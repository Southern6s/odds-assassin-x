import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from engine.depth_collector import collect_depth_snapshots, normalize_orderbook


class DepthCollectorTests(unittest.TestCase):
    def test_normalizes_best_bids_and_complementary_asks(self):
        record = normalize_orderbook(
            {"orderbook": {"yes": [[40, 7], [45, 3]], "no": [[52, 8]]}},
            "MKT", datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(record["yes_bid"], 0.45)
        self.assertEqual(record["yes_bid_size"], 3)
        self.assertAlmostEqual(record["yes_ask"], 0.48)
        self.assertEqual(record["yes_ask_size"], 8)
        self.assertAlmostEqual(record["no_ask"], 0.55)
        self.assertFalse(record["live_trading"])

    def test_empty_book_does_not_invent_quotes_or_depth(self):
        record = normalize_orderbook({"orderbook": {"yes": [], "no": []}}, "MKT")
        self.assertIsNone(record["yes_bid"])
        self.assertIsNone(record["yes_ask_size"])

    def test_rejects_bad_levels(self):
        with self.assertRaises(ValueError):
            normalize_orderbook({"orderbook": {"yes": [[110, 1]], "no": []}}, "MKT")

    def test_collects_multiple_timestamped_polls(self):
        client = Mock()
        client.iter_markets.return_value = [{"ticker": "A"}, {"ticker": "B"}]
        client.get_market_orderbook.side_effect = [
            {"orderbook": {"yes": [[40, 2]], "no": [[50, 3]]}},
            {"orderbook": {"yes": [[41, 2]], "no": [[49, 3]]}},
            {"orderbook": {"yes": [[42, 2]], "no": [[48, 3]]}},
            {"orderbook": {"yes": [[43, 2]], "no": [[47, 3]]}},
        ]
        times = iter([0, 0, 6, 6, 12, 12, 18, 18, 20, 20])
        def clock():
            return next(times, 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "depth.jsonl"
            report = collect_depth_snapshots(
                client, path, max_markets=2, duration_seconds=10,
                interval_seconds=5, sleeper=lambda _: None, clock=clock,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(report["polls"], 2)
        self.assertEqual(report["snapshots_written"], 4)
        self.assertEqual({row["ticker"] for row in rows}, {"A", "B"})

    def test_validates_resource_bounds(self):
        with self.assertRaises(ValueError):
            collect_depth_snapshots(Mock(), "unused", max_markets=101)


if __name__ == "__main__":
    unittest.main()
