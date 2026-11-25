import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.trade_collector import collect_trade_snapshots


def raw_trade(trade_id="T1"):
    return {
        "trade_id": trade_id, "ticker": "MKT-1", "count_fp": "2.50",
        "yes_price_dollars": "0.42", "no_price_dollars": "0.58",
        "taker_outcome_side": "yes", "taker_book_side": "bid",
        "created_time": "2026-01-01T00:00:00Z", "is_block_trade": False,
    }


class FakeClient:
    def iter_all_trade_tiers(self, **kwargs):
        self.kwargs = kwargs
        yield raw_trade("T1")
        yield raw_trade("T1")
        yield raw_trade("T2")
        yield {"trade_id": "BAD"}


class TradeCollectorTests(unittest.TestCase):
    def test_collects_normalized_unique_trade_snapshots_and_reports_bad_rows(self):
        client = FakeClient()
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trades.jsonl"
            report = collect_trade_snapshots(
                client, path, min_ts=100, max_ts=200, ticker="MKT-1",
                is_block_trade=False,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(report.written, 2)
        self.assertEqual(report.duplicates, 1)
        self.assertEqual(report.rejected, 1)
        self.assertEqual([row["trade"]["trade_id"] for row in rows], ["T1", "T2"])
        self.assertEqual(client.kwargs["min_ts"], 100)
        self.assertEqual(client.kwargs["max_ts"], 200)
        self.assertEqual(client.kwargs["is_block_trade"], False)
        self.assertTrue(all(row["captured_at"] == report.captured_at for row in rows))

    def test_requires_bounded_valid_window_and_page_limits(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "trades.jsonl"
            for kwargs in (
                {"min_ts": -1, "max_ts": 200},
                {"min_ts": 200, "max_ts": 200},
                {"min_ts": 100, "max_ts": 200, "limit": 1001},
                {"min_ts": 100, "max_ts": 200, "max_pages_per_tier": 0},
            ):
                with self.assertRaises(ValueError):
                    collect_trade_snapshots(FakeClient(), path, **kwargs)


if __name__ == "__main__":
    unittest.main()
