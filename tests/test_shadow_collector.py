import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from engine.shadow_collector import collect_shadow_snapshots


class ShadowCollectorTests(unittest.TestCase):
    def test_quote_and_depth_share_timestamp_and_forecast_is_not_fabricated(self):
        client = Mock()
        client.iter_markets.return_value = [
            {"ticker": "KXTEST", "event_ticker": "EVTEST", "title": "Test", "status": "open"}
        ]
        client.get_market_orderbook.return_value = {
            "orderbook_fp": {
                "yes_dollars": [["0.42", 5]],
                "no_dollars": [["0.51", 7]],
            }
        }
        times = iter([0.0, 1.0, 1.0, 1.0])
        def clock():
            return next(times, 1.0)
        captured = datetime(2026, 11, 28, 6, 0, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shadow.jsonl"
            report = collect_shadow_snapshots(
                client, path, max_markets=1, duration_seconds=1,
                interval_seconds=5, sleeper=lambda _: None, clock=clock,
                wall_clock=lambda: captured,
            )
            row = json.loads(path.read_text().strip())
        self.assertEqual(report["records_written"], 1)
        self.assertEqual(row["captured_at"], row["orderbook"]["captured_at"])
        self.assertEqual(row["quote"]["yes_bid"], 0.42)
        self.assertEqual(row["quote"]["yes_ask"], 0.49)
        self.assertTrue(row["quote_depth_same_capture"])
        self.assertIsNone(row["model_probability_yes"])
        self.assertEqual(row["forecast_status"], "not_generated_live_inference_not_wired")
        self.assertTrue(row["paper_only"])
        self.assertFalse(row["live_trading"])

    def test_rejects_naive_capture_timestamp(self):
        client = Mock()
        client.iter_markets.return_value = [{"ticker": "KXTEST"}]
        client.get_market_orderbook.return_value = {
            "orderbook_fp": {"yes_dollars": [["0.4", 1]], "no_dollars": [["0.5", 1]]}
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "No shadow records"):
                collect_shadow_snapshots(
                    client, Path(tmp) / "shadow.jsonl", duration_seconds=1,
                    wall_clock=lambda: datetime(2026, 1, 1),
                    clock=iter([0.0, 1.0]).__next__,
                )

    def test_validates_resource_bounds(self):
        with self.assertRaises(ValueError):
            collect_shadow_snapshots(Mock(), "unused", max_markets=101)


if __name__ == "__main__":
    unittest.main()
