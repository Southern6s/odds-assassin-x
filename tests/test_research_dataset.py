import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from engine.research_dataset import (
    build_dataset_from_records, read_jsonl, write_jsonl_atomic,
)


class ResearchDatasetTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
        self.markets = [
            {"captured_at": "2026-01-01T02:59:00Z", "market": {
                "ticker": "M1", "event_ticker": "EV1", "result": "yes",
                "settlement_ts": "2026-01-01T05:00:00Z",
                "yes_bid": 0.4, "yes_ask": 0.5,
            }},
            {"captured_at": "2026-01-01T03:01:00Z", "market": {
                "ticker": "M1", "event_ticker": "EV1", "result": "yes",
                "settlement_ts": "2026-01-01T05:00:00Z",
                "yes_bid": 0.9, "yes_ask": 0.95,
            }},
        ]
        self.trades = [
            {"captured_at": "2026-01-01T03:01:00Z", "trade": {
                "trade_id": "T1", "ticker": "M1", "count": 10,
                "yes_price": 0.45, "no_price": 0.55,
                "taker_outcome_side": "yes", "taker_book_side": "bid",
                "created_time": "2026-01-01T01:10:00Z", "is_block_trade": False,
            }},
            {"captured_at": "2026-01-01T03:01:00Z", "trade": {
                "trade_id": "T2", "ticker": "M1", "count": 1000,
                "yes_price": 0.95, "no_price": 0.05,
                "taker_outcome_side": "yes", "taker_book_side": "bid",
                "created_time": "2026-01-01T03:10:00Z", "is_block_trade": False,
            }},
        ]

    def test_builds_point_in_time_dataset_and_ignores_future_quotes_and_trades(self):
        rows = build_dataset_from_records(
            self.markets, self.trades, forecast_at=self.cutoff,
            max_quote_age_seconds=600,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_yes_mid"], 0.45)
        self.assertEqual(rows[0]["feature_trade_count"], 1)
        self.assertEqual(rows[0]["feature_contract_volume"], 10.0)
        self.assertEqual(rows[0]["probability_label_yes"], 1)

    def test_atomic_jsonl_export_and_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "dataset.jsonl"
            count = write_jsonl_atomic([{"ticker": "M1"}, {"ticker": "M2"}], path)
            self.assertEqual(count, 2)
            self.assertEqual(read_jsonl(path), [{"ticker": "M1"}, {"ticker": "M2"}])
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_invalid_jsonl_reports_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
                read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
