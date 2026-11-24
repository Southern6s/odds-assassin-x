import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from engine.kalshi_data import append_jsonl, normalize_market, snapshot_record


class KalshiDataTests(unittest.TestCase):
    def test_normalizes_dollars_and_legacy_cents(self):
        market = {
            "ticker": "EXAMPLE-YES",
            "series_ticker": "SERIES-1",
            "yes_bid_dollars": "0.42",
            "yes_ask": 48,
            "no_bid": 51,
            "no_ask_dollars": "0.53",
        }
        result = normalize_market(market)
        self.assertEqual(result["series_ticker"], "SERIES-1")
        self.assertAlmostEqual(result["yes_bid"], 0.42)
        self.assertAlmostEqual(result["yes_ask"], 0.48)
        self.assertAlmostEqual(result["no_bid"], 0.51)
        self.assertAlmostEqual(result["no_ask"], 0.53)

    def test_retains_settlement_metadata_for_binary_labels(self):
        result = normalize_market({
            "ticker": "SETTLED-YES", "event_ticker": "EVENT-1",
            "market_type": "binary", "status": "finalized", "result": "yes",
            "settlement_ts": "2026-01-01T12:00:00Z",
            "expected_expiration_time": "2026-01-01T11:55:00Z",
        })
        self.assertEqual(result["event_ticker"], "EVENT-1")
        self.assertEqual(result["market_type"], "binary")
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["result"], "yes")
        self.assertEqual(result["settlement_ts"], "2026-01-01T12:00:00Z")
        self.assertEqual(result["expected_expiration_time"], "2026-01-01T11:55:00Z")

    def test_missing_quotes_stay_missing(self):
        result = normalize_market({"ticker": "NO-QUOTE"})
        self.assertIsNone(result["yes_bid"])
        self.assertIsNone(result["yes_ask"])

    def test_rejects_missing_ticker_invalid_price_and_crossed_quote(self):
        with self.assertRaises(ValueError):
            normalize_market({"title": "No ticker"})
        with self.assertRaises(ValueError):
            normalize_market({"ticker": "BAD", "yes_bid": 120})
        with self.assertRaises(ValueError):
            normalize_market({"ticker": "CROSSED", "yes_bid": 60, "yes_ask": 50})

    def test_snapshot_keeps_raw_payload_and_timestamp(self):
        raw = {"ticker": "T", "yes_bid": 10}
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        record = snapshot_record(raw, when)
        self.assertEqual(record["raw"], raw)
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["market"]["yes_bid"], 0.10)

    def test_jsonl_append_preserves_prior_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "data" / "snapshots.jsonl"
            records = [{"n": 1}, {"n": 2}]
            self.assertEqual(append_jsonl(records, path), 2)
            self.assertEqual(append_jsonl([{"n": 3}], path), 1)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["n"] for line in lines], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
