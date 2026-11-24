import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.collector import collect_from_kalshi, collect_market_snapshot


class FakeClient:
    def get_historical_cutoff(self):
        return {"cutoff_ts": 123}

    def iter_categorized_markets(self, *, limit, max_pages_per_tier):
        yield {"ticker": "A", "series_ticker": "S", "yes_bid": 40}
        yield {"ticker": "A", "series_ticker": "S", "yes_bid": 40}
        yield {"ticker": "B", "yes_bid": 120}


class CollectorTests(unittest.TestCase):
    def test_appends_valid_rows_and_reports_rejected_rows(self):
        markets = [
            {"ticker": "GOOD", "yes_bid": 40, "yes_ask": 45},
            {"ticker": "BAD", "yes_bid": 80, "yes_ask": 20},
            {"ticker": "ALSO_GOOD", "yes_bid": 10, "yes_ask": 15},
        ]
        captured = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "snapshots.jsonl"
            first = collect_market_snapshot(markets, path, captured_at=captured)
            second = collect_market_snapshot(
                [{"ticker": "NEXT", "yes_bid": 20, "yes_ask": 25}], path,
                captured_at=captured,
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]

        self.assertEqual(first.written, 2)
        self.assertEqual(first.rejected, 1)
        self.assertEqual(len(first.rejection_samples), 1)
        self.assertEqual(second.written, 1)
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["captured_at"] for row in rows}, {captured.isoformat()})
        self.assertEqual(rows[-1]["market"]["ticker"], "NEXT")

    def test_requires_timezone_aware_timestamp(self):
        with TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                collect_market_snapshot([], Path(folder) / "x.jsonl",
                                        captured_at=datetime(2026, 1, 1))

    def test_full_collector_deduplicates_and_reports_bad_records(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "all.jsonl"
            report = collect_from_kalshi(FakeClient(), path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(report.written, 1)
        self.assertEqual(report.duplicates, 1)
        self.assertEqual(report.rejected, 1)
        self.assertEqual(report.historical_cutoff, {"cutoff_ts": 123})
        self.assertEqual(rows[0]["market"]["ticker"], "A")

    def test_full_collector_appends_on_subsequent_runs(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "all.jsonl"
            collect_from_kalshi(FakeClient(), path)
            collect_from_kalshi(FakeClient(), path)
            self.assertEqual(len(path.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
