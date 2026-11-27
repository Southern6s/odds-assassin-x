import json
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.catalog_discovery import discover_resumable


class PagedClient:
    def __init__(self):
        self.calls = []

    def get_historical_cutoff(self):
        return {"cutoff_ts": 123}

    def get_series(self):
        return [{"ticker": "S", "category": "Sports"}]

    def get_market_page(self, endpoint, *, cursor, limit):
        self.calls.append((endpoint, cursor, limit))
        pages = {
            ("markets", None): ([{"ticker": "A", "series_ticker": "S", "yes_bid": 40}], "live-2"),
            ("markets", "live-2"): ([{"ticker": "B", "series_ticker": "S", "yes_bid": 30}], None),
            ("historical/markets", None): ([{"ticker": "C", "series_ticker": "S", "yes_bid": 20}], None),
        }
        return pages[(endpoint, cursor)]


class ResumableCatalogTests(unittest.TestCase):
    def test_checkpoint_resumes_both_tiers_without_duplicate_rows(self):
        client = PagedClient()
        with TemporaryDirectory() as folder:
            output = Path(folder) / "catalog.jsonl"
            state = Path(folder) / "state.json"

            first = discover_resumable(
                client, output, state, limit=1000, max_pages_per_tier=1
            )
            self.assertFalse(first["complete"])
            self.assertEqual(first["tier"], "markets")
            self.assertEqual(first["cursor"], "live-2")
            self.assertEqual(len(output.read_text().splitlines()), 1)

            second = discover_resumable(
                client, output, state, limit=1000, max_pages_per_tier=1
            )
            self.assertTrue(second["complete"])
            self.assertEqual(second["pages_completed"], {
                "markets": 2, "historical/markets": 1
            })
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [row["market"]["ticker"] for row in rows], ["A", "B", "C"]
            )
            self.assertEqual([call[1] for call in client.calls], [
                None, "live-2", None
            ])

    def test_catalog_market_without_series_ticker_gets_authoritative_category(self):
        class MissingSeriesTickerClient(PagedClient):
            def get_market_page(self, endpoint, *, cursor, limit):
                if endpoint == "markets" and cursor is None:
                    return ([{
                        "ticker": "S-EVENT-MARKET",
                        "event_ticker": "S-EVENT",
                        "result": "yes",
                    }], None)
                return ([], None)

        with TemporaryDirectory() as folder:
            output = Path(folder) / "catalog.jsonl"
            state = Path(folder) / "state.json"
            discover_resumable(
                MissingSeriesTickerClient(), output, state, max_pages_per_tier=1
            )
            row = json.loads(output.read_text().splitlines()[0])
            self.assertEqual(row["market"]["series_ticker"], "S")
            self.assertEqual(row["market"]["category"], "Sports")

    def test_market_rows_use_page_capture_time_not_session_start(self):
        client = PagedClient()
        with TemporaryDirectory() as folder:
            output = Path(folder) / "catalog.jsonl"
            state = Path(folder) / "state.json"
            discover_resumable(
                client, output, state, max_pages_per_tier=1
            )
            checkpoint = json.loads(state.read_text())
            row = json.loads(output.read_text().splitlines()[0])
            row_time = datetime.fromisoformat(row["captured_at"])
            session_time = datetime.fromisoformat(checkpoint["captured_at"])
            self.assertIsNotNone(row_time.tzinfo)
            self.assertGreaterEqual(row_time, session_time)
            self.assertNotEqual(row["captured_at"], checkpoint["captured_at"])

    def test_replay_of_uncheckpointed_page_is_idempotent(self):
        client = PagedClient()
        with TemporaryDirectory() as folder:
            output = Path(folder) / "catalog.jsonl"
            state = Path(folder) / "state.json"
            first = discover_resumable(
                client, output, state, max_pages_per_tier=1
            )
            # Simulate a crash after page rows were written but before the
            # checkpoint cursor advanced: roll the state back to its start.
            checkpoint = json.loads(state.read_text())
            checkpoint["cursor"] = None
            checkpoint["pages_completed"]["markets"] = 0
            checkpoint["scanned"] = 0
            checkpoint["written"] = 0
            checkpoint["duplicates"] = 0
            state.write_text(json.dumps(checkpoint))
            replay = discover_resumable(
                client, output, state, max_pages_per_tier=1
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(replay["duplicates"], 1)
            self.assertEqual(replay["cursor"], "live-2")


if __name__ == "__main__":
    unittest.main()
