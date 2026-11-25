import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.open_market_discovery import discover_open_resumable


class OpenMarketClient:
    def __init__(self):
        self.calls = []

    def get_series(self):
        return [{"ticker": "S", "category": "Sports"}]

    def get_market_page(self, endpoint, *, cursor, limit, status=None, mve_filter=None):
        self.calls.append((endpoint, cursor, limit, status, mve_filter))
        pages = {
            None: ([{"ticker": "OPEN-A", "series_ticker": "S", "status": "open"}], "next"),
            "next": ([{"ticker": "OPEN-B", "series_ticker": "S", "status": "open"}], None),
        }
        return pages[cursor]


class OpenMarketDiscoveryTests(unittest.TestCase):
    def test_resumes_open_only_pages_and_does_not_touch_broad_catalog(self):
        client = OpenMarketClient()
        with TemporaryDirectory() as folder:
            output = Path(folder) / "open.jsonl"
            state = Path(folder) / "open-state.json"

            first = discover_open_resumable(
                client, output, state, max_pages=1
            )
            self.assertFalse(first["complete"])
            self.assertEqual(first["cursor"], "next")
            self.assertEqual(first["status"], "open")
            self.assertEqual(len(output.read_text().splitlines()), 1)

            second = discover_open_resumable(
                client, output, state, max_pages=1
            )
            self.assertTrue(second["complete"])
            self.assertEqual(second["pages_completed"], 2)
            self.assertEqual(
                [call[3:] for call in client.calls],
                [("open", "exclude"), ("open", "exclude")],
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [row["market"]["ticker"] for row in rows],
                ["OPEN-A", "OPEN-B"],
            )
            self.assertEqual(rows[0]["market"]["category"], "Sports")

    def test_checkpoint_rejects_other_status_or_page_size(self):
        client = OpenMarketClient()
        with TemporaryDirectory() as folder:
            output = Path(folder) / "open.jsonl"
            state = Path(folder) / "open-state.json"
            discover_open_resumable(client, output, state, max_pages=1)
            checkpoint = json.loads(state.read_text())
            checkpoint["mve_filter"] = "only"
            state.write_text(json.dumps(checkpoint))
            with self.assertRaises(ValueError):
                discover_open_resumable(client, output, state, max_pages=1)


if __name__ == "__main__":
    unittest.main()
