import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from engine.market_trade_collector import collect_market_trade_snapshots


def trade(trade_id, ticker, created="2026-01-01T00:01:00Z"):
    return {
        "trade_id": trade_id, "ticker": ticker, "count_fp": "2",
        "yes_price_dollars": "0.4", "no_price_dollars": "0.6",
        "taker_outcome_side": "yes", "taker_book_side": "bid",
        "created_time": created, "is_block_trade": False,
    }


class FakeClient:
    def __init__(self):
        self.queried = []

    def iter_all_trade_tiers(self, **kwargs):
        self.queried.append(kwargs["ticker"])
        if kwargs["ticker"] == "M1":
            yield trade("T1", "M1")
            yield trade("T1", "M1")
            yield trade("OLD", "M1", "2025-12-31T23:59:00Z")
        else:
            yield trade("T2", "M2")


class MarketTradeCollectorTests(unittest.TestCase):
    def test_queries_each_market_and_appends_only_unique_in_window_trades(self):
        with TemporaryDirectory() as folder:
            markets = Path(folder) / "markets.jsonl"
            output = Path(folder) / "trades.jsonl"
            markets.write_text("\\n".join([
                json.dumps({"market": {"ticker": "M1"}}),
                json.dumps({"market": {"ticker": "M2"}}),
                json.dumps({"market": {"ticker": "M1"}}),
            ]) + "\\n")
            client = FakeClient()
            report = collect_market_trade_snapshots(
                client, markets, output, min_ts=1767225600,
                max_ts=1767229200, max_pages_per_tier=3,
            )
            rows = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(client.queried, ["M1", "M2"])
        self.assertEqual(report.markets_requested, 2)
        self.assertEqual(report.markets_with_trades, 2)
        self.assertEqual(report.written, 2)
        self.assertEqual(report.duplicates, 1)
        self.assertEqual(report.outside_window, 1)
        self.assertEqual({row["trade"]["trade_id"] for row in rows}, {"T1", "T2"})

    def test_existing_trade_ids_are_not_written_twice(self):
        with TemporaryDirectory() as folder:
            markets = Path(folder) / "markets.jsonl"
            output = Path(folder) / "trades.jsonl"
            markets.write_text(json.dumps({"market": {"ticker": "M1"}}) + "\\n")
            output.write_text(json.dumps({"trade": {"trade_id": "T1"}}) + "\\n")
            report = collect_market_trade_snapshots(
                FakeClient(), markets, output, min_ts=1767225600,
                max_ts=1767229200,
            )
        self.assertEqual(report.written, 0)
        self.assertEqual(report.duplicates, 2)


if __name__ == "__main__":
    unittest.main()
