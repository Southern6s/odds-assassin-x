import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from engine.candlesticks import candlestick_snapshot
from engine.kalshi_data import snapshot_record
from engine.storage import SQLiteMarketStore
from engine.trade_data import trade_snapshot


class SQLiteMarketStoreTests(unittest.TestCase):
    def test_market_snapshots_are_atomic_and_duplicate_safe(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SQLiteMarketStore(Path(temp) / "history.sqlite3")
            when = datetime(2026, 1, 1, tzinfo=timezone.utc)
            record = snapshot_record({"ticker": "MKT-1", "yes_bid": 42}, when)
            result = store.write_market_snapshots([record, record])
            self.assertEqual((result.inserted, result.duplicates), (1, 1))
            self.assertEqual(store.market_snapshot_count(), 1)
            self.assertEqual(store.latest_market_snapshots()[0]["market"]["ticker"], "MKT-1")

    def test_market_snapshots_preserve_distinct_capture_times(self):
        store = SQLiteMarketStore(":memory:")
        first = snapshot_record(
            {"ticker": "MKT-1"}, datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        second = snapshot_record(
            {"ticker": "MKT-1"}, datetime(2026, 1, 2, tzinfo=timezone.utc)
        )
        result = store.write_market_snapshots([first, second])
        self.assertEqual(result.inserted, 2)
        self.assertEqual(store.market_snapshot_count(), 2)

    def test_candles_are_keyed_by_market_interval_period_and_capture(self):
        store = SQLiteMarketStore(":memory:")
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = {"end_period_ts": 1767225600, "price": {"close": "0.42"}}
        record = candlestick_snapshot(
            raw, market_ticker="MKT-1", period_interval=60, captured_at=when
        )
        result = store.write_candlestick_snapshots([record, record])
        self.assertEqual((result.inserted, result.duplicates), (1, 1))
        rows = store.candles_for_market("MKT-1", 60)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candle"]["price"]["close"], 0.42)


    def test_trade_snapshots_deduplicate_by_exchange_trade_id(self):
        store = SQLiteMarketStore(":memory:")
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        raw = {
            "trade_id": "TRADE-1", "ticker": "MKT-1", "count_fp": "1.00",
            "yes_price_dollars": "0.42", "no_price_dollars": "0.58",
            "taker_outcome_side": "yes", "taker_book_side": "bid",
            "created_time": "2026-01-01T00:00:00Z", "is_block_trade": False,
        }
        record = trade_snapshot(raw, captured_at=when)
        result = store.write_trade_snapshots([record, record])
        self.assertEqual((result.inserted, result.duplicates), (1, 1))
        self.assertEqual(store.trade_snapshot_count(), 1)
        rows = store.trades_for_market("MKT-1")
        self.assertEqual(rows[0]["trade"]["trade_id"], "TRADE-1")

    def test_invalid_query_limits_are_rejected(self):
        store = SQLiteMarketStore(":memory:")
        with self.assertRaises(ValueError):
            store.latest_market_snapshots(0)
        with self.assertRaises(ValueError):
            store.candles_for_market("MKT-1", 5)


if __name__ == "__main__":
    unittest.main()
