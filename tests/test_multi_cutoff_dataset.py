import unittest
from datetime import datetime, timezone

from engine.multi_cutoff_dataset import build_multi_cutoff_dataset


class MultiCutoffDatasetTests(unittest.TestCase):
    def test_builds_repeated_point_in_time_rows_with_candle_quotes(self):
        markets = [{"captured_at": "2026-01-01T00:00:00Z", "market": {
            "ticker": "M1", "event_ticker": "EV1", "result": "yes",
            "settlement_ts": "2026-01-02T00:00:00Z",
        }}]
        trades = [
            {"trade": {"trade_id": "T1", "ticker": "M1", "count": 2,
             "yes_price": 0.4, "no_price": 0.6, "taker_outcome_side": "yes",
             "taker_book_side": "bid", "created_time": "2026-01-01T00:10:00Z",
             "is_block_trade": False}},
            {"trade": {"trade_id": "T2", "ticker": "M1", "count": 3,
             "yes_price": 0.5, "no_price": 0.5, "taker_outcome_side": "yes",
             "taker_book_side": "bid", "created_time": "2026-01-01T01:10:00Z",
             "is_block_trade": False}},
        ]
        candles = [
            {"candle": {"market_ticker": "M1", "end_period_ts": 1767229200,
                        "yes_bid": {"close": 0.3}, "yes_ask": {"close": 0.4}}},
            {"candle": {"market_ticker": "M1", "end_period_ts": 1767232800,
                        "yes_bid": {"close": 0.5}, "yes_ask": {"close": 0.6}}},
        ]
        cutoffs = [
            datetime(2026, 1, 1, 1, 30, tzinfo=timezone.utc),
            datetime(2026, 1, 1, 2, 30, tzinfo=timezone.utc),
        ]
        rows = build_multi_cutoff_dataset(
            markets, trades, cutoffs, candlestick_snapshots=candles,
            max_quote_age_seconds=3600,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["market_ticker"], rows[1]["market_ticker"])
        self.assertEqual(rows[0]["market_yes_mid"], 0.35)
        self.assertEqual(rows[1]["market_yes_mid"], 0.55)
        self.assertEqual(rows[0]["feature_contract_volume"], 2.0)
        self.assertEqual(rows[1]["feature_contract_volume"], 3.0)

    def test_rejects_duplicate_or_naive_cutoffs(self):
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with self.assertRaisesRegex(ValueError, "unique"):
            build_multi_cutoff_dataset([], [], [cutoff, cutoff])
        with self.assertRaisesRegex(ValueError, "timezone"):
            build_multi_cutoff_dataset([], [], [datetime(2026, 1, 1)])


if __name__ == "__main__":
    unittest.main()
