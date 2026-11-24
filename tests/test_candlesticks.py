import unittest
from datetime import datetime, timezone

from engine.candlesticks import candlestick_snapshot, normalize_candlestick


class CandlestickTests(unittest.TestCase):
    def test_normalizes_dollar_ohlc_and_preserves_raw(self):
        raw = {
            "end_period_ts": 1790000000,
            "yes_bid": {"open_dollars": "0.40", "close_dollars": "0.45"},
            "yes_ask": {"open_dollars": "0.50", "close_dollars": "0.55"},
            "price": {"close_dollars": "0.51", "mean_dollars": "0.49"},
            "volume_fp": "12.5",
        }
        result = normalize_candlestick(raw, market_ticker="MKT", period_interval=60)
        self.assertEqual(result["yes_bid"]["close"], 0.45)
        self.assertEqual(result["yes_ask"]["close"], 0.55)
        self.assertEqual(result["price"]["mean"], 0.49)
        self.assertEqual(result["raw"], raw)

    def test_supports_legacy_values_and_snapshot_timestamp(self):
        raw = {
            "end_period_ts": 1790000000,
            "yes_bid": {"close": 40},
            "yes_ask": {"close": 50},
            "price": {"close": 45},
            "volume": 3,
        }
        stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        result = candlestick_snapshot(raw, market_ticker="MKT", period_interval=1,
                                      captured_at=stamp)
        self.assertEqual(result["candle"]["yes_bid"]["close"], 0.4)
        self.assertEqual(result["captured_at"], stamp.isoformat())

    def test_rejects_invalid_timestamp_price_interval_and_crossed_candle(self):
        with self.assertRaises(ValueError):
            normalize_candlestick({}, market_ticker="M", period_interval=60)
        with self.assertRaises(ValueError):
            normalize_candlestick({"end_period_ts": 1}, market_ticker="M", period_interval=5)
        with self.assertRaises(ValueError):
            normalize_candlestick({
                "end_period_ts": 1,
                "yes_bid": {"close_dollars": "0.8"},
                "yes_ask": {"close_dollars": "0.7"},
            }, market_ticker="M", period_interval=60)
        with self.assertRaises(ValueError):
            normalize_candlestick({
                "end_period_ts": 1, "price": {"close_dollars": "1.2"}
            }, market_ticker="M", period_interval=60)


if __name__ == "__main__":
    unittest.main()
