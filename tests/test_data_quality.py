import unittest

from engine.data_quality import assess_candlesticks, assess_market_snapshots


def candle(ts, ticker="MKT-1", interval=1):
    def make_ohlc():
        return {"open": 0.4, "high": 0.5, "low": 0.3, "close": 0.45}
    return {"candle": {
        "market_ticker": ticker, "end_period_ts": ts,
        "period_interval": interval, "yes_bid": make_ohlc(),
        "yes_ask": make_ohlc(), "price": make_ohlc(),
    }}


class DataQualityTests(unittest.TestCase):
    def test_market_snapshot_flags_missing_quotes_and_stale_capture(self):
        report = assess_market_snapshots(
            [{"captured_at": "2026-01-01T00:00:00+00:00",
              "market": {"ticker": "MKT-1", "yes_bid": 0.4}}],
            stale_after_seconds=60,
            as_of=__import__("datetime").datetime.fromisoformat("2026-01-01T00:02:00+00:00"),
        )
        self.assertEqual(report.count("missing_yes_quote"), 1)
        self.assertEqual(report.count("missing_no_quote"), 1)
        self.assertEqual(report.count("stale_snapshot"), 1)
        self.assertFalse(report.passed)

    def test_market_snapshot_rejects_naive_capture_time(self):
        report = assess_market_snapshots(
            [{"captured_at": "2026-01-01T00:00:00", "market": {"ticker": "MKT-1"}}]
        )
        self.assertEqual(report.count("invalid_capture_time"), 1)

    def test_candles_detect_duplicate_periods_gaps_and_incomplete_ohlc(self):
        incomplete = candle(120)
        incomplete["candle"]["price"]["close"] = None
        report = assess_candlesticks(
            [candle(60), candle(60), incomplete, candle(240)],
            expected_interval=1,
        )
        self.assertEqual(report.count("duplicate_period"), 1)
        self.assertEqual(report.count("missing_periods"), 1)
        self.assertEqual(report.count("incomplete_ohlc"), 1)

    def test_candles_detect_interval_mismatch(self):
        report = assess_candlesticks([candle(60, interval=60)], expected_interval=1)
        self.assertEqual(report.count("interval_mismatch"), 1)

    def test_invalid_interval_rejected(self):
        with self.assertRaises(ValueError):
            assess_candlesticks([], expected_interval=5)


if __name__ == "__main__":
    unittest.main()
