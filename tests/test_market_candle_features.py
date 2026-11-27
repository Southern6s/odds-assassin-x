import unittest
from datetime import datetime, timezone

from engine.market_candle_features import build_market_candle_features


class MarketCandleFeatureTests(unittest.TestCase):
    def test_uses_only_candles_ended_by_forecast_cutoff(self):
        cutoff = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
        base = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
        candles = []
        for i in range(27):
            p = 0.30 + i * 0.01
            candles.append({
                "candle": {
                    "market_ticker": "MKT",
                    "end_period_ts": base + (i + 1) * 60,
                    "yes_bid": {"close": p - 0.005},
                    "yes_ask": {"close": p + 0.005},
                    "price": {"close": p},
                }
            })
        features = build_market_candle_features(candles, forecast_at=cutoff)
        self.assertIn("MKT", features)
        self.assertEqual(features["MKT"]["market_candle_count"], 27)
        self.assertGreater(features["MKT"]["market_momentum_logit_5"], 0)
        self.assertIn("market_mean_reversion_z20", features["MKT"])
        self.assertNotIn("future", features["MKT"])

    def test_future_and_stale_candles_are_excluded(self):
        cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
        future = int(datetime(2026, 1, 2, tzinfo=timezone.utc).timestamp())
        stale = int(datetime(2025, 12, 30, tzinfo=timezone.utc).timestamp())
        rows = [{"candle": {"market_ticker": "M", "end_period_ts": ts,
                            "price": {"close": 0.6}}}
                for ts in (future, stale)]
        self.assertEqual(build_market_candle_features(
            rows, forecast_at=cutoff, max_age_seconds=3600), {})


if __name__ == "__main__":
    unittest.main()
