import unittest

from engine.depth_join import attach_depth_snapshots


def prediction(ticker="M1", at="2026-01-01T00:01:00Z"):
    return {
        "market_ticker": ticker,
        "forecast_at": at,
        "model_probability": 0.8,
        "market_yes_mid": 0.55,
        "market_yes_bid": 0.54,
        "market_yes_ask": 0.56,
        "market_yes_bid_size": 99,
        "market_yes_ask_size": 99,
        "quote_age_seconds": 1,
    }


def depth(at, *, ticker="M1", yes_bid=0.50, yes_ask=0.52,
          yes_bid_size=4, yes_ask_size=3):
    return {
        "ticker": ticker,
        "captured_at": at,
        "yes_bid": yes_bid,
        "yes_ask": yes_ask,
        "yes_bid_size": yes_bid_size,
        "yes_ask_size": yes_ask_size,
        "no_bid": 0.48,
        "no_ask": 0.50,
        "no_bid_size": 2,
        "no_ask_size": 5,
    }


class DepthJoinTests(unittest.TestCase):
    def test_uses_latest_snapshot_at_or_before_forecast(self):
        rows = attach_depth_snapshots(
            [prediction()],
            [
                depth("2026-01-01T00:00:20Z", yes_bid=0.49),
                depth("2026-01-01T00:00:50Z", yes_bid=0.51, yes_bid_size=7),
                depth("2026-01-01T00:01:10Z", yes_bid=0.60),
            ],
        )
        row = rows[0]
        self.assertEqual(row["market_yes_bid"], 0.51)
        self.assertEqual(row["market_yes_bid_size"], 7)
        self.assertEqual(row["depth_snapshot_at"], "2026-01-01T00:00:50+00:00")
        self.assertEqual(row["quote_age_seconds"], 10)
        self.assertTrue(row["depth_snapshot_matched"])
        # Market midpoint remains the predictive benchmark, not execution depth.
        self.assertEqual(row["market_yes_mid"], 0.55)

    def test_rejects_future_only_snapshot_and_clears_panel_depth(self):
        rows = attach_depth_snapshots(
            [prediction()],
            [depth("2026-01-01T00:01:01Z")],
        )
        row = rows[0]
        self.assertFalse(row["depth_snapshot_matched"])
        self.assertIsNone(row["market_yes_bid"])
        self.assertEqual(row["market_yes_ask_size"], 0)
        self.assertIsNone(row["quote_age_seconds"])

    def test_rejects_stale_snapshot(self):
        rows = attach_depth_snapshots(
            [prediction()],
            [depth("2026-01-01T00:00:00Z")],
            max_age_seconds=30,
        )
        self.assertFalse(rows[0]["depth_snapshot_matched"])
        self.assertIsNone(rows[0]["market_yes_ask"])

    def test_matches_tickers_independently(self):
        rows = attach_depth_snapshots(
            [prediction("M1"), prediction("M2")],
            [depth("2026-01-01T00:00:50Z", ticker="M2", yes_bid=0.42)],
        )
        self.assertFalse(rows[0]["depth_snapshot_matched"])
        self.assertTrue(rows[1]["depth_snapshot_matched"])
        self.assertEqual(rows[1]["market_yes_bid"], 0.42)

    def test_requires_timezone_aware_timestamps(self):
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            attach_depth_snapshots(
                [prediction(at="2026-01-01T00:01:00")],
                [depth("2026-01-01T00:00:50Z")],
            )

    def test_validates_age_limit(self):
        with self.assertRaises(ValueError):
            attach_depth_snapshots([], [], max_age_seconds=-1)


if __name__ == "__main__":
    unittest.main()
