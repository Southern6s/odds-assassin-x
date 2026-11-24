import unittest

from engine.market_baseline import evaluate_market_baseline


class MarketBaselineTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"market_ticker": "A", "forecast_at": "2026-01-01T00:00:00Z",
             "resolved_at": "2026-01-02T00:00:00Z", "probability_label_yes": 1,
             "market_yes_bid": 0.7, "market_yes_ask": 0.9, "market_yes_mid": 0.8},
            {"market_ticker": "B", "forecast_at": "2026-01-01T00:00:00Z",
             "resolved_at": "2026-01-02T00:00:00Z", "probability_label_yes": 0,
             "market_yes_bid": 0.1, "market_yes_ask": 0.3, "market_yes_mid": 0.2},
            {"market_ticker": "C", "forecast_at": "2026-01-01T00:00:00Z",
             "resolved_at": "2026-01-02T00:00:00Z", "probability_label_yes": 1,
             "market_yes_mid": None},
        ]

    def test_scores_midpoint_and_reports_coverage_and_neutral_baseline(self):
        report = evaluate_market_baseline(self.rows)
        self.assertEqual(report["rows_total"], 3)
        self.assertEqual(report["rows_scored"], 2)
        self.assertAlmostEqual(report["quote_coverage"], 2 / 3)
        self.assertAlmostEqual(report["brier"], 0.04)
        self.assertAlmostEqual(report["neutral_brier"], 0.25)
        self.assertAlmostEqual(report["brier_skill_vs_neutral"], 0.84)
        self.assertAlmostEqual(report["mean_yes_spread"], 0.2)
        self.assertEqual(len(report["calibration"]), 2)
        self.assertTrue(report["paper_only"])
        self.assertFalse(report["live_trading"])

    def test_no_midpoints_returns_null_scores_not_fake_zeroes(self):
        rows = [dict(self.rows[2])]
        report = evaluate_market_baseline(rows)
        self.assertEqual(report["rows_scored"], 0)
        self.assertIsNone(report["brier"])
        self.assertIsNone(report["brier_skill_vs_neutral"])

    def test_rejects_duplicate_ticker_and_invalid_timestamp(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate_market_baseline([self.rows[0], self.rows[0]])
        bad = dict(self.rows[0], forecast_at="2026-01-03T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "precede"):
            evaluate_market_baseline([bad])

    def test_rejects_bad_quote_and_bad_calibration_bins(self):
        bad = dict(self.rows[0], market_yes_mid=1.2)
        with self.assertRaises(ValueError):
            evaluate_market_baseline([bad])
        with self.assertRaises(ValueError):
            evaluate_market_baseline(self.rows, calibration_bins=1)


if __name__ == "__main__":
    unittest.main()
