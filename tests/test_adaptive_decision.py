import unittest

from engine.adaptive_decision import (
    CalibrationBin,
    MarketState,
    ModelForecast,
    adaptive_decision,
    calibrate_probability,
)


class AdaptiveDecisionTests(unittest.TestCase):
    def setUp(self):
        self.forecasts = [
            ModelForecast("logistic", 0.75, 0.18, 200),
            ModelForecast("tree", 0.70, 0.16, 180),
        ]
        self.market = MarketState(
            yes_bid=0.48, yes_ask=0.50, spread=0.02,
            liquidity=100, volatility=0.01, quote_age_seconds=10,
            fees_probability=0.005, slippage_probability=0.005,
        )

    def test_selects_side_only_when_net_edge_clears_dynamic_threshold(self):
        result = adaptive_decision(self.forecasts, self.market)
        self.assertEqual(result.action, "CONSIDER_YES")
        self.assertGreater(result.yes_net_edge, result.required_edge)
        self.assertAlmostEqual(sum(result.model_weights.values()), 1.0)
        self.assertTrue(result.paper_only)
        self.assertFalse(result.live_trading)

    def test_abstains_on_stale_wide_and_illiquid_market(self):
        market = MarketState(
            yes_bid=0.35, yes_ask=0.65, spread=0.30,
            liquidity=0.1, volatility=0.2, quote_age_seconds=500,
        )
        result = adaptive_decision(self.forecasts, market)
        self.assertEqual(result.action, "WATCH")
        self.assertIn("stale_quote", result.reasons)
        self.assertIn("spread_too_wide", result.reasons)
        self.assertIn("insufficient_liquidity", result.reasons)

    def test_abstains_when_models_disagree(self):
        forecasts = [
            ModelForecast("one", 0.95, 0.2, 100),
            ModelForecast("two", 0.05, 0.2, 100),
        ]
        result = adaptive_decision(forecasts, self.market)
        self.assertEqual(result.action, "WATCH")
        self.assertIn("models_disagree", result.reasons)

    def test_recent_skill_weights_are_normalized_and_conservative(self):
        forecasts = [
            ModelForecast("proven", 0.8, 0.10, 500),
            ModelForecast("weak", 0.6, 0.35, 500),
        ]
        result = adaptive_decision(forecasts, self.market)
        self.assertGreater(result.model_weights["proven"], result.model_weights["weak"])
        self.assertAlmostEqual(sum(result.model_weights.values()), 1.0)

    def test_insufficient_history_does_not_claim_skill_advantage(self):
        forecasts = [
            ModelForecast("new-a", 0.7, 0.01, 5),
            ModelForecast("new-b", 0.5, 0.40, 5),
        ]
        result = adaptive_decision(forecasts, self.market, min_model_observations=50)
        self.assertAlmostEqual(result.model_weights["new-a"], 0.5)
        self.assertAlmostEqual(result.model_weights["new-b"], 0.5)

    def test_calibration_shrinks_small_samples_and_uses_past_bin(self):
        bins = [CalibrationBin(0.6, 0.8, 0.6, 10)]
        calibrated = calibrate_probability(0.7, bins, prior_strength=20)
        self.assertAlmostEqual(calibrated, (0.7 * 20 + 0.6 * 10) / 30)
        self.assertEqual(calibrate_probability(0.4, bins), 0.4)

    def test_rejects_invalid_market_and_forecast_inputs(self):
        with self.assertRaises(ValueError):
            adaptive_decision(
                self.forecasts,
                MarketState(0.6, 0.5, 0.1, 10, 0.01, 1),
            )
        with self.assertRaises(ValueError):
            adaptive_decision(
                [ModelForecast("bad", float("nan"), 0.2, 100)],
                self.market,
            )


if __name__ == "__main__":
    unittest.main()
