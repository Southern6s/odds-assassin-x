import unittest

from engine.probability import (
    Forecast,
    brier_score,
    combine_forecasts,
    log_loss,
)


class ProbabilityTests(unittest.TestCase):
    def test_weighted_forecast_average(self):
        result = combine_forecasts([
            Forecast("model-a", 0.60, 2),
            Forecast("model-b", 0.30, 1),
        ])
        self.assertAlmostEqual(result.probability, 0.50)
        self.assertEqual(result.source_count, 2)
        self.assertAlmostEqual(result.disagreement, 0.1414213562)

    def test_requires_independent_sources(self):
        with self.assertRaises(ValueError):
            combine_forecasts([Forecast("same", 0.5), Forecast("same", 0.7)])

    def test_rejects_invalid_probability_and_weight(self):
        with self.assertRaises(ValueError):
            combine_forecasts([Forecast("a", 1.2), Forecast("b", 0.5)])
        with self.assertRaises(ValueError):
            combine_forecasts([Forecast("a", 0.4, 0), Forecast("b", 0.5)])

    def test_requires_minimum_sources(self):
        with self.assertRaises(ValueError):
            combine_forecasts([Forecast("a", 0.5)])

    def test_brier_score(self):
        self.assertAlmostEqual(brier_score(0.8, 1), 0.04)
        self.assertAlmostEqual(brier_score(0.8, 0), 0.64)

    def test_log_loss(self):
        self.assertAlmostEqual(log_loss(0.8, 1), -__import__("math").log(0.8))
        self.assertGreater(log_loss(0.99, 0), log_loss(0.6, 0))


if __name__ == "__main__":
    unittest.main()
