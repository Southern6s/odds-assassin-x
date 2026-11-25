import unittest

from engine.market_math import (
    analyze_market,
    analyze_quote,
    calculate_edge,
    expected_value_per_contract,
)


class MarketMathTests(unittest.TestCase):
    def test_quote_midpoint_and_spread(self):
        quote = analyze_quote(0.42, 0.48)
        self.assertAlmostEqual(quote.midpoint, 0.45)
        self.assertAlmostEqual(quote.spread, 0.06)

    def test_rejects_crossed_quote(self):
        with self.assertRaises(ValueError):
            analyze_quote(0.55, 0.45)

    def test_edge_is_model_minus_market(self):
        self.assertAlmostEqual(calculate_edge(0.60, 0.50), 0.10)

    def test_ev_uses_ask_and_excludes_fees(self):
        self.assertAlmostEqual(expected_value_per_contract(0.60, 0.50), 0.10)

    def test_no_forecast_means_no_edge(self):
        result = analyze_market(0.42, 0.48)
        self.assertIsNone(result["edge"])
        self.assertIsNone(result["expected_value_per_contract"])
        self.assertEqual(result["edge_status"], "unavailable_no_independent_forecast")

    def test_forecast_enables_edge(self):
        result = analyze_market(0.42, 0.48, 0.60)
        self.assertAlmostEqual(result["edge"], 0.15)
        self.assertAlmostEqual(result["expected_value_per_contract"], 0.12)
        self.assertEqual(result["edge_status"], "calculated")


if __name__ == "__main__":
    unittest.main()
