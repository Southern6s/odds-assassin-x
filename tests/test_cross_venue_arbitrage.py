import unittest

from engine.cross_venue_arbitrage import evaluate_cross_venue_pair


class CrossVenueArbitrageTests(unittest.TestCase):
    def setUp(self):
        self.a = {"name": "Kalshi", "yes_ask": 0.42, "yes_ask_size": 12,
                  "no_ask": 0.61, "no_ask_size": 10}
        self.b = {"name": "Polymarket", "yes_ask": 0.44, "yes_ask_size": 8,
                  "no_ask": 0.52, "no_ask_size": 7}

    def test_detects_best_fee_adjusted_direction_and_caps_depth(self):
        result = evaluate_cross_venue_pair(
            self.a, self.b, rules_verified=True,
            fee_a_per_contract=0.005, fee_b_per_contract=0.005,
            slippage_a_per_contract=0.002, slippage_b_per_contract=0.002,
            max_contracts=20,
        )
        self.assertTrue(result.eligible)
        self.assertEqual(result.yes_venue, "Kalshi")
        self.assertEqual(result.no_venue, "Polymarket")
        self.assertEqual(result.contracts, 7)
        self.assertAlmostEqual(result.net_profit_per_pair, 0.051)
        self.assertTrue(result.paper_only)
        self.assertFalse(result.live_trading)

    def test_refuses_unverified_market_rules(self):
        result = evaluate_cross_venue_pair(
            self.a, self.b, rules_verified=False,
            fee_a_per_contract=0, fee_b_per_contract=0,
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "market_resolution_rules_not_verified")

    def test_fees_and_slippage_can_remove_apparent_edge(self):
        result = evaluate_cross_venue_pair(
            self.a, self.b, rules_verified=True,
            fee_a_per_contract=0.03, fee_b_per_contract=0.03,
            slippage_a_per_contract=0.02, slippage_b_per_contract=0.02,
        )
        self.assertFalse(result.eligible)

    def test_requires_both_sides_to_have_displayed_depth(self):
        b = dict(self.b, no_ask_size=0)
        result = evaluate_cross_venue_pair(
            self.a, b, rules_verified=True,
            fee_a_per_contract=0, fee_b_per_contract=0,
        )
        self.assertFalse(result.eligible)

    def test_rejects_bad_quotes_and_invalid_configuration(self):
        bad = dict(self.a, yes_ask=1.2)
        with self.assertRaises(ValueError):
            evaluate_cross_venue_pair(
                bad, self.b, rules_verified=True,
                fee_a_per_contract=0, fee_b_per_contract=0,
            )
        with self.assertRaises(ValueError):
            evaluate_cross_venue_pair(
                self.a, self.b, rules_verified=True,
                fee_a_per_contract=-0.01, fee_b_per_contract=0,
            )


if __name__ == "__main__":
    unittest.main()
