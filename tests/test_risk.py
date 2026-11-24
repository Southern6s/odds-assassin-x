import unittest

from engine.risk import RiskLimits, size_position


class RiskTests(unittest.TestCase):
    def setUp(self):
        self.limits = RiskLimits(
            max_trade_risk=10,
            max_portfolio_risk=25,
            max_contracts=20,
            kelly_fraction=0.25,
            min_edge=0.05,
            daily_loss_limit=5,
        )

    def test_sizes_positive_edge_with_hard_caps(self):
        result = size_position(
            bankroll=1000, portfolio_risk=0, daily_pnl=0,
            model_probability=0.70, entry_price=0.50,
            fee_per_contract=0.01, limits=self.limits,
        )
        self.assertTrue(result.approved)
        self.assertGreaterEqual(result.contracts, 1)
        self.assertLessEqual(result.risk_dollars, 10)
        self.assertLessEqual(result.contracts, 20)

    def test_rejects_edge_below_threshold(self):
        result = size_position(
            bankroll=1000, portfolio_risk=0, daily_pnl=0,
            model_probability=0.53, entry_price=0.50,
            fee_per_contract=0.01, limits=self.limits,
        )
        self.assertFalse(result.approved)
        self.assertEqual(result.reason, "edge_below_minimum")

    def test_daily_loss_stop(self):
        result = size_position(
            bankroll=1000, portfolio_risk=0, daily_pnl=-5,
            model_probability=0.80, entry_price=0.50,
            fee_per_contract=0.01, limits=self.limits,
        )
        self.assertEqual(result.reason, "daily_loss_limit_reached")

    def test_portfolio_cap(self):
        result = size_position(
            bankroll=1000, portfolio_risk=25, daily_pnl=0,
            model_probability=0.80, entry_price=0.50,
            fee_per_contract=0.01, limits=self.limits,
        )
        self.assertEqual(result.reason, "portfolio_risk_limit_reached")

    def test_rejects_invalid_prices(self):
        with self.assertRaises(ValueError):
            size_position(
                bankroll=1000, portfolio_risk=0, daily_pnl=0,
                model_probability=0.8, entry_price=1,
                fee_per_contract=0, limits=self.limits,
            )


if __name__ == "__main__":
    unittest.main()
