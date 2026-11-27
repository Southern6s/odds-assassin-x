import unittest

from engine.position_management import (
    ExitPolicy, ExitQuote, OpenPosition, evaluate_exit,
)


class PositionManagementTests(unittest.TestCase):
    def setUp(self):
        self.position = OpenPosition(
            market_id="MKT-1", side="yes", contracts=10,
            entry_price=0.40, entry_fee_per_contract=0.01,
        )

    def test_takes_profit_after_entry_exit_fees_and_slippage(self):
        result = evaluate_exit(
            self.position,
            ExitQuote(bid=0.50, bid_size=10, fee_per_contract=0.01,
                      slippage_per_contract=0.01),
        )
        self.assertEqual(result.action, "EXIT")
        self.assertEqual(result.reason, "take_profit")
        self.assertEqual(result.contracts_to_close, 10)
        self.assertAlmostEqual(result.net_pnl_per_contract, 0.07)
        self.assertAlmostEqual(result.estimated_net_pnl, 0.70)

    def test_does_not_call_gross_green_mark_a_net_profit(self):
        result = evaluate_exit(
            self.position,
            ExitQuote(bid=0.43, bid_size=10, fee_per_contract=0.01,
                      slippage_per_contract=0.01),
        )
        self.assertEqual(result.action, "HOLD")
        self.assertEqual(result.reason, "profit_target_not_reached_or_no_exit_signal")
        self.assertAlmostEqual(result.net_pnl_per_contract, 0.00)

    def test_partial_exit_respects_visible_bid_depth(self):
        result = evaluate_exit(
            self.position,
            ExitQuote(bid=0.50, bid_size=3, fee_per_contract=0.01),
        )
        self.assertEqual(result.action, "PARTIAL_EXIT")
        self.assertEqual(result.contracts_to_close, 3)
        self.assertEqual(result.remaining_contracts, 7)
        self.assertAlmostEqual(result.estimated_net_pnl, 0.24)

    def test_edge_reversal_can_exit_before_profit_target(self):
        result = evaluate_exit(
            self.position,
            ExitQuote(bid=0.45, bid_size=10),
            model_fair_probability=0.40,
        )
        self.assertEqual(result.reason, "model_edge_reversed")
        self.assertEqual(result.action, "EXIT")

    def test_configured_risk_stop_exits_losing_position(self):
        result = evaluate_exit(
            self.position,
            ExitQuote(bid=0.25, bid_size=10),
            policy=ExitPolicy(max_loss_per_contract=0.10),
        )
        self.assertEqual(result.reason, "risk_stop")
        self.assertEqual(result.action, "EXIT")

    def test_no_bid_depth_means_no_fill(self):
        result = evaluate_exit(
            self.position, ExitQuote(bid=0.60, bid_size=0)
        )
        self.assertEqual(result.action, "HOLD")
        self.assertEqual(result.contracts_to_close, 0)
        self.assertEqual(result.remaining_contracts, 10)

    def test_exit_engine_is_always_paper_only(self):
        result = evaluate_exit(
            self.position, ExitQuote(bid=0.60, bid_size=10)
        )
        self.assertTrue(result.paper_only)
        self.assertFalse(result.live_trading)

    def test_rejects_invalid_side_and_negative_costs(self):
        with self.assertRaises(ValueError):
            evaluate_exit(
                OpenPosition("MKT-1", "maybe", 1, 0.4),
                ExitQuote(0.5, 1),
            )
        with self.assertRaises(ValueError):
            evaluate_exit(
                self.position, ExitQuote(0.5, 1, slippage_per_contract=-0.01)
            )


if __name__ == "__main__":
    unittest.main()
