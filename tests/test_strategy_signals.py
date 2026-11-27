import unittest

from engine.strategy_signals import (
    combine_candidate_signals,
    mean_reversion_signal,
    momentum_signal,
    order_book_imbalance,
)


class CandidateSignalTests(unittest.TestCase):
    def test_momentum_detects_rising_and_falling_prices(self):
        rising = momentum_signal([0.40, 0.42, 0.45, 0.48, 0.52, 0.56], lookback=5)
        falling = momentum_signal([0.60, 0.57, 0.53, 0.49, 0.45, 0.41], lookback=5)
        self.assertGreater(rising.score, 0)
        self.assertLess(falling.score, 0)

    def test_mean_reversion_is_contrarian_to_prior_window(self):
        history = [0.48, 0.50, 0.49, 0.51, 0.50, 0.49]
        below = mean_reversion_signal(history + [0.30], lookback=6)
        above = mean_reversion_signal(history + [0.70], lookback=6)
        self.assertGreater(below.score, 0)
        self.assertLess(above.score, 0)

    def test_mean_reversion_excludes_current_price_from_reference(self):
        values = [0.40, 0.45, 0.50, 0.55, 0.60]
        result = mean_reversion_signal(values, lookback=4)
        self.assertEqual(result.observations, 4)

    def test_book_imbalance_is_normalized(self):
        self.assertAlmostEqual(order_book_imbalance(75, 25).score, 0.5)
        self.assertAlmostEqual(order_book_imbalance(25, 75).score, -0.5)
        self.assertFalse(order_book_imbalance(0, 0).eligible)

    def test_combination_is_a_score_not_a_probability(self):
        signals = [
            momentum_signal([0.40, 0.42, 0.45, 0.48, 0.52, 0.56]),
            order_book_imbalance(80, 20),
        ]
        combined = combine_candidate_signals(signals, {"momentum": 1, "order_book_imbalance": 3})
        self.assertTrue(combined.eligible)
        self.assertLessEqual(abs(combined.score), 5)
        self.assertIn("requires_walk_forward_validation", combined.reason)

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            momentum_signal([0.4, 0.5], lookback=5)
        with self.assertRaises(ValueError):
            order_book_imbalance(-1, 2)


if __name__ == "__main__":
    unittest.main()
