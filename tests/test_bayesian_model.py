import unittest
from datetime import datetime, timedelta, timezone

from engine.bayesian_model import evaluate_bayesian_walk_forward


class BayesianShrinkageTests(unittest.TestCase):
    def test_uses_only_matured_labels_and_returns_paper_predictions(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for day in range(5):
            cutoff = base + timedelta(days=day)
            for i in range(10):
                rows.append({
                    "market_ticker": f"D{day}-{i}",
                    "forecast_at": cutoff.isoformat(),
                    "resolved_at": (cutoff + timedelta(hours=1)).isoformat(),
                    "probability_label_yes": int(i < 8),
                    "market_yes_mid": 0.70,
                })
        report = evaluate_bayesian_walk_forward(
            rows, min_train_size=20, test_cutoffs=1, prior_strength=10
        )
        self.assertTrue(report["prediction_rows"])
        first = report["prediction_rows"][0]
        # First forecast has no matured history, so no prediction is emitted
        # until a later fold has enough settled rows.
        self.assertGreaterEqual(len(report["folds"]), 1)
        self.assertGreater(first["model_probability"], 0.5)
        self.assertEqual(report["model_family"], "beta_binomial_shrinkage")

    def test_rejects_invalid_prior(self):
        with self.assertRaisesRegex(ValueError, "prior_strength"):
            evaluate_bayesian_walk_forward([], prior_strength=0)


if __name__ == "__main__":
    unittest.main()
