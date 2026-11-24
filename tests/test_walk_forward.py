import unittest
from datetime import datetime, timedelta, timezone

from engine.walk_forward import evaluate_walk_forward


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.rows = []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for day in range(8):
            cutoff = base + timedelta(days=day)
            for index in range(20):
                label = int(index % 2 == 0)
                self.rows.append({
                    "market_ticker": f"D{day}-M{index}",
                    "forecast_at": cutoff.isoformat(),
                    "resolved_at": (cutoff + timedelta(hours=12)).isoformat(),
                    "probability_label_yes": label,
                    "feature_yes_bid_contracts": 10.0 if label else 1.0,
                    "feature_no_bid_contracts": 1.0 if label else 10.0,
                    "market_yes_mid": 0.5,
                })

    def test_uses_only_matured_labels_and_scores_market_comparator(self):
        # At day 2, days 0 and 1 have matured; day 2 has not.
        report = evaluate_walk_forward(
            self.rows, ["yes_bid_contracts", "no_bid_contracts"],
            min_train_size=30, test_cutoffs=1,
        )
        self.assertGreaterEqual(len(report["folds"]), 1)
        first = report["folds"][0]
        self.assertEqual(first["training_rows"], 40)
        self.assertEqual(first["model"]["observations"], 20)
        self.assertEqual(first["market_midpoint"]["observations"], 20)
        self.assertFalse(report["live_trading"])
        self.assertFalse(report["execution_simulated"])

    def test_rejects_missing_features_and_no_valid_folds(self):
        broken = [dict(row) for row in self.rows]
        broken[0].pop("feature_yes_bid_contracts")
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_walk_forward(
                broken, ["yes_bid_contracts", "no_bid_contracts"],
                min_train_size=30,
            )
        with self.assertRaisesRegex(ValueError, "no valid folds"):
            evaluate_walk_forward(
                self.rows[:20], ["yes_bid_contracts", "no_bid_contracts"],
                min_train_size=30,
            )

    def test_rejects_invalid_labels_and_feature_allowlist(self):
        bad = [dict(row) for row in self.rows]
        bad[0]["probability_label_yes"] = 0.5
        with self.assertRaises(ValueError):
            evaluate_walk_forward(
                bad, ["yes_bid_contracts"], min_train_size=10,
            )
        with self.assertRaises(ValueError):
            evaluate_walk_forward(self.rows, [], min_train_size=10)


if __name__ == "__main__":
    unittest.main()
