import unittest
from datetime import datetime, timedelta, timezone

from engine.logistic_model import fit_logistic_model


class LogisticModelTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = datetime(2026, 1, 10, tzinfo=timezone.utc)
        self.rows = []
        for i in range(80):
            label = 1 if i % 2 == 0 else 0
            self.rows.append({
                "market_ticker": f"M{i}",
                "forecast_at": (self.cutoff - timedelta(days=3)).isoformat(),
                "resolved_at": (self.cutoff - timedelta(days=2)).isoformat(),
                "probability_label_yes": label,
                "feature_yes_bid_contracts": 10.0 if label else 1.0,
                "feature_no_bid_contracts": 1.0 if label else 10.0,
            })
        self.rows.append({
            "market_ticker": "FUTURE",
            "forecast_at": (self.cutoff - timedelta(hours=1)).isoformat(),
            "resolved_at": (self.cutoff + timedelta(days=1)).isoformat(),
            "probability_label_yes": 1,
            "feature_yes_bid_contracts": 9999.0,
            "feature_no_bid_contracts": 0.0,
        })

    def test_fit_uses_only_matured_rows_and_learns_signal(self):
        model = fit_logistic_model(
            self.rows, ["yes_bid_contracts", "no_bid_contracts"],
            as_of=self.cutoff, min_training_rows=50,
        )
        self.assertEqual(model.training_rows, 80)
        self.assertAlmostEqual(model.training_yes_rate, 0.5)
        yes = model.predict_probability({
            "feature_yes_bid_contracts": 10.0,
            "feature_no_bid_contracts": 1.0,
        })
        no = model.predict_probability({
            "feature_yes_bid_contracts": 1.0,
            "feature_no_bid_contracts": 10.0,
        })
        self.assertGreater(yes, 0.5)
        self.assertLess(no, 0.5)

    def test_requires_both_classes_and_minimum_matured_rows(self):
        with self.assertRaisesRegex(ValueError, "need at least"):
            fit_logistic_model(self.rows[:5], ["yes_bid_contracts"],
                               as_of=self.cutoff, min_training_rows=10)
        only_yes = [dict(row, probability_label_yes=1) for row in self.rows[:10]]
        with self.assertRaisesRegex(ValueError, "both YES and NO"):
            fit_logistic_model(only_yes, ["yes_bid_contracts"],
                               as_of=self.cutoff, min_training_rows=5)

    def test_rejects_missing_features_and_bad_fit_cutoff(self):
        model = fit_logistic_model(
            self.rows, ["yes_bid_contracts", "no_bid_contracts"],
            as_of=self.cutoff, min_training_rows=50,
        )
        with self.assertRaisesRegex(ValueError, "missing or non-numeric"):
            model.predict_probability({"feature_yes_bid_contracts": 1})
        with self.assertRaises(ValueError):
            fit_logistic_model(self.rows, ["yes_bid_contracts"],
                               as_of=datetime(2026, 1, 10), min_training_rows=50)


if __name__ == "__main__":
    unittest.main()
