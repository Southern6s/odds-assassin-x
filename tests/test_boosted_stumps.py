import unittest
from datetime import datetime, timedelta, timezone

from engine.boosted_stumps import evaluate_boosted_stumps_walk_forward


class BoostedStumpsTests(unittest.TestCase):
    def test_emits_chronological_paper_predictions(self):
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for day in range(7):
            cutoff = base + timedelta(days=day)
            for index in range(40):
                signal = float(index) / 39.0
                rows.append({
                    "market_ticker": f"D{day}-M{index}",
                    "forecast_at": cutoff.isoformat(),
                    "resolved_at": (cutoff + timedelta(hours=1)).isoformat(),
                    "probability_label_yes": int(signal >= 0.5),
                    "feature_signal": signal,
                    "market_yes_mid": 0.5,
                })
        report = evaluate_boosted_stumps_walk_forward(
            rows, ["signal"], min_train_size=40, min_leaf=3,
            max_stumps=10, learning_rate=0.1,
        )
        self.assertTrue(report["prediction_rows"])
        self.assertEqual(report["model_family"], "gradient_boosted_decision_stumps")
        self.assertTrue(report["paper_only"])
        self.assertFalse(report["live_trading"])
        self.assertFalse(report["execution_simulated"])
        first = report["prediction_rows"][0]
        self.assertGreater(first["model_probability"], 0.0)
        self.assertLess(first["model_probability"], 1.0)

    def test_rejects_invalid_labels(self):
        row = {
            "market_ticker": "M1",
            "forecast_at": "2026-01-01T00:00:00+00:00",
            "resolved_at": "2026-01-01T01:00:00+00:00",
            "probability_label_yes": 2,
            "feature_signal": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "0 or 1"):
            evaluate_boosted_stumps_walk_forward(
                [row], ["signal"], min_train_size=2
            )


if __name__ == "__main__":
    unittest.main()
