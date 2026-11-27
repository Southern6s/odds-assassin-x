import unittest
from datetime import datetime, timedelta, timezone

from engine.model_competition import evaluate_model_competition


class ModelCompetitionTests(unittest.TestCase):
    def test_compares_four_candidates_on_common_chronological_rows(self):
        rows = []
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for day in range(9):
            cutoff = base + timedelta(days=day)
            for index in range(30):
                label = int(index % 2 == 0)
                rows.append({
                    "market_ticker": f"D{day}-M{index}",
                    "event_ticker": f"E{index % 3}",
                    "category": ("Sports", "Politics", "Economics")[index % 3],
                    "forecast_at": cutoff.isoformat(),
                    "resolved_at": (cutoff + timedelta(hours=12)).isoformat(),
                    "probability_label_yes": label,
                    "feature_yes_bid_contracts": 10.0 if label else 1.0,
                    "feature_no_bid_contracts": 1.0 if label else 10.0,
                    "market_yes_mid": 0.75 if label else 0.25,
                })

        report = evaluate_model_competition(
            rows,
            ["yes_bid_contracts", "no_bid_contracts"],
            min_train_size=30,
            test_cutoffs=1,
        )
        self.assertEqual(set(report["candidate_models"]),
                         {"market_calibrated", "event_only", "combined", "bayesian_shrinkage"})
        self.assertEqual(report["common_out_of_sample_observations"],
                         report["market_quote_matched_observations"])
        self.assertEqual(len(report["leaderboard"]), 4)
        self.assertEqual({item["category"] for item in report["category_leaderboard"]},
                         {"Sports", "Politics", "Economics"})
        self.assertEqual(report["market_midpoint_baseline"]["observations"],
                         report["market_quote_matched_observations"])
        self.assertEqual(report["promotion_status"], "not_promoted_research_only")
        self.assertTrue(report["paper_only"])
        self.assertFalse(report["live_trading"])
        self.assertFalse(report["execution_simulated"])

    def test_rejects_empty_feature_allowlist(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            evaluate_model_competition([], [])


if __name__ == "__main__":
    unittest.main()
