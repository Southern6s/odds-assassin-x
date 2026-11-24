import unittest
from datetime import datetime, timedelta, timezone

from engine.backtest import BacktestRow, calibration_bins, evaluate


class BacktestTests(unittest.TestCase):
    def setUp(self):
        now = datetime.now(timezone.utc)
        self.rows = [
            BacktestRow("m1", now, now + timedelta(days=1), 0.8, 1, 0.7, 0.25),
            BacktestRow("m2", now + timedelta(seconds=1), now + timedelta(days=2), 0.2, 0, 0.3, 0.7),
        ]

    def test_scores_and_paper_trade_accounting(self):
        report = evaluate(self.rows, edge_threshold=0.05, fee_per_contract=0.01)
        self.assertEqual(report.observations, 2)
        self.assertEqual(report.paper_trades, 2)
        self.assertAlmostEqual(report.paper_net_pnl, 0.28)
        self.assertGreaterEqual(report.brier, 0)
        self.assertGreater(report.logloss, 0)

    def test_no_trade_without_ask_edge(self):
        rows = [
            BacktestRow("m1", self.rows[0].forecast_at, self.rows[0].resolved_at, 0.55, 1, 0.54, 0.45)
        ]
        report = evaluate(rows, edge_threshold=0.05)
        self.assertEqual(report.paper_trades, 0)
        self.assertIsNone(report.paper_roi)

    def test_rejects_time_leakage_and_duplicate_ids(self):
        row = self.rows[0]
        bad = BacktestRow("bad", row.resolved_at, row.forecast_at, 0.8, 1)
        with self.assertRaises(ValueError):
            evaluate([bad])
        with self.assertRaises(ValueError):
            evaluate([row, row])

    def test_rejects_naive_timestamps(self):
        row = self.rows[0]
        bad = BacktestRow("bad", datetime(2026, 1, 1), row.resolved_at, 0.5, 1)
        with self.assertRaises(ValueError):
            evaluate([bad])

    def test_calibration_bins(self):
        result = calibration_bins(self.rows, bins=5)
        self.assertEqual(sum(item["count"] for item in result), 2)


if __name__ == "__main__":
    unittest.main()
