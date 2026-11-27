import unittest

from engine.paper_execution import PaperExecutionConfig, simulate_paper_execution


def row(ticker, time, p, bid, ask, bid_size, ask_size, *, age=10):
    return {
        "market_ticker": ticker,
        "forecast_at": f"2026-01-01T00:{time:02d}:00+00:00",
        "model_probability": p,
        "market_yes_bid": bid,
        "market_yes_ask": ask,
        "market_yes_bid_size": bid_size,
        "market_yes_ask_size": ask_size,
        "quote_age_seconds": age,
    }


class PaperExecutionTests(unittest.TestCase):
    def test_entry_then_partial_and_full_profit_exits(self):
        rows = [
            row("M1", 0, 0.80, 0.48, 0.50, 5, 5),
            row("M1", 1, 0.90, 0.65, 0.67, 2, 2),
            row("M1", 2, 0.90, 0.70, 0.72, 3, 3),
        ]
        report = simulate_paper_execution(rows)
        self.assertTrue(report["paper_only"])
        self.assertFalse(report["live_trading"])
        self.assertEqual(report["entry_count"], 1)
        self.assertEqual(report["exit_count"], 2)
        self.assertEqual(report["open_positions"], [])
        self.assertAlmostEqual(report["realized_net_pnl"], 0.70)
        self.assertEqual(report["trades"][1]["action"], "PARTIAL_EXIT")
        self.assertEqual(report["trades"][1]["contracts"], 2)

    def test_missing_depth_never_assumes_a_fill(self):
        data = row("M1", 0, 0.90, 0.45, 0.50, 0, 0)
        report = simulate_paper_execution([data])
        self.assertEqual(report["entry_count"], 0)
        self.assertEqual(report["blocked_signal_counts"]["missing_entry_ask_depth"], 1)

    def test_stale_quotes_are_rejected(self):
        data = row("M1", 0, 0.90, 0.45, 0.50, 10, 10, age=301)
        report = simulate_paper_execution([data])
        self.assertEqual(report["entry_count"], 0)
        self.assertEqual(report["blocked_signal_counts"]["stale_quote"], 1)

    def test_hard_trade_risk_cap_limits_contracts(self):
        data = row("M1", 0, 0.90, 0.45, 0.50, 100, 100)
        config = PaperExecutionConfig(max_trade_risk=2.0, max_portfolio_risk=5.0)
        report = simulate_paper_execution([data], config=config)
        self.assertEqual(report["trades"][0]["contracts"], 3)
        self.assertLessEqual(report["peak_open_risk"], 2.0)

    def test_no_side_uses_explicit_no_quotes_and_depth(self):
        data = row("M1", 0, 0.10, 0.45, 0.50, 5, 5)
        data.update({
            "market_no_bid": 0.45, "market_no_ask": 0.50,
            "market_no_bid_size": 5, "market_no_ask_size": 5,
        })
        report = simulate_paper_execution([data])
        self.assertEqual(report["entry_count"], 1)
        self.assertEqual(report["trades"][0]["side"], "no")

    def test_invalid_probability_is_rejected(self):
        data = row("M1", 0, 1.2, 0.45, 0.50, 5, 5)
        with self.assertRaises(ValueError):
            simulate_paper_execution([data])


if __name__ == "__main__":
    unittest.main()
