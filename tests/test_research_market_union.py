import unittest

from engine.research_market_union import merge_settlement_window_markets


class SettlementWindowMarketUnionTests(unittest.TestCase):
    def test_adds_only_resolved_ordinary_binary_markets_inside_window(self):
        hydrated = [{"market": {"ticker": "TRADED", "settlement_ts": 150}}]
        def market(ticker, settlement, result="yes", category="Sports"):
            return {"market": {"ticker": ticker, "event_ticker": f"EV-{ticker}",
                               "market_type": "binary", "category": category,
                               "result": result, "settlement_ts": settlement}}
        catalog = [
            market("EARLY", 99), market("IN", 100), market("END", 200),
            market("LATE", 201), market("NO_RESULT", 150, result=None),
            market("EXOTICS", 150, category="Exotics"),
        ]
        rows, report = merge_settlement_window_markets(
            hydrated, catalog, min_ts=100, max_ts=200
        )
        self.assertEqual([row["market"]["ticker"] for row in rows], ["IN", "TRADED"])
        self.assertEqual(report["catalog_resolved_ordinary_binary_markets_in_window"], 1)
        self.assertEqual(report["added_catalog_markets"], 1)
        self.assertFalse(report["live_trading"])

    def test_deduplicates_catalog_against_trade_hydration(self):
        hydrated = [{"market": {"ticker": "SAME", "source": "trade"}}]
        catalog = [{"market": {"ticker": "SAME", "event_ticker": "EV-SAME",
                               "market_type": "binary", "category": "Sports",
                               "result": "yes", "settlement_ts": 150}}]
        rows, report = merge_settlement_window_markets(
            hydrated, catalog, min_ts=100, max_ts=200
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market"]["source"], "trade")
        self.assertEqual(report["unique_markets_after_union"], 1)

    def test_rejects_invalid_window(self):
        with self.assertRaises(ValueError):
            merge_settlement_window_markets([], [], min_ts=10, max_ts=10)


if __name__ == "__main__":
    unittest.main()
