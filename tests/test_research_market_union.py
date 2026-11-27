import unittest

from engine.research_market_union import merge_settlement_window_markets


class SettlementWindowMarketUnionTests(unittest.TestCase):
    @staticmethod
    def market(ticker, settlement, result="yes", category="Sports"):
        return {"market": {"ticker": ticker, "event_ticker": f"EV-{ticker}",
                           "market_type": "binary", "category": category,
                           "result": result, "settlement_ts": settlement}}

    def test_adds_only_resolved_ordinary_binary_markets_inside_window(self):
        hydrated = [
            self.market("TRADED", 150),
            self.market("HYDRATED_EARLY", 99),
            self.market("HYDRATED_EXOTICS", 150, category="Exotics"),
        ]
        catalog = [
            self.market("EARLY", 99), self.market("IN", 100), self.market("END", 200),
            self.market("LATE", 201), self.market("NO_RESULT", 150, result=None),
            self.market("EXOTICS", 150, category="Exotics"),
        ]
        rows, report = merge_settlement_window_markets(
            hydrated, catalog, min_ts=100, max_ts=200
        )
        self.assertEqual([row["market"]["ticker"] for row in rows], ["IN", "TRADED"])
        self.assertEqual(report["catalog_eligible_resolved_ordinary_binary_markets"], 1)
        self.assertEqual(report["trade_hydrated_ordinary_binary_resolved_in_window"], 1)
        self.assertEqual(report["trade_hydrated_markets_excluded"], 2)
        self.assertEqual(report["added_catalog_markets"], 1)
        self.assertFalse(report["live_trading"])

    def test_deduplicates_catalog_against_trade_hydration(self):
        hydrated = [self.market("SAME", 150)]
        hydrated[0]["market"]["source"] = "trade"
        catalog = [self.market("SAME", 150)]
        rows, report = merge_settlement_window_markets(
            hydrated, catalog, min_ts=100, max_ts=200
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market"]["source"], "trade")
        self.assertEqual(report["unique_markets_after_union"], 1)
        self.assertEqual(report["added_catalog_markets"], 0)

    def test_catalog_selection_is_bounded_and_spans_settlement_range(self):
        catalog = [self.market(f"M{i}", 100 + i) for i in range(10)]
        rows, report = merge_settlement_window_markets(
            [], catalog, min_ts=100, max_ts=200, max_catalog_markets=3
        )
        selected = [row["market"]["settlement_ts"] for row in rows]
        self.assertEqual(report["catalog_eligible_resolved_ordinary_binary_markets"], 10)
        self.assertEqual(report["catalog_markets_selected"], 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(selected, [100, 104, 109])

    def test_rejects_invalid_window_and_cap(self):
        with self.assertRaises(ValueError):
            merge_settlement_window_markets([], [], min_ts=10, max_ts=10)
        with self.assertRaises(ValueError):
            merge_settlement_window_markets([], [], min_ts=10, max_ts=20,
                                            max_catalog_markets=0)


if __name__ == "__main__":
    unittest.main()
