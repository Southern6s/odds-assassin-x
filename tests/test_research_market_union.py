import unittest

from engine.research_market_union import merge_settlement_window_markets


class SettlementWindowMarketUnionTests(unittest.TestCase):
    def test_adds_only_catalog_markets_settled_inside_window(self):
        hydrated = [{"market": {"ticker": "TRADED", "settlement_ts": 150}}]
        catalog = [
            {"market": {"ticker": "EARLY", "settlement_ts": 99}},
            {"market": {"ticker": "IN", "settlement_ts": 100}},
            {"market": {"ticker": "END", "settlement_ts": 200}},
            {"market": {"ticker": "LATE", "settlement_ts": 201}},
            {"market": {"ticker": "NO_DATE"}},
        ]
        rows, report = merge_settlement_window_markets(
            hydrated, catalog, min_ts=100, max_ts=200
        )
        self.assertEqual(
            [row["market"]["ticker"] for row in rows],
            ["IN", "TRADED"],
        )
        self.assertEqual(report["catalog_rows_in_settlement_window"], 1)
        self.assertEqual(report["added_catalog_markets"], 1)
        self.assertFalse(report["live_trading"])

    def test_deduplicates_catalog_against_trade_hydration(self):
        hydrated = [{"market": {"ticker": "SAME", "source": "trade"}}]
        catalog = [{"market": {"ticker": "SAME", "settlement_ts": 150,
                               "source": "catalog"}}]
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
