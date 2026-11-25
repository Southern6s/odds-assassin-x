import unittest

from engine.filter_research_universe import (
    filter_research_records, is_ordinary_binary_market,
)


class ResearchUniverseFilterTests(unittest.TestCase):
    def test_keeps_categorized_ordinary_binary_market(self):
        row = {"market": {
            "ticker": "KXNBA-GAME-TEAM", "event_ticker": "KXNBA-GAME",
            "market_type": "binary", "category": "Sports",
        }}
        self.assertTrue(is_ordinary_binary_market(row))

    def test_excludes_exotics_even_when_market_type_is_binary(self):
        row = {"market": {
            "ticker": "KXMVECROSSCATEGORY-ABC-YES",
            "event_ticker": "KXMVECROSSCATEGORY-ABC",
            "market_type": "binary", "category": "Exotics",
        }}
        self.assertFalse(is_ordinary_binary_market(row))

    def test_excludes_mve_prefix_and_uncategorized_markets(self):
        mve = {"market": {
            "ticker": "KXMVESPORTS-ABC-YES", "event_ticker": "KXMVESPORTS-ABC",
            "market_type": "binary", "category": "Sports",
        }}
        unknown = {"market": {
            "ticker": "KXUNKNOWN-ABC", "event_ticker": "KXUNKNOWN",
            "market_type": "binary", "category": None,
        }}
        self.assertFalse(is_ordinary_binary_market(mve))
        self.assertFalse(is_ordinary_binary_market(unknown))

    def test_filters_trades_and_candles_to_kept_market_tickers(self):
        markets = [
            {"market": {"ticker": "KEEP", "event_ticker": "EV1",
                        "market_type": "binary", "category": "Sports"}},
            {"market": {"ticker": "DROP", "event_ticker": "EV2",
                        "market_type": "binary", "category": "Exotics"}},
        ]
        trades = [
            {"trade": {"ticker": "KEEP", "trade_id": "T1"}},
            {"trade": {"ticker": "DROP", "trade_id": "T2"}},
        ]
        candles = [
            {"candle": {"market_ticker": "KEEP", "end_period_ts": 1}},
            {"candle": {"market_ticker": "DROP", "end_period_ts": 2}},
        ]
        kept_markets, kept_trades, kept_candles, report = filter_research_records(
            markets, trades, candles
        )
        self.assertEqual([x["market"]["ticker"] for x in kept_markets], ["KEEP"])
        self.assertEqual([x["trade"]["trade_id"] for x in kept_trades], ["T1"])
        self.assertEqual([x["candle"]["market_ticker"] for x in kept_candles], ["KEEP"])
        self.assertEqual(report["markets_excluded"], 1)
        self.assertEqual(report["trades_excluded"], 1)
        self.assertEqual(report["candles_excluded"], 1)
        self.assertFalse(report["live_trading"])


if __name__ == "__main__":
    unittest.main()
