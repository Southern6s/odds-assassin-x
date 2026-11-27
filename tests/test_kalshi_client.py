import json
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from engine.kalshi_client import (
    KalshiAPIError,
    KalshiPublicClient,
    enrich_market_category,
    enrich_market_series,
    series_category_map,
)


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.body


class KalshiClientTests(unittest.TestCase):
    def client_with_pages(self, pages):
        opener = Mock(side_effect=[FakeResponse(page) for page in pages])
        return KalshiPublicClient(opener=opener), opener

    def test_iter_markets_follows_all_cursor_pages(self):
        client, opener = self.client_with_pages([
            {"markets": [{"ticker": "A"}], "cursor": "page-2"},
            {"markets": [{"ticker": "B"}], "cursor": ""},
        ])
        markets = list(client.iter_markets(status="open", limit=500))
        self.assertEqual([m["ticker"] for m in markets], ["A", "B"])
        self.assertEqual(opener.call_count, 2)
        self.assertIn("cursor=page-2", opener.call_args_list[1].args[0].full_url)

    def test_get_market_orderbook_uses_public_read_only_endpoint(self):
        client, opener = self.client_with_pages([
            {"orderbook": {"yes": [[45, 3]], "no": [[50, 4]]}},
        ])
        result = client.get_market_orderbook("KXTEST-26NOV27")
        self.assertIn("orderbook", result)
        self.assertTrue(opener.call_args.args[0].full_url.endswith(
            "/markets/KXTEST-26NOV27/orderbook"
        ))

    def test_get_market_page_passes_live_status_filter(self):
        client, opener = self.client_with_pages([
            {"markets": [{"ticker": "OPEN"}], "cursor": ""},
        ])
        markets, cursor = client.get_market_page(
            "markets", limit=1000, status="open", mve_filter="exclude"
        )
        self.assertEqual([item["ticker"] for item in markets], ["OPEN"])
        self.assertIsNone(cursor)
        self.assertIn("status=open", opener.call_args.args[0].full_url)
        self.assertIn("mve_filter=exclude", opener.call_args.args[0].full_url)

    def test_get_market_page_rejects_status_on_historical_tier(self):
        client = KalshiPublicClient(opener=Mock())
        with self.assertRaises(ValueError):
            client.get_market_page("historical/markets", status="open")

    def test_all_market_tiers_include_archived_markets(self):
        client, opener = self.client_with_pages([
            {"markets": [{"ticker": "LIVE"}, {"ticker": "DUP"}], "cursor": ""},
            {"markets": [{"ticker": "ARCHIVED"}, {"ticker": "DUP"}], "cursor": ""},
        ])
        markets = list(client.iter_all_market_tiers())
        self.assertEqual([m["ticker"] for m in markets], ["LIVE", "DUP", "ARCHIVED"])
        self.assertTrue(opener.call_args_list[0].args[0].full_url.endswith("/markets?limit=1000"))
        self.assertTrue(opener.call_args_list[1].args[0].full_url.endswith("/historical/markets?limit=1000"))


    def test_iter_markets_by_tickers_batches_and_checks_both_tiers(self):
        client, opener = self.client_with_pages([
            {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": ""},
            {"markets": [{"ticker": "A"}, {"ticker": "B"}], "cursor": ""},
            {"markets": [{"ticker": "C"}], "cursor": ""},
            {"markets": [{"ticker": "C"}], "cursor": ""},
        ])
        markets = list(client.iter_markets_by_tickers(["A", "B", "A", "C"], batch_size=2))
        self.assertEqual([m["ticker"] for m in markets], ["A", "B", "C"])
        self.assertEqual(opener.call_count, 4)
        self.assertIn("tickers=A%2CB", opener.call_args_list[0].args[0].full_url)
        self.assertIn("/historical/markets?", opener.call_args_list[1].args[0].full_url)
        self.assertIn("tickers=C", opener.call_args_list[2].args[0].full_url)

    def test_categorized_markets_use_series_metadata(self):
        opener = Mock(side_effect=[
            FakeResponse({"series": [{"ticker": "SERIES-A", "category": "Sports"}]}),
            FakeResponse({"markets": [{"ticker": "M1", "series_ticker": "SERIES-A"}], "cursor": ""}),
            FakeResponse({"markets": [], "cursor": ""}),
        ])
        client = KalshiPublicClient(opener=opener)
        markets = list(client.iter_categorized_markets())
        self.assertEqual(markets[0]["category"], "Sports")

    def test_max_pages_is_an_explicit_safe_limit(self):
        client, opener = self.client_with_pages([
            {"markets": [{"ticker": "A"}], "cursor": "next"},
        ])
        self.assertEqual(list(client.iter_markets(max_pages=1)), [{"ticker": "A"}])
        self.assertEqual(opener.call_count, 1)

    def test_repeated_cursor_is_rejected(self):
        client, _ = self.client_with_pages([
            {"markets": [], "cursor": "same"},
            {"markets": [], "cursor": "same"},
        ])
        with self.assertRaises(KalshiAPIError):
            list(client.iter_markets())

    def test_invalid_limit_and_candle_range_are_rejected(self):
        client = KalshiPublicClient(opener=Mock())
        with self.assertRaises(ValueError):
            list(client.iter_markets(limit=1001))
        with self.assertRaises(ValueError):
            client.get_market_candlesticks(
                series_ticker="S", market_ticker="M",
                start_ts=10, end_ts=10, period_interval=60,
            )


    def test_historical_candlestick_endpoint_and_validation(self):
        client, opener = self.client_with_pages([
            {"ticker": "M1", "candlesticks": []},
        ])
        result = client.get_historical_market_candlesticks(
            market_ticker="M1", start_ts=100, end_ts=200, period_interval=60
        )
        self.assertEqual(result["ticker"], "M1")
        url = opener.call_args.args[0].full_url
        self.assertIn("/historical/markets/M1/candlesticks", url)
        self.assertIn("period_interval=60", url)
        with self.assertRaises(ValueError):
            client.get_historical_market_candlesticks(
                market_ticker="M1", start_ts=200, end_ts=100
            )

    def test_market_series_enrichment_uses_longest_authoritative_event_prefix(self):
        series = [
            {"ticker": "KXBTC", "category": "Crypto"},
            {"ticker": "KXBTC15M", "category": "Crypto Short Term"},
        ]
        result = enrich_market_series(
            {"ticker": "KXBTC15M-26NOV240615-15", "event_ticker": "KXBTC15M-26NOV240615"},
            series,
        )
        self.assertEqual(result["series_ticker"], "KXBTC15M")
        self.assertEqual(result["category"], "Crypto Short Term")

    def test_market_series_enrichment_does_not_guess_unknown_prefix(self):
        result = enrich_market_series(
            {"ticker": "UNKNOWN-1", "event_ticker": "UNKNOWN-EVENT"},
            [{"ticker": "KXBTC", "category": "Crypto"}],
        )
        self.assertNotIn("series_ticker", result)
        self.assertNotIn("category", result)

    def test_series_categories_enrich_without_title_guessing(self):
        mapping = series_category_map([
            {"ticker": "SERIES-A", "category": "Sports"},
            {"ticker": "SERIES-B", "category": "Economics"},
            {"ticker": "NO-CATEGORY"},
        ])
        result = enrich_market_category(
            {"ticker": "M1", "series_ticker": "SERIES-A"}, mapping
        )
        self.assertEqual(result["category"], "Sports")
        self.assertIsNone(enrich_market_category(
            {"ticker": "M2", "series_ticker": "UNKNOWN"}, mapping
        )["category"])
        self.assertEqual(enrich_market_category(
            {"ticker": "M3", "category": "Existing", "series_ticker": "SERIES-A"}, mapping
        )["category"], "Existing")

    def test_rate_limit_retries_and_honors_retry_after(self):
        opener = Mock(side_effect=[
            HTTPError(
                "https://example.test/markets", 429, "Too Many Requests",
                {"Retry-After": "2"}, None,
            ),
            FakeResponse({"markets": [{"ticker": "A"}], "cursor": ""}),
        ])
        sleeper = Mock()
        client = KalshiPublicClient(opener=opener, sleeper=sleeper)
        self.assertEqual(list(client.iter_markets(max_pages=1)), [{"ticker": "A"}])
        self.assertEqual(opener.call_count, 2)
        sleeper.assert_called_once_with(2.0)

    def test_http_error_is_wrapped(self):
        opener = Mock(side_effect=HTTPError(
            "https://example.test/markets", 503, "Unavailable", {}, None
        ))
        client = KalshiPublicClient(opener=opener)
        with self.assertRaises(KalshiAPIError):
            list(client.iter_markets(max_pages=1))


if __name__ == "__main__":
    unittest.main()
