import json
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError

from engine.kalshi_client import (
    KalshiAPIError,
    KalshiPublicClient,
    enrich_market_category,
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

    def test_http_error_is_wrapped(self):
        opener = Mock(side_effect=HTTPError(
            "https://example.test/markets", 503, "Unavailable", {}, None
        ))
        client = KalshiPublicClient(opener=opener)
        with self.assertRaises(KalshiAPIError):
            list(client.iter_markets(max_pages=1))


if __name__ == "__main__":
    unittest.main()
