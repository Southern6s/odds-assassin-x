import unittest
from unittest.mock import Mock

import requests

from engine.kalshi_client import (
    KalshiAPIError,
    KalshiPublicClient,
    enrich_market_category,
    series_category_map,
)


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class KalshiClientTests(unittest.TestCase):
    def client_with_pages(self, pages):
        session = Mock()
        session.get.side_effect = [FakeResponse(page) for page in pages]
        return KalshiPublicClient(session=session), session

    def test_iter_markets_follows_all_cursor_pages(self):
        client, session = self.client_with_pages([
            {"markets": [{"ticker": "A"}], "cursor": "page-2"},
            {"markets": [{"ticker": "B"}], "cursor": ""},
        ])
        markets = list(client.iter_markets(status="open", limit=500))
        self.assertEqual([m["ticker"] for m in markets], ["A", "B"])
        self.assertEqual(session.get.call_count, 2)
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["cursor"], "page-2")

    def test_max_pages_is_an_explicit_safe_limit(self):
        client, session = self.client_with_pages([
            {"markets": [{"ticker": "A"}], "cursor": "next"},
        ])
        self.assertEqual(list(client.iter_markets(max_pages=1)), [{"ticker": "A"}])
        self.assertEqual(session.get.call_count, 1)

    def test_repeated_cursor_is_rejected(self):
        client, _ = self.client_with_pages([
            {"markets": [], "cursor": "same"},
            {"markets": [], "cursor": "same"},
        ])
        with self.assertRaises(KalshiAPIError):
            list(client.iter_markets())

    def test_invalid_limit_and_candle_range_are_rejected(self):
        client = KalshiPublicClient(session=Mock())
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
        session = Mock()
        session.get.return_value = FakeResponse(
            {}, requests.HTTPError("503 Service Unavailable")
        )
        client = KalshiPublicClient(session=session)
        with self.assertRaises(KalshiAPIError):
            list(client.iter_markets(max_pages=1))


if __name__ == "__main__":
    unittest.main()
