"""Read-only client for Kalshi's public market-data endpoints.

Uses only the Python standard library. No authentication, account, order, or
trading methods are included. Pagination is cursor-based and guarded.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


class KalshiAPIError(RuntimeError):
    """Raised when Kalshi returns an invalid or unsuccessful API response."""


class KalshiPublicClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        opener: Callable[..., Any] = urlopen,
        timeout: float = 15.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.opener = opener
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{quote(path.lstrip('/'), safe='/')}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "OddsAssassinX/0.2"},
            method="GET",
        )
        try:
            with self.opener(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            raise KalshiAPIError(f"Kalshi request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KalshiAPIError("Kalshi returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise KalshiAPIError("Kalshi response must be a JSON object")
        return payload

    def iter_markets(
        self,
        *,
        status: str | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every market matching the filter, following every cursor page."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_number = 0
        while True:
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            payload = self._get("markets", params)
            markets = payload.get("markets")
            if not isinstance(markets, list):
                raise KalshiAPIError("Kalshi markets response is missing a markets list")
            for market in markets:
                if not isinstance(market, dict):
                    raise KalshiAPIError("Kalshi returned a malformed market record")
                yield market
            page_number += 1
            next_cursor = payload.get("cursor") or None
            if not next_cursor:
                break
            if not isinstance(next_cursor, str) or next_cursor in seen_cursors:
                raise KalshiAPIError("Kalshi returned an invalid or repeated pagination cursor")
            seen_cursors.add(next_cursor)
            if max_pages is not None and page_number >= max_pages:
                break
            cursor = next_cursor

    def _iter_market_endpoint(
        self,
        endpoint: str,
        *,
        status: str | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_number = 0
        while True:
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            if cursor:
                params["cursor"] = cursor
            payload = self._get(endpoint, params)
            markets = payload.get("markets")
            if not isinstance(markets, list):
                raise KalshiAPIError(f"Kalshi {endpoint} response is missing a markets list")
            for market in markets:
                if not isinstance(market, dict):
                    raise KalshiAPIError("Kalshi returned a malformed market record")
                yield market
            page_number += 1
            next_cursor = payload.get("cursor") or None
            if not next_cursor:
                break
            if not isinstance(next_cursor, str) or next_cursor in seen_cursors:
                raise KalshiAPIError("Kalshi returned an invalid or repeated pagination cursor")
            seen_cursors.add(next_cursor)
            if max_pages is not None and page_number >= max_pages:
                break
            cursor = next_cursor

    def iter_markets(
        self,
        *,
        status: str | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield live-tier markets, following every cursor page."""
        yield from self._iter_market_endpoint(
            "markets", status=status, limit=limit, max_pages=max_pages
        )

    def iter_historical_markets(
        self, *, limit: int = 1000, max_pages: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield archived/older settled markets from Kalshi's historical tier."""
        yield from self._iter_market_endpoint(
            "historical/markets", limit=limit, max_pages=max_pages
        )

    def iter_all_market_tiers(
        self, *, limit: int = 1000, max_pages_per_tier: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield live-tier then archived-tier markets; consumers should dedupe by ticker."""
        yield from self.iter_markets(limit=limit, max_pages=max_pages_per_tier)
        yield from self.iter_historical_markets(
            limit=limit, max_pages=max_pages_per_tier
        )

    def get_series(self, *, category: str | None = None) -> list[dict[str, Any]]:
        """Fetch series metadata used as the authoritative category taxonomy."""
        params = {"category": category} if category else {}
        payload = self._get("series", params)
        series = payload.get("series")
        if not isinstance(series, list):
            raise KalshiAPIError("Kalshi series response is missing a series list")
        return [item for item in series if isinstance(item, dict)]

    def get_historical_cutoff(self) -> dict[str, Any]:
        """Read Kalshi's current live/archive boundary; do not hard-code it."""
        return self._get("historical/cutoff")

    def get_market_candlesticks(
        self,
        *,
        series_ticker: str,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict[str, Any]:
        """Fetch public candle history for a market in a series."""
        if not series_ticker or not market_ticker:
            raise ValueError("series_ticker and market_ticker are required")
        if start_ts < 0 or end_ts <= start_ts:
            raise ValueError("invalid candle time range")
        if period_interval not in (1, 60, 1440):
            raise ValueError("period_interval must be 1, 60, or 1440")
        path = f"series/{series_ticker}/markets/{market_ticker}/candlesticks"
        return self._get(path, {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        })


def series_category_map(series_records: list[dict[str, Any]]) -> dict[str, str]:
    """Map series ticker to its category without guessing from market titles."""
    result: dict[str, str] = {}
    for record in series_records:
        ticker, category = record.get("ticker"), record.get("category")
        if isinstance(ticker, str) and ticker and isinstance(category, str) and category:
            result[ticker] = category
    return result


def enrich_market_category(
    market: dict[str, Any], categories_by_series: dict[str, str]
) -> dict[str, Any]:
    """Copy a market and fill category only from its known series metadata."""
    enriched = dict(market)
    if not enriched.get("category"):
        series_ticker = enriched.get("series_ticker")
        if isinstance(series_ticker, str):
            enriched["category"] = categories_by_series.get(series_ticker)
    return enriched
