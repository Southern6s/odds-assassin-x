"""Read-only client for Kalshi's public market-data endpoints.

Uses only the Python standard library. No authentication, account, order, or
trading methods are included. Pagination is cursor-based and guarded.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import time
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
        retry_attempts: int = 5,
        retry_backoff_seconds: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if isinstance(retry_attempts, bool) or retry_attempts < 0:
            raise ValueError("retry_attempts must be nonnegative")
        if retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be nonnegative")
        self.base_url = base_url.rstrip("/")
        self.opener = opener
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.sleeper = sleeper

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}/{quote(path.lstrip('/'), safe='/')}"
        if params:
            url = f"{url}?{urlencode(params)}"
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "OddsAssassinX/0.2"},
            method="GET",
        )
        for attempt in range(self.retry_attempts + 1):
            try:
                with self.opener(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except HTTPError as exc:
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.retry_attempts:
                    raise KalshiAPIError(f"Kalshi request failed: {exc}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                delay = None
                if retry_after:
                    try:
                        delay = max(0.0, float(retry_after))
                    except (TypeError, ValueError):
                        try:
                            retry_at = parsedate_to_datetime(retry_after)
                            delay = max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                        except (TypeError, ValueError, OverflowError):
                            delay = None
                if delay is None:
                    delay = min(self.retry_backoff_seconds * (2 ** attempt), 30.0)
                self.sleeper(delay)
            except (URLError, TimeoutError, OSError) as exc:
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

    def get_market_page(self, endpoint: str, *, cursor: str | None = None, limit: int = 1000) -> tuple[list[dict[str, Any]], str | None]:
        """Fetch exactly one validated page from an allowed market tier."""
        if endpoint not in ("markets", "historical/markets"):
            raise ValueError("endpoint must be markets or historical/markets")
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if cursor is not None and (not isinstance(cursor, str) or not cursor):
            raise ValueError("cursor must be a non-empty string or None")
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        payload = self._get(endpoint, params)
        markets = payload.get("markets")
        if not isinstance(markets, list) or any(not isinstance(item, dict) for item in markets):
            raise KalshiAPIError(f"Kalshi {endpoint} response contains an invalid markets list")
        next_cursor = payload.get("cursor") or None
        if next_cursor is not None and (not isinstance(next_cursor, str) or next_cursor == cursor):
            raise KalshiAPIError("Kalshi returned an invalid or unchanged pagination cursor")
        return markets, next_cursor

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
        """Yield unique markets from live and archived tiers, deduplicated by ticker."""
        seen: set[str] = set()
        for market in chain(
            self.iter_markets(limit=limit, max_pages=max_pages_per_tier),
            self.iter_historical_markets(limit=limit, max_pages=max_pages_per_tier),
        ):
            ticker = market.get("ticker")
            if isinstance(ticker, str) and ticker:
                if ticker in seen:
                    continue
                seen.add(ticker)
            yield market

    def iter_markets_by_tickers(
        self, tickers: list[str], *, batch_size: int = 100
    ) -> Iterator[dict[str, Any]]:
        """Fetch matching markets from both live and historical tiers.

        Ticker batches keep query strings bounded and make trade-to-market
        hydration deterministic instead of relying on unrelated global pages.
        """
        if not 1 <= batch_size <= 100:
            raise ValueError("batch_size must be between 1 and 100")
        unique = list(dict.fromkeys(
            ticker.strip() for ticker in tickers
            if isinstance(ticker, str) and ticker.strip()
        ))
        seen: set[str] = set()
        for offset in range(0, len(unique), batch_size):
            batch = unique[offset:offset + batch_size]
            params = {"tickers": ",".join(batch), "limit": batch_size}
            for endpoint in ("markets", "historical/markets"):
                payload = self._get(endpoint, params)
                markets = payload.get("markets")
                if not isinstance(markets, list):
                    raise KalshiAPIError(f"Kalshi {endpoint} response is missing a markets list")
                for market in markets:
                    if not isinstance(market, dict):
                        raise KalshiAPIError("Kalshi returned a malformed market record")
                    ticker = market.get("ticker")
                    if isinstance(ticker, str) and ticker not in seen:
                        seen.add(ticker)
                        yield market

    def iter_categorized_markets(
        self, *, limit: int = 1000, max_pages_per_tier: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield unique markets enriched only with authoritative series categories."""
        categories = series_category_map(self.get_series())
        for market in self.iter_all_market_tiers(
            limit=limit, max_pages_per_tier=max_pages_per_tier
        ):
            yield enrich_market_category(market, categories)


    def _iter_trades_endpoint(
        self,
        endpoint: str,
        *,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        is_block_trade: bool | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield trade records from a live or historical tier."""
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        if max_pages is not None and max_pages < 1:
            raise ValueError("max_pages must be positive")
        if min_ts is not None and min_ts < 0:
            raise ValueError("min_ts must be nonnegative Unix seconds")
        if max_ts is not None and max_ts < 0:
            raise ValueError("max_ts must be nonnegative Unix seconds")
        if min_ts is not None and max_ts is not None and max_ts <= min_ts:
            raise ValueError("max_ts must be greater than min_ts")
        cursor: str | None = None
        seen_cursors: set[str] = set()
        page_number = 0
        while True:
            params: dict[str, Any] = {"limit": limit}
            if ticker:
                params["ticker"] = ticker
            if min_ts is not None:
                params["min_ts"] = min_ts
            if max_ts is not None:
                params["max_ts"] = max_ts
            if is_block_trade is not None:
                params["is_block_trade"] = str(is_block_trade).lower()
            if cursor:
                params["cursor"] = cursor
            payload = self._get(endpoint, params)
            trades = payload.get("trades")
            if not isinstance(trades, list):
                raise KalshiAPIError(f"Kalshi {endpoint} response is missing a trades list")
            for trade in trades:
                if not isinstance(trade, dict):
                    raise KalshiAPIError("Kalshi returned a malformed trade record")
                yield trade
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

    def iter_trades(
        self, *, ticker: str | None = None, min_ts: int | None = None,
        max_ts: int | None = None, is_block_trade: bool | None = None,
        limit: int = 1000, max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield recent trades from the live trade tier."""
        yield from self._iter_trades_endpoint(
            "markets/trades", ticker=ticker, min_ts=min_ts, max_ts=max_ts,
            is_block_trade=is_block_trade, limit=limit, max_pages=max_pages,
        )

    def iter_historical_trades(
        self, *, ticker: str | None = None, min_ts: int | None = None,
        max_ts: int | None = None, is_block_trade: bool | None = None,
        limit: int = 1000, max_pages: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield archived trades from the historical tier."""
        yield from self._iter_trades_endpoint(
            "historical/trades", ticker=ticker, min_ts=min_ts, max_ts=max_ts,
            is_block_trade=is_block_trade, limit=limit, max_pages=max_pages,
        )

    def iter_all_trade_tiers(
        self, *, ticker: str | None = None, min_ts: int | None = None,
        max_ts: int | None = None, is_block_trade: bool | None = None,
        limit: int = 1000, max_pages_per_tier: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield unique trades across live and archive tiers, keyed by trade_id."""
        seen: set[str] = set()
        for trade in chain(
            self.iter_trades(
                ticker=ticker, min_ts=min_ts, max_ts=max_ts,
                is_block_trade=is_block_trade, limit=limit,
                max_pages=max_pages_per_tier,
            ),
            self.iter_historical_trades(
                ticker=ticker, min_ts=min_ts, max_ts=max_ts,
                is_block_trade=is_block_trade, limit=limit,
                max_pages=max_pages_per_tier,
            ),
        ):
            trade_id = trade.get("trade_id")
            if isinstance(trade_id, str) and trade_id:
                if trade_id in seen:
                    continue
                seen.add(trade_id)
            yield trade

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

    def get_historical_market_candlesticks(
        self,
        *,
        market_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict[str, Any]:
        """Fetch candlesticks for a market in Kalshi's historical archive."""
        if not market_ticker:
            raise ValueError("market_ticker is required")
        if start_ts < 0 or end_ts <= start_ts:
            raise ValueError("invalid candle time range")
        if period_interval not in (1, 60, 1440):
            raise ValueError("period_interval must be 1, 60, or 1440")
        return self._get(f"historical/markets/{market_ticker}/candlesticks", {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        })



def enrich_market_series(
    market: dict[str, Any], series_records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach authoritative series metadata by matching the event ticker prefix.

    Kalshi event tickers begin with their series ticker followed by a hyphen.
    Only exact prefixes present in the returned series catalog are accepted.
    """
    enriched = dict(market)
    event_ticker = enriched.get("event_ticker")
    if not enriched.get("series_ticker") and isinstance(event_ticker, str):
        matches = [
            item for item in series_records
            if isinstance(item.get("ticker"), str)
            and event_ticker.startswith(item["ticker"] + "-")
        ]
        if matches:
            series = max(matches, key=lambda item: len(item["ticker"]))
            enriched["series_ticker"] = series["ticker"]
            if not enriched.get("category") and isinstance(series.get("category"), str):
                enriched["category"] = series["category"]
    return enriched

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
