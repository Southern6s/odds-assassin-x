"""Normalization helpers for historical Kalshi candlestick data."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _ohlc(value: Any, field: str) -> dict[str, float | None] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    result: dict[str, float | None] = {}
    for key in ("open", "high", "low", "close", "mean", "previous"):
        raw = value.get(f"{key}_dollars", value.get(key))
        if raw is None:
            result[key] = None
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {field}.{key}") from exc
        if not 0 <= number <= 1:
            raise ValueError(f"{field}.{key} must be between 0 and 1")
        result[key] = number
    return result


def normalize_candlestick(
    candle: dict[str, Any],
    *,
    market_ticker: str,
    period_interval: int,
) -> dict[str, Any]:
    """Normalize a single OHLC observation while retaining the raw source row."""
    if not market_ticker.strip():
        raise ValueError("market_ticker is required")
    if period_interval not in (1, 60, 1440):
        raise ValueError("period_interval must be 1, 60, or 1440")
    end_ts = candle.get("end_period_ts")
    if not isinstance(end_ts, int) or end_ts <= 0:
        raise ValueError("end_period_ts must be a positive Unix timestamp")
    normalized = {
        "market_ticker": market_ticker,
        "end_period_ts": end_ts,
        "period_interval": period_interval,
        "yes_bid": _ohlc(candle.get("yes_bid"), "yes_bid"),
        "yes_ask": _ohlc(candle.get("yes_ask"), "yes_ask"),
        "price": _ohlc(candle.get("price"), "price"),
        "volume": candle.get("volume_fp", candle.get("volume")),
        "open_interest": candle.get("open_interest_fp", candle.get("open_interest")),
        "raw": candle,
    }
    bid, ask = normalized["yes_bid"], normalized["yes_ask"]
    if bid and ask:
        for key in ("open", "high", "low", "close"):
            if bid[key] is not None and ask[key] is not None and bid[key] > ask[key]:
                raise ValueError(f"crossed YES candle at {end_ts} ({key})")
    return normalized


def candlestick_snapshot(
    candle: dict[str, Any],
    *,
    market_ticker: str,
    period_interval: int,
    captured_at: datetime | None = None,
) -> dict[str, Any]:
    captured_at = captured_at or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    return {
        "schema_version": 1,
        "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
        "candle": normalize_candlestick(
            candle, market_ticker=market_ticker, period_interval=period_interval
        ),
    }
