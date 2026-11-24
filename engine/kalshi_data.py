"""Kalshi public-market ingestion and immutable JSONL snapshot helpers.

No credentials or trading endpoints are used. Values are normalized to decimal
probabilities while raw market payloads are retained for auditability.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable


def _decimal_price(market: dict[str, Any], dollars_key: str, cents_key: str) -> float | None:
    value = market.get(dollars_key)
    if value is None:
        value = market.get(cents_key)
        if value is None:
            return None
        try:
            number = float(value) / 100.0
        except (TypeError, ValueError):
            raise ValueError(f"invalid price field: {cents_key}")
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"invalid price field: {dollars_key}")
    if not 0 <= number <= 1:
        raise ValueError(f"price out of range: {dollars_key}/{cents_key}")
    return number


def normalize_market(market: dict[str, Any]) -> dict[str, Any]:
    """Normalize one Kalshi market without manufacturing missing quote values."""
    ticker = market.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("market ticker is required")

    normalized = {
        "ticker": ticker,
        "event_ticker": market.get("event_ticker"),
        "series_ticker": market.get("series_ticker"),
        "title": market.get("title") or market.get("subtitle"),
        "category": market.get("category"),
        "status": market.get("status"),
        "yes_bid": _decimal_price(market, "yes_bid_dollars", "yes_bid"),
        "yes_ask": _decimal_price(market, "yes_ask_dollars", "yes_ask"),
        "no_bid": _decimal_price(market, "no_bid_dollars", "no_bid"),
        "no_ask": _decimal_price(market, "no_ask_dollars", "no_ask"),
        "volume": market.get("volume_fp", market.get("volume")),
        "volume_24h": market.get("volume_24h_fp", market.get("volume_24h")),
        "open_interest": market.get("open_interest_fp", market.get("open_interest")),
        "close_time": market.get("close_time"),
        "expiration_time": market.get("expiration_time"),
    }
    for side in ("yes", "no"):
        bid, ask = normalized[f"{side}_bid"], normalized[f"{side}_ask"]
        if bid is not None and ask is not None and bid > ask:
            raise ValueError(f"crossed {side.upper()} quote for {ticker}")
    return normalized


def snapshot_record(market: dict[str, Any], captured_at: datetime | None = None) -> dict[str, Any]:
    """Wrap normalized and original market data with a timezone-aware capture time."""
    captured_at = captured_at or datetime.now(timezone.utc)
    if captured_at.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    return {
        "schema_version": 1,
        "captured_at": captured_at.astimezone(timezone.utc).isoformat(),
        "market": normalize_market(market),
        "raw": market,
    }


def append_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Append snapshot records; returns number written. Does not overwrite history."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
            count += 1
    return count
