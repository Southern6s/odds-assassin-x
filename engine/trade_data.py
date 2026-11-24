"""Normalization and immutable snapshots for public Kalshi trade prints."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def normalize_trade(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("trade must be an object")
    required = ("trade_id", "ticker", "count_fp", "yes_price_dollars",
                "no_price_dollars", "taker_outcome_side", "taker_book_side",
                "created_time", "is_block_trade")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"trade missing required fields: {', '.join(missing)}")
    trade_id, ticker = raw["trade_id"], raw["ticker"]
    if not isinstance(trade_id, str) or not trade_id.strip():
        raise ValueError("trade_id must be a non-empty string")
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")
    try:
        count = float(raw["count_fp"])
        yes_price = float(raw["yes_price_dollars"])
        no_price = float(raw["no_price_dollars"])
    except (TypeError, ValueError) as exc:
        raise ValueError("trade count and prices must be numeric") from exc
    if count <= 0:
        raise ValueError("trade count must be positive")
    if not 0 <= yes_price <= 1 or not 0 <= no_price <= 1:
        raise ValueError("trade prices must be between 0 and 1")
    outcome_side = raw["taker_outcome_side"]
    book_side = raw["taker_book_side"]
    if outcome_side not in ("yes", "no") or book_side not in ("bid", "ask"):
        raise ValueError("invalid taker direction fields")
    if (outcome_side == "yes") != (book_side == "bid"):
        raise ValueError("taker outcome side and book side disagree")
    if not isinstance(raw["is_block_trade"], bool):
        raise ValueError("is_block_trade must be boolean")
    try:
        created = datetime.fromisoformat(raw["created_time"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("created_time must be an ISO-8601 timestamp") from exc
    if created.tzinfo is None:
        raise ValueError("created_time must include a timezone")
    return {
        "trade_id": trade_id,
        "ticker": ticker,
        "count": count,
        "yes_price": yes_price,
        "no_price": no_price,
        "taker_outcome_side": outcome_side,
        "taker_book_side": book_side,
        "created_time": created.astimezone(timezone.utc).isoformat(),
        "is_block_trade": raw["is_block_trade"],
    }


def trade_snapshot(
    raw: dict[str, Any], *, captured_at: datetime | None = None
) -> dict[str, Any]:
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    return {
        "schema_version": 1,
        "captured_at": captured.astimezone(timezone.utc).isoformat(),
        "trade": normalize_trade(raw),
        "raw": raw,
    }
