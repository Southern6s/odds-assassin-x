"""Event-level descriptive features derived from public trade prints.

These are research features, not probability forecasts or trading signals. Side
counts are kept separate; this module deliberately does not infer aggressor
direction or claim causal predictive value from the exchange side fields.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping


_SIDE_KEYS = (
    ("yes", "bid", "yes_bid_contracts"),
    ("yes", "ask", "yes_ask_contracts"),
    ("no", "bid", "no_bid_contracts"),
    ("no", "ask", "no_ask_contracts"),
)


def _aware_datetime(value: str | datetime, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def build_event_trade_features(
    trades: list[dict[str, Any]],
    market_to_event: Mapping[str, str],
    *,
    as_of: datetime,
    bucket_seconds: int = 3600,
) -> list[dict[str, Any]]:
    """Aggregate trade prints by event and UTC bucket, using only data <= as_of.

    Input trades may be normalized trade objects or trade snapshots containing a
    nested "trade". Unknown markets and trades after as_of are excluded and
    counted in the returned diagnostics.
    """
    cutoff = _aware_datetime(as_of, "as_of")
    if isinstance(bucket_seconds, bool) or not isinstance(bucket_seconds, int) or bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be a positive integer")

    groups: dict[tuple[str, int], dict[str, Any]] = {}
    seen_ids: set[str] = set()
    diagnostics = {"duplicates": 0, "unknown_market": 0, "future_trade": 0}
    for item in trades:
        trade = item.get("trade", item)
        if not isinstance(trade, dict):
            raise ValueError("each trade must be an object or trade snapshot")
        trade_id = trade.get("trade_id")
        if not isinstance(trade_id, str) or not trade_id:
            raise ValueError("each trade requires a non-empty trade_id")
        if trade_id in seen_ids:
            diagnostics["duplicates"] += 1
            continue
        seen_ids.add(trade_id)

        ticker = trade.get("ticker")
        event_ticker = market_to_event.get(ticker) if isinstance(ticker, str) else None
        if not isinstance(event_ticker, str) or not event_ticker:
            diagnostics["unknown_market"] += 1
            continue
        created = _aware_datetime(trade.get("created_time"), "created_time")
        if created > cutoff:
            diagnostics["future_trade"] += 1
            continue

        try:
            count = float(trade["count"])
            yes_price = float(trade["yes_price"])
            no_price = float(trade["no_price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("trade count and prices must be numeric") from exc
        if not all(isfinite(x) for x in (count, yes_price, no_price)):
            raise ValueError("trade count and prices must be finite")
        if count <= 0 or not 0 <= yes_price <= 1 or not 0 <= no_price <= 1:
            raise ValueError("trade count or prices are outside valid ranges")
        outcome_side, book_side = trade.get("taker_outcome_side"), trade.get("taker_book_side")
        if outcome_side not in ("yes", "no") or book_side not in ("bid", "ask"):
            raise ValueError("invalid taker outcome/book side")
        if (outcome_side == "yes") != (book_side == "bid"):
            raise ValueError("inconsistent taker outcome/book side")
        block = trade.get("is_block_trade")
        if not isinstance(block, bool):
            raise ValueError("is_block_trade must be boolean")

        epoch = int(created.timestamp())
        bucket_start = epoch - epoch % bucket_seconds
        key = (event_ticker, bucket_start)
        row = groups.setdefault(key, {
            "event_ticker": event_ticker,
            "bucket_start": datetime.fromtimestamp(bucket_start, timezone.utc).isoformat(),
            "bucket_seconds": bucket_seconds,
            "trade_count": 0,
            "contract_volume": 0.0,
            "block_contract_volume": 0.0,
            "yes_price_contract_sum": 0.0,
            "no_price_contract_sum": 0.0,
            "markets": set(),
            "trade_ids": set(),
            "yes_bid_contracts": 0.0,
            "yes_ask_contracts": 0.0,
            "no_bid_contracts": 0.0,
            "no_ask_contracts": 0.0,
        })
        row["trade_count"] += 1
        row["contract_volume"] += count
        if block:
            row["block_contract_volume"] += count
        row["yes_price_contract_sum"] += yes_price * count
        row["no_price_contract_sum"] += no_price * count
        row["markets"].add(ticker)
        row["trade_ids"].add(trade_id)
        side_key = next(name for outcome, book, name in _SIDE_KEYS
                        if (outcome, book) == (outcome_side, book_side))
        row[side_key] += count

    result = []
    for key in sorted(groups):
        row = groups[key]
        volume = row["contract_volume"]
        row["yes_vwap"] = row.pop("yes_price_contract_sum") / volume
        row["no_vwap"] = row.pop("no_price_contract_sum") / volume
        row["market_count"] = len(row.pop("markets"))
        row.pop("trade_ids")
        result.append(row)
    # Attach diagnostics to every result without mutating source records.
    for row in result:
        row["diagnostics"] = dict(diagnostics)
    return result
