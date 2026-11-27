"""Leakage-safe join of captured order-book depth to walk-forward predictions.

A snapshot is eligible only when it belongs to the same market, was captured
at or before the forecast, and is no older than the configured age limit.
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping


def _time(value: Any, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def attach_depth_snapshots(
    prediction_rows: Iterable[Mapping[str, Any]],
    depth_rows: Iterable[Mapping[str, Any]],
    *,
    max_age_seconds: float = 300.0,
) -> list[dict[str, Any]]:
    """Attach latest eligible depth to each prediction; never use future data.

    Existing panel bid/ask and size values are cleared first, so an unmatched
    depth snapshot cannot silently fall back to stale or non-depth quotes.
    Market midpoint remains intact for the predictive benchmark only.
    """
    if isinstance(max_age_seconds, bool):
        raise ValueError("max_age_seconds must be finite and nonnegative")
    try:
        max_age = float(max_age_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_age_seconds must be finite and nonnegative") from exc
    if not math.isfinite(max_age) or max_age < 0:
        raise ValueError("max_age_seconds must be finite and nonnegative")

    indexed: dict[str, list[tuple[datetime, Mapping[str, Any]]]] = {}
    for index, source in enumerate(depth_rows, 1):
        ticker = source.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError(f"depth row {index} requires ticker")
        captured = _time(source.get("captured_at"), "captured_at")
        indexed.setdefault(ticker, []).append((captured, source))

    for snapshots in indexed.values():
        snapshots.sort(key=lambda item: item[0])

    times = {
        ticker: [captured for captured, _ in snapshots]
        for ticker, snapshots in indexed.items()
    }
    result: list[dict[str, Any]] = []
    for source in prediction_rows:
        row = dict(source)
        ticker = row.get("market_ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("each prediction row requires market_ticker")
        forecast = _time(row.get("forecast_at"), "forecast_at")

        # Never let the panel's shallower quote fields masquerade as depth data.
        for side in ("yes", "no"):
            for book_side in ("bid", "ask"):
                row[f"market_{side}_{book_side}"] = None
                row[f"market_{side}_{book_side}_size"] = 0
        row["quote_age_seconds"] = None
        row["depth_snapshot_at"] = None
        row["depth_snapshot_age_seconds"] = None
        row["depth_snapshot_matched"] = False

        snapshots = indexed.get(ticker, [])
        if not snapshots:
            result.append(row)
            continue
        eligible_index = bisect_right(times[ticker], forecast) - 1
        if eligible_index < 0:
            result.append(row)
            continue
        captured, snapshot = snapshots[eligible_index]
        age = (forecast - captured).total_seconds()
        if age < 0 or age > max_age:
            result.append(row)
            continue

        for side in ("yes", "no"):
            for book_side in ("bid", "ask"):
                source_key = f"{side}_{book_side}"
                size_key = f"{source_key}_size"
                row[f"market_{source_key}"] = snapshot.get(source_key)
                row[f"market_{size_key}"] = snapshot.get(size_key) or 0
        row["quote_age_seconds"] = age
        row["depth_snapshot_at"] = captured.isoformat()
        row["depth_snapshot_age_seconds"] = age
        row["depth_snapshot_matched"] = True
        result.append(row)

    return result
