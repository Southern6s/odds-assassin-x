"""Join pre-decision event trade features to subsequently resolved binary markets.

This creates labeled research observations, not forecasts. Only features from
fully completed buckets at or before forecast_at are eligible. Only binary
YES/NO markets with an actual settlement timestamp after forecast_at are used.
"""
from __future__ import annotations

from datetime import datetime, timezone
from math import isfinite
from typing import Any, Iterable, Mapping


def _timestamp(value: str | datetime, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _market_payload(item: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = item.get("market")
    return nested if isinstance(nested, Mapping) else item


def build_resolved_market_dataset(
    markets: Iterable[Mapping[str, Any]],
    event_features: Iterable[Mapping[str, Any]],
    *,
    forecast_at: datetime,
    max_feature_age_seconds: int = 86400,
) -> list[dict[str, Any]]:
    """Build one leakage-safe labeled observation per eligible binary market.

    For each event, selects the newest feature bucket that ended no later than
    forecast_at and is no older than max_feature_age_seconds. Market labels are
    included only when result is YES/NO and settlement_ts is strictly after the
    forecast timestamp. Unresolved, scalar, stale, and featureless rows are
    excluded. No outcome is inferred from prices or titles.
    """
    cutoff = _timestamp(forecast_at, "forecast_at")
    if (isinstance(max_feature_age_seconds, bool)
            or not isinstance(max_feature_age_seconds, int)
            or max_feature_age_seconds < 0):
        raise ValueError("max_feature_age_seconds must be a nonnegative integer")

    latest: dict[str, tuple[datetime, Mapping[str, Any]]] = {}
    for feature in event_features:
        event = feature.get("event_ticker")
        if not isinstance(event, str) or not event:
            raise ValueError("each feature row requires event_ticker")
        start = _timestamp(feature.get("bucket_start"), "bucket_start")
        seconds = feature.get("bucket_seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("bucket_seconds must be a positive integer")
        end = start.timestamp() + seconds
        if end > cutoff.timestamp():
            continue
        age = cutoff.timestamp() - end
        if age > max_feature_age_seconds:
            continue
        end_dt = datetime.fromtimestamp(end, timezone.utc)
        prior = latest.get(event)
        if prior is None or end_dt > prior[0]:
            latest[event] = (end_dt, feature)

    rows: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()
    for item in markets:
        market = _market_payload(item)
        ticker = market.get("ticker")
        event = market.get("event_ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("each market requires ticker")
        if ticker in seen_tickers:
            raise ValueError(f"duplicate market ticker: {ticker}")
        seen_tickers.add(ticker)
        result = market.get("result")
        if result not in ("yes", "no"):
            continue
        settled_raw = market.get("settlement_ts")
        if not settled_raw:
            continue
        settled_at = _timestamp(settled_raw, "settlement_ts")
        if settled_at <= cutoff:
            continue
        if not isinstance(event, str) or event not in latest:
            continue
        feature_end, feature = latest[event]
        age_seconds = (cutoff - feature_end).total_seconds()
        row = {
            "market_ticker": ticker,
            "event_ticker": event,
            "forecast_at": cutoff.isoformat(),
            "resolved_at": settled_at.isoformat(),
            "probability_label_yes": 1 if result == "yes" else 0,
            "feature_bucket_end": feature_end.isoformat(),
            "feature_age_seconds": age_seconds,
        }
        for key, value in feature.items():
            if key not in ("event_ticker", "bucket_start", "bucket_seconds", "diagnostics"):
                if isinstance(value, float) and not isfinite(value):
                    raise ValueError(f"feature {key} must be finite")
                row[f"feature_{key}"] = value
        rows.append(row)
    return sorted(rows, key=lambda row: row["market_ticker"])
