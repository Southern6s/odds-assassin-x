"""Point-in-time per-market price features from historical Kalshi candles."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping


def _time(value: Any, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _close(value: Any) -> float | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("close")
    if raw is None:
        return None
    number = float(raw)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        return None
    return number


def build_market_candle_features(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    forecast_at: datetime,
    max_age_seconds: int = 86400,
    momentum_lookback: int = 5,
    reversion_lookback: int = 20,
) -> dict[str, dict[str, float | int | str]]:
    """Build market-specific trailing features using candles ended by cutoff.

    The current candle close is included only when its end_period_ts is at or
    before forecast_at. Momentum uses trailing logit return. Mean reversion
    compares the latest close to the preceding window (excluding current).
    Realized volatility is the population standard deviation of prior logit
    returns. No feature is emitted without enough valid observations.
    """
    cutoff = _time(forecast_at, "forecast_at")
    if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, int) or max_age_seconds < 0:
        raise ValueError("max_age_seconds must be a nonnegative integer")
    if momentum_lookback < 2 or reversion_lookback < 3:
        raise ValueError("lookbacks are too short")

    by_ticker: dict[str, list[tuple[datetime, float]]] = {}
    for index, snapshot in enumerate(snapshots, 1):
        candle = snapshot.get("candle", snapshot)
        if not isinstance(candle, Mapping):
            raise ValueError(f"candle snapshot {index} must contain a candle object")
        ticker = candle.get("market_ticker")
        end_ts = candle.get("end_period_ts")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError(f"candle snapshot {index} requires market_ticker")
        if isinstance(end_ts, bool) or not isinstance(end_ts, int) or end_ts <= 0:
            raise ValueError(f"candle snapshot {index} has invalid end_period_ts")
        ended = datetime.fromtimestamp(end_ts, timezone.utc)
        if ended > cutoff or (cutoff - ended).total_seconds() > max_age_seconds:
            continue
        bid = _close(candle.get("yes_bid"))
        ask = _close(candle.get("yes_ask"))
        price = _close(candle.get("price"))
        midpoint = (bid + ask) / 2 if bid is not None and ask is not None and bid <= ask else price
        if midpoint is None:
            continue
        by_ticker.setdefault(ticker, []).append((ended, midpoint))

    result = {}
    for ticker, points in by_ticker.items():
        # Deduplicate repeated candle records by their end timestamp.
        unique = {ended: price for ended, price in points}
        ordered = sorted(unique.items())
        prices = [price for _, price in ordered]
        features: dict[str, float | int | str] = {
            "market_candle_count": len(prices),
            "market_candle_latest_at": ordered[-1][0].isoformat(),
        }
        logits = [math.log(price / (1.0 - price)) for price in prices]
        if len(logits) >= momentum_lookback + 1:
            changes = [b - a for a, b in zip(logits[-(momentum_lookback + 1):-1], logits[-momentum_lookback:])]
            vol = pstdev(changes)
            features["market_momentum_logit_5"] = sum(changes)
            features["market_momentum_vol_scaled_5"] = (
                sum(changes) / (vol * math.sqrt(momentum_lookback)) if vol > 1e-12 else 0.0
            )
        if len(prices) >= reversion_lookback + 1:
            history = prices[-(reversion_lookback + 1):-1]
            scale = pstdev(history)
            features["market_mean_reversion_z20"] = (
                (mean(history) - prices[-1]) / scale if scale > 1e-12 else 0.0
            )
        if len(logits) >= reversion_lookback + 1:
            returns = [b - a for a, b in zip(logits[-(reversion_lookback + 1):-1], logits[-reversion_lookback:])]
            features["market_realized_volatility_20"] = pstdev(returns)
        result[ticker] = features
    return result
