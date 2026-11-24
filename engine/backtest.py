"""Leakage-resistant evaluation primitives for historical binary-market forecasts.

This module evaluates timestamped forecasts against resolved outcomes. It does
not fetch data, train models, or submit orders. Callers must supply forecasts
that were generated before resolution and preserve the original decision-time
quotes. Fees are explicit inputs; no fee schedule is assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Iterable

from engine.probability import brier_score, log_loss


@dataclass(frozen=True)
class BacktestRow:
    market_id: str
    forecast_at: datetime
    resolved_at: datetime
    probability_yes: float
    outcome_yes: int
    yes_ask: float | None = None
    no_ask: float | None = None


@dataclass(frozen=True)
class BacktestReport:
    observations: int
    brier: float
    logloss: float
    accuracy: float
    mean_forecast: float
    observed_frequency: float
    calibration: tuple[dict, ...]
    paper_trades: int
    paper_net_pnl: float
    paper_roi: float | None


def _unit(value: float, field: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{field} must be finite and between 0 and 1")
    return value


def _validate_rows(rows: Iterable[BacktestRow]) -> list[BacktestRow]:
    items = list(rows)
    if not items:
        raise ValueError("at least one resolved observation is required")
    ids: set[str] = set()
    for row in items:
        if not row.market_id.strip():
            raise ValueError("market_id must not be empty")
        if row.market_id in ids:
            raise ValueError(f"duplicate market_id: {row.market_id}")
        ids.add(row.market_id)
        if row.forecast_at.tzinfo is None or row.resolved_at.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        if row.forecast_at >= row.resolved_at:
            raise ValueError("forecast_at must be earlier than resolved_at")
        _unit(row.probability_yes, "probability_yes")
        if row.outcome_yes not in (0, 1):
            raise ValueError("outcome_yes must be 0 or 1")
        if row.yes_ask is not None:
            _unit(row.yes_ask, "yes_ask")
        if row.no_ask is not None:
            _unit(row.no_ask, "no_ask")
    return sorted(items, key=lambda row: row.forecast_at)


def calibration_bins(rows: Iterable[BacktestRow], bins: int = 10) -> tuple[dict, ...]:
    """Return equal-width reliability bins; empty bins are omitted."""
    if bins < 2 or bins > 100:
        raise ValueError("bins must be between 2 and 100")
    items = list(rows)
    buckets: list[list[BacktestRow]] = [[] for _ in range(bins)]
    for row in items:
        p = _unit(row.probability_yes, "probability_yes")
        index = min(bins - 1, int(p * bins))
        buckets[index].append(row)
    result = []
    for index, bucket in enumerate(buckets):
        if bucket:
            result.append({
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "count": len(bucket),
                "mean_probability": sum(r.probability_yes for r in bucket) / len(bucket),
                "observed_frequency": sum(r.outcome_yes for r in bucket) / len(bucket),
            })
    return tuple(result)


def evaluate(
    rows: Iterable[BacktestRow],
    *,
    edge_threshold: float = 0.05,
    fee_per_contract: float = 0.0,
    calibration_bucket_count: int = 10,
) -> BacktestReport:
    """Evaluate forecasts and a simple one-contract paper strategy.

    A YES/NO paper position is entered only when forecast-vs-ask edge meets
    edge_threshold. P&L is realized payout minus entry ask and explicit fee.
    This is a diagnostic simulation, not a fill model: slippage, queue position,
    partial fills, and changing liquidity are not modeled.
    """
    if not math.isfinite(edge_threshold) or not 0 <= edge_threshold <= 1:
        raise ValueError("edge_threshold must be between 0 and 1")
    if not math.isfinite(fee_per_contract) or fee_per_contract < 0:
        raise ValueError("fee_per_contract must be finite and nonnegative")

    items = _validate_rows(rows)
    count = len(items)
    brier = sum(brier_score(r.probability_yes, r.outcome_yes) for r in items) / count
    losses = sum(log_loss(r.probability_yes, r.outcome_yes) for r in items) / count
    correct = sum((r.probability_yes >= 0.5) == bool(r.outcome_yes) for r in items)
    mean_p = sum(r.probability_yes for r in items) / count
    observed = sum(r.outcome_yes for r in items) / count

    trades = 0
    pnl = 0.0
    capital = 0.0
    for row in items:
        p_yes = row.probability_yes
        candidates = []
        if row.yes_ask is not None and p_yes - row.yes_ask >= edge_threshold:
            candidates.append((p_yes - row.yes_ask, row.yes_ask, row.outcome_yes))
        p_no = 1.0 - p_yes
        if row.no_ask is not None and p_no - row.no_ask >= edge_threshold:
            candidates.append((p_no - row.no_ask, row.no_ask, 1 - row.outcome_yes))
        if candidates:
            _, entry_price, won = max(candidates, key=lambda item: item[0])
            trades += 1
            capital += entry_price + fee_per_contract
            pnl += won - entry_price - fee_per_contract

    return BacktestReport(
        observations=count,
        brier=brier,
        logloss=losses,
        accuracy=correct / count,
        mean_forecast=mean_p,
        observed_frequency=observed,
        calibration=calibration_bins(items, calibration_bucket_count),
        paper_trades=trades,
        paper_net_pnl=pnl,
        paper_roi=(pnl / capital if capital > 0 else None),
    )
