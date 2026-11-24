"""Score the decision-time Kalshi midpoint as a probability baseline.

Offline research only. This module scores quotes against later settlement
labels; it does not generate forecasts or submit orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from engine.probability import brier_score, log_loss


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


def _probability(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def evaluate_market_baseline(
    rows: Iterable[Mapping[str, Any]], *, calibration_bins: int = 10
) -> dict[str, Any]:
    """Evaluate midpoint forecasts and a neutral 0.5 benchmark.

    Rows without a two-sided YES midpoint are excluded from scoring but counted
    in coverage. Each market ticker may appear only once per supplied cutoff.
    """
    if isinstance(calibration_bins, bool) or not isinstance(calibration_bins, int) or not 2 <= calibration_bins <= 100:
        raise ValueError("calibration_bins must be an integer between 2 and 100")

    items = list(rows)
    if not items:
        raise ValueError("at least one research row is required")
    seen: set[tuple[str, datetime]] = set()
    scored: list[tuple[float, int, float | None]] = []
    for row in items:
        ticker = row.get("market_ticker")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("each row requires market_ticker")
        forecast_at = _time(row.get("forecast_at"), "forecast_at")
        key = (ticker, forecast_at)
        if key in seen:
            raise ValueError(f"duplicate market_ticker and forecast_at: {ticker}")
        seen.add(key)
        resolved_at = _time(row.get("resolved_at"), "resolved_at")
        if forecast_at >= resolved_at:
            raise ValueError("forecast_at must precede resolved_at")
        label = row.get("probability_label_yes")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("probability_label_yes must be 0 or 1")
        midpoint = row.get("market_yes_mid")
        spread = None
        if midpoint is not None:
            midpoint = _probability(midpoint, "market_yes_mid")
            bid, ask = row.get("market_yes_bid"), row.get("market_yes_ask")
            if bid is not None and ask is not None:
                bid, ask = _probability(bid, "market_yes_bid"), _probability(ask, "market_yes_ask")
                if bid > ask:
                    raise ValueError("YES bid cannot exceed YES ask")
                spread = ask - bid
            scored.append((midpoint, int(label), spread))

    count = len(scored)
    result: dict[str, Any] = {
        "rows_total": len(items),
        "rows_scored": count,
        "quote_coverage": count / len(items),
        "baseline": "decision_time_yes_midpoint",
        "paper_only": True,
        "live_trading": False,
    }
    if not count:
        result.update({
            "brier": None, "logloss": None, "accuracy": None,
            "neutral_brier": None, "neutral_logloss": None,
            "brier_skill_vs_neutral": None, "mean_probability": None,
            "observed_yes_frequency": None, "mean_yes_spread": None,
            "calibration": [],
        })
        return result

    brier = sum(brier_score(p, y) for p, y, _ in scored) / count
    losses = sum(log_loss(p, y) for p, y, _ in scored) / count
    neutral_brier = sum(brier_score(0.5, y) for _, y, _ in scored) / count
    neutral_logloss = sum(log_loss(0.5, y) for _, y, _ in scored) / count
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(calibration_bins)]
    for p, y, _ in scored:
        buckets[min(calibration_bins - 1, int(p * calibration_bins))].append((p, y))
    calibration = []
    for index, bucket in enumerate(buckets):
        if bucket:
            calibration.append({
                "lower": index / calibration_bins,
                "upper": (index + 1) / calibration_bins,
                "count": len(bucket),
                "mean_probability": sum(p for p, _ in bucket) / len(bucket),
                "observed_yes_frequency": sum(y for _, y in bucket) / len(bucket),
            })
    spreads = [spread for _, _, spread in scored if spread is not None]
    result.update({
        "brier": brier,
        "logloss": losses,
        "accuracy": sum((p >= 0.5) == bool(y) for p, y, _ in scored) / count,
        "neutral_brier": neutral_brier,
        "neutral_logloss": neutral_logloss,
        "brier_skill_vs_neutral": (
            1 - brier / neutral_brier if neutral_brier > 0 else None
        ),
        "mean_probability": sum(p for p, _, _ in scored) / count,
        "observed_yes_frequency": sum(y for _, y, _ in scored) / count,
        "mean_yes_spread": sum(spreads) / len(spreads) if spreads else None,
        "calibration": calibration,
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Kalshi midpoint baseline (offline)")
    parser.add_argument("--dataset", required=True, help="Research dataset JSONL")
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args(argv)
    rows = []
    with Path(args.dataset).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                parser.error(f"invalid JSON on line {line_number}: {exc}")
            if not isinstance(row, dict):
                parser.error(f"line {line_number} must contain a JSON object")
            rows.append(row)
    try:
        report = evaluate_market_baseline(rows, calibration_bins=args.calibration_bins)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
