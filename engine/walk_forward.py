"""Chronological walk-forward scoring for the experimental event model.

Every fold fits only on observations whose outcomes settled by the first
forecast timestamp in that fold. No orders or execution simulation are run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from .logistic_model import fit_logistic_model
from .probability import brier_score, log_loss


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


def _score(probabilities: list[float], labels: list[int]) -> dict[str, float]:
    count = len(labels)
    if not count:
        raise ValueError("cannot score an empty test fold")
    return {
        "observations": count,
        "brier": sum(brier_score(p, y) for p, y in zip(probabilities, labels)) / count,
        "logloss": sum(log_loss(p, y) for p, y in zip(probabilities, labels)) / count,
        "accuracy": sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, labels)) / count,
        "mean_probability": sum(probabilities) / count,
        "observed_yes_frequency": sum(labels) / count,
    }


def _diagnostic_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    """Score diagnostic slices using identical observations for model and market."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(group_key)
        if value is None:
            continue
        groups.setdefault(str(value), []).append(row)
    summaries = []
    for group, items in sorted(groups.items()):
        if len(items) < 20:
            continue
        labels = [int(item["outcome_yes"]) for item in items]
        model = _score([float(item["model_probability"]) for item in items], labels)
        neutral = _score([0.5] * len(items), labels)
        market_items = [item for item in items if item.get("market_yes_mid") is not None]
        market = (
            _score([float(item["market_yes_mid"]) for item in market_items],
                   [int(item["outcome_yes"]) for item in market_items])
            if market_items else None
        )
        model_matched = (
            _score([float(item["model_probability"]) for item in market_items],
                   [int(item["outcome_yes"]) for item in market_items])
            if market_items else None
        )
        neutral_matched = (
            _score([0.5] * len(market_items),
                   [int(item["outcome_yes"]) for item in market_items])
            if market_items else None
        )
        summaries.append({
            "group": group,
            "model": model,
            "neutral_50": neutral,
            "model_market_matched": model_matched,
            "neutral_market_matched": neutral_matched,
            "market_midpoint": market,
            "model_brier_skill_vs_market": (
                1 - model_matched["brier"] / market["brier"]
                if model_matched and market and market["brier"] > 0 else None
            ),
        })
    return summaries


def _diagnostic_band(value: float | None, edges: tuple[float, ...], labels: tuple[str, ...]) -> str | None:
    if value is None or not math.isfinite(float(value)):
        return None
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def evaluate_walk_forward(
    rows: Iterable[Mapping[str, Any]],
    feature_names: Iterable[str],
    *,
    min_train_size: int = 100,
    test_cutoffs: int = 1,
    step_cutoffs: int | None = None,
    include_market_features: bool = True,
    allow_empty_event_features: bool = False,
) -> dict[str, Any]:
    """Evaluate by distinct forecast cutoffs, fitting once per test window."""
    names = tuple(feature_names)
    if len(set(names)) != len(names):
        raise ValueError("feature_names must be unique")
    if not names and not allow_empty_event_features:
        raise ValueError("feature_names must be non-empty")
    if not isinstance(allow_empty_event_features, bool):
        raise ValueError("allow_empty_event_features must be boolean")
    if isinstance(min_train_size, bool) or min_train_size < 2:
        raise ValueError("min_train_size must be at least 2")
    if isinstance(test_cutoffs, bool) or test_cutoffs < 1:
        raise ValueError("test_cutoffs must be positive")
    step = test_cutoffs if step_cutoffs is None else step_cutoffs
    if isinstance(step, bool) or step < 1:
        raise ValueError("step_cutoffs must be positive")

    items = list(rows)
    if "market_midpoint" in names or "market_quote_present" in names:
        raise ValueError("market_midpoint and market_quote_present are reserved model features")
    if not isinstance(include_market_features, bool):
        raise ValueError("include_market_features must be boolean")
    # Give the model the market's contemporaneous probability as a baseline
    # feature, plus an explicit missing-quote indicator. Missing quotes use a
    # neutral placeholder but are distinguishable from a genuine 50% quote.
    augmented_items = []
    for source in items:
        row = dict(source)
        midpoint = row.get("market_yes_mid")
        if midpoint is None:
            row["feature_market_midpoint"] = 0.5
            row["feature_market_quote_present"] = 0.0
        else:
            try:
                midpoint_value = float(midpoint)
            except (TypeError, ValueError) as exc:
                raise ValueError("market_yes_mid must be numeric when present") from exc
            if not math.isfinite(midpoint_value) or not 0 <= midpoint_value <= 1:
                raise ValueError("market_yes_mid must be between 0 and 1")
            row["feature_market_midpoint"] = midpoint_value
            row["feature_market_quote_present"] = 1.0
        augmented_items.append(row)
    items = augmented_items
    model_names = names + (("market_midpoint", "market_quote_present") if include_market_features else ())
    if not items:
        raise ValueError("at least one research row is required")
    cutoffs = set()
    for row in items:
        forecast = _time(row.get("forecast_at"), "forecast_at")
        resolved = _time(row.get("resolved_at"), "resolved_at")
        if forecast >= resolved:
            raise ValueError("forecast_at must precede resolved_at")
        label = row.get("probability_label_yes")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("probability_label_yes must be 0 or 1")
        cutoffs.add(forecast)
        for name in names:
            key = name if name.startswith("feature_") else f"feature_{name}"
            raw = row.get(key)
            if raw is None or isinstance(raw, bool):
                raise ValueError(f"missing or invalid feature: {key}")
            try:
                number = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"feature must be numeric: {key}") from exc
            if not math.isfinite(number):
                raise ValueError(f"feature must be finite: {key}")
    ordered = sorted(cutoffs)
    folds = []
    prediction_rows: list[dict[str, Any]] = []
    skipped_single_class_folds = 0
    for start in range(0, len(ordered), step):
        window = ordered[start:start + test_cutoffs]
        if len(window) < test_cutoffs:
            break
        cutoff = window[0]
        training = [
            row for row in items
            if _time(row["resolved_at"], "resolved_at") <= cutoff
        ]
        if len(training) < min_train_size:
            continue
        # Early history can have enough repeated observations but only one
        # resolved outcome class. Such a window cannot fit logistic regression;
        # skip it and resume once both YES and NO outcomes have matured.
        if len({int(row["probability_label_yes"]) for row in training}) < 2:
            skipped_single_class_folds += 1
            continue
        testing = [
            row for row in items
            if _time(row["forecast_at"], "forecast_at") in window
        ]
        if not testing:
            continue
        model = fit_logistic_model(
            training, model_names, as_of=cutoff, min_training_rows=min_train_size
        )
        labels = [int(row["probability_label_yes"]) for row in testing]
        model_probs = [model.predict_probability(row) for row in testing]
        market_pairs = [
            (float(row["market_yes_mid"]), label)
            for row, label in zip(testing, labels)
            if row.get("market_yes_mid") is not None
        ]
        neutral_probs = [0.5] * len(labels)
        fold = {
            "test_start": window[0].isoformat(),
            "test_end": window[-1].isoformat(),
            "training_rows": model.training_rows,
            "training_yes_rate": model.training_yes_rate,
            "model": _score(model_probs, labels),
            "neutral_50": _score(neutral_probs, labels),
            "model_market_matched": (
                _score(
                    [probability for row, probability in zip(testing, model_probs)
                     if row.get("market_yes_mid") is not None],
                    [label for row, label in zip(testing, labels)
                     if row.get("market_yes_mid") is not None],
                )
                if market_pairs else None
            ),
            "neutral_market_matched": (
                _score([0.5] * len(market_pairs), [y for _, y in market_pairs])
                if market_pairs else None
            ),
            "market_midpoint": (
                _score([p for p, _ in market_pairs], [y for _, y in market_pairs])
                if market_pairs else None
            ),
            "market_quote_coverage": len(market_pairs) / len(testing),
        }
        folds.append(fold)
        for row, probability in zip(testing, model_probs):
            market_mid = row.get("market_yes_mid")
            bid, ask = row.get("market_yes_bid"), row.get("market_yes_ask")
            spread = (
                float(ask) - float(bid)
                if bid is not None and ask is not None else None
            )
            confidence = abs(float(probability) - 0.5)
            prediction_rows.append({
                "market_ticker": row.get("market_ticker"),
                "event_ticker": row.get("event_ticker"),
                "forecast_at": row.get("forecast_at"),
                "resolved_at": row.get("resolved_at"),
                "outcome_yes": int(row["probability_label_yes"]),
                "model_probability": float(probability),
                "market_yes_mid": float(market_mid) if market_mid is not None else None,
                "market_yes_bid": float(bid) if bid is not None else None,
                "market_yes_ask": float(ask) if ask is not None else None,
                "market_no_bid": (
                    float(row["market_no_bid"]) if row.get("market_no_bid") is not None else None
                ),
                "market_no_ask": (
                    float(row["market_no_ask"]) if row.get("market_no_ask") is not None else None
                ),
                "market_yes_bid_size": row.get("market_yes_bid_size"),
                "market_yes_ask_size": row.get("market_yes_ask_size"),
                "market_no_bid_size": row.get("market_no_bid_size"),
                "market_no_ask_size": row.get("market_no_ask_size"),
                "spread": spread,
                "quote_age_seconds": row.get("quote_age_seconds"),
                "feature_age_seconds": row.get("feature_age_seconds"),
                "confidence": confidence,
                "confidence_band": _diagnostic_band(
                    confidence, (0.05, 0.10, 0.20),
                    ("0.00-0.05", "0.05-0.10", "0.10-0.20", "0.20+")
                ),
                "spread_band": _diagnostic_band(
                    spread, (0.02, 0.05, 0.10),
                    ("0.00-0.02", "0.02-0.05", "0.05-0.10", "0.10+")
                ),
                "quote_age_band": _diagnostic_band(
                    float(row["quote_age_seconds"]) if row.get("quote_age_seconds") is not None else None,
                    (30, 120, 300),
                    ("0-30s", "30-120s", "120-300s", "300s+")
                ),
                "feature_age_band": _diagnostic_band(
                    float(row["feature_age_seconds"]) if row.get("feature_age_seconds") is not None else None,
                    (3600, 21600, 86400),
                    ("0-1h", "1-6h", "6-24h", "24h+")
                ),
            })
    if not folds:
        raise ValueError(
            "no valid folds: add more forecast cutoffs and matured training outcomes"
        )

    def aggregate(metric: str, key: str) -> dict[str, float] | None:
        parts = [fold[key] for fold in folds if fold[key] is not None]
        total = sum(part["observations"] for part in parts)
        if not total:
            return None
        return {
            "observations": total,
            metric: sum(part[metric] * part["observations"] for part in parts) / total,
        }

    return {
        "folds": folds,
        "model_features": list(model_names),
        "market_baseline_features": (["market_midpoint", "market_quote_present"] if include_market_features else []),
        "skipped_single_class_folds": skipped_single_class_folds,
        "aggregate": {
            "model_brier": aggregate("brier", "model"),
            "neutral_brier": aggregate("brier", "neutral_50"),
            "model_market_matched_brier": aggregate("brier", "model_market_matched"),
            "neutral_market_matched_brier": aggregate("brier", "neutral_market_matched"),
            "market_midpoint_brier": aggregate("brier", "market_midpoint"),
            "model_logloss": aggregate("logloss", "model"),
            "model_market_matched_logloss": aggregate("logloss", "model_market_matched"),
            "market_midpoint_logloss": aggregate("logloss", "market_midpoint"),
        },
        "paper_only": True,
        "live_trading": False,
        "execution_simulated": False,
        "diagnostics": {
            "by_event": _diagnostic_summary(prediction_rows, "event_ticker"),
            "by_model_confidence": _diagnostic_summary(prediction_rows, "confidence_band"),
            "by_market_spread": _diagnostic_summary(prediction_rows, "spread_band"),
            "by_quote_age": _diagnostic_summary(prediction_rows, "quote_age_band"),
            "by_feature_age": _diagnostic_summary(prediction_rows, "feature_age_band"),
            "minimum_group_observations": 20,
            "note": (
                "Groups with fewer than 20 observations are omitted. "
                "Repeated market/cutoff observations are not independent. "
                "These are diagnostic slices, not proof of tradable edge."
            ),
        },
        "prediction_rows": prediction_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk-forward evaluate research rows (paper-only; no orders)"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--features", required=True,
                        help="Comma-separated allowlist of feature names")
    parser.add_argument("--min-train-size", type=int, default=100)
    parser.add_argument("--test-cutoffs", type=int, default=1)
    parser.add_argument("--step-cutoffs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        rows = []
        with Path(args.dataset).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {line_number} must be a JSON object")
                rows.append(row)
        report = evaluate_walk_forward(
            rows, [item.strip() for item in args.features.split(",") if item.strip()],
            min_train_size=args.min_train_size,
            test_cutoffs=args.test_cutoffs,
            step_cutoffs=args.step_cutoffs,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Walk-forward evaluation failed: {exc}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
