"""Dependency-free gradient-boosted decision stumps for binary forecasts.

This is a deliberately shallow nonlinear candidate. Each chronological fold
fits only on outcomes settled by that fold's first forecast timestamp.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping


def _time(value: Any) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("timestamps must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _sigmoid(value: float) -> float:
    value = max(-35.0, min(35.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    probability = min(0.999, max(0.001, probability))
    return math.log(probability / (1.0 - probability))


def _features(row: Mapping[str, Any], names: tuple[str, ...]) -> list[float]:
    values = []
    for name in names:
        key = name if name.startswith("feature_") else f"feature_{name}"
        raw = row.get(key)
        if raw is None or isinstance(raw, bool):
            raise ValueError(f"missing or invalid feature: {key}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature must be numeric: {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"feature must be finite: {key}")
        values.append(value)
    return values


def _fit_stumps(
    training: list[dict[str, Any]],
    names: tuple[str, ...],
    *,
    max_stumps: int,
    learning_rate: float,
    min_leaf: int,
) -> tuple[float, list[tuple[int, float, float, float]]]:
    labels = [int(row["probability_label_yes"]) for row in training]
    if any(isinstance(row.get("probability_label_yes"), bool)
           or row.get("probability_label_yes") not in (0, 1) for row in training):
        raise ValueError("probability_label_yes must be 0 or 1")
    base = _logit(sum(labels) / len(labels))
    matrix = [_features(row, names) for row in training]
    scores = [base] * len(training)
    stumps: list[tuple[int, float, float, float]] = []

    for _ in range(max_stumps):
        residuals = [label - _sigmoid(score) for label, score in zip(labels, scores)]
        best = None
        best_loss = sum(value * value for value in residuals)
        for feature_index in range(len(names)):
            ordered = sorted({values[feature_index] for values in matrix})
            for left, right in zip(ordered, ordered[1:]):
                threshold = (left + right) / 2.0
                low_indices = [i for i, values in enumerate(matrix)
                               if values[feature_index] <= threshold]
                high_indices = [i for i, values in enumerate(matrix)
                                if values[feature_index] > threshold]
                if len(low_indices) < min_leaf or len(high_indices) < min_leaf:
                    continue
                low_value = sum(residuals[i] for i in low_indices) / len(low_indices)
                high_value = sum(residuals[i] for i in high_indices) / len(high_indices)
                loss = sum((residuals[i] - low_value) ** 2 for i in low_indices)
                loss += sum((residuals[i] - high_value) ** 2 for i in high_indices)
                if loss < best_loss - 1e-12:
                    best_loss = loss
                    best = (feature_index, threshold, low_value, high_value)
        if best is None:
            break
        feature_index, threshold, low_value, high_value = best
        low_step = learning_rate * low_value
        high_step = learning_rate * high_value
        stumps.append((feature_index, threshold, low_step, high_step))
        for i, values in enumerate(matrix):
            scores[i] += low_step if values[feature_index] <= threshold else high_step
    return base, stumps


def _predict(row: Mapping[str, Any], names: tuple[str, ...],
             base: float, stumps: list[tuple[int, float, float, float]]) -> float:
    values = _features(row, names)
    score = base
    for index, threshold, low_step, high_step in stumps:
        score += low_step if values[index] <= threshold else high_step
    return _sigmoid(score)


def evaluate_boosted_stumps_walk_forward(
    rows: Iterable[Mapping[str, Any]],
    feature_names: Iterable[str],
    *,
    min_train_size: int = 100,
    test_cutoffs: int = 1,
    step_cutoffs: int | None = None,
    max_stumps: int = 30,
    learning_rate: float = 0.05,
    min_leaf: int = 10,
) -> dict[str, Any]:
    """Evaluate a shallow boosted-stump ensemble using matured labels only."""
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if isinstance(min_train_size, bool) or min_train_size < 2:
        raise ValueError("min_train_size must be at least 2")
    if isinstance(test_cutoffs, bool) or test_cutoffs < 1:
        raise ValueError("test_cutoffs must be positive")
    step = test_cutoffs if step_cutoffs is None else step_cutoffs
    if isinstance(step, bool) or step < 1:
        raise ValueError("step_cutoffs must be positive")
    if isinstance(max_stumps, bool) or max_stumps < 1:
        raise ValueError("max_stumps must be positive")
    if not math.isfinite(learning_rate) or not 0 < learning_rate <= 0.5:
        raise ValueError("learning_rate must be in (0, 0.5]")
    if isinstance(min_leaf, bool) or min_leaf < 2:
        raise ValueError("min_leaf must be at least 2")

    items = [dict(row) for row in rows]
    if not items:
        raise ValueError("at least one research row is required")
    cutoffs = sorted({_time(row["forecast_at"]) for row in items})
    for row in items:
        forecast, resolved = _time(row["forecast_at"]), _time(row["resolved_at"])
        if forecast >= resolved:
            raise ValueError("forecast_at must precede resolved_at")
        label = row.get("probability_label_yes")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("probability_label_yes must be 0 or 1")
        _features(row, names)

    predictions = []
    folds = []
    skipped_single_class_folds = 0
    for start in range(0, len(cutoffs), step):
        window = cutoffs[start:start + test_cutoffs]
        if len(window) < test_cutoffs:
            break
        cutoff = window[0]
        training = [row for row in items if _time(row["resolved_at"]) <= cutoff]
        if len(training) < min_train_size:
            continue
        if len({int(row["probability_label_yes"]) for row in training}) < 2:
            skipped_single_class_folds += 1
            continue
        base, stumps = _fit_stumps(
            training, names, max_stumps=max_stumps,
            learning_rate=learning_rate, min_leaf=min_leaf,
        )
        testing = [row for row in items if _time(row["forecast_at"]) in window]
        fold_count = 0
        for row in testing:
            probability = _predict(row, names, base, stumps)
            raw = row.get("market_yes_mid")
            predictions.append({
                "market_ticker": row.get("market_ticker"),
                "event_ticker": row.get("event_ticker"),
                "category": row.get("category"),
                "series_ticker": row.get("series_ticker"),
                "forecast_at": row["forecast_at"],
                "resolved_at": row["resolved_at"],
                "outcome_yes": int(row["probability_label_yes"]),
                "model_probability": probability,
                "market_yes_mid": float(raw) if raw is not None else None,
                "market_yes_bid": row.get("market_yes_bid"),
                "market_yes_ask": row.get("market_yes_ask"),
                "market_no_bid": row.get("market_no_bid"),
                "market_no_ask": row.get("market_no_ask"),
                "market_yes_bid_size": row.get("market_yes_bid_size"),
                "market_yes_ask_size": row.get("market_yes_ask_size"),
                "market_no_bid_size": row.get("market_no_bid_size"),
                "market_no_ask_size": row.get("market_no_ask_size"),
            })
            fold_count += 1
        folds.append({
            "test_start": window[0].isoformat(),
            "test_end": window[-1].isoformat(),
            "training_rows": len(training),
            "prediction_rows": fold_count,
            "stump_count": len(stumps),
        })
    if not folds:
        raise ValueError("no valid folds: add more forecast cutoffs and matured training outcomes")
    return {
        "folds": folds,
        "prediction_rows": predictions,
        "model_features": list(names),
        "model_family": "gradient_boosted_decision_stumps",
        "paper_only": True,
        "live_trading": False,
        "execution_simulated": False,
        "skipped_single_class_folds": skipped_single_class_folds,
    }
