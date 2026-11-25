"""Small dependency-free, regularized logistic regression for offline research.

The model consumes explicitly selected numeric event features only. Training
rows are filtered to outcomes settled by the supplied fit cutoff. It does not
place orders and is not a claim of predictive edge.
"""
from __future__ import annotations

from dataclasses import dataclass
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


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 700))
        return 1 / (1 + z)
    z = math.exp(max(value, -700))
    return z / (1 + z)


def _vector(row: Mapping[str, Any], names: tuple[str, ...]) -> list[float]:
    values = []
    for name in names:
        key = name if name.startswith("feature_") else f"feature_{name}"
        raw = row.get(key)
        if isinstance(raw, bool) or raw is None:
            raise ValueError(f"missing or non-numeric feature: {key}")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"feature must be numeric: {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"feature must be finite: {key}")
        values.append(value)
    return values


@dataclass(frozen=True)
class LogisticModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    training_rows: int
    training_yes_rate: float
    fitted_at: str
    l2: float
    iterations: int

    def predict_probability(self, row: Mapping[str, Any]) -> float:
        values = _vector(row, self.feature_names)
        score = self.intercept + sum(
            coefficient * ((value - mean) / scale)
            for coefficient, value, mean, scale in zip(
                self.coefficients, values, self.means, self.scales
            )
        )
        return _sigmoid(score)


def fit_logistic_model(
    rows: Iterable[Mapping[str, Any]],
    feature_names: Iterable[str],
    *,
    as_of: datetime,
    min_training_rows: int = 30,
    l2: float = 1.0,
    iterations: int = 1500,
    learning_rate: float = 0.05,
) -> LogisticModel:
    """Fit only on labels resolved by as_of; feature list must be explicit."""
    cutoff = _time(as_of, "as_of")
    names = tuple(feature_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("feature_names must be non-empty and unique")
    if min_training_rows < 2:
        raise ValueError("min_training_rows must be at least 2")
    if not math.isfinite(l2) or l2 < 0:
        raise ValueError("l2 must be finite and nonnegative")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")

    training = []
    for row in rows:
        forecast_at = _time(row.get("forecast_at"), "forecast_at")
        resolved_at = _time(row.get("resolved_at"), "resolved_at")
        if forecast_at >= resolved_at:
            raise ValueError("training forecast_at must precede resolved_at")
        if resolved_at > cutoff:
            continue
        label = row.get("probability_label_yes")
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("training label must be 0 or 1")
        training.append((_vector(row, names), int(label)))
    if len(training) < min_training_rows:
        raise ValueError(
            f"need at least {min_training_rows} matured training rows; got {len(training)}"
        )
    labels = [label for _, label in training]
    if len(set(labels)) < 2:
        raise ValueError("training data must contain both YES and NO outcomes")

    width = len(names)
    means = [sum(x[j] for x, _ in training) / len(training) for j in range(width)]
    variances = [
        sum((x[j] - means[j]) ** 2 for x, _ in training) / len(training)
        for j in range(width)
    ]
    scales = [math.sqrt(v) if v > 1e-12 else 1.0 for v in variances]
    matrix = [[(x[j] - means[j]) / scales[j] for j in range(width)]
              for x, _ in training]
    weights = [0.0] * width
    intercept = math.log(
        (sum(labels) + 0.5) / (len(labels) - sum(labels) + 0.5)
    )
    n = len(training)
    for _ in range(iterations):
        grad = [0.0] * width
        bias_grad = 0.0
        for vector, label in zip(matrix, labels):
            error = _sigmoid(intercept + sum(w * x for w, x in zip(weights, vector))) - label
            bias_grad += error
            for j, value in enumerate(vector):
                grad[j] += error * value
        intercept -= learning_rate * bias_grad / n
        for j in range(width):
            grad[j] = grad[j] / n + l2 * weights[j] / n
            weights[j] -= learning_rate * grad[j]

    return LogisticModel(
        feature_names=names, means=tuple(means), scales=tuple(scales),
        coefficients=tuple(weights), intercept=intercept,
        training_rows=n, training_yes_rate=sum(labels) / n,
        fitted_at=cutoff.isoformat(), l2=l2, iterations=iterations,
    )
