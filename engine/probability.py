"""Independent forecast aggregation and evaluation for binary markets.

Forecasts must come from external/model inputs; market quotes are deliberately
not accepted here. Weights are explicit configuration, not learned performance
claims. This module never places orders.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class Forecast:
    source: str
    probability: float
    weight: float = 1.0


@dataclass(frozen=True)
class EnsembleResult:
    probability: float
    source_count: int
    total_weight: float
    disagreement: float


def _probability(value: float, field: str = "probability") -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field} must be a finite number between 0 and 1")
    return value


def combine_forecasts(forecasts: Iterable[Forecast], min_sources: int = 2) -> EnsembleResult:
    """Return a weighted mean and weighted disagreement of independent forecasts."""
    items = list(forecasts)
    if min_sources < 1:
        raise ValueError("min_sources must be at least 1")
    if len(items) < min_sources:
        raise ValueError(f"at least {min_sources} forecasts are required")

    seen: set[str] = set()
    clean: list[Forecast] = []
    for item in items:
        source = item.source.strip()
        if not source:
            raise ValueError("forecast source must not be empty")
        if source in seen:
            raise ValueError("forecast sources must be unique")
        seen.add(source)
        probability = _probability(item.probability)
        weight = float(item.weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("forecast weight must be a positive finite number")
        clean.append(Forecast(source, probability, weight))

    total_weight = sum(item.weight for item in clean)
    mean = sum(item.probability * item.weight for item in clean) / total_weight
    variance = sum(
        item.weight * (item.probability - mean) ** 2 for item in clean
    ) / total_weight
    return EnsembleResult(
        probability=mean,
        source_count=len(clean),
        total_weight=total_weight,
        disagreement=math.sqrt(variance),
    )


def brier_score(probability: float, outcome: int) -> float:
    """Squared probability error; lower is better."""
    p = _probability(probability)
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    return (p - outcome) ** 2


def log_loss(probability: float, outcome: int, epsilon: float = 1e-15) -> float:
    """Binary logarithmic loss with numerical clipping at the endpoints."""
    p = _probability(probability)
    if outcome not in (0, 1):
        raise ValueError("outcome must be 0 or 1")
    if not 0 < epsilon < 0.5:
        raise ValueError("epsilon must be between 0 and 0.5")
    p = min(1.0 - epsilon, max(epsilon, p))
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))
