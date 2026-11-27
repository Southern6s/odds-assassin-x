"""Chronological beta-binomial shrinkage calibration of market midpoints."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping


def _time(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _bucket(probability: float, width: float) -> int:
    return min(int(math.floor(probability / width)), int(math.ceil(1 / width)) - 1)


def evaluate_bayesian_walk_forward(
    rows: Iterable[Mapping[str, Any]],
    *,
    min_train_size: int = 100,
    test_cutoffs: int = 1,
    step_cutoffs: int | None = None,
    bucket_width: float = 0.05,
    prior_strength: float = 20.0,
) -> dict[str, Any]:
    """Estimate empirical YES rates by midpoint bucket with a Beta prior.

    The prior mean is the midpoint of each price bucket. Only outcomes settled
    by a fold's first forecast timestamp contribute. This is calibration of
    the market prior, not an independent event-information model.
    """
    if not math.isfinite(bucket_width) or not 0 < bucket_width <= 0.25:
        raise ValueError("bucket_width must be in (0, 0.25]")
    if not math.isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError("prior_strength must be positive and finite")
    if min_train_size < 2 or test_cutoffs < 1:
        raise ValueError("invalid fold sizing")
    step = test_cutoffs if step_cutoffs is None else step_cutoffs
    if step < 1:
        raise ValueError("step_cutoffs must be positive")

    items = [dict(row) for row in rows]
    cutoffs = sorted({_time(row["forecast_at"]) for row in items})
    predictions = []
    folds = []
    for start in range(0, len(cutoffs), step):
        window = cutoffs[start:start + test_cutoffs]
        if len(window) < test_cutoffs:
            break
        cutoff = window[0]
        training = [row for row in items if _time(row["resolved_at"]) <= cutoff
                    and row.get("market_yes_mid") is not None]
        if len(training) < min_train_size:
            continue
        counts: dict[int, list[int]] = {}
        for row in training:
            p = float(row["market_yes_mid"])
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError("market_yes_mid must be between 0 and 1")
            bucket = _bucket(p, bucket_width)
            counts.setdefault(bucket, [0, 0])[int(row["probability_label_yes"])] += 1

        testing = [row for row in items if _time(row["forecast_at"]) in window]
        fold_predictions = []
        for row in testing:
            raw = row.get("market_yes_mid")
            if raw is None:
                probability = 0.5
            else:
                p = float(raw)
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("market_yes_mid must be between 0 and 1")
                bucket = _bucket(p, bucket_width)
                no, yes = counts.get(bucket, [0, 0])
                prior_mean = min(max((bucket + 0.5) * bucket_width, 0.001), 0.999)
                probability = (prior_strength * prior_mean + yes) / (prior_strength + yes + no)
            prediction = {
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
            }
            predictions.append(prediction)
            fold_predictions.append(prediction)
        folds.append({"test_start": window[0].isoformat(),
                      "test_end": window[-1].isoformat(),
                      "training_rows": len(training),
                      "prediction_rows": len(fold_predictions)})
    return {"folds": folds, "prediction_rows": predictions,
            "model_features": ["market_yes_midpoint_bucket"],
            "model_family": "beta_binomial_shrinkage"}
