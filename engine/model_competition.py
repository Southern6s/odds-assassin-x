"""Common-fold comparison of market-calibrated, event-only, and combined models.

All candidates use the same chronological walk-forward implementation and
settled-label cutoff. This is a research leaderboard, not a trade selector.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

from .probability import brier_score, log_loss
from .walk_forward import evaluate_walk_forward


def _metrics(probabilities: list[float], labels: list[int]) -> dict[str, Any]:
    if not labels:
        return {"observations": 0, "brier": None, "logloss": None, "accuracy": None}
    count = len(labels)
    return {
        "observations": count,
        "brier": sum(brier_score(p, y) for p, y in zip(probabilities, labels)) / count,
        "logloss": sum(log_loss(p, y) for p, y in zip(probabilities, labels)) / count,
        "accuracy": sum((p >= 0.5) == bool(y) for p, y in zip(probabilities, labels)) / count,
    }


def evaluate_model_competition(
    rows: Iterable[Mapping[str, Any]],
    event_feature_names: Iterable[str],
    *,
    min_train_size: int = 100,
    test_cutoffs: int = 1,
    step_cutoffs: int | None = None,
) -> dict[str, Any]:
    """Compare three logistic variants on identical forecast cutoffs.

    Candidates:
      * market_calibrated: logistic calibration of midpoint + quote presence
      * event_only: regularized logistic using only supplied event features
      * combined: event features plus midpoint + quote-presence features

    Reports both full-sample and quote-matched metrics. Market midpoint itself
    is scored as the unmodified benchmark. Repeated market/cutoff rows are
    observations, not independent bets; no execution or P&L is inferred.
    """
    items = list(rows)
    names = tuple(event_feature_names)
    if not items:
        raise ValueError("at least one research row is required")
    if not names or len(set(names)) != len(names):
        raise ValueError("event_feature_names must be non-empty and unique")

    common = {
        "min_train_size": min_train_size,
        "test_cutoffs": test_cutoffs,
        "step_cutoffs": step_cutoffs,
    }
    reports = {
        "market_calibrated": evaluate_walk_forward(
            items, (), include_market_features=True, **common
        ),
        "event_only": evaluate_walk_forward(
            items, names, include_market_features=False, **common
        ),
        "combined": evaluate_walk_forward(
            items, names, include_market_features=True, **common
        ),
    }

    prediction_maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for model_name, report in reports.items():
        mapping: dict[tuple[str, str], dict[str, Any]] = {}
        for prediction in report["prediction_rows"]:
            key = (str(prediction["market_ticker"]), str(prediction["forecast_at"]))
            if key in mapping:
                raise ValueError(f"duplicate prediction key for {model_name}: {key}")
            mapping[key] = prediction
        prediction_maps[model_name] = mapping

    common_keys = set.intersection(*(set(value) for value in prediction_maps.values()))
    if not common_keys:
        raise ValueError("candidate models produced no common out-of-sample predictions")
    ordered_keys = sorted(common_keys, key=lambda key: (key[1], key[0]))
    labels = [int(prediction_maps["combined"][key]["outcome_yes"]) for key in ordered_keys]
    for model_name, mapping in prediction_maps.items():
        if [int(mapping[key]["outcome_yes"]) for key in ordered_keys] != labels:
            raise ValueError(f"candidate outcome labels differ for {model_name}")

    quote_keys = [
        key for key in ordered_keys
        if prediction_maps["combined"][key].get("market_yes_mid") is not None
    ]
    quote_labels = [int(prediction_maps["combined"][key]["outcome_yes"]) for key in quote_keys]
    market_probabilities = [
        float(prediction_maps["combined"][key]["market_yes_mid"]) for key in quote_keys
    ]

    leaderboard = []
    for model_name, mapping in prediction_maps.items():
        full_probabilities = [float(mapping[key]["model_probability"]) for key in ordered_keys]
        matched_probabilities = [float(mapping[key]["model_probability"]) for key in quote_keys]
        leaderboard.append({
            "model": model_name,
            "full_sample": _metrics(full_probabilities, labels),
            "market_quote_matched": _metrics(matched_probabilities, quote_labels),
            "fold_count": len(reports[model_name]["folds"]),
            "model_features": reports[model_name]["model_features"],
        })
    leaderboard.sort(
        key=lambda item: (
            item["market_quote_matched"]["brier"]
            if item["market_quote_matched"]["brier"] is not None else float("inf")
        )
    )

    return {
        "leaderboard": leaderboard,
        "market_midpoint_baseline": _metrics(market_probabilities, quote_labels),
        "common_out_of_sample_observations": len(ordered_keys),
        "market_quote_matched_observations": len(quote_keys),
        "candidate_models": list(reports),
        "promotion_status": "not_promoted_research_only",
        "promotion_rule": (
            "No model is promoted from Brier/logloss alone. It must also beat "
            "the market baseline on untouched chronological data and pass "
            "fee-, spread-, slippage-, depth-, and early-exit-aware paper execution."
        ),
        "paper_only": True,
        "live_trading": False,
        "execution_simulated": False,
        "note": (
            "Repeated market/cutoff observations are not independent. "
            "This leaderboard compares probability forecasts only; it does "
            "not claim tradable edge or net profitability."
        ),
    }
