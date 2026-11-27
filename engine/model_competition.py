"""Common-fold comparison of market-calibrated, event-only, and combined models.

All candidates use the same chronological walk-forward implementation and
settled-label cutoff. This is a research leaderboard, not a trade selector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    if not names or len(set(names)) != len(names):
        raise ValueError("event_feature_names must be non-empty and unique")
    if not items:
        raise ValueError("at least one research row is required")

    common = {
        "min_train_size": min_train_size,
        "test_cutoffs": test_cutoffs,
        "step_cutoffs": step_cutoffs,
    }
    reports = {
        "market_calibrated": evaluate_walk_forward(
            items, (), include_market_features=True,
            allow_empty_event_features=True, **common
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
    market_baseline_metrics = _metrics(market_probabilities, quote_labels)
    for item in leaderboard:
        candidate_brier = item["market_quote_matched"]["brier"]
        baseline_brier = market_baseline_metrics["brier"]
        item["brier_skill_vs_market"] = (
            1.0 - candidate_brier / baseline_brier
            if candidate_brier is not None and baseline_brier not in (None, 0.0) else None
        )
        item["beats_market_midpoint_brier"] = (
            candidate_brier < baseline_brier
            if candidate_brier is not None and baseline_brier is not None else False
        )
    leaderboard.sort(
        key=lambda item: (
            item["market_quote_matched"]["brier"]
            if item["market_quote_matched"]["brier"] is not None else float("inf")
        )
    )

    # Category slices are emitted only from explicit exchange metadata and
    # only when there are enough quote-matched observations to be informative.
    categories: dict[str, list[tuple[str, str]]] = {}
    missing_category = 0
    for key in quote_keys:
        category = prediction_maps["combined"][key].get("category")
        if not isinstance(category, str) or not category.strip():
            missing_category += 1
            continue
        categories.setdefault(category.strip(), []).append(key)
    category_leaderboard = []
    for category, category_keys in sorted(categories.items()):
        if len(category_keys) < 20:
            continue
        category_labels = [int(prediction_maps["combined"][key]["outcome_yes"]) for key in category_keys]
        category_models = {}
        for model_name, mapping in prediction_maps.items():
            category_models[model_name] = _metrics(
                [float(mapping[key]["model_probability"]) for key in category_keys],
                category_labels,
            )
        category_market = _metrics(
            [float(prediction_maps["combined"][key]["market_yes_mid"]) for key in category_keys],
            category_labels,
        )
        for metrics in category_models.values():
            metrics["brier_skill_vs_market"] = (
                1.0 - metrics["brier"] / category_market["brier"]
                if category_market["brier"] not in (None, 0.0) else None
            )
            metrics["beats_market_midpoint_brier"] = (
                metrics["brier"] < category_market["brier"]
                if category_market["brier"] is not None else False
            )
        category_leaderboard.append({
            "category": category,
            "observations": len(category_keys),
            "models": category_models,
            "market_midpoint_baseline": category_market,
            "best_candidate": min(
                category_models,
                key=lambda name: category_models[name]["brier"],
            ),
        })

    return {
        "leaderboard": leaderboard,
        "market_midpoint_baseline": market_baseline_metrics,
        "common_out_of_sample_observations": len(ordered_keys),
        "market_quote_matched_observations": len(quote_keys),
        "category_leaderboard": category_leaderboard,
        "category_observations_without_metadata": missing_category,
        "minimum_category_observations": 20,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare paper-only Kalshi model candidates on chronological folds"
    )
    parser.add_argument("--dataset", required=True, help="Point-in-time research panel JSONL")
    parser.add_argument("--features", required=True,
                        help="Comma-separated numeric event/candle feature allowlist")
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
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"line {line_number} must be a JSON object")
                rows.append(record)
        features = [item.strip() for item in args.features.split(",") if item.strip()]
        report = evaluate_model_competition(
            rows, features, min_train_size=args.min_train_size,
            test_cutoffs=args.test_cutoffs, step_cutoffs=args.step_cutoffs,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Model competition failed: {exc}\\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
