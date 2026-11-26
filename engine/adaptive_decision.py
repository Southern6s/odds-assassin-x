"""Adaptive, paper-only decision support for binary prediction markets.

This module does not place orders, size positions, or promise an edge. It
combines independently generated forecasts using recent out-of-sample skill,
optionally applies a past-only calibration map, detects adverse market
conditions, and abstains unless estimated edge clears dynamic cost/risk gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _probability(value: float, name: str = "probability") -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class ModelForecast:
    """A model probability plus its recent *out-of-sample* Brier history."""
    name: str
    probability: float
    recent_brier: float
    scored_observations: int


@dataclass(frozen=True)
class CalibrationBin:
    """Past-only reliability bin; count controls shrinkage toward identity."""
    lower: float
    upper: float
    observed_frequency: float
    count: int


@dataclass(frozen=True)
class MarketState:
    yes_bid: float
    yes_ask: float
    spread: float
    liquidity: float
    volatility: float
    quote_age_seconds: float
    fees_probability: float = 0.0
    slippage_probability: float = 0.0


@dataclass(frozen=True)
class AdaptiveDecision:
    action: str
    probability_yes: float
    probability_no: float
    yes_net_edge: float
    no_net_edge: float
    required_edge: float
    ensemble_disagreement: float
    regime: str
    model_weights: Mapping[str, float]
    reasons: tuple[str, ...]
    paper_only: bool = True
    live_trading: bool = False


def calibrate_probability(
    probability: float,
    bins: Sequence[CalibrationBin],
    *,
    prior_strength: float = 20.0,
) -> float:
    """Smooth a probability using a reliability bin built only from past data.

    Empty/missing bins leave the forecast unchanged. Shrinkage prevents tiny
    samples from causing extreme probability corrections.
    """
    p = _probability(probability)
    prior = _finite(prior_strength, "prior_strength")
    if prior <= 0:
        raise ValueError("prior_strength must be positive")
    matches = [b for b in bins if b.lower <= p <= b.upper]
    if not matches:
        return p
    if len(matches) > 1:
        # A shared boundary is harmless; choose the narrowest matching bin.
        matches.sort(key=lambda b: b.upper - b.lower)
    bucket = matches[0]
    lower = _probability(bucket.lower, "bin.lower")
    upper = _probability(bucket.upper, "bin.upper")
    observed = _probability(bucket.observed_frequency, "observed_frequency")
    if lower >= upper or not isinstance(bucket.count, int) or bucket.count < 0:
        raise ValueError("calibration bin bounds/count are invalid")
    weight = bucket.count / (bucket.count + prior)
    return min(1.0, max(0.0, (1 - weight) * p + weight * observed))


def _ensemble(
    forecasts: Iterable[ModelForecast],
    *,
    skill_temperature: float,
    min_model_observations: int,
) -> tuple[float, float, dict[str, float]]:
    items = list(forecasts)
    if not items:
        raise ValueError("at least one model forecast is required")
    names: set[str] = set()
    clean = []
    for item in items:
        name = item.name.strip()
        if not name or name in names:
            raise ValueError("model names must be non-empty and unique")
        names.add(name)
        p = _probability(item.probability, f"{name}.probability")
        score = _finite(item.recent_brier, f"{name}.recent_brier")
        if not 0 <= score <= 1:
            raise ValueError("recent_brier must be between 0 and 1")
        if not isinstance(item.scored_observations, int) or item.scored_observations < 0:
            raise ValueError("scored_observations must be a nonnegative integer")
        clean.append((name, p, score, item.scored_observations))

    temperature = _finite(skill_temperature, "skill_temperature")
    if temperature <= 0:
        raise ValueError("skill_temperature must be positive")
    # Reliability weighting is deliberately conservative. Until enough
    # out-of-sample observations exist, a model receives no skill advantage.
    raw = []
    for _, _, score, count in clean:
        if count < min_model_observations:
            raw.append(1.0)
        else:
            raw.append(math.exp(-max(0.0, score - 0.15) / temperature))
    total = sum(raw)
    weights = [w / total for w in raw]
    mean = sum(w * item[1] for w, item in zip(weights, clean))
    variance = sum(w * (item[1] - mean) ** 2 for w, item in zip(weights, clean))
    return mean, math.sqrt(variance), {
        item[0]: weight for item, weight in zip(clean, weights)
    }


def adaptive_decision(
    forecasts: Iterable[ModelForecast],
    market: MarketState,
    *,
    calibration_bins: Sequence[CalibrationBin] = (),
    base_edge: float = 0.03,
    volatility_sensitivity: float = 0.10,
    disagreement_sensitivity: float = 0.50,
    max_spread: float = 0.15,
    min_liquidity: float = 1.0,
    max_quote_age_seconds: float = 120.0,
    min_model_observations: int = 50,
    skill_temperature: float = 0.10,
    calibration_prior_strength: float = 20.0,
) -> AdaptiveDecision:
    """Return CONSIDER_YES, CONSIDER_NO, or WATCH; never submit an order.

    Market inputs are decimal probabilities/normalized diagnostics. Liquidity
    is a caller-defined nonnegative depth score; volatility is a nonnegative
    recent return-volatility measure. Fees and slippage are probability units.
    """
    bid = _probability(market.yes_bid, "yes_bid")
    ask = _probability(market.yes_ask, "yes_ask")
    spread = _probability(market.spread, "spread")
    liquidity = _finite(market.liquidity, "liquidity")
    volatility = _finite(market.volatility, "volatility")
    age = _finite(market.quote_age_seconds, "quote_age_seconds")
    fees = _probability(market.fees_probability, "fees_probability")
    slippage = _probability(market.slippage_probability, "slippage_probability")
    if bid > ask:
        raise ValueError("yes_bid cannot exceed yes_ask")
    if abs((ask - bid) - spread) > 0.02:
        raise ValueError("spread is inconsistent with yes_bid and yes_ask")
    if liquidity < 0 or volatility < 0 or age < 0:
        raise ValueError("liquidity, volatility, and quote age must be nonnegative")
    if not isinstance(min_model_observations, int) or min_model_observations < 1:
        raise ValueError("min_model_observations must be positive")

    base = _probability(base_edge, "base_edge")
    vol_sensitivity = _finite(volatility_sensitivity, "volatility_sensitivity")
    disagreement_sensitivity = _finite(disagreement_sensitivity, "disagreement_sensitivity")
    spread_limit = _probability(max_spread, "max_spread")
    depth_floor = _finite(min_liquidity, "min_liquidity")
    age_limit = _finite(max_quote_age_seconds, "max_quote_age_seconds")
    if min(vol_sensitivity, disagreement_sensitivity, depth_floor, age_limit) < 0:
        raise ValueError("sensitivities and market limits must be nonnegative")

    raw_probability, disagreement, weights = _ensemble(
        forecasts, skill_temperature=skill_temperature,
        min_model_observations=min_model_observations,
    )
    probability = calibrate_probability(
        raw_probability, calibration_bins,
        prior_strength=calibration_prior_strength,
    )
    probability_no = 1.0 - probability

    regime_flags = []
    if volatility >= 0.05:
        regime_flags.append("high_volatility")
    if spread >= 0.08:
        regime_flags.append("wide_spread")
    if liquidity < max(depth_floor * 2, 1.0):
        regime_flags.append("thin_liquidity")
    regime = "+".join(regime_flags) if regime_flags else "normal"

    # Buy YES at the ask; buy NO at 1 - YES bid. Costs are deducted from edge.
    yes_net = probability - ask - fees - slippage
    no_net = probability_no - (1.0 - bid) - fees - slippage
    required = (
        base
        + vol_sensitivity * min(volatility, 1.0)
        + disagreement_sensitivity * disagreement
        + 0.25 * spread
    )
    reasons = []
    if age > age_limit:
        reasons.append("stale_quote")
    if spread > spread_limit:
        reasons.append("spread_too_wide")
    if liquidity < depth_floor:
        reasons.append("insufficient_liquidity")
    if disagreement > 0.20:
        reasons.append("models_disagree")
    if not reasons:
        if yes_net >= required and yes_net > no_net:
            action = "CONSIDER_YES"
            reasons.append("estimated_yes_edge_clears_adaptive_threshold")
        elif no_net >= required and no_net > yes_net:
            action = "CONSIDER_NO"
            reasons.append("estimated_no_edge_clears_adaptive_threshold")
        else:
            action = "WATCH"
            reasons.append("no_side_clears_cost_and_risk_adjusted_threshold")
    else:
        action = "WATCH"

    return AdaptiveDecision(
        action=action,
        probability_yes=probability,
        probability_no=probability_no,
        yes_net_edge=yes_net,
        no_net_edge=no_net,
        required_edge=required,
        ensemble_disagreement=disagreement,
        regime=regime,
        model_weights=weights,
        reasons=tuple(reasons),
    )
