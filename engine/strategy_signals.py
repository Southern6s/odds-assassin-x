"""Transparent candidate signals for paper-only strategy research.

These are feature generators, not standalone probability forecasts or trade
instructions. They use only supplied historical observations and must be
evaluated in chronological walk-forward tests before receiving ensemble weight.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import mean, pstdev
from typing import Sequence


@dataclass(frozen=True)
class Signal:
    name: str
    score: float
    observations: int
    eligible: bool
    reason: str


def _prices(values: Sequence[float], *, minimum: int = 2) -> list[float]:
    if len(values) < minimum:
        raise ValueError(f"at least {minimum} price observations are required")
    result = []
    for value in values:
        number = float(value)
        if not math.isfinite(number) or not 0.0 < number < 1.0:
            raise ValueError("prices must be finite probabilities strictly between 0 and 1")
        result.append(number)
    return result


def momentum_signal(prices: Sequence[float], lookback: int = 5) -> Signal:
    """Return a volatility-scaled trailing logit momentum score.

    Positive values indicate upward momentum; negative values indicate
    downward momentum. No future prices or fitted parameters are used.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback must be an integer >= 2")
    values = _prices(prices, minimum=lookback + 1)
    window = values[-(lookback + 1):]

    def logit(p: float) -> float:
        return math.log(p / (1.0 - p))

    changes = [logit(b) - logit(a) for a, b in zip(window, window[1:])]
    scale = pstdev(changes)
    cumulative = sum(changes)
    score = cumulative / (scale * math.sqrt(lookback)) if scale > 1e-12 else 0.0
    return Signal("momentum", max(-5.0, min(5.0, score)), len(window), True,
                  "trailing_logit_return_scaled_by_realized_volatility")


def mean_reversion_signal(prices: Sequence[float], lookback: int = 20) -> Signal:
    """Return a contrarian z-score against the prior trailing price window.

    A positive score means the current price is below its historical mean and
    therefore points toward YES recovery; negative means the reverse.
    The current observation is excluded from the reference mean/volatility.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 3:
        raise ValueError("lookback must be an integer >= 3")
    values = _prices(prices, minimum=lookback + 1)
    history = values[-(lookback + 1):-1]
    current = values[-1]
    center = mean(history)
    scale = pstdev(history)
    if scale <= 1e-12:
        return Signal("mean_reversion", 0.0, len(history), False,
                      "historical_window_has_zero_variance")
    score = (center - current) / scale
    return Signal("mean_reversion", max(-5.0, min(5.0, score)), len(history), True,
                  "contrarian_zscore_vs_prior_window")


def order_book_imbalance(bid_size: float, ask_size: float) -> Signal:
    """Return normalized top-of-book size imbalance in [-1, 1]."""
    bid = float(bid_size)
    ask = float(ask_size)
    if not math.isfinite(bid) or not math.isfinite(ask) or bid < 0 or ask < 0:
        raise ValueError("bid_size and ask_size must be finite and nonnegative")
    total = bid + ask
    if total <= 0:
        return Signal("order_book_imbalance", 0.0, 0, False,
                      "no_displayed_size")
    return Signal("order_book_imbalance", (bid - ask) / total, 1, True,
                  "normalized_top_of_book_depth_imbalance")


def combine_candidate_signals(signals: Sequence[Signal],
                              weights: dict[str, float] | None = None) -> Signal:
    """Combine eligible signals as a research score, never as a probability."""
    eligible = [item for item in signals if item.eligible]
    if not eligible:
        return Signal("candidate_ensemble", 0.0, 0, False, "no_eligible_signals")
    supplied = weights or {}
    raw = []
    for item in eligible:
        weight = float(supplied.get(item.name, 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("signal weights must be finite and nonnegative")
        raw.append((item, weight))
    total = sum(weight for _, weight in raw)
    if total <= 0:
        raise ValueError("at least one eligible signal must have positive weight")
    score = sum(item.score * weight for item, weight in raw) / total
    return Signal("candidate_ensemble", max(-5.0, min(5.0, score)),
                  sum(item.observations for item, _ in raw), True,
                  "weighted_candidate_score_requires_walk_forward_validation")
