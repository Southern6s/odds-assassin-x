"""Pure, testable probability and quote math for binary prediction markets.

This module does not fetch data, place orders, or infer a forecast from market prices.
A model probability must be supplied independently before edge/EV is calculated.
All prices and probabilities are decimals in [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Quote:
    bid: Optional[float]
    ask: Optional[float]
    midpoint: Optional[float]
    spread: Optional[float]


def _valid_probability(value: Optional[float], field: str) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return number


def analyze_quote(bid: Optional[float], ask: Optional[float]) -> Quote:
    """Validate a binary contract quote and calculate midpoint/spread."""
    bid = _valid_probability(bid, "bid")
    ask = _valid_probability(ask, "ask")
    if bid is not None and ask is not None and bid > ask:
        raise ValueError("bid cannot exceed ask")
    midpoint = (bid + ask) / 2 if bid is not None and ask is not None else (
        bid if ask is None else ask
    )
    spread = ask - bid if bid is not None and ask is not None else None
    return Quote(bid=bid, ask=ask, midpoint=midpoint, spread=spread)


def calculate_edge(model_probability: float, market_probability: float) -> float:
    """Return model minus market probability, in probability points."""
    model = _valid_probability(model_probability, "model_probability")
    market = _valid_probability(market_probability, "market_probability")
    return model - market


def expected_value_per_contract(
    model_probability: float, purchase_price: float
) -> float:
    """Gross expected profit per $1-payout contract, before fees.

    A winning contract pays $1; a losing contract pays $0.
    """
    model = _valid_probability(model_probability, "model_probability")
    price = _valid_probability(purchase_price, "purchase_price")
    return model - price


def analyze_market(
    yes_bid: Optional[float],
    yes_ask: Optional[float],
    model_yes_probability: Optional[float] = None,
) -> dict:
    """Return transparent quote metrics; edge stays unavailable without a forecast."""
    quote = analyze_quote(yes_bid, yes_ask)
    result = {
        "yes_bid": quote.bid,
        "yes_ask": quote.ask,
        "market_probability": quote.midpoint,
        "spread": quote.spread,
        "model_probability": None,
        "edge": None,
        "expected_value_per_contract": None,
        "edge_status": "unavailable_no_independent_forecast",
    }
    if model_yes_probability is not None:
        model = _valid_probability(model_yes_probability, "model_yes_probability")
        result.update({
            "model_probability": model,
            "edge": calculate_edge(model, quote.midpoint) if quote.midpoint is not None else None,
            "expected_value_per_contract": (
                expected_value_per_contract(model, quote.ask)
                if quote.ask is not None else None
            ),
            "edge_status": "calculated" if quote.midpoint is not None else "unavailable_no_market_quote",
        })
    return result
