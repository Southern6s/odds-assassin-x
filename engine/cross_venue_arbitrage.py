"""Fee-aware cross-venue arbitrage candidate detection.

This module only evaluates supplied executable quotes. It does not fetch data,
match markets automatically, submit orders, or guarantee simultaneous fills.
A pair is eligible only after a separate process verifies identical resolution
rules and event identity. Paper research only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


@dataclass(frozen=True)
class ArbitrageCandidate:
    eligible: bool
    reason: str
    yes_venue: str | None = None
    no_venue: str | None = None
    yes_ask: float | None = None
    no_ask: float | None = None
    contracts: int = 0
    gross_cost_per_pair: float | None = None
    total_cost_per_pair: float | None = None
    net_profit_per_pair: float | None = None
    net_return_on_cost: float | None = None
    paper_only: bool = True
    live_trading: bool = False


def _number(value: Any, field: str, *, low: float = 0.0,
            high: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result) or result < low or (high is not None and result > high):
        raise ValueError(f"{field} is outside its valid range")
    return result


def evaluate_cross_venue_pair(
    venue_a: Mapping[str, Any],
    venue_b: Mapping[str, Any],
    *,
    rules_verified: bool,
    fee_a_per_contract: float,
    fee_b_per_contract: float,
    slippage_a_per_contract: float = 0.0,
    slippage_b_per_contract: float = 0.0,
    min_net_profit_per_pair: float = 0.01,
    max_contracts: int = 100,
) -> ArbitrageCandidate:
    """Evaluate buying YES at one venue and equivalent NO at the other.

    Inputs require executable YES/NO asks and displayed ask sizes for both
    venues. The function chooses the cheaper valid direction, limits pair size
    to the minimum displayed depth, and deducts fees and slippage on both legs.
    It never infers that two similarly named markets resolve identically.
    """
    if not isinstance(rules_verified, bool):
        raise ValueError("rules_verified must be boolean")
    if isinstance(max_contracts, bool) or not isinstance(max_contracts, int) or max_contracts < 1:
        raise ValueError("max_contracts must be a positive integer")
    fees = (
        _number(fee_a_per_contract, "fee_a_per_contract"),
        _number(fee_b_per_contract, "fee_b_per_contract"),
    )
    slips = (
        _number(slippage_a_per_contract, "slippage_a_per_contract"),
        _number(slippage_b_per_contract, "slippage_b_per_contract"),
    )
    minimum = _number(min_net_profit_per_pair, "min_net_profit_per_pair")
    if not rules_verified:
        return ArbitrageCandidate(False, "market_resolution_rules_not_verified")
    for label, venue in (("venue_a", venue_a), ("venue_b", venue_b)):
        if not isinstance(venue.get("name"), str) or not venue["name"].strip():
            raise ValueError(f"{label}.name is required")
        for key in ("yes_ask", "no_ask"):
            _number(venue.get(key), f"{label}.{key}", high=1.0)
        for key in ("yes_ask_size", "no_ask_size"):
            _number(venue.get(key), f"{label}.{key}", high=1_000_000_000)

    # Each candidate buys YES on one venue and NO on the other.
    directions = [
        (venue_a, venue_b, 0, 1),
        (venue_b, venue_a, 1, 0),
    ]
    candidates = []
    for yes_venue, no_venue, yes_index, no_index in directions:
        yes_price = float(yes_venue["yes_ask"])
        no_price = float(no_venue["no_ask"])
        size = min(
            max_contracts,
            math.floor(float(yes_venue["yes_ask_size"])),
            math.floor(float(no_venue["no_ask_size"])),
        )
        gross = yes_price + no_price
        costs = fees[yes_index] + fees[no_index] + slips[yes_index] + slips[no_index]
        total = gross + costs
        profit = 1.0 - total
        if size > 0 and profit >= minimum:
            candidates.append((profit, size, yes_venue, no_venue,
                               yes_price, no_price, gross, total))

    if not candidates:
        return ArbitrageCandidate(False, "no_positive_fee_adjusted_edge_with_available_depth")
    profit, size, yes_venue, no_venue, yes_price, no_price, gross, total = max(
        candidates, key=lambda item: item[0]
    )
    return ArbitrageCandidate(
        eligible=True,
        reason="fee_adjusted_opposite_outcome_basket_candidate",
        yes_venue=yes_venue["name"],
        no_venue=no_venue["name"],
        yes_ask=yes_price,
        no_ask=no_price,
        contracts=size,
        gross_cost_per_pair=gross,
        total_cost_per_pair=total,
        net_profit_per_pair=profit,
        net_return_on_cost=profit / total if total > 0 else None,
    )
