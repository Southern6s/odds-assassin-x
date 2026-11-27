"""Paper-only position exits for binary prediction-market contracts.

The quote must be the executable bid for the held outcome token (YES or NO).
This module never connects to an exchange or submits orders. It models only the
visible bid size supplied by the caller; callers must not treat a quote as a
guaranteed fill.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _finite(value: float, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _probability(value: float, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return result


@dataclass(frozen=True)
class OpenPosition:
    """An existing paper position in YES or NO contracts."""

    market_id: str
    side: str
    contracts: int
    entry_price: float
    entry_fee_per_contract: float = 0.0


@dataclass(frozen=True)
class ExitQuote:
    """Current executable bid and visible size for the held outcome token.

    slippage_per_contract is an explicit adverse-price allowance in dollars.
    fee_per_contract is the estimated exit fee in dollars.
    """

    bid: float
    bid_size: int
    fee_per_contract: float = 0.0
    slippage_per_contract: float = 0.0


@dataclass(frozen=True)
class ExitPolicy:
    """Configurable net-profit and defensive exit thresholds."""

    min_net_profit_per_contract: float = 0.02
    min_profit_roi: float = 0.05
    min_hold_edge: float = 0.01
    max_loss_per_contract: float | None = None


@dataclass(frozen=True)
class ExitDecision:
    action: str
    reason: str
    contracts_to_close: int
    remaining_contracts: int
    executable_exit_price: float | None
    net_pnl_per_contract: float | None
    estimated_net_pnl: float
    paper_only: bool = True
    live_trading: bool = False


def evaluate_exit(
    position: OpenPosition,
    quote: ExitQuote,
    *,
    policy: ExitPolicy = ExitPolicy(),
    model_fair_probability: float | None = None,
) -> ExitDecision:
    """Evaluate whether to close some/all of a position before settlement.

    Profit is measured after both entry and exit fees and the supplied
    slippage allowance. A close is sized only to the visible bid depth.
    model_fair_probability is the current fair probability of the held side,
    not necessarily the YES probability.
    """
    if not isinstance(position.market_id, str) or not position.market_id.strip():
        raise ValueError("market_id must be a non-empty string")
    if position.side not in ("yes", "no"):
        raise ValueError("side must be 'yes' or 'no'")
    if not isinstance(position.contracts, int) or isinstance(position.contracts, bool) or position.contracts < 1:
        raise ValueError("contracts must be a positive integer")
    entry = _probability(position.entry_price, "entry_price")
    if entry <= 0:
        raise ValueError("entry_price must be greater than 0")
    entry_fee = _finite(position.entry_fee_per_contract, "entry_fee_per_contract")
    bid = _probability(quote.bid, "bid")
    bid_size = quote.bid_size
    exit_fee = _finite(quote.fee_per_contract, "fee_per_contract")
    slippage = _finite(quote.slippage_per_contract, "slippage_per_contract")
    if not isinstance(bid_size, int) or isinstance(bid_size, bool) or bid_size < 0:
        raise ValueError("bid_size must be a nonnegative integer")
    if min(entry_fee, exit_fee, slippage) < 0:
        raise ValueError("fees and slippage must be nonnegative")
    if entry_fee + exit_fee + slippage >= 1:
        raise ValueError("combined costs are outside the supported range")

    target = _finite(policy.min_net_profit_per_contract, "min_net_profit_per_contract")
    target_roi = _finite(policy.min_profit_roi, "min_profit_roi")
    hold_edge = _finite(policy.min_hold_edge, "min_hold_edge")
    if target < 0 or target_roi < 0 or hold_edge < 0:
        raise ValueError("exit thresholds must be nonnegative")
    max_loss = policy.max_loss_per_contract
    if max_loss is not None:
        max_loss = _finite(max_loss, "max_loss_per_contract")
        if max_loss <= 0:
            raise ValueError("max_loss_per_contract must be positive when enabled")

    fair = None
    if model_fair_probability is not None:
        fair = _probability(model_fair_probability, "model_fair_probability")

    if bid_size == 0 or bid <= 0:
        return ExitDecision(
            "HOLD", "no_executable_bid_liquidity", 0, position.contracts,
            None, None, 0.0,
        )

    executable_price = max(0.0, bid - slippage)
    net_per_contract = (
        executable_price - exit_fee - entry - entry_fee
    )
    entry_cost = entry + entry_fee
    profit_target = max(target, entry_cost * target_roi)

    reason = None
    if max_loss is not None and net_per_contract <= -max_loss:
        reason = "risk_stop"
    elif fair is not None and fair <= bid - hold_edge:
        reason = "model_edge_reversed"
    elif net_per_contract >= profit_target:
        reason = "take_profit"

    if reason is None:
        return ExitDecision(
            "HOLD", "profit_target_not_reached_or_no_exit_signal", 0,
            position.contracts, executable_price, net_per_contract, 0.0,
        )

    close_count = min(position.contracts, bid_size)
    remaining = position.contracts - close_count
    return ExitDecision(
        action="PARTIAL_EXIT" if remaining else "EXIT",
        reason=reason,
        contracts_to_close=close_count,
        remaining_contracts=remaining,
        executable_exit_price=executable_price,
        net_pnl_per_contract=net_per_contract,
        estimated_net_pnl=net_per_contract * close_count,
    )
