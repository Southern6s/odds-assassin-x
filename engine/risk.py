"""Conservative paper position sizing and portfolio risk gates.

All dollar amounts are USD. This module has no exchange connectivity and never
submits orders. Fee inputs must be supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RiskLimits:
    max_trade_risk: float
    max_portfolio_risk: float
    max_contracts: int
    kelly_fraction: float = 0.25
    min_edge: float = 0.03
    daily_loss_limit: float | None = None


@dataclass(frozen=True)
class SizingDecision:
    approved: bool
    contracts: int
    risk_dollars: float
    edge: float
    expected_value_per_contract: float
    full_kelly_fraction: float
    reason: str


def _finite_nonnegative(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return number


def size_position(
    *,
    bankroll: float,
    portfolio_risk: float,
    daily_pnl: float,
    model_probability: float,
    entry_price: float,
    fee_per_contract: float,
    limits: RiskLimits,
) -> SizingDecision:
    """Size a binary YES contract using fractional Kelly and hard dollar caps."""
    bankroll = _finite_nonnegative(bankroll, "bankroll")
    portfolio_risk = _finite_nonnegative(portfolio_risk, "portfolio_risk")
    fee = _finite_nonnegative(fee_per_contract, "fee_per_contract")
    daily_pnl = float(daily_pnl)
    if not math.isfinite(daily_pnl):
        raise ValueError("daily_pnl must be finite")
    p = float(model_probability)
    price = float(entry_price)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("model_probability must be between 0 and 1")
    if not math.isfinite(price) or not 0 < price < 1:
        raise ValueError("entry_price must be greater than 0 and less than 1")
    if limits.max_trade_risk <= 0 or not math.isfinite(limits.max_trade_risk):
        raise ValueError("max_trade_risk must be positive and finite")
    if limits.max_portfolio_risk <= 0 or not math.isfinite(limits.max_portfolio_risk):
        raise ValueError("max_portfolio_risk must be positive and finite")
    if not isinstance(limits.max_contracts, int) or limits.max_contracts < 1:
        raise ValueError("max_contracts must be a positive integer")
    if not 0 < limits.kelly_fraction <= 1 or not math.isfinite(limits.kelly_fraction):
        raise ValueError("kelly_fraction must be in (0, 1]")
    if not 0 <= limits.min_edge <= 1 or not math.isfinite(limits.min_edge):
        raise ValueError("min_edge must be between 0 and 1")
    if limits.daily_loss_limit is not None:
        if not math.isfinite(limits.daily_loss_limit) or limits.daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be positive and finite")

    edge = p - price
    ev = p - price - fee
    loss_per_contract = price + fee
    win_profit = 1.0 - price - fee

    def reject(reason: str, kelly: float = 0.0) -> SizingDecision:
        return SizingDecision(False, 0, 0.0, edge, ev, kelly, reason)

    if limits.daily_loss_limit is not None and daily_pnl <= -limits.daily_loss_limit:
        return reject("daily_loss_limit_reached")
    if edge < limits.min_edge:
        return reject("edge_below_minimum")
    if ev <= 0:
        return reject("nonpositive_expected_value_after_fees")
    if win_profit <= 0:
        return reject("no_positive_win_profit_after_fees")
    if bankroll <= 0:
        return reject("no_bankroll")
    if portfolio_risk >= limits.max_portfolio_risk:
        return reject("portfolio_risk_limit_reached")

    # Binary Kelly fraction of bankroll for a YES contract.
    b = win_profit / loss_per_contract
    full_kelly = max(0.0, (p * b - (1.0 - p)) / b)
    desired_risk = bankroll * full_kelly * limits.kelly_fraction
    remaining_portfolio = limits.max_portfolio_risk - portfolio_risk
    risk_budget = min(desired_risk, limits.max_trade_risk, remaining_portfolio, bankroll)
    contracts = min(limits.max_contracts, int(risk_budget // loss_per_contract))
    if contracts < 1:
        return reject("risk_budget_below_one_contract", full_kelly)

    total_risk = contracts * loss_per_contract
    return SizingDecision(
        approved=True,
        contracts=contracts,
        risk_dollars=total_risk,
        edge=edge,
        expected_value_per_contract=ev,
        full_kelly_fraction=full_kelly,
        reason="paper_risk_checks_passed",
    )
