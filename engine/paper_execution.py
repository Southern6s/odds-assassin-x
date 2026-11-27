"""Chronological, depth-aware paper execution for walk-forward predictions.

Only timestamped decision-time quotes and explicit displayed size are used.
No settlement labels are consulted for trade decisions. Displayed depth is an
optimistic upper bound, not a guarantee of fills or queue priority.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping

from .position_management import ExitPolicy, ExitQuote, OpenPosition, evaluate_exit


@dataclass(frozen=True)
class PaperExecutionConfig:
    min_entry_edge: float = 0.05
    fee_per_contract: float = 0.01
    slippage_per_contract: float = 0.01
    max_trade_risk: float = 10.0
    max_portfolio_risk: float = 25.0
    max_contracts: int = 20
    daily_loss_limit: float = 5.0
    max_quote_age_seconds: float = 300.0
    exit_policy: ExitPolicy = ExitPolicy(min_net_profit_per_contract=0.02,
                                         min_profit_roi=0.05,
                                         min_hold_edge=0.01,
                                         max_loss_per_contract=0.10)


def _number(value: Any, field: str, *, minimum: float | None = None,
            maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def _time(value: Any, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _size(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None:
        return 0
    number = _number(value, key, minimum=0)
    return math.floor(number)


def _quote(row: Mapping[str, Any], side: str, book_side: str) -> tuple[float | None, int]:
    price = row.get(f"market_{side}_{book_side}")
    if price is None:
        return None, 0
    return _number(price, f"market_{side}_{book_side}", minimum=0, maximum=1), _size(
        row, f"market_{side}_{book_side}_size")


def simulate_paper_execution(
    prediction_rows: Iterable[Mapping[str, Any]],
    *,
    config: PaperExecutionConfig = PaperExecutionConfig(),
) -> dict[str, Any]:
    """Replay predictions chronologically with hard risk caps and depth limits.

    Rows must contain market_ticker, forecast_at, model_probability,
    market_yes_bid/ask, and quote_age_seconds. Optional NO quotes are supported.
    Entry and exit fills require explicit ask/bid sizes; absent size means no fill.
    Open positions are marked to the latest executable bid, net of estimated
    exit fees/slippage, but are not counted as realized profit.
    """
    if not 0 <= config.min_entry_edge <= 1 or not math.isfinite(config.min_entry_edge):
        raise ValueError("min_entry_edge must be between 0 and 1")
    for name in ("fee_per_contract", "slippage_per_contract", "max_trade_risk",
                 "max_portfolio_risk", "daily_loss_limit", "max_quote_age_seconds"):
        _number(getattr(config, name), name, minimum=0)
    if config.max_trade_risk <= 0 or config.max_portfolio_risk <= 0:
        raise ValueError("trade and portfolio risk caps must be positive")
    if isinstance(config.max_contracts, bool) or not isinstance(config.max_contracts, int) or config.max_contracts < 1:
        raise ValueError("max_contracts must be a positive integer")

    rows = list(prediction_rows)
    if not rows:
        raise ValueError("at least one prediction row is required")
    normalized = []
    for row in rows:
        ticker = row.get("market_ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("each row requires market_ticker")
        timestamp = _time(row.get("forecast_at"), "forecast_at")
        probability = _number(row.get("model_probability"), "model_probability",
                              minimum=0, maximum=1)
        age = row.get("quote_age_seconds")
        if age is not None:
            age = _number(age, "quote_age_seconds", minimum=0)
        normalized.append((timestamp, ticker, probability, age, row))
    normalized.sort(key=lambda item: (item[0], item[1]))

    positions: dict[str, OpenPosition] = {}
    latest_marks: dict[str, tuple[float, float, float]] = {}
    trades: list[dict[str, Any]] = []
    blocked: dict[str, int] = {}
    realized = 0.0
    daily_realized: dict[str, float] = {}
    peak_open_risk = 0.0

    def block(reason: str) -> None:
        blocked[reason] = blocked.get(reason, 0) + 1

    for timestamp, ticker, p_yes, quote_age, row in normalized:
        trading_day = timestamp.date().isoformat()
        if quote_age is None:
            block("missing_quote_age")
            continue
        if quote_age > config.max_quote_age_seconds:
            block("stale_quote")
            continue

        yes_bid, yes_bid_size = _quote(row, "yes", "bid")
        yes_ask, yes_ask_size = _quote(row, "yes", "ask")
        no_bid, no_bid_size = _quote(row, "no", "bid")
        no_ask, no_ask_size = _quote(row, "no", "ask")
        if ticker in positions:
            position = positions[ticker]
            bid, bid_size = ((yes_bid, yes_bid_size) if position.side == "yes"
                             else (no_bid, no_bid_size))
            fair = p_yes if position.side == "yes" else 1.0 - p_yes
            if bid is None:
                block("missing_exit_bid")
                continue
            latest_marks[ticker] = (bid, config.fee_per_contract,
                                    config.slippage_per_contract)
            decision = evaluate_exit(
                position,
                ExitQuote(bid=bid, bid_size=bid_size,
                          fee_per_contract=config.fee_per_contract,
                          slippage_per_contract=config.slippage_per_contract),
                policy=config.exit_policy,
                model_fair_probability=fair,
            )
            if decision.contracts_to_close:
                realized += decision.estimated_net_pnl
                daily_realized[trading_day] = (
                    daily_realized.get(trading_day, 0.0) + decision.estimated_net_pnl
                )
                trades.append({
                    "timestamp": timestamp.isoformat(), "market_ticker": ticker,
                    "side": position.side, "action": decision.action,
                    "reason": decision.reason,
                    "contracts": decision.contracts_to_close,
                    "price": decision.executable_exit_price,
                    "net_pnl": decision.estimated_net_pnl,
                })
                if decision.remaining_contracts:
                    positions[ticker] = replace(position, contracts=decision.remaining_contracts)
                else:
                    del positions[ticker]
                    latest_marks.pop(ticker, None)
            else:
                block(decision.reason)
            continue

        if daily_realized.get(trading_day, 0.0) <= -config.daily_loss_limit:
            block("daily_loss_limit_reached")
            continue
        candidates = []
        if yes_ask is not None and yes_ask_size:
            candidates.append(("yes", p_yes, yes_ask, yes_ask_size))
        if no_ask is not None and no_ask_size:
            candidates.append(("no", 1.0 - p_yes, no_ask, no_ask_size))
        if not candidates:
            block("missing_entry_ask_depth")
            continue
        candidates = [candidate for candidate in candidates
                      if candidate[1] - candidate[2] - config.fee_per_contract
                      - config.slippage_per_contract >= config.min_entry_edge]
        if not candidates:
            block("edge_below_minimum_after_costs")
            continue
        side, fair, ask, ask_size = max(
            candidates, key=lambda candidate: candidate[1] - candidate[2])
        entry_price = ask + config.slippage_per_contract
        if entry_price > 1.0:
            block("entry_cost_exceeds_contract_payout")
            continue
        risk_per_contract = entry_price + config.fee_per_contract
        open_risk = sum(p.contracts * (p.entry_price + p.entry_fee_per_contract)
                        for p in positions.values())
        remaining_portfolio = config.max_portfolio_risk - open_risk
        budget = min(config.max_trade_risk, remaining_portfolio)
        contracts = min(config.max_contracts, ask_size,
                        int(budget // risk_per_contract) if risk_per_contract > 0 else 0)
        if contracts < 1:
            block("risk_budget_below_one_contract")
            continue
        position = OpenPosition(
            market_id=ticker, side=side, contracts=contracts,
            entry_price=entry_price,
            entry_fee_per_contract=config.fee_per_contract,
        )
        positions[ticker] = position
        bid = yes_bid if side == "yes" else no_bid
        if bid is not None:
            latest_marks[ticker] = (bid, config.fee_per_contract,
                                    config.slippage_per_contract)
        peak_open_risk = max(peak_open_risk, open_risk + contracts * risk_per_contract)
        trades.append({
            "timestamp": timestamp.isoformat(), "market_ticker": ticker,
            "side": side, "action": "ENTRY", "reason": "edge_and_risk_checks_passed",
            "contracts": contracts, "price": entry_price, "net_pnl": 0.0,
        })

    unrealized = 0.0
    unmarked_positions = 0
    open_position_rows = []
    for ticker, position in positions.items():
        mark = latest_marks.get(ticker)
        if mark is None:
            unmarked_positions += 1
            mark_bid = None
            position_unrealized = None
        else:
            mark_bid, exit_fee, slippage = mark
            position_unrealized = (
                max(0.0, mark_bid - slippage) - exit_fee
                - position.entry_price - position.entry_fee_per_contract
            ) * position.contracts
            unrealized += position_unrealized
        open_position_rows.append({
            "market_ticker": ticker, "side": position.side,
            "contracts": position.contracts, "entry_price": position.entry_price,
            "mark_bid": mark_bid, "unrealized_pnl": position_unrealized,
        })

    return {
        "paper_only": True,
        "live_trading": False,
        "fills_are_estimates": True,
        "requires_displayed_depth": True,
        "observations": len(normalized),
        "entry_count": sum(trade["action"] == "ENTRY" for trade in trades),
        "exit_count": sum(trade["action"] in ("EXIT", "PARTIAL_EXIT") for trade in trades),
        "realized_net_pnl": realized,
        "unrealized_net_pnl_estimate": unrealized,
        "combined_net_pnl_estimate": realized + unrealized,
        "peak_open_risk": peak_open_risk,
        "open_positions": open_position_rows,
        "unmarked_open_positions": unmarked_positions,
        "blocked_signal_counts": blocked,
        "trades": trades,
    }
