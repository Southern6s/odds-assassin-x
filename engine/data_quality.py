"""Historical market-data quality checks; diagnostics never repair source data."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True)
class QualityIssue:
    code: str
    ticker: str
    detail: str


@dataclass(frozen=True)
class QualityReport:
    records_checked: int
    issues: tuple[QualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues

    def count(self, code: str) -> int:
        return sum(issue.code == code for issue in self.issues)


def _utc(value: str | datetime, field: str) -> datetime:
    parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def assess_market_snapshots(
    records: Iterable[dict[str, Any]],
    *,
    stale_after_seconds: int = 300,
    as_of: datetime | None = None,
) -> QualityReport:
    """Flag absent two-sided quotes and old capture timestamps without imputing values."""
    if stale_after_seconds < 0:
        raise ValueError("stale_after_seconds must be nonnegative")
    reference = _utc(as_of or datetime.now(timezone.utc), "as_of")
    issues: list[QualityIssue] = []
    checked = 0
    for record in records:
        checked += 1
        market = record.get("market") or {}
        ticker = str(market.get("ticker") or "<unknown>")
        if market.get("yes_bid") is None or market.get("yes_ask") is None:
            issues.append(QualityIssue("missing_yes_quote", ticker, "YES bid or ask is absent"))
        if market.get("no_bid") is None or market.get("no_ask") is None:
            issues.append(QualityIssue("missing_no_quote", ticker, "NO bid or ask is absent"))
        try:
            captured = _utc(record["captured_at"], "captured_at")
        except (KeyError, TypeError, ValueError) as exc:
            issues.append(QualityIssue("invalid_capture_time", ticker, str(exc)))
            continue
        age = (reference - captured).total_seconds()
        if age > stale_after_seconds:
            issues.append(QualityIssue("stale_snapshot", ticker, f"capture is {int(age)} seconds old"))
        elif age < 0:
            issues.append(QualityIssue("future_snapshot", ticker, "capture time is after as_of"))
    return QualityReport(checked, tuple(issues))


def assess_candlesticks(
    records: Iterable[dict[str, Any]],
    *,
    expected_interval: int,
) -> QualityReport:
    """Detect duplicate periods, gaps, and incomplete OHLC observations per ticker."""
    if expected_interval not in (1, 60, 1440):
        raise ValueError("expected_interval must be 1, 60, or 1440")
    rows = list(records)
    issues: list[QualityIssue] = []
    grouped: dict[str, list[int]] = {}
    seen: set[tuple[str, int]] = set()
    for record in rows:
        candle = record.get("candle") or {}
        ticker = str(candle.get("market_ticker") or "<unknown>")
        end_ts = candle.get("end_period_ts")
        interval = candle.get("period_interval")
        if interval != expected_interval:
            issues.append(QualityIssue("interval_mismatch", ticker, f"expected {expected_interval}, got {interval}"))
        if not isinstance(end_ts, int) or end_ts <= 0:
            issues.append(QualityIssue("invalid_period", ticker, "end_period_ts must be a positive integer"))
            continue
        identity = (ticker, end_ts)
        if identity in seen:
            issues.append(QualityIssue("duplicate_period", ticker, f"duplicate end_period_ts {end_ts}"))
        else:
            seen.add(identity)
            grouped.setdefault(ticker, []).append(end_ts)
        for field in ("yes_bid", "yes_ask", "price"):
            ohlc = candle.get(field)
            if ohlc is None or any(ohlc.get(key) is None for key in ("open", "high", "low", "close")):
                issues.append(QualityIssue("incomplete_ohlc", ticker, f"{field} OHLC is incomplete"))
    step = expected_interval * 60
    for ticker, timestamps in grouped.items():
        ordered = sorted(set(timestamps))
        for previous, current in zip(ordered, ordered[1:]):
            delta = current - previous
            if delta > step and delta % step == 0:
                missing = delta // step - 1
                issues.append(QualityIssue("missing_periods", ticker, f"{missing} missing period(s) between {previous} and {current}"))
            elif delta > step:
                issues.append(QualityIssue("irregular_gap", ticker, f"non-aligned gap of {delta} seconds"))
    return QualityReport(len(rows), tuple(issues))
