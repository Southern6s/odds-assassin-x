"""Build a chronological panel of point-in-time, resolved-market observations.

This is offline research only. Each cutoff is evaluated independently; quotes
and completed event-trade buckets after that cutoff are excluded by the shared
dataset builder. Outcomes are included only when settlement is later.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Iterable

from .event_research import build_event_trade_features
from .outcome_dataset import build_resolved_market_dataset
from .market_candle_features import build_market_candle_features
from .research_dataset import read_jsonl, write_jsonl_atomic, _market_payload


def _aware(value: str | datetime, field: str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def build_dataset_panel_from_records(
    market_snapshots: Iterable[dict[str, Any]],
    trade_snapshots: Iterable[dict[str, Any]],
    *,
    forecast_start: datetime,
    forecast_end: datetime,
    step_seconds: int = 3600,
    bucket_seconds: int = 3600,
    max_feature_age_seconds: int = 86400,
    max_quote_age_seconds: int = 300,
    candlestick_snapshots: Iterable[dict[str, Any]] = (),
    max_cutoffs: int = 1000,
) -> list[dict[str, Any]]:
    """Build a panel at regular UTC cutoffs, inclusive of both endpoints.

    Event features are aggregated once through forecast_end. The outcome
    dataset independently rejects incomplete/future buckets at every cutoff.
    Historical candle quote timestamps use end_period_ts, never download time.
    """
    start = _aware(forecast_start, "forecast_start")
    end = _aware(forecast_end, "forecast_end")
    if end < start:
        raise ValueError("forecast_end must not precede forecast_start")
    if isinstance(step_seconds, bool) or not isinstance(step_seconds, int) or step_seconds < 1:
        raise ValueError("step_seconds must be a positive integer")
    if isinstance(max_cutoffs, bool) or not isinstance(max_cutoffs, int) or max_cutoffs < 1:
        raise ValueError("max_cutoffs must be a positive integer")
    cutoff_count = int((end - start).total_seconds() // step_seconds) + 1
    if cutoff_count > max_cutoffs:
        raise ValueError(f"panel requires {cutoff_count} cutoffs; max_cutoffs is {max_cutoffs}")

    markets_input = list(market_snapshots)
    trades_input = list(trade_snapshots)
    candles_input = list(candlestick_snapshots)
    quote_snapshots = list(markets_input)
    for snapshot in candles_input:
        candle = snapshot.get("candle", snapshot)
        if not isinstance(candle, dict):
            raise ValueError("candlestick snapshot must contain a candle object")
        ticker = candle.get("market_ticker")
        end_ts = candle.get("end_period_ts")
        if not isinstance(ticker, str) or not ticker:
            raise ValueError("candlestick snapshot requires market_ticker")
        if isinstance(end_ts, bool) or not isinstance(end_ts, int) or end_ts <= 0:
            raise ValueError("candlestick end_period_ts must be a positive Unix timestamp")
        bid = candle.get("yes_bid")
        ask = candle.get("yes_ask")
        quote_snapshots.append({
            "captured_at": datetime.fromtimestamp(end_ts, timezone.utc).isoformat(),
            "market": {
                "ticker": ticker,
                "yes_bid": bid.get("close") if isinstance(bid, dict) else None,
                "yes_ask": ask.get("close") if isinstance(ask, dict) else None,
            },
        })

    market_to_event: dict[str, str] = {}
    latest_market: dict[str, tuple[datetime, dict[str, Any]]] = {}
    for snapshot in markets_input:
        market = _market_payload(snapshot)
        ticker, event = market.get("ticker"), market.get("event_ticker")
        if not isinstance(ticker, str) or not ticker:
            continue
        if isinstance(event, str) and event:
            market_to_event[ticker] = event
        captured_raw = snapshot.get("captured_at")
        if not isinstance(captured_raw, str):
            continue
        captured = _aware(captured_raw, f"captured_at for {ticker}")
        previous = latest_market.get(ticker)
        if previous is None or captured > previous[0]:
            latest_market[ticker] = (captured, market)
    if not market_to_event:
        raise ValueError("market snapshots contain no ticker-to-event mappings")

    markets = [entry[1] for entry in latest_market.values()]
    event_features = build_event_trade_features(
        trades_input, market_to_event, as_of=end, bucket_seconds=bucket_seconds
    )

    rows: list[dict[str, Any]] = []
    cutoff = start
    while cutoff <= end:
        cutoff_rows = build_resolved_market_dataset(
            markets,
            event_features,
            forecast_at=cutoff,
            max_feature_age_seconds=max_feature_age_seconds,
            market_snapshots=quote_snapshots,
            max_quote_age_seconds=max_quote_age_seconds,
        )
        candle_features = build_market_candle_features(
            candles_input, forecast_at=cutoff,
            max_age_seconds=max_feature_age_seconds,
        )
        for row in cutoff_rows:
            features = candle_features.get(row["market_ticker"], {})
            for name, value in features.items():
                if name != "market_candle_latest_at":
                    row[f"feature_{name}"] = value
            if features:
                row["market_candle_latest_at"] = features["market_candle_latest_at"]
        rows.extend(cutoff_rows)
        cutoff += timedelta(seconds=step_seconds)
    return sorted(rows, key=lambda row: (row["forecast_at"], row["market_ticker"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a leakage-safe chronological Kalshi research panel (offline only)"
    )
    parser.add_argument("--markets", required=True, help="Market snapshot JSONL")
    parser.add_argument("--trades", required=True, help="Trade snapshot JSONL")
    parser.add_argument("--candlesticks", required=True, help="Historical candle snapshot JSONL")
    parser.add_argument("--forecast-start", required=True, help="Timezone-aware ISO-8601 first cutoff")
    parser.add_argument("--forecast-end", required=True, help="Timezone-aware ISO-8601 final cutoff")
    parser.add_argument("--step-seconds", type=int, default=3600)
    parser.add_argument("--bucket-seconds", type=int, default=3600)
    parser.add_argument("--max-feature-age-seconds", type=int, default=86400)
    parser.add_argument("--max-quote-age-seconds", type=int, default=300)
    parser.add_argument("--max-cutoffs", type=int, default=1000)
    parser.add_argument("--output", default="data/research_panel.jsonl")
    args = parser.parse_args(argv)
    try:
        rows = build_dataset_panel_from_records(
            read_jsonl(args.markets),
            read_jsonl(args.trades),
            forecast_start=args.forecast_start,
            forecast_end=args.forecast_end,
            step_seconds=args.step_seconds,
            bucket_seconds=args.bucket_seconds,
            max_feature_age_seconds=args.max_feature_age_seconds,
            max_quote_age_seconds=args.max_quote_age_seconds,
            candlestick_snapshots=read_jsonl(args.candlesticks),
            max_cutoffs=args.max_cutoffs,
        )
        written = write_jsonl_atomic(rows, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Research panel build failed: {exc}\\n")
    print(json.dumps({
        "output": args.output,
        "rows_written": written,
        "forecast_start": _aware(args.forecast_start, "forecast_start").isoformat(),
        "forecast_end": _aware(args.forecast_end, "forecast_end").isoformat(),
        "step_seconds": args.step_seconds,
        "paper_only": True,
        "live_trading": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
