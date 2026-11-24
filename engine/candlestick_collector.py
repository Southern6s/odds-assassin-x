"""Collect historical Kalshi candles for a hydrated market snapshot file."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .collector import collect_candlesticks
from .kalshi_client import KalshiPublicClient


def read_markets(path: str | Path) -> list[dict]:
    markets = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            market = record.get("market") if isinstance(record, dict) else None
            if not isinstance(market, dict):
                raise ValueError(f"{path}:{line_number}: missing market object")
            ticker = market.get("ticker")
            if not isinstance(ticker, str) or not ticker:
                raise ValueError(f"{path}:{line_number}: missing market ticker")
            if ticker not in seen:
                seen.add(ticker)
                markets.append(market)
    return markets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect read-only historical Kalshi candlesticks for hydrated markets"
    )
    parser.add_argument("--markets", required=True, help="Hydrated market snapshot JSONL")
    parser.add_argument("--start-ts", required=True, type=int, help="Inclusive Unix start time")
    parser.add_argument("--end-ts", required=True, type=int, help="Exclusive Unix end time")
    parser.add_argument("--period-interval", type=int, choices=(1, 60, 1440), default=1)
    parser.add_argument("--output", default="data/candlestick_snapshots.jsonl")
    args = parser.parse_args(argv)
    try:
        markets = read_markets(args.markets)
        report = collect_candlesticks(
            KalshiPublicClient(), markets, args.output,
            start_ts=args.start_ts, end_ts=args.end_ts,
            period_interval=args.period_interval,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Candlestick collection failed: {exc}\\n")
    print(json.dumps({
        **asdict(report),
        "unique_markets_in_input": len(markets),
        "period_interval_minutes": args.period_interval,
        "live_trading": False,
    }, indent=2, sort_keys=True))
    return 0 if report.markets_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
