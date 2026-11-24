"""Hydrate market metadata for tickers observed in a bounded trade sample."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .collector import collect_market_snapshot
from .kalshi_client import KalshiPublicClient, enrich_market_series


@dataclass
class HydrationReport:
    trade_rows: int
    unique_tickers: int
    markets_written: int
    markets_rejected: int
    missing_tickers: int


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hydrate read-only Kalshi market metadata for collected trades"
    )
    parser.add_argument("--trades", required=True, help="Trade snapshot JSONL")
    parser.add_argument("--output", default="data/market_snapshots.jsonl")
    args = parser.parse_args(argv)

    try:
        trade_rows = 0
        tickers: list[str] = []
        seen: set[str] = set()
        with Path(args.trades).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                trade = record.get("trade") if isinstance(record, dict) else None
                ticker = trade.get("ticker") if isinstance(trade, dict) else None
                trade_rows += 1
                if isinstance(ticker, str) and ticker and ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)

        client = KalshiPublicClient()
        series_records = client.get_series()
        markets = [
            enrich_market_series(market, series_records)
            for market in client.iter_markets_by_tickers(tickers)
        ]
        found = {m.get("ticker") for m in markets if isinstance(m.get("ticker"), str)}
        report = collect_market_snapshot(markets, args.output)
        result = HydrationReport(
            trade_rows=trade_rows,
            unique_tickers=len(tickers),
            markets_written=report.written,
            markets_rejected=report.rejected,
            missing_tickers=len(set(tickers) - found),
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Market hydration failed: {exc}\\n")

    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.markets_rejected == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
