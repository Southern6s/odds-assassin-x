"""Collect bounded public trades market-by-market to avoid newest-first global-page bias."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .kalshi_client import KalshiPublicClient
from .kalshi_data import append_jsonl
from .trade_data import trade_snapshot


@dataclass
class MarketTradeReport:
    captured_at: str
    markets_requested: int = 0
    markets_queried: int = 0
    markets_with_trades: int = 0
    markets_without_trades: int = 0
    markets_failed: int = 0
    written: int = 0
    duplicates: int = 0
    outside_window: int = 0
    rejected: int = 0
    failed_tickers: list[str] = field(default_factory=list)


def read_market_tickers(path: str | Path) -> list[str]:
    tickers: list[str] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            market = record.get("market") if isinstance(record, dict) else None
            ticker = market.get("ticker") if isinstance(market, dict) else None
            if not isinstance(ticker, str) or not ticker.strip():
                raise ValueError(f"{path}:{line_number}: missing market ticker")
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    return tickers


def _trade_time(raw: dict[str, Any]) -> datetime:
    value = raw.get("created_time")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("created_time must include a timezone")
    return parsed.astimezone(timezone.utc)


def collect_market_trade_snapshots(
    client: KalshiPublicClient,
    markets_path: str | Path,
    output_path: str | Path,
    *,
    min_ts: int,
    max_ts: int,
    max_pages_per_tier: int = 5,
) -> MarketTradeReport:
    """Query each hydrated ticker independently and append unique in-window prints."""
    if isinstance(min_ts, bool) or not isinstance(min_ts, int) or min_ts < 0:
        raise ValueError("min_ts must be a nonnegative Unix-second integer")
    if isinstance(max_ts, bool) or not isinstance(max_ts, int) or max_ts <= min_ts:
        raise ValueError("max_ts must be greater than min_ts")
    if isinstance(max_pages_per_tier, bool) or not 1 <= max_pages_per_tier <= 1000:
        raise ValueError("max_pages_per_tier must be between 1 and 1000")

    tickers = read_market_tickers(markets_path)
    captured = datetime.now(timezone.utc)
    report = MarketTradeReport(captured_at=captured.isoformat(), markets_requested=len(tickers))
    seen_ids: set[str] = set()
    output = Path(output_path)
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                existing = json.loads(line)
                trade = existing.get("trade", {})
                trade_id = trade.get("trade_id") if isinstance(trade, dict) else None
                if isinstance(trade_id, str) and trade_id:
                    seen_ids.add(trade_id)

    for ticker in tickers:
        report.markets_queried += 1
        market_rows: list[dict[str, Any]] = []
        market_trade_ids: set[str] = set()
        try:
            for raw in client.iter_all_trade_tiers(
                ticker=ticker,
                min_ts=min_ts,
                max_ts=max_ts,
                limit=1000,
                max_pages_per_tier=max_pages_per_tier,
            ):
                try:
                    created = _trade_time(raw)
                    if not min_ts <= created.timestamp() < max_ts:
                        report.outside_window += 1
                        continue
                    trade_id = raw.get("trade_id")
                    if not isinstance(trade_id, str) or not trade_id:
                        report.rejected += 1
                        continue
                    if trade_id in seen_ids or trade_id in market_trade_ids:
                        report.duplicates += 1
                        continue
                    row = trade_snapshot(raw, captured_at=captured)
                    market_rows.append(row)
                    market_trade_ids.add(trade_id)
                except (AttributeError, TypeError, ValueError):
                    report.rejected += 1
            if market_rows:
                report.written += append_jsonl(market_rows, output)
                seen_ids.update(market_trade_ids)
                report.markets_with_trades += 1
            else:
                report.markets_without_trades += 1
        except Exception:
            report.markets_failed += 1
            if len(report.failed_tickers) < 50:
                report.failed_tickers.append(ticker)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect bounded Kalshi trades per market (read-only; no orders)"
    )
    parser.add_argument("--markets", required=True, help="Hydrated market snapshot JSONL")
    parser.add_argument("--min-ts", required=True, type=int, help="Inclusive Unix seconds")
    parser.add_argument("--max-ts", required=True, type=int, help="Exclusive Unix seconds")
    parser.add_argument("--max-pages-per-tier", type=int, default=5)
    parser.add_argument("--output", default="data/trade_snapshots.jsonl")
    args = parser.parse_args(argv)
    try:
        report = collect_market_trade_snapshots(
            KalshiPublicClient(), args.markets, args.output,
            min_ts=args.min_ts, max_ts=args.max_ts,
            max_pages_per_tier=args.max_pages_per_tier,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Market trade collection failed: {exc}\\n")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.markets_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
