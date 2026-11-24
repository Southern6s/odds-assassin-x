"""Bounded, read-only public Kalshi trade collector.

Use explicit Unix-second time bounds. Output is append-only JSONL and intended
for local research; this command never authenticates or places orders.
"""
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
class TradeCollectionReport:
    captured_at: str
    min_ts: int
    max_ts: int
    written: int = 0
    rejected: int = 0
    duplicates: int = 0
    rejection_samples: list[str] = field(default_factory=list)


def collect_trade_snapshots(
    client: KalshiPublicClient,
    output_path: str | Path,
    *,
    min_ts: int,
    max_ts: int,
    ticker: str | None = None,
    is_block_trade: bool | None = None,
    limit: int = 1000,
    max_pages_per_tier: int | None = None,
) -> TradeCollectionReport:
    """Collect unique public trades within a required, bounded time interval."""
    if isinstance(min_ts, bool) or not isinstance(min_ts, int) or min_ts < 0:
        raise ValueError("min_ts must be a nonnegative Unix-second integer")
    if isinstance(max_ts, bool) or not isinstance(max_ts, int) or max_ts <= min_ts:
        raise ValueError("max_ts must be greater than min_ts")
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_pages_per_tier is not None and (
        isinstance(max_pages_per_tier, bool) or max_pages_per_tier < 1
    ):
        raise ValueError("max_pages_per_tier must be positive")
    if ticker is not None and (not isinstance(ticker, str) or not ticker.strip()):
        raise ValueError("ticker must be a non-empty string")

    captured = datetime.now(timezone.utc)
    report = TradeCollectionReport(
        captured_at=captured.isoformat(), min_ts=min_ts, max_ts=max_ts
    )
    seen: set[str] = set()

    def records():
        for raw in client.iter_all_trade_tiers(
            ticker=ticker, min_ts=min_ts, max_ts=max_ts,
            is_block_trade=is_block_trade, limit=limit,
            max_pages_per_tier=max_pages_per_tier,
        ):
            trade_id = raw.get("trade_id") if isinstance(raw, dict) else None
            if isinstance(trade_id, str) and trade_id in seen:
                report.duplicates += 1
                continue
            try:
                record = trade_snapshot(raw, captured_at=captured)
            except (TypeError, ValueError) as exc:
                report.rejected += 1
                if len(report.rejection_samples) < 20:
                    report.rejection_samples.append(f"{trade_id or '<missing>'}: {exc}")
                continue
            seen.add(record["trade"]["trade_id"])
            yield record

    report.written = append_jsonl(records(), output_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect bounded public Kalshi trades (read-only; no trading)"
    )
    parser.add_argument("--min-ts", type=int, required=True, help="Inclusive Unix seconds")
    parser.add_argument("--max-ts", type=int, required=True, help="Exclusive Unix seconds")
    parser.add_argument("--ticker", default=None, help="Optional market ticker filter")
    parser.add_argument("--block-trades", choices=("all", "yes", "no"), default="all")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages-per-tier", type=int, default=None)
    parser.add_argument("--output", default="data/trade_snapshots.jsonl")
    args = parser.parse_args(argv)
    block_filter = {"all": None, "yes": True, "no": False}[args.block_trades]
    try:
        report = collect_trade_snapshots(
            KalshiPublicClient(), args.output,
            min_ts=args.min_ts, max_ts=args.max_ts, ticker=args.ticker,
            is_block_trade=block_filter, limit=args.limit,
            max_pages_per_tier=args.max_pages_per_tier,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Trade collection failed: {exc}\n")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.rejected == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
