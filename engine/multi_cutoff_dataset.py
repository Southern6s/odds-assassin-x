"""Build one leakage-safe research observation per market and forecast cutoff."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .research_dataset import build_dataset_from_records, read_jsonl, write_jsonl_atomic


def build_multi_cutoff_dataset(
    market_snapshots: list[dict[str, Any]],
    trade_snapshots: list[dict[str, Any]],
    forecast_times: list[datetime],
    *,
    candlestick_snapshots: list[dict[str, Any]] | None = None,
    bucket_seconds: int = 3600,
    max_feature_age_seconds: int = 86400,
    max_quote_age_seconds: int = 300,
) -> list[dict[str, Any]]:
    """Build repeated point-in-time rows; no later observations enter earlier rows."""
    if not forecast_times:
        raise ValueError("at least one forecast cutoff is required")
    cutoffs = sorted(forecast_times)
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("forecast cutoffs must be unique")
    result = []
    for cutoff in cutoffs:
        if cutoff.tzinfo is None:
            raise ValueError("every forecast cutoff must be timezone-aware")
        result.extend(build_dataset_from_records(
            market_snapshots, trade_snapshots, forecast_at=cutoff,
            bucket_seconds=bucket_seconds,
            max_feature_age_seconds=max_feature_age_seconds,
            max_quote_age_seconds=max_quote_age_seconds,
            candlestick_snapshots=candlestick_snapshots or (),
        ))
    keys = [(row["market_ticker"], row["forecast_at"]) for row in result]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate market/cutoff observation generated")
    return sorted(result, key=lambda row: (row["forecast_at"], row["market_ticker"]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build multi-cutoff historical research data (offline; no trading)"
    )
    parser.add_argument("--markets", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--forecast-times", required=True,
                        help="Comma-separated timezone-aware ISO-8601 cutoffs")
    parser.add_argument("--candlesticks", default=None)
    parser.add_argument("--output", default="data/multi_cutoff_research.jsonl")
    parser.add_argument("--bucket-seconds", type=int, default=3600)
    parser.add_argument("--max-feature-age-seconds", type=int, default=86400)
    parser.add_argument("--max-quote-age-seconds", type=int, default=300)
    args = parser.parse_args(argv)
    try:
        cutoffs = [
            datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            for value in args.forecast_times.split(",") if value.strip()
        ]
        if any(cutoff.tzinfo is None for cutoff in cutoffs):
            raise ValueError("all --forecast-times values must include a timezone")
        rows = build_multi_cutoff_dataset(
            read_jsonl(args.markets), read_jsonl(args.trades), cutoffs,
            candlestick_snapshots=(
                read_jsonl(args.candlesticks) if args.candlesticks else []
            ),
            bucket_seconds=args.bucket_seconds,
            max_feature_age_seconds=args.max_feature_age_seconds,
            max_quote_age_seconds=args.max_quote_age_seconds,
        )
        written = write_jsonl_atomic(rows, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Multi-cutoff dataset build failed: {exc}\\n")
    print(json.dumps({
        "output": args.output, "rows_written": written,
        "forecast_cutoffs": len(cutoffs), "paper_only": True,
        "live_trading": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
