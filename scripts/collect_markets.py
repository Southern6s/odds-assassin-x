"""CLI entry point for a single full-tier Kalshi public-data collection.

Example:
    python -m scripts.collect_markets --output data/market_snapshots.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.collector import collect_market_snapshot
from engine.kalshi_client import KalshiPublicClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect read-only Kalshi market snapshots.")
    parser.add_argument("--output", default="data/market_snapshots.jsonl",
                        help="Append-only JSONL destination.")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Markets per API page (1-1000).")
    parser.add_argument("--max-pages-per-tier", type=int, default=None,
                        help="Optional safety cap; omit to follow every page.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be between 1 and 1000")
    if args.max_pages_per_tier is not None and args.max_pages_per_tier < 1:
        raise SystemExit("--max-pages-per-tier must be positive")

    client = KalshiPublicClient()
    try:
        cutoff = client.get_historical_cutoff()
        markets = client.iter_categorized_markets(
            limit=args.limit, max_pages_per_tier=args.max_pages_per_tier
        )
        report = collect_market_snapshot(markets, Path(args.output))
    except Exception as exc:
        raise SystemExit(f"Collection failed; inspect API/network and retry: {exc}") from exc

    print(json.dumps({
        "status": "completed",
        "historical_cutoff": cutoff,
        "captured_at": report.captured_at,
        "written": report.written,
        "rejected": report.rejected,
        "rejection_samples": report.rejection_samples,
        "output": str(Path(args.output)),
        "limited_run": args.max_pages_per_tier is not None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
