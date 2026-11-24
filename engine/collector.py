"""Read-only Kalshi market snapshot collection and CLI orchestration."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .kalshi_client import KalshiPublicClient
from .kalshi_data import append_jsonl, snapshot_record


@dataclass
class CollectionReport:
    captured_at: str
    written: int = 0
    rejected: int = 0
    duplicates: int = 0
    historical_cutoff: dict[str, Any] = field(default_factory=dict)
    rejection_samples: list[str] = field(default_factory=list)


def collect_market_snapshot(
    markets: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    captured_at: datetime | None = None,
) -> CollectionReport:
    """Append a timestamp-consistent snapshot; malformed rows are counted, not guessed."""
    timestamp = captured_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    report = CollectionReport(captured_at=timestamp.isoformat())

    def valid_records():
        for market in markets:
            try:
                record = snapshot_record(market, timestamp)
            except (TypeError, ValueError) as exc:
                report.rejected += 1
                if len(report.rejection_samples) < 20:
                    ticker = market.get("ticker", "<missing>") if isinstance(market, dict) else "<invalid-record>"
                    report.rejection_samples.append(f"{ticker}: {exc}")
                continue
            yield record

    report.written = append_jsonl(valid_records(), output_path)
    return report


def collect_from_kalshi(
    client: KalshiPublicClient,
    output_path: str | Path,
    *,
    limit: int = 1000,
    max_pages_per_tier: int | None = None,
) -> CollectionReport:
    """Collect unique markets from live and historical tiers into append-only JSONL."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_pages_per_tier is not None and max_pages_per_tier < 1:
        raise ValueError("max_pages_per_tier must be positive")

    cutoff = client.get_historical_cutoff()
    seen: set[str] = set()
    report = CollectionReport(
        captured_at=datetime.now(timezone.utc).isoformat(),
        historical_cutoff=cutoff,
    )

    def unique_markets():
        for market in client.iter_categorized_markets(
            limit=limit, max_pages_per_tier=max_pages_per_tier
        ):
            ticker = market.get("ticker")
            if isinstance(ticker, str) and ticker:
                if ticker in seen:
                    report.duplicates += 1
                    continue
                seen.add(ticker)
            try:
                record = snapshot_record(market, datetime.fromisoformat(report.captured_at))
            except (TypeError, ValueError) as exc:
                report.rejected += 1
                if len(report.rejection_samples) < 20:
                    report.rejection_samples.append(f"{ticker or '<missing>'}: {exc}")
                continue
            yield record

    report.written = append_jsonl(unique_markets(), output_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect public Kalshi market snapshots (read-only)")
    parser.add_argument("--output", default="data/market_snapshots.jsonl")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages-per-tier", type=int, default=None,
                        help="Optional safety cap for test runs; omit to page through all markets")
    args = parser.parse_args(argv)
    try:
        report = collect_from_kalshi(
            KalshiPublicClient(),
            args.output,
            limit=args.limit,
            max_pages_per_tier=args.max_pages_per_tier,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Collection failed: {exc}\\n")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0 if report.rejected == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
