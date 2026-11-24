"""Build a point-in-time labeled research dataset from collected Kalshi JSONL.

This is an offline research command. It never contacts the exchange or places
orders. Market outcomes are labels; trade prints and quotes are restricted to
the requested forecast cutoff before they become features.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

from .event_research import build_event_trade_features
from .outcome_dataset import build_resolved_market_dataset


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL strictly, reporting the source line for malformed records."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: each record must be an object")
            records.append(record)
    return records


def _market_payload(record: dict[str, Any]) -> dict[str, Any]:
    market = record.get("market", record)
    if not isinstance(market, dict):
        raise ValueError("market snapshot must contain a market object")
    return market


def build_dataset_from_records(
    market_snapshots: Iterable[dict[str, Any]],
    trade_snapshots: Iterable[dict[str, Any]],
    *,
    forecast_at: datetime,
    bucket_seconds: int = 3600,
    max_feature_age_seconds: int = 86400,
    max_quote_age_seconds: int = 300,
) -> list[dict[str, Any]]:
    """Create labeled rows using only information available by forecast_at."""
    markets_input = list(market_snapshots)
    trades_input = list(trade_snapshots)
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
        try:
            captured = datetime.fromisoformat(captured_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid captured_at for market {ticker}") from exc
        if captured.tzinfo is None:
            raise ValueError(f"captured_at for market {ticker} must include timezone")
        captured = captured.astimezone(timezone.utc)
        previous = latest_market.get(ticker)
        if previous is None or captured > previous[0]:
            latest_market[ticker] = (captured, market)

    if not market_to_event:
        raise ValueError("market snapshots contain no ticker-to-event mappings")

    markets = [entry[1] for entry in latest_market.values()]
    event_features = build_event_trade_features(
        trades_input, market_to_event, as_of=forecast_at,
        bucket_seconds=bucket_seconds,
    )
    return build_resolved_market_dataset(
        markets,
        event_features,
        forecast_at=forecast_at,
        max_feature_age_seconds=max_feature_age_seconds,
        market_snapshots=markets_input,
        max_quote_age_seconds=max_quote_age_seconds,
    )


def write_jsonl_atomic(rows: Iterable[dict[str, Any]], output_path: str | Path) -> int:
    """Write a complete export then replace the destination to avoid partial files."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True,
                                        allow_nan=False))
                handle.write("\\n")
                count += 1
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a point-in-time Kalshi research dataset (offline; no trading)"
    )
    parser.add_argument("--markets", required=True, help="Market snapshot JSONL")
    parser.add_argument("--trades", required=True, help="Trade snapshot JSONL")
    parser.add_argument("--forecast-at", required=True,
                        help="Timezone-aware ISO-8601 decision timestamp")
    parser.add_argument("--output", default="data/research_dataset.jsonl")
    parser.add_argument("--bucket-seconds", type=int, default=3600)
    parser.add_argument("--max-feature-age-seconds", type=int, default=86400)
    parser.add_argument("--max-quote-age-seconds", type=int, default=300)
    args = parser.parse_args(argv)

    try:
        cutoff = datetime.fromisoformat(args.forecast_at.replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            raise ValueError("--forecast-at must include a timezone")
        rows = build_dataset_from_records(
            read_jsonl(args.markets),
            read_jsonl(args.trades),
            forecast_at=cutoff,
            bucket_seconds=args.bucket_seconds,
            max_feature_age_seconds=args.max_feature_age_seconds,
            max_quote_age_seconds=args.max_quote_age_seconds,
        )
        written = write_jsonl_atomic(rows, args.output)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Dataset build failed: {exc}\\n")

    print(json.dumps({
        "output": str(args.output),
        "rows_written": written,
        "forecast_at": cutoff.astimezone(timezone.utc).isoformat(),
        "paper_only": True,
        "live_trading": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
