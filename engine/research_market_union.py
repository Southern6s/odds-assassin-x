"""Add resolved markets from the requested settlement window to the trade sample.

This is bounded by the catalog scan page budget and exact settlement timestamps.
It performs no authenticated requests and never places orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected object")
            rows.append(row)
    return rows


def _unix_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return None
            return parsed.astimezone(timezone.utc).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def merge_settlement_window_markets(
    hydrated_markets: list[dict[str, Any]],
    catalog_markets: list[dict[str, Any]],
    *,
    min_ts: int,
    max_ts: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Union trade-hydrated markets with catalog markets settled in [min_ts,max_ts)."""
    if isinstance(min_ts, bool) or not isinstance(min_ts, int) or min_ts < 0:
        raise ValueError("min_ts must be a nonnegative integer")
    if isinstance(max_ts, bool) or not isinstance(max_ts, int) or max_ts <= min_ts:
        raise ValueError("max_ts must be an integer greater than min_ts")

    merged: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any]) -> None:
        market = row.get("market", row)
        ticker = market.get("ticker") if isinstance(market, dict) else None
        if isinstance(ticker, str) and ticker.strip():
            merged.setdefault(ticker, row)

    for row in hydrated_markets:
        add(row)

    catalog_candidates = 0
    for row in catalog_markets:
        market = row.get("market", row)
        if not isinstance(market, dict):
            continue
        settled = _unix_seconds(market.get("settlement_ts"))
        if settled is None or not min_ts <= settled < max_ts:
            continue
        catalog_candidates += 1
        add(row)

    result = [merged[key] for key in sorted(merged)]
    return result, {
        "trade_hydrated_markets": len({(r.get("market", r)).get("ticker") for r in hydrated_markets
                                      if isinstance(r.get("market", r), dict)
                                      and isinstance((r.get("market", r)).get("ticker"), str)}),
        "catalog_rows_in_settlement_window": catalog_candidates,
        "unique_markets_after_union": len(result),
        "added_catalog_markets": max(0, len(result) - len({
            (r.get("market", r)).get("ticker") for r in hydrated_markets
            if isinstance(r.get("market", r), dict)
            and isinstance((r.get("market", r)).get("ticker"), str)
        })),
        "paper_only": True,
        "live_trading": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Union trade-hydrated markets with catalog markets settled in a time window"
    )
    parser.add_argument("--hydrated", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--min-ts", required=True, type=int)
    parser.add_argument("--max-ts", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        merged, report = merge_settlement_window_markets(
            _read_jsonl(args.hydrated), _read_jsonl(args.catalog),
            min_ts=args.min_ts, max_ts=args.max_ts,
        )
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for row in merged:
                handle.write(json.dumps(row, separators=(",", ":"), sort_keys=True, allow_nan=False))
                handle.write("\n")
        temporary.replace(destination)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Settlement-window market merge failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
