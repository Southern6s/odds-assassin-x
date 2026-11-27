"""Add a bounded, time-stratified set of resolved markets to the trade sample."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .filter_research_universe import is_ordinary_binary_market


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
    max_catalog_markets: int = 300,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Union only resolved ordinary-binary markets inside the requested window."""
    if isinstance(min_ts, bool) or not isinstance(min_ts, int) or min_ts < 0:
        raise ValueError("min_ts must be a nonnegative integer")
    if isinstance(max_ts, bool) or not isinstance(max_ts, int) or max_ts <= min_ts:
        raise ValueError("max_ts must be an integer greater than min_ts")
    if (isinstance(max_catalog_markets, bool) or not isinstance(max_catalog_markets, int)
            or not 1 <= max_catalog_markets <= 2000):
        raise ValueError("max_catalog_markets must be between 1 and 2000")

    merged: dict[str, dict[str, Any]] = {}

    def market_payload(row: dict[str, Any]) -> dict[str, Any]:
        market = row.get("market", row)
        return market if isinstance(market, dict) else {}

    def add(row: dict[str, Any]) -> None:
        ticker = market_payload(row).get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            merged.setdefault(ticker, row)

    hydrated_seen: set[str] = set()
    hydrated_tickers: set[str] = set()
    for row in hydrated_markets:
        market = market_payload(row)
        ticker = market.get("ticker")
        if not isinstance(ticker, str) or not ticker.strip():
            continue
        hydrated_seen.add(ticker)
        settled = _unix_seconds(market.get("settlement_ts"))
        if (is_ordinary_binary_market(row) and market.get("result") in ("yes", "no")
                and settled is not None and min_ts <= settled < max_ts):
            hydrated_tickers.add(ticker)
            add(row)

    eligible: list[tuple[float, str, dict[str, Any]]] = []
    seen_catalog: set[str] = set()
    for row in catalog_markets:
        market = market_payload(row)
        ticker = market.get("ticker")
        if (not ticker or ticker in seen_catalog or not is_ordinary_binary_market(row)
                or market.get("result") not in ("yes", "no")):
            continue
        settled = _unix_seconds(market.get("settlement_ts"))
        if settled is None or not min_ts <= settled < max_ts:
            continue
        seen_catalog.add(ticker)
        eligible.append((settled, ticker, row))

    eligible.sort(key=lambda item: (item[0], item[1]))
    if len(eligible) > max_catalog_markets:
        if max_catalog_markets == 1:
            selected = [eligible[len(eligible) // 2]]
        else:
            indices = {
                round(i * (len(eligible) - 1) / (max_catalog_markets - 1))
                for i in range(max_catalog_markets)
            }
            selected = [eligible[index] for index in sorted(indices)]
    else:
        selected = eligible

    for _, _, row in selected:
        add(row)

    result = [merged[key] for key in sorted(merged)]
    return result, {
        "trade_hydrated_markets": len(hydrated_seen),
        "trade_hydrated_ordinary_binary_resolved_in_window": len(hydrated_tickers),
        "trade_hydrated_markets_excluded": len(hydrated_seen - hydrated_tickers),
        "catalog_eligible_resolved_ordinary_binary_markets": len(eligible),
        "catalog_markets_selected": len(selected),
        "unique_markets_after_union": len(result),
        "added_catalog_markets": len(set(merged) - hydrated_tickers),
        "paper_only": True,
        "live_trading": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Union a bounded resolved ordinary-binary research cohort"
    )
    parser.add_argument("--hydrated", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--min-ts", required=True, type=int)
    parser.add_argument("--max-ts", required=True, type=int)
    parser.add_argument("--max-catalog-markets", type=int, default=300)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        merged, report = merge_settlement_window_markets(
            _read_jsonl(args.hydrated), _read_jsonl(args.catalog),
            min_ts=args.min_ts, max_ts=args.max_ts,
            max_catalog_markets=args.max_catalog_markets,
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
