"""Filter collected Kalshi snapshots to ordinary, categorized binary markets.

This is an offline research-preparation step. It excludes the Exotics category
and known MVE-prefixed contracts even when the exchange labels them binary.
It does not fetch data or place orders.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: each record must be an object")
            records.append(record)
    return records


def _market(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("market", record)
    return value if isinstance(value, dict) else {}


def is_ordinary_binary_market(record: dict[str, Any]) -> bool:
    """Return true only for categorized binary markets outside known MVE sets."""
    market = _market(record)
    ticker = market.get("ticker")
    event_ticker = market.get("event_ticker")
    category = market.get("category")
    market_type = market.get("market_type")
    if market_type != "binary":
        return False
    if not isinstance(category, str) or not category.strip():
        return False
    if category.strip().casefold() == "exotics":
        return False
    if isinstance(ticker, str) and ticker.upper().startswith("KXMVE"):
        return False
    if isinstance(event_ticker, str) and event_ticker.upper().startswith("KXMVE"):
        return False
    return isinstance(ticker, str) and bool(ticker.strip())


def filter_research_records(
    market_snapshots: list[dict[str, Any]],
    trade_snapshots: list[dict[str, Any]],
    candlestick_snapshots: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Filter trades and candles to the retained ordinary binary market tickers."""
    kept_markets = [row for row in market_snapshots if is_ordinary_binary_market(row)]
    kept_tickers = {_market(row)["ticker"] for row in kept_markets}

    kept_trades = []
    for row in trade_snapshots:
        trade = row.get("trade", row)
        if isinstance(trade, dict) and trade.get("ticker") in kept_tickers:
            kept_trades.append(row)

    kept_candles = []
    for row in candlestick_snapshots:
        candle = row.get("candle", row)
        if isinstance(candle, dict) and candle.get("market_ticker") in kept_tickers:
            kept_candles.append(row)

    categories = Counter(
        str(_market(row).get("category") or "UNKNOWN") for row in market_snapshots
    )
    report = {
        "markets_input": len(market_snapshots),
        "markets_kept": len(kept_markets),
        "markets_excluded": len(market_snapshots) - len(kept_markets),
        "trades_input": len(trade_snapshots),
        "trades_kept": len(kept_trades),
        "trades_excluded": len(trade_snapshots) - len(kept_trades),
        "candles_input": len(candlestick_snapshots),
        "candles_kept": len(kept_candles),
        "candles_excluded": len(candlestick_snapshots) - len(kept_candles),
        "categories_input": dict(sorted(categories.items())),
        "excluded_categories": ["Exotics"],
        "excluded_ticker_prefixes": ["KXMVE"],
        "paper_only": True,
        "live_trading": False,
    }
    return kept_markets, kept_trades, kept_candles, report


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(
                    record, separators=(",", ":"), sort_keys=True, allow_nan=False
                ))
                handle.write("\n")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Filter Kalshi research snapshots to ordinary binary markets"
    )
    parser.add_argument("--markets", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--candlesticks", required=True)
    parser.add_argument("--output-dir", default="data/ordinary_binary")
    args = parser.parse_args(argv)
    try:
        markets, trades, candles, report = filter_research_records(
            read_jsonl(args.markets), read_jsonl(args.trades),
            read_jsonl(args.candlesticks),
        )
        output = Path(args.output_dir)
        write_jsonl(output / "market_snapshots.jsonl", markets)
        write_jsonl(output / "trade_snapshots.jsonl", trades)
        write_jsonl(output / "candlestick_snapshots.jsonl", candles)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Research universe filtering failed: {exc}\n")
    report["output_dir"] = str(output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
