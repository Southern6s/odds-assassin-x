"""Filter collected Kalshi snapshots to ordinary, categorized binary markets.

The command-line path streams large trade/candle JSONL files so multi-million-row
artifacts can be filtered without retaining them all in memory. This is offline
research preparation only; it does not fetch data or place orders.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable


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
    if market.get("market_type") != "binary":
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
    """In-memory helper retained for unit tests and small inputs."""
    kept_markets = [row for row in market_snapshots if is_ordinary_binary_market(row)]
    kept_tickers = {_market(row)["ticker"] for row in kept_markets}
    kept_trades = [
        row for row in trade_snapshots
        if isinstance(row.get("trade", row), dict)
        and row.get("trade", row).get("ticker") in kept_tickers
    ]
    kept_candles = [
        row for row in candlestick_snapshots
        if isinstance(row.get("candle", row), dict)
        and row.get("candle", row).get("market_ticker") in kept_tickers
    ]
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


def filter_jsonl_stream(
    source: str | Path,
    destination: str | Path,
    *,
    ticker_field: str,
    kept_tickers: set[str],
) -> tuple[int, int]:
    """Filter one JSONL file with bounded memory and atomically publish output."""
    source_path, destination_path = Path(source), Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination_path.with_name(destination_path.name + ".tmp")
    total = kept = 0
    try:
        with source_path.open("r", encoding="utf-8") as input_handle, temporary.open(
            "w", encoding="utf-8"
        ) as output_handle:
            for line_number, line in enumerate(input_handle, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{source_path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"{source_path}:{line_number}: each record must be an object"
                    )
                payload = record.get("trade" if ticker_field == "ticker" else "candle", record)
                if isinstance(payload, dict) and payload.get(ticker_field) in kept_tickers:
                    output_handle.write(json.dumps(
                        record, separators=(",", ":"), sort_keys=True, allow_nan=False
                    ))
                    output_handle.write("\n")
                    kept += 1
        temporary.replace(destination_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return total, kept


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream-filter Kalshi research snapshots to ordinary binary markets"
    )
    parser.add_argument("--markets", required=True)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--candlesticks", required=True)
    parser.add_argument("--output-dir", default="data/ordinary_binary")
    args = parser.parse_args(argv)
    try:
        # Market catalog is comparatively small; trades and candles can be millions
        # of rows, so those files are processed one record at a time.
        market_input = read_jsonl(args.markets)
        markets = [row for row in market_input if is_ordinary_binary_market(row)]
        kept_tickers = {_market(row)["ticker"] for row in markets}
        output = Path(args.output_dir)
        write_jsonl(output / "market_snapshots.jsonl", markets)
        trades_total, trades_kept = filter_jsonl_stream(
            args.trades, output / "trade_snapshots.jsonl",
            ticker_field="ticker", kept_tickers=kept_tickers,
        )
        candles_total, candles_kept = filter_jsonl_stream(
            args.candlesticks, output / "candlestick_snapshots.jsonl",
            ticker_field="market_ticker", kept_tickers=kept_tickers,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Research universe filtering failed: {exc}\n")

    categories = Counter(
        str(_market(row).get("category") or "UNKNOWN") for row in market_input
    )
    report = {
        "markets_input": len(market_input),
        "markets_kept": len(markets),
        "markets_excluded": len(market_input) - len(markets),
        "trades_input": trades_total,
        "trades_kept": trades_kept,
        "trades_excluded": trades_total - trades_kept,
        "candles_input": candles_total,
        "candles_kept": candles_kept,
        "candles_excluded": candles_total - candles_kept,
        "categories_input": dict(sorted(categories.items())),
        "excluded_categories": ["Exotics"],
        "excluded_ticker_prefixes": ["KXMVE"],
        "output_dir": str(output),
        "paper_only": True,
        "live_trading": False,
        "streaming_large_files": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
