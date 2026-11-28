"""Synchronized read-only market quote and order-book shadow capture.

Each record derives its executable top-of-book quote and displayed depth from
one order-book response, with one capture timestamp. No forecast is fabricated:
live inference is explicitly marked unavailable until a point-in-time inference
pipeline is wired in. This module never places orders.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Callable

from .depth_collector import normalize_orderbook
from .kalshi_client import KalshiPublicClient


def _market_metadata(market: dict[str, Any]) -> dict[str, Any]:
    """Keep only stable descriptive fields from the market catalog response."""
    return {
        key: market.get(key)
        for key in (
            "ticker", "event_ticker", "series_ticker", "title", "subtitle",
            "category", "status", "market_type", "close_time",
            "expiration_time", "expected_expiration_time",
        )
    }


def collect_shadow_snapshots(
    client: Any,
    output_path: str | Path,
    *,
    max_markets: int = 20,
    duration_seconds: int = 300,
    interval_seconds: int = 30,
    max_market_pages: int = 1,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Capture quote and depth from the same response for a bounded open-market set."""
    if not 1 <= max_markets <= 100:
        raise ValueError("max_markets must be between 1 and 100")
    if not 1 <= duration_seconds <= 3600:
        raise ValueError("duration_seconds must be between 1 and 3600")
    if not 5 <= interval_seconds <= 600:
        raise ValueError("interval_seconds must be between 5 and 600")
    if not 1 <= max_market_pages <= 10:
        raise ValueError("max_market_pages must be between 1 and 10")

    catalog = list(client.iter_markets(
        status="open", limit=1000, max_pages=max_market_pages
    ))
    unique: dict[str, dict[str, Any]] = {}
    for market in catalog:
        ticker = market.get("ticker")
        if isinstance(ticker, str) and ticker.strip():
            unique.setdefault(ticker, market)
    if not unique:
        raise RuntimeError("No open market tickers were discovered")

    # Catalog order is not a liquidity signal. Screen a bounded candidate
    # set first, then keep the markets with real, preferably two-sided depth.
    # This avoids spending the entire capture window polling empty books.
    screened: list[tuple[dict[str, Any], dict[str, Any]]] = []
    screening_failures = 0
    empty_books = 0
    screen_limit = min(len(unique), max_markets * 5)
    for market in list(unique.values())[:screen_limit]:
        try:
            book = normalize_orderbook(
                client.get_market_orderbook(market["ticker"]), market["ticker"]
            )
            if not book["has_visible_liquidity"]:
                empty_books += 1
                continue
            screened.append((market, book))
        except (TypeError, ValueError, RuntimeError, OSError) as exc:
            screening_failures += 1
            print(
                f"shadow_screen_failed ticker={market['ticker']} error={exc}",
                flush=True,
            )

    screened.sort(
        key=lambda item: (
            item[1]["two_sided_book"],
            item[1]["top_level_depth"],
        ),
        reverse=True,
    )
    selected_pairs = screened[:max_markets]
    if not selected_pairs:
        raise RuntimeError(
            f"No visible order-book liquidity found across {screen_limit} candidates; "
            "increase max_market_pages or retry later"
        )
    selected = [market for market, _ in selected_pairs]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = clock()
    deadline = started + duration_seconds
    polls = written = 0
    failures = screening_failures + empty_books
    tickers_with_books: set[str] = set()

    while True:
        for market in selected:
            ticker = market["ticker"]
            try:
                payload = client.get_market_orderbook(ticker)
                captured = wall_clock()
                if captured.tzinfo is None:
                    raise ValueError("wall_clock must return a timezone-aware datetime")
                book = normalize_orderbook(payload, ticker, captured)
                if not book["has_visible_liquidity"]:
                    failures += 1
                    continue
                timestamp = captured.astimezone(timezone.utc).isoformat()
                record = {
                    "schema_version": 1,
                    "capture_id": f"{ticker}:{timestamp}",
                    "captured_at": timestamp,
                    "ticker": ticker,
                    "market": _market_metadata(market),
                    "quote": {
                        "yes_bid": book["yes_bid"],
                        "yes_ask": book["yes_ask"],
                        "no_bid": book["no_bid"],
                        "no_ask": book["no_ask"],
                        "quote_source": "same_orderbook_response",
                    },
                    "orderbook": book,
                    "quote_depth_same_capture": True,
                    "forecast_status": "not_generated_live_inference_not_wired",
                    "model_probability_yes": None,
                    "paper_only": True,
                    "live_trading": False,
                }
                with output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(
                        record, separators=(",", ":"), sort_keys=True,
                        allow_nan=False,
                    ) + "\n")
                written += 1
                tickers_with_books.add(ticker)
            except (TypeError, ValueError, RuntimeError, OSError) as exc:
                failures += 1
                print(f"shadow_capture_failed ticker={ticker} error={exc}", flush=True)

        polls += 1
        elapsed = clock() - started
        print(json.dumps({
            "poll": polls, "markets": len(selected), "records_written": written,
            "failures": failures, "elapsed_seconds": round(elapsed, 1),
            "paper_only": True, "live_trading": False,
        }), flush=True)
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleeper(min(interval_seconds, remaining))

    if written == 0:
        raise RuntimeError("No shadow records with visible order-book liquidity were captured")
    return {
        "status": "completed",
        "markets_selected": len(selected),
        "markets_with_books": len(tickers_with_books),
        "polls": polls,
        "records_written": written,
        "failures": failures,
        "selected_tickers": [market["ticker"] for market in selected],
        "forecast_status": "not_generated_live_inference_not_wired",
        "output": str(output),
        "paper_only": True,
        "live_trading": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture synchronized public Kalshi quotes and depth (paper-only)"
    )
    parser.add_argument("--output", default="data/shadow_snapshots.jsonl")
    parser.add_argument("--max-markets", type=int, default=20)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--max-market-pages", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = collect_shadow_snapshots(
            KalshiPublicClient(), args.output,
            max_markets=args.max_markets,
            duration_seconds=args.duration_seconds,
            interval_seconds=args.interval_seconds,
            max_market_pages=args.max_market_pages,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Shadow capture failed: {exc}\\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
