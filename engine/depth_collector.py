"""Repeated read-only Kalshi order-book snapshots for paper execution research.

Supports both legacy YES/NO arrays and Kalshi's current orderbook_fp
YES/NO dollar arrays. Markets are pre-screened for actual visible liquidity;
empty books are never counted as successful snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from .kalshi_client import KalshiPublicClient


def _levels(value: Any, side: str) -> list[tuple[float, float]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{side} orderbook levels must be a list")
    result = []
    for level in value:
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            raise ValueError(f"malformed {side} orderbook level")
        price, size = float(level[0]), float(level[1])
        # Integer-cent legacy prices and dollar/probability strings are both
        # accepted. Current orderbook_fp values are already dollar prices.
        if price > 1:
            price /= 100.0
        if not 0 <= price <= 1 or size < 0:
            raise ValueError(f"invalid {side} orderbook level")
        if size > 0:
            result.append((price, size))
    return result


def normalize_orderbook(payload: dict[str, Any], ticker: str,
                        captured_at: datetime | None = None) -> dict[str, Any]:
    """Normalize YES/NO bid depth and derive complementary ask depth."""
    if not isinstance(payload, dict):
        raise ValueError("orderbook response must be an object")
    book = payload.get("orderbook", payload)
    if not isinstance(book, dict):
        raise ValueError("orderbook payload is missing")
    # Kalshi's current public response uses orderbook_fp with *_dollars keys.
    yes_raw = book.get("yes")
    no_raw = book.get("no")
    if yes_raw is None:
        yes_raw = book.get("yes_dollars", book.get("orderbook_fp", {}).get("yes_dollars", []))
    if no_raw is None:
        no_raw = book.get("no_dollars", book.get("orderbook_fp", {}).get("no_dollars", []))
    yes = _levels(yes_raw, "yes")
    no = _levels(no_raw, "no")
    yes_best = max(yes, key=lambda item: item[0], default=None)
    no_best = max(no, key=lambda item: item[0], default=None)
    captured = captured_at or datetime.now(timezone.utc)
    if captured.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    return {
        "schema_version": 2,
        "captured_at": captured.astimezone(timezone.utc).isoformat(),
        "ticker": ticker,
        "yes_bid": yes_best[0] if yes_best else None,
        "yes_bid_size": yes_best[1] if yes_best else None,
        "yes_ask": 1.0 - no_best[0] if no_best else None,
        "yes_ask_size": no_best[1] if no_best else None,
        "no_bid": no_best[0] if no_best else None,
        "no_bid_size": no_best[1] if no_best else None,
        "no_ask": 1.0 - yes_best[0] if yes_best else None,
        "no_ask_size": yes_best[1] if yes_best else None,
        "yes_bid_levels": [{"price": p, "size": s} for p, s in sorted(yes, reverse=True)],
        "no_bid_levels": [{"price": p, "size": s} for p, s in sorted(no, reverse=True)],
        "has_visible_liquidity": bool(yes or no),
        "two_sided_book": bool(yes and no),
        "top_level_depth": (yes_best[1] if yes_best else 0) + (no_best[1] if no_best else 0),
        "raw_orderbook": payload,
        "paper_only": True,
        "live_trading": False,
    }


def collect_depth_snapshots(client: Any, output_path: str | Path, *,
                            max_markets: int = 20, duration_seconds: int = 300,
                            interval_seconds: int = 30, max_market_pages: int = 1,
                            sleeper=time.sleep, clock=time.monotonic) -> dict[str, Any]:
    """Poll a bounded set of liquid open markets until duration expires."""
    if not 1 <= max_markets <= 100:
        raise ValueError("max_markets must be between 1 and 100")
    if duration_seconds < 1 or duration_seconds > 3600:
        raise ValueError("duration_seconds must be between 1 and 3600")
    if interval_seconds < 5 or interval_seconds > 600:
        raise ValueError("interval_seconds must be between 5 and 600")
    if max_market_pages < 1 or max_market_pages > 10:
        raise ValueError("max_market_pages must be between 1 and 10")

    markets = list(client.iter_markets(status="open", limit=1000,
                                       max_pages=max_market_pages))
    candidates = list(dict.fromkeys(
        item.get("ticker") for item in markets
        if isinstance(item.get("ticker"), str) and item.get("ticker")
    ))
    if not candidates:
        raise RuntimeError("No open market tickers were discovered")

    # Screen candidates against actual books rather than trusting catalog order.
    # Prefer two-sided books and greater displayed top-level depth.
    screened: list[dict[str, Any]] = []
    empty_books = 0
    screening_failures = 0
    for ticker in candidates:
        try:
            record = normalize_orderbook(client.get_market_orderbook(ticker), ticker)
            if not record["has_visible_liquidity"]:
                empty_books += 1
                continue
            screened.append(record)
        except (TypeError, ValueError, RuntimeError) as exc:
            screening_failures += 1
            print(f"orderbook_screen_failed ticker={ticker} error={exc}", flush=True)
        if len(screened) >= max_markets * 5:
            # Enough candidates to rank a useful liquid subset without scanning
            # every market page when the catalog is already healthy.
            break
    screened.sort(key=lambda row: (
        row["two_sided_book"], row["top_level_depth"]
    ), reverse=True)
    selected = screened[:max_markets]
    if not selected:
        raise RuntimeError(
            f"No visible order-book liquidity found across {len(candidates)} candidates; "
            "increase max_market_pages or retry later"
        )
    tickers = [row["ticker"] for row in selected]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = clock()
    deadline = started + duration_seconds
    polls = 0
    written = 0
    failures = screening_failures
    empty_polls = 0
    while True:
        captured = datetime.now(timezone.utc)
        with output.open("a", encoding="utf-8") as handle:
            for ticker in tickers:
                try:
                    payload = client.get_market_orderbook(ticker)
                    record = normalize_orderbook(payload, ticker, captured)
                    if not record["has_visible_liquidity"]:
                        empty_polls += 1
                        failures += 1
                        print(f"orderbook_empty ticker={ticker}", flush=True)
                        continue
                    handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
                    written += 1
                except (TypeError, ValueError, RuntimeError) as exc:
                    failures += 1
                    print(f"orderbook_failed ticker={ticker} error={exc}", flush=True)
        polls += 1
        elapsed = clock() - started
        print(json.dumps({
            "poll": polls, "markets": len(tickers), "snapshots_written": written,
            "failures": failures, "empty_books": empty_polls,
            "elapsed_seconds": round(elapsed, 1),
            "paper_only": True, "live_trading": False,
        }), flush=True)
        remaining = deadline - clock()
        if remaining <= 0:
            break
        sleeper(min(interval_seconds, remaining))
    return {
        "status": "completed", "markets": len(tickers), "polls": polls,
        "snapshots_written": written, "failures": failures,
        "empty_books": empty_polls, "screened_candidates": len(candidates),
        "selected_tickers": tickers,
        "output": str(output), "paper_only": True, "live_trading": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect repeated public Kalshi order-book snapshots")
    parser.add_argument("--output", default="data/orderbook_snapshots.jsonl")
    parser.add_argument("--max-markets", type=int, default=20)
    parser.add_argument("--duration-seconds", type=int, default=300)
    parser.add_argument("--interval-seconds", type=int, default=30)
    parser.add_argument("--max-market-pages", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        report = collect_depth_snapshots(
            KalshiPublicClient(), args.output, max_markets=args.max_markets,
            duration_seconds=args.duration_seconds, interval_seconds=args.interval_seconds,
            max_market_pages=args.max_market_pages,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Collection failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
