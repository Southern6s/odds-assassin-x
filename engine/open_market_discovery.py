"""Resumable discovery of currently open Kalshi markets only.

This dataset is intentionally separate from the broad live/archive catalog so
active-market research never mutates or inherits the large historical checkpoint.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .catalog_discovery import _atomic_json, _load_seen_tickers
from .kalshi_client import KalshiPublicClient, enrich_market_category, series_category_map
from .kalshi_data import append_jsonl, snapshot_record

STATE_VERSION = 1
STATUS = "open"


def discover_open_resumable(
    client: KalshiPublicClient,
    output_path: str | Path,
    state_path: str | Path,
    *,
    limit: int = 1000,
    max_pages: int = 10,
) -> dict[str, Any]:
    """Append complete pages from the live open-market filter and checkpoint."""
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_pages < 1:
        raise ValueError("max_pages must be positive")

    output = Path(output_path)
    state_file = Path(state_path)
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if (
            state.get("version") != STATE_VERSION
            or state.get("limit") != limit
            or state.get("status") != STATUS
        ):
            raise ValueError("Open-market checkpoint settings mismatch")
    else:
        state = {
            "version": STATE_VERSION,
            "limit": limit,
            "status": STATUS,
            "cursor": None,
            "pages_completed": 0,
            "scanned": 0,
            "written": 0,
            "rejected": 0,
            "duplicates": 0,
            "captured_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(state_file, state)

    seen = _load_seen_tickers(output)
    categories = series_category_map(client.get_series())
    pages_this_run = 0
    started = datetime.now(timezone.utc)

    while not state.get("complete", False) and pages_this_run < max_pages:
        page, next_cursor = client.get_market_page(
            "markets", cursor=state["cursor"], limit=limit, status=STATUS
        )
        captured_at = datetime.now(timezone.utc)
        records = []
        page_rejected = 0
        page_duplicates = 0
        for raw_market in page:
            state["scanned"] += 1
            market = enrich_market_category(raw_market, categories)
            ticker = market.get("ticker")
            if isinstance(ticker, str) and ticker:
                if ticker in seen:
                    page_duplicates += 1
                    continue
            try:
                record = snapshot_record(market, captured_at)
            except (TypeError, ValueError):
                page_rejected += 1
                continue
            if isinstance(ticker, str) and ticker:
                seen.add(ticker)
            records.append(record)

        written = append_jsonl(records, output)
        state["written"] += written
        state["rejected"] += page_rejected
        state["duplicates"] += page_duplicates
        state["pages_completed"] += 1
        pages_this_run += 1
        state["cursor"] = next_cursor
        state["complete"] = next_cursor is None
        _atomic_json(state_file, state)
        elapsed = int((datetime.now(timezone.utc) - started).total_seconds())
        print(
            f"Open-market checkpoint: page={state['pages_completed']}, "
            f"page_markets={len(page)}, total_scanned={state['scanned']}, "
            f"total_written={state['written']}, duplicates={state['duplicates']}, "
            f"elapsed_seconds={elapsed}",
            flush=True,
        )

    result = dict(state)
    result["pages_this_run"] = pages_this_run
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable read-only Kalshi open-market discovery")
    parser.add_argument("--output", default="data/open_market_snapshots.jsonl")
    parser.add_argument("--state-file", default="data/open_market_state.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        report = discover_open_resumable(
            KalshiPublicClient(), args.output, args.state_file,
            limit=args.limit, max_pages=args.max_pages,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Open-market discovery failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
