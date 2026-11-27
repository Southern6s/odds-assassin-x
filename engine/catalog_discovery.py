"""Resumable, page-committed Kalshi market-universe discovery."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .kalshi_client import (
    KalshiPublicClient,
    enrich_market_category,
    enrich_market_series,
    series_category_map,
)
from .kalshi_data import append_jsonl, snapshot_record

# Search the archive first: it is where the requested pre-cutoff settled
# cohort lives. The live settled tier is still scanned after the archive.
TIERS = ("historical/markets", "markets")
STATE_VERSION = 2


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_seen_tickers(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ticker = row.get("market", {}).get("ticker")
                if isinstance(ticker, str) and ticker:
                    seen.add(ticker)
            except (json.JSONDecodeError, AttributeError) as exc:
                raise ValueError(f"Existing catalog JSONL is invalid at line {line_number}") from exc
    return seen


def discover_resumable(
    client: KalshiPublicClient,
    output_path: str | Path,
    state_path: str | Path,
    *,
    limit: int = 1000,
    max_pages_per_tier: int = 10,
) -> dict[str, Any]:
    """Append complete pages, then atomically checkpoint their next cursor.

    If a process stops after appending but before checkpointing, existing ticker
    keys make replay of that page idempotent on the next invocation.
    """
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_pages_per_tier < 1:
        raise ValueError("max_pages_per_tier must be positive")
    output = Path(output_path)
    state_file = Path(state_path)
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if state.get("version") != STATE_VERSION or state.get("limit") != limit:
            raise ValueError("Checkpoint version or page size mismatch; use matching settings")
        if state.get("tier") not in (*TIERS, "complete"):
            raise ValueError("Checkpoint contains an unknown tier")
    else:
        state = {
            "version": STATE_VERSION,
            "limit": limit,
            "tier": TIERS[0],
            "cursor": None,
            "pages_completed": {tier: 0 for tier in TIERS},
            "scanned": 0,
            "written": 0,
            "rejected": 0,
            "duplicates": 0,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "historical_cutoff": client.get_historical_cutoff(),
        }
        _atomic_json(state_file, state)

    seen = _load_seen_tickers(output)
    series_records = client.get_series()
    categories = series_category_map(series_records)
    pages_this_run = {tier: 0 for tier in TIERS}
    started = datetime.now(timezone.utc)

    while state["tier"] != "complete":
        tier = state["tier"]
        if pages_this_run[tier] >= max_pages_per_tier:
            break
        # The live market catalog defaults toward active listings. Explicitly
        # request settled markets so the resolved cohort is not drowned out by
        # currently open contracts. The historical tier contains archived
        # settled markets and does not accept a status filter.
        page, next_cursor = client.get_market_page(
            tier,
            cursor=state["cursor"],
            limit=limit,
            status="settled" if tier == "markets" else None,
            mve_filter="exclude",
        )
        # Each page gets its own capture time. The checkpoint captured_at
        # remains the discovery session start, not a timestamp for later pages.
        timestamp = datetime.now(timezone.utc)
        records = []
        page_rejected = 0
        page_duplicates = 0
        for raw_market in page:
            state["scanned"] += 1
            # Market-list payloads can omit series_ticker. Resolve it from the
            # authoritative event-ticker prefix before applying the category map.
            market = enrich_market_category(
                enrich_market_series(raw_market, series_records), categories
            )
            ticker = market.get("ticker")
            if isinstance(ticker, str) and ticker:
                if ticker in seen:
                    page_duplicates += 1
                    continue
            try:
                record = snapshot_record(market, timestamp)
            except (TypeError, ValueError):
                page_rejected += 1
                continue
            if isinstance(ticker, str) and ticker:
                seen.add(ticker)
            records.append(record)

        # Do not move the cursor until the entire page append has returned.
        written = append_jsonl(records, output)
        state["written"] += written
        state["rejected"] += page_rejected
        state["duplicates"] += page_duplicates
        state["pages_completed"][tier] += 1
        pages_this_run[tier] += 1
        if next_cursor is None:
            if tier == TIERS[0]:
                state["tier"] = TIERS[1]
                state["cursor"] = None
            else:
                state["tier"] = "complete"
                state["cursor"] = None
        else:
            state["cursor"] = next_cursor
        _atomic_json(state_file, state)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(
            f"Catalog checkpoint: tier={tier}, page={state['pages_completed'][tier]}, "
            f"page_markets={len(page)}, total_scanned={state['scanned']}, "
            f"total_written={state['written']}, duplicates={state['duplicates']}, "
            f"elapsed_seconds={int(elapsed)}",
            flush=True,
        )

    result = dict(state)
    result["complete"] = state["tier"] == "complete"
    result["pages_this_run"] = pages_this_run
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable read-only Kalshi catalog discovery")
    parser.add_argument("--output", default="data/market_universe_snapshots.jsonl")
    parser.add_argument("--state-file", default="data/market_universe_state.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages-per-tier", type=int, default=10)
    args = parser.parse_args(argv)
    try:
        report = discover_resumable(
            KalshiPublicClient(), args.output, args.state_file,
            limit=args.limit, max_pages_per_tier=args.max_pages_per_tier,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Discovery failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
