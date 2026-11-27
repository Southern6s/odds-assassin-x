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
STATE_VERSION = 4


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


def discover_resumable(
    client: KalshiPublicClient,
    output_path: str | Path,
    state_path: str | Path,
    *,
    limit: int = 1000,
    max_pages_per_tier: int = 250,
    min_ts: int | None = None,
    max_ts: int | None = None,
) -> dict[str, Any]:
    """Append complete pages, then atomically checkpoint their next cursor.

    If a process stops after appending but before checkpointing, existing ticker
    keys make replay of that page idempotent on the next invocation.
    """
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if max_pages_per_tier < 1:
        raise ValueError("max_pages_per_tier must be positive")
    if (min_ts is None) != (max_ts is None):
        raise ValueError("min_ts and max_ts must be supplied together")
    if min_ts is not None and (isinstance(min_ts, bool) or isinstance(max_ts, bool)
            or not isinstance(min_ts, int) or not isinstance(max_ts, int)
            or min_ts < 0 or max_ts <= min_ts):
        raise ValueError("invalid settlement window")
    window = {"min_ts": min_ts, "max_ts": max_ts} if min_ts is not None else None
    output = Path(output_path)
    state_file = Path(state_path)
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        if (state.get("version") != STATE_VERSION or state.get("limit") != limit
                or state.get("window") != window):
            raise ValueError("Checkpoint version, page size, or settlement window mismatch; use matching settings")
        if (set(state.get("cursors", {})) != set(TIERS)
                or set(state.get("tier_complete", {})) != set(TIERS)
                or set(state.get("pages_completed", {})) != set(TIERS)):
            raise ValueError("Checkpoint contains invalid per-tier pagination state")
    else:
        state = {
            "version": STATE_VERSION,
            "limit": limit,
            "window": window,
            "cursors": {tier: None for tier in TIERS},
            "tier_complete": {tier: False for tier in TIERS},
            "pages_completed": {tier: 0 for tier in TIERS},
            "scanned": 0,
            "written": 0,
            "rejected": 0,
            "outside_window": 0,
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

    # Round-robin so each catalog tier receives its own per-run page budget.
    while any(
        not state["tier_complete"][tier]
        and pages_this_run[tier] < max_pages_per_tier
        for tier in TIERS
    ):
        for tier in TIERS:
            if state["tier_complete"][tier] or pages_this_run[tier] >= max_pages_per_tier:
                continue
        # The live market catalog defaults toward active listings. Explicitly
        # request settled markets so the resolved cohort is not drowned out by
        # currently open contracts. The historical tier contains archived
        # settled markets and does not accept a status filter.
        page, next_cursor = client.get_market_page(
            tier,
            cursor=state["cursors"][tier],
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
        page_outside_window = 0
        for raw_market in page:
            state["scanned"] += 1
            # Market-list payloads can omit series_ticker. Resolve it from the
            # authoritative event-ticker prefix before applying the category map.
            market = enrich_market_category(
                enrich_market_series(raw_market, series_records), categories
            )
            ticker = market.get("ticker")
            if window is not None:
                settled_ts = _unix_seconds(market.get("settlement_ts"))
                if settled_ts is None or not min_ts <= settled_ts < max_ts:
                    page_outside_window += 1
                    continue
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
        state["outside_window"] += page_outside_window
        state["duplicates"] += page_duplicates
        state["pages_completed"][tier] += 1
        pages_this_run[tier] += 1
        if next_cursor is None:
            state["tier_complete"][tier] = True
            state["cursors"][tier] = None
        else:
            state["cursors"][tier] = next_cursor
        _atomic_json(state_file, state)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(
            f"Catalog checkpoint: tier={tier}, page={state['pages_completed'][tier]}, "
            f"page_markets={len(page)}, total_scanned={state['scanned']}, "
            f"total_written={state['written']}, outside_window={state['outside_window']}, "
            f"duplicates={state['duplicates']}, "
            f"elapsed_seconds={int(elapsed)}",
            flush=True,
        )

    result = dict(state)
    result["complete"] = all(state["tier_complete"].values())
    result["pages_this_run"] = pages_this_run
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumable read-only Kalshi catalog discovery")
    parser.add_argument("--output", default="data/market_universe_snapshots.jsonl")
    parser.add_argument("--state-file", default="data/market_universe_state.json")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--max-pages-per-tier", type=int, default=250)
    parser.add_argument("--min-ts", type=int)
    parser.add_argument("--max-ts", type=int)
    args = parser.parse_args(argv)
    try:
        report = discover_resumable(
            KalshiPublicClient(), args.output, args.state_file,
            limit=args.limit, max_pages_per_tier=args.max_pages_per_tier,
            min_ts=args.min_ts, max_ts=args.max_ts,
        )
    except (ValueError, RuntimeError) as exc:
        parser.exit(1, f"Discovery failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
