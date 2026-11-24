"""One-shot, read-only market snapshot collection orchestration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .kalshi_data import append_jsonl, snapshot_record


@dataclass
class CollectionReport:
    captured_at: str
    written: int = 0
    rejected: int = 0
    rejection_samples: list[str] = field(default_factory=list)


def collect_market_snapshot(
    markets: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    captured_at: datetime | None = None,
) -> CollectionReport:
    """Append a timestamp-consistent snapshot; malformed rows are counted, not guessed."""
    timestamp = captured_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("captured_at must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    report = CollectionReport(captured_at=timestamp.isoformat())

    def valid_records():
        for market in markets:
            try:
                record = snapshot_record(market, timestamp)
            except (TypeError, ValueError) as exc:
                report.rejected += 1
                if len(report.rejection_samples) < 20:
                    ticker = market.get("ticker", "<missing>") if isinstance(market, dict) else "<invalid-record>"
                    report.rejection_samples.append(f"{ticker}: {exc}")
                continue
            yield record

    report.written = append_jsonl(valid_records(), output_path)
    return report
