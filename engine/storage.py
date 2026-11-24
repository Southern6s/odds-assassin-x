"""Transactional SQLite storage adapter for immutable market history.

SQLite is a local-development adapter, not a recommendation to keep production
history on an ephemeral Render filesystem. The API is intentionally small so a
managed PostgreSQL adapter can implement the same write contract later.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable


@dataclass(frozen=True)
class StoreWriteReport:
    inserted: int
    duplicates: int


class SQLiteMarketStore:
    """Append-only local store with uniqueness guards and atomic batch writes."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS market_snapshots (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (ticker, captured_at)
                );
                CREATE INDEX IF NOT EXISTS idx_market_snapshots_time
                    ON market_snapshots(captured_at);
                CREATE INDEX IF NOT EXISTS idx_market_snapshots_ticker_time
                    ON market_snapshots(ticker, captured_at);

                CREATE TABLE IF NOT EXISTS candlestick_snapshots (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    period_interval INTEGER NOT NULL
                        CHECK (period_interval IN (1, 60, 1440)),
                    end_period_ts INTEGER NOT NULL,
                    captured_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (ticker, period_interval, end_period_ts, captured_at)
                );
                CREATE INDEX IF NOT EXISTS idx_candles_ticker_interval_time
                    ON candlestick_snapshots(ticker, period_interval, end_period_ts);
                CREATE INDEX IF NOT EXISTS idx_candles_capture_time
                    ON candlestick_snapshots(captured_at);
                """
            )

    @staticmethod
    def _json(record: dict[str, Any]) -> str:
        return json.dumps(record, separators=(",", ":"), sort_keys=True, allow_nan=False)

    def write_market_snapshots(
        self, records: Iterable[dict[str, Any]]
    ) -> StoreWriteReport:
        inserted = duplicates = 0
        with self._connect() as connection:
            for record in records:
                try:
                    market = record["market"]
                    ticker = market["ticker"]
                    captured_at = record["captured_at"]
                    version = record["schema_version"]
                    if not isinstance(ticker, str) or not ticker or not isinstance(captured_at, str):
                        raise ValueError("invalid market snapshot identity")
                    cursor = connection.execute(
                        """INSERT OR IGNORE INTO market_snapshots
                           (ticker, captured_at, schema_version, payload_json)
                           VALUES (?, ?, ?, ?)""",
                        (ticker, captured_at, version, self._json(record)),
                    )
                    if cursor.rowcount:
                        inserted += 1
                    else:
                        duplicates += 1
                except (KeyError, TypeError) as exc:
                    raise ValueError("malformed market snapshot record") from exc
        return StoreWriteReport(inserted, duplicates)

    def write_candlestick_snapshots(
        self, records: Iterable[dict[str, Any]]
    ) -> StoreWriteReport:
        inserted = duplicates = 0
        with self._connect() as connection:
            for record in records:
                try:
                    candle = record["candle"]
                    ticker = candle["market_ticker"]
                    interval = candle["period_interval"]
                    end_ts = candle["end_period_ts"]
                    captured_at = record["captured_at"]
                    version = record["schema_version"]
                    if not isinstance(ticker, str) or not ticker or not isinstance(captured_at, str):
                        raise ValueError("invalid candlestick snapshot identity")
                    cursor = connection.execute(
                        """INSERT OR IGNORE INTO candlestick_snapshots
                           (ticker, period_interval, end_period_ts, captured_at,
                            schema_version, payload_json)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (ticker, interval, end_ts, captured_at, version, self._json(record)),
                    )
                    if cursor.rowcount:
                        inserted += 1
                    else:
                        duplicates += 1
                except (KeyError, TypeError) as exc:
                    raise ValueError("malformed candlestick snapshot record") from exc
        return StoreWriteReport(inserted, duplicates)

    def market_snapshot_count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM market_snapshots").fetchone()[0]

    def candlestick_snapshot_count(self) -> int:
        with self._connect() as connection:
            return connection.execute("SELECT COUNT(*) FROM candlestick_snapshots").fetchone()[0]

    def latest_market_snapshots(self, limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be between 1 and 10000")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT payload_json FROM market_snapshots
                   ORDER BY captured_at DESC, ticker ASC LIMIT ?""", (limit,)
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def candles_for_market(
        self, ticker: str, period_interval: int, *, limit: int = 10000
    ) -> list[dict[str, Any]]:
        if period_interval not in (1, 60, 1440):
            raise ValueError("period_interval must be 1, 60, or 1440")
        if not 1 <= limit <= 100000:
            raise ValueError("limit must be between 1 and 100000")
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT payload_json FROM candlestick_snapshots
                   WHERE ticker = ? AND period_interval = ?
                   ORDER BY end_period_ts ASC, captured_at ASC LIMIT ?""",
                (ticker, period_interval, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]
