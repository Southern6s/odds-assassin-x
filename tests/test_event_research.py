import unittest
from datetime import datetime, timezone

from engine.event_research import build_event_trade_features
from engine.trade_data import normalize_trade, trade_snapshot


def raw_trade(trade_id="T1", ticker="MKT-1", created="2026-01-01T00:15:00Z",
              outcome="yes", book="bid", count="2.50", block=False):
    return {
        "trade_id": trade_id, "ticker": ticker, "count_fp": count,
        "yes_price_dollars": "0.42", "no_price_dollars": "0.58",
        "taker_outcome_side": outcome, "taker_book_side": book,
        "created_time": created, "is_block_trade": block,
    }


class EventTradeResearchTests(unittest.TestCase):
    def test_aggregates_event_buckets_with_documented_direction_semantics(self):
        raws = [
            raw_trade("A", "M1", outcome="yes", book="bid", count="2.0"),
            raw_trade("B", "M2", outcome="no", book="ask", count="3.0", block=True),
            raw_trade("C", "M1", outcome="yes", book="bid", count="4.0"),
            raw_trade("D", "M2", outcome="no", book="ask", count="1.0"),
        ]
        trades = [normalize_trade(item) for item in raws]
        rows = build_event_trade_features(
            trades, {"M1": "EV1", "M2": "EV1"},
            as_of=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            bucket_seconds=3600,
        )
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["event_ticker"], "EV1")
        self.assertEqual(row["trade_count"], 4)
        self.assertEqual(row["contract_volume"], 10.0)
        self.assertEqual(row["block_contract_volume"], 3.0)
        self.assertEqual(row["market_count"], 2)
        self.assertEqual(row["yes_bid_contracts"], 6.0)
        self.assertEqual(row["yes_ask_contracts"], 0.0)
        self.assertEqual(row["no_bid_contracts"], 0.0)
        self.assertEqual(row["no_ask_contracts"], 4.0)
        self.assertAlmostEqual(row["yes_vwap"], 0.42)

    def test_as_of_cutoff_unknown_markets_and_duplicate_ids(self):
        known = normalize_trade(raw_trade("A"))
        future = normalize_trade(raw_trade("F", created="2026-01-01T02:00:00Z"))
        unknown = normalize_trade(raw_trade("U", ticker="MISSING"))
        rows = build_event_trade_features(
            [known, known, future, unknown], {"MKT-1": "EV1"},
            as_of=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(rows[0]["trade_count"], 1)
        self.assertEqual(rows[0]["diagnostics"], {
            "duplicates": 1, "unknown_market": 1, "future_trade": 1,
        })

    def test_accepts_snapshot_input_and_rejects_naive_cutoff(self):
        snapshot = trade_snapshot(
            raw_trade(), captured_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
        rows = build_event_trade_features(
            [snapshot], {"MKT-1": "EV1"},
            as_of=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(rows[0]["trade_count"], 1)
        with self.assertRaises(ValueError):
            build_event_trade_features(
                [], {}, as_of=datetime(2026, 1, 1)
            )

    def test_rejects_invalid_bucket_and_trade_values(self):
        with self.assertRaises(ValueError):
            build_event_trade_features(
                [], {}, as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
                bucket_seconds=0,
            )
        bad = normalize_trade(raw_trade())
        bad["count"] = float("nan")
        with self.assertRaises(ValueError):
            build_event_trade_features(
                [bad], {"MKT-1": "EV1"},
                as_of=datetime(2026, 1, 1, 1, tzinfo=timezone.utc),
            )


if __name__ == "__main__":
    unittest.main()
