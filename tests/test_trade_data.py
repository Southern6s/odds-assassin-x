import unittest
from datetime import datetime, timezone
from unittest.mock import Mock
import json

from engine.trade_data import normalize_trade, trade_snapshot
from engine.kalshi_client import KalshiAPIError, KalshiPublicClient


def raw_trade(trade_id="T1", ticker="MKT-1"):
    return {
        "trade_id": trade_id, "ticker": ticker, "count_fp": "2.50",
        "yes_price_dollars": "0.42", "no_price_dollars": "0.58",
        "taker_outcome_side": "yes", "taker_book_side": "bid",
        "created_time": "2026-01-01T00:00:00Z", "is_block_trade": False,
    }


class TradeDataTests(unittest.TestCase):
    def test_normalizes_fixed_point_trade_and_direction(self):
        trade = normalize_trade(raw_trade())
        self.assertEqual(trade["count"], 2.5)
        self.assertEqual(trade["yes_price"], 0.42)
        self.assertEqual(trade["taker_outcome_side"], "yes")
        self.assertEqual(trade["created_time"], "2026-01-01T00:00:00+00:00")

    def test_rejects_invalid_price_and_inconsistent_direction_fields(self):
        bad = raw_trade()
        bad["yes_price_dollars"] = "1.2"
        with self.assertRaises(ValueError):
            normalize_trade(bad)
        bad = raw_trade()
        bad["taker_book_side"] = "ask"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            normalize_trade(bad)
        bad = raw_trade()
        bad["taker_book_side"] = "sideways"
        with self.assertRaises(ValueError):
            normalize_trade(bad)

    def test_snapshot_retains_raw_and_requires_aware_capture_time(self):
        when = datetime(2026, 1, 1, tzinfo=timezone.utc)
        snap = trade_snapshot(raw_trade(), captured_at=when)
        self.assertEqual(snap["raw"]["trade_id"], "T1")
        self.assertEqual(snap["captured_at"], when.isoformat())
        with self.assertRaises(ValueError):
            trade_snapshot(raw_trade(), captured_at=datetime(2026, 1, 1))


class FakeResponse:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self): return self.body


class TradeClientTests(unittest.TestCase):
    def test_live_and_historical_trades_paginate_and_deduplicate(self):
        opener = Mock(side_effect=[
            FakeResponse({"trades": [{"trade_id": "A"}], "cursor": "next"}),
            FakeResponse({"trades": [{"trade_id": "B"}], "cursor": ""}),
            FakeResponse({"trades": [{"trade_id": "A"}, {"trade_id": "C"}], "cursor": ""}),
        ])
        client = KalshiPublicClient(opener=opener)
        rows = list(client.iter_all_trade_tiers(ticker="M1", min_ts=100, max_ts=200))
        self.assertEqual([r["trade_id"] for r in rows], ["A", "B", "C"])
        self.assertIn("/markets/trades?", opener.call_args_list[0].args[0].full_url)
        self.assertIn("/historical/trades?", opener.call_args_list[2].args[0].full_url)
        self.assertIn("min_ts=100", opener.call_args_list[0].args[0].full_url)

    def test_trade_pagination_rejects_repeated_cursor(self):
        opener = Mock(side_effect=[
            FakeResponse({"trades": [], "cursor": "same"}),
            FakeResponse({"trades": [], "cursor": "same"}),
        ])
        with self.assertRaises(KalshiAPIError):
            list(KalshiPublicClient(opener=opener).iter_trades())


if __name__ == "__main__":
    unittest.main()
