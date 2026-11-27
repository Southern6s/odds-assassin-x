from datetime import datetime, timezone
import unittest

from engine.research_panel import build_dataset_panel_from_records


def _trade(trade_id: str, created: str) -> dict:
    return {
        "trade": {
            "trade_id": trade_id,
            "ticker": "TEST-MARKET",
            "created_time": created,
            "count": "3",
            "yes_price": "0.40",
            "no_price": "0.60",
            "taker_outcome_side": "yes",
            "taker_book_side": "bid",
            "is_block_trade": False,
        }
    }


def _candle(end_ts: int, bid: float, ask: float) -> dict:
    return {
        "candle": {
            "market_ticker": "TEST-MARKET",
            "end_period_ts": end_ts,
            "yes_bid": {"close": bid},
            "yes_ask": {"close": ask},
        },
        # Deliberately later than the candle timestamp: quote time must come
        # from end_period_ts, not the time the historical file was downloaded.
        "captured_at": "2025-01-01T05:00:00+00:00",
    }


class ResearchPanelTests(unittest.TestCase):
    def test_panel_uses_only_completed_features_quotes_and_future_outcomes(self):
        market_snapshots = [{
            "captured_at": "2025-01-01T04:00:00+00:00",
            "market": {
                "ticker": "TEST-MARKET",
                "event_ticker": "TEST-EVENT",
                "result": "yes",
                "settlement_ts": "2025-01-01T03:00:00+00:00",
            },
        }]
        trades = [
            _trade("t1", "2025-01-01T00:15:00+00:00"),
            _trade("t2", "2025-01-01T01:15:00+00:00"),
        ]
        candles = [
            _candle(1735693140, 0.35, 0.45),  # 00:59 UTC
            _candle(1735696740, 0.45, 0.55),  # 01:59 UTC
        ]

        rows = build_dataset_panel_from_records(
            market_snapshots,
            trades,
            forecast_start=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            forecast_end=datetime(2025, 1, 1, 3, 0, tzinfo=timezone.utc),
            step_seconds=3600,
            candlestick_snapshots=candles,
        )

        self.assertEqual([row["forecast_at"] for row in rows], [
            "2025-01-01T01:00:00+00:00",
            "2025-01-01T02:00:00+00:00",
        ])
        self.assertEqual([row["feature_trade_count"] for row in rows], [1, 1])
        for actual, expected in zip(
            [row["market_yes_mid"] for row in rows], [0.40, 0.50]
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertTrue(all(row["probability_label_yes"] == 1 for row in rows))
        self.assertTrue(all(
            row["resolved_at"] == "2025-01-01T03:00:00+00:00" for row in rows
        ))

    def test_panel_rejects_future_quotes_and_cutoff_after_settlement(self):
        markets = [{
            "captured_at": "2025-01-01T04:00:00+00:00",
            "market": {
                "ticker": "TEST-MARKET",
                "event_ticker": "TEST-EVENT",
                "result": "no",
                "settlement_ts": "2025-01-01T02:00:00+00:00",
            },
        }]
        rows = build_dataset_panel_from_records(
            markets,
            [_trade("t1", "2025-01-01T00:15:00+00:00")],
            forecast_start=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            forecast_end=datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc),
            step_seconds=3600,
            candlestick_snapshots=[_candle(1735693260, 0.2, 0.3)],
        )

        # The 01:00 row has no fresh quote but remains a valid outcome observation;
        # the 02:00 row is excluded because settlement is not strictly in the future.
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["forecast_at"], "2025-01-01T01:00:00+00:00")
        self.assertNotIn("market_yes_mid", rows[0])

    def test_panel_adds_market_specific_candle_features_without_future_data(self):
        market_snapshots = [{
            "captured_at": "2025-01-01T00:59:00+00:00",
            "market": {
                "ticker": "TEST-MARKET",
                "event_ticker": "TEST-EVENT",
                "result": "yes",
                "settlement_ts": "2025-01-01T02:00:00+00:00",
                "yes_bid": 0.5,
                "yes_ask": 0.6,
            },
        }]
        base = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
        candles = [
            _candle(base + minute * 60, 0.30 + minute * 0.01,
                    0.32 + minute * 0.01)
            for minute in range(55, 61)
        ]
        rows = build_dataset_panel_from_records(
            market_snapshots,
            [_trade("candle-test-trade", "2025-01-01T00:30:00+00:00")],
            forecast_start=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            forecast_end=datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc),
            candlestick_snapshots=candles,
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("feature_market_momentum_logit_5", rows[0])
        self.assertGreater(rows[0]["feature_market_momentum_logit_5"], 0)
        self.assertEqual(rows[0]["feature_market_candle_count"], 6)

    def test_panel_caps_number_of_cutoffs(self):
        with self.assertRaisesRegex(ValueError, "max_cutoffs"):
            build_dataset_panel_from_records(
                [{
                    "captured_at": "2025-01-01T04:00:00+00:00",
                    "market": {"ticker": "T", "event_ticker": "E"},
                }],
                [],
                forecast_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
                forecast_end=datetime(2025, 1, 2, tzinfo=timezone.utc),
                step_seconds=60,
                max_cutoffs=5,
            )


if __name__ == "__main__":
    unittest.main()
