import unittest
from datetime import datetime, timezone

from engine.outcome_dataset import build_resolved_market_dataset


class OutcomeDatasetTests(unittest.TestCase):
    def setUp(self):
        self.cutoff = datetime(2026, 1, 1, 3, tzinfo=timezone.utc)
        self.features = [
            {
                "event_ticker": "EV1",
                "bucket_start": "2026-01-01T00:00:00Z",
                "bucket_seconds": 3600,
                "trade_count": 10,
                "contract_volume": 100.0,
                "yes_bid_contracts": 40.0,
            },
            {
                "event_ticker": "EV1",
                "bucket_start": "2026-01-01T01:00:00Z",
                "bucket_seconds": 3600,
                "trade_count": 20,
                "contract_volume": 200.0,
                "yes_bid_contracts": 80.0,
            },
            {
                "event_ticker": "EV1",
                "bucket_start": "2026-01-01T03:00:00Z",
                "bucket_seconds": 3600,
                "trade_count": 999,
                "contract_volume": 999.0,
            },
        ]

    def test_joins_latest_completed_feature_bucket_to_future_yes_no_labels(self):
        markets = [
            {
                "ticker": "MKT-YES", "event_ticker": "EV1",
                "result": "yes", "settlement_ts": "2026-01-01T05:00:00Z",
            },
            {
                "ticker": "MKT-NO", "event_ticker": "EV1",
                "result": "no", "settlement_ts": "2026-01-01T06:00:00Z",
            },
        ]
        rows = build_resolved_market_dataset(
            markets, self.features, forecast_at=self.cutoff
        )
        self.assertEqual([r["market_ticker"] for r in rows], ["MKT-NO", "MKT-YES"])
        self.assertEqual(rows[0]["probability_label_yes"], 0)
        self.assertEqual(rows[1]["probability_label_yes"], 1)
        self.assertEqual(rows[0]["feature_trade_count"], 20)
        self.assertEqual(rows[0]["feature_contract_volume"], 200.0)
        self.assertEqual(rows[0]["feature_bucket_end"], "2026-01-01T02:00:00+00:00")
        self.assertEqual(rows[0]["feature_age_seconds"], 3600.0)

    def test_excludes_unresolved_scalar_settled_before_cutoff_and_stale_features(self):
        markets = [
            {"ticker": "A", "event_ticker": "EV1", "result": "", "settlement_ts": None},
            {"ticker": "B", "event_ticker": "EV1", "result": "scalar",
             "settlement_ts": "2026-01-01T05:00:00Z"},
            {"ticker": "C", "event_ticker": "EV1", "result": "yes",
             "settlement_ts": "2026-01-01T02:00:00Z"},
            {"ticker": "D", "event_ticker": "EV2", "result": "yes",
             "settlement_ts": "2026-01-01T05:00:00Z"},
        ]
        rows = build_resolved_market_dataset(
            markets, self.features, forecast_at=self.cutoff,
            max_feature_age_seconds=0,
        )
        self.assertEqual(rows, [])

    def test_accepts_snapshot_wrapped_market_and_requires_settlement_after_forecast(self):
        markets = [{
            "schema_version": 1,
            "market": {
                "ticker": "MKT-1", "event_ticker": "EV1",
                "result": "yes", "settlement_ts": "2026-01-01T03:00:00Z",
            },
        }]
        rows = build_resolved_market_dataset(
            markets, self.features, forecast_at=self.cutoff
        )
        self.assertEqual(rows, [])

    def test_rejects_naive_forecast_time_and_duplicate_market_ticker(self):
        with self.assertRaises(ValueError):
            build_resolved_market_dataset(
                [], [], forecast_at=datetime(2026, 1, 1)
            )
        market = {
            "ticker": "DUP", "event_ticker": "EV1", "result": "yes",
            "settlement_ts": "2026-01-01T05:00:00Z",
        }
        with self.assertRaises(ValueError):
            build_resolved_market_dataset(
                [market, market], self.features, forecast_at=self.cutoff
            )

    def test_attaches_latest_pre_forecast_market_quote_for_baseline(self):
        market = {
            "ticker": "MKT-YES", "event_ticker": "EV1",
            "result": "yes", "settlement_ts": "2026-01-01T05:00:00Z",
        }
        snapshots = [
            {
                "captured_at": "2026-01-01T01:30:00Z",
                "market": {"ticker": "MKT-YES", "yes_bid": 0.40, "yes_ask": 0.50,
                           "no_bid": 0.50, "no_ask": 0.60},
            },
            {
                "captured_at": "2026-01-01T02:30:00Z",
                "market": {"ticker": "MKT-YES", "yes_bid": 0.45, "yes_ask": 0.55,
                           "no_bid": 0.45, "no_ask": 0.55},
            },
            {
                "captured_at": "2026-01-01T03:01:00Z",
                "market": {"ticker": "MKT-YES", "yes_bid": 0.90, "yes_ask": 0.95},
            },
        ]
        rows = build_resolved_market_dataset(
            [market], self.features, forecast_at=self.cutoff,
            market_snapshots=snapshots, max_quote_age_seconds=3600,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market_yes_bid"], 0.45)
        self.assertEqual(rows[0]["market_yes_ask"], 0.55)
        self.assertEqual(rows[0]["market_yes_mid"], 0.50)
        self.assertEqual(rows[0]["quote_captured_at"], "2026-01-01T02:30:00+00:00")
        self.assertEqual(rows[0]["quote_age_seconds"], 1800.0)

    def test_excludes_stale_and_future_market_quotes(self):
        market = {
            "ticker": "MKT-YES", "event_ticker": "EV1",
            "result": "yes", "settlement_ts": "2026-01-01T05:00:00Z",
        }
        snapshots = [
            {"captured_at": "2025-12-31T00:00:00Z",
             "market": {"ticker": "MKT-YES", "yes_bid": 0.2, "yes_ask": 0.3}},
            {"captured_at": "2026-01-01T03:01:00Z",
             "market": {"ticker": "MKT-YES", "yes_bid": 0.9, "yes_ask": 0.95}},
        ]
        rows = build_resolved_market_dataset(
            [market], self.features, forecast_at=self.cutoff,
            market_snapshots=snapshots, max_quote_age_seconds=300,
        )
        self.assertNotIn("market_yes_mid", rows[0])

    def test_rejects_crossed_decision_time_quotes(self):
        market = {
            "ticker": "MKT-YES", "event_ticker": "EV1",
            "result": "yes", "settlement_ts": "2026-01-01T05:00:00Z",
        }
        snapshots = [{
            "captured_at": "2026-01-01T02:59:00Z",
            "market": {"ticker": "MKT-YES", "yes_bid": 0.6, "yes_ask": 0.5},
        }]
        with self.assertRaises(ValueError):
            build_resolved_market_dataset(
                [market], self.features, forecast_at=self.cutoff,
                market_snapshots=snapshots,
            )

    def test_rejects_invalid_max_age(self):
        with self.assertRaises(ValueError):
            build_resolved_market_dataset(
                [], [], forecast_at=self.cutoff, max_feature_age_seconds=-1
            )


if __name__ == "__main__":
    unittest.main()
