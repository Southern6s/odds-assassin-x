from datetime import datetime, timezone

import pytest

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


def test_panel_uses_only_completed_features_quotes_and_future_outcomes():
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

    assert [row["forecast_at"] for row in rows] == [
        "2025-01-01T01:00:00+00:00",
        "2025-01-01T02:00:00+00:00",
    ]
    assert [row["feature_trade_count"] for row in rows] == [1, 2]
    assert [row["market_yes_mid"] for row in rows] == pytest.approx([0.40, 0.50])
    assert all(row["probability_label_yes"] == 1 for row in rows)
    assert all(row["resolved_at"] == "2025-01-01T03:00:00+00:00" for row in rows)


def test_panel_rejects_future_quotes_and_cutoff_after_settlement():
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
        candlestick_snapshots=[_candle(1735693260, 0.2, 0.3)],  # 01:01, after cutoff
    )

    # The 01:00 row has no fresh quote but remains a valid outcome observation;
    # the 02:00 row is excluded because settlement is not strictly in the future.
    assert len(rows) == 1
    assert rows[0]["forecast_at"] == "2025-01-01T01:00:00+00:00"
    assert "market_yes_mid" not in rows[0]


def test_panel_caps_number_of_cutoffs():
    with pytest.raises(ValueError, match="max_cutoffs"):
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
