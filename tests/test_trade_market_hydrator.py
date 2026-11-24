import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from engine import trade_market_hydrator


class FakeClient:
    def get_series(self):
        return [{"ticker": "EVT", "category": "Sports"}]

    def iter_markets_by_tickers(self, tickers):
        self.tickers = tickers
        yield {"ticker": "MKT-1", "event_ticker": "EVT-1", "result": "yes"}


class TradeMarketHydratorTests(unittest.TestCase):
    def test_hydrates_only_unique_tickers_present_in_trade_file(self):
        with TemporaryDirectory() as folder:
            trades = Path(folder) / "trades.jsonl"
            output = Path(folder) / "markets.jsonl"
            trades.write_text("\n".join([
                json.dumps({"trade": {"ticker": "MKT-1"}}),
                json.dumps({"trade": {"ticker": "MKT-1"}}),
                json.dumps({"trade": {"ticker": "MKT-2"}}),
            ]) + "\n")
            client = FakeClient()
            with patch.object(trade_market_hydrator, "KalshiPublicClient", return_value=client):
                result = trade_market_hydrator.main([
                    "--trades", str(trades), "--output", str(output)
                ])
            rows = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(result, 0)
        self.assertEqual(client.tickers, ["MKT-1", "MKT-2"])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["market"]["event_ticker"], "EVT-1")

    def test_hydrator_adds_series_ticker_from_verified_event_prefix(self):
        with TemporaryDirectory() as folder:
            trades = Path(folder) / "trades.jsonl"
            output = Path(folder) / "markets.jsonl"
            trades.write_text(json.dumps({"trade": {"ticker": "MKT-1"}}) + "\\n")
            client = FakeClient()
            with patch.object(trade_market_hydrator, "KalshiPublicClient", return_value=client):
                result = trade_market_hydrator.main([
                    "--trades", str(trades), "--output", str(output)
                ])
            row = json.loads(output.read_text().splitlines()[0])
        self.assertEqual(result, 0)
        self.assertEqual(row["market"]["series_ticker"], "EVT")
        self.assertEqual(row["market"]["category"], "Sports")


if __name__ == "__main__":
    unittest.main()
