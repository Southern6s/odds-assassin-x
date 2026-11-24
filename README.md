# ODDS ASSASSIN X — Free mobile prototype

Read-only Kalshi public market scanner. No account login, private API key, order placement, or real-money trading.

## Deploy free from an iPhone

1. Create a free GitHub account at https://github.com.
2. In Safari, create a new public repository named `odds-assassin-x`.
3. Upload `app.py`, `requirements.txt`, and `render.yaml` from this project.
4. Create a free Render account at https://render.com.
5. In Render choose **New → Web Service**, connect the GitHub repository, and choose the **Free** instance.
6. Render should detect the Python app. Build command: `pip install -r requirements.txt`. Start command: `gunicorn app:app`.
7. Deploy, then open the generated `onrender.com` URL on your iPhone.

Free Render services sleep after 15 minutes without inbound traffic and may take about a minute to wake up. Do not use this prototype for automated or live trading.

## Local development

```bash
pip install -r requirements.txt
python app.py
```

## Offline research dataset (paper-only)

The research engine can build a point-in-time labeled dataset from previously
collected market and trade snapshot JSONL files. It does not call Kalshi, infer
missing outcomes, or place orders.

```bash
python -m engine.research_dataset \
  --markets data/market_snapshots.jsonl \
  --trades data/trade_snapshots.jsonl \
  --forecast-at 2026-01-01T12:00:00Z \
  --output data/research_dataset.jsonl
```

The command:
- uses only trade prints created at or before the forecast cutoff;
- uses only completed event buckets no older than the configured feature age;
- attaches the latest market quote captured at or before the cutoff, subject to
  the quote-age limit;
- labels only binary markets with settlement after the forecast cutoff;
- writes the export atomically, so a failed run does not leave a partial file.

The default feature bucket is 1 hour, maximum feature age is 24 hours, and
maximum quote age is 5 minutes. Override these with
`--bucket-seconds`, `--max-feature-age-seconds`, and
`--max-quote-age-seconds`. This creates research observations, not validated
forecasts. Compare any model against the market-implied baseline using
chronological walk-forward testing before considering paper signals.

SQLite is currently a local-development adapter. Do not rely on a local
SQLite file or Render's ephemeral filesystem for durable production history.
