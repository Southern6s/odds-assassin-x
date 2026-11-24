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

### Score the market-implied baseline

After creating the research dataset, score the last valid pre-cutoff YES midpoint:

```bash
python -m engine.market_baseline --dataset data/research_dataset.jsonl
```

The report includes Brier score, log loss, accuracy, calibration bins, quote
coverage, average YES spread, and Brier skill versus a neutral 50% forecast.
Rows without a two-sided YES midpoint are excluded from scoring and remain in
the coverage denominator. This is a descriptive benchmark, not evidence of a
tradable edge: it does not model fills, fees, slippage, or liquidity.

### Fit a leakage-filtered event-feature model (research only)

The experimental logistic model uses an explicit allowlist of numeric
`feature_*` columns and excludes every training label not settled by the fit
cutoff. Example from Python:

```python
from datetime import datetime, timezone
from engine.logistic_model import fit_logistic_model

model = fit_logistic_model(
    research_rows,
    ["yes_bid_contracts", "no_bid_contracts", "contract_volume"],
    as_of=datetime(2026, 1, 10, tzinfo=timezone.utc),
    min_training_rows=100,
)
probability_yes = model.predict_probability(research_rows[0])
```

This is an experimental, dependency-free L2-regularized logistic regression,
not a validated trading model. Use chronological out-of-sample evaluation and
compare against the market midpoint baseline before interpreting any signal.
The model intentionally does not select features automatically or use market
midpoint as an input feature.

### Collect public trade history (read-only)

Kalshi trade collection is bounded to an explicit Unix-second interval. The
collector pages through both live and historical public trade tiers, normalizes
the documented fixed-point fields, deduplicates trade IDs, and writes append-
only JSONL snapshots:

```bash
python -m engine.trade_collector \
  --min-ts 1767225600 \
  --max-ts 1767312000 \
  --output data/trade_snapshots.jsonl
```

The interval is required to prevent an accidental unbounded backfill. Use
`--ticker` to restrict collection to one market, `--block-trades yes` or
`--block-trades no` to filter block trades, and `--max-pages-per-tier` for a
small test run. Kalshi's documented live endpoint is `GET /markets/trades`;
archived trades use `GET /historical/trades`. The collector is unauthenticated,
read-only, and does not place orders. Local JSONL is research storage, not
durable hosted storage.

### Walk-forward model evaluation (paper-only)

The evaluator expects a combined research JSONL file containing multiple
historical `forecast_at` cutoffs. It fits a fresh logistic model at each test
window using only rows whose outcomes had settled by that window's first
forecast timestamp, then scores later observations against both a neutral
50% forecast and the decision-time market midpoint.

```bash
python -m engine.walk_forward \
  --dataset data/multi_cutoff_research.jsonl \
  --features yes_bid_contracts,no_bid_contracts,contract_volume \
  --min-train-size 100 \
  --test-cutoffs 1
```

Cutoffs are kept intact as groups so observations from the same forecast
timestamp cannot be split between training and testing. The report aggregates
Brier score and log loss across folds. This evaluator does not simulate fills,
fees, slippage, or orders. One-cutoff datasets cannot produce a walk-forward
result; collect or construct multiple historical point-in-time snapshots first.

### Build a multi-cutoff historical dataset

For genuine walk-forward work, build observations at several historical decision
times rather than reusing one cutoff. Supply collected market metadata, public
trade prints, and (ideally) historical candlestick snapshots. Candle close
quotes are timestamped at each candle's end time, not the later collection
time, and are filtered against each forecast cutoff.

```bash
python -m engine.multi_cutoff_dataset \
  --markets data/market_snapshots.jsonl \
  --trades data/trade_snapshots.jsonl \
  --candlesticks data/candlestick_snapshots.jsonl \
  --forecast-times 2026-01-01T12:00:00Z,2026-01-02T12:00:00Z \
  --max-quote-age-seconds 3600 \
  --output data/multi_cutoff_research.jsonl
```

Then evaluate the repeated observations chronologically with
`python -m engine.walk_forward`. This workflow is still research-only; the
quality and representativeness of collected historical data must be checked
before interpreting scores. Do not assume missing candle closes or missing
historical snapshots can be reconstructed.
