# Market-Making Hybrid Research Plan

## Objective

Research a paper-only market-making module for ODDS ASSASSIN X, informed by publicly documented Kalshi market-making architectures. Preserve the existing walk-forward probability model as an independent signal and benchmark. Do not assume that a reported win rate transfers across strategies or market regimes.

The reported 70.6% result belongs to a separate, privately implemented Kalshi market-making system. Its author says the core market-selection criteria, signals, and parameter values are proprietary. This project therefore studies published design principles and public reference implementations; it does not copy or claim access to that private source code.

## Candidate design to investigate

1. **Market selection**
   - Restrict the first experiment to simple binary markets.
   - Rank candidates using normalized executable spread, recent traded volume, open interest, and displayed depth.
   - Exclude thin books, stale quotes, wide/unstable spreads, and markets with insufficient two-sided depth.
   - Limit concentration by event and series so correlated contracts do not consume the portfolio.

2. **Inventory-aware quoting**
   - Evaluate an Avellaneda–Stoikov-style reservation price and spread as a research baseline, not as a presumed profitable formula.
   - Skew quotes as inventory grows; reduce or stop adding exposure near per-market and portfolio caps.
   - Use the existing model probability as one fair-value input, alongside the live book midpoint and volatility/flow diagnostics.
   - Compare fixed-spread, inventory-skewed, and model-informed quote variants on identical data.

3. **Execution realism**
   - Model post-only order placement, queue position, partial fills, cancellations, quote expiry, latency, and adverse selection.
   - Never count a displayed quote as a fill merely because price touched it.
   - Charge the applicable maker/taker fees and conservative slippage; report fill assumptions and sensitivity.
   - Measure markouts after fills (for example 1m, 5m, and 15m where data permits).

4. **Risk and operations**
   - Keep `paper_only=true` and `live_trading=false` in every test and workflow.
   - Enforce per-market, per-event, gross portfolio, and daily-loss caps; add a kill switch to the paper strategy state.
   - On market deselection or shutdown, cancel simulated resting orders and verify no stale working orders remain.
   - No credential handling or live order-routing code is part of this research phase.

## Evaluation protocol

- Freeze the current model and execution baselines before adding the new strategy.
- Use chronological walk-forward folds; train only on information available at each decision timestamp.
- Keep a final untouched holdout period and avoid tuning repeatedly against it.
- Compare on identical markets and timestamps:
  1. current probability model taking strategy;
  2. market-midpoint / passive benchmark;
  3. simple fixed-spread maker;
  4. inventory-aware maker;
  5. model-informed inventory-aware maker.
- Report net P&L after fees, win rate, average win/loss, profit factor, Sharpe (with stated frequency), max drawdown, tail loss/CVaR, turnover, fill rate, quote time at touch, inventory exposure, and adverse-selection markouts.
- Break results out by series/category, spread, depth, volatility, time-to-resolution, and market regime.
- Include confidence intervals and minimum sample requirements. A high win rate alone is not an acceptance criterion.

## Promotion gates

1. Unit and deterministic simulation tests pass.
2. Historical replay beats relevant baselines after realistic costs and conservative fill assumptions.
3. Results persist on untouched chronological holdout data and across more than one regime/category.
4. A forward paper-only shadow run confirms fill/markout assumptions.
5. Any failed gate triggers diagnosis or rejection; do not promote by relaxing execution assumptions.

## Immediate dependency

Complete the synchronized depth + research-panel evaluation already in progress. Validate actual timestamp overlap first. If the depth capture does not overlap the forecast decision times, collect a new synchronized sample rather than using future snapshots or loosening the no-lookahead rule.

## Reference material

- Ian Wang, *Kalshi market-making bot*: published performance summary and architecture; strategy details are proprietary.
- `rodlaf/KalshiMarketMaker`: public reference implementation describing binary-market selection, volume/spread filters, Avellaneda–Stoikov workers, inventory skew, and portfolio caps. It is an educational reference, not evidence that its default parameters are profitable.
- ODDS ASSASSIN X existing `engine/walk_forward.py`, `engine/depth_join.py`, and `engine/paper_execution.py` remain the evaluation and safety foundation.
