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


## What the reported 70.6% system actually documents

The published operator description is more specific than a generic Avellaneda–Stoikov bot. It describes:
- WebSocket book and fill ingestion with sequence-continuity checks; a sequence gap invalidates the local book and forces a fresh snapshot.
- Market-specific eligibility and quote placement that adapt to spread regime and external depth, after subtracting the bot's own resting orders.
- Quote sizing that responds to realized fills and time of day; multi-level quote ladders and coordinated handling of correlated markets.
- Inventory skew toward flat under per-market caps, event-level stop-losses, and cooldowns that pull quotes when flow appears informed.
- Debounced amend/cancel lifecycle, exact-decimal price/quantity handling, and fill attribution through order lineage.
- Single-writer fencing, market-level quarantine on ambiguous state, incident halts, and position reconstruction from exchange state after restart.

The author's market-selection criteria, signals, and parameter values remain proprietary. These documented operational and control principles are useful research targets; they are not a reproducible specification of the private strategy.

## Additional experiments added from that architecture

1. **Book integrity gate:** reject a quote decision if the book is stale, crossed/invalid, or sequence continuity is unknown. In historical snapshot data, explicitly label this as a limitation because snapshots cannot reproduce WebSocket sequence guarantees.
2. **Own-order accounting:** reserve a separate simulated own-order ledger and subtract own displayed size from external depth before estimating available liquidity. Do not infer queue priority from aggregate snapshots.
3. **Fill-responsive sizing:** compare constant size with conservative sizing based on recent realized fill rate and inventory utilization. Keep time-of-day effects as a separately tested feature, not a hard-coded assumption.
4. **Correlated event exposure:** cap aggregate exposure by event/series and stress the joint outcome where contracts are logically related.
5. **Adverse-flow cooldown:** test quote withdrawal after adverse short-horizon markouts or abnormal trade/book imbalance; cooldown duration must be selected on training folds only.
6. **Order-state invariants in simulation:** model order lineage, idempotent amend/cancel transitions, stale-order expiry, and fail-closed behavior after ambiguous state. This is a simulator-only state machine, not live order-routing.

## Source links

- Private-system operator description and reported results: https://ian-wang.com/projects/kalshi
- Public Kalshi market-maker reference implementation: https://github.com/rodlaf/KalshiMarketMaker
- Avellaneda–Stoikov adverse-selection comparison / synthetic simulator reference: https://github.com/kayasolomon/market-making

The public repositories are independent reference projects. Neither is represented as the source code behind the reported 70.6% system.

## Reference material

- Ian Wang, *Kalshi market-making bot*: published performance summary and architecture; strategy details are proprietary.
- `rodlaf/KalshiMarketMaker`: public reference implementation describing binary-market selection, volume/spread filters, Avellaneda–Stoikov workers, inventory skew, and portfolio caps. It is an educational reference, not evidence that its default parameters are profitable.
- ODDS ASSASSIN X existing `engine/walk_forward.py`, `engine/depth_join.py`, and `engine/paper_execution.py` remain the evaluation and safety foundation.
