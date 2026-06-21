# CLAUDE.md — Swing System project brief

> Read this fully before writing code. It encodes non-negotiable discipline.
> Companion files: `swing_engine.py` (working v1 engine), `methodology.md` (full spec).

## What this project is
A systematic, end-of-day (EOD) swing-trading research system for Indian
**large-cap cash equity (delivery), universe = Nifty 50**. Holding period
**1–3 weeks**. Capital ~₹5L. The goal of the current phase is a **rigorously
backtested, trustworthy methodology — not live trading.** We are validating
whether a real edge exists before risking money.

## Honest mandate (do not let me chase fantasy)
- Target = **beat the Nifty 50 (total return) risk-adjusted, through the cycle.**
  Large caps are efficient, so the realistic output of doing this well is more
  like **~12–16% CAGR as a *multi-year average*** (NOT an annual floor), with the
  payoff being LOWER drawdown and effectively unlimited capacity — not smallcap-
  style 20–25%. Beating Nifty 50 risk-adjusted at all is already a real result.
- **Drawdown is the binding constraint:** design for max DD ≤ 15–20%.
- Live always underperforms backtest. We only believe live >20% if the
  out-of-sample backtest shows ~30%+ gross with ≤20% max DD.
- If a backtest prints implausibly high returns, assume a **bug or look-ahead
  leak** and hunt for it. Honest backtests on weak data return ~nothing.

## Architecture: gate → select → time → size → execute
1. **Macro GATE** — market-state (FII/DII flows, India VIX, breadth,
   Nifty/Smallcap trend) → continuous exposure 0–1. Risk-off → sit in cash.
   "Macro" here = market-state, NOT economic data releases.
2. **SELECTION** — company price momentum + relative strength + rising
   delivery % (real-buying proxy). Turnaround names with ugly trailing
   fundamentals are ALLOWED if price/volume show accumulation.
3. **TIMING** — technical trigger (breakout/pullback + volume confirmation)
   for the entry moment.
4. **SIZING** — volatility-based, conviction-scaled, hard-capped. Concentration
   is a TESTED PARAMETER (`max_position_pct`), not a belief. A per-name
   catastrophic-loss cap stays regardless (smallcaps gap through stops).
5. **EXECUTE** — EOD order + ATR stop + time stop (max 15 days).

Fuse by **pipeline logic, NOT a fitted weighted blend.** A black-box weighted
score overfits in-sample and hides which layer broke.

## NON-NEGOTIABLE backtest discipline
1. **Survivorship bias** → point-in-time universe; include delisted/crashed
   names that existed on each historical date. Never use "today's index list".
2. **Look-ahead bias** → only data knowable at decision time (EOD close).
   Fundamentals usable ~45 days after quarter-end (announcement), not period-end.
3. **Cost realism** → charge every trade: STT, exchange, stamp, GST, AND
   slippage/impact scaled to position-size-vs-volume. An edge that dies after
   costs does not exist.
4. **Overfitting** → walk-forward (fit one window, test the NEXT untouched one,
   roll). Keep a final hold-out untouched until the very end. If many parameter
   sets are tried, assume the best is partly luck and haircut it.
- **Ablate every layer** (gate on/off, timing on/off, etc.). A layer stays only
  if it earns its place out-of-sample.
5. **Data cleaning vs cherry-picking (CRITICAL).** You MAY remove data
   *artifacts* — things never real or never tradeable (unadjusted
   splits/bonuses, bad ticks, suspended/halted names, zero-liquidity days). You
   MUST NOT remove real tradeable *losses* because they can be explained in
   hindsight ("freak news, doesn't count"). Bright-line test: *"Could I have
   known to exclude this BEFORE the trade, using only data available then?"*
   Yes -> exclude; hindsight-only -> it stays. Never trim outliers
   asymmetrically (deleting losers, keeping winners). If removing ONE event
   flips the result, that is a fragility warning, not something to delete.
6. **Statistical vs economic significance.** On huge stock-day samples a trivial
   effect still clears t=2. Require signals to be BOTH significant AND
   economically meaningful (e.g. IC >= ~0.02), not just significant.
- **Kill criteria:** abandon a candidate if edge vanishes after costs, works
  only in-sample, rests on <30 trades, degrades out-of-sample, or can't be tied
  to a real mechanism (capacity advantage / forced flows / regime behaviour).

## Universe filters (point-in-time, daily)
Universe = **Nifty 50** (large-cap). Use point-in-time membership (NOT today's
list) to stay survivorship-free; until historical membership is wired, proxy by
the top ~50 names by trailing market cap / traded value each day. Liquidity and
ASM/GSM/circuit/F&O-ban filters rarely bind for Nifty 50 names but stay in as
cheap guards. Catastrophic-gap risk is far lower than in smallcaps but the
per-name loss cap stays. NOTE the tension: 50 names is a SMALL pond — a selective
breakout may yield ~no simultaneous candidates; if so, widen to Nifty 100.

## Metrics we judge by
CAGR, Sharpe, Sortino, MaxDD, **Calmar (CAGR/MaxDD)**, hit rate, avg win/loss,
turnover, exposure %, rolling 12-month return distribution. Not CAGR alone.

## Current state
- `swing_engine.py` v1 EXISTS and is validated on synthetic data: cost model,
  gate→select→time loop, vol sizing + concentration knob, walk-forward, metrics,
  layer ablations. Synthetic results are ~flat after costs — correct (no edge
  baked in). The engine consumes a `panel` dict:
  `{"close": DF, "value": DF, "deliv": DF, "market": Series, "sectors": array}`
  (DataFrames indexed by date, columns = stocks).
- `regression_diagnostics.py` EXISTS: (1) factor attribution — regress strategy
  returns on market/momentum factors to test whether residual ALPHA is
  significant (real skill) or the return is disguised beta; (2) information
  coefficient — rank-correlate a signal vs forward returns to vet predictive
  power BEFORE simulating. Judge by t-stats + economic size, never R^2 alone.
  Every new signal must clear a meaningful IC; every candidate strategy must
  show significant positive alpha after factor attribution, or it is rejected.

## >>> NEXT TASK: build the Fyers + NSE data pipeline <<<
Replace `make_synthetic()` with real data. The pipeline MUST output the **exact
same `panel` dict** so the engine runs unchanged.
- `config.py` — Fyers app_id/secret/access-token. **GITIGNORED. Never hardcode
  credentials in any other file. Never print them.**
- `pipeline/fetch.py` — Fyers historical daily OHLCV, paginated to respect API
  lookback limits, cached to local parquet (`data/`). Fetch incrementally.
- `pipeline/delivery.py` — NSE bhavcopy / delivery files → delivery %.
- `pipeline/universe.py` — point-in-time small/mid-cap universe per the filters
  above (survivorship-free).
- `pipeline/fundamentals.py` — quarterly & annual EPS, ROE, and shareholding /
  MF-FII holding (for CAN SLIM C/A/I). Stamp each with its ANNOUNCEMENT date and
  expose it only from that date forward (look-ahead discipline). Sources:
  screener.in, NSE/BSE shareholding filings.
- `build_panel.py` — assemble cached data into the `panel` dict.
Build ONE piece at a time, show output, then proceed. Validate no look-ahead at
each step.

## Candidate strategy: consolidation breakout  (`consolidation_breakout.py` + `test_signals.py` BUILT & TESTED)
> REAL-DATA RESULT (3y large-cap, 2023-06→2026-06, via `test_signals.py`):
> KILLED. The setup has a statistically significant NEGATIVE edge — 5d fwd
> return after firing = -1.2% vs ~0 baseline, t = -4.0 (10d t = -3.3); backtest
> CAGR -2.7% vs +3.8% B&H; OOS Sharpe -2.05 ≈ IS -1.97 (consistent, not overfit);
> gate/selection ablations don't rescue it. Interpretation: large-cap breakouts
> short-term MEAN-REVERT here — fading the breakout / buying pullbacks is the
> hypothesis worth testing next (test it, don't just flip the sign on faith).
Volatility-contraction / base-breakout (VCP / Darvas family), operationalized:
- BASE: ~20–30 day consolidation, range (high−low)/mean < ~12% (vol contracted)
- SENTIMENT: delivery % over confirm days > base-period delivery %
- TRIGGER: close breaks base high, AND last `confirm_days` each close up on
  volume > ~1.5× base avg — require DELIVERY-backed volume (screens pump-traps,
  which show high volume but low delivery)
- GATE: only when market risk-on
Tested by **event study** (not IC — sparse event signal): forward returns when
the setup fires vs random baseline, AFTER costs, with a t-stat.
KEY TENSION (seen on synthetic): more `confirm_days` → fewer false breakouts BUT
far fewer signals (419→108→23 for confirm 1→2→3) and less power. confirm_days=3
already trips the <30-trades kill rule. Must show each extra confirm day buys
more edge than it costs in sample/power. Tune confirm_days, base_window,
tightness, vol_mult on real data — and beware overfitting these four knobs
(few params, walk-forward, hold-out).

## Candidate strategy: CAN SLIM quality overlay  (`canslim.py` + real fundamentals BUILT & TESTED)
> REAL-DATA RESULT (`test_canslim.py`, monthly factor test on large-cap universe):
> NO EDGE. Top-quintile minus bottom-quintile CAN SLIM score = -1.6%/month,
> t = -1.2 (insignificant, wrong sign). The strict hard screen (C&A&L&M) passes
> only ~0.8 names/month — CAN SLIM is a growth/smallcap method and large caps
> almost never clear 25% qtr EPS + 20% 3y CAGR + top-20% RS at once. Caveats:
> universe still contaminated (ETFs/penny) + a momentum-unfriendly 2024-25 window,
> so this is directional. But CAN SLIM as a Nifty-50 screen looks unusable.
O'Neil's CAN SLIM — a months-to-years GROWTH method, adapted here as a slow
QUALITY OVERLAY on selection (NOT classic O'Neil; C & A can't move in 1-3 wks).
Maps onto our stack: M=gate, L=RS/momentum selection, N+S=consolidation-breakout
timing, C+A+I=the fundamental quality filter (turned back on).
- C: current-qtr EPS YoY ≥ 25% (bonus accelerating)
- A: 3-yr EPS growth ≥ ~20% AND ROE ≥ 17%
- N: price near/above 52-week high   S: up-day volume (NOTE: classic "small
  float" does NOT apply to large caps — drop the float test, keep volume)
- L: relative-strength rank ≥ 80 (top 20% by ~6m perf)
- I: institutional/fund holding rising QoQ    M: gate risk-on
USE: final buy list = CAN SLIM leader AND breakout firing. CAN SLIM = WHO,
breakout = WHEN. Selectivity is per-stock, so a big universe (500-1000 names)
yields enough simultaneous candidates; a small universe yields ~none.
TEST: does the CAN SLIM filter raise the breakout event-study edge & Calmar vs
breakouts ALONE, after costs? Fewer-but-better only wins if "better" is real.
Beware: growth/bull-biased; qualitative "N" (new product/mgmt) not automatable.

## Conventions
- Python 3, numpy/pandas. Keep signals pluggable (functions returning per-stock
  scores). Cache aggressively; never re-fetch what's on disk. Small, reviewable
  commits. Ask before any destructive file/data operation.