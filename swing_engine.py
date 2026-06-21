"""
================================================================================
 swing_engine.py  —  v1 EOD swing engine (gate -> select -> time -> size -> exec)
================================================================================
 Systematic end-of-day swing system for Indian small/mid-cap cash equity.
 Holding 1-3 weeks. This module is the BACKBONE: it owns the panel data
 contract, the macro gate, the selection/timing/sizing primitives, a realistic
 Indian-delivery cost model, the backtest loop, the metrics, walk-forward, and
 layer ablations.

 DESIGN DISCIPLINE (see CLAUDE.md):
   - Pipeline logic, NOT a fitted weighted blend. Each layer is separable so we
     can ablate it (gate on/off, timing on/off, ...) and see what earns its keep.
   - Only data knowable at EOD decision time. Signals at day t look at rows
     [.. t-1] (close of t-1 known) and act on the close of t-1 / open of t.
     We are deliberately conservative: a signal computed "at day t" uses bars
     strictly before t, and the trade is filled at the close of t.
   - Costs are charged on every trade (STT, exchange, stamp, GST, slippage).
   - Synthetic data has NO edge baked in, so an honest backtest returns ~flat
     after costs. If it prints rich returns, that is a BUG / look-ahead leak.

 PANEL CONTRACT (the one dict every data source must produce):
   panel = {
     "close":   DataFrame [dates x stocks]  adjusted close
     "value":   DataFrame [dates x stocks]  traded value (Rs) ~ price*volume
     "deliv":   DataFrame [dates x stocks]  delivery fraction in [0,1]
     "market":  Series    [dates]           benchmark (smallcap) index level
     "sectors": np.ndarray [stocks]         integer sector id per column
   }
 DataFrames are indexed by date; columns are integer stock ids 0..n-1.
================================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# =============================================================================
#  SYNTHETIC DATA  —  honest null: random returns, no predictive structure
# =============================================================================
def make_synthetic(n_stocks: int = 40, n_days: int = 800, seed: int = 7) -> dict:
    """Build a panel of random-walk equities with a market + sector factor and
    NO baked-in predictability. Used to validate wiring: an honest backtest on
    this data must come out ~flat after costs.

    Columns are integer stock ids 0..n_stocks-1 (canslim.py indexes funds by id).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_days)

    # ---- market (smallcap benchmark): regime-switching drift + vol ----------
    # Bull/bear regimes so the gate has something real to switch on. The regime
    # drives the MARKET, not individual-stock predictability.
    regime = np.zeros(n_days)
    state, p_switch = 1, 0.015
    for i in range(n_days):
        if rng.random() < p_switch:
            state *= -1
        regime[i] = state
    mkt_mu = np.where(regime > 0, 0.0006, -0.0009)          # daily drift by regime
    mkt_vol = np.where(regime > 0, 0.009, 0.016)            # vol rises in bears
    mkt_ret = rng.normal(mkt_mu, mkt_vol)
    market = pd.Series(100 * np.exp(np.cumsum(mkt_ret)), index=dates, name="market")

    # ---- sectors ------------------------------------------------------------
    n_sectors = max(2, n_stocks // 8)
    sectors = rng.integers(0, n_sectors, n_stocks)
    sec_ret = rng.normal(0, 0.006, (n_days, n_sectors))     # daily sector factor

    # ---- stocks: beta*market + sector + idiosyncratic (all noise) -----------
    betas = rng.uniform(0.7, 1.4, n_stocks)
    idio_vol = rng.uniform(0.012, 0.030, n_stocks)
    R = np.empty((n_days, n_stocks))
    for j in range(n_stocks):
        R[:, j] = (betas[j] * mkt_ret
                   + sec_ret[:, sectors[j]]
                   + rng.normal(0, idio_vol[j], n_days))
    px0 = rng.uniform(40, 600, n_stocks)
    close = pd.DataFrame(px0 * np.exp(np.cumsum(R, axis=0)), index=dates,
                         columns=range(n_stocks))

    # ---- traded value (Rs): base liquidity * price * volume-noise -----------
    base_adv = rng.uniform(5e7, 5e8, n_stocks)              # Rs/day median liquidity
    vol_noise = np.exp(rng.normal(0, 0.4, (n_days, n_stocks)))
    value = pd.DataFrame(base_adv * vol_noise, index=dates, columns=range(n_stocks))

    # ---- delivery fraction: mean-reverting random walk in [0.2, 0.85] -------
    deliv = np.empty((n_days, n_stocks))
    d = rng.uniform(0.3, 0.6, n_stocks)
    for i in range(n_days):
        d = np.clip(d + rng.normal(0, 0.03, n_stocks) - 0.05 * (d - 0.45), 0.15, 0.9)
        deliv[i] = d
    deliv = pd.DataFrame(deliv, index=dates, columns=range(n_stocks))

    return {"close": close, "value": value, "deliv": deliv,
            "market": market, "sectors": sectors}


# =============================================================================
#  COST MODEL  —  Indian cash-equity DELIVERY, charged both legs
# =============================================================================
def round_trip_cost_frac(notional: float, adv: float,
                         impact_k: float = 0.10) -> float:
    """Fraction of notional lost to costs over a full buy+sell round trip.

    Statutory (delivery, approx, 2024):
      STT          0.10% buy + 0.10% sell
      Exchange txn 0.00297% per leg (NSE)
      SEBI         0.0001% per leg
      Stamp        0.015% on buy only
      GST          18% on (exchange txn + brokerage); brokerage ~0 (discount)
    Slippage/impact: square-root market-impact model, scaled to how big the
      trade is vs average daily traded value (adv). Charged per leg.
        impact_frac = impact_k * sqrt(notional / adv)
    """
    stt = 0.0010 + 0.0010
    exch = 0.0000297 * 2
    sebi = 0.000001 * 2
    stamp = 0.00015
    gst = 0.18 * exch
    statutory = stt + exch + sebi + stamp + gst

    participation = max(notional, 1.0) / max(adv, 1.0)
    impact = impact_k * np.sqrt(participation)              # per leg
    slippage = 2 * impact                                  # both legs
    return statutory + slippage


# =============================================================================
#  LAYER 1 — MACRO GATE  (market-state -> continuous exposure 0..1)
# =============================================================================
def gate_exposure(market: pd.Series,
                  fast: int = 20, slow: int = 50,
                  vol_window: int = 20, vol_cap: float = 0.018) -> pd.Series:
    """Continuous market-state exposure in [0,1] from the benchmark series.

    "Macro" = market-state, not economic releases. On synthetic data we only
    have the price series, so we proxy the gate with TREND + inverse-VOL:
      - trend: fast SMA above slow SMA  (risk-on regime)
      - vol  : realised vol below a cap (de-risk in turbulent tape)
    Real version also folds in FII/DII flows, India VIX, market breadth.

    Returns a Series aligned to `market`; values are knowable at each EOD (uses
    only trailing data via rolling()).
    """
    sma_f = market.rolling(fast).mean()
    sma_s = market.rolling(slow).mean()
    ret = market.pct_change()
    rvol = ret.rolling(vol_window).std()

    # trend score: how far fast sits above slow, squashed to 0..1
    trend = ((sma_f - sma_s) / sma_s / 0.02).clip(-1, 1)
    trend = (trend + 1) / 2
    # vol score: 1 when calm, ->0 as realised vol exceeds cap
    volscore = (1 - (rvol / vol_cap)).clip(0, 1)

    expo = (trend * volscore).clip(0, 1)
    return expo.fillna(0.0)


# =============================================================================
#  LAYER 2 — SELECTION  (who: momentum + RS + rising delivery)
# =============================================================================
def selection_scores(panel: dict, t: int,
                     mom_window: int = 126, rs_window: int = 63,
                     deliv_window: int = 20) -> pd.Series:
    """Cross-sectional selection score per stock at day t (uses bars < t only).

    Combines, by RANK (robust, not a fitted blend of raw magnitudes):
      - price momentum   (6m return)
      - relative strength (3m return vs the market)
      - rising delivery % (real-buying proxy: recent deliv vs prior deliv)
    Higher = more attractive. Turnaround names with ugly fundamentals are
    allowed here; the fundamental veto lives elsewhere (CAN SLIM overlay).
    """
    c = panel["close"]
    if t <= mom_window + 5:
        return pd.Series(0.0, index=c.columns)

    mom = c.iloc[t - 1] / c.iloc[t - 1 - mom_window] - 1
    stock_rs = c.iloc[t - 1] / c.iloc[t - 1 - rs_window] - 1
    mkt = panel["market"]
    mkt_rs = mkt.iloc[t - 1] / mkt.iloc[t - 1 - rs_window] - 1
    rs = stock_rs - mkt_rs

    dv = panel["deliv"]
    deliv_now = dv.iloc[t - deliv_window:t].mean()
    deliv_prev = dv.iloc[t - 2 * deliv_window:t - deliv_window].mean()
    deliv_trend = deliv_now - deliv_prev

    score = (mom.rank(pct=True)
             + rs.rank(pct=True)
             + deliv_trend.rank(pct=True)) / 3.0
    return score.fillna(0.0)


# =============================================================================
#  LAYER 3 — TIMING  (when: breakout + volume/delivery confirmation)
# =============================================================================
def timing_trigger(panel: dict, t: int,
                   breakout_window: int = 20, vol_mult: float = 1.3) -> pd.Series:
    """Boolean per-stock entry trigger at day t (uses bars < t only).

    Simple base-breakout: prior close breaks the trailing N-day high AND the
    last bar's traded value exceeds vol_mult x its trailing average. This is the
    engine's built-in timing; consolidation_breakout.py is the richer VCP cousin.
    """
    c = panel["close"]; v = panel["value"]
    if t <= breakout_window + 2:
        return pd.Series(False, index=c.columns)

    prior_high = c.iloc[t - 1 - breakout_window:t - 1].max()
    broke = c.iloc[t - 1] > prior_high
    vavg = v.iloc[t - 1 - breakout_window:t - 1].mean()
    vol_ok = v.iloc[t - 1] > vol_mult * vavg
    return (broke & vol_ok).fillna(False)


# =============================================================================
#  LAYER 4 — SIZING  (vol-targeted, conviction-scaled, hard-capped)
# =============================================================================
def position_sizes(panel: dict, t: int, candidates: list, scores: pd.Series,
                   equity: float, exposure: float,
                   risk_per_trade: float = 0.0075,
                   atr_window: int = 14, atr_mult: float = 2.5,
                   max_position_pct: float = 0.15,
                   catastrophic_pct: float = 0.06) -> dict:
    """Rupee sizing per candidate. Returns {stock: notional}.

    - Volatility target: risk a fixed fraction of equity per trade; distance to
      the ATR stop sets share count  ->  size = risk_budget / (atr_mult*ATR).
    - Conviction scaling: tilt the risk budget by selection score.
    - Hard caps: max_position_pct of equity (TESTED knob) and a per-name
      catastrophic cap (notional s.t. a full gap-through stop loses
      <= catastrophic_pct of equity) that ALWAYS binds (smallcaps gap stops).
    - Exposure from the gate scales the whole book.
    """
    c = panel["close"]
    if not candidates:
        return {}
    px = c.iloc[t - 1]
    # ATR proxy from close-to-close abs moves (no intraday H/L in panel)
    atr = c.diff().abs().iloc[t - atr_window:t].mean()

    sizes = {}
    for s in candidates:
        price = px[s]
        stop_dist = atr_mult * atr[s]
        if not np.isfinite(price) or price <= 0 or not np.isfinite(stop_dist) or stop_dist <= 0:
            continue
        conviction = 0.5 + scores.get(s, 0.5)               # 0.5..1.5x
        risk_budget = equity * risk_per_trade * conviction * exposure
        notional_vol = risk_budget / (stop_dist / price)    # vol target
        notional_cap = equity * max_position_pct * exposure
        notional_cat = equity * catastrophic_pct / 1.0      # full-stop-loss cap
        sizes[s] = max(0.0, min(notional_vol, notional_cap, notional_cat))
    return sizes


# =============================================================================
#  BACKTEST LOOP
# =============================================================================
def simulate(panel: dict, start: int = 260, end: int | None = None,
             use_gate: bool = True, use_timing: bool = True,
             use_selection: bool = True,
             top_k: int = 5, max_hold: int = 15,
             atr_mult: float = 2.5, max_position_pct: float = 0.15,
             init_equity: float = 500_000.0, impact_k: float = 0.10,
             signal_fn=None) -> dict:
    """Run the EOD pipeline day by day and return an equity curve + trade log.

    Pipeline per day t (decided on bars < t, filled at close of t):
        gate -> (selection) -> (timing) -> sizing -> execute
    Open positions exit on: ATR stop hit, max_hold time stop, or gate collapse.

    Ablation switches: use_gate / use_timing / use_selection flip layers off so
    each layer must earn its keep out-of-sample.

    signal_fn(panel, t) -> boolean Series may REPLACE the built-in timing
    trigger (e.g. consolidation_breakout.fires or a CAN SLIM buy mask).
    """
    c = panel["close"]; v = panel["value"]
    dates = c.index
    end = len(c) - 1 if end is None else end
    gate = gate_exposure(panel["market"]) if use_gate else pd.Series(1.0, index=dates)

    equity = init_equity
    cash = init_equity
    positions = {}          # stock -> dict(shares, entry_px, entry_t, stop, adv)
    eq_curve = []
    trades = []

    for t in range(start, end + 1):
        px_now = c.iloc[t]
        exposure = float(gate.iloc[t]) if use_gate else 1.0

        # ---- mark-to-market & manage exits (using close of t) --------------
        for s in list(positions.keys()):
            pos = positions[s]
            price = px_now[s]
            held = t - pos["entry_t"]
            exit_reason = None
            if not np.isfinite(price):
                exit_reason = "delisted"
            elif price <= pos["stop"]:
                exit_reason = "stop"
            elif held >= max_hold:
                exit_reason = "time"
            elif use_gate and exposure < 0.2:
                exit_reason = "gate"
            if exit_reason:
                gross = pos["shares"] * price
                cost = gross * round_trip_cost_frac(pos["entry_notional"], pos["adv"], impact_k)
                cash += gross - cost
                pnl = gross - cost - pos["entry_notional"]
                trades.append({"stock": s, "entry_t": pos["entry_t"], "exit_t": t,
                               "held": held, "ret": pnl / pos["entry_notional"],
                               "pnl": pnl, "reason": exit_reason})
                del positions[s]
            else:
                # trail the stop up (chandelier-style), never down
                atr = c.diff().abs().iloc[t - 14:t].mean()[s]
                new_stop = price - atr_mult * atr
                pos["stop"] = max(pos["stop"], new_stop) if np.isfinite(new_stop) else pos["stop"]

        # ---- mark equity ---------------------------------------------------
        mtm = sum(p["shares"] * px_now[s] for s, p in positions.items()
                  if np.isfinite(px_now[s]))
        equity = cash + mtm
        eq_curve.append((dates[t], equity, exposure, len(positions)))

        # ---- entries: only if the gate allows risk ------------------------
        if use_gate and exposure < 0.2:
            continue
        slots = top_k - len(positions)
        if slots <= 0:
            continue

        sel = selection_scores(panel, t) if use_selection else pd.Series(1.0, index=c.columns)
        if signal_fn is not None:
            trig = signal_fn(panel, t)
            trig = pd.Series(trig).reindex(c.columns).fillna(False)
        elif use_timing:
            trig = timing_trigger(panel, t)
        else:
            trig = pd.Series(True, index=c.columns)

        eligible = [s for s in c.columns
                    if bool(trig.get(s, False)) and s not in positions
                    and np.isfinite(px_now[s]) and px_now[s] > 0
                    and np.isfinite(v.iloc[t][s])]
        eligible.sort(key=lambda s: sel.get(s, 0.0), reverse=True)
        picks = eligible[:slots]

        sizes = position_sizes(panel, t, picks, sel, equity, exposure,
                               atr_mult=atr_mult, max_position_pct=max_position_pct)
        for s in picks:
            notional = sizes.get(s, 0.0)
            price = px_now[s]
            if notional < 1000 or price <= 0:
                continue
            shares = notional / price
            if shares <= 0:
                continue
            entry_cost = notional * round_trip_cost_frac(notional, v.iloc[t][s], impact_k) / 2
            cash -= notional + entry_cost
            atr = c.diff().abs().iloc[t - 14:t].mean()[s]
            positions[s] = {"shares": shares, "entry_px": price, "entry_t": t,
                            "stop": price - atr_mult * atr,
                            "entry_notional": notional, "adv": v.iloc[t][s]}

    curve = pd.DataFrame(eq_curve, columns=["date", "equity", "exposure", "n_pos"]).set_index("date")
    return {"curve": curve, "trades": pd.DataFrame(trades), "final_equity": equity}


# =============================================================================
#  METRICS
# =============================================================================
def metrics(curve: pd.DataFrame, trades: pd.DataFrame, init_equity: float = 500_000.0,
            periods_per_year: int = 252) -> dict:
    eq = curve["equity"]
    ret = eq.pct_change().dropna()
    years = len(eq) / periods_per_year
    cagr = (eq.iloc[-1] / init_equity) ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan

    ann = np.sqrt(periods_per_year)
    sharpe = ret.mean() / ret.std() * ann if ret.std() > 0 else np.nan
    downside = ret[ret < 0].std()
    sortino = ret.mean() / downside * ann if downside and downside > 0 else np.nan

    roll_max = eq.cummax()
    dd = eq / roll_max - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd and maxdd < 0 and not np.isnan(cagr) else np.nan

    if len(trades):
        wins = trades[trades["ret"] > 0]["ret"]
        losses = trades[trades["ret"] <= 0]["ret"]
        hit = len(wins) / len(trades)
        avg_win = wins.mean() if len(wins) else 0.0
        avg_loss = losses.mean() if len(losses) else 0.0
        n_trades = len(trades)
    else:
        hit = avg_win = avg_loss = np.nan
        n_trades = 0

    # rolling 12m return distribution
    roll_12m = eq.pct_change(periods_per_year).dropna()
    exposure_avg = curve["exposure"].mean()
    # turnover proxy: trades per year
    turnover = n_trades / years if years > 0 else np.nan

    return {"CAGR": cagr, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd,
            "Calmar": calmar, "HitRate": hit, "AvgWin": avg_win, "AvgLoss": avg_loss,
            "NTrades": n_trades, "Turnover/yr": turnover, "AvgExposure": exposure_avg,
            "Roll12m_med": roll_12m.median() if len(roll_12m) else np.nan,
            "Roll12m_min": roll_12m.min() if len(roll_12m) else np.nan}


def print_metrics(m: dict, title: str = "") -> None:
    if title:
        print(f"\n  {title}")
    def f(x, pct=True):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "   n/a"
        return f"{x:6.1%}" if pct else f"{x:6.2f}"
    print(f"    CAGR {f(m['CAGR'])} | Sharpe {f(m['Sharpe'],0)} | Sortino {f(m['Sortino'],0)} "
          f"| MaxDD {f(m['MaxDD'])} | Calmar {f(m['Calmar'],0)}")
    print(f"    Hit {f(m['HitRate'])} | AvgWin {f(m['AvgWin'])} | AvgLoss {f(m['AvgLoss'])} "
          f"| Trades {m['NTrades']} | Exp {f(m['AvgExposure'])}")


# =============================================================================
#  WALK-FORWARD  (fit one window, test the NEXT untouched one, roll)
# =============================================================================
def walk_forward(panel: dict, train: int = 252, test: int = 126,
                 param_grid: list | None = None, **sim_kwargs) -> pd.DataFrame:
    """Roll train/test windows forward. For each fold, pick the best params on
    TRAIN by Calmar, then score them on the untouched TEST window. Reports
    in-sample vs out-of-sample so we can see degradation (the overfit tell).
    """
    if param_grid is None:
        param_grid = [{"atr_mult": a, "max_position_pct": p}
                      for a in (2.0, 2.5, 3.0) for p in (0.10, 0.15, 0.20)]
    n = len(panel["close"])
    rows = []
    fold = 0
    start = 260
    while start + train + test <= n:
        tr0, tr1 = start, start + train
        te0, te1 = tr1, tr1 + test
        # fit on train
        best, best_calmar = None, -np.inf
        for params in param_grid:
            r = simulate(panel, start=tr0, end=tr1, **{**sim_kwargs, **params})
            m = metrics(r["curve"], r["trades"], sim_kwargs.get("init_equity", 500_000.0))
            calmar = m["Calmar"] if not np.isnan(m["Calmar"]) else -1
            if calmar > best_calmar:
                best_calmar, best = calmar, params
        # test on next untouched window with the chosen params
        r_is = simulate(panel, start=tr0, end=tr1, **{**sim_kwargs, **best})
        m_is = metrics(r_is["curve"], r_is["trades"])
        r_oos = simulate(panel, start=te0, end=te1, **{**sim_kwargs, **best})
        m_oos = metrics(r_oos["curve"], r_oos["trades"])
        rows.append({"fold": fold, **{f"is_{k}": v for k, v in
                     {"CAGR": m_is["CAGR"], "Sharpe": m_is["Sharpe"], "Calmar": m_is["Calmar"]}.items()},
                     **{f"oos_{k}": v for k, v in
                     {"CAGR": m_oos["CAGR"], "Sharpe": m_oos["Sharpe"], "Calmar": m_oos["Calmar"]}.items()},
                     "params": best})
        fold += 1
        start += test
    return pd.DataFrame(rows)


# =============================================================================
#  LAYER ABLATIONS  (does each layer earn its place?)
# =============================================================================
def ablate(panel: dict, **sim_kwargs) -> pd.DataFrame:
    configs = {
        "full (gate+sel+time)": dict(use_gate=True, use_selection=True, use_timing=True),
        "no gate":              dict(use_gate=False, use_selection=True, use_timing=True),
        "no timing":            dict(use_gate=True, use_selection=True, use_timing=False),
        "no selection":         dict(use_gate=True, use_selection=False, use_timing=True),
        "gate only":            dict(use_gate=True, use_selection=False, use_timing=False),
    }
    rows = []
    for name, cfg in configs.items():
        r = simulate(panel, **{**sim_kwargs, **cfg})
        m = metrics(r["curve"], r["trades"])
        rows.append({"config": name, "CAGR": m["CAGR"], "Sharpe": m["Sharpe"],
                     "MaxDD": m["MaxDD"], "Calmar": m["Calmar"], "Trades": m["NTrades"]})
    return pd.DataFrame(rows)


# =============================================================================
#  DEMO
# =============================================================================
if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 68)
    print("  swing_engine v1  —  synthetic validation (expect ~flat after costs)")
    print("=" * 68)

    panel = make_synthetic()
    print(f"\n  Panel: {panel['close'].shape[0]} days x {panel['close'].shape[1]} stocks, "
          f"{len(np.unique(panel['sectors']))} sectors")

    r = simulate(panel)
    m = metrics(r["curve"], r["trades"])
    print_metrics(m, "FULL PIPELINE (synthetic, no edge baked in):")
    print("\n  -> On honest random data this should be ~flat/slightly-negative after")
    print("     costs. Rich returns here would mean a look-ahead leak. Good if dull.")

    print("\n" + "-" * 68)
    print("  LAYER ABLATIONS (each layer must earn its keep out-of-sample):")
    print("-" * 68)
    print(ablate(panel).to_string(index=False, float_format=lambda x: f"{x:6.2%}" if abs(x) < 5 else f"{x:6.2f}"))

    print("\n" + "-" * 68)
    print("  WALK-FORWARD (in-sample vs out-of-sample; watch for degradation):")
    print("-" * 68)
    wf = walk_forward(panel)
    if len(wf):
        show = wf[["fold", "is_CAGR", "oos_CAGR", "is_Sharpe", "oos_Sharpe", "is_Calmar", "oos_Calmar"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
        print(f"\n  Mean IS Sharpe {wf['is_Sharpe'].mean():.2f} vs OOS Sharpe {wf['oos_Sharpe'].mean():.2f}"
              "  (OOS << IS is the overfit tell).")
