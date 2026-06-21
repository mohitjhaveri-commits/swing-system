"""
================================================================================
 CAN SLIM  (O'Neil)  —  quality scorer + integration with breakout timing
================================================================================
 CAN SLIM is a MONTHS-to-YEARS growth method, NOT a swing method (C & A —
 earnings — can't change inside a 1-3 week hold). We use it as a slow-moving
 QUALITY OVERLAY on selection: only swing-trade breakouts in CAN SLIM leaders.

   C  Current quarterly EPS YoY >= 25% (bonus: accelerating)
   A  Annual: 3-yr EPS growth >= ~20% AND ROE >= 17%
   N  New base/52-week high (price near/above 52w high)         [price-derived]
   S  Supply/demand: small float + volume up on up-days         [price-derived]
   L  Leader: relative-strength rank >= 80 (top 20% by 6m perf) [price-derived]
   I  Institutional sponsorship rising (fund holding up QoQ)
   M  Market direction risk-on  -> our GATE                     [price-derived]

 REAL DATA NEEDS (this is where fundamentals come back): quarterly & annual EPS,
 ROE (screener.in), shareholding pattern / MF-FII holding (NSE-BSE filings).
 Use ANNOUNCEMENT dates, never period-end (look-ahead). Synthetic fundamentals
 below only demo the scoring machinery.
================================================================================
"""
import numpy as np
import pandas as pd
from swing_engine import make_synthetic, gate_exposure
from consolidation_breakout import fires


def synth_fundamentals(n_stocks=40, seed=11):
    """Per-stock fundamental snapshot (synthetic). Real version is time-varying
    by announcement date."""
    rng = np.random.default_rng(seed)
    return {
        "eps_q_yoy": rng.normal(0.20, 0.35, n_stocks),     # current qtr EPS YoY
        "eps_q_accel": rng.normal(0.0, 0.15, n_stocks),    # growth vs prior qtr
        "eps_3y_cagr": rng.normal(0.18, 0.20, n_stocks),   # annual growth
        "roe": rng.normal(0.16, 0.10, n_stocks),           # return on equity
        "inst_chg": rng.normal(0.0, 0.02, n_stocks),       # QoQ fund-holding change
        "float_cr": rng.uniform(50, 5000, n_stocks),       # free-float (Cr) - small better
    }


def canslim_score(panel, fund, t):
    """Composite 0-1 CAN SLIM score per stock at day t, plus a pass-mask for the
    hard screens (C, A, L, M). Higher = stronger leader."""
    c = panel["close"]; v = panel["value"]
    n = c.shape[1]

    # --- price-derived components ---
    perf_6m = c.iloc[t-1] / c.iloc[t-126] - 1 if t > 130 else pd.Series(0, index=c.columns)
    L_rank = perf_6m.rank(pct=True)                                   # L: leader RS
    hi_52w = c.iloc[t-252:t].max() if t > 252 else c.iloc[:t].max()
    N_prox = (c.iloc[t-1] / hi_52w).clip(upper=1.0)                   # N: near 52w high
    up = c.iloc[t-20:t].diff() > 0
    S_demand = (v.iloc[t-20:t][up].mean() / (v.iloc[t-20:t].mean() + 1e-9)).fillna(1.0)
    float_small = pd.Series(fund["float_cr"], index=c.columns).rank(pct=True, ascending=False)

    # --- fundamental components (cross-sectional, snapshot) ---
    C_score = pd.Series(np.clip(fund["eps_q_yoy"] / 0.25, 0, 2), index=c.columns) / 2
    A_eps = pd.Series(np.clip(fund["eps_3y_cagr"] / 0.20, 0, 2), index=c.columns) / 2
    A_roe = pd.Series((np.array(fund["roe"]) > 0.17).astype(float), index=c.columns)
    I_score = pd.Series((np.array(fund["inst_chg"]) > 0).astype(float), index=c.columns)

    # --- hard screens (O'Neil treats C, A, L, M as must-pass) ---
    M_on = gate_exposure(panel["market"]).iloc[t] > 0.5
    passC = pd.Series(fund["eps_q_yoy"], index=c.columns) >= 0.25
    passA = (pd.Series(fund["eps_3y_cagr"], index=c.columns) >= 0.20) & (A_roe > 0)
    passL = L_rank >= 0.80
    pass_mask = passC & passA & passL & bool(M_on)

    # --- composite (weighted; tune/validate, don't trust the weights) ---
    score = (0.22*C_score + 0.15*A_eps + 0.08*A_roe + 0.15*L_rank +
             0.12*N_prox + 0.10*S_demand.clip(0,1) + 0.08*float_small +
             0.10*I_score)
    return score, pass_mask


if __name__ == "__main__":
    print("=" * 60)
    print("  CAN SLIM scorer + breakout integration (synthetic)")
    print("=" * 60)
    panel = make_synthetic()
    fund = synth_fundamentals(panel["close"].shape[1])
    t = len(panel["close"]) - 30
    c = panel["close"]

    score, passes = canslim_score(panel, fund, t)
    leaders = score[passes].sort_values(ascending=False)

    print(f"\n  CAN SLIM hard-screen survivors (C+A+L+M): {int(passes.sum())} of {c.shape[1]}")
    print("  Top leaders by composite score:")
    print("  stock   score   EPSq_YoY   3yEPS    ROE   instChg")
    print("  " + "-"*52)
    for s in leaders.index[:8]:
        print(f"   #{s:<4} {score[s]:5.2f}   {fund['eps_q_yoy'][s]:6.1%}   "
              f"{fund['eps_3y_cagr'][s]:5.1%}  {fund['roe'][s]:5.1%}  "
              f"{fund['inst_chg'][s]:+5.1%}")

    # --- THE INTEGRATION: CAN SLIM leader AND breaking out today ---
    breaking = fires(panel, t, confirm_days=2)
    buy_list = [s for s in leaders.index if breaking.get(s, False)]
    print(f"\n  FINAL BUY LIST = CAN SLIM leader AND breakout firing today:")
    print(f"   {buy_list if buy_list else '(none today — normal; both filters are strict)'}")

    print("\n  Read: CAN SLIM picks WHO (quality leaders); the breakout picks WHEN.")
    print("  Synthetic fundamentals are random => this only proves the wiring.")
    print("  Real test: does adding the CAN SLIM filter to breakouts raise the")
    print("  event-study edge & Calmar vs breakouts alone? Let the data decide.\n")
