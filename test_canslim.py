"""
================================================================================
 test_canslim.py  —  does CAN SLIM quality predict large-cap returns? (REAL data)
================================================================================
 The breakout TIMING was killed (test_signals.py), so testing CAN SLIM as an
 overlay on it would yield ~no signals. Instead we test the more basic question
 the fundamentals let us finally ask honestly:

   Among large caps, do high CAN SLIM-score names beat low-score names?

 METHOD — monthly cross-sectional factor test (look-ahead-safe):
   - On the first trading day of each month, score every in-universe stock with
     canslim.canslim_score (price components from panel + REAL fundamentals as of
     that date, using only data >=45 days old).
   - Form equal-weight TOP-quintile and BOTTOM-quintile portfolios; hold ~21
     trading days; record forward return (adjusted close).
   - Compare top vs bottom vs universe average; t-stat on the top-minus-bottom
     spread. Also track the strict hard-screen "leaders" (C&A&L&M) count/return.

 A real fundamental edge => top > bottom with an economically meaningful spread
 and |t| > ~2. Otherwise CAN SLIM quality adds nothing here -> kill it.
================================================================================
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from build_panel import build_panel
from pipeline.universe import universe_mask
from pipeline.fundamentals import parse_company, asof_features
from canslim import canslim_score


def load_parsed(symbols):
    out = {}
    for s in symbols:
        p = parse_company(s)
        if p:
            out[s] = p
    return out


def build_fund_arrays(parsed_by_sym, symbols, n, asof):
    """fund dict (arrays length n, indexed by stock id) with point-in-time values."""
    keys = ["eps_q_yoy", "eps_q_accel", "eps_3y_cagr", "roe", "inst_chg"]
    arr = {k: np.full(n, np.nan) for k in keys}
    for i, sym in enumerate(symbols):
        p = parsed_by_sym.get(sym)
        if not p:
            continue
        f = asof_features(p, asof)
        for k in keys:
            arr[k][i] = f[k]
    arr["float_cr"] = np.ones(n)            # no float data for large caps -> neutral
    return arr


def month_starts(dates, start_i):
    out = []
    prev_m = None
    for i in range(start_i, len(dates)):
        ym = (dates[i].year, dates[i].month)
        if ym != prev_m:
            out.append(i)
            prev_m = ym
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 72)
    print("  CAN SLIM factor test  —  do quality leaders beat laggards? (real data)")
    print("=" * 72)

    panel = build_panel()
    c = panel["close"]
    dates = c.index
    symbols = panel["symbols"]
    n = len(symbols)
    umask = universe_mask(panel["value"])

    uni_syms = json.load(open("data/universe_symbols.json"))
    parsed = load_parsed(uni_syms)
    print(f"\n  fundamentals parsed for {len(parsed)}/{len(uni_syms)} universe names")

    HOLD = 21
    rebs = [i for i in month_starts(dates, 260) if i + HOLD < len(dates)]
    top_r, bot_r, uni_r, lead_r, lead_n = [], [], [], [], []

    for t in rebs:
        uni_ids = [j for j in umask.columns if bool(umask.iloc[t][j])]
        if len(uni_ids) < 10:
            continue
        fund = build_fund_arrays(parsed, symbols, n, dates[t])
        score, passmask = canslim_score(panel, fund, t)

        fwd = c.iloc[t + HOLD] / c.iloc[t] - 1     # enter close[t], exit close[t+HOLD]
        sc = score.reindex(uni_ids)
        fw = fwd.reindex(uni_ids)
        ok = sc.notna() & fw.notna()
        sc, fw = sc[ok], fw[ok]
        if len(sc) < 10:
            continue

        k = max(1, len(sc) // 5)
        order = sc.sort_values(ascending=False)
        top = fw[order.index[:k]].mean()
        bot = fw[order.index[-k:]].mean()
        top_r.append(top); bot_r.append(bot); uni_r.append(fw.mean())

        leaders = [j for j in uni_ids if bool(passmask.get(j, False))]
        lead_n.append(len(leaders))
        lf = fwd.reindex(leaders).dropna()
        lead_r.append(lf.mean() if len(lf) else np.nan)

    def ann(mean_m):
        return (1 + mean_m) ** 12 - 1

    top_r = np.array(top_r); bot_r = np.array(bot_r); uni_r = np.array(uni_r)
    spread = top_r - bot_r
    tstat = spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))
    lead_r = np.array([x for x in lead_r if not np.isnan(x)])

    print(f"\n  rebalances: {len(top_r)} months, hold {HOLD} trading days\n")
    print(f"  {'portfolio':<22}{'avg monthly':>14}{'annualized':>14}")
    print("  " + "-" * 50)
    print(f"  {'TOP quintile':<22}{top_r.mean():>13.2%}{ann(top_r.mean()):>14.1%}")
    print(f"  {'universe average':<22}{uni_r.mean():>13.2%}{ann(uni_r.mean()):>14.1%}")
    print(f"  {'BOTTOM quintile':<22}{bot_r.mean():>13.2%}{ann(bot_r.mean()):>14.1%}")
    print(f"  {'hard-screen leaders':<22}{lead_r.mean():>13.2%}{ann(lead_r.mean()):>14.1%}"
          f"   (avg {np.mean(lead_n):.1f} names/month)")
    print(f"\n  TOP - BOTTOM spread: {spread.mean():.2%}/month  (t-stat {tstat:.2f})")
    print("\n  " + "=" * 50)
    print("  VERDICT: a real quality edge => TOP > BOTTOM, meaningful spread,")
    print("  |t| > ~2. Flat/negative spread => CAN SLIM adds no edge on large caps.")
