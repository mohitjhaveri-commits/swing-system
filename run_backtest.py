"""
================================================================================
 run_backtest.py  —  FIRST real-data backtest (large-cap / Nifty-50 proxy)
================================================================================
 End-to-end: cached NSE bhavcopy -> adjusted panel -> universe-restricted engine.
 Entries are restricted to the point-in-time large-cap universe by ANDing the
 universe mask into the timing trigger via signal_fn.

 This is a FIRST LOOK to confirm the machinery runs on real data and the numbers
 are sane (not a final, trustworthy result — universe is a liquidity proxy, market
 is a proxy index, adjustment not yet cross-validated). Read the printed caveats.
================================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from build_panel import build_panel
from pipeline.universe import universe_mask
from swing_engine import (simulate, metrics, print_metrics, timing_trigger,
                          ablate)


def make_signal_fn(umask: pd.DataFrame):
    """signal_fn(panel, t) = in-universe AND base-breakout trigger."""
    def fn(panel, t):
        trig = timing_trigger(panel, t)
        uni = umask.iloc[t].reindex(trig.index).fillna(False)
        return trig & uni
    return fn


def benchmark_metrics(market: pd.Series, start: int, periods_per_year: int = 252):
    """Buy-and-hold the (proxy) benchmark over the same window, for comparison."""
    m = market.iloc[start:].dropna()
    ret = m.pct_change().dropna()
    years = len(m) / periods_per_year
    cagr = (m.iloc[-1] / m.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    sharpe = ret.mean() / ret.std() * np.sqrt(periods_per_year) if ret.std() > 0 else np.nan
    dd = (m / m.cummax() - 1).min()
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd}


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 70)
    print("  FIRST REAL-DATA BACKTEST  —  large-cap (Nifty-50 liquidity proxy)")
    print("=" * 70)

    panel = build_panel()
    n_days, n_stk = panel["close"].shape
    print(f"\n  panel: {n_days} trading days x {n_stk} stocks "
          f"({panel['close'].index.min().date()} -> {panel['close'].index.max().date()})")

    umask = universe_mask(panel["value"])
    sig = make_signal_fn(umask)

    start = 260
    r = simulate(panel, start=start, signal_fn=sig)
    m = metrics(r["curve"], r["trades"])
    print_metrics(m, "STRATEGY (gate + selection + universe-breakout, after costs):")

    bm = benchmark_metrics(panel["market"], start)
    print(f"\n  BENCHMARK (buy & hold proxy index, same window):")
    print(f"    CAGR {bm['CAGR']:6.1%} | Sharpe {bm['Sharpe']:5.2f} | MaxDD {bm['MaxDD']:6.1%}")

    print(f"\n  trades: {m['NTrades']}  (watch this — Nifty-50 is a small pond)")

    print("\n" + "-" * 70)
    print("  HONESTY CHECKLIST (why this is a FIRST LOOK, not a verdict):")
    print("  - universe = top-50 by liquidity (lets in IDEA/ETFs) — not true Nifty50")
    print("  - market = equal-weight liquid proxy — not the real Nifty 50 TR index")
    print("  - corporate-action adjustment not yet cross-validated vs a known split")
    print("  - if CAGR looks rich, SUSPECT A LEAK before believing it (per CLAUDE.md)")
