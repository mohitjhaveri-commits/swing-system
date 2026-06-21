"""
================================================================================
 test_signals.py  —  is there an edge? consolidation breakout on REAL data
================================================================================
 Runs the consolidation-breakout signal through the full discipline on 3y of real
 large-cap data:
   1. ENGINE BACKTEST  — simulate (gate + selection + universe-breakout), vs B&H
   2. EVENT STUDY      — forward returns when the setup fires, after costs, t-stat
   3. WALK-FORWARD     — fit risk params on train, score untouched test (overfit tell)
   4. ABLATIONS        — does the gate / selection layer earn its keep?

 CAN SLIM is intentionally NOT tested here: it needs REAL fundamentals (EPS/ROE/
 institutional holding) we have not wired yet. Testing it on random synthetic
 fundamentals would be meaningless. See CLAUDE.md NEXT TASK (screener.in).
================================================================================
"""
from __future__ import annotations
import pandas as pd

from build_panel import build_panel
from pipeline.universe import universe_mask
from consolidation_breakout import fires, event_study
from swing_engine import simulate, metrics, print_metrics, walk_forward
from run_backtest import benchmark_metrics


def make_signal_fn(umask, **fire_kw):
    def fn(panel, t):
        sig = fires(panel, t, **fire_kw)
        uni = umask.iloc[t].reindex(sig.index).fillna(False)
        return sig & uni
    return fn


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 70)
    print("  CONSOLIDATION BREAKOUT  —  real-data test (large-cap, 3y)")
    print("=" * 70)

    panel = build_panel()
    umask = universe_mask(panel["value"])
    sig = make_signal_fn(umask)
    start = 260

    # --- 1. ENGINE BACKTEST ---------------------------------------------------
    r = simulate(panel, start=start, signal_fn=sig)
    m = metrics(r["curve"], r["trades"])
    print_metrics(m, "1) ENGINE BACKTEST (gate+sel+consolidation, after costs):")
    bm = benchmark_metrics(panel["market"], start)
    print(f"\n   BENCHMARK (buy & hold proxy): CAGR {bm['CAGR']:.1%} | "
          f"Sharpe {bm['Sharpe']:.2f} | MaxDD {bm['MaxDD']:.1%}")

    # --- 2. EVENT STUDY -------------------------------------------------------
    print("\n" + "-" * 70)
    print("  2) EVENT STUDY (fwd returns when setup fires, after ~0.4% cost):")
    es = event_study(panel, sig, start=start)
    print(es.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print("   edge is real only if mean_event > baseline AND |t_stat| > ~2")

    # --- 3. WALK-FORWARD ------------------------------------------------------
    print("\n" + "-" * 70)
    print("  3) WALK-FORWARD (in-sample vs out-of-sample; OOS<<IS = overfit):")
    wf = walk_forward(panel, signal_fn=sig)
    if len(wf):
        show = wf[["fold", "is_CAGR", "oos_CAGR", "is_Sharpe", "oos_Sharpe",
                   "is_Calmar", "oos_Calmar"]]
        print(show.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))
        print(f"\n   mean IS Sharpe {wf['is_Sharpe'].mean():.2f} vs "
              f"OOS Sharpe {wf['oos_Sharpe'].mean():.2f}")
    else:
        print("   (not enough history for a full walk-forward fold)")

    # --- 4. ABLATIONS ---------------------------------------------------------
    print("\n" + "-" * 70)
    print("  4) ABLATIONS (does each layer earn its keep, with this signal?):")
    configs = {
        "gate + selection": dict(use_gate=True, use_selection=True),
        "no gate":          dict(use_gate=False, use_selection=True),
        "no selection":     dict(use_gate=True, use_selection=False),
    }
    rows = []
    for name, cfg in configs.items():
        rr = simulate(panel, start=start, signal_fn=sig, **cfg)
        mm = metrics(rr["curve"], rr["trades"])
        rows.append({"config": name, "CAGR": mm["CAGR"], "Sharpe": mm["Sharpe"],
                     "MaxDD": mm["MaxDD"], "Calmar": mm["Calmar"], "Trades": mm["NTrades"]})
    print(pd.DataFrame(rows).to_string(index=False,
          float_format=lambda x: f"{x:7.2%}" if abs(x) < 5 else f"{x:7.2f}"))

    print("\n" + "=" * 70)
    print("  VERDICT GUIDE (CLAUDE.md kill criteria): keep the signal ONLY if it")
    print("  beats B&H risk-adjusted, the event-study t-stat clears ~2, OOS does")
    print("  not collapse vs IS, and trades >= 30. Otherwise it has no edge.")
