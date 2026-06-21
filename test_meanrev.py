"""
================================================================================
 test_meanrev.py  —  mean-reversion on the CLEAN real Nifty 50 universe
================================================================================
 Runs the short-term reversal signal through the full discipline, on the real
 Nifty 50 membership + real equal-weight benchmark (no ETF/penny contamination):
   1. EVENT STUDY  — param sweep; forward returns after the dip, after costs, t
   2. BACKTEST     — engine vs benchmark
   3. WALK-FORWARD — IS vs OOS
   4. ABLATIONS    — gate / selection
================================================================================
"""
from __future__ import annotations
import pandas as pd

from build_panel import build_panel
from pipeline.nifty50 import apply_to_panel
from mean_reversion import fires
from consolidation_breakout import event_study
from swing_engine import simulate, metrics, print_metrics, walk_forward
from run_backtest import benchmark_metrics


def make_signal_fn(umask, **fk):
    def fn(panel, t):
        sig = fires(panel, t, **fk)
        uni = umask.iloc[t].reindex(sig.index).fillna(False)
        return sig & uni
    return fn


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 72)
    print("  MEAN-REVERSION  —  real Nifty 50 universe, real benchmark (3y)")
    print("=" * 72)

    panel = build_panel()
    nf = apply_to_panel(panel)
    panel["market"] = nf["benchmark"]                 # real EW Nifty 50 as the gate/bench
    panel["sectors"] = nf["sectors"]
    umask = nf["umask"]
    start = 260

    # --- 1. EVENT STUDY param sweep ------------------------------------------
    print("\n  1) EVENT STUDY (fwd returns after the dip; want mean>0, t>~2):")
    print(f"  {'drop_days':>9}{'drop_pct':>9}{'H':>4}{'n':>6}{'mean_ev':>10}{'baseline':>10}{'t_stat':>8}")
    for dd in (2, 3, 5):
        for dp in (0.03, 0.05):
            sig = make_signal_fn(umask, drop_days=dd, drop_pct=dp)
            es = event_study(panel, sig, start=start)
            r = es[es["horizon"] == 5].iloc[0]
            print(f"  {dd:>9}{dp:>9.2f}{5:>4}{int(r['n_events']):>6}"
                  f"{r['mean_event']:>10.4f}{r['mean_baseline']:>10.4f}{r['t_stat']:>8.2f}")

    # --- 2. BACKTEST (pick a middle setting) ---------------------------------
    sig = make_signal_fn(umask, drop_days=3, drop_pct=0.04)
    print("\n" + "-" * 72)
    r = simulate(panel, start=start, signal_fn=sig, max_hold=7)
    m = metrics(r["curve"], r["trades"])
    print_metrics(m, "2) BACKTEST (mean-rev, max_hold=7, after costs):")
    bm = benchmark_metrics(panel["market"], start)
    print(f"\n   BENCHMARK (Nifty 50 EW): CAGR {bm['CAGR']:.1%} | "
          f"Sharpe {bm['Sharpe']:.2f} | MaxDD {bm['MaxDD']:.1%}")

    # --- 3. WALK-FORWARD -----------------------------------------------------
    print("\n" + "-" * 72)
    print("  3) WALK-FORWARD (IS vs OOS):")
    wf = walk_forward(panel, signal_fn=sig, max_hold=7)
    if len(wf):
        print(wf[["fold", "is_CAGR", "oos_CAGR", "is_Sharpe", "oos_Sharpe",
                  "is_Calmar", "oos_Calmar"]].to_string(index=False,
              float_format=lambda x: f"{x:7.2f}"))

    # --- 4. ABLATIONS --------------------------------------------------------
    print("\n" + "-" * 72)
    print("  4) ABLATIONS:")
    rows = []
    for name, cfg in {"gate+sel": dict(use_gate=True, use_selection=True),
                      "no gate": dict(use_gate=False, use_selection=True),
                      "no selection": dict(use_gate=True, use_selection=False)}.items():
        rr = simulate(panel, start=start, signal_fn=sig, max_hold=7, **cfg)
        mm = metrics(rr["curve"], rr["trades"])
        rows.append({"config": name, "CAGR": mm["CAGR"], "Sharpe": mm["Sharpe"],
                     "MaxDD": mm["MaxDD"], "Calmar": mm["Calmar"], "Trades": mm["NTrades"]})
    print(pd.DataFrame(rows).to_string(index=False,
          float_format=lambda x: f"{x:7.2%}" if abs(x) < 5 else f"{x:7.2f}"))
