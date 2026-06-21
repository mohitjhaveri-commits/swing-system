"""
================================================================================
 mean_reversion.py  —  short-term reversal ("buy the dip in an uptrend")
================================================================================
 Motivated by the consolidation-breakout result: large-cap breakouts had a
 SIGNIFICANT NEGATIVE forward edge (t=-4), i.e. buying strength got faded. The
 testable flip side is short-term MEAN-REVERSION: after a brief pullback WITHIN a
 longer uptrend, large caps tend to bounce.

   TREND   : close > trailing `trend_window`-day average  (only buy dips in names
             still in an uptrend — avoid catching falling knives)
   PULLBACK: price fell >= `drop_pct` over the last `drop_days` days (oversold)
   SENTIMENT (optional): delivery % rising into the weakness (accumulation)
   GATE    : applied by the engine (risk-on only)

 Expect POSITIVE forward returns when this fires (the opposite sign to the
 breakout). All inputs use bars strictly before t. Test it honestly — a story
 that fits the last result is still just a hypothesis until the data confirms it.
================================================================================
"""
from __future__ import annotations
import pandas as pd


def fires(panel: dict, t: int,
          trend_window: int = 100, drop_days: int = 3, drop_pct: float = 0.04,
          deliv_backed: bool = False) -> pd.Series:
    """Boolean per-stock mean-reversion trigger at day t (bars < t only)."""
    c = panel["close"]; d = panel["deliv"]
    need = trend_window + drop_days + 2
    if t < need:
        return pd.Series(False, index=c.columns)

    last = c.iloc[t - 1]
    sma = c.iloc[t - 1 - trend_window:t - 1].mean()
    uptrend = last > sma

    drop = c.iloc[t - 1] / c.iloc[t - 1 - drop_days] - 1
    pulled = drop < -drop_pct

    sig = uptrend & pulled
    if deliv_backed:
        dd = d.iloc[t - 5:t].mean() > d.iloc[t - 25:t - 5].mean()
        sig = sig & dd
    return sig.fillna(False)


if __name__ == "__main__":
    from swing_engine import make_synthetic
    from consolidation_breakout import event_study
    pd.set_option("display.width", 120)
    print("=" * 68)
    print("  mean_reversion  —  wiring check on synthetic (expect ~0 edge)")
    print("=" * 68)
    panel = make_synthetic()
    es = event_study(panel, lambda p, t: fires(p, t))
    print(es.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
