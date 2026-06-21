"""
================================================================================
 build_panel.py  —  assemble cached NSE data into the engine's `panel` dict
================================================================================
 Turns the wide close/value/deliv frames (from pipeline.load_bhavcopy) into the
 EXACT panel contract swing_engine.py consumes, so the engine runs unchanged:

   panel = {
     "close":   DataFrame [dates x stock-ids]   (Rs)
     "value":   DataFrame [dates x stock-ids]   (Rs traded value)
     "deliv":   DataFrame [dates x stock-ids]   delivery fraction [0,1]
     "market":  Series    [dates]               benchmark level
     "sectors": np.ndarray[stock-ids]           integer sector id per column
   }

 Columns are integer ids 0..n-1 (engine + canslim index by id). The id<->symbol
 map is returned alongside as `symbols` so we can read results back in English.

 ---------------------------------------------------------------------------
 HONESTY NOTES (what is real here vs still a placeholder):
   close/value/deliv : REAL NSE EOD data.
   market            : PROXY — equal-weight index of the most liquid names,
                       rebased to 100. Gives the gate a real trend/vol series.
                       TODO: replace with the actual Nifty Smallcap 250 level.
   sectors           : PLACEHOLDER zeros. The engine's simulate() does not use
                       sectors yet; CAN SLIM does not need them. TODO: real NSE
                       industry classification when we wire fundamentals.
   ADJUSTMENT        : close is UNADJUSTED. Multi-year backtests need a
                       split/bonus adjustment first (pipeline/adjust.py — TODO).
 ---------------------------------------------------------------------------
================================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pipeline.load_bhavcopy import load_frames
from pipeline.adjust import adjust_close


def _market_proxy(close: pd.DataFrame, value: pd.DataFrame,
                  top_n: int = 200) -> pd.Series:
    """Equal-weight daily-return index of the `top_n` most-liquid names,
    rebased to 100. A liquid, broad proxy so gate_exposure() has a real
    trend+vol series to switch on until we drop in the true smallcap index."""
    med_val = value.median(axis=0)
    liquid = med_val.dropna().sort_values(ascending=False).head(top_n).index
    rets = close[liquid].pct_change(fill_method=None)
    # equal-weight mean of cross-section, ignoring names not trading that day
    idx_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    return pd.Series(100 * np.exp(np.cumsum(idx_ret.values)),
                     index=close.index, name="market")


def build_panel(min_history: int = 1, min_price: float = 5.0) -> dict:
    """Assemble the panel dict from cached bhavcopy data.

    min_history : drop symbols seen on fewer than this many days (thin/just-listed)
    min_price   : drop penny symbols whose median close < this (Rs)
    """
    f = load_frames()
    # ---- split/bonus adjustment FIRST (continuous, honest return series) -----
    close_adj, events = adjust_close(f["close"], f["prev_close"])
    close, value, deliv = close_adj, f["value"], f["deliv"]

    # ---- light hygiene (NOT cherry-picking: structural, knowable upfront) ----
    keep = (close.notna().sum(axis=0) >= min_history) & (close.median() >= min_price)
    syms = close.columns[keep]
    close, value, deliv = close[syms], value[syms], deliv[syms]

    # ---- map symbols -> integer ids (stable, alphabetical) ------------------
    symbols = list(close.columns)
    ids = range(len(symbols))
    close.columns = list(ids)
    value.columns = list(ids)
    deliv.columns = list(ids)

    market = _market_proxy(close_adj, f["value"]).reindex(close.index)
    sectors = np.zeros(len(symbols), dtype=int)     # placeholder (see notes)

    return {"close": close, "value": value, "deliv": deliv,
            "market": market, "sectors": sectors, "symbols": symbols}


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 68)
    print("  build_panel  —  assemble cached NSE data into engine panel dict")
    print("=" * 68)
    p = build_panel()
    print(f"\n  close : {p['close'].shape[0]} dates x {p['close'].shape[1]} stocks")
    print(f"  market: {p['market'].iloc[0]:.2f} -> {p['market'].iloc[-1]:.2f} "
          f"(proxy, rebased 100)")
    print(f"  symbols[:8] : {p['symbols'][:8]}")
    print("\n  panel keys + types match swing_engine contract:")
    for k in ("close", "value", "deliv", "market", "sectors"):
        print(f"    {k:8s} {type(p[k]).__name__}")
    print("\n  NOTE: only a few days cached so far — fetch a multi-year range")
    print("  before running simulate() (it needs start=260+ trading days).")
