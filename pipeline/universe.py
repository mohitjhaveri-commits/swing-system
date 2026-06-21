"""
================================================================================
 pipeline/universe.py  —  point-in-time large-cap (Nifty-50-like) membership
================================================================================
 The engine trades only LARGE CAPS (universe = Nifty 50). The rigorous way is
 historical Nifty 50 membership, but that is not freely available; bhavcopy also
 lacks shares-outstanding, so we cannot compute true market cap.

 PROXY (survivorship-free, point-in-time): on each day, the universe is the
 top-N names by TRAILING median traded value (liquidity). Nifty 50 constituents
 dominate cash-market turnover, so top-50-by-liquidity tracks the index closely.
 Crucially this is computed from ONLY trailing data, so it is survivorship-free
 and look-ahead-safe: a name enters when it actually became liquid, and a name
 that later vanished is simply absent on later dates.

 LIMITATION (logged, not hidden): liquidity != market cap. A few high-turnover
 mid-caps can sneak in and a quiet large-cap can drop out. TODO: replace with
 real point-in-time Nifty 50 membership when a source is wired.

 Returns a boolean DataFrame [dates x ids] -> True where the name is in-universe
 that day. Combine with the timing trigger to restrict entries to large caps.
================================================================================
"""
from __future__ import annotations
import pandas as pd


def universe_mask(value: pd.DataFrame, top_n: int = 50,
                  lookback: int = 60, min_days: int = 60) -> pd.DataFrame:
    """Boolean [dates x ids]: True where the stock is among the top_n by trailing
    `lookback`-day median traded value as of that day.

    min_days : require at least this much trailing history before a name can be
               in-universe (avoids ranking freshly-listed names on 2-3 prints).
    """
    # trailing median traded value, knowable at each EOD (shift(1) = strictly past)
    trail = value.shift(1).rolling(lookback, min_periods=min_days).median()
    # rank per day: 1 = most liquid; keep ranks <= top_n
    ranks = trail.rank(axis=1, ascending=False, method="first")
    return ranks <= top_n


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    from build_panel import build_panel
    print("=" * 68)
    print("  universe  —  point-in-time top-50-by-liquidity (Nifty-50 proxy)")
    print("=" * 68)
    p = build_panel()
    mask = universe_mask(p["value"])
    syms = p["symbols"]

    sizes = mask.sum(axis=1)
    print(f"\n  daily universe size: min {sizes.min()}, max {sizes.max()}, "
          f"median {sizes.median():.0f} (target {50})")

    # show the universe on the last day, in English
    last = mask.iloc[-1]
    members = sorted(syms[i] for i in last.index[last.values])
    print(f"\n  in-universe on {mask.index[-1].date()} ({len(members)} names):")
    for i in range(0, len(members), 6):
        print("    " + "  ".join(f"{m:<12}" for m in members[i:i + 6]))
