"""
================================================================================
 pipeline/adjust.py  —  split / bonus back-adjustment from NSE PREV_CLOSE
================================================================================
 WHY: bhavcopy CLOSE_PRICE is UNADJUSTED. On a 1:1 bonus a stock's raw close
 halves overnight; an unadjusted backtest reads that as a -50% loss that never
 happened. CLAUDE.md requires ADJUSTED close. We must fix this before any honest
 return is computed.

 HOW (self-contained — no extra data source):
   NSE publishes PREV_CLOSE = the OFFICIAL adjusted previous close. On a normal
   day  PREV_CLOSE[t] == CLOSE[t-1].  On a corporate-action ex-date NSE sets
   PREV_CLOSE[t] to the adjusted basis, so

       ratio[t] = PREV_CLOSE[t] / CLOSE[t-1]

   departs from 1.0 by exactly the action's factor (1:1 bonus -> ~0.5,
   2:1 split -> ~0.333, etc.). To get a continuous adjusted series we multiply
   all bars STRICTLY BEFORE each ex-date by the cumulative product of the future
   ratios. (Equivalently: walk backwards, scaling older prices down/up so the
   series is continuous across the action.)

 GUARDRAILS (avoid mistaking noise/bad ticks for actions):
   - Only treat |ratio-1| > `tol` as an action (normal gaps stay ~1.0).
   - Ignore ratios outside [`lo`, `hi`] as bad data (e.g. a 100x tick error),
     not a real action — left as 1.0, logged.
   - Dividends are NOT adjusted here (NSE does not fold ordinary dividends into
     PREV_CLOSE for equities), which matches how we trade price-only.

 This adjusts `close`. `value` and `deliv` are flow/ratio series and need no
 price adjustment. (Volume would, if we used raw shares; we use rupee value.)
================================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def adjustment_factors(close: pd.DataFrame, prev_close: pd.DataFrame,
                       tol: float = 0.05, lo: float = 0.2, hi: float = 1.8,
                       max_day_frac: float = 0.02
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (factors, events).

    factors : DataFrame [dates x ids] cumulative multiplier to apply to raw close
              so the series is continuous (==1.0 on/after the last action).
    events  : long DataFrame of detected actions (date, id, ratio) for audit.

    tol          : min |ratio-1| to count as an action (above session/round noise;
                   a 1:10 bonus is ~9%, most actions far larger).
    max_day_frac : if MORE than this fraction of that day's traded names look like
                   actions, it is a calendar artifact (e.g. Muhurat session gap),
                   not real corporate actions -> the whole day is rejected.
    """
    prior = close.shift(1)
    ratio = prev_close / prior                       # ~1.0 normally

    in_band = (ratio >= lo) & (ratio <= hi)
    candidate = (ratio.sub(1).abs() > tol) & in_band
    # bad data: large deviation but outside plausible action band -> ignore
    bad = (ratio.sub(1).abs() > tol) & ~in_band

    # mass-event guard: drop whole days where too many names "move" together
    traded = ratio.notna().sum(axis=1).clip(lower=1)
    day_frac = candidate.sum(axis=1) / traded
    artifact_days = day_frac > max_day_frac
    is_action = candidate & ~artifact_days.to_numpy()[:, None]

    eff = ratio.where(is_action, 1.0).fillna(1.0)    # 1.0 except on ex-dates

    # cumulative product of all ratios at/after each date, computed from the end:
    # a bar before an ex-date must be scaled by every future action ratio.
    rev_cum = eff[::-1].cumprod()[::-1]
    factors = rev_cum.shift(-1).fillna(1.0)          # apply FUTURE actions to today

    # audit log of what we acted on
    ev = is_action.stack()
    ev = ev[ev]
    events = pd.DataFrame({
        "date": ev.index.get_level_values(0),
        "id": ev.index.get_level_values(1),
        "ratio": [ratio.loc[d, i] for d, i in ev.index],
    })
    events.attrs["n_bad_ticks"] = int(bad.to_numpy().sum())
    events.attrs["n_artifact_days"] = int(artifact_days.sum())
    return factors, events


def adjust_close(close: pd.DataFrame, prev_close: pd.DataFrame, **kw
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adjusted_close, events)."""
    factors, events = adjustment_factors(close, prev_close, **kw)
    return close * factors, events


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    from pipeline.load_bhavcopy import load_frames
    print("=" * 68)
    print("  adjust  —  detect split/bonus from PREV_CLOSE and back-adjust close")
    print("=" * 68)
    f = load_frames()
    adj, events = adjust_close(f["close"], f["prev_close"])
    n = len(events)
    print(f"\n  detected {n} corporate-action ex-dates across the cached window")
    print(f"  ignored {events.attrs.get('n_bad_ticks', 0)} implausible-ratio bad ticks")
    if n:
        print("\n  sample events (date, stock-id, ratio):")
        print(events.head(12).to_string(index=False))
    print("\n  (few/none expected on a 5-day window — real actions show up once")
    print("   the multi-year history finishes downloading.)")
