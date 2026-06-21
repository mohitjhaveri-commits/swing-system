"""
================================================================================
 consolidation_breakout.py  —  volatility-contraction / base-breakout signal
================================================================================
 The VCP / Darvas-base family, operationalized for EOD large-cap swing entries
 (see CLAUDE.md "Candidate strategy: consolidation breakout").

   BASE      : ~base_window-day consolidation that is TIGHT
               (high-low)/mean < `tightness`  (volatility has contracted)
   TRIGGER   : latest close breaks the base high, AND each of the last
               `confirm_days` closes is an UP day on traded value
               > `vol_mult` x the base-period average value
   SENTIMENT : delivery % over the confirm days > delivery % over the base
               (delivery-backed buying — screens pump-traps that show big
               volume but low delivery)
   GATE      : applied by the engine (entries only when market risk-on)

 ALL inputs use bars STRICTLY BEFORE t (latest known bar = t-1); the engine
 fills the entry at the close of t. No look-ahead.

 NOTE on "volume": the panel carries traded VALUE (Rs), not share volume, so the
 volume confirmation uses value (price*volume). Consistent with the engine's
 built-in timing and fine as a liquidity-of-interest proxy.

 KEY TENSION (CLAUDE.md): more confirm_days -> fewer false breakouts but far
 fewer signals. Tune on real data; beware the <30-trade kill rule and overfit.
================================================================================
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def fires(panel: dict, t: int,
          base_window: int = 25, tightness: float = 0.12,
          confirm_days: int = 2, vol_mult: float = 1.5,
          deliv_backed: bool = True) -> pd.Series:
    """Boolean per-stock trigger at day t (uses bars < t only)."""
    c = panel["close"]; v = panel["value"]; d = panel["deliv"]
    need = base_window + confirm_days + 2
    if t < need:
        return pd.Series(False, index=c.columns)

    conf = slice(t - confirm_days, t)                       # last confirm_days bars
    base = slice(t - confirm_days - base_window, t - confirm_days)

    base_c = c.iloc[base]
    base_high = base_c.max()
    rng = (base_high - base_c.min()) / base_c.mean()
    tight = rng < tightness                                 # volatility contracted

    broke = c.iloc[t - 1] > base_high                       # latest close breaks out

    # each of the last confirm_days closes is an up day
    conf_c = c.iloc[t - confirm_days - 1:t]
    ups = (conf_c.diff().iloc[1:] > 0).all()

    # value (volume) confirmation vs base average
    vol_ok = v.iloc[conf].mean() > vol_mult * v.iloc[base].mean()

    # delivery-backed sentiment
    if deliv_backed:
        deliv_ok = d.iloc[conf].mean() > d.iloc[base].mean()
    else:
        deliv_ok = pd.Series(True, index=c.columns)

    return (tight & broke & ups & vol_ok & deliv_ok).fillna(False)


def event_study(panel: dict, signal_t_fn, horizons=(5, 10, 15),
                start: int = 260, cost: float = 0.004) -> pd.DataFrame:
    """Forward-return event study: when the setup FIRES, what are the average
    H-day forward returns (after a flat round-trip cost), vs the universe
    baseline (all stock-days)? Reports mean, t-stat, n. A real edge shows
    setup-returns meaningfully ABOVE baseline AND a t-stat clearing ~2.

    signal_t_fn(panel, t) -> boolean Series (e.g. universe & fires).
    cost : flat round-trip cost haircut applied to the event leg only.
    """
    c = panel["close"]
    n = len(c)
    rows = []
    for H in horizons:
        ev_rets, base_rets = [], []
        for t in range(start, n - H):
            fwd = c.iloc[t - 1 + H] / c.iloc[t - 1] - 1      # enter t-1 close, exit +H
            fwd = fwd.replace([np.inf, -np.inf], np.nan)
            sig = signal_t_fn(panel, t).reindex(c.columns).fillna(False)
            ev = fwd[sig].dropna()
            if len(ev):
                ev_rets.append(ev.values - cost)
            base_rets.append(fwd.dropna().values)
        ev_all = np.concatenate(ev_rets) if ev_rets else np.array([])
        base_all = np.concatenate(base_rets) if base_rets else np.array([])
        if len(ev_all) > 1:
            mean_ev = ev_all.mean()
            tstat = mean_ev / (ev_all.std(ddof=1) / np.sqrt(len(ev_all)))
        else:
            mean_ev, tstat = np.nan, np.nan
        rows.append({"horizon": H, "n_events": len(ev_all),
                     "mean_event": mean_ev, "mean_baseline": base_all.mean() if len(base_all) else np.nan,
                     "edge_vs_base": (mean_ev - base_all.mean()) if len(base_all) else np.nan,
                     "t_stat": tstat})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from swing_engine import make_synthetic
    pd.set_option("display.width", 120)
    print("=" * 68)
    print("  consolidation_breakout  —  wiring check on synthetic (expect weak)")
    print("=" * 68)
    panel = make_synthetic()
    t = len(panel["close"]) - 30
    f = fires(panel, t)
    print(f"\n  signals firing at t={t}: {int(f.sum())} of {panel['close'].shape[1]}")
    es = event_study(panel, lambda p, t: fires(p, t))
    print("\n  event study on synthetic (no edge baked in -> edge ~0, |t| small):")
    print(es.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
