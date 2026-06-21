"""
================================================================================
 pipeline/nifty50.py  —  real Nifty 50 membership, sectors, and benchmark
================================================================================
 Replaces the top-50-by-liquidity PROXY (which let in ETFs like SILVERBEES and
 penny names like IDEA) with the OFFICIAL NSE Nifty 50 constituent list. The same
 file carries each name's Industry, so we also get REAL sector ids (the panel's
 sectors were placeholder zeros before).

 SURVIVORSHIP CAVEAT (logged, not hidden): NSE only publishes the CURRENT list,
 so applying it across the 3y backtest carries mild survivorship bias (~a handful
 of constituent changes over the window). Far cleaner than the liquidity proxy;
 a fully point-in-time membership history is a later upgrade.

 Benchmark: equal-weight total return of the in-universe members from our ADJUSTED
 closes, rebased to 100. (The real Nifty 50 is float-cap-weighted; equal-weight is
 a fair, self-consistent yardstick for an equal-weight strategy and uses the same
 names. TODO: float-cap weights if/when shares-outstanding is wired.)
================================================================================
"""
from __future__ import annotations
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "ind_nifty50list.csv"
URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 Chrome/124.0", "Accept": "*/*"}


def fetch_list() -> Path:
    if not CACHE.exists():
        req = urllib.request.Request(URL, headers=HEADERS)
        data = urllib.request.urlopen(req, timeout=20).read()
        CACHE.write_bytes(data)
    return CACHE


def load_list() -> pd.DataFrame:
    """DataFrame with columns: symbol, industry."""
    df = pd.read_csv(fetch_list())
    df.columns = [c.strip() for c in df.columns]
    return pd.DataFrame({"symbol": df["Symbol"].str.strip(),
                         "industry": df["Industry"].str.strip()})


def apply_to_panel(panel: dict) -> dict:
    """Return (umask, benchmark, sectors) for the Nifty 50 universe, aligned to
    the panel's integer stock ids.

    umask     : bool DataFrame [dates x ids] — True where a member has data that day
    benchmark : Series [dates] — equal-weight TR of members, rebased 100
    sectors   : np.ndarray[ids] — integer sector id from NSE Industry (-1 if none)
    """
    lst = load_list()
    sym2ind = dict(zip(lst["symbol"], lst["industry"]))
    members = set(lst["symbol"])
    symbols = panel["symbols"]

    member_ids = [i for i, s in enumerate(symbols) if s in members]
    close = panel["close"]

    # static membership mask, but only where the stock actually traded that day
    mask = pd.DataFrame(False, index=close.index, columns=close.columns)
    has = close[member_ids].notna()
    mask[member_ids] = has

    # equal-weight TR benchmark from adjusted closes of members
    rets = close[member_ids].pct_change(fill_method=None)
    bench_ret = rets.mean(axis=1, skipna=True).fillna(0.0)
    benchmark = pd.Series(100 * np.exp(np.cumsum(bench_ret.values)),
                          index=close.index, name="nifty50_ew")

    # real sectors from Industry
    inds = sorted({sym2ind[s] for s in symbols if s in sym2ind})
    ind2id = {name: k for k, name in enumerate(inds)}
    sectors = np.array([ind2id.get(sym2ind.get(s, None), -1) for s in symbols], dtype=int)

    return {"umask": mask, "benchmark": benchmark, "sectors": sectors,
            "n_members": len(member_ids)}


if __name__ == "__main__":
    from build_panel import build_panel
    pd.set_option("display.width", 120)
    print("=" * 70)
    print("  nifty50  —  real membership, sectors, benchmark")
    print("=" * 70)
    panel = build_panel()
    nf = apply_to_panel(panel)
    print(f"\n  Nifty 50 members found in panel: {nf['n_members']}/50")
    print(f"  benchmark (EW TR): {nf['benchmark'].iloc[260]:.1f} -> "
          f"{nf['benchmark'].iloc[-1]:.1f}  "
          f"({(nf['benchmark'].iloc[-1]/nf['benchmark'].iloc[260]-1):.1%} over test window)")
    nsec = len({s for s in nf['sectors'] if s >= 0})
    print(f"  real sectors mapped: {nsec} industries")
    sizes = nf['umask'].sum(axis=1)
    print(f"  daily universe size: median {sizes.median():.0f} "
          f"(vs 50 — gaps = members not yet listed early in window)")
