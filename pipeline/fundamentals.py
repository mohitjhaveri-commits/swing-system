"""
================================================================================
 pipeline/fundamentals.py  —  real fundamentals from screener.in (for CAN SLIM)
================================================================================
 Fetches & parses each company's public screener.in page into the fundamentals
 CAN SLIM needs:
   - quarterly EPS         -> C: current-qtr EPS YoY (+ acceleration)
   - annual EPS            -> A: 3-yr EPS CAGR
   - Return on Equity (3y) -> A: quality
   - FII + DII holding     -> I: institutional sponsorship trend (QoQ)

 LOOK-AHEAD DISCIPLINE (CLAUDE.md): a quarter labelled "Mar 2024" (period end
 2024-03-31) is only KNOWABLE after results are announced. We stamp each period
 available at period_end + `LAG_DAYS` (default 45) and `asof_features()` uses
 ONLY periods available by the as-of date. Never the period-end date itself.

 Tables are located by CONTENT, not index (screener reorders tables per company,
 e.g. extra segment tables), so parsing is robust across names.

 Politeness: cache each company's HTML under data/fundamentals/; never re-fetch
 what is on disk. Stdlib urllib + pandas.read_html(lxml).
================================================================================
"""
from __future__ import annotations
import time
import urllib.request
import urllib.error
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parent.parent / "data" / "fundamentals"
LAG_DAYS = 45                                   # quarter-end -> announcement lag
URL = "https://www.screener.in/company/{sym}/"
URL_CONS = "https://www.screener.in/company/{sym}/consolidated/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
           "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
           "Accept": "text/html"}


def fetch_company(symbol: str, retries: int = 3, pause: float = 1.0) -> Path | None:
    """Download & cache a company's screener page. Tries consolidated first
    (preferred for groups), falls back to standalone. Returns cached path or None."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{symbol}.html"
    if out.exists():
        return out
    for url in (URL_CONS.format(sym=symbol), URL.format(sym=symbol)):
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=30) as r:
                    html = r.read().decode("utf-8", "ignore")
                if "Quarterly Results" in html:
                    out.write_text(html)
                    return out
                break                            # page exists but no data -> try next url
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break                        # consolidated missing -> try standalone
                if attempt == retries:
                    break
            except Exception:
                if attempt == retries:
                    break
            time.sleep(pause * attempt)
    return None


def _num(x) -> float:
    """Parse a screener cell ('1,234', '23%', '', '-12.3') -> float/NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").replace("%", "").replace("\xa0", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _period_end(label: str):
    """'Mar 2024' -> Timestamp 2024-03-31. Non-period headers -> NaT."""
    try:
        return pd.to_datetime(label.strip(), format="%b %Y") + pd.offsets.MonthEnd(0)
    except (ValueError, TypeError):
        return pd.NaT


def _row_by_label(tbl: pd.DataFrame, target: str) -> pd.Series | None:
    """Return the data row whose first-column label matches `target`, indexed by
    period-end date (non-period columns dropped)."""
    labels = tbl.iloc[:, 0].astype(str).str.replace("\xa0", " ").str.strip()
    hit = tbl[labels.str.startswith(target)]
    if hit.empty:
        return None
    row = hit.iloc[0, 1:]
    idx = [_period_end(c) for c in row.index]
    s = pd.Series([_num(v) for v in row.values], index=idx)
    return s[s.index.notna()].sort_index()


def parse_company(symbol: str) -> dict | None:
    """Parse cached screener HTML -> structured fundamentals (period-indexed)."""
    p = CACHE / f"{symbol}.html"
    if not p.exists():
        return None
    tables = pd.read_html(StringIO(p.read_text()))

    q_eps = a_eps = fii = dii = None
    roe_3y = np.nan
    for t in tables:
        head = [str(c) for c in t.columns]
        col0 = t.iloc[:, 0].astype(str).str.replace("\xa0", " ").str.strip()
        months = {h.split()[0] for h in head if _period_end(h) is not pd.NaT}
        # EPS table: has an "EPS in Rs" row. Quarterly if it spans >1 month-of-year.
        if col0.str.startswith("EPS in Rs").any():
            eps = _row_by_label(t, "EPS in Rs")
            if eps is not None and len(eps):
                if len({m for m in months}) > 1:
                    q_eps = eps                  # quarterly (Mar/Jun/Sep/Dec)
                else:
                    a_eps = eps                  # annual (all March)
        # ROE compounded table
        if head and "Return on Equity" in head[0]:
            d = dict(zip(t.iloc[:, 0].astype(str).str.strip(),
                         t.iloc[:, 1].map(_num)))
            roe_3y = d.get("3 Years:", d.get("Last Year:", np.nan))
        # shareholding table
        if col0.str.startswith("FIIs").any():
            fii = _row_by_label(t, "FIIs")
            dii = _row_by_label(t, "DIIs")

    if q_eps is None:
        return None
    return {"symbol": symbol, "q_eps": q_eps, "a_eps": a_eps,
            "roe_3y": (roe_3y / 100.0 if pd.notna(roe_3y) else np.nan),
            "fii": fii, "dii": dii}


def asof_features(parsed: dict, asof, lag_days: int = LAG_DAYS) -> dict:
    """CAN SLIM fundamental features knowable at `asof` (look-ahead-safe).
    Returns eps_q_yoy, eps_q_accel, eps_3y_cagr, roe, inst_chg (fractions)."""
    asof = pd.Timestamp(asof)
    cutoff = asof - pd.Timedelta(days=lag_days)   # period_end must be <= this

    def avail(s):
        return None if s is None else s[s.index <= cutoff]

    q = avail(parsed["q_eps"])
    eps_q_yoy = eps_q_accel = eps_3y_cagr = inst_chg = np.nan
    if q is not None and len(q) >= 5:
        yoy = q.iloc[-1] / q.iloc[-5] - 1 if q.iloc[-5] > 0 else np.nan
        prev = q.iloc[-2] / q.iloc[-6] - 1 if len(q) >= 6 and q.iloc[-6] > 0 else np.nan
        eps_q_yoy, eps_q_accel = yoy, (yoy - prev if pd.notna(prev) else np.nan)

    a = avail(parsed["a_eps"])
    if a is not None and len(a) >= 4 and a.iloc[-4] > 0 and a.iloc[-1] > 0:
        eps_3y_cagr = (a.iloc[-1] / a.iloc[-4]) ** (1 / 3) - 1

    fii, dii = avail(parsed["fii"]), avail(parsed["dii"])
    if fii is not None and dii is not None and len(fii) >= 2 and len(dii) >= 2:
        inst = (fii.fillna(0) + dii.fillna(0))
        inst_chg = (inst.iloc[-1] - inst.iloc[-2]) / 100.0

    return {"eps_q_yoy": eps_q_yoy, "eps_q_accel": eps_q_accel,
            "eps_3y_cagr": eps_3y_cagr, "roe": parsed["roe_3y"],
            "inst_chg": inst_chg}


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 70)
    print("  fundamentals  —  real screener.in data (look-ahead-safe)")
    print("=" * 70)
    for sym in ("RELIANCE", "INFY", "TCS", "HDFCBANK"):
        fetch_company(sym)
        parsed = parse_company(sym)
        if not parsed:
            print(f"\n  {sym}: parse failed")
            continue
        print(f"\n  {sym}:  latest quarters of EPS available")
        print(f"    q_eps tail: {parsed['q_eps'].tail(5).round(2).to_dict()}")
        # features as-of a date deep in our backtest window
        f = asof_features(parsed, "2025-06-01")
        print(f"    AS-OF 2025-06-01 (uses only data >=45d old):")
        print(f"      EPS qtr YoY  : {f['eps_q_yoy']:.1%}" if pd.notna(f['eps_q_yoy']) else "      EPS qtr YoY  : n/a")
        print(f"      EPS 3y CAGR  : {f['eps_3y_cagr']:.1%}" if pd.notna(f['eps_3y_cagr']) else "      EPS 3y CAGR  : n/a")
        print(f"      ROE (3y)     : {f['roe']:.1%}" if pd.notna(f['roe']) else "      ROE (3y)     : n/a")
        print(f"      FII+DII QoQ  : {f['inst_chg']:+.2%}" if pd.notna(f['inst_chg']) else "      FII+DII QoQ  : n/a")
