"""
================================================================================
 pipeline/load_bhavcopy.py  —  cached CSVs  ->  wide close / value / deliv frames
================================================================================
 Reads every cached full-bhavcopy CSV and pivots it into the three price/flow
 DataFrames the engine's panel needs:

   close  [dates x symbols]  CLOSE_PRICE   (Rs)
   value  [dates x symbols]  traded value  (Rs)   <- TURNOVER_LACS * 1e5
   deliv  [dates x symbols]  delivery frac [0,1]   <- DELIV_PER / 100

 Filtering discipline:
   - SERIES == 'EQ' only  (cash-equity delivery segment; drops bonds/ETFs/SME).
   - Columns carry leading spaces in NSE's header -> we strip them.
   - DELIV_PER is blank for a few illiquid names / special sessions -> NaN.

 Output columns are SYMBOL strings here (human-readable). build_panel.py later
 maps them to the integer ids the engine indexes by.
================================================================================
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "data" / "bhavcopy"


def _read_one(path: Path) -> pd.DataFrame:
    """Parse a single cached bhavcopy CSV into tidy long rows (EQ only)."""
    df = pd.read_csv(path, skipinitialspace=True)   # kills the leading spaces
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"].copy()

    df["date"] = pd.to_datetime(df["DATE1"].str.strip(), format="%d-%b-%Y")
    df["symbol"] = df["SYMBOL"].str.strip()
    df["close"] = pd.to_numeric(df["CLOSE_PRICE"], errors="coerce")
    # PREV_CLOSE is NSE's OFFICIAL adjusted prior close -> drives split/bonus adj
    df["prev_close"] = pd.to_numeric(df["PREV_CLOSE"], errors="coerce")
    # TURNOVER_LACS is rupees-in-lakhs -> rupees
    df["value"] = pd.to_numeric(df["TURNOVER_LACS"], errors="coerce") * 1e5
    # DELIV_PER is a percentage -> fraction; blanks become NaN
    df["deliv"] = pd.to_numeric(df["DELIV_PER"], errors="coerce") / 100.0
    return df[["date", "symbol", "close", "prev_close", "value", "deliv"]]


def load_frames(cache: Path = CACHE) -> dict:
    """Load all cached CSVs -> {'close','value','deliv'} wide DataFrames
    indexed by date, columns = SYMBOL. Dates sorted ascending."""
    files = sorted(cache.glob("sec_bhavdata_full_*.csv"))
    if not files:
        raise FileNotFoundError(f"no bhavcopy CSVs in {cache} — run fetch_bhavcopy first")

    long = pd.concat([_read_one(f) for f in files], ignore_index=True)
    long = long.drop_duplicates(["date", "symbol"], keep="last")

    frames = {}
    for col in ("close", "prev_close", "value", "deliv"):
        frames[col] = (long.pivot(index="date", columns="symbol", values=col)
                           .sort_index())
    return frames


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print("=" * 68)
    print("  load_bhavcopy  —  pivot cached CSVs into close/value/deliv")
    print("=" * 68)
    f = load_frames()
    c, v, d = f["close"], f["value"], f["deliv"]
    print(f"\n  shape: {c.shape[0]} dates x {c.shape[1]} EQ symbols")
    print(f"  dates: {c.index.min().date()} -> {c.index.max().date()}")

    for sym in ("RELIANCE", "TATAMOTORS", "INFY"):
        if sym in c.columns:
            print(f"\n  {sym}:")
            print(f"    close : {c[sym].round(2).tolist()}")
            print(f"    value : {(v[sym] / 1e7).round(2).tolist()}  (Rs crore)")
            print(f"    deliv : {(d[sym] * 100).round(1).tolist()}  (%)")

    miss_deliv = d.isna().sum().sum() / d.size
    print(f"\n  delivery NaN fraction: {miss_deliv:.2%} "
          f"(blank for a few illiquid names — expected)")
