"""
================================================================================
 pipeline/fetch_bhavcopy.py  —  NSE "full bhavcopy with delivery" downloader
================================================================================
 The single richest free EOD source for Indian cash equity. One CSV per trading
 day gives, per security: OHLC, close, traded quantity, TURNOVER (Rs), number of
 trades, delivered quantity and DELIVERY %. That covers three of the five panel
 pieces at once:  close  ->  value  ->  deliv.

 WHY THIS SOURCE (see CLAUDE.md discipline):
   - Survivorship-free BY CONSTRUCTION. Each day's file is a snapshot of what
     actually traded THAT day, so delisted/crashed names appear on their historic
     dates and silently drop out afterwards. We never use "today's index list".
   - Look-ahead-safe: it is the official EOD file, knowable only after close.

 Source URL (current NSE archive host, 2024+):
   https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv

 We cache every raw CSV under data/bhavcopy/ and NEVER re-download what is on
 disk (incremental). Non-trading days (weekends/holidays) 404 -> recorded as
 empty sentinels so we don't retry them every run.

 Stdlib only (urllib) -> no extra dependencies. Be polite: small delay, retries.
================================================================================
"""
from __future__ import annotations
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import date, timedelta

BASE = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv"
CACHE = Path(__file__).resolve().parent.parent / "data" / "bhavcopy"
HOLIDAY_MARK = ".holiday"   # sentinel suffix for known non-trading days

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}


def _cache_path(d: date) -> Path:
    return CACHE / f"sec_bhavdata_full_{d:%d%m%Y}.csv"


def _holiday_path(d: date) -> Path:
    return CACHE / f"sec_bhavdata_full_{d:%d%m%Y}{HOLIDAY_MARK}"


def fetch_day(d: date, retries: int = 3, pause: float = 0.6) -> Path | None:
    """Download one day's full bhavcopy. Returns the cached CSV path, or None if
    that date is a non-trading day (no file published). Cached + sentinel-aware:
    a second call for the same date does no network I/O."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = _cache_path(d)
    if out.exists():
        return out                      # already have it
    if _holiday_path(d).exists():
        return None                     # known non-trading day

    url = BASE.format(ddmmyyyy=f"{d:%d%m%Y}")
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            # sanity: a real file starts with the SYMBOL header
            if not data[:6].upper().startswith(b"SYMBOL"):
                raise ValueError("unexpected body (not a bhavcopy)")
            out.write_bytes(data)
            return out
        except urllib.error.HTTPError as e:
            if e.code == 404:           # weekend / market holiday
                _holiday_path(d).touch()
                return None
            if attempt == retries:
                raise
        except Exception:
            if attempt == retries:
                raise
        time.sleep(pause * attempt)     # backoff
    return None


def fetch_range(start: date, end: date, pause: float = 0.6,
                verbose: bool = True) -> list[Path]:
    """Fetch every trading day in [start, end]. Skips weekends up front and any
    date already cached. Returns the list of CSV paths actually available."""
    paths: list[Path] = []
    d = start
    n_dl = n_cache = n_hol = 0
    while d <= end:
        if d.weekday() >= 5:            # Sat/Sun: skip without hitting NSE
            d += timedelta(days=1)
            continue
        had = _cache_path(d).exists()
        p = fetch_day(d, pause=pause)
        if p is not None:
            paths.append(p)
            if had:
                n_cache += 1
            else:
                n_dl += 1
                if verbose:
                    print(f"  downloaded {d:%Y-%m-%d}")
                time.sleep(pause)       # be polite to NSE only on real fetches
        else:
            n_hol += 1
        d += timedelta(days=1)
    if verbose:
        print(f"\n  range {start} -> {end}: {len(paths)} trading days "
              f"({n_dl} new, {n_cache} cached, {n_hol} non-trading)")
    return paths


if __name__ == "__main__":
    # SMOKE TEST: pull one recent week and show what we got.
    print("=" * 68)
    print("  fetch_bhavcopy smoke test  —  one week of NSE full bhavcopy")
    print("=" * 68)
    paths = fetch_range(date(2025, 6, 2), date(2025, 6, 8))
    print(f"\n  cached files now in {CACHE}:")
    for p in sorted(CACHE.glob('sec_bhavdata_full_*.csv')):
        print(f"    {p.name:40s} {p.stat().st_size:>8,} bytes")
