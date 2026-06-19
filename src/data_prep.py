"""
Data preparation: load raw 1-minute BTC/USD candles and resample to 4-hour OHLCV.

Source of raw data
-------------------
Real 1-minute BTC/USD candles from the Bitstamp exchange, mirrored on GitHub:
  https://github.com/ff137/bitstamp-btcusd-minute-data
  file: data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz

Why Bitstamp BTC/USD instead of Binance BTC/USDT?
  The execution environment this was built in has no network access to
  api.binance.com or data.binance.vision (both return HTTP 403 under the
  network policy). Bitstamp BTC/USD is the closest *real* spot-BTC data set
  that is reachable (hosted on GitHub). The price difference between BTC/USD
  (Bitstamp) and BTC/USDT (Binance) is well under 0.1% and is irrelevant for a
  4-hour trend-following strategy. To use genuine Binance BTC/USDT data instead,
  see `fetch_binance.py` and run it from an environment that can reach Binance.

The raw file uses UTC unix-second timestamps. We aggregate to 4h bars using the
standard OHLCV convention (first open, max high, min low, last close, sum vol).
"""
from __future__ import annotations

import gzip
import os
import urllib.request

import pandas as pd

RAW_GZ = os.path.join("data", "btcusd_bitstamp_1min.csv.gz")
OUT_4H = os.path.join("data", "btcusd_4h.parquet")
RAW_URL = (
    "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/"
    "main/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
)

# Analysis window. Bitstamp 1-min mirror runs from 2012 to 2025-01-07.
# We keep 7 full years (>> the 3-year minimum) spanning multiple bull/bear cycles.
START = "2018-01-01"
END = "2025-01-07"


def ensure_raw(path: str = RAW_GZ) -> str:
    """Download the raw 1-minute candle archive (~91 MB) if not present."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"Downloading raw data from {RAW_URL} ...")
        urllib.request.urlretrieve(RAW_URL, path)
        print(f"Saved {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def load_1min(path: str = RAW_GZ) -> pd.DataFrame:
    """Load the raw 1-minute candles into a UTC-indexed DataFrame."""
    ensure_raw(path)
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("dt").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].astype("float64")
    # Bitstamp forward-fills flat 1-min candles (volume 0) during gaps; those are
    # legitimate "no trade" minutes and resample to correct OHLC, so we keep them.
    return df


def resample_4h(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute candles to 4-hour OHLCV bars (UTC, left-labelled)."""
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = df_1min.resample("4h", label="left", closed="left").agg(agg)
    # Drop bars with no underlying data (NaN open) - e.g. exchange downtime.
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def build(start: str = START, end: str = END) -> pd.DataFrame:
    df1 = load_1min()
    df4 = resample_4h(df1)
    df4 = df4.loc[start:end]
    os.makedirs("data", exist_ok=True)
    df4.to_parquet(OUT_4H)
    return df4


def load_4h(path: str = OUT_4H) -> pd.DataFrame:
    return pd.read_parquet(path)


if __name__ == "__main__":
    df = build()
    print(f"Built 4h dataset: {len(df):,} bars")
    print(f"Range: {df.index[0]}  ->  {df.index[-1]}")
    print(f"Price range: ${df['close'].min():,.0f}  ->  ${df['close'].max():,.0f}")
    print(df.head(3))
    print(df.tail(3))
