"""
OPTIONAL: fetch genuine Binance BTC/USDT 4h klines via ccxt.

This is the data source the task originally asked for. It is NOT used by the
default pipeline because the environment this project was built in cannot reach
api.binance.com / data.binance.vision (HTTP 403 under the network policy). Run
this from any machine with Binance connectivity to produce `data/btcusd_4h.parquet`
in the exact same schema the backtest expects, then run `run_backtest.py`.

    pip install ccxt pandas pyarrow
    python3 -m src.fetch_binance --since 2018-01-01

The backtest code is data-source agnostic: it only needs a 4h OHLCV DataFrame.
"""
from __future__ import annotations

import argparse
import os
import time

import pandas as pd

OUT_4H = os.path.join("data", "btcusd_4h.parquet")


def fetch(symbol: str = "BTC/USDT", timeframe: str = "4h", since: str = "2018-01-01") -> pd.DataFrame:
    import ccxt  # imported lazily so the default pipeline has no ccxt dependency

    ex = ccxt.binance({"enableRateLimit": True})
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    limit = 1000
    all_rows: list[list] = []
    cursor = since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows += batch
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(all_rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype("float64")
    return df.sort_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--since", default="2018-01-01")
    args = ap.parse_args()

    df = fetch(args.symbol, args.timeframe, args.since)
    os.makedirs("data", exist_ok=True)
    df.to_parquet(OUT_4H)
    print(f"Fetched {len(df):,} {args.timeframe} bars: {df.index[0]} -> {df.index[-1]}")
    print(f"Saved {OUT_4H}")
