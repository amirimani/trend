#!/usr/bin/env python3
"""
Collect fresh OHLCV history from Binance and save it into the repo so it can be
committed/pushed — then analysed elsewhere (e.g. in a sandbox without Binance
access).

Run this ON YOUR SERVER (it needs Binance connectivity + ccxt):

    pip install ccxt pandas
    python3 fetch_market_data.py \
        --symbols BTC/USDT,SOL/USDT,ETH/USDT,XRP/USDT \
        --timeframes 4h,1d --since 2019-01-01 --market future --push

Files are written to data/market/<SYMBOL>_<TF>.csv (UTC, last/forming candle
dropped) plus data/market/manifest.json. With --push it also commits and pushes
those data files to the current branch.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import pandas as pd

OUTDIR = os.path.join("data", "market")


def make_exchange(market: str):
    import ccxt
    if market == "future":
        return ccxt.binanceusdm({"enableRateLimit": True})   # USDT-M futures
    return ccxt.binance({"enableRateLimit": True})           # spot


def fetch_one(ex, symbol: str, timeframe: str, since: str) -> pd.DataFrame:
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    limit, rows, cursor = 1500, [], since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows += batch
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(ex.rateLimit / 1000.0)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype("float64")
    return df.sort_index().iloc[:-1]      # drop the still-forming last candle


def safe(symbol: str) -> str:
    return symbol.replace("/", "").replace(":", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTC/USDT,SOL/USDT,ETH/USDT,XRP/USDT")
    ap.add_argument("--timeframes", default="4h,1d")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--market", choices=["future", "spot"], default="future")
    ap.add_argument("--push", action="store_true", help="git add/commit/push the data")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    timeframes = [t.strip() for t in args.timeframes.split(",") if t.strip()]
    os.makedirs(OUTDIR, exist_ok=True)
    ex = make_exchange(args.market)

    manifest = {"fetched_at": datetime.now(timezone.utc).isoformat(),
                "market": args.market, "since": args.since, "files": {}}
    for sym in symbols:
        for tf in timeframes:
            try:
                df = fetch_one(ex, sym, tf, args.since)
            except Exception as e:
                print(f"!! {sym} {tf}: {e}")
                continue
            fn = f"{safe(sym)}_{tf}.csv"
            path = os.path.join(OUTDIR, fn)
            df.to_csv(path)
            manifest["files"][fn] = {
                "symbol": sym, "timeframe": tf, "rows": len(df),
                "start": str(df.index[0]), "end": str(df.index[-1]),
            }
            print(f"OK {sym} {tf}: {len(df):,} rows  {df.index[0].date()} -> {df.index[-1].date()}  ({fn})")

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote {len(manifest['files'])} files + manifest to {OUTDIR}/")

    if args.push:
        msg = f"data: refresh {args.market} market data {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z"
        for cmd in (["git", "add", OUTDIR],
                    ["git", "commit", "-m", msg],
                    ["git", "push"]):
            print("$", " ".join(cmd))
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
