#!/usr/bin/env python3
"""
Analyse the real market data collected by fetch_market_data.py.

Reads every data/market/<SYMBOL>_<TF>.csv and, for each, runs the honest
walk-forward (aggregated out-of-sample expectancy) plus a single IS/OOS split,
for both strategy families (trend and hold). Prints a comparison so you can see
which coins/timeframes/strategies actually have a positive average edge.

    python3 analyze_market.py                 # all files
    python3 analyze_market.py --tf 4h         # only 4h files
    python3 analyze_market.py --leverage 1 --short
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

from src.analysis import analyze_symbol, walk_forward
from src.backtest import Costs
from src.strategy import Params

MKT = os.path.join("data", "market")
COLS = ["open", "high", "low", "close", "volume"]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[COLS].astype("float64").sort_index()


def parse_name(fn: str):
    m = re.match(r"(.+)_([0-9]+[mhdwM]+)\.csv$", os.path.basename(fn))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default=None, help="only this timeframe (e.g. 4h)")
    ap.add_argument("--leverage", type=float, default=1.0)
    ap.add_argument("--short", action="store_true", help="allow shorts (futures)")
    ap.add_argument("--fee", type=float, default=0.0004, help="per-side taker fee")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(MKT, "*.csv")))
    if not files:
        print(f"No data in {MKT}/. Run fetch_market_data.py on the server first.")
        return
    costs = Costs(fee=args.fee)

    print(f"{'symbol':12s} {'tf':4s} {'strat':6s} {'WF exp/trade':>12s} {'WF win':>7s} "
          f"{'WF total':>9s} {'OOS Sharpe':>11s} {'verdict':>8s}")
    print("-" * 78)
    for path in files:
        sym, tf = parse_name(path)
        if sym is None or (args.tf and tf != args.tf):
            continue
        df = load(path)
        disp = sym.replace("USDT", "/USDT")
        for strat in ("trend", "hold"):
            base = Params(strategy=strat, leverage=args.leverage,
                          allow_short=(args.short and strat == "trend"))
            try:
                wf = walk_forward(df, costs, tf, base=base)
                s = analyze_symbol(df, costs, timeframe=tf, base=base)
            except Exception as e:
                print(f"{disp:12s} {tf:4s} {strat:6s}  error: {e}")
                continue
            exp = wf.get("expectancy")
            exp_s = f"{exp*100:+.2f}%" if exp is not None else "n/a"
            win_s = f"{wf['win_rate']*100:.0f}%" if wf.get("trades") else "—"
            tot_s = f"{wf['total_return']*100:+.0f}%" if wf.get("trades") else "—"
            oos = s["out_sample"]["sharpe"]
            mark = "POS" if (exp is not None and exp > 0) else "neg"
            print(f"{disp:12s} {tf:4s} {strat:6s} {exp_s:>12s} {win_s:>7s} {tot_s:>9s} "
                  f"{oos:>11.2f} {mark:>8s}")


if __name__ == "__main__":
    main()
