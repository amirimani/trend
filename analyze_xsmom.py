#!/usr/bin/env python3
"""
Cross-sectional momentum analysis over the collected universe (data/market).

Builds a close panel from all <SYMBOL>_<TF>.csv files, grid-searches the momentum
parameters on the in-sample half, and reports the out-of-sample result vs two
benchmarks: equal-weight buy & hold of the universe, and BTC buy & hold. Few
parameters -> low over-fitting risk.

    python3 analyze_xsmom.py --tf 1d
    python3 analyze_xsmom.py --tf 1d --long-short
"""
from __future__ import annotations

import argparse
import glob
import itertools
import os
import re

import pandas as pd

from src.metrics import compute_metrics, periods_per_year
from src.xsmom import build_panel, equal_weight_hold, xsmom_equity

MKT = os.path.join("data", "market")


def _load_closes(tf: str) -> dict:
    closes = {}
    for path in sorted(glob.glob(os.path.join(MKT, f"*_{tf}.csv"))):
        m = re.match(rf"(.+)_{re.escape(tf)}\.csv$", os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path, parse_dates=[0], index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        closes[m.group(1)] = df["close"].astype("float64").sort_index()
    return closes


def _slice_metrics(eq: pd.Series, lo, hi, ppy) -> dict:
    s = eq.loc[lo:hi]
    s = s / s.iloc[0] * 10_000.0
    return compute_metrics(s, [], ppy=ppy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1d")
    ap.add_argument("--long-short", action="store_true")
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--funding", type=float, default=0.0,
                    help="avg perpetual funding per 8h (e.g. 0.0001); longs pay / shorts receive")
    args = ap.parse_args()

    closes = _load_closes(args.tf)
    if len(closes) < 3:
        print(f"Need more coins in {MKT}/ (found {len(closes)}). Fetch a larger "
              f"universe with fetch_market_data.py first.")
        return
    panel = build_panel(closes)
    ppy = periods_per_year(args.tf)
    daily = args.tf == "1d"
    bar_hours = 24.0 if args.tf == "1d" else (168.0 if args.tf == "1w" else 4.0)
    n = len(panel)
    split = panel.index[int(n * 0.7)]
    print(f"Universe: {len(closes)} coins | {args.tf} | {panel.index[0].date()} -> "
          f"{panel.index[-1].date()} | IS/OOS split {str(split)[:10]}")

    # Small parameter grid (few knobs -> low over-fit risk).
    lbs = [30, 60, 90] if daily else [120, 240, 360]
    rbs = [7, 14, 30] if daily else [42, 84]
    ks = [3, 5, 8]
    best = None
    for lb, k, rb in itertools.product(lbs, ks, rbs):
        if k > len(closes):
            continue
        eq = xsmom_equity(panel, lb, k, rb, args.fee, args.slippage, args.long_short,
                          funding_8h=args.funding, bar_hours=bar_hours)
        is_m = _slice_metrics(eq, panel.index[0], split, ppy)
        if is_m["sharpe"] == is_m["sharpe"] and (best is None or is_m["sharpe"] > best[0]):
            best = (is_m["sharpe"], lb, k, rb, eq)

    if best is None:
        print("No valid parameter set.")
        return
    _, lb, k, rb, eq = best
    is_m = _slice_metrics(eq, panel.index[0], split, ppy)
    oos = _slice_metrics(eq, split, panel.index[-1], ppy)

    ew = equal_weight_hold(panel)
    ew_oos = _slice_metrics(ew, split, panel.index[-1], ppy)
    btc = next((c for c in panel.columns if c.startswith("BTC")), None)
    btc_oos = None
    if btc is not None:
        btc_oos = _slice_metrics(panel[btc].dropna(), split, panel.index[-1], ppy)

    def line(name, m):
        return (f"  {name:24s} ret {m['total_return']*100:+8.1f}% | CAGR {m['cagr']*100:+6.1f}% | "
                f"Sharpe {m['sharpe']:.2f} | maxDD {m['max_drawdown']*100:6.1f}%")

    print(f"\nBest params: lookback={lb}  top_k={k}  rebalance={rb}  "
          f"{'(long/short)' if args.long_short else '(long-only)'}")
    print(f"IS  Sharpe {is_m['sharpe']:.2f}")
    print("\nOUT-OF-SAMPLE:")
    print(line("XS-Momentum", oos))
    print(line("Equal-weight hold", ew_oos))
    if btc_oos:
        print(line("BTC buy & hold", btc_oos))


if __name__ == "__main__":
    main()
