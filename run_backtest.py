"""
End-to-end backtest of the EMA-cross + RSI-filter + ATR-stop strategy on real
4-hour BTC data.

Methodology
-----------
* Indicators are causal and signals execute on the next bar's open (no look-ahead).
* Fees (0.10%/side) and slippage (0.05%/side) are charged on every fill.
* The data is split into IN-SAMPLE (param selection) and OUT-OF-SAMPLE (held out).
* A small grid search picks parameters by in-sample Sharpe, then we report how
  those exact parameters perform on the untouched out-of-sample period.

Run:  python3 run_backtest.py
"""
from __future__ import annotations

import itertools
import json
import os

import pandas as pd

from src.backtest import Costs, run_backtest
from src.data_prep import load_4h
from src.metrics import compute_metrics, format_metrics
from src.strategy import Params

SPLIT = "2023-01-01"   # data before -> in-sample, on/after -> out-of-sample
OUTDIR = "results"


def _ts(x):
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def evaluate(full_equity, all_trades, lo, hi):
    """Slice the full-history equity/trades to [lo, hi) and rebase for metrics."""
    lo, hi = _ts(lo), _ts(hi)
    eq = full_equity.loc[lo:hi]
    eq = eq / eq.iloc[0] * 10_000.0  # rebase to the standard starting cash
    trades = [t for t in all_trades if lo <= t.entry_time < hi]
    return compute_metrics(eq, trades)


def run_one(df, p, costs):
    res = run_backtest(df, p, costs)
    return res


def grid_search(df, costs, split):
    """Optimise parameters on the in-sample window only (by Sharpe)."""
    grid = {
        "ema_fast": [20, 50],
        "ema_slow": [100, 200],
        "rsi_long_max": [65, 75],
        "atr_sl_mult": [1.5, 2.5],
        "atr_tp_mult": [3.0, 5.0],
    }
    keys = list(grid)
    best = None
    is_end = pd.Timestamp(split, tz="UTC")
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        p = Params(**kw)
        res = run_backtest(df, p, costs)
        is_m = evaluate(res.equity, res.trades, df.index[0], split)
        # require a minimum trade count so we don't "win" on 2 lucky trades
        if is_m["num_trades"] < 15:
            continue
        score = is_m["sharpe"]
        if score == score and (best is None or score > best[0]):
            best = (score, p, is_m, res)
    return best


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = load_4h()
    costs = Costs()

    print(f"Loaded {len(df):,} 4h bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f"In-sample : {df.index[0].date()} .. {SPLIT}")
    print(f"Out-sample: {SPLIT} .. {df.index[-1].date()}")
    print(f"Costs: fee={costs.fee*100:.3f}%/side  slippage={costs.slippage*100:.3f}%/side")

    # ---- Default parameters --------------------------------------------------
    default_p = Params()
    res_def = run_backtest(df, default_p, costs)
    full_def = compute_metrics(res_def.equity, res_def.trades)
    is_def = evaluate(res_def.equity, res_def.trades, df.index[0], SPLIT)
    oos_def = evaluate(res_def.equity, res_def.trades, SPLIT, df.index[-1])
    bnh_full = compute_metrics(res_def.bnh_equity, [])

    section("DEFAULT PARAMS  (EMA 50/200, RSI<70 & >=50, SL 2*ATR, TP 4*ATR)")
    print("\n[Full period]\n" + format_metrics(full_def))
    print("\n[In-sample]\n" + format_metrics(is_def))
    print("\n[Out-of-sample]\n" + format_metrics(oos_def))
    print("\n[Buy & Hold, full period]\n" + format_metrics(bnh_full))

    # ---- Grid search on in-sample, validate on out-of-sample ----------------
    section("GRID SEARCH  (selected on IN-SAMPLE Sharpe, validated OUT-OF-SAMPLE)")
    best = grid_search(df, costs, SPLIT)
    if best is None:
        print("No parameter set met the minimum-trade constraint.")
        chosen = default_p
        res_best = res_def
    else:
        score, chosen, is_m, res_best = best
        print(f"Best in-sample params: {chosen}")
        print(f"In-sample Sharpe: {score:.2f}")

    is_best = evaluate(res_best.equity, res_best.trades, df.index[0], SPLIT)
    oos_best = evaluate(res_best.equity, res_best.trades, SPLIT, df.index[-1])
    full_best = compute_metrics(res_best.equity, res_best.trades)
    print("\n[In-sample]\n" + format_metrics(is_best))
    print("\n[Out-of-sample  <-- the honest test]\n" + format_metrics(oos_best))

    # ---- Persist results -----------------------------------------------------
    out = {
        "split": SPLIT,
        "costs": vars(costs),
        "default_params": vars(default_p),
        "chosen_params": vars(chosen),
        "default": {"full": full_def, "in_sample": is_def, "out_sample": oos_def},
        "chosen": {"full": full_best, "in_sample": is_best, "out_sample": oos_best},
        "buy_hold_full": bnh_full,
    }
    with open(os.path.join(OUTDIR, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    res_best.equity.to_frame().assign(buy_hold=res_def.bnh_equity).to_csv(
        os.path.join(OUTDIR, "equity_curve.csv")
    )

    # trade log
    rows = [vars(t) for t in res_best.trades]
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "trades.csv"), index=False)

    _plot(res_best.equity, res_def.bnh_equity, SPLIT)
    print(f"\nSaved metrics.json, equity_curve.csv, trades.csv, equity_curve.png in {OUTDIR}/")


def _plot(equity, bnh, split):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(equity.index, equity.values, label="Strategy", lw=1.3)
    ax[0].plot(bnh.index, bnh.values, label="Buy & Hold", lw=1.0, alpha=0.7)
    ax[0].axvline(pd.Timestamp(split, tz="UTC"), color="k", ls="--", lw=1, label="IS/OOS split")
    ax[0].set_yscale("log")
    ax[0].set_title("Equity curve (log scale) - BTC 4h EMA/RSI/ATR strategy vs Buy & Hold")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    dd = equity / equity.cummax() - 1.0
    ax[1].fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.4)
    ax[1].set_title("Strategy drawdown (%)"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "equity_curve.png"), dpi=110)


if __name__ == "__main__":
    main()
