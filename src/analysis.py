"""
Reusable analysis logic shared by the offline backtest (`run_backtest.py`) and
the live auto-tuning engine (`src/live/analyzer.py`).

Kept inside the package so it ships in the Docker image (the Dockerfile copies
`src/` only). No plotting / no I/O here.
"""
from __future__ import annotations

import itertools

import pandas as pd

from src.backtest import Costs, run_backtest
from src.metrics import compute_metrics
from src.strategy import Params

# Default grid for parameter selection. ~32 combinations -> a few seconds.
DEFAULT_GRID = {
    "ema_fast": [20, 50],
    "ema_slow": [100, 200],
    "rsi_long_max": [65, 75],
    "atr_sl_mult": [1.5, 2.5],
    "atr_tp_mult": [3.0, 5.0],
}


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def evaluate(full_equity: pd.Series, all_trades: list, lo, hi) -> dict:
    """Slice a full-history equity curve / trade list to [lo, hi) and rebase."""
    lo, hi = _ts(lo), _ts(hi)
    eq = full_equity.loc[lo:hi]
    if len(eq) == 0:
        return compute_metrics(full_equity.iloc[:1] * 0 + 10_000.0, [])
    eq = eq / eq.iloc[0] * 10_000.0
    trades = [t for t in all_trades if lo <= t.entry_time < hi]
    return compute_metrics(eq, trades)


def grid_search(df: pd.DataFrame, costs: Costs, split, grid: dict | None = None,
                min_trades: int = 15, base: Params | None = None):
    """Select parameters by IN-SAMPLE Sharpe (with a minimum-trade guard).

    Returns (best_Params, is_metrics, oos_metrics, full_result) or None.
    """
    grid = grid or DEFAULT_GRID
    base = base or Params()
    keys = list(grid)
    best = None
    for combo in itertools.product(*grid.values()):
        kw = {**vars(base), **dict(zip(keys, combo))}
        # Params only accepts its declared fields:
        p = Params(**{k: v for k, v in kw.items() if k in vars(Params())})
        res = run_backtest(df, p, costs)
        is_m = evaluate(res.equity, res.trades, df.index[0], split)
        if is_m["num_trades"] < min_trades:
            continue
        score = is_m["sharpe"]
        if score == score and (best is None or score > best[0]):
            oos_m = evaluate(res.equity, res.trades, split, df.index[-1])
            best = (score, p, is_m, oos_m, res)
    if best is None:
        return None
    _, p, is_m, oos_m, res = best
    return p, is_m, oos_m, res


def analyze_symbol(df: pd.DataFrame, costs: Costs | None = None,
                   split_frac: float = 0.7, grid: dict | None = None) -> dict:
    """Auto-tune parameters for one symbol's 4h history.

    Splits the data into in-sample (param selection) and out-of-sample
    (validation), grid-searches on IS, and returns the chosen parameters plus
    IS/OOS performance — exactly the methodology used in run_backtest.py.
    """
    costs = costs or Costs()
    split_i = int(len(df) * split_frac)
    split = df.index[split_i]

    found = grid_search(df, costs, split, grid=grid)
    if found is None:
        # Fall back to defaults if nothing cleared the trade threshold.
        p = Params()
        res = run_backtest(df, p, costs)
        is_m = evaluate(res.equity, res.trades, df.index[0], split)
        oos_m = evaluate(res.equity, res.trades, split, df.index[-1])
        tuned = False
    else:
        p, is_m, oos_m, res = found
        tuned = True

    return {
        "params": {k: v for k, v in vars(p).items()},
        "tuned": tuned,
        "split": str(split),
        "n_bars": int(len(df)),
        "range": [str(df.index[0]), str(df.index[-1])],
        "in_sample": is_m,
        "out_sample": oos_m,
    }
