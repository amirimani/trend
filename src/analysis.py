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
from src.metrics import BARS_PER_YEAR, compute_metrics, periods_per_year, timeframe_hours
from src.strategy import Params

# Default grid for parameter selection. It searches the high-impact strategic
# choices: RSI filter on/off, exit style (fixed TP vs trailing vs partial), and
# the trend-regime filter. ~96 combinations.
DEFAULT_GRID = {
    "ema_fast": [20, 50],
    "ema_slow": [100, 200],
    "use_rsi_filter": [True, False],
    "exit_mode": ["fixed", "trailing", "partial"],
    "regime_filter": [False, True],
    "atr_sl_mult": [1.5, 2.5],
}


def grid_for(timeframe: str) -> dict:
    """Timeframe-aware grid: higher timeframes need FASTER EMA lookbacks (in
    bars) to generate enough trades to evaluate."""
    grid = dict(DEFAULT_GRID)
    if timeframe_hours(timeframe) >= 24:        # daily and above
        grid["ema_fast"] = [10, 20]
        grid["ema_slow"] = [50, 100]
    return grid


def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def evaluate(full_equity: pd.Series, all_trades: list, lo, hi,
             ppy: float = BARS_PER_YEAR) -> dict:
    """Slice a full-history equity curve / trade list to [lo, hi) and rebase."""
    lo, hi = _ts(lo), _ts(hi)
    eq = full_equity.loc[lo:hi]
    if len(eq) == 0:
        return compute_metrics(full_equity.iloc[:1] * 0 + 10_000.0, [], ppy=ppy)
    eq = eq / eq.iloc[0] * 10_000.0
    trades = [t for t in all_trades if lo <= t.entry_time < hi]
    return compute_metrics(eq, trades, ppy=ppy)


def grid_search(df: pd.DataFrame, costs: Costs, split, grid: dict | None = None,
                min_trades: int = 15, base: Params | None = None,
                ppy: float = BARS_PER_YEAR):
    """Select parameters robustly via IN-SAMPLE cross-validation.

    The in-sample window is split into two contiguous halves; a config is scored
    by the *minimum* of its Sharpe across the two halves (so it must work in
    BOTH sub-periods, not just on aggregate). This fights the over-fitting that
    a plain "best in-sample Sharpe" suffers — without ever touching the
    out-of-sample data. Returns (best_Params, is_metrics, oos_metrics, result)
    or None.
    """
    grid = grid or DEFAULT_GRID
    base = base or Params()
    keys = list(grid)

    split_ts = _ts(split)
    is_index = df.index[df.index < split_ts]
    mid = is_index[len(is_index) // 2] if len(is_index) >= 4 else split

    best = None
    for combo in itertools.product(*grid.values()):
        kw = {**vars(base), **dict(zip(keys, combo))}
        p = Params(**{k: v for k, v in kw.items() if k in vars(Params())})
        res = run_backtest(df, p, costs)
        is_m = evaluate(res.equity, res.trades, df.index[0], split, ppy)
        if is_m["num_trades"] < min_trades:
            continue
        sh1 = evaluate(res.equity, res.trades, df.index[0], mid, ppy)["sharpe"]
        sh2 = evaluate(res.equity, res.trades, mid, split, ppy)["sharpe"]
        # A half with no/too-few trades can't be trusted -> treat as poor.
        s1 = sh1 if sh1 == sh1 else -9.0
        s2 = sh2 if sh2 == sh2 else -9.0
        score = min(s1, s2)        # robust: worst sub-period must still be ok
        if best is None or score > best[0]:
            oos_m = evaluate(res.equity, res.trades, split, df.index[-1], ppy)
            best = (score, p, is_m, oos_m, res)
    if best is None:
        return None
    _, p, is_m, oos_m, res = best
    return p, is_m, oos_m, res


def analyze_symbol(df: pd.DataFrame, costs: Costs | None = None,
                   split_frac: float = 0.7, grid: dict | None = None,
                   timeframe: str = "4h") -> dict:
    """Auto-tune parameters for one symbol's history (any timeframe).

    Splits the data into in-sample (param selection) and out-of-sample
    (validation), grid-searches on IS with timeframe-correct annualisation, and
    returns the chosen parameters plus IS/OOS performance.
    """
    costs = costs or Costs()
    ppy = periods_per_year(timeframe)
    grid = grid or grid_for(timeframe)
    # Higher timeframes have fewer bars -> fewer signals; relax the trade floor.
    min_trades = 8 if timeframe_hours(timeframe) >= 24 else 15
    split_i = int(len(df) * split_frac)
    split = df.index[split_i]

    found = grid_search(df, costs, split, grid=grid, ppy=ppy, min_trades=min_trades)
    if found is None:
        # Fall back to defaults if nothing cleared the trade threshold.
        p = Params()
        res = run_backtest(df, p, costs)
        is_m = evaluate(res.equity, res.trades, df.index[0], split, ppy)
        oos_m = evaluate(res.equity, res.trades, split, df.index[-1], ppy)
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
