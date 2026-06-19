"""
Reusable analysis logic shared by the offline backtest (`run_backtest.py`) and
the live auto-tuning engine (`src/live/analyzer.py`).

Kept inside the package so it ships in the Docker image (the Dockerfile copies
`src/` only). No plotting / no I/O here.
"""
from __future__ import annotations

import itertools
import math

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
    "htf_filter": [False, True],   # HTF trend gate (regime_filter dropped: redundant)
    "atr_sl_mult": [1.5, 2.5],
}


def grid_for(timeframe: str, allow_short: bool = False) -> dict:
    """Timeframe-aware grid: higher timeframes need FASTER EMA lookbacks (in
    bars) to generate enough trades to evaluate. On futures (allow_short) the
    search also decides per-coin whether enabling shorts helps out-of-sample."""
    grid = dict(DEFAULT_GRID)
    if timeframe_hours(timeframe) >= 24:        # daily and above
        grid["ema_fast"] = [10, 20]
        grid["ema_slow"] = [50, 100]
    if allow_short:
        grid["allow_short"] = [False, True]
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
                min_trades: int = 20, base: Params | None = None,
                ppy: float = BARS_PER_YEAR, half_min: int = 6):
    """Select parameters with strict anti-over-fitting gates (no OOS leakage).

    The in-sample window is split into two contiguous halves; a config is kept
    only if it:
      * trades enough overall (`min_trades`) AND in each half (`half_min`);
      * is profitable (positive Sharpe AND positive return) in BOTH halves
        (consistency gate); and
      * clears a sample-size- & trials-deflated Sharpe hurdle:
            score = min(half Sharpes) - 0.5*sqrt(ln(#combos)) / sqrt(#trades)
        which must be > 0 (so a config "found" among many trials on few trades
        is penalised the way a Deflated Sharpe Ratio would penalise it).
    If nothing clears the bar, returns None (caller falls back to defaults
    rather than shipping an over-fit config).
    """
    grid = grid or DEFAULT_GRID
    base = base or Params()
    keys = list(grid)
    combos = list(itertools.product(*grid.values()))
    trials_penalty = 0.5 * math.sqrt(math.log(max(2, len(combos))))

    split_ts = _ts(split)
    is_index = df.index[df.index < split_ts]
    mid = is_index[len(is_index) // 2] if len(is_index) >= 4 else split

    best = None
    for combo in combos:
        kw = {**vars(base), **dict(zip(keys, combo))}
        p = Params(**{k: v for k, v in kw.items() if k in vars(Params())})
        res = run_backtest(df, p, costs)
        is_m = evaluate(res.equity, res.trades, df.index[0], split, ppy)
        nt = is_m["num_trades"]
        if nt < min_trades:
            continue
        m1 = evaluate(res.equity, res.trades, df.index[0], mid, ppy)
        m2 = evaluate(res.equity, res.trades, mid, split, ppy)
        if m1["num_trades"] < half_min or m2["num_trades"] < half_min:
            continue
        sh1, sh2 = m1["sharpe"], m2["sharpe"]
        if not (sh1 == sh1 and sh2 == sh2):
            continue
        # Consistency gate: must be profitable in BOTH in-sample halves.
        if sh1 <= 0 or sh2 <= 0 or m1["total_return"] <= 0 or m2["total_return"] <= 0:
            continue
        score = min(sh1, sh2) - trials_penalty / math.sqrt(nt)   # deflated hurdle
        if score <= 0:                                           # didn't clear it
            continue
        if best is None or score > best[0]:
            oos_m = evaluate(res.equity, res.trades, split, df.index[-1], ppy)
            best = (score, p, is_m, oos_m, res)
    if best is None:
        return None
    _, p, is_m, oos_m, res = best
    return p, is_m, oos_m, res


def analyze_symbol(df: pd.DataFrame, costs: Costs | None = None,
                   split_frac: float = 0.7, grid: dict | None = None,
                   timeframe: str = "4h", base: Params | None = None) -> dict:
    """Auto-tune parameters for one symbol's history (any timeframe).

    Splits the data into in-sample (param selection) and out-of-sample
    (validation), grid-searches on IS with timeframe-correct annualisation, and
    returns the chosen parameters plus IS/OOS performance. `base` carries the
    user settings that are NOT searched (e.g. leverage, and whether shorts are
    permitted at all on this account).
    """
    costs = costs or Costs()
    base = base or Params()
    ppy = periods_per_year(timeframe)
    grid = grid or grid_for(timeframe, allow_short=base.allow_short)
    # Trade floors: enough samples to trust the stats (anti-over-fit). Daily has
    # fewer bars, but we still demand a meaningful sample or we fall back.
    daily = timeframe_hours(timeframe) >= 24
    min_trades = 12 if daily else 20
    half_min = 4 if daily else 6
    split_i = int(len(df) * split_frac)
    split = df.index[split_i]

    found = grid_search(df, costs, split, grid=grid, ppy=ppy,
                        min_trades=min_trades, half_min=half_min, base=base)
    if found is None:
        # Fall back to the base settings if nothing cleared the trade threshold.
        p = base
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
