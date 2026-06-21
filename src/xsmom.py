"""
Cross-sectional momentum (XS-MOM) backtest.

At each rebalance we rank the whole universe by recent momentum and hold an
equal-weight basket of the strongest coins (optionally short the weakest), then
rotate on the next rebalance. This captures *relative* strength across many
coins — a different (and historically more robust) edge than single-asset
timing, with few parameters (lookback, how many to hold, rebalance period).

Causality: the momentum at bar i uses closes up to and including i; the chosen
weights take effect on the NEXT bar's return, so there is no look-ahead.
"""
from __future__ import annotations

import pandas as pd


def build_panel(closes: dict[str, pd.Series]) -> pd.DataFrame:
    """closes: {symbol: close series}. Returns a date x symbol close panel."""
    return pd.DataFrame(closes).sort_index()


def xsmom_equity(panel: pd.DataFrame, lookback: int, top_k: int, rebalance: int,
                 fee: float = 0.0004, slippage: float = 0.0005,
                 long_short: bool = False, init: float = 10_000.0) -> pd.Series:
    """Equity curve of the cross-sectional momentum portfolio (equal weight)."""
    rets = panel.pct_change()
    cols = panel.columns
    prev_w = pd.Series(0.0, index=cols)
    eq = init
    last_rebal = -10 ** 9
    out = []
    cost_rate = fee + slippage
    for i in range(len(panel)):
        if i > 0:                                   # apply held weights to today's return
            r = rets.iloc[i].fillna(0.0)
            eq *= (1.0 + float((prev_w * r).sum()))
        if i >= lookback and (i - last_rebal) >= rebalance:
            mom = panel.iloc[i] / panel.iloc[i - lookback] - 1.0
            valid = mom.dropna()
            if len(valid) >= 1:
                k = min(top_k, len(valid))
                new_w = pd.Series(0.0, index=cols)
                new_w[valid.nlargest(k).index] = 1.0 / k
                if long_short and len(valid) >= 2 * k:
                    new_w[valid.nsmallest(k).index] = -1.0 / k
                turnover = float((new_w - prev_w).abs().sum())
                eq *= (1.0 - turnover * cost_rate)  # rebalancing cost on turnover
                prev_w = new_w
                last_rebal = i
        out.append(eq)
    return pd.Series(out, index=panel.index, name="xsmom")


def equal_weight_hold(panel: pd.DataFrame, init: float = 10_000.0) -> pd.Series:
    """Benchmark: equal-weight buy & hold of whatever coins exist at each time."""
    rets = panel.pct_change()
    eq, out = init, []
    for i in range(len(panel)):
        if i > 0:
            r = rets.iloc[i]
            out_r = r.mean(skipna=True)          # equal weight across available coins
            eq *= (1.0 + (0.0 if pd.isna(out_r) else float(out_r)))
        out.append(eq)
    return pd.Series(out, index=panel.index, name="equal_weight")
