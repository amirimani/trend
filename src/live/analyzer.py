"""
Live auto-tuning engine.

Given a symbol, it fetches that symbol's full 4h history from Binance, runs the
in-sample / out-of-sample grid search (src.analysis) to derive the best signal
parameters, and returns a summary. The monitor saves the chosen params into the
watchlist so the symbol is henceforth monitored with its own tuned settings.

`fetch_fn` is injected so this is testable offline with local data.
"""
from __future__ import annotations

from src.analysis import analyze_symbol
from src.backtest import Costs
from src.strategy import Params


def run_analysis(symbol: str, fetch_fn, timeframe: str = "4h",
                 since: str = "2020-01-01", costs: Costs | None = None,
                 base: Params | None = None) -> dict:
    """Fetch history for `symbol` and auto-tune its parameters.

    `base` carries account settings kept fixed during the search (leverage,
    whether shorts are allowed). Returns the analyze_symbol() summary plus the
    symbol. Raises on data errors (caller reports the failure to the user).
    """
    df = fetch_fn(symbol, timeframe, since)
    warmup = 220  # need enough bars for EMA200 + a meaningful sample
    if df is None or len(df) < warmup + 200:
        raise ValueError(
            f"دادهٔ کافی برای {symbol} نیست (فقط {0 if df is None else len(df)} کندل)."
        )
    summary = analyze_symbol(df, costs=costs, timeframe=timeframe, base=base)
    summary["symbol"] = symbol
    summary["timeframe"] = timeframe
    return summary
