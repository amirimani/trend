"""Live market data feed: recent OHLCV from Binance via ccxt.

Returns a UTC-indexed 4h OHLCV DataFrame in the same schema the strategy uses.
The most recent (still-forming) candle is dropped so we only ever act on
*closed* bars — this is what keeps the live signals look-ahead-free, exactly
like the backtest.
"""
from __future__ import annotations

import pandas as pd


def fetch_recent(symbol: str = "BTC/USDT", timeframe: str = "4h", limit: int = 400) -> pd.DataFrame:
    """Fetch the last `limit` candles. Drops the final, not-yet-closed candle."""
    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype("float64")
    df = df.sort_index()
    # The last row is the current, incomplete candle -> drop it.
    return df.iloc[:-1]


def fetch_history(symbol: str = "BTC/USDT", timeframe: str = "4h",
                  since: str = "2020-01-01") -> pd.DataFrame:
    """Fetch the full 4h history (paged) for auto-tuning a symbol's parameters.

    Drops the final, not-yet-closed candle. Used by the /analyze engine.
    """
    import time as _t

    import ccxt

    ex = ccxt.binance({"enableRateLimit": True})
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    limit, rows, cursor = 1000, [], since_ms
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        rows += batch
        cursor = batch[-1][0] + 1
        if len(batch) < limit:
            break
        _t.sleep(ex.rateLimit / 1000.0)

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="ts")
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype("float64")
    return df.sort_index().iloc[:-1]
