"""
Strategy definition: EMA-cross trend following with an RSI entry filter and
ATR-based stop-loss / take-profit.

All indicators are causal: the value at bar t uses only data up to and including
bar t's *close*. Signals derived here are acted on by the backtest engine on the
*next* bar's open, which is what prevents look-ahead bias (see backtest.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Params:
    ema_fast: int = 50
    ema_slow: int = 200
    rsi_period: int = 14
    # Long entry requires RSI below this level: do not buy into an already
    # overbought market (the "avoid entering at saturation" filter).
    rsi_long_max: float = 70.0
    # Optional momentum floor for longs (RSI must also be above this).
    rsi_long_min: float = 50.0
    atr_period: int = 14
    atr_sl_mult: float = 2.0   # stop-loss distance = atr_sl_mult * ATR
    atr_tp_mult: float = 4.0   # take-profit distance = atr_tp_mult * ATR
    allow_short: bool = False  # spot BTC: long-only by default

    # ---- Entry options -----------------------------------------------------
    entry_mode: str = "ema_cross"   # "ema_cross" | "donchian"
    donchian_period: int = 20       # breakout lookback for donchian entries
    use_rsi_filter: bool = True     # apply the RSI entry band

    # ---- Trend regime filter ----------------------------------------------
    regime_filter: bool = False     # only go long above (short below) a long MA
    regime_ema: int = 200

    # ---- Exit options ------------------------------------------------------
    exit_mode: str = "fixed"        # "fixed" | "trailing" | "partial"
    trail_atr_mult: float = 3.0     # trailing-stop distance (trailing/partial)
    partial_tp_mult: float = 2.0    # first target (R multiple) for partial mode
    partial_tp_frac: float = 0.5    # fraction closed at the first target


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (causal)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    out = out.where(avg_loss != 0.0, 100.0)  # no losses -> RSI 100
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range (causal)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], p.ema_fast)
    out["ema_slow"] = ema(out["close"], p.ema_slow)
    out["rsi"] = rsi(out["close"], p.rsi_period)
    out["atr"] = atr(out, p.atr_period)
    if p.regime_filter:
        out["regime_ema"] = ema(out["close"], p.regime_ema)
    if p.entry_mode == "donchian":
        # Prior N-bar channel (shifted so the current bar isn't included).
        out["dc_high"] = out["high"].rolling(p.donchian_period).max().shift(1)
        out["dc_low"] = out["low"].rolling(p.donchian_period).min().shift(1)
    return out


def generate_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """
    Produce per-bar entry/exit *intentions* evaluated on closed bars.

    Columns added:
      long_entry  : EMA fast crosses above slow AND RSI within [min, max] band
      long_exit   : EMA fast crosses below slow (trend flip)
      short_entry : EMA fast crosses below slow AND RSI not oversold (if shorts on)
      short_exit  : EMA fast crosses above slow

    A "cross" at bar t means the relationship differs from bar t-1, so it depends
    only on closed bars t and t-1.
    """
    out = add_indicators(df, p)
    fast, slow = out["ema_fast"], out["ema_slow"]

    above = fast > slow
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_dn = ~above & above.shift(1, fill_value=False)

    if p.entry_mode == "donchian":
        # Breakout entries; trend-flip exit on the opposite channel break.
        long_raw = out["close"] > out["dc_high"]
        long_flip = out["close"] < out["dc_low"]
        short_raw = out["close"] < out["dc_low"]
        short_flip = out["close"] > out["dc_high"]
    else:  # ema_cross
        long_raw, long_flip = cross_up, cross_dn
        short_raw, short_flip = cross_dn, cross_up

    # RSI entry band (optional).
    if p.use_rsi_filter:
        rsi_ok_long = (out["rsi"] < p.rsi_long_max) & (out["rsi"] >= p.rsi_long_min)
        rsi_ok_short = (out["rsi"] > (100.0 - p.rsi_long_max)) & (
            out["rsi"] <= (100.0 - p.rsi_long_min))
    else:
        rsi_ok_long = rsi_ok_short = pd.Series(True, index=out.index)

    # Trend regime filter (optional): only long above / short below the long EMA.
    if p.regime_filter:
        regime_long = out["close"] > out["regime_ema"]
        regime_short = out["close"] < out["regime_ema"]
    else:
        regime_long = regime_short = pd.Series(True, index=out.index)

    out["long_entry"] = long_raw & rsi_ok_long & regime_long
    out["long_exit"] = long_flip

    if p.allow_short:
        out["short_entry"] = short_raw & rsi_ok_short & regime_short
        out["short_exit"] = short_flip
    else:
        out["short_entry"] = False
        out["short_exit"] = False

    # Indicators are undefined until the slow EMA / ATR / channel warm up;
    # suppress signals there so we never trade on half-formed indicators.
    warmup = max(p.ema_slow, p.atr_period, p.rsi_period, p.donchian_period,
                 p.regime_ema if p.regime_filter else 0)
    out.iloc[:warmup, out.columns.get_indexer(
        ["long_entry", "long_exit", "short_entry", "short_exit"]
    )] = False
    return out
