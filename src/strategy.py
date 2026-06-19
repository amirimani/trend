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

    # ---- Higher-timeframe (MTF) trend filter -------------------------------
    # Take the TREND from a higher timeframe and only allow entries that agree
    # with it (entries still trigger on this — the lower — timeframe).
    htf_filter: bool = False
    htf: str = ""                   # e.g. "1D"/"1W"; "" -> auto (one step up)
    htf_ema: int = 50               # HTF trend = close vs EMA(htf_ema) on the HTF

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


def _base_hours(idx: pd.DatetimeIndex) -> float:
    if len(idx) < 3:
        return 4.0
    return float(pd.Series(idx).diff().median().total_seconds() / 3600.0)


def _default_htf(base_hours: float) -> str:
    if base_hours < 24:
        return "1D"
    if base_hours < 168:
        return "1W"
    return "1MS"


def _rule_hours(rule: str) -> float:
    """Approximate hours per bar for a pandas resample rule (1D/1W/1MS/4H...)."""
    r = rule.upper()
    num = int("".join(c for c in r if c.isdigit()) or 1)
    if "MS" in r or r.endswith("M"):
        return num * 24 * 30.0
    if r.endswith("W"):
        return num * 24 * 7.0
    if r.endswith("D"):
        return num * 24.0
    if r.endswith("H"):
        return float(num)
    return 24.0


def _htf_trend(df: pd.DataFrame, p: Params):
    """Causal higher-timeframe trend (close > EMA) aligned to `df.index`.

    Resamples to the higher timeframe, computes the trend on HTF closes, then
    uses only the PREVIOUS completed HTF bar (shift(1)) before forward-filling
    onto the base bars — so a base bar never sees an unfinished HTF bar.
    Returns (trend_up_series, warmup_in_base_bars).
    """
    base_h = _base_hours(df.index)
    rule = p.htf or _default_htf(base_h)
    htf_close = df["close"].resample(rule).last().dropna()
    htf_ema = htf_close.ewm(span=p.htf_ema, adjust=False).mean()
    up_htf = (htf_close > htf_ema).shift(1)            # only completed HTF bars
    up = up_htf.reindex(df.index, method="ffill").fillna(False).astype(bool)
    ratio = max(1.0, _rule_hours(rule) / base_h)       # base bars per HTF bar
    warmup = int(p.htf_ema * 3 * ratio)                # let the HTF EMA settle
    return up, warmup


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

    # Higher-timeframe (MTF) trend filter (optional): align entries with the
    # trend on a higher timeframe. Causal — uses only completed HTF bars.
    htf_warmup = 0
    if p.htf_filter:
        up, htf_warmup = _htf_trend(out, p)
        regime_long = regime_long & up
        regime_short = regime_short & (~up)

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
                 p.regime_ema if p.regime_filter else 0, htf_warmup)
    out.iloc[:warmup, out.columns.get_indexer(
        ["long_entry", "long_exit", "short_entry", "short_exit"]
    )] = False
    return out
