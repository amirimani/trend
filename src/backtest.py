"""
Event-driven backtest engine (long/short, single asset, full-equity sizing).

Look-ahead safety
------------------
* Indicators and signals are computed on closed bars (see strategy.py).
* A signal observed at the close of bar t is executed at the OPEN of bar t+1.
* Stop-loss / take-profit levels are fixed at entry (from that bar's ATR) and are
  only ever checked against *future* bars' high/low.

Costs
-----
* `fee` is charged on every fill (entry and exit), as a fraction of notional.
* `slippage` worsens every fill price (buys fill higher, sells fill lower).

Intrabar SL/TP resolution
--------------------------
If a single bar's range touches both the stop and the target, we conservatively
assume the STOP filled first (pessimistic). Gap-through fills at the open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .position import open_position, step
from .strategy import Params, generate_signals


@dataclass
class Costs:
    fee: float = 0.001      # 0.10% per side (Binance spot taker)
    slippage: float = 0.0005  # 0.05% per side
    init_cash: float = 10_000.0


@dataclass
class Trade:
    side: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    ret: float          # net return on the trade (fraction, incl. costs)
    pnl: float          # cash pnl
    bars_held: int


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade] = field(default_factory=list)
    bnh_equity: pd.Series | None = None


def run_backtest(df: pd.DataFrame, p: Params, costs: Costs) -> BacktestResult:
    sig = generate_signals(df, p)

    o = sig["open"].to_numpy()
    h = sig["high"].to_numpy()
    l = sig["low"].to_numpy()
    c = sig["close"].to_numpy()
    atr = sig["atr"].to_numpy()
    idx = sig.index

    long_entry = sig["long_entry"].to_numpy()
    long_exit = sig["long_exit"].to_numpy()
    short_entry = sig["short_entry"].to_numpy()
    short_exit = sig["short_exit"].to_numpy()

    fee, slip = costs.fee, costs.slippage
    cash = costs.init_cash
    n = len(df)

    # Position state (supports partial exits via `pos["remaining"]`).
    pos = None
    qty0 = 0.0            # units opened at entry (full size)
    entry_price = 0.0
    entry_time = None
    entry_bar = 0
    entry_capital = 0.0
    realized_cash = 0.0   # long: proceeds banked from exited legs
    realized_pnl = 0.0    # short: pnl banked from covered legs
    entry_fee = 0.0
    last_reason = None

    equity = np.empty(n, dtype="float64")
    trades: list[Trade] = []

    def buy_fill(price):   # adverse for entering long / covering short
        return price * (1.0 + slip)

    def sell_fill(price):  # adverse for exiting long / entering short
        return price * (1.0 - slip)

    for i in range(n):
        # ---- 1. Handle an open position: process exit legs on THIS bar -------
        if pos is not None:
            flip = bool(long_exit[i]) if pos["side"] == 1 else bool(short_exit[i])
            for leg in step(pos, o[i], h[i], l[i], c[i], flip, p):
                units = leg["frac"] * qty0
                last_reason = leg["reason"]
                if pos["side"] == 1:
                    realized_cash += units * sell_fill(leg["price"]) * (1.0 - fee)
                else:
                    px = buy_fill(leg["price"])
                    realized_pnl += units * (entry_price - px) - units * px * fee
            if pos["remaining"] <= 1e-12:    # fully closed -> record the trade
                if pos["side"] == 1:
                    cash = realized_cash
                else:
                    cash = entry_capital - entry_fee + realized_pnl
                pnl = cash - entry_capital
                trades.append(Trade(
                    side="long" if pos["side"] == 1 else "short",
                    entry_time=entry_time, entry_price=entry_price,
                    exit_time=idx[i], exit_price=sell_fill(c[i]) if pos["side"] == 1 else buy_fill(c[i]),
                    exit_reason=last_reason, ret=pnl / entry_capital, pnl=pnl,
                    bars_held=i - entry_bar,
                ))
                pos = None

        # ---- 2. Look for an entry to execute on THIS bar's open --------------
        # Entry signal was generated at bar i-1's close (acted on i's open).
        if pos is None and i > 0:
            want_long = bool(long_entry[i - 1])
            want_short = bool(short_entry[i - 1]) and p.allow_short
            a = atr[i - 1]
            if (want_long or want_short) and np.isfinite(a) and a > 0:
                entry_capital = cash
                if want_long:
                    entry_price = buy_fill(o[i])
                    qty0 = (cash * (1.0 - fee)) / entry_price  # entry fee embedded
                    entry_fee = 0.0
                    realized_cash = 0.0
                    pos = open_position(1, entry_price, a, p)
                else:
                    entry_price = sell_fill(o[i])
                    qty0 = cash / entry_price                  # notional == cash
                    entry_fee = qty0 * entry_price * fee       # charged explicitly
                    realized_pnl = 0.0
                    pos = open_position(-1, entry_price, a, p)
                entry_time = idx[i]
                entry_bar = i

        # ---- 3. Mark-to-market equity at THIS bar's close --------------------
        if pos is not None:
            rem_units = pos["remaining"] * qty0
            if pos["side"] == 1:
                equity[i] = realized_cash + rem_units * c[i]
            else:
                equity[i] = (entry_capital - entry_fee + realized_pnl
                             + rem_units * (entry_price - c[i]))
        else:
            equity[i] = cash

    equity_s = pd.Series(equity, index=idx, name="equity")

    # Buy & hold benchmark (with one round-trip cost).
    bnh = costs.init_cash * (1.0 - fee) * (c / (c[0] * (1.0 + slip)))
    bnh_s = pd.Series(bnh, index=idx, name="buy_hold")

    return BacktestResult(equity=equity_s, trades=trades, bnh_equity=bnh_s)
