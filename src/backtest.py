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

    # Position state.
    in_pos = False
    side = 0           # +1 long, -1 short
    qty = 0.0
    entry_price = 0.0
    entry_time = None
    entry_bar = 0
    sl = tp = 0.0

    equity = np.empty(n, dtype="float64")
    trades: list[Trade] = []

    def buy_fill(price):  # adverse for entering long / covering short
        return price * (1.0 + slip)

    def sell_fill(price):  # adverse for exiting long / entering short
        return price * (1.0 - slip)

    for i in range(n):
        # ---- 1. Handle an open position: check exits on THIS bar -------------
        if in_pos:
            exit_price = None
            reason = None

            if side == 1:  # long: stop below, target above
                # Pessimistic ordering: stop checked before target.
                if l[i] <= sl:
                    # gap-through: fill at open if it opened below the stop
                    raw = min(o[i], sl) if o[i] <= sl else sl
                    exit_price = sell_fill(raw)
                    reason = "stop"
                elif h[i] >= tp:
                    raw = max(o[i], tp) if o[i] >= tp else tp
                    exit_price = sell_fill(raw)
                    reason = "target"
            else:          # short: stop above, target below
                if h[i] >= sl:
                    raw = max(o[i], sl) if o[i] >= sl else sl
                    exit_price = buy_fill(raw)
                    reason = "stop"
                elif l[i] <= tp:
                    raw = min(o[i], tp) if o[i] <= tp else tp
                    exit_price = buy_fill(raw)
                    reason = "target"

            # Signal-based exit (trend flip) executes at THIS bar's open, but only
            # if the SL/TP wasn't already hit on this bar.
            if exit_price is None:
                flip = long_exit[i] if side == 1 else short_exit[i]
                if flip:
                    exit_price = sell_fill(o[i]) if side == 1 else buy_fill(o[i])
                    reason = "signal"

            if exit_price is not None:
                if side == 1:
                    # exit fee already embedded in sell_fill? No - charge here.
                    cash = qty * exit_price * (1.0 - fee)
                else:
                    # close short: pnl = qty*(entry-exit), minus the exit fee.
                    # (entry fee was deducted from cash at entry.)
                    exit_fee = qty * exit_price * fee
                    cash = (
                        prev_cash_at_entry
                        + qty * (entry_price - exit_price)
                        - entry_fee
                        - exit_fee
                    )
                pnl = cash - prev_cash_at_entry
                net_ret = pnl / prev_cash_at_entry
                trades.append(
                    Trade(
                        side="long" if side == 1 else "short",
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=idx[i],
                        exit_price=exit_price,
                        exit_reason=reason,
                        ret=net_ret,
                        pnl=pnl,
                        bars_held=i - entry_bar,
                    )
                )
                in_pos = False
                side = 0
                qty = 0.0

        # ---- 2. Look for an entry to execute on THIS bar's open --------------
        # Entry signal was generated at bar i-1's close (acted on i's open).
        if not in_pos and i > 0:
            want_long = long_entry[i - 1]
            want_short = short_entry[i - 1] and p.allow_short
            a = atr[i - 1]
            if (want_long or want_short) and np.isfinite(a) and a > 0:
                prev_cash_at_entry = cash
                if want_long:
                    fill = buy_fill(o[i])
                    qty = (cash * (1.0 - fee)) / fill  # entry fee embedded
                    entry_fee = 0.0
                    side = 1
                    sl = fill - p.atr_sl_mult * a
                    tp = fill + p.atr_tp_mult * a
                else:
                    fill = sell_fill(o[i])
                    qty = cash / fill                  # notional == cash
                    entry_fee = qty * fill * fee       # charged explicitly
                    side = -1
                    sl = fill + p.atr_sl_mult * a
                    tp = fill - p.atr_tp_mult * a
                entry_price = fill
                entry_time = idx[i]
                entry_bar = i
                in_pos = True

        # ---- 3. Mark-to-market equity at THIS bar's close --------------------
        if in_pos:
            if side == 1:
                equity[i] = qty * c[i]
            else:
                # short: capital base + unrealised pnl - entry fee already paid
                equity[i] = prev_cash_at_entry + qty * (entry_price - c[i]) - entry_fee
        else:
            equity[i] = cash

    equity_s = pd.Series(equity, index=idx, name="equity")

    # Buy & hold benchmark (with one round-trip cost).
    bnh = costs.init_cash * (1.0 - fee) * (c / (c[0] * (1.0 + slip)))
    bnh_s = pd.Series(bnh, index=idx, name="buy_hold")

    return BacktestResult(equity=equity_s, trades=trades, bnh_equity=bnh_s)
