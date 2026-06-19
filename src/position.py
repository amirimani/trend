"""
Shared position / exit logic — the single source of truth used by BOTH the
offline backtest engine and the live monitor, so live signals exactly match
what was backtested.

A *position* is a plain JSON-serialisable dict (so the live monitor can persist
it across restarts). `open_position` builds it; `step` advances it by one closed
bar and returns the exit "legs" that occurred on that bar.

Exit modes (Params.exit_mode):
  "fixed"    : fixed ATR stop-loss + fixed ATR take-profit (the original).
  "trailing" : ATR stop-loss, no fixed target; a chandelier trailing stop
               ratchets up (long) from the highest high since entry, letting
               winners run until the trend reverses.
  "partial"  : take `partial_tp_frac` at `partial_tp_mult`×ATR, move the stop to
               break-even, then trail the remainder by `trail_atr_mult`×ATR.

Conventions (mirror the backtest's pessimism):
  * The stop is checked before the target on the same bar.
  * A gap through a level fills at the bar open.
  * A trend-flip exits the remainder at the bar close.
  * The trailing stop is updated from this bar's extreme AFTER exits are checked
    (so it only affects subsequent bars — no look-ahead).

`step` returns a list of legs: {"frac", "price", "reason"} where `price` is the
RAW level (the caller applies slippage and fees) and `reason` ∈
{"SL","TP","TP1","TRAIL","EXIT"}. `frac` is the fraction of the ORIGINAL size.
"""
from __future__ import annotations

from .strategy import Params


def open_position(side: int, entry: float, atr: float, p: Params,
                  maint_margin: float = 0.005) -> dict:
    """side: +1 long, -1 short. `entry` is the (slippage-adjusted) fill price.

    Computes the (isolated-margin) liquidation price from leverage so the
    backtest/live can model getting liquidated before the stop when leverage is
    high relative to the stop distance.
    """
    pos = {
        "side": int(side),
        "entry": float(entry),
        "atr": float(atr),
        "remaining": 1.0,
        "anchor": float(entry),          # running extreme for trailing
        "mode": p.exit_mode,
        "trailing_engaged": False,
        "partial_done": p.exit_mode != "partial",
    }
    if side == 1:
        pos["stop"] = entry - p.atr_sl_mult * atr
        pos["tp"] = entry + p.atr_tp_mult * atr if p.exit_mode == "fixed" else None
        pos["t1"] = entry + p.partial_tp_mult * atr if p.exit_mode == "partial" else None
    else:
        pos["stop"] = entry + p.atr_sl_mult * atr
        pos["tp"] = entry - p.atr_tp_mult * atr if p.exit_mode == "fixed" else None
        pos["t1"] = entry - p.partial_tp_mult * atr if p.exit_mode == "partial" else None

    lev = max(1.0, float(getattr(p, "leverage", 1.0)))
    if lev > 1.0:
        dist = max(0.0, 1.0 / lev - maint_margin)   # adverse move that wipes margin
        pos["liq"] = entry * (1.0 - dist) if side == 1 else entry * (1.0 + dist)
    else:
        pos["liq"] = None
    return pos


def _trails(pos: dict, p: Params) -> bool:
    return pos["mode"] == "trailing" or (pos["mode"] == "partial" and pos["partial_done"])


def step(pos: dict, o: float, h: float, l: float, c: float,
         flip: bool, p: Params) -> list[dict]:
    """Advance the open position by one closed bar; return the exit legs."""
    events: list[dict] = []
    stop = pos["stop"]
    liq = pos.get("liq")
    rem = pos["remaining"]

    if pos["side"] == 1:  # ---- LONG ----
        # Liquidation bites before the stop only if it sits ABOVE the stop.
        loss_level = stop if liq is None else max(stop, liq)
        if l <= loss_level:
            px = min(o, loss_level) if o <= loss_level else loss_level   # gap at open
            is_liq = liq is not None and liq > stop
            reason = "LIQ" if is_liq else ("TRAIL" if pos["trailing_engaged"] else "SL")
            events.append({"frac": rem, "price": px, "reason": reason})
            pos["remaining"] = 0.0
            return events
        if pos["mode"] == "fixed" and h >= pos["tp"]:
            px = max(o, pos["tp"]) if o >= pos["tp"] else pos["tp"]
            events.append({"frac": rem, "price": px, "reason": "TP"})
            pos["remaining"] = 0.0
            return events
        if pos["mode"] == "partial" and not pos["partial_done"] and h >= pos["t1"]:
            px = max(o, pos["t1"]) if o >= pos["t1"] else pos["t1"]
            events.append({"frac": p.partial_tp_frac, "price": px, "reason": "TP1"})
            pos["remaining"] = rem - p.partial_tp_frac
            pos["partial_done"] = True
            pos["stop"] = max(pos["stop"], pos["entry"])       # move to break-even
        if flip and pos["remaining"] > 0:
            events.append({"frac": pos["remaining"], "price": c, "reason": "EXIT"})
            pos["remaining"] = 0.0
            return events
        if _trails(pos, p):                                    # update trail (next bar)
            pos["anchor"] = max(pos["anchor"], h)
            new_stop = pos["anchor"] - p.trail_atr_mult * pos["atr"]
            if new_stop > pos["stop"]:
                pos["stop"] = new_stop
                pos["trailing_engaged"] = True
    else:                 # ---- SHORT ----
        loss_level = stop if liq is None else min(stop, liq)
        if h >= loss_level:
            px = max(o, loss_level) if o >= loss_level else loss_level
            is_liq = liq is not None and liq < stop
            reason = "LIQ" if is_liq else ("TRAIL" if pos["trailing_engaged"] else "SL")
            events.append({"frac": rem, "price": px, "reason": reason})
            pos["remaining"] = 0.0
            return events
        if pos["mode"] == "fixed" and l <= pos["tp"]:
            px = min(o, pos["tp"]) if o <= pos["tp"] else pos["tp"]
            events.append({"frac": rem, "price": px, "reason": "TP"})
            pos["remaining"] = 0.0
            return events
        if pos["mode"] == "partial" and not pos["partial_done"] and l <= pos["t1"]:
            px = min(o, pos["t1"]) if o <= pos["t1"] else pos["t1"]
            events.append({"frac": p.partial_tp_frac, "price": px, "reason": "TP1"})
            pos["remaining"] = rem - p.partial_tp_frac
            pos["partial_done"] = True
            pos["stop"] = min(pos["stop"], pos["entry"])
        if flip and pos["remaining"] > 0:
            events.append({"frac": pos["remaining"], "price": c, "reason": "EXIT"})
            pos["remaining"] = 0.0
            return events
        if _trails(pos, p):
            pos["anchor"] = min(pos["anchor"], l)
            new_stop = pos["anchor"] + p.trail_atr_mult * pos["atr"]
            if new_stop < pos["stop"]:
                pos["stop"] = new_stop
                pos["trailing_engaged"] = True
    return events
