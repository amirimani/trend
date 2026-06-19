"""
Live market monitor for the EMA-cross + RSI-filter + ATR-stop strategy.

It polls Binance for closed 4h candles and, when a fresh entry (or trend-flip
exit) signal appears on the just-closed bar, sends a Telegram alert describing
the proposed position: direction, entry, stop-loss, take-profit and R/R.

This is an ADVISORY service — it never places orders.

Look-ahead safety: signals are evaluated only on *closed* candles (the live,
still-forming candle is dropped by the feed), and each candle is alerted at most
once (deduplicated via a small state file on the mounted volume).

Config via environment variables (see .env.example):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  SYMBOL=BTC/USDT  TIMEFRAME=4h  POLL_SECONDS=300
  EMA_FAST EMA_SLOW RSI_PERIOD RSI_LONG_MIN RSI_LONG_MAX
  ATR_PERIOD ATR_SL_MULT ATR_TP_MULT ALLOW_SHORT ALERT_ON_EXIT
  STATE_FILE=/data/state.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.strategy import Params, generate_signals
from src.live.feed import fetch_recent
from src.live.notifier import TelegramNotifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("monitor")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _f(name, default):
    return float(os.getenv(name, default))


def _i(name, default):
    return int(os.getenv(name, default))


def _b(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def load_params() -> Params:
    return Params(
        ema_fast=_i("EMA_FAST", 50),
        ema_slow=_i("EMA_SLOW", 200),
        rsi_period=_i("RSI_PERIOD", 14),
        rsi_long_min=_f("RSI_LONG_MIN", 50),
        rsi_long_max=_f("RSI_LONG_MAX", 70),
        atr_period=_i("ATR_PERIOD", 14),
        atr_sl_mult=_f("ATR_SL_MULT", 2.0),
        atr_tp_mult=_f("ATR_TP_MULT", 4.0),
        allow_short=_b("ALLOW_SHORT", False),
    )


SYMBOL = os.getenv("SYMBOL", "BTC/USDT")
TIMEFRAME = os.getenv("TIMEFRAME", "4h")
POLL_SECONDS = _i("POLL_SECONDS", 300)
STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")


# --------------------------------------------------------------------------- #
# State (dedup across restarts)
# --------------------------------------------------------------------------- #
def load_state() -> dict:
    try:
        with open(STATE_FILE) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE) or ".", exist_ok=True)
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh)
    os.replace(tmp, STATE_FILE)


# --------------------------------------------------------------------------- #
# Signal evaluation on the most recently closed bar
# --------------------------------------------------------------------------- #
def signal_from_row(row, ts, p: Params, alert_on_exit: bool = True) -> dict | None:
    """Return an alert dict for a single (closed) signal row, or None."""
    price = float(row["close"])
    atr = float(row["atr"])
    rsi = float(row["rsi"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    base = {
        "time": ts,
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "ema_fast": float(row["ema_fast"]),
        "ema_slow": float(row["ema_slow"]),
    }

    if bool(row["long_entry"]):
        sl = price - p.atr_sl_mult * atr
        tp = price + p.atr_tp_mult * atr
        rr = (tp - price) / (price - sl) if price > sl else float("nan")
        return {**base, "kind": "entry", "side": "LONG", "sl": sl, "tp": tp, "rr": rr}

    if p.allow_short and bool(row["short_entry"]):
        sl = price + p.atr_sl_mult * atr
        tp = price - p.atr_tp_mult * atr
        rr = (price - tp) / (sl - price) if sl > price else float("nan")
        return {**base, "kind": "entry", "side": "SHORT", "sl": sl, "tp": tp, "rr": rr}

    if alert_on_exit and bool(row["long_exit"]):
        return {**base, "kind": "exit", "side": "LONG"}
    if alert_on_exit and p.allow_short and bool(row["short_exit"]):
        return {**base, "kind": "exit", "side": "SHORT"}

    return None


def evaluate_last_bar(df, p: Params, alert_on_exit: bool = True) -> dict | None:
    """Return an alert dict for the last closed bar, or None if no signal."""
    sig = generate_signals(df, p)
    return signal_from_row(sig.iloc[-1], sig.index[-1], p, alert_on_exit)


def check_exit(pos: dict, row, ts, p: Params) -> dict | None:
    """
    Has the tracked position `pos` been resolved on this closed bar?

    Mirrors the backtest exit rules: stop checked before target (pessimistic),
    gap-through fills at the bar open, and a trend-flip (opposite EMA cross)
    closes at the bar close. Returns an exit dict or None.
    """
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    sl, tp = pos["sl"], pos["tp"]

    if pos["side"] == "LONG":
        if l <= sl:
            px = o if o <= sl else sl       # gap-down through the stop
            return {"exit_price": px, "reason": "SL", "time": ts}
        if h >= tp:
            px = o if o >= tp else tp        # gap-up through the target
            return {"exit_price": px, "reason": "TP", "time": ts}
        if bool(row["long_exit"]):
            return {"exit_price": c, "reason": "EXIT", "time": ts}
    else:  # SHORT
        if h >= sl:
            px = o if o >= sl else sl
            return {"exit_price": px, "reason": "SL", "time": ts}
        if l <= tp:
            px = o if o <= tp else tp
            return {"exit_price": px, "reason": "TP", "time": ts}
        if bool(row["short_exit"]):
            return {"exit_price": c, "reason": "EXIT", "time": ts}
    return None


# --------------------------------------------------------------------------- #
# Message formatting (Persian)
# --------------------------------------------------------------------------- #
def format_alert(a: dict, symbol: str, timeframe: str) -> str:
    t = a["time"].strftime("%Y-%m-%d %H:%M UTC")
    if a["kind"] == "exit":
        emoji = "⚪️"
        head = f"{emoji} *سیگنال خروج / فلیپ روند* — `{symbol}` {timeframe}"
        body = (
            f"کراس معکوس EMA رخ داد؛ اگر پوزیشن *{a['side']}* باز داری، خروج/بستن رو در نظر بگیر.\n"
            f"قیمت فعلی: `${a['price']:,.0f}`"
        )
        meta = f"RSI: `{a['rsi']:.1f}` | EMA: `{a['ema_fast']:,.0f}/{a['ema_slow']:,.0f}`"
        return f"{head}\n⏰ بستن کندل: `{t}`\n\n{body}\n\n{meta}"

    is_long = a["side"] == "LONG"
    emoji = "🟢" if is_long else "🔴"
    word = "خرید (LONG)" if is_long else "فروش (SHORT)"
    entry, sl, tp = a["price"], a["sl"], a["tp"]
    tp_pct = (tp / entry - 1) * 100
    sl_pct = (sl / entry - 1) * 100
    return (
        f"{emoji} *سیگنال {word}* — `{symbol}` {timeframe}\n"
        f"⏰ بستن کندل: `{t}`\n\n"
        f"📍 ورود (در بازار/کندل بعد): `${entry:,.1f}`\n"
        f"🎯 حد سود (TP): `${tp:,.1f}`  ({tp_pct:+.2f}%)\n"
        f"🛑 حد ضرر (SL): `${sl:,.1f}`  ({sl_pct:+.2f}%)\n"
        f"⚖️ نسبت R/R: `{a['rr']:.2f}`\n\n"
        f"RSI: `{a['rsi']:.1f}` | EMA: `{a['ema_fast']:,.0f}/{a['ema_slow']:,.0f}` | "
        f"ATR: `{a['atr']:,.0f}`\n"
        f"_سیستم هشداردهنده است و سفارش ثبت نمی‌کند._"
    )


_REASON_FA = {
    "TP": "اصابت به حد سود (TP) 🎯",
    "SL": "اصابت به حد ضرر (SL) 🛑",
    "EXIT": "فلیپ روند / کراس معکوس EMA ⚪️",
}


def format_result(pos: dict, ex: dict, symbol: str, timeframe: str) -> str:
    """Announce the realised P/L of a closed position."""
    entry = pos["entry"]
    exit_px = ex["exit_price"]
    is_long = pos["side"] == "LONG"

    pnl_pct = (exit_px / entry - 1.0) * 100.0 if is_long else (entry / exit_px - 1.0) * 100.0
    risk = abs(entry - pos["sl"])
    reward = (exit_px - entry) if is_long else (entry - exit_px)
    r_mult = reward / risk if risk > 0 else float("nan")

    entry_ts = pd.Timestamp(pos["entry_time"])
    bars = max(1, round((ex["time"] - entry_ts).total_seconds() / (4 * 3600)))
    days = bars * 4 / 24

    win = pnl_pct >= 0
    head_emoji = "✅" if win else "❌"
    verb = "سود" if win else "ضرر"
    return (
        f"{head_emoji} *نتیجهٔ پوزیشن {pos['side']} بسته شد* — `{symbol}` {timeframe}\n"
        f"دلیل خروج: {_REASON_FA.get(ex['reason'], ex['reason'])}\n\n"
        f"📍 ورود: `${entry:,.1f}`\n"
        f"🚪 خروج: `${exit_px:,.1f}`\n"
        f"💰 {verb}: `{pnl_pct:+.2f}%`  (R: `{r_mult:+.2f}`)\n"
        f"🕒 مدت نگهداری: `{bars}` کندل (~`{days:.1f}` روز)\n"
        f"⏰ زمان خروج: `{ex['time'].strftime('%Y-%m-%d %H:%M UTC')}`\n"
        f"_سود/زیان بر مبنای قیمت و بدون احتساب کارمزد است._"
    )


# --------------------------------------------------------------------------- #
# One polling cycle
# --------------------------------------------------------------------------- #
def _register_position(state: dict, alert: dict) -> None:
    state["position"] = {
        "side": alert["side"],
        "entry": alert["price"],
        "sl": alert["sl"],
        "tp": alert["tp"],
        "rr": alert["rr"],
        "entry_time": alert["time"].isoformat(),
    }


def check_once(p: Params, notifier: TelegramNotifier, state: dict) -> dict:
    warmup = max(p.ema_slow, p.atr_period, p.rsi_period) + 20
    df = fetch_recent(SYMBOL, TIMEFRAME, limit=max(400, warmup))
    if len(df) < warmup:
        log.warning("Not enough candles (%d) for warmup (%d).", len(df), warmup)
        return state

    sig = generate_signals(df, p)
    latest_ts = sig.index[-1]

    # First run ever: don't replay history, just mark the latest bar.
    if "last_bar" not in state:
        state["last_bar"] = latest_ts.isoformat()
        save_state(state)
        log.info("Initialised at %s (no historical replay).", latest_ts)
        return state

    last_seen = pd.Timestamp(state["last_bar"])
    new_bars = sig.index[sig.index > last_seen]
    if len(new_bars) == 0:
        return state

    for ts in new_bars:
        row = sig.loc[ts]

        # 1) Resolve an open tracked position on this bar.
        closed_this_bar = False
        pos = state.get("position")
        if pos:
            ex = check_exit(pos, row, ts, p)
            if ex is not None:
                notifier.send(format_result(pos, ex, SYMBOL, TIMEFRAME))
                log.info("Closed %s via %s @ %s (entry %s)",
                         pos["side"], ex["reason"], ts, pos["entry"])
                state["position"] = None
                closed_this_bar = True

        # 2) If flat, look for a new entry to open (and track) a position.
        #    A trend-flip exit is reported when it actually closes a tracked
        #    position (step 1, reason EXIT), so we don't emit standalone
        #    "consider exiting" noise when already flat.
        _ = closed_this_bar  # (kept for clarity; entries can follow a reversal)
        if not state.get("position"):
            s = signal_from_row(row, ts, p, alert_on_exit=False)
            if s and s["kind"] == "entry":
                notifier.send(format_alert(s, SYMBOL, TIMEFRAME))
                _register_position(state, s)
                log.info("Entry %s @ %s price %s", s["side"], ts, s["price"])

    state["last_bar"] = latest_ts.isoformat()
    save_state(state)
    return state


def main():
    p = load_params()
    notifier = TelegramNotifier()
    state = load_state()

    log.info("Live monitor starting: %s %s | params=%s | poll=%ss | shorts=%s",
             SYMBOL, TIMEFRAME, p, POLL_SECONDS, p.allow_short)
    notifier.send(
        f"🤖 *ربات رصد بازار فعال شد*\n`{SYMBOL}` {TIMEFRAME} | "
        f"EMA `{p.ema_fast}/{p.ema_slow}` | RSI `{p.rsi_period}` | "
        f"SL `{p.atr_sl_mult}×ATR` / TP `{p.atr_tp_mult}×ATR`\n"
        f"شروع: `{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}`"
    )

    while True:
        try:
            state = check_once(p, notifier, state)
        except Exception as e:  # never let the loop die on a transient error
            log.exception("Cycle error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        p = load_params()
        check_once(p, TelegramNotifier(), load_state())
    else:
        main()
