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
ALERT_ON_EXIT = _b("ALERT_ON_EXIT", True)
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
def evaluate_last_bar(df, p: Params, alert_on_exit: bool = True) -> dict | None:
    """Return an alert dict for the last closed bar, or None if no signal."""
    sig = generate_signals(df, p)
    last = sig.iloc[-1]
    ts = sig.index[-1]
    price = float(last["close"])
    atr = float(last["atr"])
    rsi = float(last["rsi"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    base = {
        "time": ts,
        "price": price,
        "atr": atr,
        "rsi": rsi,
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
    }

    if bool(last["long_entry"]):
        sl = price - p.atr_sl_mult * atr
        tp = price + p.atr_tp_mult * atr
        rr = (tp - price) / (price - sl) if price > sl else float("nan")
        return {**base, "kind": "entry", "side": "LONG", "sl": sl, "tp": tp, "rr": rr}

    if p.allow_short and bool(last["short_entry"]):
        sl = price + p.atr_sl_mult * atr
        tp = price - p.atr_tp_mult * atr
        rr = (price - tp) / (sl - price) if sl > price else float("nan")
        return {**base, "kind": "entry", "side": "SHORT", "sl": sl, "tp": tp, "rr": rr}

    if alert_on_exit and bool(last["long_exit"]):
        return {**base, "kind": "exit", "side": "LONG"}
    if alert_on_exit and p.allow_short and bool(last["short_exit"]):
        return {**base, "kind": "exit", "side": "SHORT"}

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


# --------------------------------------------------------------------------- #
# One polling cycle
# --------------------------------------------------------------------------- #
def check_once(p: Params, notifier: TelegramNotifier, state: dict) -> dict:
    warmup = max(p.ema_slow, p.atr_period, p.rsi_period) + 20
    df = fetch_recent(SYMBOL, TIMEFRAME, limit=max(400, warmup))
    if len(df) < warmup:
        log.warning("Not enough candles (%d) for warmup (%d).", len(df), warmup)
        return state

    last_ts = df.index[-1].isoformat()
    if state.get("last_bar") == last_ts:
        return state  # already processed this closed bar

    alert = evaluate_last_bar(df, p, ALERT_ON_EXIT)
    if alert is not None:
        msg = format_alert(alert, SYMBOL, TIMEFRAME)
        notifier.send(msg)
        log.info("Signal: %s %s @ %s", alert["kind"], alert["side"], last_ts)
    else:
        log.info("New closed bar %s — no signal.", last_ts)

    state["last_bar"] = last_ts
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
