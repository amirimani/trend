"""
Multi-symbol live market monitor — Telegram-managed.

For every *enabled* symbol in the watchlist it polls Binance for closed 4h
candles and, when a fresh entry signal appears, sends a Telegram alert with the
proposed position (entry / SL / TP / R-R). It then tracks that position and,
when it resolves (TP, SL or trend-flip), sends a follow-up result with the
realised P/L and R. Each symbol uses its own parameters.

Symbols are managed live from Telegram (/add /remove /enable /disable) and each
can be auto-tuned with /analyze (the engine fetches that symbol's history and
grid-searches its best parameters). State (watchlist, params, positions,
history) is persisted on the mounted volume.

This is an ADVISORY service — it never places orders.

Config via environment variables (see .env.example):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  WATCHLIST=BTC/USDT,SOL/USDT,ETH/USDT   (seeded on first run)
  TIMEFRAME=4h  POLL_SECONDS=300  ANALYZE_SINCE=2020-01-01
  EMA_FAST EMA_SLOW RSI_PERIOD RSI_LONG_MIN RSI_LONG_MAX
  ATR_PERIOD ATR_SL_MULT ATR_TP_MULT ALLOW_SHORT
  STATE_FILE=/data/state.json
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.live.analyzer import run_analysis
from src.live.feed import fetch_history, fetch_recent
from src.live.notifier import TelegramNotifier
from src.live import state as st
from src.strategy import Params, generate_signals

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
    """The default/seed parameters for newly added symbols (until /analyze)."""
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


TIMEFRAME = os.getenv("TIMEFRAME", "4h")
POLL_SECONDS = _i("POLL_SECONDS", 300)
ANALYZE_SINCE = os.getenv("ANALYZE_SINCE", "2020-01-01")
WATCHLIST_SEED = [s for s in os.getenv("WATCHLIST", "BTC/USDT").split(",") if s.strip()]
# Quality guard: a symbol whose out-of-sample Sharpe is below this fails
# validation and is auto-disabled by /analyze. OOS Sharpe in [MIN, WEAK) is
# kept but flagged as weak.
MIN_OOS_SHARPE = _f("MIN_OOS_SHARPE", 0.0)
WEAK_OOS_SHARPE = _f("WEAK_OOS_SHARPE", 0.5)
# Weekly performance report: sent once per ISO week, on/after REPORT_DAY
# (0=Mon) at REPORT_HOUR UTC. Also available on demand via /summary.
WEEKLY_REPORT = _b("WEEKLY_REPORT", True)
REPORT_DAY = _i("REPORT_DAY", 0)
REPORT_HOUR = _i("REPORT_HOUR", 9)


# --------------------------------------------------------------------------- #
# Shared runtime context (one per process)
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    notifier: TelegramNotifier
    state: dict
    default_params: Params
    timeframe: str = TIMEFRAME
    lock: threading.Lock = field(default_factory=threading.Lock)
    runtime: dict = field(default_factory=lambda: {
        "started_at": datetime.now(timezone.utc).isoformat(), "symbols": {}})
    analyzing: set = field(default_factory=set)
    # injected so tests can substitute local data
    fetch_recent_fn: callable = fetch_recent
    fetch_history_fn: callable = fetch_history

    def start_analysis(self, symbol: str) -> str:
        symbol = st.normalize_symbol(symbol)
        if symbol in self.analyzing:
            return f"⏳ تحلیل `{symbol}` همین الان در حال اجراست."
        self.analyzing.add(symbol)
        threading.Thread(target=_analysis_worker, args=(self, symbol), daemon=True).start()
        return f"🔬 تحلیل `{symbol}` شروع شد؛ نتیجه را به‌زودی می‌فرستم…"


# --------------------------------------------------------------------------- #
# Signal evaluation on a single closed bar
# --------------------------------------------------------------------------- #
def signal_from_row(row, ts, p: Params, alert_on_exit: bool = True) -> dict | None:
    price = float(row["close"])
    atr = float(row["atr"])
    rsi = float(row["rsi"])
    if not np.isfinite(atr) or atr <= 0:
        return None
    base = {
        "time": ts, "price": price, "atr": atr, "rsi": rsi,
        "ema_fast": float(row["ema_fast"]), "ema_slow": float(row["ema_slow"]),
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


def check_exit(pos: dict, row, ts, p: Params) -> dict | None:
    """Mirror the backtest exit rules (stop-before-target, gap-through at open)."""
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    sl, tp = pos["sl"], pos["tp"]
    if pos["side"] == "LONG":
        if l <= sl:
            return {"exit_price": (o if o <= sl else sl), "reason": "SL", "time": ts}
        if h >= tp:
            return {"exit_price": (o if o >= tp else tp), "reason": "TP", "time": ts}
        if bool(row["long_exit"]):
            return {"exit_price": c, "reason": "EXIT", "time": ts}
    else:
        if h >= sl:
            return {"exit_price": (o if o >= sl else sl), "reason": "SL", "time": ts}
        if l <= tp:
            return {"exit_price": (o if o <= tp else tp), "reason": "TP", "time": ts}
        if bool(row["short_exit"]):
            return {"exit_price": c, "reason": "EXIT", "time": ts}
    return None


def realised(pos: dict, ex: dict) -> dict:
    entry, exit_px = pos["entry"], ex["exit_price"]
    is_long = pos["side"] == "LONG"
    pnl_pct = (exit_px / entry - 1.0) * 100.0 if is_long else (entry / exit_px - 1.0) * 100.0
    risk = abs(entry - pos["sl"])
    reward = (exit_px - entry) if is_long else (entry - exit_px)
    r_mult = reward / risk if risk > 0 else float("nan")
    bars = max(1, round((ex["time"] - pd.Timestamp(pos["entry_time"])).total_seconds() / (4 * 3600)))
    return {
        "side": pos["side"], "entry": entry, "exit": exit_px, "reason": ex["reason"],
        "pnl_pct": pnl_pct, "r": r_mult, "bars": bars,
        "entry_time": pos["entry_time"], "exit_time": ex["time"].isoformat(),
    }


def _position_dict(alert: dict) -> dict:
    return {
        "side": alert["side"], "entry": alert["price"], "sl": alert["sl"],
        "tp": alert["tp"], "rr": alert["rr"], "entry_time": alert["time"].isoformat(),
    }


# --------------------------------------------------------------------------- #
# Message formatting (Persian)
# --------------------------------------------------------------------------- #
def fmt_price(x: float) -> str:
    """Readable price across scales: BTC ($102,300), SOL ($148.20), DOGE ($0.1234)."""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:,.2f}"
    if ax >= 0.01:
        return f"{x:.4f}"
    return f"{x:.6f}"


def format_alert(a: dict, symbol: str, timeframe: str) -> str:
    t = a["time"].strftime("%Y-%m-%d %H:%M UTC")
    is_long = a["side"] == "LONG"
    emoji = "🟢" if is_long else "🔴"
    word = "خرید (LONG)" if is_long else "فروش (SHORT)"
    entry, sl, tp = a["price"], a["sl"], a["tp"]
    return (
        f"{emoji} *سیگنال {word}* — `{symbol}` {timeframe}\n"
        f"⏰ بستن کندل: `{t}`\n\n"
        f"📍 ورود (در بازار/کندل بعد): `${fmt_price(entry)}`\n"
        f"🎯 حد سود (TP): `${fmt_price(tp)}`  ({(tp/entry-1)*100:+.2f}%)\n"
        f"🛑 حد ضرر (SL): `${fmt_price(sl)}`  ({(sl/entry-1)*100:+.2f}%)\n"
        f"⚖️ نسبت R/R: `{a['rr']:.2f}`\n\n"
        f"RSI: `{a['rsi']:.1f}` | EMA: `{fmt_price(a['ema_fast'])}/{fmt_price(a['ema_slow'])}`\n"
        f"_سیستم هشداردهنده است و سفارش ثبت نمی‌کند._"
    )


_REASON_FA = {
    "TP": "اصابت به حد سود (TP) 🎯",
    "SL": "اصابت به حد ضرر (SL) 🛑",
    "EXIT": "فلیپ روند / کراس معکوس EMA ⚪️",
}


def format_result(pos: dict, ex: dict, symbol: str, timeframe: str) -> str:
    r = realised(pos, ex)
    days = r["bars"] * 4 / 24
    win = r["pnl_pct"] >= 0
    return (
        f"{'✅' if win else '❌'} *نتیجهٔ پوزیشن {pos['side']} بسته شد* — `{symbol}` {timeframe}\n"
        f"دلیل خروج: {_REASON_FA.get(ex['reason'], ex['reason'])}\n\n"
        f"📍 ورود: `${fmt_price(r['entry'])}`\n"
        f"🚪 خروج: `${fmt_price(r['exit'])}`\n"
        f"💰 {'سود' if win else 'ضرر'}: `{r['pnl_pct']:+.2f}%`  (R: `{r['r']:+.2f}`)\n"
        f"🕒 مدت نگهداری: `{r['bars']}` کندل (~`{days:.1f}` روز)\n"
        f"⏰ زمان خروج: `{ex['time'].strftime('%Y-%m-%d %H:%M UTC')}`\n"
        f"_سود/زیان بر مبنای قیمت و بدون احتساب کارمزد است._"
    )


def quality_verdict(oos: dict) -> str:
    """Rate a tuned strategy by its out-of-sample result: good / weak / fail."""
    sh = oos.get("sharpe", float("nan"))
    ret = oos.get("total_return", float("nan"))
    if sh != sh:  # NaN -> not enough trades to judge
        return "weak"
    if sh < MIN_OOS_SHARPE or ret < 0:
        return "fail"
    if sh < WEAK_OOS_SHARPE:
        return "weak"
    return "good"


def verdict_note(verdict: str, symbol: str, enabled: bool) -> str:
    if verdict == "fail":
        if not enabled:
            return (f"🛑 *اعتبارسنجی رد شد* (برون‌نمونه منفی) — `{symbol}` خودکار "
                    f"*غیرفعال* شد. برای نادیده‌گرفتن: `/enable {symbol}`")
        return (f"🛑 *برون‌نمونه منفی است* — استراتژی روی `{symbol}` اعتبارسنجی نشد؛ "
                f"توصیه: `/disable {symbol}`")
    if verdict == "weak":
        return "⚠️ نتیجهٔ برون‌نمونه ضعیف است؛ با احتیاط استفاده کن."
    return "✅ اعتبارسنجی موفق بود — این پارامترها ذخیره و برای رصد استفاده می‌شود."


def format_analysis(summary: dict, symbol: str, timeframe: str, footer: str = "") -> str:
    p = summary["params"]
    is_m, oos = summary["in_sample"], summary["out_sample"]
    tuned = "بهینه‌شده" if summary.get("tuned") else "پیش‌فرض (بهینه‌سازی نتیجه نداد)"
    return (
        f"🔬 *تحلیل {symbol}* — {timeframe}  ({tuned})\n"
        f"بازه: {summary['range'][0][:10]} → {summary['range'][1][:10]} "
        f"({summary['n_bars']} کندل)\n\n"
        f"⚙️ پارامترهای انتخابی:\n"
        f"EMA `{p['ema_fast']}/{p['ema_slow']}` | RSI<`{p['rsi_long_max']}` | "
        f"SL `{p['atr_sl_mult']}×ATR` / TP `{p['atr_tp_mult']}×ATR`\n\n"
        f"📊 *درون‌نمونه (انتخاب):* بازده `{is_m['total_return']*100:+.1f}%` | "
        f"Sharpe `{is_m['sharpe']:.2f}` | برد `{is_m['win_rate']*100:.0f}%` | "
        f"معاملات `{is_m['num_trades']}`\n"
        f"🧪 *برون‌نمونه (آزمون صادقانه):* بازده `{oos['total_return']*100:+.1f}%` | "
        f"Sharpe `{oos['sharpe']:.2f}` | برد `{oos['win_rate']*100:.0f}%` | "
        f"DD `{oos['max_drawdown']*100:.1f}%` | معاملات `{oos['num_trades']}`\n\n"
        f"{footer}"
    )


# --------------------------------------------------------------------------- #
# Per-symbol processing
# --------------------------------------------------------------------------- #
def _update_runtime(runtime: dict, symbol: str, sig, latest_ts) -> None:
    last = sig.loc[latest_ts]
    runtime.setdefault("symbols", {})[symbol] = {
        "last_check": datetime.now(timezone.utc).isoformat(),
        "last_bar": latest_ts.isoformat(),
        "price": float(last["close"]),
        "rsi": float(last["rsi"]),
        "ema_fast": float(last["ema_fast"]),
        "ema_slow": float(last["ema_slow"]),
        "atr": float(last["atr"]),
    }


def _process_bars(entry: dict, sig, p: Params, notifier, symbol: str, timeframe: str) -> None:
    latest_ts = sig.index[-1]
    if not entry.get("last_bar"):
        entry["last_bar"] = latest_ts.isoformat()      # first run: no replay
        entry.setdefault("history", [])
        return
    last_seen = pd.Timestamp(entry["last_bar"])
    for ts in sig.index[sig.index > last_seen]:
        row = sig.loc[ts]
        pos = entry.get("position")
        if pos:
            ex = check_exit(pos, row, ts, p)
            if ex is not None:
                notifier.send(format_result(pos, ex, symbol, timeframe))
                entry.setdefault("history", []).append(realised(pos, ex))
                entry["history"] = entry["history"][-500:]
                entry["position"] = None
        if not entry.get("position"):
            s = signal_from_row(row, ts, p, alert_on_exit=False)
            if s and s["kind"] == "entry":
                notifier.send(format_alert(s, symbol, timeframe))
                entry["position"] = _position_dict(s)
    entry["last_bar"] = latest_ts.isoformat()


def process_symbol(ctx: Context, symbol: str) -> None:
    with ctx.lock:
        entry = st.watchlist(ctx.state).get(symbol)
        if not entry or not entry.get("enabled"):
            return
        p = st.dict_to_params(entry["params"])

    warmup = max(p.ema_slow, p.atr_period, p.rsi_period) + 20
    df = ctx.fetch_recent_fn(symbol, ctx.timeframe, max(400, warmup))  # slow I/O, no lock
    if len(df) < warmup:
        log.warning("%s: only %d candles (< warmup %d).", symbol, len(df), warmup)
        return
    sig = generate_signals(df, p)
    latest_ts = sig.index[-1]

    with ctx.lock:
        entry = st.watchlist(ctx.state).get(symbol)  # re-read (may have changed)
        if entry is None or not entry.get("enabled"):
            return
        _update_runtime(ctx.runtime, symbol, sig, latest_ts)
        _process_bars(entry, sig, p, ctx.notifier, symbol, ctx.timeframe)
        st.save_state(ctx.state)


def check_all(ctx: Context) -> None:
    with ctx.lock:
        symbols = [s for s, e in st.watchlist(ctx.state).items() if e.get("enabled")]
    for symbol in symbols:
        try:
            process_symbol(ctx, symbol)
        except Exception as e:
            log.exception("%s cycle error: %s", symbol, e)


# --------------------------------------------------------------------------- #
# Background analysis worker
# --------------------------------------------------------------------------- #
def _analysis_worker(ctx: Context, symbol: str) -> None:
    try:
        summary = run_analysis(symbol, ctx.fetch_history_fn, ctx.timeframe, ANALYZE_SINCE)
        verdict = quality_verdict(summary["out_sample"])
        with ctx.lock:
            wl = st.watchlist(ctx.state)
            if symbol not in wl:
                wl[symbol] = st.new_entry(st.dict_to_params(summary["params"]))
            wl[symbol]["params"] = summary["params"]
            wl[symbol]["analyzed_at"] = datetime.now(timezone.utc).isoformat()
            wl[symbol]["analysis"] = {
                "tuned": summary["tuned"], "range": summary["range"],
                "n_bars": summary["n_bars"], "verdict": verdict,
                "in_sample": summary["in_sample"], "out_sample": summary["out_sample"],
            }
            if verdict == "fail":  # quality guard: auto-disable poor performers
                wl[symbol]["enabled"] = False
            enabled = wl[symbol]["enabled"]
            st.save_state(ctx.state)
        footer = verdict_note(verdict, symbol, enabled)
        ctx.notifier.send(format_analysis(summary, symbol, ctx.timeframe, footer))
        log.info("Analysis done for %s (tuned=%s, verdict=%s).",
                 symbol, summary["tuned"], verdict)
    except Exception as e:
        log.exception("Analysis failed for %s: %s", symbol, e)
        ctx.notifier.send(f"❌ تحلیل `{symbol}` ناموفق بود: {e}")
    finally:
        ctx.analyzing.discard(symbol)


# --------------------------------------------------------------------------- #
# Weekly performance report
# --------------------------------------------------------------------------- #
def _period_stats(hist: list) -> tuple:
    n = len(hist)
    wins = sum(1 for t in hist if t["pnl_pct"] >= 0)
    rs = [t["r"] for t in hist if t.get("r") == t.get("r")]
    pnl = sum(t["pnl_pct"] for t in hist)
    return n, wins, sum(rs), pnl


def build_weekly_report(ctx: "Context", days: int = 7) -> str:
    """Summarise closed trades and open positions across the watchlist.

    Reads shared state/runtime; callers hold ctx.lock (it does not lock itself).
    """
    now = datetime.now(timezone.utc)
    since = now - timedelta(days=days)
    wl = st.watchlist(ctx.state)

    lines, tot_n, tot_w, tot_r, tot_pnl, open_count = [], 0, 0, 0.0, 0.0, 0
    best = worst = None
    for sym, e in wl.items():
        hist = [t for t in e.get("history", []) if pd.Timestamp(t["exit_time"]) >= since]
        pos = e.get("position")
        if pos:
            open_count += 1
        if not hist and not pos:
            continue
        n, w, r, pnl = _period_stats(hist)
        tot_n += n; tot_w += w; tot_r += r; tot_pnl += pnl
        flag = "🟢" if e.get("enabled") else "⚪️"
        seg = f"{flag} `{sym}`"
        seg += (f": {n} معامله | برد {w/n*100:.0f}% | R `{r:+.1f}` | `{pnl:+.1f}%`"
                if n else ": بدون معاملهٔ بسته")
        if pos:
            rt = ctx.runtime.get("symbols", {}).get(sym, {})
            price = rt.get("price")
            if price:
                up = (price / pos["entry"] - 1) * 100 if pos["side"] == "LONG" \
                    else (pos["entry"] / price - 1) * 100
                seg += f"  📌{pos['side']} شناور `{up:+.1f}%`"
            else:
                seg += "  📌پوزیشن باز"
        lines.append(seg)
        if n:
            best = (sym, pnl) if best is None or pnl > best[1] else best
            worst = (sym, pnl) if worst is None or pnl < worst[1] else worst

    enabled = sum(1 for e in wl.values() if e.get("enabled"))
    head = f"📅 *گزارش هفتگی* ({since:%Y-%m-%d} → {now:%Y-%m-%d} UTC)"
    if not lines:
        return head + "\n\nاین بازه معامله‌ای بسته نشد و پوزیشن بازی هم نیست."
    foot = (f"\n\n📊 *مجموع:* {tot_n} معامله"
            + (f" | برد {tot_w/tot_n*100:.0f}%" if tot_n else "")
            + f" | R `{tot_r:+.1f}` | `{tot_pnl:+.1f}%`\n"
            f"🟢 فعال: {enabled} از {len(wl)} | 📌 پوزیشن باز: {open_count}")
    if best:
        foot += f"\n🏆 بهترین: `{best[0]}` (`{best[1]:+.1f}%`)"
    if worst and worst[0] != best[0]:
        foot += f"  |  🔻 بدترین: `{worst[0]}` (`{worst[1]:+.1f}%`)"
    return head + "\n\n" + "\n".join(lines) + foot


def maybe_send_weekly(ctx: "Context") -> None:
    """Send the weekly report once per ISO week, on/after the scheduled time."""
    if not WEEKLY_REPORT or not ctx.notifier.enabled:
        return
    now = datetime.now(timezone.utc)
    iso = now.isocalendar()
    key = f"{iso[0]}-W{iso[1]:02d}"
    with ctx.lock:
        if ctx.state.get("last_report_week") == key:
            return
        if now.weekday() < REPORT_DAY or (now.weekday() == REPORT_DAY and now.hour < REPORT_HOUR):
            return
        text = build_weekly_report(ctx)
        ctx.state["last_report_week"] = key
        st.save_state(ctx.state)
    ctx.notifier.send(text)
    log.info("Weekly report sent (%s).", key)


# --------------------------------------------------------------------------- #
# Telegram command listener
# --------------------------------------------------------------------------- #
def command_loop(ctx: Context) -> None:
    from src.live import commands

    if not ctx.notifier.enabled:
        log.warning("Telegram not configured -> command listener disabled.")
        return
    ctx.notifier.set_my_commands(commands.MENU)
    offset = ctx.notifier.drain_updates()
    log.info("Command listener ready (offset=%s).", offset)
    while True:
        try:
            for u in ctx.notifier.get_updates(offset=offset, timeout=25):
                offset = u["update_id"] + 1
                msg = u.get("message") or u.get("edited_message") or {}
                if str(msg.get("chat", {}).get("id")) != str(ctx.notifier.chat_id):
                    continue
                text = (msg.get("text") or "").strip()
                if not text.startswith("/"):
                    continue
                with ctx.lock:  # serialise command handling vs the market/analysis threads
                    reply = commands.dispatch(text, ctx)
                if reply:
                    ctx.notifier.send(reply)
        except Exception as e:
            log.exception("Command loop error: %s", e)
            time.sleep(5)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_context() -> Context:
    notifier = TelegramNotifier()
    default_p = load_params()
    state = st.migrate(st.load_state(), st.normalize_symbol(WATCHLIST_SEED[0]),
                       default_p, seed_symbols=WATCHLIST_SEED)
    st.save_state(state)
    return Context(notifier=notifier, state=state, default_params=default_p)


def main():
    ctx = build_context()
    symbols = list(st.watchlist(ctx.state))
    log.info("Live monitor: watching %s | poll=%ss", symbols, POLL_SECONDS)
    ctx.notifier.send(
        "🤖 *موتور رصد چندارزی فعال شد*\n"
        f"واچ‌لیست: {', '.join(f'`{s}`' for s in symbols) or '—'}\n"
        f"تایم‌فریم: {TIMEFRAME} | بازهٔ بررسی: هر {POLL_SECONDS//60} دقیقه\n"
        f"برای فهرست دستورها /help را بفرست."
    )
    threading.Thread(target=command_loop, args=(ctx,), daemon=True).start()
    while True:
        try:
            check_all(ctx)
            maybe_send_weekly(ctx)
        except Exception as e:
            log.exception("Cycle error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
