"""
Telegram command handlers.

Pure functions: each command reads the shared `state` (persisted: position,
history, last_bar) and `runtime` (in-memory snapshot of the latest price /
indicators, refreshed every polling cycle) and returns a Markdown reply string.
No network here — sending is done by the caller in monitor.command_loop.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

# Menu registered with Telegram (setMyCommands). Names must be lowercase.
MENU = [
    ("status", "وضعیت سرویس و پوزیشن فعلی"),
    ("position", "جزئیات پوزیشن باز و سود/زیان شناور"),
    ("stats", "آمار کلی معاملات بسته‌شده"),
    ("history", "آخرین معاملات (مثلا: /history 10)"),
    ("price", "قیمت لحظه‌ای و اندیکاتورها"),
    ("params", "پارامترهای استراتژی"),
    ("help", "راهنمای دستورها"),
]


def _age(iso: str | None) -> str:
    if not iso:
        return "—"
    t = pd.Timestamp(iso)
    delta = datetime.now(timezone.utc) - t.to_pydatetime()
    secs = int(delta.total_seconds())
    if secs < 60:
        return f"{secs} ثانیه پیش"
    if secs < 3600:
        return f"{secs // 60} دقیقه پیش"
    if secs < 86400:
        return f"{secs // 3600} ساعت پیش"
    return f"{secs // 86400} روز پیش"


def _fmt_dt(iso: str | None) -> str:
    if not iso:
        return "—"
    return pd.Timestamp(iso).strftime("%Y-%m-%d %H:%M UTC")


def cmd_help(*_args) -> str:
    rows = "\n".join(f"/{c} — {d}" for c, d in MENU)
    return "📖 *دستورهای موجود*\n" + rows


def cmd_params(state, runtime, p, symbol, timeframe) -> str:
    return (
        f"⚙️ *پارامترهای استراتژی* — `{symbol}` {timeframe}\n"
        f"EMA: `{p.ema_fast}/{p.ema_slow}`\n"
        f"RSI: دوره `{p.rsi_period}`، فیلتر ورود `{p.rsi_long_min}–{p.rsi_long_max}`\n"
        f"ATR: دوره `{p.atr_period}`\n"
        f"حد ضرر: `{p.atr_sl_mult}×ATR` | حد سود: `{p.atr_tp_mult}×ATR`\n"
        f"شورت فعال؟ `{'بله' if p.allow_short else 'خیر'}`"
    )


def cmd_price(state, runtime, p, symbol, timeframe) -> str:
    if "price" not in runtime:
        return "⏳ هنوز داده‌ای دریافت نشده؛ کمی بعد دوباره امتحان کن."
    trend = "صعودی 🟢" if runtime["ema_fast"] >= runtime["ema_slow"] else "نزولی 🔴"
    return (
        f"💹 *قیمت لحظه‌ای* — `{symbol}` {timeframe}\n"
        f"قیمت (کلوز آخرین کندل): `${runtime['price']:,.1f}`\n"
        f"RSI: `{runtime['rsi']:.1f}` | ATR: `{runtime['atr']:,.0f}`\n"
        f"EMA: `{runtime['ema_fast']:,.0f}/{runtime['ema_slow']:,.0f}` → روند {trend}\n"
        f"_آخرین کندل: {_fmt_dt(runtime.get('last_bar'))} ({_age(runtime.get('last_check'))})_"
    )


def _position_block(pos: dict, runtime: dict) -> str:
    is_long = pos["side"] == "LONG"
    entry, sl, tp = pos["entry"], pos["sl"], pos["tp"]
    lines = [
        f"نوع: *{pos['side']}*",
        f"📍 ورود: `${entry:,.1f}`  ({_fmt_dt(pos.get('entry_time'))})",
        f"🎯 TP: `${tp:,.1f}`  |  🛑 SL: `${sl:,.1f}`  |  ⚖️ R/R: `{pos.get('rr', float('nan')):.2f}`",
    ]
    price = runtime.get("price")
    if price:
        upnl = (price / entry - 1) * 100 if is_long else (entry / price - 1) * 100
        risk = abs(entry - sl)
        cur_r = ((price - entry) if is_long else (entry - price)) / risk if risk > 0 else float("nan")
        emoji = "🟢" if upnl >= 0 else "🔴"
        lines.append(f"قیمت فعلی: `${price:,.1f}`")
        lines.append(f"{emoji} سود/زیان شناور: `{upnl:+.2f}%`  (R: `{cur_r:+.2f}`)")
    return "\n".join(lines)


def cmd_position(state, runtime, p, symbol, timeframe) -> str:
    pos = state.get("position")
    if not pos:
        return "📭 پوزیشن بازی وجود ندارد. منتظر سیگنال بعدی هستم."
    return f"📌 *پوزیشن باز* — `{symbol}` {timeframe}\n" + _position_block(pos, runtime)


def cmd_status(state, runtime, p, symbol, timeframe) -> str:
    pos = state.get("position")
    hist = state.get("history", [])
    head = (
        f"🤖 *وضعیت سرویس* — `{symbol}` {timeframe}\n"
        f"فعال از: {_fmt_dt(runtime.get('started_at'))}\n"
        f"آخرین بررسی: {_age(runtime.get('last_check'))}\n"
        f"آخرین کندل بسته: {_fmt_dt(runtime.get('last_bar'))}\n"
    )
    if "price" in runtime:
        head += f"قیمت: `${runtime['price']:,.1f}` | RSI `{runtime['rsi']:.1f}`\n"
    head += f"معاملات بسته‌شده: `{len(hist)}`\n\n"
    if pos:
        head += "📌 *پوزیشن باز:*\n" + _position_block(pos, runtime)
    else:
        head += "📭 پوزیشن باز نداریم — منتظر سیگنال."
    return head


def cmd_stats(state, runtime, p, symbol, timeframe) -> str:
    hist = state.get("history", [])
    if not hist:
        return "📊 هنوز معاملهٔ بسته‌شده‌ای ثبت نشده."
    n = len(hist)
    rs = [t["r"] for t in hist if not math.isnan(t.get("r", float("nan")))]
    pnls = [t["pnl_pct"] for t in hist]
    wins = [t for t in hist if t["pnl_pct"] >= 0]
    win_rate = len(wins) / n * 100
    total_r = sum(rs)
    avg_r = total_r / len(rs) if rs else float("nan")
    by_reason = {}
    for t in hist:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1
    best = max(hist, key=lambda t: t["pnl_pct"])
    worst = min(hist, key=lambda t: t["pnl_pct"])
    reason_fa = {"TP": "حد سود", "SL": "حد ضرر", "EXIT": "فلیپ روند"}
    reason_str = " | ".join(f"{reason_fa.get(k, k)}: {v}" for k, v in by_reason.items())
    return (
        f"📊 *آمار معاملات* — `{symbol}` {timeframe}\n"
        f"تعداد: `{n}` | برد: `{len(wins)}` | باخت: `{n-len(wins)}`\n"
        f"نرخ برد: `{win_rate:.1f}%`\n"
        f"مجموع R: `{total_r:+.2f}` | میانگین R: `{avg_r:+.2f}`\n"
        f"میانگین سود/زیان هر معامله: `{sum(pnls)/n:+.2f}%`\n"
        f"بهترین: `{best['pnl_pct']:+.2f}%` | بدترین: `{worst['pnl_pct']:+.2f}%`\n"
        f"تفکیک خروج → {reason_str}"
    )


def cmd_history(state, runtime, p, symbol, timeframe, arg: str = "5") -> str:
    hist = state.get("history", [])
    if not hist:
        return "🗒 تاریخچه‌ای موجود نیست."
    try:
        n = max(1, min(20, int(arg)))
    except (ValueError, TypeError):
        n = 5
    rows = []
    reason_fa = {"TP": "🎯TP", "SL": "🛑SL", "EXIT": "⚪️flip"}
    for t in hist[-n:][::-1]:
        emoji = "✅" if t["pnl_pct"] >= 0 else "❌"
        rows.append(
            f"{emoji} {t['side']} `{t['pnl_pct']:+.2f}%` (R `{t['r']:+.2f}`) "
            f"{reason_fa.get(t['reason'], t['reason'])} — {_fmt_dt(t['exit_time'])}"
        )
    return f"🗒 *آخرین {len(rows)} معامله*\n" + "\n".join(rows)


_HANDLERS = {
    "help": cmd_help,
    "start": cmd_help,
    "status": cmd_status,
    "position": cmd_position,
    "pos": cmd_position,
    "stats": cmd_stats,
    "price": cmd_price,
    "params": cmd_params,
    "config": cmd_params,
}


def dispatch(text: str, state, runtime, p, symbol, timeframe) -> str | None:
    """Route a '/command [args]' string to its handler. Returns reply or None."""
    parts = text.strip().split()
    cmd = parts[0].lstrip("/").lower()
    cmd = cmd.split("@")[0]  # strip @botname in group chats
    arg = parts[1] if len(parts) > 1 else None

    if cmd == "history":
        return cmd_history(state, runtime, p, symbol, timeframe, arg or "5")
    handler = _HANDLERS.get(cmd)
    if handler is None:
        return f"❓ دستور ناشناخته: `/{cmd}`\nبرای فهرست دستورها /help را بفرست."
    if handler is cmd_help:
        return cmd_help()
    return handler(state, runtime, p, symbol, timeframe)
