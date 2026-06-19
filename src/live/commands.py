"""
Telegram command handlers for the multi-symbol engine.

Each handler reads the shared Context (state watchlist + runtime snapshot) and
returns a Markdown reply. Management commands mutate the watchlist and persist.
No long-running work here: /analyze just launches a background job via ctx.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd

from src.live import state as st

MENU = [
    ("list", "فهرست ارزها و وضعیت‌شان"),
    ("status", "وضعیت کلی یا یک ارز: /status SOL"),
    ("position", "پوزیشن باز یک ارز: /position SOL"),
    ("stats", "آمار معاملات یک ارز: /stats SOL"),
    ("history", "آخرین معاملات: /history SOL 10"),
    ("price", "قیمت و اندیکاتورها: /price SOL"),
    ("params", "پارامترهای یک ارز: /params SOL"),
    ("analyze", "بهینه‌سازی پارامترهای یک ارز: /analyze SOL"),
    ("add", "افزودن ارز: /add SOL/USDT"),
    ("remove", "حذف ارز: /remove SOL"),
    ("enable", "فعال‌سازی ارز: /enable SOL"),
    ("disable", "غیرفعال‌سازی ارز: /disable SOL"),
    ("help", "راهنمای دستورها"),
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _age(iso):
    if not iso:
        return "—"
    secs = int((datetime.now(timezone.utc) - pd.Timestamp(iso).to_pydatetime()).total_seconds())
    if secs < 60:
        return f"{secs} ثانیه پیش"
    if secs < 3600:
        return f"{secs // 60} دقیقه پیش"
    if secs < 86400:
        return f"{secs // 3600} ساعت پیش"
    return f"{secs // 86400} روز پیش"


def _dt(iso):
    return pd.Timestamp(iso).strftime("%Y-%m-%d %H:%M UTC") if iso else "—"


def _g(x):
    """Readable price across BTC ($100k), SOL ($148.20) and DOGE ($0.1234) scales."""
    ax = abs(x)
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 1:
        return f"{x:,.2f}"
    if ax >= 0.01:
        return f"{x:.4f}"
    return f"{x:.6f}"


def _rt(ctx, symbol):
    return ctx.runtime.get("symbols", {}).get(symbol, {})


def _resolve_symbol(ctx, arg):
    """Pick the target symbol: the given arg, or the sole watched symbol."""
    wl = st.watchlist(ctx.state)
    if arg:
        return st.normalize_symbol(arg)
    if len(wl) == 1:
        return next(iter(wl))
    return None


def _position_block(pos, rt):
    is_long = pos["side"] == "LONG"
    entry, sl, tp = pos["entry"], pos["sl"], pos["tp"]
    lines = [
        f"نوع: *{pos['side']}*  |  ورود: `${_g(entry)}`  ({_dt(pos.get('entry_time'))})",
        f"🎯 TP `${_g(tp)}`  |  🛑 SL `${_g(sl)}`  |  ⚖️ R/R `{pos.get('rr', float('nan')):.2f}`",
    ]
    price = rt.get("price")
    if price:
        upnl = (price / entry - 1) * 100 if is_long else (entry / price - 1) * 100
        risk = abs(entry - sl)
        cur_r = ((price - entry) if is_long else (entry - price)) / risk if risk > 0 else float("nan")
        lines.append(f"قیمت فعلی `${_g(price)}` → {'🟢' if upnl >= 0 else '🔴'} "
                     f"شناور `{upnl:+.2f}%` (R `{cur_r:+.2f}`)")
    return "\n".join(lines)


def _stats_of(hist):
    n = len(hist)
    rs = [t["r"] for t in hist if not math.isnan(t.get("r", float("nan")))]
    wins = [t for t in hist if t["pnl_pct"] >= 0]
    return {
        "n": n, "wins": len(wins), "losses": n - len(wins),
        "win_rate": len(wins) / n * 100 if n else 0.0,
        "total_r": sum(rs), "avg_r": (sum(rs) / len(rs)) if rs else float("nan"),
        "avg_pnl": (sum(t["pnl_pct"] for t in hist) / n) if n else 0.0,
    }


# --------------------------------------------------------------------------- #
# query commands
# --------------------------------------------------------------------------- #
def cmd_help(ctx, arg=None):
    return "📖 *دستورهای موجود*\n" + "\n".join(f"/{c} — {d}" for c, d in MENU)


def cmd_list(ctx, arg=None):
    wl = st.watchlist(ctx.state)
    if not wl:
        return "📭 واچ‌لیست خالی است. با `/add SOL/USDT` اضافه کن."
    rows = []
    for sym, e in wl.items():
        flag = "🟢" if e.get("enabled") else "⚪️"
        pos = "📌 پوزیشن باز" if e.get("position") else "—"
        if sym in ctx.analyzing:
            pos = "🔬 در حال تحلیل"
        p = e["params"]
        tuned = "🛠" if e.get("analyzed_at") else ""
        n = len(e.get("history", []))
        rows.append(f"{flag} `{sym}` {tuned} — EMA `{p['ema_fast']}/{p['ema_slow']}` | "
                    f"معاملات `{n}` | {pos}")
    return "📋 *واچ‌لیست*\n" + "\n".join(rows) + "\n\n🟢فعال ⚪️غیرفعال 🛠بهینه‌شده"


def cmd_status(ctx, arg=None):
    sym = _resolve_symbol(ctx, arg)
    if arg and sym not in st.watchlist(ctx.state):
        return f"`{sym}` در واچ‌لیست نیست. /list را ببین."
    if sym is None:
        # overview of all
        head = (f"🤖 *وضعیت موتور*\nفعال از: {_dt(ctx.runtime.get('started_at'))}\n"
                f"تعداد ارزها: `{len(st.watchlist(ctx.state))}`\n\n")
        return head + cmd_list(ctx)
    e = st.watchlist(ctx.state)[sym]
    rt = _rt(ctx, sym)
    out = [f"🤖 *وضعیت* `{sym}` — {ctx.timeframe}",
           f"وضعیت: {'فعال 🟢' if e.get('enabled') else 'غیرفعال ⚪️'}",
           f"آخرین بررسی: {_age(rt.get('last_check'))} | آخرین کندل: {_dt(rt.get('last_bar'))}"]
    if "price" in rt:
        out.append(f"قیمت: `${_g(rt['price'])}` | RSI `{rt['rsi']:.1f}`")
    out.append(f"معاملات بسته‌شده: `{len(e.get('history', []))}`")
    if e.get("position"):
        out.append("\n📌 *پوزیشن باز:*\n" + _position_block(e["position"], rt))
    else:
        out.append("📭 پوزیشن باز ندارد.")
    return "\n".join(out)


def cmd_position(ctx, arg=None):
    sym = _resolve_symbol(ctx, arg)
    if sym is None:
        return "نماد را مشخص کن: مثلا `/position SOL`"
    e = st.watchlist(ctx.state).get(sym)
    if not e:
        return f"`{sym}` در واچ‌لیست نیست."
    if not e.get("position"):
        return f"📭 `{sym}` پوزیشن باز ندارد."
    return f"📌 *پوزیشن باز* `{sym}`\n" + _position_block(e["position"], _rt(ctx, sym))


def cmd_stats(ctx, arg=None):
    sym = _resolve_symbol(ctx, arg)
    if sym is None:
        return "نماد را مشخص کن: مثلا `/stats SOL`"
    e = st.watchlist(ctx.state).get(sym)
    if not e:
        return f"`{sym}` در واچ‌لیست نیست."
    hist = e.get("history", [])
    if not hist:
        return f"📊 `{sym}` هنوز معاملهٔ بسته‌شده‌ای ندارد."
    s = _stats_of(hist)
    by = {}
    for t in hist:
        by[t["reason"]] = by.get(t["reason"], 0) + 1
    rf = {"TP": "حد سود", "SL": "حد ضرر", "EXIT": "فلیپ"}
    best, worst = max(hist, key=lambda t: t["pnl_pct"]), min(hist, key=lambda t: t["pnl_pct"])
    return (
        f"📊 *آمار* `{sym}`\n"
        f"تعداد `{s['n']}` | برد `{s['wins']}` | باخت `{s['losses']}` | نرخ برد `{s['win_rate']:.1f}%`\n"
        f"مجموع R `{s['total_r']:+.2f}` | میانگین R `{s['avg_r']:+.2f}` | میانگین سود `{s['avg_pnl']:+.2f}%`\n"
        f"بهترین `{best['pnl_pct']:+.2f}%` | بدترین `{worst['pnl_pct']:+.2f}%`\n"
        f"خروج‌ها → " + " | ".join(f"{rf.get(k, k)}: {v}" for k, v in by.items())
    )


def cmd_history(ctx, arg=None, n_arg=None):
    sym = _resolve_symbol(ctx, arg)
    if sym is None:
        return "نماد را مشخص کن: مثلا `/history SOL 10`"
    e = st.watchlist(ctx.state).get(sym)
    if not e:
        return f"`{sym}` در واچ‌لیست نیست."
    hist = e.get("history", [])
    if not hist:
        return f"🗒 `{sym}` تاریخچه‌ای ندارد."
    try:
        n = max(1, min(20, int(n_arg)))
    except (ValueError, TypeError):
        n = 5
    rf = {"TP": "🎯TP", "SL": "🛑SL", "EXIT": "⚪️flip"}
    rows = [f"{'✅' if t['pnl_pct'] >= 0 else '❌'} {t['side']} `{t['pnl_pct']:+.2f}%` "
            f"(R `{t['r']:+.2f}`) {rf.get(t['reason'], t['reason'])} — {_dt(t['exit_time'])}"
            for t in hist[-n:][::-1]]
    return f"🗒 *آخرین {len(rows)} معاملهٔ* `{sym}`\n" + "\n".join(rows)


def cmd_price(ctx, arg=None):
    sym = _resolve_symbol(ctx, arg)
    if sym is None:
        return "نماد را مشخص کن: مثلا `/price SOL`"
    rt = _rt(ctx, sym)
    if "price" not in rt:
        return f"⏳ هنوز دادهٔ `{sym}` دریافت نشده."
    trend = "صعودی 🟢" if rt["ema_fast"] >= rt["ema_slow"] else "نزولی 🔴"
    return (f"💹 *{sym}* — {ctx.timeframe}\n"
            f"قیمت `${_g(rt['price'])}` | RSI `{rt['rsi']:.1f}` | ATR `{_g(rt['atr'])}`\n"
            f"EMA `{_g(rt['ema_fast'])}/{_g(rt['ema_slow'])}` → روند {trend}\n"
            f"_آخرین کندل: {_dt(rt.get('last_bar'))} ({_age(rt.get('last_check'))})_")


def cmd_params(ctx, arg=None):
    sym = _resolve_symbol(ctx, arg)
    if sym is None:
        return "نماد را مشخص کن: مثلا `/params SOL`"
    e = st.watchlist(ctx.state).get(sym)
    if not e:
        return f"`{sym}` در واچ‌لیست نیست."
    p = e["params"]
    src = f"بهینه‌شده در {_dt(e['analyzed_at'])}" if e.get("analyzed_at") else "پیش‌فرض"
    return (f"⚙️ *پارامترهای* `{sym}`  ({src})\n"
            f"EMA `{p['ema_fast']}/{p['ema_slow']}`\n"
            f"RSI دوره `{p['rsi_period']}`، فیلتر `{p['rsi_long_min']}–{p['rsi_long_max']}`\n"
            f"ATR دوره `{p['atr_period']}` | SL `{p['atr_sl_mult']}×` | TP `{p['atr_tp_mult']}×`\n"
            f"شورت: `{'بله' if p['allow_short'] else 'خیر'}`")


# --------------------------------------------------------------------------- #
# management commands (mutate + persist)
# --------------------------------------------------------------------------- #
def cmd_add(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/add SOL/USDT`"
    ok, res = st.add_symbol(ctx.state, arg, ctx.default_params)
    if not ok:
        return res
    st.save_state(ctx.state)
    return (f"✅ `{res}` اضافه شد (با پارامترهای پیش‌فرض، فعال).\n"
            f"برای استخراج تنظیمات اختصاصی‌اش: `/analyze {res}`")


def cmd_remove(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/remove SOL`"
    ok, res = st.remove_symbol(ctx.state, arg)
    if ok:
        st.save_state(ctx.state)
        return f"🗑 `{res}` حذف شد."
    return res


def cmd_enable(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/enable SOL`"
    ok, res = st.set_enabled(ctx.state, arg, True)
    if ok:
        st.save_state(ctx.state)
        return f"🟢 `{res}` فعال شد."
    return res


def cmd_disable(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/disable SOL`"
    ok, res = st.set_enabled(ctx.state, arg, False)
    if ok:
        st.save_state(ctx.state)
        return f"⚪️ `{res}` غیرفعال شد (رصد متوقف، تنظیمات حفظ می‌شود)."
    return res


def cmd_analyze(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/analyze SOL/USDT`"
    return ctx.start_analysis(arg)


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
_HANDLERS = {
    "help": cmd_help, "start": cmd_help, "list": cmd_list,
    "status": cmd_status, "position": cmd_position, "pos": cmd_position,
    "stats": cmd_stats, "price": cmd_price, "params": cmd_params, "config": cmd_params,
    "add": cmd_add, "remove": cmd_remove, "delete": cmd_remove,
    "enable": cmd_enable, "disable": cmd_disable, "analyze": cmd_analyze,
}


def dispatch(text: str, ctx) -> str | None:
    parts = text.strip().split()
    cmd = parts[0].lstrip("/").lower().split("@")[0]
    args = parts[1:]
    if cmd == "history":
        return cmd_history(ctx, args[0] if args else None, args[1] if len(args) > 1 else None)
    handler = _HANDLERS.get(cmd)
    if handler is None:
        return f"❓ دستور ناشناخته: `/{cmd}`\nبرای فهرست دستورها /help را بفرست."
    return handler(ctx, args[0] if args else None)
