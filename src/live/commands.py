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
    ("menu", "منوی دکمه‌ای (شیشه‌ای)"),
    ("list", "فهرست ارزها و وضعیت‌شان"),
    ("compare", "مقایسهٔ ارزها بر اساس برون‌نمونه"),
    ("summary", "گزارش هفتگی عملکرد همهٔ ارزها"),
    ("status", "وضعیت کلی یا یک ارز: /status SOL"),
    ("position", "پوزیشن باز یک ارز: /position SOL"),
    ("stats", "آمار معاملات یک ارز: /stats SOL"),
    ("history", "آخرین معاملات: /history SOL 10"),
    ("price", "قیمت و اندیکاتورها: /price SOL"),
    ("params", "پارامترهای یک ارز: /params SOL"),
    ("analyze", "بهینه‌سازی پارامترهای یک ارز: /analyze SOL"),
    ("analyze_all", "تحلیل مجدد همهٔ ارزهای فعال"),
    ("backtest", "بک‌تست و نمودار یک ارز: /backtest SOL"),
    ("backtest_all", "تحلیل نموداری همهٔ ارزها"),
    ("walkforward", "آزمون Walk-Forward (expectancy واقعی): /walkforward SOL"),
    ("xsmom", "سبد مومنتوم مقطعی Long/Short"),
    ("xsmom_status", "آخرین سبد مومنتوم ذخیره‌شده"),
    ("xsmom_chart", "نمودار بک‌تست سبد مومنتوم"),
    ("report", "نمایش دوبارهٔ نتیجهٔ تحلیل ذخیره‌شده: /report SOL"),
    ("add", "افزودن ارز: /add SOL/USDT"),
    ("remove", "حذف ارز: /remove SOL"),
    ("enable", "فعال‌سازی ارز: /enable SOL"),
    ("disable", "غیرفعال‌سازی ارز: /disable SOL"),
    ("help", "راهنمای دستورها"),
]


# --------------------------------------------------------------------------- #
# Glass (inline) keyboards
# --------------------------------------------------------------------------- #
def _btn(text, data):
    return {"text": text, "callback_data": data}


def main_menu_kb(ctx) -> dict:
    wl = st.watchlist(ctx.state)
    rows = [[_btn("📋 لیست", "list"), _btn("📅 گزارش هفتگی", "summary")]]
    rows.append([_btn("⚖️ مقایسهٔ ارزها", "compare"),
                 _btn("🔬 تحلیل مجدد همه", "analyze_all")])
    rows.append([_btn("📈 تحلیل نموداری همه", "backtest_all"),
                 _btn("🔀 سبد مومنتوم", "xsmom")])
    row = []
    for s in wl:
        flag = "🟢" if wl[s].get("enabled") else "⚪️"
        row.append(_btn(f"{flag} {s.split('/')[0]}", f"menu {s}"))
        if len(row) == 2:
            rows.append(row); row = []
    if row:
        rows.append(row)
    rows.append([_btn("❓ راهنما", "help")])
    return {"inline_keyboard": rows}


def xsmom_kb() -> dict:
    return {"inline_keyboard": [
        [_btn("🔄 محاسبهٔ دوباره", "xsmom"), _btn("📈 نمودار", "xsmom_chart")],
        [_btn("⬅️ منوی اصلی", "menu")],
    ]}


def symbol_kb(sym: str) -> dict:
    return {"inline_keyboard": [
        [_btn("📈 قیمت", f"price {sym}"), _btn("📊 آمار", f"stats {sym}")],
        [_btn("📌 پوزیشن", f"position {sym}"), _btn("🗒 تاریخچه", f"history {sym}")],
        [_btn("🔬 تحلیل", f"analyze {sym}"), _btn("🧪 بک‌تست", f"backtest {sym}")],
        [_btn("📄 گزارش تحلیل", f"report {sym}")],
        [_btn("✅ فعال", f"enable {sym}"), _btn("⛔ غیرفعال", f"disable {sym}")],
        [_btn("⬅️ منوی اصلی", "menu")],
    ]}


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
    side = pos.get("side_str") or ("LONG" if pos.get("side") == 1 else "SHORT")
    is_long = side == "LONG"
    entry = pos["entry"]
    stop = pos.get("stop", pos.get("sl"))     # tolerate legacy key
    tp = pos.get("tp")
    lev = pos.get("leverage", 1.0)
    head = f"نوع: *{side}*"
    if lev and lev > 1.0:
        head += f" `{lev:g}x`"
    lines = [f"{head}  |  ورود: `${_g(entry)}`  ({_dt(pos.get('entry_time'))})"]
    tline = f"🛑 استاپ فعلی `${_g(stop)}`"
    if tp:
        tline += f"  |  🎯 هدف `${_g(tp)}`"
    if pos.get("rr"):
        tline += f"  |  ⚖️ R/R `{pos['rr']:.2f}`"
    lines.append(tline)
    if pos.get("liq"):
        lines.append(f"☠️ لیکویید ≈ `${_g(pos['liq'])}`")
    if pos.get("mode") == "partial" and pos.get("partial_done"):
        lines.append(f"✅ سود جزئی گرفته شد | مابقی `{pos.get('remaining', 1)*100:.0f}%` در حال تریل")
    elif pos.get("mode") == "trailing" or (pos.get("mode") == "partial"):
        lines.append("🏃 خروج با تریلینگ‌استاپ (استاپ با روند بالا می‌آید)")
    price = rt.get("price")
    if price:
        move = (price / entry - 1) if is_long else (entry / price - 1)
        upnl = move * 100 * lev      # leveraged P/L on margin
        risk = pos.get("risk") or abs(entry - stop)
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
    text = "📖 *دستورهای موجود*\n" + "\n".join(f"/{c} — {d}" for c, d in MENU)
    return {"text": text, "keyboard": main_menu_kb(ctx)}


def cmd_menu(ctx, arg=None):
    """Glass-button menu. With a symbol -> that coin's submenu; else main menu."""
    if arg:
        sym = _resolve_symbol(ctx, arg)
        if sym not in st.watchlist(ctx.state):
            return f"`{sym}` در واچ‌لیست نیست."
        return {"text": cmd_status(ctx, sym), "keyboard": symbol_kb(sym)}
    return {"text": "🔹 *منوی اصلی* — یک گزینه را انتخاب کن:",
            "keyboard": main_menu_kb(ctx)}


def cmd_backtest(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/backtest SOL/USDT`"
    return ctx.start_backtest(arg)


def cmd_walkforward(ctx, arg=None):
    if not arg:
        return "نماد را بده: مثلا `/walkforward SOL/USDT`"
    return ctx.start_walkforward(arg)


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


def cmd_analyze_all(ctx, arg=None):
    return ctx.start_analysis_all()


def cmd_backtest_all(ctx, arg=None):
    return ctx.start_backtest_all()


def cmd_xsmom(ctx, arg=None):
    """Compute the current long/short cross-sectional momentum basket on demand."""
    return ctx.start_xsmom(announce=True)


def cmd_xsmom_status(ctx, arg=None):
    """Show the LAST posted momentum basket from state (no recompute)."""
    x = ctx.state.get("xsmom")
    if not x or not (x.get("longs") or x.get("shorts")):
        return ("🔀 هنوز سبد مومنتومی محاسبه نشده.\n"
                "برای محاسبهٔ سبد فعلی `/xsmom` و برای نمودار تاریخی `/xsmom_chart` را بزن.")
    longs = "، ".join(f"`{s.split('/')[0]}`" for s in x.get("longs", [])) or "—"
    shorts = "، ".join(f"`{s.split('/')[0]}`" for s in x.get("shorts", [])) or "—"
    lines = [
        "🔀 *آخرین سبد مومنتوم (ذخیره‌شده)*",
        f"بازچینش: {_dt(x.get('last_rebalance'))} ({_age(x.get('last_rebalance'))})",
        f"تا کندلِ: {_dt(x.get('asof'))}",
        f"📈 لانگ: {longs}",
    ]
    if x.get("shorts"):
        lines.append(f"📉 شورت: {shorts}")
    lines.append("\n_برای به‌روزرسانی `/xsmom`، برای نمودار `/xsmom_chart`._")
    return "\n".join(lines)


def cmd_xsmom_chart(ctx, arg=None):
    """Backtest the momentum basket over full history and send an equity chart."""
    return ctx.start_xsmom_chart()


def cmd_compare(ctx, arg=None):
    """Rank watched coins by their WALK-FORWARD expectancy (the honest metric)."""
    wl = st.watchlist(ctx.state)
    if not wl:
        return "📭 واچ‌لیست خالی است. با `/add SOL/USDT` اضافه کن."
    analyzed, pending = [], []
    for sym, e in wl.items():
        a = e.get("analysis")
        if a and a.get("walk_forward") and a["walk_forward"].get("trades"):
            analyzed.append((sym, e, a))
        else:
            pending.append(sym)

    analyzed.sort(key=lambda it: it[2]["walk_forward"].get("expectancy", -1e9), reverse=True)

    medals = ["🥇", "🥈", "🥉"]
    vmark = {"good": "✅", "weak": "⚠️", "fail": "🛑"}
    lines = []
    for i, (sym, e, a) in enumerate(analyzed):
        wf = a["walk_forward"]
        rank = medals[i] if i < 3 else "•"
        flag = "🟢" if e.get("enabled") else "⚪️"
        lines.append(
            f"{rank} `{sym}` {flag}{vmark.get(a.get('verdict'), '')} — "
            f"میانگین/معامله `{wf['expectancy']*100:+.2f}%` | "
            f"برد `{wf['win_rate']*100:.0f}%` | مرکب `{wf['total_return']*100:+.0f}%` | "
            f"معاملات `{wf['trades']}`"
        )
    head = "📊 *مقایسهٔ واچ‌لیست* — بر اساس Walk-Forward (expectancy برون‌نمونه)\n\n"
    body = "\n".join(lines) if lines else "هنوز هیچ ارزی تحلیل نشده."
    foot = ""
    if analyzed and analyzed[0][2]["walk_forward"]["expectancy"] > 0:
        foot += f"\n\n🏆 بهترین: `{analyzed[0][0]}`"
    if pending:
        foot += "\n🔘 تحلیل‌نشده: " + "، ".join(f"`{s}`" for s in pending) + " — `/analyze` بزن"
    foot += "\n_میانگینِ مثبت = اِجِ تعمیم‌پذیر؛ تضمین آینده نیست._"
    return head + body + foot


def cmd_summary(ctx, arg=None):
    """On-demand weekly performance report (default 7 days; /summary 30 for 30d)."""
    from src.live import monitor  # lazy to avoid import cycle
    try:
        days = max(1, min(365, int(arg)))
    except (ValueError, TypeError):
        days = 7
    return monitor.build_weekly_report(ctx, days=days)


def cmd_report(ctx, arg=None):
    """Re-show the stored analysis (IS/OOS + verdict) for a symbol."""
    from src.live import monitor  # lazy to avoid import cycle

    sym = _resolve_symbol(ctx, arg)
    if sym is None:
        return "نماد را مشخص کن: مثلا `/report SOL`"
    e = st.watchlist(ctx.state).get(sym)
    if not e:
        return f"`{sym}` در واچ‌لیست نیست."
    a = e.get("analysis")
    if not a:
        return f"📭 `{sym}` هنوز تحلیل نشده. اول `/analyze {sym}` را بزن."
    summary = {
        "params": e["params"], "tuned": a.get("tuned", False),
        "range": a.get("range", ["?", "?"]), "n_bars": a.get("n_bars", 0),
        "in_sample": a["in_sample"], "out_sample": a["out_sample"],
        "walk_forward": a.get("walk_forward"),
    }
    verdict = a.get("verdict") or monitor.wf_verdict(a.get("walk_forward"))
    footer = monitor.verdict_note(verdict, sym, e.get("enabled", True))
    when = f"\n_آخرین تحلیل: {_dt(e.get('analyzed_at'))}_"
    return monitor.format_analysis(summary, sym, ctx.timeframe, footer) + when


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
_HANDLERS = {
    "help": cmd_help, "start": cmd_help, "menu": cmd_menu,
    "list": cmd_list, "summary": cmd_summary, "compare": cmd_compare,
    "status": cmd_status, "position": cmd_position, "pos": cmd_position,
    "stats": cmd_stats, "price": cmd_price, "params": cmd_params, "config": cmd_params,
    "add": cmd_add, "remove": cmd_remove, "delete": cmd_remove,
    "enable": cmd_enable, "disable": cmd_disable, "analyze": cmd_analyze,
    "analyze_all": cmd_analyze_all, "analyzeall": cmd_analyze_all,
    "backtest": cmd_backtest, "backtest_all": cmd_backtest_all,
    "backtestall": cmd_backtest_all, "walkforward": cmd_walkforward,
    "wf": cmd_walkforward, "report": cmd_report,
    "xsmom": cmd_xsmom, "momentum": cmd_xsmom, "basket": cmd_xsmom,
    "xsmom_status": cmd_xsmom_status, "xsmomstatus": cmd_xsmom_status,
    "xsmom_chart": cmd_xsmom_chart, "xsmomchart": cmd_xsmom_chart,
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
