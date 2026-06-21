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
from src.metrics import periods_per_year, timeframe_hours
from src.position import open_position, step
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
    """The default/seed parameters for newly added symbols (until /analyze).
    Also carries the account settings (leverage, shorts) that /analyze keeps
    fixed while it searches the strategy knobs."""
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
        leverage=_f("LEVERAGE", 1.0),
        strategy=os.getenv("STRATEGY", "trend").strip().lower(),
        regime_ema=_i("REGIME_EMA", 200),
    )


# Futures cost/risk settings.
TAKER_FEE = _f("TAKER_FEE", 0.0004)      # Binance USDT-M futures taker ~0.04%
MAINT_MARGIN = _f("MAINT_MARGIN", 0.005)  # maintenance margin (liquidation price)


def _costs():
    from src.backtest import Costs
    return Costs(fee=TAKER_FEE, maint_margin=MAINT_MARGIN)


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
# Periodic auto re-analysis: re-tune each coin whose params are older than this
# many days (also tunes never-analysed coins). 0 disables. One coin per cycle.
REANALYZE_DAYS = _i("REANALYZE_DAYS", 30)


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
    backtesting: set = field(default_factory=set)
    walkforwarding: set = field(default_factory=set)
    # injected so tests can substitute local data
    fetch_recent_fn: callable = fetch_recent
    fetch_history_fn: callable = fetch_history

    def start_analysis(self, symbol: str, auto: bool = False) -> str:
        symbol = st.normalize_symbol(symbol)
        if symbol in self.analyzing:
            return f"⏳ تحلیل `{symbol}` همین الان در حال اجراست."
        self.analyzing.add(symbol)
        threading.Thread(target=_analysis_worker, args=(self, symbol, auto), daemon=True).start()
        return f"🔬 تحلیل `{symbol}` شروع شد؛ نتیجه را به‌زودی می‌فرستم…"

    def start_analysis_all(self) -> str:
        """Re-tune EVERY coin (enabled or disabled), SEQUENTIALLY. Coins that
        pass validation are re-enabled; failing ones stay disabled."""
        syms = [s for s in st.watchlist(self.state) if s not in self.analyzing]
        if not syms:
            return "ارزی برای تحلیل نیست (یا همه در حال تحلیل‌اند)."
        for s in syms:
            self.analyzing.add(s)

        def run():
            for s in syms:
                _analysis_worker(self, s, auto=False)   # discards itself in finally
        threading.Thread(target=run, daemon=True).start()
        return ("🔬 تحلیل مجدد همهٔ ارزها شروع شد (شاملِ غیرفعال‌ها):\n"
                + "، ".join(f"`{s}`" for s in syms)
                + "\nنتایج به‌ترتیب آماده‌شدن ارسال می‌شوند…")

    def start_backtest(self, symbol: str) -> str:
        symbol = st.normalize_symbol(symbol)
        if symbol not in st.watchlist(self.state):
            return f"`{symbol}` در واچ‌لیست نیست. اول `/add {symbol}` کن."
        if symbol in self.backtesting:
            return f"⏳ بک‌تست `{symbol}` در حال اجراست."
        self.backtesting.add(symbol)
        threading.Thread(target=_backtest_worker, args=(self, symbol), daemon=True).start()
        return f"🧪 بک‌تست `{symbol}` شروع شد؛ نتیجه و نمودار به‌زودی می‌آید…"

    def start_walkforward(self, symbol: str) -> str:
        symbol = st.normalize_symbol(symbol)
        if symbol not in st.watchlist(self.state):
            return f"`{symbol}` در واچ‌لیست نیست."
        if symbol in self.walkforwarding:
            return f"⏳ Walk-Forward `{symbol}` در حال اجراست."
        self.walkforwarding.add(symbol)
        threading.Thread(target=_walkforward_worker, args=(self, symbol), daemon=True).start()
        return (f"🔁 Walk-Forward `{symbol}` شروع شد — تیون روی پنجره‌های غلتان و "
                f"معامله روی دادهٔ ندیده. کمی طول می‌کشد…")

    def start_backtest_all(self) -> str:
        """Chart-backtest every coin, SEQUENTIALLY (matplotlib isn't thread-safe
        and concurrent exchange fetches hit rate limits)."""
        syms = [s for s in st.watchlist(self.state) if s not in self.backtesting]
        if not syms:
            return "ارزی برای بک‌تست نیست (یا همه در حال اجرا)."
        for s in syms:
            self.backtesting.add(s)

        def run():
            for s in syms:
                _backtest_worker(self, s)   # discards itself in finally
        threading.Thread(target=run, daemon=True).start()
        return ("📈 بک‌تست نموداری همهٔ ارزها شروع شد:\n"
                + "، ".join(f"`{s}`" for s in syms)
                + "\nنمودارها به‌ترتیب آماده‌شدن ارسال می‌شوند…")


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
        "exit_mode": p.exit_mode, "trail_mult": p.trail_atr_mult,
        "partial_frac": p.partial_tp_frac,
    }
    if bool(row["long_entry"]) or (p.allow_short and bool(row["short_entry"])):
        is_long = bool(row["long_entry"])
        side = "LONG" if is_long else "SHORT"
        sl = price - p.atr_sl_mult * atr if is_long else price + p.atr_sl_mult * atr
        # Display target depends on the exit style.
        if p.exit_mode == "fixed":
            tgt = price + p.atr_tp_mult * atr if is_long else price - p.atr_tp_mult * atr
        elif p.exit_mode == "partial":
            tgt = price + p.partial_tp_mult * atr if is_long else price - p.partial_tp_mult * atr
        else:  # trailing -> no fixed target
            tgt = None
        rr = abs(tgt - price) / abs(price - sl) if (tgt and price != sl) else None
        lev = max(1.0, float(getattr(p, "leverage", 1.0)))
        liq = None
        if lev > 1.0:
            dist = max(0.0, 1.0 / lev - MAINT_MARGIN)
            liq = price * (1 - dist) if is_long else price * (1 + dist)
        return {**base, "kind": "entry", "side": side, "sl": sl, "tp": tgt, "rr": rr,
                "leverage": lev, "liq": liq}
    if alert_on_exit and bool(row["long_exit"]):
        return {**base, "kind": "exit", "side": "LONG"}
    if alert_on_exit and p.allow_short and bool(row["short_exit"]):
        return {**base, "kind": "exit", "side": "SHORT"}
    return None


def _open_live_position(alert: dict, p: Params) -> dict:
    """Build a persisted position from an entry alert, using the shared module."""
    side = 1 if alert["side"] == "LONG" else -1
    pos = open_position(side, alert["price"], alert["atr"], p, maint_margin=MAINT_MARGIN)
    pos["side_str"] = alert["side"]
    pos["entry_time"] = alert["time"].isoformat()
    pos["rr"] = alert.get("rr")
    pos["risk"] = abs(alert["price"] - pos["stop"])   # per-unit risk for R calc
    pos["leverage"] = float(alert.get("leverage", 1.0))
    pos["acc_pct"] = 0.0                               # accumulated realised % (on margin)
    pos["acc_r"] = 0.0                                 # accumulated realised R
    return pos


def _leg_pct_r(pos: dict, price: float) -> tuple[float, float]:
    """Return (pnl % on margin, R). The % is leveraged; R is leverage-independent."""
    e = pos["entry"]
    lev = pos.get("leverage", 1.0)
    if pos["side"] == 1:
        pct = (price / e - 1.0) * 100.0 * lev
        r = (price - e) / pos["risk"] if pos["risk"] > 0 else float("nan")
    else:
        pct = (e / price - 1.0) * 100.0 * lev
        r = (e - price) / pos["risk"] if pos["risk"] > 0 else float("nan")
    return pct, r


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


_EXIT_DESC = {
    "fixed": lambda a: f"🎯 حد سود (TP): `${fmt_price(a['tp'])}`  ({(a['tp']/a['price']-1)*100:+.2f}%)\n"
                       f"⚖️ نسبت R/R: `{a['rr']:.2f}`",
    "partial": lambda a: f"🎯 هدف اول (بستن {a['partial_frac']*100:.0f}%): `${fmt_price(a['tp'])}`"
                         f"  ({abs(a['tp']/a['price']-1)*100:+.2f}%)\n"
                         f"سپس مابقی با تریلینگ‌استاپ `{a['trail_mult']}×ATR` ادامه می‌یابد"
                         + (f" | R اولیه `{a['rr']:.2f}`" if a.get("rr") else ""),
    "trailing": lambda a: f"🏃 خروج: تریلینگ‌استاپ `{a['trail_mult']}×ATR` (بدون حد سود ثابت — روند را تا انتها سوار می‌شود)",
}


def format_alert(a: dict, symbol: str, timeframe: str) -> str:
    t = a["time"].strftime("%Y-%m-%d %H:%M UTC")
    is_long = a["side"] == "LONG"
    emoji = "🟢" if is_long else "🔴"
    word = "خرید (LONG)" if is_long else "فروش (SHORT)"
    entry, sl = a["price"], a["sl"]
    exit_line = _EXIT_DESC.get(a.get("exit_mode", "fixed"), _EXIT_DESC["fixed"])(a)
    lev = a.get("leverage", 1.0)
    lev_line = ""
    if lev and lev > 1.0:
        lev_line = f"⚡ اهرم: `{lev:g}x` (سود/زیان روی مارجین ×{lev:g})"
        if a.get("liq"):
            lev_line += f" | ☠️ لیکویید ≈ `${fmt_price(a['liq'])}`"
        lev_line += "\n"
    return (
        f"{emoji} *سیگنال {word}* — `{symbol}` {timeframe}\n"
        f"⏰ بستن کندل: `{t}`\n\n"
        f"📍 ورود (در بازار/کندل بعد): `${fmt_price(entry)}`\n"
        f"{exit_line}\n"
        f"🛑 حد ضرر (SL): `${fmt_price(sl)}`  ({(sl/entry-1)*100:+.2f}%)\n"
        f"{lev_line}\n"
        f"RSI: `{a['rsi']:.1f}` | EMA: `{fmt_price(a['ema_fast'])}/{fmt_price(a['ema_slow'])}`\n"
        f"_سیستم هشداردهنده است و سفارش ثبت نمی‌کند._"
    )


_REASON_FA = {
    "TP": "اصابت به حد سود (TP) 🎯",
    "TP1": "هدف اول (بخشی) 🎯",
    "SL": "اصابت به حد ضرر (SL) 🛑",
    "TRAIL": "تریلینگ‌استاپ 🏃",
    "EXIT": "فلیپ روند / کراس معکوس ⚪️",
}


def format_partial(pos: dict, leg: dict, pct: float, r: float, symbol: str, timeframe: str) -> str:
    return (
        f"🟡 *سود جزئی گرفته شد* — `{symbol}` {timeframe}\n"
        f"بستن `{leg['frac']*100:.0f}%` از پوزیشن {pos['side_str']} در `${fmt_price(leg['price'])}`"
        f"  (`{pct:+.2f}%`، R `{r:+.2f}`)\n"
        f"مابقی `{pos['remaining']*100:.0f}%` با تریلینگ‌استاپ ادامه دارد؛ "
        f"استاپ به `${fmt_price(pos['stop'])}` منتقل شد."
    )


def format_result(trade: dict, symbol: str, timeframe: str) -> str:
    days = trade["bars"] * timeframe_hours(timeframe) / 24
    win = trade["pnl_pct"] >= 0
    return (
        f"{'✅' if win else '❌'} *نتیجهٔ پوزیشن {trade['side']} بسته شد* — `{symbol}` {timeframe}\n"
        f"دلیل خروج نهایی: {_REASON_FA.get(trade['reason'], trade['reason'])}\n\n"
        f"📍 ورود: `${fmt_price(trade['entry'])}`\n"
        f"🚪 خروج نهایی: `${fmt_price(trade['exit'])}`\n"
        f"💰 {'سود' if win else 'ضرر'} کل: `{trade['pnl_pct']:+.2f}%`  (R: `{trade['r']:+.2f}`)\n"
        f"🕒 مدت نگهداری: `{trade['bars']}` کندل (~`{days:.1f}` روز)\n"
        f"⏰ زمان خروج: `{pd.Timestamp(trade['exit_time']).strftime('%Y-%m-%d %H:%M UTC')}`\n"
        f"_سود/زیان بر مبنای قیمت و بدون احتساب کارمزد است._"
    )


def quality_verdict(oos: dict) -> str:
    """Single-split out-of-sample rating (legacy / reference)."""
    sh = oos.get("sharpe", float("nan"))
    ret = oos.get("total_return", float("nan"))
    if sh != sh:
        return "weak"
    if sh < MIN_OOS_SHARPE or ret < 0:
        return "fail"
    if sh < WEAK_OOS_SHARPE:
        return "weak"
    return "good"


def wf_verdict(wf: dict) -> str:
    """Rate by WALK-FORWARD expectancy (window-robust, aggregated over many
    unseen folds). This drives the live enable/disable decision."""
    if not wf:
        return "weak"
    n = wf.get("trades", 0)
    exp = wf.get("expectancy")
    tot = wf.get("total_return", 0.0)
    if n < 5 or exp is None:
        return "weak"                 # too few out-of-sample trades to judge
    if exp <= 0 or tot <= 0:
        return "fail"                 # negative average -> no edge
    if exp < 0.005:
        return "weak"                 # positive but thin (<0.5%/trade)
    return "good"


def verdict_note(verdict: str, symbol: str, enabled: bool) -> str:
    if verdict == "fail":
        return (f"🛑 *Walk-Forward منفی* — `{symbol}` اِجِ تعمیم‌پذیر ندارد و خودکار "
                f"*غیرفعال* شد. برای نادیده‌گرفتن: `/enable {symbol}`")
    if verdict == "weak":
        return ("⚠️ Walk-Forward مثبتِ نازک یا نمونهٔ کم — با احتیاط (فعال ماند).")
    return "✅ *Walk-Forward مثبت* — اِجِ تعمیم‌پذیر دارد؛ فعال شد."


def _params_summary(p: dict) -> str:
    """Human-readable description of the ACTUAL chosen options."""
    if p.get("strategy") == "hold":
        lev = p.get("leverage", 1.0)
        lv = f" | اهرم {lev:g}x" if lev and lev != 1.0 else ""
        return (f"استراتژی: نگه‌داری + فیلتر خرسی\n"
                f"در بازار می‌مانی تا قیمت بالای EMA{p.get('regime_ema', 200)} است؛ "
                f"زیر آن به نقد می‌روی.{lv}")
    entry = ("بریک‌اوت دانچیان" if p.get("entry_mode") == "donchian"
             else f"کراس EMA {p['ema_fast']}/{p['ema_slow']}")
    em = p.get("exit_mode", "fixed")
    if em == "trailing":
        ex = f"تریلینگ‌استاپ `{p['trail_atr_mult']}×ATR` | SL `{p['atr_sl_mult']}×`"
    elif em == "partial":
        ex = (f"پلکانی @`{p['partial_tp_mult']}×`+تریل `{p['trail_atr_mult']}×` | "
              f"SL `{p['atr_sl_mult']}×`")
    else:
        ex = f"TP `{p['atr_tp_mult']}×` / SL `{p['atr_sl_mult']}×ATR`"
    filt = []
    if p.get("use_rsi_filter", True):
        filt.append(f"RSI<{p['rsi_long_max']:.0f}")
    if p.get("regime_filter"):
        filt.append("فیلتر روند")
    if p.get("htf_filter"):
        filt.append(f"روندِ تایم‌بالاتر ({p.get('htf') or 'خودکار'})")
    filt_s = "، ".join(filt) if filt else "بدون فیلتر"
    lev = p.get("leverage", 1.0)
    direction = "long/short" if p.get("allow_short") else "long-only"
    extra = f"\nاهرم: {lev:g}x | جهت: {direction}" if (lev and lev != 1.0) else f"\nجهت: {direction}"
    return (f"ورود: {entry}\nفیلتر: {filt_s}\nخروج: {ex}{extra}")


def format_analysis(summary: dict, symbol: str, timeframe: str, footer: str = "") -> str:
    p = summary["params"]
    is_m, oos = summary["in_sample"], summary["out_sample"]
    wf = summary.get("walk_forward") or {}
    tuned = "بهینه‌شده" if summary.get("tuned") else "پیش‌فرض (بهینه‌سازی نتیجه نداد)"
    wf_line = "🔁 *Walk-Forward:* دادهٔ کافی نبود"
    if wf.get("trades"):
        wf_line = (f"🔁 *Walk-Forward (معیارِ اصلی):* میانگین هر معامله "
                   f"`{wf['expectancy']*100:+.2f}%` | برد `{wf['win_rate']*100:.0f}%` | "
                   f"مرکب `{wf['total_return']*100:+.0f}%` | معاملات `{wf['trades']}`")
    return (
        f"🔬 *تحلیل {symbol}* — {timeframe}  ({tuned})\n"
        f"بازه: {summary['range'][0][:10]} → {summary['range'][1][:10]} "
        f"({summary['n_bars']} کندل)\n\n"
        f"⚙️ *پارامترهای انتخابی:*\n{_params_summary(p)}\n\n"
        f"{wf_line}\n"
        f"_تکِ‌اسپلیت (مرجع):_ درون‌نمونه Sharpe `{is_m['sharpe']:.2f}` / "
        f"برون‌نمونه Sharpe `{oos['sharpe']:.2f}`\n\n"
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
        o, h, l, c = (float(row["open"]), float(row["high"]),
                      float(row["low"]), float(row["close"]))
        pos = entry.get("position")
        if pos and "remaining" not in pos:
            # Legacy position from an older schema (pre-trailing) open during an
            # upgrade — drop tracking; a fresh signal will re-enter cleanly.
            entry["position"] = None
            pos = None
        if pos:
            flip = bool(row["long_exit"]) if pos["side"] == 1 else bool(row["short_exit"])
            for leg in step(pos, o, h, l, c, flip, p):
                pct, r = _leg_pct_r(pos, leg["price"])
                pos["acc_pct"] += leg["frac"] * pct
                pos["acc_r"] += leg["frac"] * r
                pos["_last_price"] = leg["price"]
                pos["_last_reason"] = leg["reason"]
                if leg["reason"] == "TP1" and pos["remaining"] > 1e-12:
                    notifier.send(format_partial(pos, leg, pct, r, symbol, timeframe))
            if pos["remaining"] <= 1e-12:   # fully closed -> record + report
                hours = timeframe_hours(timeframe)
                bars = max(1, round((ts - pd.Timestamp(pos["entry_time"])).total_seconds() / (hours * 3600)))
                trade = {
                    "side": pos["side_str"], "entry": pos["entry"],
                    "exit": pos["_last_price"], "reason": pos["_last_reason"],
                    "pnl_pct": pos["acc_pct"], "r": pos["acc_r"], "bars": bars,
                    "entry_time": pos["entry_time"], "exit_time": ts.isoformat(),
                }
                notifier.send(format_result(trade, symbol, timeframe))
                entry.setdefault("history", []).append(trade)
                entry["history"] = entry["history"][-500:]
                entry["position"] = None
        if not entry.get("position"):
            s = signal_from_row(row, ts, p, alert_on_exit=False)
            if s and s["kind"] == "entry":
                notifier.send(format_alert(s, symbol, timeframe))
                entry["position"] = _open_live_position(s, p)
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
def format_walkforward(symbol: str, timeframe: str, wf: dict) -> str:
    if wf.get("trades", 0) == 0:
        return f"🔁 *Walk-Forward {symbol}* — {timeframe}\nمعاملهٔ برون‌نمونه‌ای تولید نشد."
    exp = wf["expectancy"] * 100
    verdict = ("✅ میانگین *مثبت* (اِجِ کوچک)" if exp > 0
               else "🛑 میانگین *منفی* — اِج پایدار نیست")
    folds = "، ".join(f"{f['return']*100:+.0f}%" for f in wf["fold_detail"])
    return (
        f"🔁 *Walk-Forward {symbol}* — {timeframe}\n"
        f"معاملاتِ کاملاً برون‌نمونه روی `{wf['folds']}` پنجرهٔ غلتان:\n\n"
        f"🎯 *میانگین هر معامله (expectancy):* `{exp:+.2f}%`\n"
        f"نرخ برد: `{wf['win_rate']*100:.0f}%` | تعداد معاملات: `{wf['trades']}`\n"
        f"بازده مرکبِ OOS: `{wf['total_return']*100:+.0f}%`\n"
        f"بهترین/بدترین معامله: `{wf['best']*100:+.0f}%` / `{wf['worst']*100:+.0f}%`\n\n"
        f"به‌ازای هر پنجره: {folds}\n\n"
        f"*{verdict}*\n"
        f"_این صادقانه‌ترین آزمون است: هر معامله روی دادهٔ ندیده. تضمین آینده نیست._"
    )


def _walkforward_worker(ctx: Context, symbol: str) -> None:
    from src.analysis import walk_forward
    try:
        df = ctx.fetch_history_fn(symbol, ctx.timeframe, ANALYZE_SINCE)
        if df is None or len(df) < 400:
            raise ValueError("دادهٔ کافی نیست")
        wf = walk_forward(df, _costs(), ctx.timeframe, base=ctx.default_params)
        ctx.notifier.send(format_walkforward(symbol, ctx.timeframe, wf))
        log.info("Walk-forward done for %s.", symbol)
    except Exception as e:
        log.exception("Walk-forward failed for %s: %s", symbol, e)
        ctx.notifier.send(f"❌ Walk-Forward `{symbol}` ناموفق بود: {e}")
    finally:
        ctx.walkforwarding.discard(symbol)


def _analysis_worker(ctx: Context, symbol: str, auto: bool = False) -> None:
    try:
        summary = run_analysis(symbol, ctx.fetch_history_fn, ctx.timeframe, ANALYZE_SINCE,
                               costs=_costs(), base=ctx.default_params)
        verdict = wf_verdict(summary.get("walk_forward"))   # decision = walk-forward
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
                "walk_forward": summary.get("walk_forward"),
            }
            # Quality guard: disable on fail, RE-ENABLE when it passes again.
            wl[symbol]["enabled"] = verdict != "fail"
            enabled = wl[symbol]["enabled"]
            st.save_state(ctx.state)
        footer = verdict_note(verdict, symbol, enabled)
        prefix = "🔄 *بازتحلیل خودکار*\n" if auto else ""
        ctx.notifier.send(prefix + format_analysis(summary, symbol, ctx.timeframe, footer))
        log.info("Analysis done for %s (auto=%s, tuned=%s, verdict=%s).",
                 symbol, auto, summary["tuned"], verdict)
    except Exception as e:
        log.exception("Analysis failed for %s: %s", symbol, e)
        if not auto:
            ctx.notifier.send(f"❌ تحلیل `{symbol}` ناموفق بود: {e}")
    finally:
        ctx.analyzing.discard(symbol)


def maybe_reanalyze(ctx: Context) -> None:
    """Re-tune the single most-overdue enabled coin (one per cycle to stay gentle)."""
    if REANALYZE_DAYS <= 0:
        return
    now = datetime.now(timezone.utc)
    target = None
    with ctx.lock:
        due = []
        for s, e in st.watchlist(ctx.state).items():
            if not e.get("enabled") or s in ctx.analyzing:
                continue
            at = e.get("analyzed_at")
            age = (now - pd.Timestamp(at)).days if at else 10 ** 6
            if age >= REANALYZE_DAYS:
                due.append((age, s))
        if due:
            due.sort(reverse=True)
            target = due[0][1]
            ctx.analyzing.add(target)  # reserve before spawning
    if target:
        log.info("Auto re-analysis triggered for %s.", target)
        threading.Thread(target=_analysis_worker, args=(ctx, target, True), daemon=True).start()


# --------------------------------------------------------------------------- #
# On-demand backtest (/backtest) — runs current params over full history
# --------------------------------------------------------------------------- #
_PLOT_LOCK = threading.Lock()  # matplotlib is NOT thread-safe — serialise plotting


def _plot_equity(symbol: str, res, timeframe: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure   # OO API: avoids pyplot global state
    except Exception:
        return None
    import tempfile

    eq, bnh = res.equity, res.bnh_equity
    path = os.path.join(tempfile.gettempdir(), f"bt_{symbol.replace('/', '')}.png")
    with _PLOT_LOCK:
        fig = Figure(figsize=(11, 7))
        ax = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})
        ax[0].plot(eq.index, eq.values, label="Strategy", lw=1.3)
        if bnh is not None:
            ax[0].plot(bnh.index, bnh.values, label="Buy & Hold", lw=1.0, alpha=0.7)
        ax[0].set_yscale("log")
        ax[0].set_title(f"{symbol} {timeframe} — backtest equity (log)")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        dd = eq / eq.cummax() - 1.0
        ax[1].fill_between(dd.index, dd.values * 100, 0, color="red", alpha=0.4)
        ax[1].set_title("Drawdown (%)"); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
    return path


def _backtest_caption(symbol, p, m, bnh, df, timeframe) -> str:
    return (
        f"🧪 *بک‌تست {symbol}* — {timeframe}\n"
        f"بازه: {str(df.index[0])[:10]} → {str(df.index[-1])[:10]} ({len(df)} کندل)\n"
        f"پارامتر: EMA `{p.ema_fast}/{p.ema_slow}` | SL `{p.atr_sl_mult}×` / TP `{p.atr_tp_mult}×`\n\n"
        f"بازده کل: `{m['total_return']*100:+.1f}%` | CAGR `{m['cagr']*100:+.1f}%`\n"
        f"Sharpe `{m['sharpe']:.2f}` | حداکثر افت `{m['max_drawdown']*100:.1f}%`\n"
        f"برد `{m['win_rate']*100:.0f}%` | معاملات `{m['num_trades']}` | "
        f"Profit Factor `{m['profit_factor']:.2f}`\n"
        f"📉 خرید-و-نگه‌داری: بازده `{bnh['total_return']*100:+.1f}%` | "
        f"افت `{bnh['max_drawdown']*100:.1f}%`\n"
        f"_شامل کارمزد ۰.۱٪ و slippage ۰.۰۵٪ هر طرف._"
    )


def _backtest_worker(ctx: Context, symbol: str) -> None:
    from src.backtest import run_backtest
    from src.metrics import compute_metrics
    try:
        with ctx.lock:
            e = st.watchlist(ctx.state).get(symbol)
            p = st.dict_to_params(e["params"]) if e else ctx.default_params
        df = ctx.fetch_history_fn(symbol, ctx.timeframe, ANALYZE_SINCE)
        if df is None or len(df) < 250:
            raise ValueError("دادهٔ کافی نیست")
        res = run_backtest(df, p, _costs())
        ppy = periods_per_year(ctx.timeframe)
        m = compute_metrics(res.equity, res.trades, ppy=ppy)
        bnh = compute_metrics(res.bnh_equity, [], ppy=ppy)
        caption = _backtest_caption(symbol, p, m, bnh, df, ctx.timeframe)
        png = _plot_equity(symbol, res, ctx.timeframe)
        if png:
            ctx.notifier.send_photo(png, caption)
            try:
                os.remove(png)
            except OSError:
                pass
        else:
            ctx.notifier.send(caption)
        log.info("Backtest done for %s.", symbol)
    except Exception as e:
        log.exception("Backtest failed for %s: %s", symbol, e)
        ctx.notifier.send(f"❌ بک‌تست `{symbol}` ناموفق بود: {e}")
    finally:
        ctx.backtesting.discard(symbol)


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
                cq = u.get("callback_query")
                if cq:  # a glass (inline) button was tapped
                    chat = cq.get("message", {}).get("chat", {}).get("id")
                    ctx.notifier.answer_callback(cq.get("id", ""))
                    if str(chat) != str(ctx.notifier.chat_id):
                        continue
                    data = (cq.get("data") or "").strip()
                    if data:
                        with ctx.lock:
                            reply = commands.dispatch("/" + data, ctx)
                        _send_reply(ctx, reply)
                    continue
                msg = u.get("message") or u.get("edited_message") or {}
                if str(msg.get("chat", {}).get("id")) != str(ctx.notifier.chat_id):
                    continue
                text = (msg.get("text") or "").strip()
                if not text.startswith("/"):
                    continue
                with ctx.lock:  # serialise command handling vs the market/analysis threads
                    reply = commands.dispatch(text, ctx)
                _send_reply(ctx, reply)
        except Exception as e:
            log.exception("Command loop error: %s", e)
            time.sleep(5)


def _send_reply(ctx: Context, reply) -> None:
    """A handler reply may be plain text or {'text':..., 'keyboard':...}."""
    if not reply:
        return
    if isinstance(reply, dict):
        ctx.notifier.send(reply.get("text", ""), reply_markup=reply.get("keyboard"))
    else:
        ctx.notifier.send(reply)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_context() -> Context:
    notifier = TelegramNotifier()
    default_p = load_params()
    raw = st.load_state()
    existed = raw.get("version") == 2 and "watchlist" in raw
    state = st.migrate(raw, st.normalize_symbol(WATCHLIST_SEED[0]),
                       default_p, seed_symbols=WATCHLIST_SEED)
    # If the timeframe changed (e.g. 4h -> 1d), reset per-symbol bar tracking and
    # open positions (they belong to the old timeframe); params/history are kept.
    if state.get("timeframe") and state.get("timeframe") != TIMEFRAME:
        for e in st.watchlist(state).values():
            e["last_bar"] = None
            e["position"] = None
        log.info("Timeframe changed %s -> %s: reset bar tracking & open positions.",
                 state.get("timeframe"), TIMEFRAME)
    state["timeframe"] = TIMEFRAME
    try:
        st.save_state(state)
    except OSError as e:
        log.error("CANNOT WRITE STATE at %s (%s) — watchlist will NOT persist! "
                  "Check the Docker volume mount/permissions.", st.STATE_FILE, e)
    log.info("State %s at %s | watchlist: %s",
             "LOADED (persisted)" if existed else "INITIALISED (fresh seed)",
             st.STATE_FILE, list(st.watchlist(state)))
    return Context(notifier=notifier, state=state, default_params=default_p)


def main():
    from src.live import commands
    ctx = build_context()
    symbols = list(st.watchlist(ctx.state))
    log.info("Live monitor: watching %s | poll=%ss", symbols, POLL_SECONDS)
    ctx.notifier.send(
        "🤖 *موتور رصد چندارزی فعال شد*\n"
        f"واچ‌لیست: {', '.join(f'`{s}`' for s in symbols) or '—'}\n"
        f"تایم‌فریم: {TIMEFRAME} | بازهٔ بررسی: هر {POLL_SECONDS//60} دقیقه\n"
        f"بازتحلیل خودکار: هر {REANALYZE_DAYS} روز\n"
        f"یک گزینه را انتخاب کن یا /help را بفرست:",
        reply_markup=commands.main_menu_kb(ctx),
    )
    threading.Thread(target=command_loop, args=(ctx,), daemon=True).start()
    while True:
        try:
            check_all(ctx)
            maybe_send_weekly(ctx)
            maybe_reanalyze(ctx)
        except Exception as e:
            log.exception("Cycle error: %s", e)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
