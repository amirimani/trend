"""
Live cross-sectional momentum (XS-MOM) basket.

At each rebalance we rank the configured universe by recent momentum and hold an
equal-weight basket of the strongest coins, optionally shorting the weakest
(market-neutral long/short). This is the live counterpart of src/xsmom.py and is
the strongest edge found in the research: walk-forward ~+18%/yr, ~market-neutral
and ~funding-neutral when run long/short.

It is ADVISORY — it tells you which coins to long/short on this rebalance; it
never places orders. Momentum at the latest CLOSED bar uses closes up to and
including that bar, so there is no look-ahead.
"""
from __future__ import annotations

from datetime import datetime, timezone


def compute_basket(fetch_recent_fn, universe: list[str], timeframe: str,
                   lookback: int, top_k: int, long_short: bool) -> dict:
    """Rank `universe` by `lookback`-bar momentum on the latest closed bars.

    Returns {asof, longs, shorts, moms, n, missing}. `longs`/`shorts` are lists
    of symbols (equal weight); `moms` maps symbol -> momentum (fraction).
    Coins without enough history are skipped (reported in `missing`).
    """
    moms: dict[str, float] = {}
    asof = None
    missing: list[str] = []
    need = lookback + 1
    for sym in universe:
        try:
            df = fetch_recent_fn(sym, timeframe, max(need + 5, 60))
        except Exception:
            missing.append(sym)
            continue
        if df is None or len(df) < need:
            missing.append(sym)
            continue
        closes = df["close"].astype("float64")
        m = float(closes.iloc[-1] / closes.iloc[-1 - lookback] - 1.0)
        if m != m:                       # NaN guard
            missing.append(sym)
            continue
        moms[sym] = m
        bar_ts = df.index[-1]
        if asof is None or bar_ts > asof:
            asof = bar_ts

    ranked = sorted(moms.items(), key=lambda kv: kv[1], reverse=True)
    n = len(ranked)
    k = min(top_k, n)
    longs = [s for s, _ in ranked[:k]]
    shorts = []
    if long_short and n >= 2 * k and k > 0:
        shorts = [s for s, _ in ranked[-k:]]
    return {
        "asof": (asof or datetime.now(timezone.utc)),
        "longs": longs,
        "shorts": shorts,
        "moms": moms,
        "n": n,
        "missing": missing,
    }


def _coin(sym: str) -> str:
    """Display name: 'BTC/USDT' -> 'BTC'."""
    return sym.split("/")[0]


def format_basket(basket: dict, timeframe: str, rebalance_days: int,
                  prev: dict | None = None) -> str:
    """Telegram message for a rebalance: which coins to long/short, with the
    change vs the previous basket highlighted."""
    longs, shorts, moms = basket["longs"], basket["shorts"], basket["moms"]
    asof = basket["asof"]
    asof_s = asof.strftime("%Y-%m-%d %H:%M") if hasattr(asof, "strftime") else str(asof)
    prev_longs = set(prev.get("longs", [])) if prev else set()
    prev_shorts = set(prev.get("shorts", [])) if prev else set()

    def row(sym: str, is_new: bool) -> str:
        tag = " 🆕" if is_new else ""
        return f"  `{_coin(sym):<6}` {moms.get(sym, 0.0) * 100:+6.1f}%{tag}"

    lines = [
        "🔀 *مومنتوم مقطعی (Long/Short)*",
        f"تایم‌فریم: {timeframe} | بازچینش: هر {rebalance_days} روز | "
        f"جهان: {basket['n']} ارز",
        f"تا تاریخ: {asof_s}",
        "",
        f"📈 *لانگ* ({len(longs)}):",
    ]
    if longs:
        lines += [row(s, s not in prev_longs) for s in longs]
    else:
        lines.append("  —")
    if shorts:
        lines += ["", f"📉 *شورت* ({len(shorts)}):"]
        lines += [row(s, s not in prev_shorts) for s in shorts]
    elif basket.get("n", 0) and not shorts:
        lines += ["", "_شورت غیرفعال یا ارز کافی نیست (long-only)._"]

    if prev:
        dropped = (prev_longs | prev_shorts) - (set(longs) | set(shorts))
        if dropped:
            lines += ["", "🔁 خروج از سبد: " + "، ".join(f"`{_coin(s)}`" for s in dropped)]
    if basket.get("missing"):
        lines += ["", "⚠️ بدون دادهٔ کافی: "
                  + "، ".join(f"`{_coin(s)}`" for s in basket["missing"])]
    lines += ["", "_وزن‌ها مساوی‌اند. این صرفاً پیشنهاد سبد است؛ سفارش ثبت نمی‌شود._"]
    return "\n".join(lines)
