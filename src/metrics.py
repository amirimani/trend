"""Performance metrics computed from an equity curve and a list of trades."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Default: 4h bars (6 per day). Use 365.25 days/yr for annualisation.
BARS_PER_YEAR = 6 * 365.25


def timeframe_hours(tf: str) -> float:
    """Hours per bar for a ccxt timeframe string ('15m','1h','4h','1d','1w')."""
    tf = (tf or "4h").strip().lower()
    unit = {"m": 1 / 60.0, "h": 1.0, "d": 24.0, "w": 168.0}.get(tf[-1], 1.0)
    digits = "".join(c for c in tf if c.isdigit())
    return (int(digits) if digits else 1) * unit


def periods_per_year(tf: str) -> float:
    return (365.25 * 24.0) / timeframe_hours(tf)


def _max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def compute_metrics(equity: pd.Series, trades: list, rf: float = 0.0,
                    ppy: float = BARS_PER_YEAR) -> dict:
    eq = equity.dropna()
    rets = eq.pct_change().dropna()

    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = (eq.index[-1] - eq.index[0]).total_seconds() / (365.25 * 24 * 3600)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    if rets.std(ddof=0) > 0:
        sharpe = float((rets.mean() - rf / ppy) / rets.std(ddof=0) * np.sqrt(ppy))
        # Standard downside deviation: RMS of negative excess returns over ALL bars.
        neg = np.minimum(rets - rf / ppy, 0.0)
        downside = float(np.sqrt(np.mean(neg ** 2)))
        sortino = float(rets.mean() / downside * np.sqrt(ppy)) if downside > 0 else float("nan")
    else:
        sharpe = sortino = float("nan")

    mdd = _max_drawdown(eq)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("nan")

    n = len(trades)
    if n:
        rs = np.array([t.ret for t in trades])
        wins = rs[rs > 0]
        losses = rs[rs <= 0]
        win_rate = float(len(wins) / n)
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf")
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        avg_bars = float(np.mean([t.bars_held for t in trades]))
        expectancy = float(rs.mean())
    else:
        win_rate = profit_factor = avg_win = avg_loss = avg_bars = expectancy = float("nan")

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "num_trades": n,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_bars_held": avg_bars,
        "expectancy_per_trade": expectancy,
        "final_equity": float(eq.iloc[-1]),
        "years": years,
    }


def format_metrics(m: dict) -> str:
    def pct(x):
        return f"{x*100:,.2f}%" if x == x else "n/a"
    lines = [
        f"  Total return     : {pct(m['total_return'])}",
        f"  CAGR             : {pct(m['cagr'])}",
        f"  Sharpe (ann.)    : {m['sharpe']:.2f}" if m['sharpe'] == m['sharpe'] else "  Sharpe           : n/a",
        f"  Sortino (ann.)   : {m['sortino']:.2f}" if m['sortino'] == m['sortino'] else "  Sortino          : n/a",
        f"  Max drawdown     : {pct(m['max_drawdown'])}",
        f"  Calmar           : {m['calmar']:.2f}" if m['calmar'] == m['calmar'] else "  Calmar           : n/a",
        f"  # trades         : {m['num_trades']}",
        f"  Win rate         : {pct(m['win_rate'])}",
        f"  Profit factor    : {m['profit_factor']:.2f}" if m['profit_factor'] == m['profit_factor'] else "  Profit factor    : n/a",
        f"  Avg win / loss   : {pct(m['avg_win'])} / {pct(m['avg_loss'])}",
        f"  Avg bars held    : {m['avg_bars_held']:.1f}  (~{m['avg_bars_held']*4/24:.1f} days)" if m['avg_bars_held'] == m['avg_bars_held'] else "",
        f"  Expectancy/trade : {pct(m['expectancy_per_trade'])}",
        f"  Final equity     : ${m['final_equity']:,.0f}",
    ]
    return "\n".join(l for l in lines if l)
