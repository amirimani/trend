# BTC 4h Swing Backtest — EMA cross + RSI filter + ATR stops

A from-scratch, look-ahead-free backtesting system for a trend-following swing
strategy on 4-hour BTC candles, with realistic fees and slippage and a proper
in-sample / out-of-sample split.

## Strategy

- **Entry:** EMA fast/slow cross (default 50 / 200) **or** Donchian breakout
  (`entry_mode`). Long-only by default; shorts via `allow_short=True`.
- **Entry filter (optional):** RSI(14) band (`use_rsi_filter`, default
  `50 ≤ RSI < 70`); an optional **trend-regime filter** (`regime_filter`: only
  long above a long EMA); and an optional **higher-timeframe (MTF) trend filter**
  (`htf_filter`): take the trend from a higher timeframe (e.g. daily) and only
  allow entries that agree with it, while entries still trigger on the base
  timeframe. The HTF trend is computed causally (only completed HTF bars).
- **Exit (`exit_mode`):**
  - `fixed` — ATR stop-loss + fixed ATR take-profit (default `SL 2·ATR`, `TP 4·ATR`).
  - `trailing` — ATR stop, no fixed target; a chandelier trailing stop lets
    winners run until the trend reverses (best for big trends).
  - `partial` — take part of the position at the first target, move the stop to
    break-even, then trail the remainder.
  - All modes also exit on the opposite signal (trend flip).

The offline backtest and the live engine share **one** exit implementation
(`src/position.py`), so live alerts behave exactly as backtested. `/analyze`
grid-searches the entry filter, exit mode and regime filter per coin.

## Data

Real **1-minute BTC/USD candles from Bitstamp** (2012→2025), mirrored on GitHub
([ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)),
resampled to 4h. We analyse **2018-01-01 → 2025-01-07** = 15,379 4h bars (~7 years,
well above the 3-year minimum), spanning the 2018 bear, 2020 crash, 2021 bull,
2022 bear and 2023-24 recovery.

> **Why not Binance BTC/USDT?** The environment this was built in has no network
> access to `api.binance.com` / `data.binance.vision` (HTTP 403 under the network
> policy). Bitstamp BTC/USD is the closest *reachable real* spot-BTC dataset; the
> price difference vs Binance BTC/USDT is < 0.1% and irrelevant at the 4h scale.
> To use genuine Binance data, run `python3 -m src.fetch_binance --since 2018-01-01`
> from a Binance-reachable machine (uses `ccxt`); it writes `data/btcusd_4h.parquet`
> in the same schema and the rest of the pipeline is unchanged.

## No look-ahead bias

- All indicators are causal (value at bar *t* uses only data up to *t*'s close).
- A signal seen at the close of bar *t* is executed at the **open of bar *t+1***.
- Stop/target levels are fixed at entry and only checked against *future* bars.
- If a bar touches both stop and target, the **stop is assumed to fill first**
  (pessimistic); gap-throughs fill at the bar open.

## Costs

`fee = 0.10%` per side (Binance spot taker) and `slippage = 0.05%` per side, both
charged on every fill. Position sizing is full-equity (compounding), start $10,000.

## Run

```bash
pip install -r requirements.txt
python3 run_backtest.py        # auto-downloads raw data (~91MB) on first run
```

Outputs land in `results/`: `metrics.json`, `equity_curve.csv`, `trades.csv`,
`equity_curve.png`.

## Real results (this repo, as run)

Split: in-sample `2018-01-01 → 2023-01-01`, out-of-sample `2023-01-01 → 2025-01-07`.

### Default params (EMA 50/200, RSI 50–70, SL 2·ATR, TP 4·ATR)

| Metric | Full | In-sample | Out-of-sample |
|---|---:|---:|---:|
| Total return | +36.4% | −0.7% | **+37.4%** |
| CAGR | 4.5% | −0.1% | 17.1% |
| Sharpe (ann.) | 0.48 | 0.04 | **1.87** |
| Max drawdown | −17.5% | −17.5% | −6.1% |
| Win rate | 50.0% | 36.8% | 77.8% |
| Profit factor | 1.63 | 1.05 | 7.34 |
| # trades | 28 | 19 | 9 |

### Grid search (selected on in-sample Sharpe → EMA 50/100, RSI<75, SL 1.5·ATR, TP 5·ATR)

| Metric | In-sample | Out-of-sample |
|---|---:|---:|
| Total return | +87.5% | +17.0% |
| Sharpe (ann.) | 0.83 | 0.70 |
| Max drawdown | −16.3% | −12.4% |
| Win rate | 38.1% | 35.7% |
| # trades | 42 | 14 |

### Buy & Hold benchmark (full period)
Total return **+633.7%**, CAGR 32.8%, Sharpe 0.76, **max drawdown −81.5%**.

## Honest reading of the results

- In this **strong secular bull market**, long-only buy & hold beats the strategy
  on raw return by a wide margin — but at a brutal **−81% drawdown**. The strategy
  trades return for *much* lower risk (max DD −17%) and a higher Sharpe.
- The in-sample-optimised parameters did **not** beat the default params
  out-of-sample (OOS Sharpe 0.70 vs 1.87). That is the OOS split doing its job:
  it exposes that grid-search "improvements" were partly in-sample overfitting.
- Trade counts are low (trend strategies trade rarely), so per-period metrics are
  noisy — treat single-run OOS numbers as indicative, not precise.

## Live engine — multi-symbol, Telegram-managed (Docker)

An always-on service that watches **many symbols at once**, each with its own
parameters, open position and trade history. For every enabled symbol it polls
Binance for closed 4h candles and sends a **Telegram alert** on each entry signal
(direction, entry, TP, SL, R/R), then **tracks the position** and sends a
follow-up **result** (realised P/L + R) when it resolves (TP / SL / trend-flip).

You manage the whole watchlist **from Telegram** — add/remove/enable/disable
coins — and auto-tune any coin with **`/analyze`**: the engine fetches that
symbol's history, grid-searches its best parameters (in-sample/out-of-sample),
saves them, and reports the result. It is an **advisory** service — it never
places orders.

> Unlike the backtest sandbox, your Hetzner host has normal internet access, so
> the live feed uses **genuine Binance** data (any symbol) via `ccxt`.

### What an alert looks like

```
🟢 سیگنال خرید (LONG) — BTC/USDT 4h
⏰ بستن کندل: 2025-01-04 08:00 UTC

📍 ورود (در بازار/کندل بعد): $97,882.0
🎯 حد سود (TP): $101,936.2  (+4.14%)
🛑 حد ضرر (SL): $95,854.9  (-2.07%)
⚖️ نسبت R/R: 2.00

RSI: 61.8 | EMA: 95,928/95,903 | ATR: 1,014
```

And when that position later resolves:

```
✅ نتیجهٔ پوزیشن LONG بسته شد — BTC/USDT 4h
دلیل خروج: اصابت به حد سود (TP) 🎯

📍 ورود: $10,401.0
🚪 خروج: $11,586.0
💰 سود: +11.39%  (R: +2.00)
🕒 مدت نگهداری: 24 کندل (~4.0 روز)
```

The open position is persisted in the dedup volume, so the follow-up result is
sent even if the container restarts between entry and exit. Exit resolution uses
the same intrabar rules as the backtest (stop-before-target, gap-through at open).

### Telegram commands

The bot listens for commands (long-polling `getUpdates` in a background thread),
answers only the configured `TELEGRAM_CHAT_ID`, and registers a "/" menu.

**Manage the watchlist:**

| Command | Action |
|---|---|
| `/list` | all coins with status, params summary, open position |
| `/add SOL/USDT` | add a coin (default params, enabled) |
| `/remove SOL` | remove a coin |
| `/enable SOL` / `/disable SOL` | start / pause watching (params kept) |
| `/analyze SOL/USDT` | fetch history, auto-tune params, save & report (background job) |

**Query a coin** (symbol optional when only one is watched):

| Command | Shows |
|---|---|
| `/status [SOL]` | overview, or one coin: last candle, price, open position |
| `/position SOL` | open position + **floating** P/L and R |
| `/stats SOL` | closed trades: count, win rate, total/avg R, best/worst, exit breakdown |
| `/history SOL 10` | last *n* closed trades (default 5, max 20) |
| `/price SOL` | latest price, RSI, ATR, EMAs and trend |
| `/params SOL` | that coin's active parameters (and whether tuned) |
| `/backtest SOL` | full backtest with current params → metrics + **equity-curve chart** |
| `/report SOL` | re-show the stored `/analyze` result (IS/OOS + verdict) |
| `/summary [days]` | weekly performance report (default 7d) |
| `/menu` | **glass (inline-button) menu** — tap to navigate, no typing |
| `/help` | command list |

**Glass-button menu:** `/menu` (and `/help`, and the startup message) sends an
inline keyboard. The main menu has 📋 list / 📅 weekly / one button per coin;
tapping a coin opens its submenu (price, stats, position, history, analyze,
backtest, enable/disable) — all driven by callback buttons, no commands to type.

**Periodic auto-tuning:** every `REANALYZE_DAYS` (default 30) the engine
automatically re-runs `/analyze` on the most-overdue coin (one per cycle), so
parameters stay fresh as the market regime changes; brand-new coins are tuned
automatically too. Each auto-tune posts its result (with the OOS quality guard).

The watchlist, per-coin tuned params, open positions and trade history are all
persisted in the state volume, so everything survives restarts.

### What an `/analyze` reply looks like

```
🔬 تحلیل SOL/USDT — 4h  (بهینه‌شده)
بازه: 2020-08-11 → 2025-06-18 (10500 کندل)

⚙️ پارامترهای انتخابی:
EMA 50/100 | RSI<75 | SL 1.5×ATR / TP 5.0×ATR

📊 درون‌نمونه (انتخاب): بازده +90.3% | Sharpe 0.86 | برد 39% | معاملات 41
🧪 برون‌نمونه (آزمون صادقانه): بازده +15.2% | Sharpe 0.62 | برد 33% | DD -12.4% | معاملات 15
```

### Deploy on your Hetzner Docker host

```bash
git clone <this-repo> && cd trend
cp .env.example .env          # then edit: put your bot token + chat id in .env
docker compose up -d --build  # builds the image and starts the monitor
docker compose logs -f        # watch it; you'll get a "bot activated" message
```

Get a bot token from **@BotFather** and your chat id from **@userinfobot**. Set
the initial `WATCHLIST` in `.env`; afterwards manage coins live from Telegram.
The service:
- only makes **outbound** calls (Binance + Telegram) — no open ports needed;
- `restart: unless-stopped`, so it survives reboots;
- persists the watchlist / params / positions / history in a Docker **volume**
  (`monitor-state`), so restarts never re-alert the same candle or lose tuning;
- evaluates signals only on **closed** candles (same no-look-ahead rule as the
  backtest); poll cadence is `POLL_SECONDS` (default 5 min).

`.env` only sets the **seed/default** params for new coins — each coin's real
parameters come from `/analyze`.

## Layout

```
src/data_prep.py    load 1-min raw, resample to 4h (auto-downloads data)
src/strategy.py     EMA / RSI / ATR indicators + causal signal generation
src/backtest.py     event-driven engine (next-bar fills, intrabar ATR exits, costs)
src/metrics.py      return / Sharpe / Sortino / drawdown / win-rate / PF
src/fetch_binance.py  OPTIONAL genuine Binance BTC/USDT loader (ccxt)
src/analysis.py     shared IS/OOS grid-search (used by backtest + live /analyze)
run_backtest.py     orchestration: IS/OOS split, grid search, reporting, plots

src/live/feed.py      Binance feed (ccxt): fetch_recent + fetch_history (paged)
src/live/notifier.py  Telegram Bot-API: send + getUpdates + setMyCommands
src/live/state.py     persisted watchlist + per-symbol state, migration, ops
src/live/analyzer.py  /analyze engine: fetch history -> auto-tune params
src/live/commands.py  command handlers (manage watchlist + per-symbol queries)
src/live/monitor.py   multi-symbol loop + command/analysis threads; alert->track->P/L
Dockerfile            image for the live engine
docker-compose.yml    one-command deploy (volume for state, auto-restart)
.env.example          configuration template
```
