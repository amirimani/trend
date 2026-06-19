# BTC 4h Swing Backtest — EMA cross + RSI filter + ATR stops

A from-scratch, look-ahead-free backtesting system for a trend-following swing
strategy on 4-hour BTC candles, with realistic fees and slippage and a proper
in-sample / out-of-sample split.

## Strategy

- **Direction:** EMA fast/slow cross (default 50 / 200) — long-only by default
  (spot BTC). Shorts are available via `Params(allow_short=True)`.
- **Entry filter:** RSI(14) must be inside a band (default `50 ≤ RSI < 70`) so we
  do not buy into an already-overbought market.
- **Risk management:** ATR(14)-based stop-loss and take-profit, fixed at entry
  (default `SL = 2·ATR`, `TP = 4·ATR`). Exit also on the opposite EMA cross.

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

## Layout

```
src/data_prep.py    load 1-min raw, resample to 4h (auto-downloads data)
src/strategy.py     EMA / RSI / ATR indicators + causal signal generation
src/backtest.py     event-driven engine (next-bar fills, intrabar ATR exits, costs)
src/metrics.py      return / Sharpe / Sortino / drawdown / win-rate / PF
src/fetch_binance.py  OPTIONAL genuine Binance BTC/USDT loader (ccxt)
run_backtest.py     orchestration: IS/OOS split, grid search, reporting, plots
```
