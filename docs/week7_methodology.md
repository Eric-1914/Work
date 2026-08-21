# Week 7: Backtesting & Performance Evaluation

## Objective

Week 7 evaluates the final risk-managed portfolio produced in Week 6. It does not retrain the Week 4 models, rebuild Week 5 signals, or change Week 6 optimization parameters. This keeps the project pipeline sequential and avoids using Week 7 realized returns to alter the strategy being evaluated.

The implementation covers:

1. a monthly backtesting framework,
2. transaction costs and slippage,
3. Sharpe ratio, Sortino ratio, CAGR, drawdown, win rate, and turnover,
4. comparison with SPY and a monthly equal-weighted sector benchmark,
5. rolling-window validation,
6. trading-friction sensitivity analysis.

## Inputs

The final Week 6 target portfolio:

```text
results/tables/week6_optimized_portfolio.csv
```

Daily XLF, XLK, and XLE returns:

```text
data/processed/sector_features.csv
```

A real SPY series is required. Week 7 first searches existing CSV files under `data/` for common SPY return/price layouts and rejects stale local files that do not adequately cover the backtest endpoints. If suitable SPY data are not already present, it uses the project's existing `yfinance` dependency to download SPY once and caches normalized daily returns at `data/processed/spy_benchmark.csv`. It never silently substitutes a synthetic proxy.

## Execution timing and look-ahead control

Week 6 target weights are indexed by `Signal_Date`. A signal is assumed to be known only after that day's close.

Week 7 therefore:

1. finds the first available trading date strictly after each signal date,
2. treats that date as the execution date,
3. uses the pre-trade portfolio for that execution day's close-to-close return,
4. trades after that day's close,
5. applies the new target weights from the following daily return observation.

This is deliberately conservative because the project stores daily close-to-close returns rather than open/intraday execution data.

## Portfolio accounting

Weights drift naturally between monthly rebalances as sectors earn different returns. At each rebalance, turnover is calculated from the drifted pre-trade portfolio rather than simply comparing one saved target with the previous saved target.

One-way turnover is:

```text
0.5 * sum(abs(new_weight - pre_trade_weight))
```

Trading costs are charged only on risky ETF trades; changing the accounting cash weight itself does not create an extra trading charge.

Default implementation assumptions are:

```text
Transaction cost = 5 bps per risky dollar traded
Slippage = 5 bps per risky dollar traded
Total friction = 10 bps per risky dollar traded
```

These are transparent evaluation assumptions, not values specified by the assignment.

## Benchmarks

### SPY

SPY is used as a broad U.S. equity benchmark. The code requires an actual local SPY return or price series.

### Equal-weighted sectors

The second benchmark starts from equal one-third weights in XLF, XLK, and XLE and rebalances monthly on the same execution dates as the strategy. Between rebalances, weights drift with market returns. It is kept frictionless as a transparent reference portfolio.

Benchmarks are intentionally shown frictionless because they are reference indices rather than alternative implementation portfolios.

## Performance metrics

Week 7 reports:

- Total Return
- CAGR
- Annualized Volatility
- Sharpe Ratio, using a zero risk-free rate
- Sortino Ratio, using zero minimum acceptable return
- Maximum Drawdown
- Daily Win Rate
- Rebalance Count
- Total, average, and annualized one-way turnover
- Trading-cost fraction

## Rolling validation

Strategy stability is evaluated with trailing rolling windows. The default window is 126 trading days (approximately six months), evaluated approximately every 21 trading days. A minimum of 63 daily observations is required before a window is reported.

This is an evaluation-only robustness check; it does not tune strategy parameters using future observations.

## Trading-cost sensitivity

Week 7 re-runs the identical saved Week 6 targets under total trading-friction assumptions of:

```text
0, 5, 10, 20, and 40 bps per risky dollar traded
```

For each scenario, transaction cost and slippage are split equally. This shows whether conclusions depend on an unrealistically low cost assumption without altering the Week 6 investment decisions.

## Main outputs

Tables:

```text
results/tables/week7_daily_backtest.csv
results/tables/week7_rebalance_log.csv
results/tables/week7_execution_schedule.csv
results/tables/week7_benchmark_daily.csv
results/tables/week7_performance_summary.csv
results/tables/week7_rolling_validation.csv
results/tables/week7_cost_sensitivity.csv
```

Figures:

```text
results/figures/week7_equity_curve.png
results/figures/week7_drawdown_comparison.png
results/figures/week7_rolling_sharpe.png
results/figures/week7_cost_sensitivity.png
```

Configuration and generated report:

```text
data/backtesting/week7_backtest_config.json
docs/week7_backtest_report.md
```

## Interpretation

Week 7 is designed to answer whether the final Week 6 strategy remains reasonable after realistic execution assumptions and when compared with simple benchmarks. Historical backtest results are evidence about the tested period, not a guarantee of future performance.


## Benchmark timing fairness
SPY and the equal-weight sector benchmark use the same initial execution convention as the strategy. They do not receive the first execution-date close-to-close return. The equal-weight sector benchmark is rebalanced monthly on the same execution dates as the strategy, but is kept frictionless as a simple benchmark.

## End-of-sample guard
A Week 6 signal is only backtestable when the data contain both an execution session after the signal and at least one later return observation. Week 7 fails clearly instead of charging a last-day trading cost for a position whose post-trade return cannot be observed.
