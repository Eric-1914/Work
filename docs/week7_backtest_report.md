# Week 7 Backtesting & Performance Evaluation Report

## Backtest design

The final Week 6 optimized portfolio was backtested with monthly execution, 5.0 bps transaction cost and 5.0 bps slippage per risky dollar traded.
Signals are executed conservatively after the close of the next available trading session, and the new weights become effective on the following daily return observation.
SPY benchmark data source: `data/processed/spy_benchmark.csv (downloaded via yfinance)`.
SPY is treated as a frictionless buy-and-hold benchmark after the common initial execution close. The equal-weight sector benchmark is frictionless and rebalanced monthly on the same execution dates as the strategy.

## Main results

- Week 7 strategy CAGR: 10.72%
- Week 7 strategy Sharpe: 0.848
- Week 7 strategy Sortino: 1.139
- Week 7 strategy maximum drawdown: -18.03%
- Week 7 strategy daily win rate: 48.68%
- Average one-way turnover per rebalance: 42.22%
- SPY CAGR: 19.10%; Sharpe: 1.135; max drawdown: -18.76%
- Equal-weight sectors CAGR: 22.06%; Sharpe: 1.257; max drawdown: -19.11%

## Robustness checks

58.3% of reported rolling windows had a positive Sharpe ratio.
At the highest tested trading-friction assumption (40 bps), strategy CAGR was 8.51%.

## Interpretation

Week 7 is an evaluation layer. It does not change Week 6 portfolio targets or tune parameters using future returns. Results should therefore be interpreted as historical evidence, not as a guarantee of future performance.
