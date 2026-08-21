# Week 8: Final Analysis & Reporting

## Scope

Week 8 consolidates the validated Week 6 portfolio and Week 7 backtest into a reproducible reporting layer. It does not retrain models or retune strategy parameters after observing final performance.

## Final strategy scorecard

- Total return: 23.97%
- CAGR: 10.72%
- Annualized volatility: 13.01%
- Sharpe ratio: 0.848
- Sortino ratio: 1.139
- Maximum drawdown: -18.03%
- CAGR difference vs. SPY: -8.38%
- CAGR difference vs. equal-weight sectors: -11.34%

| Metric | Week7_Strategy_Net | SPY_Benchmark | Equal_Weighted_Sectors | Strategy_minus_SPY | Strategy_minus_EqualWeight |
| --- | --- | --- | --- | --- | --- |
| Total_Return | 0.2397 | 0.4463 | 0.5232 | -0.2065 | -0.2834 |
| CAGR | 0.1072 | 0.1910 | 0.2206 | -0.0838 | -0.1134 |
| Annualized_Volatility | 0.1301 | 0.1662 | 0.1701 | -0.0360 | -0.0400 |
| Sharpe_0RF | 0.8478 | 1.1346 | 1.2572 | -0.2868 | -0.4094 |
| Sortino_0MAR | 1.1388 | 1.6813 | 1.7824 | -0.5425 | -0.6436 |
| Max_Drawdown | -0.1803 | -0.1876 | -0.1911 | 0.0072 | 0.0108 |
| Daily_Win_Rate | 0.4868 | 0.5714 | 0.5695 | -0.0846 | -0.0827 |

## Robustness checks

- Rolling windows evaluated: 24
- Median rolling Sharpe: 0.668
- Share of rolling windows with positive Sharpe: 58.33%
- Share of rolling windows with positive CAGR: 58.33%
- Worst rolling maximum drawdown: -17.28%
- Return drag from zero friction to the highest tested friction: 6.94%

| Measure | Value |
| --- | --- |
| Rolling_Window_Count | 24.0000 |
| Rolling_Sharpe_Median | 0.6683 |
| Rolling_Sharpe_Min | -1.1433 |
| Positive_Rolling_Sharpe_Share | 0.5833 |
| Positive_Rolling_CAGR_Share | 0.5833 |
| Worst_Rolling_Max_Drawdown | -0.1728 |
| Zero_Friction_Total_Return | 0.2574 |
| Base_10bps_Total_Return | 0.2397 |
| Highest_Friction_Total_Return | 0.1881 |
| Return_Drag_Zero_to_Highest_Friction | 0.0694 |
| Total_One_Way_Turnover | 10.5554 |
| Total_Trading_Cost_Fraction | 0.0142 |

## Latest optimized allocation

| Signal_Date | Asset | Target_Weight |
| --- | --- | --- |
| 2026-05-29 | XLK | 0.3738 |
| 2026-05-29 | XLF | 0.3570 |
| 2026-05-29 | CASH | 0.2692 |
| 2026-05-29 | XLE | 0.0000 |

## Data-quality status

All Week 8 reporting outputs are generated only after the required Week 6/7 inputs pass structural checks. Current audit result: **PASS**.

| Check | Passed |
| --- | --- |
| Required_Strategies_Present | True |
| Daily_Dates_Unique | True |
| Benchmark_Dates_Unique | True |
| Daily_and_Benchmark_Date_Range_Match | True |
| Daily_Equity_Positive | True |
| Daily_Returns_Finite | True |
| Rolling_Windows_Available | True |
| Cost_Sensitivity_Available | True |
| Rebalance_Log_Available | True |
| Week6_Targets_Available | True |

## Reporting outputs

- Static dashboard: `results/dashboard/week8_dashboard.html`
- Final scorecard: `results/tables/week8_final_scorecard.csv`
- Robustness summary: `results/tables/week8_robustness_summary.csv`
- Latest allocation: `results/tables/week8_latest_allocation.csv`
- Module inventory: `results/tables/week8_module_inventory.csv`
- Reproducibility manifest: `data/reporting/week8_reproducibility_manifest.json`
- Project process documentation: `docs/week8_project_documentation.md`

## Interpretation note

Week 8 is a reporting and reproducibility layer. It summarizes already-produced out-of-sample/backtest results and deliberately avoids changing model, allocation, risk, or backtest parameters after seeing final performance.
