# Week 5 Strategy Design Report

## Scope

Week 5 converts the Week 4 selected-model probabilities into monthly long-only sector portfolio targets. The implementation covers Top-N selection, probability-threshold filtering, score weighting, monthly rebalancing, and position sizing.

## Strategy Parameters

- Top-N: 2
- Probability threshold: 0.50
- Maximum sector weight: 60%
- Rebalancing frequency: monthly
- Execution rule: next trading session after each signal date
- Portfolio constraints: long-only, no leverage, no short positions; residual allocation remains in cash

## Strategy Summary

```text
                  Strategy  Months Average_Selected_Sectors Average_Invested_Weight Average_Cash_Weight Average_One_Way_Turnover Maximum_Sector_Weight
        Top_N_Equal_Weight      25                   2.0000                  1.0000              0.0000                   0.3800                0.5000
    Threshold_Equal_Weight      25                   1.5600                  0.7120              0.2880                   0.5320                0.6000
Probability_Score_Weighted      25                   3.0000                  1.0000              0.0000                   0.1318                0.5058
          Combined_Default      25                   1.3600                  0.7120              0.2880                   0.5077                0.6000
```

## Latest Combined-Default Target

Signal date: 2026-05-29

```text
Asset Outperformance_Probability  Sector_Rank Target_Weight
  XLK                     0.6453            1        0.5209
  XLF                     0.5936            2        0.4791
  XLE                     0.2546            3        0.0000
```

Cash target: 0.0000

## Rebalancing Logic

Each month the strategy computes new target weights from the current Week 4 signal snapshot. Trade weights are the difference between the new target and the previous month's target. Reported one-way turnover is half of the absolute weight changes across XLF, XLK, XLE, and cash. The first rebalance assumes the portfolio starts fully in cash.

## Data Handling

Week 5 consumes only the leakage-safe Week 4 signal file. Realized forward returns and classification targets are not used in portfolio construction.

## Main Outputs

- `results/tables/week5_actionable_signals.csv`
- `results/tables/week5_top_n_weights.csv`
- `results/tables/week5_threshold_weights.csv`
- `results/tables/week5_score_weighted_weights.csv`
- `results/tables/week5_default_portfolio.csv`
- `results/tables/week5_all_portfolio_weights.csv`
- `results/tables/week5_rebalance_trades.csv`
- `results/tables/week5_rebalance_summary.csv`
- `results/tables/week5_strategy_summary.csv`
