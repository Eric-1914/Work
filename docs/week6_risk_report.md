# Week 6 Risk Management & Optimization Report

## Scope

Week 6 extends the Week 5 combined portfolio with explicit sector caps, turnover control, drawdown protection, constrained mean-variance optimization, inverse-volatility allocation, volatility targeting, and a rule-based defensive overlay.

## Core Risk Rules

- Maximum sector weight: 60%
- Normal maximum one-way turnover per rebalance: 50%
- Volatility target: 12% annualized
- Drawdown trigger: -10%
- Defensive maximum sector exposure: 50%
- Initial portfolio establishment is not treated as a rebalance for turnover control
- Risk-reduction exception: defensive deleveraging may exceed the normal turnover limit
- Long-only, no leverage, residual weight held in cash

## Optimization

MVO uses trailing 60-trading-day annualized mean returns and a shrunk covariance matrix. The second allocation uses inverse trailing volatility and a 12% volatility target. Both are restricted to sectors selected by the Week 5 combined signal and preserve the Week 5 risky budget before risk scaling.

The final Week 6 target blends MVO and inverse-volatility allocations 50/50, applies the 12% volatility target, limits normal rebalancing turnover, and then applies the drawdown defensive overlay when required.

## Risk Metrics

```text
                   Strategy  Daily_Observations Total_Return Annualized_Return Annualized_Volatility Sharpe_0RF Max_Drawdown Average_Gross_Exposure Average_One_Way_Turnover
Inverse_Volatility_Targeted                 532       0.2447            0.1092                0.1344     0.8388      -0.1700                 0.6140                   0.4368
            MVO_Constrained                 532       0.3305            0.1448                0.1675     0.8919      -0.2160                 0.7120                   0.4833
             Week5_Baseline                 532       0.3503            0.1529                0.1613     0.9636      -0.2038                 0.7120                   0.4872
            Week6_Optimized                 532       0.2506            0.1118                0.1305     0.8775      -0.1693                 0.5680                   0.3994
```

## Latest Week 6 Target

Signal date: 2026-05-29

```text
Asset Target_Weight Outperformance_Probability Sector_Rank
  XLF        0.3570                     0.5936           2
  XLK        0.3738                     0.6453           1
  XLE        0.0000                     0.2546           3
 CASH        0.2692                                       
```

## Constraint Audit

All saved Week 6 targets passed the configured constraint audit: True.
Defensive overlay activations: 2.

## Look-Ahead Control

Every optimization window ends on or before its signal date. Target weights are intended for the next trading session, and realized returns are used only after the corresponding signal date when computing portfolio performance and drawdown.
