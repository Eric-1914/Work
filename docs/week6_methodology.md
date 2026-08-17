# Week 6: Risk Management & Optimization

## Objective

Week 6 adds a risk-management layer on top of the Week 5 combined sector portfolio. It does not create new directional signals. It changes position sizing and portfolio risk for sectors already selected by Week 5.

The implementation covers the assigned tasks:

1. integrate portfolio constraints: sector caps, turnover limits, and drawdown control,
2. apply Mean-Variance Optimization (MVO),
3. apply inverse-volatility allocation with volatility targeting,
4. implement a rule-based defensive overlay.

## Inputs

Week 6 uses:

```text
results/tables/week5_default_portfolio.csv
```

for the monthly Week 5 target portfolio, and:

```text
data/processed/sector_features.csv
```

for daily XLF, XLK, and XLE returns.

## No-Look-Ahead Rule

For each signal date, the trailing risk window contains only observations dated on or before that signal date. The resulting target is intended for the next trading session. Returns after the signal date are used only for realized performance and drawdown measurement.

## 1. Portfolio Constraints

The final Week 6 portfolio is long-only, unlevered, and uses cash for any unallocated weight.

### Sector cap

Each sector is capped at 60%. This keeps Week 6 consistent with Week 5 and prevents a single-sector concentration.

### Turnover limit

Normal one-way turnover is limited to 50% per monthly rebalance:

```text
0.5 * sum(abs(New_Target_Weight - Previous_Target_Weight))
```

The first portfolio establishment is not treated as a rebalance, so the strategy is not artificially forced to start mostly in cash.

If Week 5 removes a sector, Week 6 exits that sector even when the mandatory exit exceeds the normal turnover limit. The implementation then reallocates as much of the sale proceeds as possible without adding unnecessary turnover. Risk-reducing trades may also exceed the normal turnover limit when required by the volatility or drawdown controls.

### Drawdown control

When the live Week 6 strategy drawdown is at or below -10%, total sector exposure is capped at 50% and the remainder is held in cash.

This is a drawdown-triggered defensive rule, not a guarantee that realized maximum drawdown can never exceed 10%.

## 2. Mean-Variance Optimization

For each signal date, Week 6 estimates:

- trailing annualized mean returns,
- trailing annualized covariance,
- trailing annualized volatility.

The default lookback is 60 trading days with at least 40 complete observations. The covariance matrix is shrunk 20% toward its diagonal to reduce estimation noise.

MVO maximizes:

```text
Expected Return - Risk_Aversion * Portfolio Variance
```

subject to:

- long-only weights,
- only sectors already selected by Week 5,
- 60% maximum sector weight,
- the same gross sector exposure authorized by Week 5 before later risk scaling.

Because the project has only three sector ETFs, a deterministic 1-percentage-point grid search is used instead of adding another optimization dependency.

## 3. Inverse-Volatility Allocation and Volatility Targeting

The second optimization view uses inverse trailing volatility:

```text
Score_i = 1 / Volatility_i
```

The weights are normalized subject to the 60% sector cap and the Week 5 risky budget.

The portfolio is then scaled down if forecast annualized volatility is above 12%. Any reduced exposure is moved to cash.

This is intentionally described as inverse-volatility allocation rather than exact equal-risk-contribution risk parity. The assignment allows risk parity **or** volatility targeting, and this implementation uses volatility targeting directly.

## 4. Final Week 6 Portfolio

The final target uses a transparent 50/50 blend:

```text
50% constrained MVO
+
50% inverse-volatility allocation
```

The processing order is:

1. read the Week 5 eligible sectors and risky budget,
2. estimate trailing return and covariance inputs,
3. build constrained MVO weights,
4. build inverse-volatility weights,
5. blend the two views 50/50,
6. apply the 12% volatility target,
7. apply the 50% normal one-way turnover limit on true rebalances,
8. re-check the volatility target after turnover control,
9. apply the drawdown defensive overlay when required,
10. hold residual weight in cash.

## 5. Performance Metrics

Week 6 reports:

- total return,
- annualized return,
- annualized volatility,
- zero-risk-free-rate Sharpe ratio,
- maximum drawdown,
- average gross sector exposure,
- average one-way turnover across rebalances.

The Sharpe ratio is calculated from mean daily return divided by daily return standard deviation, annualized by the square root of 252. Initial portfolio establishment is excluded from average rebalance turnover.

## Main Outputs

Optimization targets:

```text
results/tables/week6_mvo_weights.csv
results/tables/week6_inverse_vol_weights.csv
results/tables/week6_optimized_portfolio.csv
```

Diagnostics and risk controls:

```text
results/tables/week6_optimization_diagnostics.csv
results/tables/week6_rebalance_trades.csv
results/tables/week6_constraint_audit.csv
```

Performance and risk metrics:

```text
results/tables/week6_daily_performance.csv
results/tables/week6_risk_metrics.csv
```

Figures:

```text
results/figures/week6_equity_curve.png
results/figures/week6_drawdown_comparison.png
results/figures/week6_optimized_weights.png
results/figures/week6_turnover_control.png
```

Configuration and report:

```text
data/strategy/week6_risk_config.json
docs/week6_risk_report.md
```
