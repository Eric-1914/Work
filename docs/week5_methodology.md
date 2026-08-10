# Week 5: Strategy Design & Signal Processing

## Objective

Week 5 converts the Week 4 selected-model probabilities into actionable monthly sector-allocation targets. The implementation directly covers the requested tasks:

1. translate model outputs into actionable signals,
2. construct sector portfolios using Top-N selection, threshold filtering, and score weighting,
3. define monthly rebalancing and position-sizing logic.

## Input Interface

Week 5 reads only:

```text
results/tables/week4_selected_model_signals.csv
```

The file contains the selected Week 4 model's probability, class prediction, sector rank, rank score, and macro regime for XLF, XLK, and XLE. It intentionally excludes realized forward returns and target labels, so Week 5 portfolio construction cannot use future outcomes.

## 1. Actionable Signals

The model output is converted into the following decision fields:

- `Pass_Threshold`: probability is at least 0.50,
- `Top_N_Selected`: sector rank is in the top 2,
- `Signal_Score`: selected-model outperformance probability,
- `Default_Selected`: sector passes the probability threshold and is in the top 2,
- `Default_Target_Weight`: final combined-strategy sector weight,
- `Actionable_Signal`: `TARGET_LONG` or `NO_POSITION`.

The 0.50 threshold is the same probability cutoff already used by the Week 4 binary classifier, so no additional fitted threshold is introduced in Week 5.

## 2. Portfolio Construction

Four portfolio views are produced so each requested technique is explicit and the combined rule is also available.

### Top-N Equal Weight

- Rank XLF, XLK, and XLE by Week 4 probability.
- Select the top 2 sectors.
- Allocate 50% to each selected sector.

### Threshold Equal Weight

- Select sectors with outperformance probability greater than or equal to 0.50.
- Equal-weight all qualifying sectors.
- A single qualifying sector is capped at 60%, with the remainder held in cash.
- If no sector qualifies, the portfolio remains 100% in cash.

### Probability Score Weighted

- Use the three Week 4 outperformance probabilities as nonnegative sizing scores.
- Normalize the scores into portfolio weights.
- Apply the 60% per-sector cap and redistribute excess to the remaining sectors when possible.

### Combined Default

The default strategy uses all three requested methods in sequence:

1. threshold filter at probability >= 0.50,
2. keep at most the top 2 sectors,
3. size eligible sectors in proportion to their model probabilities,
4. cap each sector at 60%,
5. hold any unallocated weight in cash.

This produces a simple long-only allocation that favors higher-confidence sectors without forcing full investment when only one sector is eligible.

## 3. Position Sizing Constraints

The Week 5 portfolios follow these rules:

- long-only,
- no short positions,
- no leverage,
- maximum sector weight: 60%,
- portfolio weights sum to 100% including cash,
- cash absorbs weight that cannot be allocated under the selection and cap rules.

The explicit `CASH` row makes under-investment visible instead of silently renormalizing a low-conviction portfolio back to 100% risk exposure.

## 4. Rebalancing Logic

The strategy rebalances monthly using each Week 4 signal date as the decision timestamp. The target is intended for implementation on the next trading session after the signal date. Week 5 defines the execution rule but does not invent an exchange calendar date; exact trading dates can be attached later when the backtest/execution calendar is introduced.

For each asset:

```text
Trade_Weight = New_Target_Weight - Previous_Target_Weight
```

The first rebalance assumes the portfolio starts at 100% cash.

One-way turnover is reported as:

```text
0.5 * sum(abs(New_Target_Weight - Previous_Target_Weight))
```

across XLF, XLK, XLE, and cash.

## Main Outputs

Actionable signal table:

```text
results/tables/week5_actionable_signals.csv
```

Portfolio targets:

```text
results/tables/week5_top_n_weights.csv
results/tables/week5_threshold_weights.csv
results/tables/week5_score_weighted_weights.csv
results/tables/week5_default_portfolio.csv
results/tables/week5_all_portfolio_weights.csv
```

Rebalancing and strategy summaries:

```text
results/tables/week5_rebalance_trades.csv
results/tables/week5_rebalance_summary.csv
results/tables/week5_strategy_summary.csv
```

Figures:

```text
results/figures/week5_latest_target_weights.png
results/figures/week5_turnover_comparison.png
```

Configuration and report:

```text
data/strategy/week5_strategy_config.json
docs/week5_strategy_report.md
```
