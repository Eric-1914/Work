# Week 4: Machine Learning Model Development II

## Objective

Week 4 continues directly from the Week 3 monthly sector-rotation dataset and covers the four requested tasks:

1. improve robustness with ensemble learning,
2. incorporate macroeconomic regime switching,
3. rank sectors from model outputs,
4. evaluate models with precision, ROC AUC, confusion matrix, and ranking correlation.

## Week 3 Interface Preserved

Input:

```text
data/modeling/model_dataset.csv
```

The target remains:

```text
Target_Outperform = 1 if Relative_Return_21D > 0, otherwise 0
```

`Relative_Return_21D` is the sector's forward 21-trading-day return minus the equal-weight forward return of XLF, XLK, and XLE. Week 4 does not use forward-return fields or `Adj_Close` as predictors.

## 1. Ensemble Learning

Five models are evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Soft-Voting Ensemble
- Regime-Switching Ensemble

The soft-voting model combines Logistic Regression, Random Forest, and Gradient Boosting by averaging their predicted probabilities with equal weights.

Equal weights are intentional. Choosing custom weights from the same cross-validation folds and then reporting those same folds would create an optimistic tuning loop unless nested validation were added. Equal weighting therefore gives a simple, reproducible robustness improvement without adding that bias.

## 2. Macroeconomic Regime Switching

Week 3 already contains:

- `Growth_Regime`
- `Inflation_Regime`
- `Rate_Regime`

Week 4 combines them into:

```text
Growth=<state>|Inflation=<state>|Rate=<state>
```

The resulting `Macro_Regime` is used as a routing variable. The underlying base models continue to use the three Week 3 regime tags as categorical predictors; the composite routing label is not redundantly one-hot encoded inside the base models.

The regime-switching estimator trains:

- one global soft-voting ensemble on all training observations,
- one local soft-voting ensemble for a regime only when it has at least 36 observations and at least 6 observations from each target class.

Local predictions are shrunk toward the global model:

```text
local_weight = 0.70 * n_regime / (n_regime + 36)
final_probability = local_weight * local_probability
                  + (1 - local_weight) * global_probability
```

This reduces small-regime overfitting. Sparse or unseen regimes automatically fall back to the global model.

## 3. Sector Ranking

For each holdout month and each model, XLF, XLK, and XLE are ranked by predicted outperformance probability.

```text
Sector_Rank = 1  -> highest predicted probability
Sector_Rank = 2  -> middle
Sector_Rank = 3  -> lowest
```

`Rank_Score_0_1` converts the three ranks to `1.0`, `0.5`, and `0.0`.

The complete ranking output for all models is:

```text
results/tables/week4_holdout_rankings.csv
```

The selected-model evaluation ranking file is:

```text
results/tables/week4_selected_model_rankings.csv
```

The clean Week 5 signal handoff is:

```text
results/tables/week4_selected_model_signals.csv
```

The Week 5 signal file intentionally excludes `Target_Outperform` and `Relative_Return_21D` so later signal construction cannot accidentally use realized outcomes.

## 4. Evaluation

Required metrics:

- Precision
- ROC AUC
- Confusion matrix
- Ranking correlation

Additional Week 3 diagnostics are retained:

- Accuracy
- Balanced accuracy
- Recall
- F1

Ranking correlation is the monthly cross-sectional Spearman correlation between predicted outperformance probability and realized `Relative_Return_21D`. The reported holdout value is the mean across valid months.

## Time-Aware Evaluation

The Week 3 time-aware structure is preserved:

- 5-fold expanding-window cross-validation over unique monthly signal dates,
- one signal-date gap between training and each validation fold,
- final approximately 20% of signal dates as holdout, with at least 12 holdout months,
- one signal-date purge gap before the holdout.

All three sector rows from a given month stay together. No random train/test shuffle is used.

## Model Selection

The holdout set is not used for model selection. Models are selected from development cross-validation using:

1. mean ROC AUC,
2. mean ranking correlation,
3. mean precision,
4. mean balanced accuracy.

## Main Outputs

```text
results/tables/week4_cv_metrics.csv
results/tables/week4_cv_summary.csv
results/tables/week4_holdout_metrics.csv
results/tables/week4_confusion_matrix.csv
results/tables/week4_holdout_rankings.csv
results/tables/week4_selected_model_rankings.csv
results/tables/week4_selected_model_signals.csv
results/tables/week4_ranking_correlation.csv
results/tables/week4_regime_summary.csv
results/tables/week4_regime_model_usage.csv
results/tables/week4_split_summary.json
results/tables/week4_model_selection.json
```

Models:

```text
results/models/week4_logistic_regression.joblib
results/models/week4_random_forest.joblib
results/models/week4_gradient_boosting.joblib
results/models/week4_soft_voting_ensemble.joblib
results/models/week4_regime_switching_ensemble.joblib
results/models/week4_selected_model.joblib
```

Figures:

```text
results/figures/week4_model_comparison.png
results/figures/week4_*_confusion_matrix.png
```

Documentation/configuration:

```text
data/modeling/week4_modeling_config.json
docs/week4_model_report.md
```
