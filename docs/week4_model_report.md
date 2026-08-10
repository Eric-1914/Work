# Week 4 Model Report

## Objective

Week 4 extends the Week 3 monthly sector-outperformance classifier with ensemble
learning, macroeconomic regime switching, sector ranking, and the required
precision, ROC AUC, confusion-matrix, and ranking-correlation evaluation.

## Selected Model

`Gradient_Boosting` was selected using development cross-validation only. The
chronological holdout set was not used for model selection.

Selection priority:

1. mean cross-validation ROC AUC,
2. mean cross-validation ranking correlation,
3. mean cross-validation precision,
4. mean cross-validation balanced accuracy.

## Holdout Results for Selected Model

- Precision: 0.5128
- ROC AUC: 0.5638
- Balanced accuracy: 0.5624
- Recall: 0.5882
- F1: 0.5479
- Mean monthly ranking correlation: 0.1000

## Ensemble Learning

The Week 4 soft-voting ensemble combines Logistic Regression, Random Forest, and
Gradient Boosting using equal probability weights. Equal weights were used to
avoid introducing a tuned weighting rule that would require an additional nested
validation layer.

## Macroeconomic Regime Switching

A composite `Macro_Regime` is formed from the existing Week 3 `Growth_Regime`,
`Inflation_Regime`, and `Rate_Regime` tags. The regime-switching model always
fits a global soft-voting ensemble. A local regime model is fitted only when the
regime has at least 36 training observations and at least 6 observations from
each target class.

Local probabilities are shrunk toward the global probability. Larger regimes
receive more local weight, with a maximum local weight of 0.70. Sparse or unseen
regimes automatically use the global ensemble.

Regime-specific models fitted on the final training period:

- Growth=Expansion|Inflation=High|Rate=Rising: 42 observations, local weight 0.377
- Growth=Expansion|Inflation=High|Rate=Stable: 54 observations, local weight 0.420
- Growth=Expansion|Inflation=Low|Rate=Stable: 81 observations, local weight 0.485
- Growth=Expansion|Inflation=Moderate|Rate=Stable: 51 observations, local weight 0.410

## Sector Ranking

For every holdout signal date, XLF, XLK, and XLE are ranked from highest to lowest
predicted outperformance probability. `Sector_Rank = 1` is the highest-ranked
sector. `Rank_Score_0_1` maps ranks 1/2/3 to 1.0/0.5/0.0 for direct Week 5 use.

Ranking quality is evaluated as the monthly Spearman correlation between model
probabilities and realized `Relative_Return_21D` values.

The clean Week 5 signal handoff excludes realized outcomes and is:

`results/tables/week4_selected_model_signals.csv`

`week4_selected_model_rankings.csv` keeps realized outcomes only for Week 4 evaluation/audit.

## Cross-Validation Summary

```text
                    model  mean_accuracy  mean_balanced_accuracy  mean_precision  std_precision  mean_f1  mean_roc_auc  std_roc_auc  mean_ranking_correlation  std_ranking_correlation  valid_folds
        Gradient_Boosting       0.454167                0.453204        0.466967       0.071221 0.498031      0.472778     0.100989                  -0.04375                 0.232387            5
            Random_Forest       0.441667                0.438194        0.455158       0.087626 0.469902      0.470905     0.091756                  -0.05000                 0.140799            5
     Soft_Voting_Ensemble       0.458333                0.458812        0.476820       0.072337 0.510628      0.467906     0.087846                  -0.04375                 0.230277            5
      Logistic_Regression       0.479167                0.474797        0.475450       0.102798 0.491797      0.458197     0.050885                  -0.00625                 0.153730            5
Regime_Switching_Ensemble       0.458333                0.457754        0.476559       0.074090 0.513939      0.456619     0.084574                  -0.05625                 0.211256            5
```
