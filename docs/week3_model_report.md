# Week 3 Machine Learning Model Report

## 1. Prediction Target

The binary target predicts whether each sector ETF will outperform the equal-weight
average of XLF, XLK, and XLE over the following 21 trading days.

A monthly signal dataset is used. Each observation is taken from the final available
trading date in a calendar month.

## 2. Models

The baseline comparison includes:

- a majority-class Dummy benchmark,
- Logistic Regression,
- Random Forest,
- an optional soft-voting ensemble combining Logistic Regression and Random Forest.

## 3. Time-Aware Evaluation

The training and test periods are separated chronologically.

- Training period: 2016-02-29 to 2024-03-28
- Purge-gap month: 2024-04-30
- Holdout test period: 2024-05-31 to 2026-05-29
- Training signal dates: 98
- Holdout signal dates: 25

Expanding-window cross-validation is performed only on the training period, with a
one-month gap before each validation fold.

## 4. Cross-Validation Summary

- Dummy: mean CV balanced accuracy 0.500, mean CV F1 0.669.
- Soft_Voting_Ensemble: mean CV balanced accuracy 0.480, mean CV F1 0.505.
- Logistic_Regression: mean CV balanced accuracy 0.475, mean CV F1 0.492.
- Random_Forest: mean CV balanced accuracy 0.446, mean CV F1 0.469.

## 5. Holdout Results

The model selected using mean cross-validation balanced accuracy was
**Soft_Voting_Ensemble**. Its previously unseen holdout results were:

- Balanced accuracy: **0.577**
- Accuracy: **0.587**
- Precision: **0.552**
- Recall: **0.471**
- F1: **0.508**
- ROC AUC: **0.560**

The Dummy benchmark holdout balanced accuracy was
**0.500**. The CV-selected model's holdout improvement
over the Dummy benchmark was
**0.077**.

## 6. Feature Importance

The highest holdout permutation-importance features for the CV-selected model were:

- Rank_MACD_Hist_Pct: 0.0379 mean permutation importance
- Relative_MACD_Hist_Pct: 0.0317 mean permutation importance
- Sector: 0.0304 mean permutation importance
- Market_Mean_MACD_Hist_Pct: 0.0227 mean permutation importance
- Rank_Rolling_Return_60D: 0.0195 mean permutation importance

Permutation importance measures the reduction in holdout balanced accuracy after a
feature is shuffled. Correlated features can share or dilute measured importance.

## 7. Interpretation

This is a baseline classification experiment, not a completed trading strategy.
Model performance does not include transaction costs, portfolio construction, or
backtesting. Those steps belong to the next project stage.

The model universe contains only three sector ETFs, so the conclusions should not be
generalized to the full market without expanding the asset universe and repeating
the time-aware evaluation.
