# Week 3 Methodology

## Objective

Week 3 develops baseline machine-learning models that predict sector
outperformance.

The implementation covers:

- Logistic Regression
- Random Forest
- next-month relative-return target definition
- time-aware cross-validation
- chronological holdout evaluation
- feature-importance analysis
- optional macroeconomic variables
- optional soft-voting ensemble

## Prediction Target

For each monthly sector observation:

```text
next_21d_sector_return
    = sector adjusted close 21 trading days later
      / current sector adjusted close - 1

equal_weight_benchmark_return
    = mean next-21-day return of XLF, XLK, and XLE

relative_return_21d
    = sector return - equal-weight benchmark return

target_outperform
    = 1 if relative_return_21d > 0
      0 otherwise
```

The signal observation is the final available trading date in each calendar
month. Twenty-one trading days serve as an approximate one-month horizon.

## Features

### Sector technical features

- daily return
- 5-, 20-, and 60-day rolling returns
- 20-day momentum
- 20- and 60-day annualized rolling volatility
- RSI(14)
- price-normalized MACD, signal, and histogram

### Cross-sectional features

For selected indicators, the model also receives:

- the three-sector mean,
- the sector's difference from that mean,
- the sector's percentile rank among XLF, XLK, and XLE.

These features are consistent with a relative-sector prediction target.

### Optional macroeconomic variables

- GDP growth
- CPI-derived year-over-year inflation
- Federal Funds Effective Rate
- growth, inflation, and rate regime tags

The macroeconomic data were approximately lagged and aligned during Week 2.

## Models

### Dummy benchmark

Predicts the most frequent training class. This is included to determine
whether the ML models improve over a trivial baseline.

### Logistic Regression

Uses:

- median imputation,
- numeric standardization,
- one-hot encoding,
- balanced class weights,
- L2 regularization.

### Random Forest

Uses:

- median imputation,
- one-hot encoding,
- 500 trees,
- limited tree depth,
- minimum leaf size,
- balanced subsample class weights.

The depth and leaf limits reduce overfitting risk on a relatively small
monthly dataset.

### Soft-voting ensemble

Averages predicted probabilities from Logistic Regression and Random Forest.

## Time-Aware Evaluation

Random train/test splitting is not used.

### Cross-validation

Expanding-window cross-validation is applied to unique monthly signal dates.
All three sector rows from one date remain in the same fold.

A one-month gap is inserted between each training and validation period.

### Final holdout

The final approximately 20% of monthly dates, with at least 12 months, are
reserved as a chronological holdout test period.

One monthly date between training and testing is excluded as a purge gap.

## Metrics

- Accuracy
- Balanced accuracy
- Precision
- Recall
- F1
- ROC AUC

Balanced accuracy is emphasized because class frequencies may not be exactly
equal.

## Feature Importance

The pipeline produces:

- holdout permutation importance for each non-dummy model,
- Logistic Regression coefficients,
- Random Forest impurity-based importance.

Permutation importance is calculated on the untouched holdout period using
balanced accuracy.

## Limitations

- The universe includes only XLF, XLK, and XLE.
- A 21-trading-day horizon is an approximation of one month.
- Macro release lags are approximate rather than exact point-in-time vintages.
- Model evaluation does not yet include transaction costs, position sizing,
  turnover, or portfolio backtesting.
- Feature importance can be diluted when predictors are correlated.

The Week 3 output is a model-development baseline, not an investment
recommendation.
