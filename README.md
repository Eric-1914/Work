# Machine Learning-Based Sector Rotation Strategy

Innovation AI Internship Project

## Baseline Sector ETFs

- XLF — Financials
- XLK — Technology
- XLE — Energy

## Week 1: Data Infrastructure and Acquisition

```bash
python src/data_collection/collect_week1_data.py
python src/data_collection/validate_week1_data.py
```

## Week 2: Data Cleaning, Feature Engineering, and EDA

```bash
python run_week2.py
```

## Environment Setup

From the repository root:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Week 3: Machine Learning Model Development

Week 3:

- defines a next-21-trading-day relative outperformance target,
- builds a monthly sector-level model dataset,
- trains Logistic Regression and Random Forest,
- trains an optional soft-voting ensemble,
- performs expanding-window cross-validation,
- evaluates a chronological holdout period,
- calculates feature importance,
- generates an automatic model report,
- validates all outputs.

Prerequisite:

```text
data/processed/sector_features.csv
```

This file must already exist from the completed Week 2 pipeline.

Run:

```bash
python run_week3.py
```

A successful run ends with:

```text
[DONE] All Week 3 outputs passed validation.
[DONE] Full Week 3 pipeline completed successfully.
```

Or run each stage:

```bash
python src/modeling/build_model_dataset.py
python src/modeling/train_models.py
python src/validation/validate_week3.py
```

Expected Week 3 outputs:

```text
data/modeling/
├── model_dataset.csv
├── target_summary.csv
└── modeling_config.json

results/models/
├── dummy.joblib
├── logistic_regression.joblib
├── random_forest.joblib
└── soft_voting_ensemble.joblib

results/tables/
├── week3_cv_metrics.csv
├── week3_holdout_metrics.csv
├── week3_holdout_predictions.csv
├── week3_split_summary.json
├── week3_model_selection.json
├── week3_model_selection.json
├── logistic_regression_permutation_importance.csv
├── random_forest_permutation_importance.csv
├── soft_voting_ensemble_permutation_importance.csv
├── logistic_regression_native_importance.csv
└── random_forest_native_importance.csv

results/figures/
├── week3_model_comparison.png
├── logistic_regression_confusion_matrix.png
├── random_forest_confusion_matrix.png
├── soft_voting_ensemble_confusion_matrix.png
├── logistic_regression_permutation_importance.png
├── random_forest_permutation_importance.png
└── soft_voting_ensemble_permutation_importance.png

docs/
├── week3_methodology.md
└── week3_model_report.md
```
