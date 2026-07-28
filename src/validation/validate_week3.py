"""
Validate all required Week 3 outputs.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "modeling"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

REQUIRED_FILES = [
    DATA_DIR / "model_dataset.csv",
    DATA_DIR / "target_summary.csv",
    DATA_DIR / "modeling_config.json",
    TABLE_DIR / "week3_cv_metrics.csv",
    TABLE_DIR / "week3_holdout_metrics.csv",
    TABLE_DIR / "week3_holdout_predictions.csv",
    TABLE_DIR / "week3_split_summary.json",
    TABLE_DIR / "week3_model_selection.json",
    DOCS_DIR / "week3_model_report.md",
    FIGURE_DIR / "week3_model_comparison.png",
]

REQUIRED_MODELS = [
    MODEL_DIR / "dummy.joblib",
    MODEL_DIR / "logistic_regression.joblib",
    MODEL_DIR / "random_forest.joblib",
    MODEL_DIR / "soft_voting_ensemble.joblib",
]

NON_DUMMY = [
    "logistic_regression",
    "random_forest",
    "soft_voting_ensemble",
]


def check_files() -> None:
    for path in REQUIRED_FILES + REQUIRED_MODELS:
        if not path.exists():
            raise FileNotFoundError(f"Missing required Week 3 output: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Week 3 output is empty: {path}")

    for model in NON_DUMMY:
        for suffix in [
            "confusion_matrix.png",
            "permutation_importance.png",
        ]:
            path = FIGURE_DIR / f"{model}_{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing figure: {path}")

        path = TABLE_DIR / f"{model}_permutation_importance.csv"
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing importance table: {path}")


def check_dataset() -> None:
    data = pd.read_csv(DATA_DIR / "model_dataset.csv", parse_dates=["Date"])

    if data.empty:
        raise ValueError("model_dataset.csv is empty.")
    if data[["Date", "Sector"]].duplicated().any():
        raise ValueError("Duplicate Date/Sector rows found in model dataset.")
    if not set(data["Target_Outperform"].unique()).issubset({0, 1}):
        raise ValueError("Target_Outperform must contain only 0 and 1.")
    if data["Date"].nunique() < 48:
        raise ValueError("Insufficient monthly dates for configured evaluation.")
    if data.groupby("Date")["Sector"].nunique().ne(3).any():
        raise ValueError("Every modeling date must contain all three sectors.")


def check_metrics() -> None:
    cv = pd.read_csv(TABLE_DIR / "week3_cv_metrics.csv")
    holdout = pd.read_csv(TABLE_DIR / "week3_holdout_metrics.csv")

    required_metrics = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
    ]
    for frame_name, frame in [("CV", cv), ("holdout", holdout)]:
        if frame.empty:
            raise ValueError(f"{frame_name} metric table is empty.")
        for metric in required_metrics:
            values = pd.to_numeric(frame[metric], errors="coerce")
            if values.isna().any() or not values.between(0, 1).all():
                raise ValueError(
                    f"{frame_name} metric {metric} falls outside [0, 1]."
                )

    required_models = {
        "Dummy",
        "Logistic_Regression",
        "Random_Forest",
        "Soft_Voting_Ensemble",
    }
    if set(holdout["model"]) != required_models:
        raise ValueError("Holdout metrics do not contain all required models.")


def check_predictions_and_split() -> None:
    predictions = pd.read_csv(
        TABLE_DIR / "week3_holdout_predictions.csv",
        parse_dates=["Date"],
    )
    if predictions.empty:
        raise ValueError("Holdout prediction file is empty.")

    probability_columns = [
        column
        for column in predictions.columns
        if column.endswith("_Probability")
    ]
    for column in probability_columns:
        values = pd.to_numeric(predictions[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"Invalid prediction probabilities in {column}.")

    summary = json.loads(
        (TABLE_DIR / "week3_split_summary.json").read_text(encoding="utf-8")
    )
    train_end = pd.Timestamp(summary["train_end"])
    gap_date = pd.Timestamp(summary["gap_date"])
    test_start = pd.Timestamp(summary["test_start"])

    if not (train_end < gap_date < test_start):
        raise ValueError("Chronological split and purge gap are invalid.")


def main() -> int:
    try:
        check_files()
        check_dataset()
        check_metrics()
        check_predictions_and_split()
        print("[PASS] Required Week 3 files and trained models exist.")
        print("[PASS] Monthly model dataset and binary target passed validation.")
        print("[PASS] Cross-validation and holdout metrics passed validation.")
        print("[PASS] Prediction probabilities and chronological split passed validation.")
        print("[DONE] All Week 3 outputs passed validation.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 3 validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
