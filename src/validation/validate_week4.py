"""Validate the Week 4 sector-rotation outputs and recompute key metrics."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.train_models_week4 import (
    CONTINUOUS_TARGET,
    DATASET_PATH,
    DATE_COL,
    EXPECTED_SECTORS,
    MACRO_REGIME_COL,
    MODEL_INPUT_COLUMNS,
    REQUIRED_MODELS,
    SECTOR_COL,
    TABLE_DIR,
    MODEL_DIR,
    FIGURE_DIR,
    DOCS_DIR,
    TARGET,
    chronological_split,
    load_dataset,
    safe_spearman,
    select_model,
)
from src.modeling.week4_estimators import positive_probability


TOLERANCE = 1e-10


def close_enough(actual: float, expected: float, label: str) -> None:
    if pd.isna(actual) and pd.isna(expected):
        return
    if not np.isclose(float(actual), float(expected), atol=TOLERANCE, rtol=1e-8):
        raise ValueError(f"{label} mismatch: expected {expected}, found {actual}.")


def check_files() -> None:
    required = [
        DATASET_PATH,
        DATASET_PATH.parent / "week4_modeling_config.json",
        TABLE_DIR / "week4_cv_metrics.csv",
        TABLE_DIR / "week4_cv_summary.csv",
        TABLE_DIR / "week4_holdout_metrics.csv",
        TABLE_DIR / "week4_confusion_matrix.csv",
        TABLE_DIR / "week4_holdout_rankings.csv",
        TABLE_DIR / "week4_selected_model_rankings.csv",
        TABLE_DIR / "week4_selected_model_signals.csv",
        TABLE_DIR / "week4_ranking_correlation.csv",
        TABLE_DIR / "week4_regime_summary.csv",
        TABLE_DIR / "week4_regime_model_usage.csv",
        TABLE_DIR / "week4_split_summary.json",
        TABLE_DIR / "week4_model_selection.json",
        MODEL_DIR / "week4_selected_model.joblib",
        FIGURE_DIR / "week4_model_comparison.png",
        DOCS_DIR / "week4_model_report.md",
    ]
    for model in REQUIRED_MODELS:
        required.append(MODEL_DIR / f"week4_{model.lower()}.joblib")
        required.append(FIGURE_DIR / f"week4_{model.lower()}_confusion_matrix.png")

    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Week 4 outputs: {missing}")


def check_cv_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = pd.read_csv(TABLE_DIR / "week4_cv_metrics.csv")
    summary = pd.read_csv(TABLE_DIR / "week4_cv_summary.csv")
    if cv.empty or summary.empty:
        raise ValueError("Week 4 cross-validation tables are empty.")
    if set(cv["model"]) != set(REQUIRED_MODELS) or set(summary["model"]) != set(REQUIRED_MODELS):
        raise ValueError("Week 4 cross-validation outputs are missing required models.")

    for model in REQUIRED_MODELS:
        folds = sorted(cv.loc[cv["model"] == model, "fold"].astype(int).tolist())
        if folds != [1, 2, 3, 4, 5]:
            raise ValueError(f"{model} does not contain exactly five CV folds.")

    bounded = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc"]
    for metric in bounded:
        values = pd.to_numeric(cv[metric], errors="coerce")
        if values.isna().all() or not values.dropna().between(0, 1).all():
            raise ValueError(f"CV metric {metric} is invalid.")

    rank = pd.to_numeric(cv["ranking_correlation"], errors="coerce").dropna()
    if not rank.empty and not rank.between(-1, 1).all():
        raise ValueError("CV ranking correlation falls outside [-1, 1].")

    # Verify summary means from raw folds instead of trusting the saved table.
    for _, row in summary.iterrows():
        raw = cv[cv["model"] == row["model"]]
        checks = {
            "mean_accuracy": raw["accuracy"].mean(),
            "mean_balanced_accuracy": raw["balanced_accuracy"].mean(),
            "mean_precision": raw["precision"].mean(),
            "mean_f1": raw["f1"].mean(),
            "mean_roc_auc": raw["roc_auc"].mean(),
            "mean_ranking_correlation": raw["ranking_correlation"].mean(),
        }
        for field, expected in checks.items():
            close_enough(row[field], expected, f"CV summary {row['model']} {field}")
    return cv, summary


def check_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_train, expected_test, expected_gap = chronological_split(data)
    split = json.loads((TABLE_DIR / "week4_split_summary.json").read_text(encoding="utf-8"))
    expected = {
        "train_start": str(expected_train[DATE_COL].min().date()),
        "train_end": str(expected_train[DATE_COL].max().date()),
        "gap_date": str(expected_gap.date()),
        "test_start": str(expected_test[DATE_COL].min().date()),
        "test_end": str(expected_test[DATE_COL].max().date()),
        "train_rows": len(expected_train),
        "test_rows": len(expected_test),
        "train_months": expected_train[DATE_COL].nunique(),
        "test_months": expected_test[DATE_COL].nunique(),
    }
    for key, value in expected.items():
        if split[key] != value:
            raise ValueError(f"Split summary field {key} is inconsistent with the dataset.")
    return expected_train, expected_test


def load_rankings() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rankings = pd.read_csv(TABLE_DIR / "week4_holdout_rankings.csv", parse_dates=[DATE_COL])
    selected_rankings = pd.read_csv(
        TABLE_DIR / "week4_selected_model_rankings.csv", parse_dates=[DATE_COL]
    )
    selected_signals = pd.read_csv(
        TABLE_DIR / "week4_selected_model_signals.csv", parse_dates=[DATE_COL]
    )
    correlations = pd.read_csv(TABLE_DIR / "week4_ranking_correlation.csv", parse_dates=[DATE_COL])
    if rankings.empty or selected_rankings.empty or selected_signals.empty or correlations.empty:
        raise ValueError("Week 4 ranking outputs are empty.")
    return rankings, selected_rankings, selected_signals, correlations


def check_rankings(
    rankings: pd.DataFrame,
    correlations: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    if set(rankings["model"]) != set(REQUIRED_MODELS):
        raise ValueError("Ranking output is missing one or more models.")

    expected_dates = set(test[DATE_COL])
    expected_rows_per_model = len(test)
    for model in REQUIRED_MODELS:
        group_model = rankings[rankings["model"] == model]
        if len(group_model) != expected_rows_per_model:
            raise ValueError(f"Unexpected ranking row count for {model}.")
        if set(group_model[DATE_COL]) != expected_dates:
            raise ValueError(f"Ranking dates do not match holdout dates for {model}.")

    probability = pd.to_numeric(rankings["Outperformance_Probability"], errors="coerce")
    if probability.isna().any() or not probability.between(0, 1).all():
        raise ValueError("Outperformance probabilities must fall inside [0, 1].")
    predicted = pd.to_numeric(rankings["Predicted_Class"], errors="coerce")
    if predicted.isna().any() or not set(predicted.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Predicted_Class must contain only 0 and 1.")
    if not np.array_equal(predicted.astype(int).to_numpy(), (probability >= 0.5).astype(int).to_numpy()):
        raise ValueError("Predicted_Class is inconsistent with the 0.5 probability threshold.")

    for (model, date), group in rankings.groupby(["model", DATE_COL]):
        if set(group[SECTOR_COL].astype(str)) != set(EXPECTED_SECTORS):
            raise ValueError(f"{model} {date.date()} does not contain exactly XLF, XLK, XLE.")
        if group[MACRO_REGIME_COL].isna().any():
            raise ValueError("Macro_Regime contains missing values in ranking output.")
        ranks = group["Sector_Rank"].astype(int)
        if set(ranks) != {1, 2, 3}:
            raise ValueError(f"Invalid sector ranks for {model} {date.date()}.")
        ordered = group.sort_values(["Outperformance_Probability", "Sector_Rank"], ascending=[False, True])
        if not ordered["Sector_Rank"].astype(int).is_monotonic_increasing:
            raise ValueError(f"Ranks are not ordered by probability for {model} {date.date()}.")
        expected_score = 1.0 - (ranks - 1) / 2.0
        if not np.allclose(group["Rank_Score_0_1"].astype(float), expected_score.astype(float)):
            raise ValueError(f"Rank_Score_0_1 is inconsistent for {model} {date.date()}.")

    # Recompute every monthly ranking correlation from the ranking file.
    expected_corr_rows = []
    for (model, date), group in rankings.groupby(["model", DATE_COL]):
        corr = safe_spearman(group["Outperformance_Probability"], group[CONTINUOUS_TARGET])
        expected_corr_rows.append((model, pd.Timestamp(date), corr, len(group)))

    actual_lookup = correlations.set_index(["model", DATE_COL])
    if len(actual_lookup) != len(expected_corr_rows):
        raise ValueError("Ranking-correlation table has the wrong number of rows.")
    for model, date, corr, n_sectors in expected_corr_rows:
        if (model, date) not in actual_lookup.index:
            raise ValueError(f"Missing ranking correlation for {model} {date.date()}.")
        row = actual_lookup.loc[(model, date)]
        close_enough(row["ranking_correlation"], corr, f"ranking correlation {model} {date.date()}")
        if int(row["n_sectors"]) != n_sectors:
            raise ValueError("Ranking correlation n_sectors is inconsistent.")


def check_holdout_metrics(rankings: pd.DataFrame, correlations: pd.DataFrame) -> None:
    metrics = pd.read_csv(TABLE_DIR / "week4_holdout_metrics.csv")
    matrix_table = pd.read_csv(TABLE_DIR / "week4_confusion_matrix.csv")
    if set(metrics["model"]) != set(REQUIRED_MODELS):
        raise ValueError("Holdout metrics are missing required models.")
    if set(matrix_table["model"]) != set(REQUIRED_MODELS):
        raise ValueError("Confusion matrix table is missing required models.")

    for model in REQUIRED_MODELS:
        group = rankings[rankings["model"] == model]
        y_true = group[TARGET].astype(int)
        y_pred = group["Predicted_Class"].astype(int)
        prob = group["Outperformance_Probability"].astype(float)
        metric_row = metrics[metrics["model"] == model].iloc[0]

        recomputed = {
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": (
                roc_auc_score(y_true, prob) if y_true.nunique() == 2 else np.nan
            ),
        }
        for field, value in recomputed.items():
            close_enough(metric_row[field], value, f"holdout {model} {field}")

        corr_rows = correlations[correlations["model"] == model]["ranking_correlation"].dropna()
        close_enough(
            metric_row["mean_ranking_correlation"],
            corr_rows.mean(),
            f"holdout {model} mean_ranking_correlation",
        )
        if int(metric_row["valid_ranking_months"]) != int(corr_rows.shape[0]):
            raise ValueError(f"valid_ranking_months is inconsistent for {model}.")
        if int(metric_row["n_holdout"]) != len(group):
            raise ValueError(f"n_holdout is inconsistent for {model}.")

        expected_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        actual_matrix = matrix_table[matrix_table["model"] == model].iloc[0]
        for field, expected_value in zip(["TN", "FP", "FN", "TP"], expected_matrix):
            if int(actual_matrix[field]) != int(expected_value):
                raise ValueError(f"Confusion matrix {field} is inconsistent for {model}.")


def check_selection_and_selected_rankings(
    summary: pd.DataFrame,
    rankings: pd.DataFrame,
    selected_rankings: pd.DataFrame,
    selected_signals: pd.DataFrame,
) -> str:
    selection = json.loads((TABLE_DIR / "week4_model_selection.json").read_text(encoding="utf-8"))
    selected = selection["selected_model"]
    expected_selected = select_model(summary)
    if selected != expected_selected:
        raise ValueError("Saved selected model does not match the stated CV selection rule.")

    expected = rankings[rankings["model"] == selected].copy()
    key_columns = [
        DATE_COL,
        SECTOR_COL,
        MACRO_REGIME_COL,
        TARGET,
        CONTINUOUS_TARGET,
        "model",
        "Outperformance_Probability",
        "Predicted_Class",
        "Sector_Rank",
        "Rank_Score_0_1",
    ]
    expected = expected[key_columns].sort_values([DATE_COL, "Sector_Rank", SECTOR_COL]).reset_index(drop=True)
    actual = selected_rankings[key_columns].sort_values([DATE_COL, "Sector_Rank", SECTOR_COL]).reset_index(drop=True)
    assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-10, atol=1e-12)

    signal_columns = [
        DATE_COL,
        SECTOR_COL,
        MACRO_REGIME_COL,
        "model",
        "Outperformance_Probability",
        "Predicted_Class",
        "Sector_Rank",
        "Rank_Score_0_1",
    ]
    if TARGET in selected_signals.columns or CONTINUOUS_TARGET in selected_signals.columns:
        raise ValueError("Week 5 signal handoff must not contain realized target columns.")
    expected_signals = expected[signal_columns].copy()
    actual_signals = selected_signals[signal_columns].sort_values(
        [DATE_COL, "Sector_Rank", SECTOR_COL]
    ).reset_index(drop=True)
    expected_signals = expected_signals.sort_values(
        [DATE_COL, "Sector_Rank", SECTOR_COL]
    ).reset_index(drop=True)
    assert_frame_equal(actual_signals, expected_signals, check_dtype=False, rtol=1e-10, atol=1e-12)
    return selected


def check_regime_usage() -> None:
    usage = pd.read_csv(TABLE_DIR / "week4_regime_model_usage.csv")
    if usage.empty:
        raise ValueError("Regime model usage table is empty.")
    if usage[MACRO_REGIME_COL].duplicated().any():
        raise ValueError("Regime model usage contains duplicate regimes.")
    weights = pd.to_numeric(usage["local_model_weight"], errors="coerce")
    if weights.isna().any() or not weights.between(0, 0.70).all():
        raise ValueError("Regime local weights fall outside the configured range.")
    fitted = usage["local_model_fitted"].astype(str).str.lower().map({"true": True, "false": False})
    if fitted.isna().any():
        raise ValueError("local_model_fitted contains invalid values.")
    if ((~fitted) & (weights != 0)).any():
        raise ValueError("A non-fitted regime has a non-zero local-model weight.")


def check_saved_models(data: pd.DataFrame, selected_model: str) -> None:
    sample = data[MODEL_INPUT_COLUMNS].head(9)
    for model in REQUIRED_MODELS:
        path = MODEL_DIR / f"week4_{model.lower()}.joblib"
        loaded = joblib.load(path)
        probability = positive_probability(loaded, sample)
        if probability.shape != (len(sample),) or not np.isfinite(probability).all():
            raise ValueError(f"Saved model failed prediction check: {path.name}")

    selected_loaded = joblib.load(MODEL_DIR / "week4_selected_model.joblib")
    saved_reference = joblib.load(MODEL_DIR / f"week4_{selected_model.lower()}.joblib")
    p1 = positive_probability(selected_loaded, sample)
    p2 = positive_probability(saved_reference, sample)
    if not np.allclose(p1, p2, rtol=1e-12, atol=1e-12):
        raise ValueError("week4_selected_model.joblib does not match the selected model file.")


def main() -> int:
    try:
        check_files()
        data = load_dataset()
        _, summary = check_cv_tables()
        _, test = check_split(data)
        rankings, selected_rankings, selected_signals, correlations = load_rankings()
        check_rankings(rankings, correlations, test)
        check_holdout_metrics(rankings, correlations)
        selected_model = check_selection_and_selected_rankings(
            summary, rankings, selected_rankings, selected_signals
        )
        check_regime_usage()
        check_saved_models(data, selected_model)

        print("[PASS] Required Week 4 files exist and the Week 3 interface is valid.")
        print("[PASS] CV summaries were recomputed from the five expanding-window folds.")
        print("[PASS] Chronological train/gap/holdout split matches the dataset.")
        print("[PASS] Precision, ROC AUC, confusion matrices, and ranking correlations were recomputed.")
        print("[PASS] Selected-model rankings and leakage-safe Week 5 signals match exactly.")
        print("[PASS] Regime-switching usage and serialized-model predictions passed validation.")
        print("[DONE] All Week 4 outputs passed validation.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 4 validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
