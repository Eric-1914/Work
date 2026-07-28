"""
Week 3 - Train and evaluate baseline classification models.

Models:
- Dummy majority-class benchmark
- Logistic Regression
- Random Forest
- Optional soft-voting ensemble of Logistic Regression and Random Forest

Evaluation:
- Expanding-window cross-validation over unique monthly signal dates
- One-month gap between train and validation folds
- Final chronological holdout test set with a one-month purge gap
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "modeling" / "model_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

TARGET = "Target_Outperform"
DATE_COL = "Date"

NUMERIC_FEATURES = [
    "Daily_Return",
    "Rolling_Return_5D",
    "Rolling_Return_20D",
    "Rolling_Return_60D",
    "Momentum_20D",
    "Rolling_Volatility_20D",
    "Rolling_Volatility_60D",
    "RSI_14",
    "MACD_Pct",
    "MACD_Signal_Pct",
    "MACD_Hist_Pct",
    "Market_Mean_Rolling_Return_20D",
    "Relative_Rolling_Return_20D",
    "Rank_Rolling_Return_20D",
    "Market_Mean_Rolling_Return_60D",
    "Relative_Rolling_Return_60D",
    "Rank_Rolling_Return_60D",
    "Market_Mean_Momentum_20D",
    "Relative_Momentum_20D",
    "Rank_Momentum_20D",
    "Market_Mean_Rolling_Volatility_20D",
    "Relative_Rolling_Volatility_20D",
    "Rank_Rolling_Volatility_20D",
    "Market_Mean_RSI_14",
    "Relative_RSI_14",
    "Rank_RSI_14",
    "Market_Mean_MACD_Hist_Pct",
    "Relative_MACD_Hist_Pct",
    "Rank_MACD_Hist_Pct",
    "GDP_Growth",
    "Inflation_YoY",
    "FedFunds_Rate",
]

CATEGORICAL_FEATURES = [
    "Sector",
    "Growth_Regime",
    "Inflation_Regime",
    "Rate_Regime",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
RANDOM_STATE = 42


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models() -> dict[str, object]:
    logistic = Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=6,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )

    ensemble = VotingClassifier(
        estimators=[
            ("logistic", clone(logistic)),
            ("random_forest", clone(random_forest)),
        ],
        voting="soft",
        n_jobs=1,
    )

    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "Logistic_Regression": logistic,
        "Random_Forest": random_forest,
        "Soft_Voting_Ensemble": ensemble,
    }


def score_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probability: np.ndarray | None,
) -> dict[str, float]:
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }

    if probability is not None and pd.Series(y_true).nunique() == 2:
        result["roc_auc"] = roc_auc_score(y_true, probability)

    return result


def predict_probability(model: object, X: pd.DataFrame) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)
        if probability.shape[1] == 2:
            return probability[:, 1]
    return None


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    unique_dates = np.array(sorted(data[DATE_COL].unique()))
    if len(unique_dates) < 48:
        raise ValueError(
            "At least 48 monthly signal dates are required for the configured "
            "training, purge gap, and holdout evaluation."
        )

    test_months = max(12, int(np.ceil(len(unique_dates) * 0.20)))
    first_test_position = len(unique_dates) - test_months
    gap_position = first_test_position - 1

    train_dates = unique_dates[:gap_position]
    gap_date = pd.Timestamp(unique_dates[gap_position])
    test_dates = unique_dates[first_test_position:]

    train = data[data[DATE_COL].isin(train_dates)].copy()
    test = data[data[DATE_COL].isin(test_dates)].copy()

    if train.empty or test.empty:
        raise RuntimeError("Chronological train/test split produced an empty set.")

    if train[DATE_COL].max() >= test[DATE_COL].min():
        raise ValueError("Training dates overlap the holdout test period.")

    return train, test, gap_date


def expanding_date_splits(
    train_data: pd.DataFrame,
    n_splits: int = 5,
):
    unique_dates = np.array(sorted(train_data[DATE_COL].unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)

    for fold, (train_date_idx, valid_date_idx) in enumerate(
        splitter.split(unique_dates), start=1
    ):
        fold_train_dates = unique_dates[train_date_idx]
        fold_valid_dates = unique_dates[valid_date_idx]

        row_train_idx = train_data.index[
            train_data[DATE_COL].isin(fold_train_dates)
        ].to_numpy()
        row_valid_idx = train_data.index[
            train_data[DATE_COL].isin(fold_valid_dates)
        ].to_numpy()

        yield fold, row_train_idx, row_valid_idx


def save_confusion_matrix(
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Underperform", "Outperform"])
    ax.set_yticks([0, 1], labels=["Underperform", "Outperform"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{name.replace('_', ' ')} Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{name.lower()}_confusion_matrix.png",
        dpi=160,
    )
    plt.close(fig)


def save_permutation_importance(
    name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="balanced_accuracy",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    table = pd.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    table.to_csv(
        TABLE_DIR / f"{name.lower()}_permutation_importance.csv",
        index=False,
    )

    top = table.head(15).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importance_mean"])
    ax.set_xlabel("Decrease in Holdout Balanced Accuracy")
    ax.set_title(f"{name.replace('_', ' ')} Permutation Importance")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / f"{name.lower()}_permutation_importance.png",
        dpi=160,
    )
    plt.close(fig)

    return table


def save_native_importance(
    name: str,
    model: object,
) -> None:
    if name not in {"Logistic_Regression", "Random_Forest"}:
        return

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()

    if name == "Logistic_Regression":
        values = classifier.coef_[0]
        table = pd.DataFrame(
            {
                "transformed_feature": names,
                "signed_coefficient": values,
                "absolute_importance": np.abs(values),
            }
        ).sort_values("absolute_importance", ascending=False)
    else:
        values = classifier.feature_importances_
        table = pd.DataFrame(
            {
                "transformed_feature": names,
                "importance": values,
            }
        ).sort_values("importance", ascending=False)

    table.to_csv(
        TABLE_DIR / f"{name.lower()}_native_importance.csv",
        index=False,
    )


def save_model_comparison(metrics: pd.DataFrame) -> None:
    plot_data = metrics.sort_values("balanced_accuracy")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_data["model"], plot_data["balanced_accuracy"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Holdout Balanced Accuracy")
    ax.set_title("Week 3 Baseline Model Comparison")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "week3_model_comparison.png", dpi=160)
    plt.close(fig)


def write_model_report(
    metrics: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    split_summary: dict,
    importances: dict[str, pd.DataFrame],
) -> None:
    dummy = metrics[metrics["model"] == "Dummy"].iloc[0]

    cv_average = (
        cv_metrics.groupby("model", as_index=False)[
            ["balanced_accuracy", "f1", "roc_auc"]
        ]
        .mean(numeric_only=True)
        .sort_values("balanced_accuracy", ascending=False)
    )

    cv_non_dummy = cv_average[cv_average["model"] != "Dummy"].copy()
    selected_name = cv_non_dummy.iloc[0]["model"]
    selected = metrics[metrics["model"] == selected_name].iloc[0]

    selection = {
        "selection_rule": "Highest mean expanding-window CV balanced accuracy among non-dummy models",
        "selected_model": selected_name,
        "selected_model_mean_cv_balanced_accuracy": float(
            cv_non_dummy.iloc[0]["balanced_accuracy"]
        ),
    }
    (TABLE_DIR / "week3_model_selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    best_importance = importances.get(selected_name)
    top_features = []
    if best_importance is not None:
        for _, row in best_importance.head(5).iterrows():
            top_features.append(
                f"- {row['feature']}: "
                f"{row['importance_mean']:.4f} mean permutation importance"
            )

    cv_lines = []
    for _, row in cv_average.iterrows():
        cv_lines.append(
            f"- {row['model']}: mean CV balanced accuracy "
            f"{row['balanced_accuracy']:.3f}, mean CV F1 {row['f1']:.3f}."
        )

    report = f"""# Week 3 Machine Learning Model Report

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

- Training period: {split_summary['train_start']} to {split_summary['train_end']}
- Purge-gap month: {split_summary['gap_date']}
- Holdout test period: {split_summary['test_start']} to {split_summary['test_end']}
- Training signal dates: {split_summary['train_signal_dates']}
- Holdout signal dates: {split_summary['test_signal_dates']}

Expanding-window cross-validation is performed only on the training period, with a
one-month gap before each validation fold.

## 4. Cross-Validation Summary

{chr(10).join(cv_lines)}

## 5. Holdout Results

The model selected using mean cross-validation balanced accuracy was
**{selected_name}**. Its previously unseen holdout results were:

- Balanced accuracy: **{selected['balanced_accuracy']:.3f}**
- Accuracy: **{selected['accuracy']:.3f}**
- Precision: **{selected['precision']:.3f}**
- Recall: **{selected['recall']:.3f}**
- F1: **{selected['f1']:.3f}**
- ROC AUC: **{selected['roc_auc']:.3f}**

The Dummy benchmark holdout balanced accuracy was
**{dummy['balanced_accuracy']:.3f}**. The CV-selected model's holdout improvement
over the Dummy benchmark was
**{selected['balanced_accuracy'] - dummy['balanced_accuracy']:.3f}**.

## 6. Feature Importance

The highest holdout permutation-importance features for the CV-selected model were:

{chr(10).join(top_features) if top_features else "- No importance results available."}

Permutation importance measures the reduction in holdout balanced accuracy after a
feature is shuffled. Correlated features can share or dilute measured importance.

## 7. Interpretation

This is a baseline classification experiment, not a completed trading strategy.
Model performance does not include transaction costs, portfolio construction, or
backtesting. Those steps belong to the next project stage.

The model universe contains only three sector ETFs, so the conclusions should not be
generalized to the full market without expanding the asset universe and repeating
the time-aware evaluation.
"""

    (DOCS_DIR / "week3_model_report.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> int:
    try:
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                "data/modeling/model_dataset.csv is missing. "
                "Run build_model_dataset.py first."
            )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

        data = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COL])
        data = data.sort_values([DATE_COL, "Sector"]).reset_index(drop=True)

        missing = [column for column in FEATURES + [TARGET] if column not in data]
        if missing:
            raise ValueError(f"Model dataset is missing columns: {missing}")

        train, test, gap_date = chronological_split(data)

        X_train = train[FEATURES]
        y_train = train[TARGET].astype(int)
        X_test = test[FEATURES]
        y_test = test[TARGET].astype(int)

        models = build_models()
        cv_rows: list[dict] = []

        indexed_train = train.reset_index(drop=True)
        for name, base_model in models.items():
            for fold, row_train_idx, row_valid_idx in expanding_date_splits(
                indexed_train,
                n_splits=5,
            ):
                fold_model = clone(base_model)
                fold_train = indexed_train.iloc[row_train_idx]
                fold_valid = indexed_train.iloc[row_valid_idx]

                fold_model.fit(
                    fold_train[FEATURES],
                    fold_train[TARGET].astype(int),
                )
                prediction = fold_model.predict(fold_valid[FEATURES])
                probability = predict_probability(
                    fold_model,
                    fold_valid[FEATURES],
                )
                scores = score_predictions(
                    fold_valid[TARGET].astype(int),
                    prediction,
                    probability,
                )
                cv_rows.append(
                    {
                        "model": name,
                        "fold": fold,
                        "train_start": fold_train[DATE_COL].min().date(),
                        "train_end": fold_train[DATE_COL].max().date(),
                        "validation_start": fold_valid[DATE_COL].min().date(),
                        "validation_end": fold_valid[DATE_COL].max().date(),
                        "train_rows": len(fold_train),
                        "validation_rows": len(fold_valid),
                        **scores,
                    }
                )

        cv_metrics = pd.DataFrame(cv_rows)
        cv_metrics.to_csv(
            TABLE_DIR / "week3_cv_metrics.csv",
            index=False,
        )

        holdout_rows: list[dict] = []
        prediction_frame = test[
            [
                DATE_COL,
                "Sector",
                TARGET,
                "Relative_Return_21D",
                "Next_21D_Sector_Return",
                "Next_21D_EqualWeight_Return",
            ]
        ].copy()

        permutation_tables: dict[str, pd.DataFrame] = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            probability = predict_probability(model, X_test)
            scores = score_predictions(y_test, prediction, probability)

            holdout_rows.append(
                {
                    "model": name,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    **scores,
                }
            )

            prediction_frame[f"{name}_Prediction"] = prediction
            if probability is not None:
                prediction_frame[f"{name}_Probability"] = probability

            safe_name = name.lower()
            joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")
            save_confusion_matrix(name, y_test, prediction)

            if name != "Dummy":
                importance = save_permutation_importance(
                    name,
                    model,
                    X_test,
                    y_test,
                )
                permutation_tables[name] = importance
                save_native_importance(name, model)

        metrics = pd.DataFrame(holdout_rows)
        metrics.to_csv(
            TABLE_DIR / "week3_holdout_metrics.csv",
            index=False,
        )
        prediction_frame.to_csv(
            TABLE_DIR / "week3_holdout_predictions.csv",
            index=False,
            date_format="%Y-%m-%d",
        )
        save_model_comparison(metrics)

        split_summary = {
            "train_start": str(train[DATE_COL].min().date()),
            "train_end": str(train[DATE_COL].max().date()),
            "gap_date": str(gap_date.date()),
            "test_start": str(test[DATE_COL].min().date()),
            "test_end": str(test[DATE_COL].max().date()),
            "train_signal_dates": int(train[DATE_COL].nunique()),
            "test_signal_dates": int(test[DATE_COL].nunique()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
        (TABLE_DIR / "week3_split_summary.json").write_text(
            json.dumps(split_summary, indent=2),
            encoding="utf-8",
        )

        write_model_report(
            metrics,
            cv_metrics,
            split_summary,
            permutation_tables,
        )

        print("[PASS] Trained Dummy, Logistic Regression, and Random Forest.")
        print("[PASS] Trained optional soft-voting ensemble.")
        print("[PASS] Completed expanding-window cross-validation.")
        print("[PASS] Completed chronological holdout evaluation.")
        print("[PASS] Saved model files, metrics, predictions, and importance results.")
        print("[PASS] Saved docs/week3_model_report.md")
        print("[DONE] Week 3 baseline model development completed.")
        return 0

    except Exception as exc:
        print(f"[ERROR] Week 3 model training failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
