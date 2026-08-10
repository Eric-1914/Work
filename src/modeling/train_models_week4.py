"""Week 4 - Machine Learning Model Development II.

This module extends the Week 3 monthly sector-classification dataset
with ensemble learning, macroeconomic regime switching, sector ranking, and the
required Week 4 evaluation metrics.
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling.week4_estimators import RegimeSwitchingEnsemble, positive_probability


DATASET_PATH = PROJECT_ROOT / "data" / "modeling" / "model_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "results" / "models"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

TARGET = "Target_Outperform"
CONTINUOUS_TARGET = "Relative_Return_21D"
DATE_COL = "Date"
SECTOR_COL = "Sector"
MACRO_REGIME_COL = "Macro_Regime"
EXPECTED_SECTORS = ("XLF", "XLK", "XLE")
RANDOM_STATE = 42

# Exact Week 3 numeric predictor interface. Forward-looking target columns and
# Adj_Close are intentionally excluded from the predictor list.
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

# The three Week 3 regime tags remain ordinary categorical predictors. The new
# composite Macro_Regime is used to route observations in the regime-switching
# estimator but is deliberately not one-hot encoded again inside each base model.
CATEGORICAL_FEATURES = [
    "Sector",
    "Growth_Regime",
    "Inflation_Regime",
    "Rate_Regime",
]
MODEL_INPUT_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [MACRO_REGIME_COL]

REQUIRED_MODELS = [
    "Logistic_Regression",
    "Random_Forest",
    "Gradient_Boosting",
    "Soft_Voting_Ensemble",
    "Regime_Switching_Ensemble",
]

SELECTION_COLUMNS = [
    "mean_roc_auc",
    "mean_ranking_correlation",
    "mean_precision",
    "mean_balanced_accuracy",
]


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"model_dataset.csv is missing required Week 3 columns: {missing}")


def add_macro_regime(data: pd.DataFrame) -> pd.DataFrame:
    """Combine the Week 3 growth, inflation, and rate regime tags."""
    data = data.copy()
    require_columns(data, CATEGORICAL_FEATURES)
    growth = data["Growth_Regime"].astype("string").fillna("Unknown").astype(str)
    inflation = data["Inflation_Regime"].astype("string").fillna("Unknown").astype(str)
    rate = data["Rate_Regime"].astype("string").fillna("Unknown").astype(str)
    data[MACRO_REGIME_COL] = (
        "Growth=" + growth + "|Inflation=" + inflation + "|Rate=" + rate
    )
    return data


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "data/modeling/model_dataset.csv is missing. Complete Week 3 first."
        )

    data = pd.read_csv(DATASET_PATH, parse_dates=[DATE_COL])
    require_columns(
        data,
        [DATE_COL, SECTOR_COL, TARGET, CONTINUOUS_TARGET]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES,
    )
    if data.empty:
        raise ValueError("model_dataset.csv is empty.")

    data = data.sort_values([DATE_COL, SECTOR_COL]).reset_index(drop=True)
    data = add_macro_regime(data)

    if data[[DATE_COL, SECTOR_COL]].duplicated().any():
        raise ValueError("Duplicate Date/Sector observations found in model_dataset.csv.")

    actual_sectors = set(data[SECTOR_COL].astype(str).unique())
    if actual_sectors != set(EXPECTED_SECTORS):
        raise ValueError(
            f"Week 4 expects sectors {list(EXPECTED_SECTORS)}, found {sorted(actual_sectors)}."
        )
    for date, group in data.groupby(DATE_COL, sort=False):
        if set(group[SECTOR_COL].astype(str)) != set(EXPECTED_SECTORS):
            raise ValueError(f"Signal date {date.date()} does not contain exactly XLF, XLK, XLE.")

    target = pd.to_numeric(data[TARGET], errors="coerce")
    if target.isna().any() or not set(target.unique()).issubset({0, 1}):
        raise ValueError("Target_Outperform must contain only non-missing 0 and 1 values.")
    if target.nunique() != 2:
        raise ValueError("Target_Outperform must contain both classes 0 and 1.")
    data[TARGET] = target.astype(int)

    continuous = pd.to_numeric(data[CONTINUOUS_TARGET], errors="coerce")
    if continuous.isna().any() or not np.isfinite(continuous.to_numpy()).all():
        raise ValueError("Relative_Return_21D must contain finite numeric values.")
    data[CONTINUOUS_TARGET] = continuous.astype(float)

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Numeric feature {column} contains missing/non-finite values.")
        data[column] = values.astype(float)

    for column in CATEGORICAL_FEATURES:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Categorical feature {column} contains missing/blank values.")

    if data[DATE_COL].nunique() < 48:
        raise ValueError("At least 48 monthly signal dates are required for Week 4.")
    return data


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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


def build_base_models() -> dict[str, object]:
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
                    n_estimators=300,
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

    gradient_boosting = Pipeline(
        [
            ("preprocessor", make_preprocessor(scale_numeric=False)),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=180,
                    learning_rate=0.035,
                    max_depth=2,
                    min_samples_leaf=4,
                    subsample=0.85,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    return {
        "Logistic_Regression": logistic,
        "Random_Forest": random_forest,
        "Gradient_Boosting": gradient_boosting,
    }


def build_models() -> dict[str, object]:
    base = build_base_models()

    # Equal weights avoid introducing an arbitrary tuning decision that would
    # require nested validation to evaluate without optimistic bias.
    soft_voting = VotingClassifier(
        estimators=[
            ("logistic", clone(base["Logistic_Regression"])),
            ("random_forest", clone(base["Random_Forest"])),
            ("gradient_boosting", clone(base["Gradient_Boosting"])),
        ],
        voting="soft",
        weights=[1.0, 1.0, 1.0],
        n_jobs=1,
    )

    regime_switching = RegimeSwitchingEnsemble(
        base_estimator=clone(soft_voting),
        regime_col=MACRO_REGIME_COL,
        min_regime_samples=36,   # 12 full monthly observations for three sectors
        min_class_samples=6,
        max_regime_weight=0.70,
        shrinkage_samples=36.0,
    )

    return {
        **base,
        "Soft_Voting_Ensemble": soft_voting,
        "Regime_Switching_Ensemble": regime_switching,
    }


def score_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": np.nan,
    }
    if pd.Series(y_true).nunique() == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, probability))
    return result


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Preserve the Week 3 chronological holdout and one-month purge gap."""
    unique_dates = np.array(sorted(data[DATE_COL].unique()))
    if len(unique_dates) < 48:
        raise ValueError("Insufficient signal dates for the configured time-aware split.")

    test_months = max(12, int(np.ceil(len(unique_dates) * 0.20)))
    first_test_position = len(unique_dates) - test_months
    gap_position = first_test_position - 1
    if gap_position <= 0:
        raise ValueError("Insufficient training history after reserving holdout and gap.")

    train_dates = unique_dates[:gap_position]
    gap_date = pd.Timestamp(unique_dates[gap_position])
    test_dates = unique_dates[first_test_position:]

    train = data[data[DATE_COL].isin(train_dates)].copy().reset_index(drop=True)
    test = data[data[DATE_COL].isin(test_dates)].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise RuntimeError("Chronological train/test split produced an empty set.")
    if train[DATE_COL].max() >= gap_date or gap_date >= test[DATE_COL].min():
        raise ValueError("Chronological train/gap/test ordering is invalid.")
    return train, test, gap_date


def expanding_date_splits(train_data: pd.DataFrame, n_splits: int = 5):
    """Yield expanding-window folds over unique dates with a one-date gap."""
    unique_dates = np.array(sorted(train_data[DATE_COL].unique()))
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=1)

    for fold, (train_date_idx, valid_date_idx) in enumerate(splitter.split(unique_dates), start=1):
        fold_train_dates = unique_dates[train_date_idx]
        fold_valid_dates = unique_dates[valid_date_idx]
        fold_train = train_data[train_data[DATE_COL].isin(fold_train_dates)].copy()
        fold_valid = train_data[train_data[DATE_COL].isin(fold_valid_dates)].copy()
        if fold_train[DATE_COL].max() >= fold_valid[DATE_COL].min():
            raise ValueError(f"CV fold {fold} is not chronological.")
        yield fold, fold_train, fold_valid


def safe_spearman(predicted: pd.Series, realized: pd.Series) -> float:
    """Spearman correlation without introducing a SciPy dependency."""
    pair = pd.DataFrame(
        {
            "predicted": pd.to_numeric(predicted, errors="coerce"),
            "realized": pd.to_numeric(realized, errors="coerce"),
        }
    ).dropna()
    if len(pair) < 2:
        return np.nan
    if pair["predicted"].nunique() < 2 or pair["realized"].nunique() < 2:
        return np.nan
    predicted_rank = pair["predicted"].rank(method="average")
    realized_rank = pair["realized"].rank(method="average")
    return float(predicted_rank.corr(realized_rank))


def mean_ranking_correlation(frame: pd.DataFrame, probability_col: str) -> tuple[float, int]:
    correlations: list[float] = []
    for _, group in frame.groupby(DATE_COL):
        correlation = safe_spearman(group[probability_col], group[CONTINUOUS_TARGET])
        if not np.isnan(correlation):
            correlations.append(correlation)
    if not correlations:
        return np.nan, 0
    return float(np.mean(correlations)), len(correlations)


def save_confusion_matrix_figure(name: str, y_true: pd.Series, y_pred: np.ndarray) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    ax.set_xticks([0, 1], labels=["Underperform", "Outperform"])
    ax.set_yticks([0, 1], labels=["Underperform", "Outperform"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Week 4 {name.replace('_', ' ')} Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"week4_{name.lower()}_confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_model_comparison_figure(holdout_metrics: pd.DataFrame) -> None:
    plot_data = holdout_metrics.sort_values("roc_auc")
    y = np.arange(len(plot_data))
    height = 0.24
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(y - height, plot_data["precision"], height=height, label="Precision")
    ax.barh(y, plot_data["roc_auc"], height=height, label="ROC AUC")
    rank_scaled = (plot_data["mean_ranking_correlation"].fillna(0.0) + 1.0) / 2.0
    ax.barh(y + height, rank_scaled, height=height, label="Rank corr. scaled to [0,1]")
    ax.set_yticks(y, labels=plot_data["model"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Holdout metric value")
    ax.set_title("Week 4 Model Comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "week4_model_comparison.png", dpi=160)
    plt.close(fig)


def cross_validate(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for fold, fold_train, fold_valid in expanding_date_splits(train, n_splits=5):
        X_train = fold_train[MODEL_INPUT_COLUMNS]
        y_train = fold_train[TARGET]
        X_valid = fold_valid[MODEL_INPUT_COLUMNS]
        y_valid = fold_valid[TARGET]

        if y_train.nunique() < 2:
            raise ValueError(f"Week 4 CV fold {fold} training set contains one class.")

        for name, model in build_models().items():
            model.fit(X_train, y_train)
            probability = positive_probability(model, X_valid)
            prediction = (probability >= 0.5).astype(int)
            metrics = score_predictions(y_valid, prediction, probability)

            ranking_frame = fold_valid[[DATE_COL, CONTINUOUS_TARGET]].copy()
            ranking_frame["Probability"] = probability
            rank_corr, valid_rank_months = mean_ranking_correlation(
                ranking_frame, "Probability"
            )

            rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "train_start": str(fold_train[DATE_COL].min().date()),
                    "train_end": str(fold_train[DATE_COL].max().date()),
                    "validation_start": str(fold_valid[DATE_COL].min().date()),
                    "validation_end": str(fold_valid[DATE_COL].max().date()),
                    "n_train": len(fold_train),
                    "n_validation": len(fold_valid),
                    **metrics,
                    "ranking_correlation": rank_corr,
                    "valid_ranking_months": valid_rank_months,
                }
            )

    cv = pd.DataFrame(rows)
    summary = (
        cv.groupby("model", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_f1=("f1", "mean"),
            mean_roc_auc=("roc_auc", "mean"),
            std_roc_auc=("roc_auc", "std"),
            mean_ranking_correlation=("ranking_correlation", "mean"),
            std_ranking_correlation=("ranking_correlation", "std"),
            valid_folds=("fold", "count"),
        )
        .sort_values(SELECTION_COLUMNS, ascending=[False] * len(SELECTION_COLUMNS), na_position="last")
        .reset_index(drop=True)
    )
    return cv, summary


def select_model(cv_summary: pd.DataFrame) -> str:
    ordered = cv_summary.sort_values(
        SELECTION_COLUMNS,
        ascending=[False] * len(SELECTION_COLUMNS),
        na_position="last",
    )
    if ordered.empty:
        raise RuntimeError("No Week 4 cross-validation summary is available.")
    return str(ordered.iloc[0]["model"])


def rank_holdout_predictions(
    test: pd.DataFrame,
    model_name: str,
    probability: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    ranked = test[
        [DATE_COL, SECTOR_COL, MACRO_REGIME_COL, TARGET, CONTINUOUS_TARGET]
    ].copy()
    ranked["model"] = model_name
    ranked["Outperformance_Probability"] = probability
    ranked["Predicted_Class"] = prediction
    ranked["Sector_Rank"] = (
        ranked.groupby(DATE_COL)["Outperformance_Probability"]
        .rank(ascending=False, method="first")
        .astype(int)
    )
    group_size = ranked.groupby(DATE_COL)[SECTOR_COL].transform("size")
    ranked["Rank_Score_0_1"] = np.where(
        group_size <= 1,
        1.0,
        1.0 - (ranked["Sector_Rank"] - 1) / (group_size - 1),
    )
    return ranked.sort_values([DATE_COL, "Sector_Rank", SECTOR_COL]).reset_index(drop=True)


def evaluate_holdout(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    ranking_frames: list[pd.DataFrame] = []
    correlation_rows: list[dict[str, object]] = []
    fitted_models: dict[str, object] = {}

    X_train = train[MODEL_INPUT_COLUMNS]
    y_train = train[TARGET]
    X_test = test[MODEL_INPUT_COLUMNS]
    y_test = test[TARGET]

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        probability = positive_probability(model, X_test)
        prediction = (probability >= 0.5).astype(int)
        metrics = score_predictions(y_test, prediction, probability)

        ranked = rank_holdout_predictions(test, name, probability, prediction)
        ranking_frames.append(ranked)

        month_correlations: list[float] = []
        for date, group in ranked.groupby(DATE_COL):
            correlation = safe_spearman(
                group["Outperformance_Probability"], group[CONTINUOUS_TARGET]
            )
            correlation_rows.append(
                {
                    "model": name,
                    "Date": str(pd.Timestamp(date).date()),
                    "ranking_correlation": correlation,
                    "n_sectors": len(group),
                }
            )
            if not np.isnan(correlation):
                month_correlations.append(correlation)

        metric_rows.append(
            {
                "model": name,
                **metrics,
                "mean_ranking_correlation": (
                    float(np.mean(month_correlations)) if month_correlations else np.nan
                ),
                "valid_ranking_months": len(month_correlations),
                "n_holdout": len(test),
            }
        )

        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        confusion_rows.append(
            {"model": name, "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        )
        save_confusion_matrix_figure(name, y_test, prediction)
        joblib.dump(model, MODEL_DIR / f"week4_{name.lower()}.joblib")

    rankings = pd.concat(ranking_frames, ignore_index=True)
    rankings = rankings.sort_values(["model", DATE_COL, "Sector_Rank", SECTOR_COL]).reset_index(drop=True)
    correlations = pd.DataFrame(correlation_rows).sort_values(["model", "Date"]).reset_index(drop=True)
    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(confusion_rows),
        rankings,
        correlations,
        fitted_models,
    )


def build_regime_summary(train: pd.DataFrame) -> pd.DataFrame:
    return (
        train.groupby(MACRO_REGIME_COL)
        .agg(
            observations=(TARGET, "size"),
            months=(DATE_COL, "nunique"),
            outperformance_rate=(TARGET, "mean"),
            first_date=(DATE_COL, "min"),
            last_date=(DATE_COL, "max"),
        )
        .reset_index()
        .sort_values(["observations", MACRO_REGIME_COL], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_regime_model_usage(regime_model: RegimeSwitchingEnsemble) -> pd.DataFrame:
    rows = []
    for regime, sample_count in sorted(regime_model.regime_sample_counts_.items()):
        counts = regime_model.regime_class_counts_[regime]
        rows.append(
            {
                MACRO_REGIME_COL: regime,
                "training_observations": int(sample_count),
                "class_0_count": int(counts[0]),
                "class_1_count": int(counts[1]),
                "local_model_fitted": regime in regime_model.regime_models_,
                "local_model_weight": float(regime_model.regime_weights_.get(regime, 0.0)),
            }
        )
    return pd.DataFrame(rows)


def write_model_report(
    selected_model: str,
    holdout_metrics: pd.DataFrame,
    cv_summary: pd.DataFrame,
    regime_usage: pd.DataFrame,
) -> None:
    selected_holdout = holdout_metrics[holdout_metrics["model"] == selected_model].iloc[0]
    fitted_regimes = regime_usage[regime_usage["local_model_fitted"]]
    regime_lines = []
    for _, row in fitted_regimes.iterrows():
        regime_lines.append(
            f"- {row[MACRO_REGIME_COL]}: {int(row['training_observations'])} observations, "
            f"local weight {row['local_model_weight']:.3f}"
        )

    report = f"""# Week 4 Model Report

## Objective

Week 4 extends the Week 3 monthly sector-outperformance classifier with ensemble
learning, macroeconomic regime switching, sector ranking, and the required
precision, ROC AUC, confusion-matrix, and ranking-correlation evaluation.

## Selected Model

`{selected_model}` was selected using development cross-validation only. The
chronological holdout set was not used for model selection.

Selection priority:

1. mean cross-validation ROC AUC,
2. mean cross-validation ranking correlation,
3. mean cross-validation precision,
4. mean cross-validation balanced accuracy.

## Holdout Results for Selected Model

- Precision: {selected_holdout['precision']:.4f}
- ROC AUC: {selected_holdout['roc_auc']:.4f}
- Balanced accuracy: {selected_holdout['balanced_accuracy']:.4f}
- Recall: {selected_holdout['recall']:.4f}
- F1: {selected_holdout['f1']:.4f}
- Mean monthly ranking correlation: {selected_holdout['mean_ranking_correlation']:.4f}

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

{chr(10).join(regime_lines) if regime_lines else '- None; all regimes used the global fallback.'}

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
{cv_summary.to_string(index=False)}
```
"""
    (DOCS_DIR / "week4_model_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

        data = load_dataset()
        train, test, gap_date = chronological_split(data)

        cv_metrics, cv_summary = cross_validate(train)
        selected_model = select_model(cv_summary)
        (
            holdout_metrics,
            confusion_table,
            rankings,
            ranking_correlations,
            fitted_models,
        ) = evaluate_holdout(train, test)

        regime_summary = build_regime_summary(train)
        regime_model = fitted_models["Regime_Switching_Ensemble"]
        regime_usage = build_regime_model_usage(regime_model)
        selected_rankings = rankings[rankings["model"] == selected_model].copy()
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
        selected_signals = selected_rankings[signal_columns].copy()

        cv_metrics.to_csv(TABLE_DIR / "week4_cv_metrics.csv", index=False)
        cv_summary.to_csv(TABLE_DIR / "week4_cv_summary.csv", index=False)
        holdout_metrics.to_csv(TABLE_DIR / "week4_holdout_metrics.csv", index=False)
        confusion_table.to_csv(TABLE_DIR / "week4_confusion_matrix.csv", index=False)
        rankings.to_csv(TABLE_DIR / "week4_holdout_rankings.csv", index=False)
        selected_rankings.to_csv(TABLE_DIR / "week4_selected_model_rankings.csv", index=False)
        selected_signals.to_csv(TABLE_DIR / "week4_selected_model_signals.csv", index=False)
        ranking_correlations.to_csv(TABLE_DIR / "week4_ranking_correlation.csv", index=False)
        regime_summary.to_csv(TABLE_DIR / "week4_regime_summary.csv", index=False)
        regime_usage.to_csv(TABLE_DIR / "week4_regime_model_usage.csv", index=False)
        save_model_comparison_figure(holdout_metrics)

        joblib.dump(fitted_models[selected_model], MODEL_DIR / "week4_selected_model.joblib")

        split_summary = {
            "train_start": str(train[DATE_COL].min().date()),
            "train_end": str(train[DATE_COL].max().date()),
            "gap_date": str(gap_date.date()),
            "test_start": str(test[DATE_COL].min().date()),
            "test_end": str(test[DATE_COL].max().date()),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_months": int(train[DATE_COL].nunique()),
            "test_months": int(test[DATE_COL].nunique()),
        }
        (TABLE_DIR / "week4_split_summary.json").write_text(
            json.dumps(split_summary, indent=2), encoding="utf-8"
        )

        fitted_regimes = sorted(regime_model.regime_models_.keys())
        model_selection = {
            "selected_model": selected_model,
            "selection_data": "development cross-validation only",
            "selection_rule": [
                "highest mean ROC AUC",
                "then highest mean ranking correlation",
                "then highest mean precision",
                "then highest mean balanced accuracy",
            ],
            "soft_voting_weights": {
                "Logistic_Regression": 1.0,
                "Random_Forest": 1.0,
                "Gradient_Boosting": 1.0,
            },
            "regime_switching": {
                "composite_source": ["Growth_Regime", "Inflation_Regime", "Rate_Regime"],
                "min_regime_samples": int(regime_model.min_regime_samples),
                "min_class_samples": int(regime_model.min_class_samples),
                "max_regime_weight": float(regime_model.max_regime_weight),
                "shrinkage_samples": float(regime_model.shrinkage_samples),
                "fitted_regimes": fitted_regimes,
            },
            "cv_summary": cv_summary.replace({np.nan: None}).to_dict(orient="records"),
        }
        (TABLE_DIR / "week4_model_selection.json").write_text(
            json.dumps(model_selection, indent=2), encoding="utf-8"
        )

        week4_config = {
            "week": 4,
            "input_dataset": "data/modeling/model_dataset.csv",
            "target": TARGET,
            "continuous_ranking_target": CONTINUOUS_TARGET,
            "forecast_horizon_trading_days": 21,
            "benchmark": "Equal-weight mean of XLF, XLK, and XLE",
            "signal_frequency": "Calendar month-end",
            "sectors": list(EXPECTED_SECTORS),
            "models": REQUIRED_MODELS,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "routing_feature": MACRO_REGIME_COL,
            "evaluation": {
                "cv": "5-fold expanding window over unique signal dates",
                "cv_gap_signal_dates": 1,
                "holdout": "last max(12 months, 20% of signal dates)",
                "holdout_purge_gap_signal_dates": 1,
                "required_metrics": [
                    "precision",
                    "roc_auc",
                    "confusion_matrix",
                    "ranking_correlation",
                ],
            },
            "selected_model": selected_model,
            "random_state": RANDOM_STATE,
        }
        config_path = PROJECT_ROOT / "data" / "modeling" / "week4_modeling_config.json"
        config_path.write_text(json.dumps(week4_config, indent=2), encoding="utf-8")

        write_model_report(selected_model, holdout_metrics, cv_summary, regime_usage)

        print(
            f"[PASS] Week 4 evaluated {len(REQUIRED_MODELS)} models on "
            f"{data[DATE_COL].nunique()} monthly signal dates."
        )
        print("[PASS] Added diversified soft-voting ensemble learning.")
        print(
            f"[PASS] Added macro regime switching; fitted {len(fitted_regimes)} "
            "regime-specific model(s) on the final training period."
        )
        print("[PASS] Ranked XLF, XLK, and XLE from outperformance probabilities.")
        print("[PASS] Saved precision, ROC AUC, confusion matrix, and ranking correlation outputs.")
        print(f"[PASS] CV-selected Week 4 model: {selected_model}")
        print("[DONE] Week 4 model development completed.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 4 model development failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
