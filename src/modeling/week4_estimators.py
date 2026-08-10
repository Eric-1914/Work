"""Custom estimators used by the Week 4 sector-rotation models."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def positive_probability(estimator: object, X: pd.DataFrame) -> np.ndarray:
    """Return P(y=1) from an estimator that implements predict_proba()."""
    probability = np.asarray(estimator.predict_proba(X), dtype=float)
    if probability.ndim != 2 or probability.shape[0] != len(X):
        raise ValueError("predict_proba() returned an invalid probability array.")

    classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
    positive_positions = np.where(classes == 1)[0]
    if len(positive_positions) == 0:
        result = np.zeros(len(X), dtype=float)
    else:
        result = probability[:, int(positive_positions[0])].astype(float)

    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite outperformance probabilities.")
    return np.clip(result, 0.0, 1.0)


class RegimeSwitchingEnsemble(BaseEstimator, ClassifierMixin):
    """Shrink regime-specific ensemble probabilities toward a global ensemble.

    A global model is always trained. A local model is trained only for macro
    regimes with enough observations and enough examples of both target classes.
    The local-model weight grows with the regime sample size, which makes sparse
    regimes rely more heavily on the global model instead of overfitting.
    """

    def __init__(
        self,
        base_estimator: object,
        regime_col: str = "Macro_Regime",
        min_regime_samples: int = 36,
        min_class_samples: int = 6,
        max_regime_weight: float = 0.70,
        shrinkage_samples: float = 36.0,
    ) -> None:
        self.base_estimator = base_estimator
        self.regime_col = regime_col
        self.min_regime_samples = min_regime_samples
        self.min_class_samples = min_class_samples
        self.max_regime_weight = max_regime_weight
        self.shrinkage_samples = shrinkage_samples

    def _validate_parameters(self) -> None:
        if int(self.min_regime_samples) < 2:
            raise ValueError("min_regime_samples must be at least 2.")
        if int(self.min_class_samples) < 1:
            raise ValueError("min_class_samples must be at least 1.")
        if not 0.0 <= float(self.max_regime_weight) <= 1.0:
            raise ValueError("max_regime_weight must be inside [0, 1].")
        if float(self.shrinkage_samples) <= 0:
            raise ValueError("shrinkage_samples must be positive.")

    def fit(self, X: pd.DataFrame, y) -> "RegimeSwitchingEnsemble":
        self._validate_parameters()
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RegimeSwitchingEnsemble requires a pandas DataFrame.")
        if self.regime_col not in X.columns:
            raise ValueError(f"Missing macro regime column: {self.regime_col}")

        y_array = np.asarray(y, dtype=int)
        if len(y_array) != len(X):
            raise ValueError("X and y have different lengths.")
        if not set(np.unique(y_array)).issubset({0, 1}):
            raise ValueError("Target must contain only binary classes 0 and 1.")
        if len(np.unique(y_array)) < 2:
            raise ValueError("Training data must contain target classes 0 and 1.")

        self.global_model_ = clone(self.base_estimator)
        self.global_model_.fit(X, y_array)

        self.regime_models_: Dict[str, object] = {}
        self.regime_sample_counts_: Dict[str, int] = {}
        self.regime_class_counts_: Dict[str, Dict[int, int]] = {}
        self.regime_weights_: Dict[str, float] = {}

        regimes = X[self.regime_col].astype("string").fillna("Unknown").astype(str)
        for regime in sorted(regimes.unique()):
            mask = regimes.eq(regime).to_numpy()
            y_local = y_array[mask]
            sample_count = int(mask.sum())
            class_counts = {
                0: int(np.sum(y_local == 0)),
                1: int(np.sum(y_local == 1)),
            }
            self.regime_sample_counts_[regime] = sample_count
            self.regime_class_counts_[regime] = class_counts

            if sample_count < int(self.min_regime_samples):
                continue
            if min(class_counts.values()) < int(self.min_class_samples):
                continue

            local_model = clone(self.base_estimator)
            local_model.fit(X.loc[mask], y_local)
            self.regime_models_[regime] = local_model

            # Empirical-Bayes-style shrinkage: sparse regimes stay close to the
            # global model, while larger regimes receive progressively more weight.
            local_weight = float(self.max_regime_weight) * (
                sample_count / (sample_count + float(self.shrinkage_samples))
            )
            self.regime_weights_[regime] = float(
                np.clip(local_weight, 0.0, float(self.max_regime_weight))
            )

        self.classes_ = np.array([0, 1], dtype=int)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_model_"):
            raise ValueError("RegimeSwitchingEnsemble is not fitted.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("RegimeSwitchingEnsemble requires a pandas DataFrame.")
        if self.regime_col not in X.columns:
            raise ValueError(f"Missing macro regime column: {self.regime_col}")

        global_probability = positive_probability(self.global_model_, X)
        final_probability = global_probability.copy()
        regimes = X[self.regime_col].astype("string").fillna("Unknown").astype(str)

        for regime, local_model in self.regime_models_.items():
            mask = regimes.eq(regime).to_numpy()
            if not mask.any():
                continue
            local_probability = positive_probability(local_model, X.loc[mask])
            local_weight = float(self.regime_weights_[regime])
            final_probability[mask] = (
                local_weight * local_probability
                + (1.0 - local_weight) * global_probability[mask]
            )

        final_probability = np.clip(final_probability, 0.0, 1.0)
        return np.column_stack([1.0 - final_probability, final_probability])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
