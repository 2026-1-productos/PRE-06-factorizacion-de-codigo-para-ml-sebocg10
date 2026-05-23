"""Regression evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    mae: float
    r2: float


def evaluate_regression(y_true, y_pred) -> RegressionMetrics:
    """Compute common regression metrics."""

    return RegressionMetrics(
        mse=mean_squared_error(y_true, y_pred),
        mae=mean_absolute_error(y_true, y_pred),
        r2=r2_score(y_true, y_pred),
    )