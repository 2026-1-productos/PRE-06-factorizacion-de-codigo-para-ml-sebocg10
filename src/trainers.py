"""Model training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

from .data import load_wine_quality_data
from .metrics import RegressionMetrics, evaluate_regression


@dataclass(frozen=True)
class TrainResult:
    estimator: object
    train_metrics: RegressionMetrics
    test_metrics: RegressionMetrics


def split_wine_data(
    test_size: float = 0.25,
    random_state: int = 123456,
):
    """Split the wine quality data into train and test sets."""

    x, y = load_wine_quality_data()
    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_elasticnet(
    alpha: float,
    l1_ratio: float,
    random_state: int = 12345,
) -> TrainResult:
    """Train a single ElasticNet model with given hyperparameters."""

    x_train, x_test, y_train, y_test = split_wine_data()
    estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    estimator.fit(x_train, y_train)

    train_metrics = evaluate_regression(y_train, estimator.predict(x_train))
    test_metrics = evaluate_regression(y_test, estimator.predict(x_test))

    return TrainResult(estimator, train_metrics, test_metrics)


def select_best_elasticnet(
    candidates: Iterable[tuple[float, float]],
) -> TrainResult:
    """Train multiple ElasticNet models and return the best by test MSE."""

    best_result: TrainResult | None = None
    best_mse = float("inf")

    for alpha, l1_ratio in candidates:
        result = train_elasticnet(alpha=alpha, l1_ratio=l1_ratio)
        if result.test_metrics.mse < best_mse:
            best_mse = result.test_metrics.mse
            best_result = result

    if best_result is None:
        raise ValueError("No ElasticNet candidates provided")

    return best_result


def train_knn(n_neighbors: int) -> TrainResult:
    """Train a KNN regressor with a given neighbor count."""

    x_train, x_test, y_train, y_test = split_wine_data()
    estimator = KNeighborsRegressor(n_neighbors=n_neighbors)
    estimator.fit(x_train, y_train)

    train_metrics = evaluate_regression(y_train, estimator.predict(x_train))
    test_metrics = evaluate_regression(y_test, estimator.predict(x_test))

    return TrainResult(estimator, train_metrics, test_metrics)


def select_best_knn(neighbors: Iterable[int]) -> TrainResult:
    """Train multiple KNN models and return the best by test MSE."""

    best_result: TrainResult | None = None
    best_mse = float("inf")

    for n_neighbors in neighbors:
        result = train_knn(n_neighbors=n_neighbors)
        if result.test_metrics.mse < best_mse:
            best_mse = result.test_metrics.mse
            best_result = result

    if best_result is None:
        raise ValueError("No KNN candidates provided")

    return best_result