"""Train the best ElasticNet model and save it to disk."""

from __future__ import annotations

from pathlib import Path

import joblib

from .trainers import select_best_elasticnet


CANDIDATES = [
    (0.5, 0.5),
    (0.2, 0.2),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.3, 0.2),
]


def build_and_save_model(output_path: str | Path) -> None:
    """Train the best ElasticNet and persist it to output_path."""

    result = select_best_elasticnet(CANDIDATES)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.estimator, output_path)


if __name__ == "__main__":
    build_and_save_model(Path("models") / "estimator.pkl")