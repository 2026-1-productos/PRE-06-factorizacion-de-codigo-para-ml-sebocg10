"""Data loading utilities."""

from __future__ import annotations

import pandas as pd

WINE_QUALITY_URL = (
    "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    "winequality-red.csv"
)


def load_wine_quality_data(url: str = WINE_QUALITY_URL) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target for the wine quality dataset."""

    df = pd.read_csv(url, sep=";")
    y = df["quality"]
    x = df.drop(columns=["quality"])
    return x, y