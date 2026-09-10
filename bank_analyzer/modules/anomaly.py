"""Anomaly detection strategies (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class AnomalyDetector(ABC):
    """Strategy: flag unusual expense rows in a classified DataFrame."""

    @abstractmethod
    def detect(self, expenses: pd.DataFrame) -> pd.DataFrame:
        """Return the subset of ``expenses`` that are anomalous."""


class ZScoreAnomalyDetector(AnomalyDetector):
    """Flag amounts whose z-score vs the category mean exceeds a threshold."""

    def __init__(self, threshold_std: float = 2.5) -> None:
        self.threshold_std = threshold_std

    def detect(self, expenses: pd.DataFrame) -> pd.DataFrame:
        if expenses.empty or "category" not in expenses.columns:
            return pd.DataFrame()

        work = expenses.copy()
        stats = work.groupby("category")["amount"].agg(["mean", "std"]).rename(
            columns={"mean": "category_mean", "std": "category_std"}
        )
        work = work.merge(stats, left_on="category", right_index=True, how="left")
        work["category_std"] = work["category_std"].fillna(0)
        work["z_score"] = np.where(
            work["category_std"] > 0,
            (work["amount"] - work["category_mean"]) / work["category_std"],
            0.0,
        )
        work["is_anomaly"] = work["z_score"] >= self.threshold_std
        return (
            work[work["is_anomaly"]]
            .sort_values("z_score", ascending=False)
            .reset_index(drop=True)
        )
