"""Application facade: one entry point for load, classify, override and analytics."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .analytics import AnalyticsService
from .anomaly import AnomalyDetector
from .classifier import TransactionClassifier
from .ingestion import BankStatementLoader
from .models import (
    ClassificationMethod,
    ClassifierProtocol,
    KPIMetrics,
    SelfTrainingClassifier,
    StatementLoaderProtocol,
)

class BankAnalyzerFacade:
    """Facade + Dependency Injection over loader, classifier and analytics.

    UIs (Streamlit, Qt) call only this object; they never reach into the
    subsystems (Law of Demeter).
    """

    def __init__(
        self,
        loader: StatementLoaderProtocol | None = None,
        classifier: ClassifierProtocol | None = None,
        analytics: AnalyticsService | None = None,
    ) -> None:
        self._loader: StatementLoaderProtocol = loader or BankStatementLoader()
        self._classifier: ClassifierProtocol = classifier or TransactionClassifier()
        self._analytics = analytics or AnalyticsService()

    # Read-only access for tests and tools; UI code should use the methods below.
    @property
    def loader(self) -> StatementLoaderProtocol:
        return self._loader

    @property
    def classifier(self) -> ClassifierProtocol:
        return self._classifier

    @property
    def analytics(self) -> AnalyticsService:
        return self._analytics


    # load/classify
    def load_and_classify(
        self, file_path: str | Path
    ) -> pd.DataFrame:
        raw = self._loader.load(file_path)
        classified = self._classifier.classify(raw)
        if not isinstance(classified, pd.DataFrame):
            raise TypeError("Classifier must return a DataFrame for statement files.")

        # Learn from what the rules recognised in THIS statement, then give
        # the still-uncertain rows a second pass with the updated model.
        if isinstance(self._classifier, SelfTrainingClassifier):
            if self._classifier.learn_from_dataframe(classified):
                classified = self._classifier.reclassify_uncertain(classified)
        return classified

    def get_categories(self) -> list[str]:
        return self._classifier.get_categories()

    def override_category(
        self, df: pd.DataFrame, transaction_id: str, new_category: str
    ) -> pd.DataFrame:
        """Relabel one transaction and every row with the same cleaned text."""
        match = df[df["transaction_id"] == transaction_id]
        if match.empty:
            return df

        cleaned = str(match.iloc[0]["cleaned_description"])
        self._classifier.manual_override(transaction_id, new_category, cleaned)

        out = df.copy()
        affected = (out["transaction_id"] == transaction_id) | (
            out["cleaned_description"] == cleaned
        )
        out.loc[affected, "category"] = new_category
        out.loc[affected, "method"] = ClassificationMethod.MANUAL.value
        out.loc[affected, "confidence"] = 1.0
        return out

    # filters
    @staticmethod
    def apply_date_filter(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()

    # analytics
    def compute_kpis(self, df: pd.DataFrame) -> KPIMetrics:
        return self._analytics.compute_kpis(df)

    def category_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._analytics.category_summary(df)

    def monthly_trends(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._analytics.monthly_trends(df)

    def top_expenses(self, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return self._analytics.top_expenses(df, n)

    def check_budget_limits(
        self, df: pd.DataFrame, budgets: dict[str, float]
    ) -> pd.DataFrame:
        return self._analytics.check_budget_limits(df, budgets)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        threshold_std: float = 2.5,
        detector: Optional[AnomalyDetector] = None,
    ) -> pd.DataFrame:
        return self._analytics.detect_anomalies(df, threshold_std, detector)
