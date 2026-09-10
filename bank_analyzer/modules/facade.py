"""Application facade: one entry point for load, classify, override, and analytics."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .analytics import AnalyticsService
from .classifier import TransactionClassifier
from .ingestion import BankStatementLoader
from .models import ClassifierProtocol, KPIMetrics, StatementLoaderProtocol


class BankAnalyzerFacade:
    """Facade + Dependency Injection over loader, classifier, and analytics.

    Streamlit and Qt both talk to this object instead of wiring the three
    subsystems themselves.
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

    @property
    def loader(self) -> StatementLoaderProtocol:
        return self._loader

    @property
    def classifier(self) -> ClassifierProtocol:
        return self._classifier

    @property
    def analytics(self) -> AnalyticsService:
        return self._analytics

    def list_banks(self) -> list[str]:
        return self._loader.list_banks()

    def bank_labels(self) -> dict[str, str]:
        return self._loader.bank_labels()

    def load_and_classify(
        self,
        file_path: str | Path,
        bank_key: Optional[str] = None,
    ) -> pd.DataFrame:
        raw = self._loader.load(file_path, bank_key=bank_key)
        classified = self._classifier.classify(raw)
        if not isinstance(classified, pd.DataFrame):
            raise TypeError("Classifier must return a DataFrame for statement files.")
        return classified

    def override_category(
        self,
        df: pd.DataFrame,
        transaction_id: str,
        new_category: str,
    ) -> pd.DataFrame:
        match = df[df["transaction_id"] == transaction_id]
        if match.empty:
            return df

        cleaned = str(match.iloc[0]["cleaned_description"])
        self._classifier.manual_override(transaction_id, new_category, cleaned)
        out = df.copy()
        mask = out["transaction_id"] == transaction_id
        out.loc[mask, "category"] = new_category
        out.loc[mask, "method"] = "manual"
        out.loc[mask, "confidence"] = 1.0
        return out

    def get_categories(self) -> list[str]:
        return self._classifier.get_categories()

    def apply_date_filter(
        self, df: pd.DataFrame, start: date, end: date
    ) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        return df.loc[mask].copy()

    def compute_kpis(self, df: pd.DataFrame) -> KPIMetrics:
        return self._analytics.compute_kpis(df)
