"""Application facade: one entry point for load, classify, override and analytics."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

from .recurring import RecurringDetector, RecurringRepository, payments_frame
from .history import TransactionRepository
from .analytics import AnalyticsService
from .anomaly import AnomalyDetector
from .classifier import TransactionClassifier
from .ingestion import BankStatementLoader
from .ml_model import LearnedCategoryModel
from .mysql_repositories import (
    MySqlRecurringRepository,
    MySqlRuleRepository,
    MySqlTrainingExampleStore,
    MySqlTransactionRepository,
)
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
        history: TransactionRepository | None = None,
        recurring: RecurringRepository | None = None,
        recurring_detector: RecurringDetector | None = None,
    ) -> None:
        self._loader: StatementLoaderProtocol = loader or BankStatementLoader()
        self._classifier: ClassifierProtocol = classifier or TransactionClassifier()
        self._analytics = analytics or AnalyticsService()
        self._history = history
        self._recurring = recurring
        self._recurring_detector = recurring_detector or RecurringDetector()

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

    @classmethod
    def for_user(cls, user_id: int) -> "BankAnalyzerFacade":
        """Facade whose rules and training data belong to one user."""
        classifier = TransactionClassifier(
            repository=MySqlRuleRepository(user_id),
            model=LearnedCategoryModel(store=MySqlTrainingExampleStore(user_id))
        )
        return cls(
            classifier=classifier,
            history=MySqlTransactionRepository(user_id),
            recurring=MySqlRecurringRepository(user_id),
        )

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

    def save_transactions(self, df: pd.DataFrame) -> int:
        """Store new rows and re-detect recurring payments."""
        if self._history is None:
            return 0
        added = self._history.save(df)
        self.refresh_recurring()
        return added

    def refresh_recurring(self) -> None:
        if self._history is not None and self._recurring is not None:
            self._recurring.sync(self._recurring_detector.detect(self._history.load()))

    def recurring_payments(self, df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Stored list with a database, otherwise detected from ``df`` on the fly."""
        if self._recurring is not None:
            return self._recurring.load()
        return payments_frame(self._recurring_detector.detect(df))

    def dismiss_recurring(self, payment_id: int) -> None:
        if self._recurring is not None:
            self._recurring.dismiss(payment_id)

    def load_history(self) -> Optional[pd.DataFrame]:
        """Everything stored for this user, or None without a history backend."""
        if self._history is None:
            return None
        df = self._history.load()
        return None if df.empty else df

    def clear_history(self) -> None:
        """Delete stored transactions and what was detected from them.

        The user's category corrections (rules, ML examples) are kept.
        """
        if self._history is not None:
            self._history.clear()
        if self._recurring is not None:
            self._recurring.clear()

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

        if self._history is not None and 'fingerprint' in out.columns:
            self._history.update_category(
                out.loc[affected, 'fingerprint'],
                new_category,
                ClassificationMethod.MANUAL.value
            )
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
