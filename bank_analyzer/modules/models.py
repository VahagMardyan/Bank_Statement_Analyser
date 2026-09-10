"""Domain models and typed result objects for the analyzer pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, Union

import pandas as pd


class TransactionType(str, Enum):
    """Income vs expense in the unified transaction schema."""

    INCOME = "income"
    EXPENSE = "expense"


class ClassificationMethod(str, Enum):
    """How a category was assigned (stored as DataFrame ``method`` values)."""

    MERCHANT_RULE = "Merchant rule"
    KEYWORD_RULE = "Keyword rule"
    VECTOR = "Vector"
    MANUAL = "manual"
    NONE = "None"


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of classifying a single description."""

    category: str
    confidence: float
    method: str

    def as_tuple(self) -> tuple[str, float, str]:
        return self.category, self.confidence, self.method


@dataclass(frozen=True)
class KPIMetrics:
    """Top-level dashboard metrics."""

    total_income: float
    total_expenses: float
    net_savings: float
    transaction_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_income": self.total_income,
            "total_expenses": self.total_expenses,
            "net_savings": self.net_savings,
            "transaction_count": self.transaction_count,
        }


class ClassifierProtocol(Protocol):
    """Interface the UI/facade depends on (Dependency Inversion)."""

    def get_categories(self) -> list[str]: ...

    def classify(
        self, target: Union[str, pd.DataFrame]
    ) -> Union[str, pd.DataFrame]: ...

    def manual_override(
        self,
        transaction_id: str,
        new_category: str,
        cleaned_description: str,
        persist: bool = True,
    ) -> None: ...


class StatementLoaderProtocol(Protocol):
    """Interface for loading statements into the unified schema."""

    def list_banks(self) -> list[str]: ...

    def bank_display_name(self, bank_key: str) -> str: ...

    def bank_labels(self) -> dict[str, str]: ...

    def load(
        self,
        file_path: str | Path,
        bank_key: str | None = None,
        encoding: str | None = None,
    ) -> pd.DataFrame: ...
