"""Domain models, typed result objects and the interfaces between layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Union, runtime_checkable

import pandas as pd

if TYPE_CHECKING:  # avoid a runtime import cycle with text_utils
    from .text_utils import TextNormalizer


OTHER_CATEGORY = "Other"


class TransactionType(str, Enum):
    """Income vs expense in the unified transaction schema."""

    INCOME = "income"
    EXPENSE = "expense"


class ClassificationMethod(str, Enum):
    """How a category was assigned (stored as DataFrame ``method`` values)."""

    MERCHANT_RULE = "Merchant rule"
    KEYWORD_RULE = "Keyword rule"
    NAME_HEURISTIC = "Name heuristic"
    ML_MODEL = "ML model"
    VECTOR = "Vector"
    MANUAL = "manual"
    NONE = "None"


# Rows labelled by these methods are trusted enough to train on.
TRUSTED_METHODS: frozenset[str] = frozenset(
    {ClassificationMethod.MERCHANT_RULE.value, ClassificationMethod.KEYWORD_RULE.value}
)


@dataclass(frozen=True)
class ClassificationResult:
    """Outcome of classifying a single description."""

    category: str
    confidence: float
    method: str

    @classmethod
    def unclassified(cls, confidence: float = 0.0) -> "ClassificationResult":
        """The 'Other' result every fallback path returns."""
        return cls(OTHER_CATEGORY, confidence, ClassificationMethod.NONE.value)

    def as_tuple(self) -> tuple[str, float, str]:
        return self.category, self.confidence, self.method


@dataclass(frozen=True)
class ClassificationRequest:
    """Parameter Object passed along the classification chain.

    Holds the raw description once and derives every view a handler may need,
    so each handler no longer cleans or splits the text itself.

    The loader appends merchant/payee/place after ``" | "``. The bank's own
    description (``primary``) says WHAT the payment is; the payee only says WHO.
    """

    raw: str
    normalizer: "TextNormalizer" = field(repr=False, compare=False)
    exclude: frozenset[str] = frozenset()

    PAYEE_SEPARATOR = " | "

    @cached_property
    def cleaned(self) -> str:
        return self.normalizer.clean(self.raw)

    @cached_property
    def primary(self) -> str:
        """Cleaned description WITHOUT the appended payee ('' if there is none)."""
        if self.PAYEE_SEPARATOR not in self.raw:
            return ""
        return self.normalizer.clean(self.raw.split(self.PAYEE_SEPARATOR, 1)[0])

    def allows(self, category: str) -> bool:
        return category not in self.exclude


@dataclass(frozen=True)
class KPIMetrics:
    """Top-level dashboard metrics."""

    total_income: float
    total_expenses: float
    net_savings: float
    transaction_count: int

    @classmethod
    def empty(cls) -> "KPIMetrics":
        return cls(0.0, 0.0, 0.0, 0)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_income": self.total_income,
            "total_expenses": self.total_expenses,
            "net_savings": self.net_savings,
            "transaction_count": self.transaction_count,
        }


# --------------------------------------------------------------------------
# Interfaces (Dependency Inversion + Interface Segregation)
# --------------------------------------------------------------------------
class ClassifierProtocol(Protocol):
    """Minimal interface the UI/facade needs from any classifier."""

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


@runtime_checkable
class SelfTrainingClassifier(Protocol):
    """Optional capability: classifiers that learn from what they labelled.

    Kept separate from :class:`ClassifierProtocol` so a pure rule-based
    classifier does not have to implement learning (Interface Segregation).
    """

    def learn_from_dataframe(
        self, classified: pd.DataFrame, persist: bool = True
    ) -> int: ...

    def reclassify_uncertain(self, classified: pd.DataFrame) -> pd.DataFrame: ...


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
