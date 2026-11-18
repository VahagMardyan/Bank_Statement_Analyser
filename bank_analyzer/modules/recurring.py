"""Recurring payments: detection from transaction history and the storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable, Optional

import pandas as pd

from .models import TransactionType


@dataclass(frozen=True)
class Period:
    key: str     # locale key: recurring.periods.<key>
    days: int    # typical gap, also stored as interval_days
    months: int  # calendar months for the next due date; 0 = count in days

    @property
    def tolerance(self) -> int:
        return max(2, round(self.days * 0.15))

    def matches(self, gap_days: float) -> bool:
        return abs(gap_days - self.days) <= self.tolerance

    def next_after(self, last: date) -> date:
        if self.months:
            return (pd.Timestamp(last) + pd.DateOffset(months=self.months)).date()
        return last + timedelta(days=self.days)


PERIODS = (
    Period("weekly", 7, 0),
    Period("monthly", 30, 1),
    Period("quarterly", 91, 3),
    Period("yearly", 365, 12),
)
PERIOD_BY_DAYS = {p.days: p for p in PERIODS}

PAYMENT_COLUMNS = [
    "id", "name", "category", "expected_amount", "period", "last_paid", "next_due",
]


@dataclass(frozen=True)
class RecurringPayment:
    name: str
    category: str
    expected_amount: float
    period: Period
    last_paid: date
    next_due: date
    occurrences: int


class RecurringDetector:
    """Finds merchants paid at regular intervals with a stable amount."""

    def __init__(
        self,
        min_occurrences: int = 3,
        min_regular_share: float = 0.75,
        max_amount_spread: float = 0.35,
        variable_amount_categories: Iterable[str] = ("Utilities",),
    ) -> None:
        self.min_occurrences = min_occurrences
        self.min_regular_share = min_regular_share
        self.max_amount_spread = max_amount_spread
        self.variable_amount_categories = set(variable_amount_categories)

    def detect(self, df: Optional[pd.DataFrame]) -> list[RecurringPayment]:
        if df is None or df.empty:
            return []
        expenses = df[df["transaction_type"] == TransactionType.EXPENSE.value]
        expenses = expenses[expenses["cleaned_description"].str.len() > 0]
        as_of = df["date"].max().date()
        found = []
        for name, group in expenses.groupby("cleaned_description"):
            payment = self._check(str(name), group.sort_values("date"), as_of)
            if payment is not None:
                found.append(payment)
        return sorted(found, key=lambda p: p.next_due)

    def _check(
        self, name: str, group: pd.DataFrame, as_of: date
    ) -> Optional[RecurringPayment]:
        days = group["date"].dt.normalize().drop_duplicates()
        if len(days) < self.min_occurrences:
            return None

        period = self._match_period(days.diff().dt.days.dropna())
        if period is None:
            return None

        category = str(group["category"].mode().iloc[0])
        amounts = group["amount"]
        if category not in self.variable_amount_categories:
            spread = amounts.std(ddof=0) / amounts.mean()
            if spread > self.max_amount_spread:
                return None

        last_paid = days.iloc[-1].date()
        if (as_of - last_paid).days > 2 * period.days + period.tolerance:
            return None  # stopped paying

        return RecurringPayment(
            name=name[:255],
            category=category,
            expected_amount=round(float(amounts.tail(3).median()), 2),
            period=period,
            last_paid=last_paid,
            next_due=period.next_after(last_paid),
            occurrences=len(days),
        )

    def _match_period(self, gaps: pd.Series) -> Optional[Period]:
        median = gaps.median()
        for period in PERIODS:
            if period.matches(median):
                regular_share = gaps.map(period.matches).mean()
                return period if regular_share >= self.min_regular_share else None
        return None


def payments_frame(payments: Iterable[RecurringPayment]) -> pd.DataFrame:
    """Detected payments in the same columns the repository returns."""
    return pd.DataFrame(
        [
            {
                "id": None,
                "name": p.name,
                "category": p.category,
                "expected_amount": p.expected_amount,
                "period": p.period.key,
                "last_paid": p.last_paid,
                "next_due": p.next_due,
            }
            for p in payments
        ],
        columns=PAYMENT_COLUMNS,
    )


class RecurringRepository(ABC):
    @abstractmethod
    def sync(self, detected: list[RecurringPayment]) -> None:
        """Replace the detected list; keep the user's dismissals."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Active, not dismissed payments in PAYMENT_COLUMNS, soonest first."""

    @abstractmethod
    def clear(self) -> None:
        """Forget all detected payments (used when the history is cleared)."""

    @abstractmethod
    def dismiss(self, payment_id: int) -> None:
        pass

