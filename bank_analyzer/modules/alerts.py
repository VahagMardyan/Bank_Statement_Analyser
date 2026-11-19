"""Suspicious activity: transactions worth a second look.

Each check is a separate ``AlertRule`` (Strategy). ``SuspiciousActivityDetector``
runs all of them, so a new check is one new class added to the list.
These are hints for the user, not proof of fraud: a duplicate charge can be a
real second purchase, and a night payment can be a real night out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum
from typing import Iterable, Optional

import pandas as pd

from .constants import INCOME_CATEGORIES
from .models import TransactionType

ALERT_COLUMNS = ["date", "kind", "description", "category", "amount", "detail"]


class AlertKind(str, Enum):
    DUPLICATE_CHARGE = "duplicate_charge"
    NEW_MERCHANT_LARGE = "new_merchant_large"
    PRICE_INCREASE = "price_increase"
    NIGHT_PAYMENT = "night_payment"


@dataclass(frozen=True)
class Alert:
    kind: AlertKind
    date: pd.Timestamp
    description: str
    category: str
    amount: float
    params: dict = field(default_factory=dict)  # placeholders for the locale message


class AlertRule(ABC):
    @abstractmethod
    def check(self, expenses: pd.DataFrame) -> list[Alert]:
        """``expenses``: expense rows of the whole history, sorted by date."""

    @staticmethod
    def _alert(kind: AlertKind, row, **params) -> Alert:
        return Alert(kind, row.date, str(row.description), str(row.category),
                     round(float(row.amount), 2), params)


def has_time(ts: pd.Timestamp) -> bool:
    """Many statements give only the date (00:00:00)."""
    return not (ts.hour == 0 and ts.minute == 0 and ts.second == 0)


class DuplicateChargeRule(AlertRule):
    """Same merchant, same amount, charged again within a short time."""

    def __init__(self, window_minutes: int = 10, min_amount: float = 5000.0) -> None:
        self.window = timedelta(minutes=window_minutes)
        self.min_amount = min_amount

    def check(self, expenses: pd.DataFrame) -> list[Alert]:
        alerts = []
        big = expenses[expenses["amount"] >= self.min_amount]
        for _, group in big.groupby(["cleaned_description", "amount"]):
            rows = list(group.sort_values("date").itertuples(index=False))
            for prev, row in zip(rows, rows[1:]):
                if has_time(prev.date) and has_time(row.date):
                    close = row.date - prev.date <= self.window
                    gap = int((row.date - prev.date).total_seconds() // 60)
                else:  # date only: same day is the best we can say
                    close = row.date.normalize() == prev.date.normalize()
                    gap = None
                if close:
                    alerts.append(self._alert(AlertKind.DUPLICATE_CHARGE, row, minutes=gap))
        return alerts


class NewMerchantLargeRule(AlertRule):
    """A big payment to a merchant never paid before (and not a known regular payment)."""

    def __init__(self, known_names: Iterable[str] = (), min_amount: float = 30000.0,
                 percentile: float = 0.95, min_history: int = 30) -> None:
        self.known = set(known_names)
        self.min_amount = min_amount
        self.percentile = percentile
        self.min_history = min_history  # earlier payments needed before "new" means anything

    def check(self, expenses: pd.DataFrame) -> list[Alert]:
        if len(expenses) < self.min_history:
            return []
        threshold = max(self.min_amount, float(expenses["amount"].quantile(self.percentile)))
        # A merchant only counts as "new" once there is some history to compare with.
        start = expenses["date"].iloc[self.min_history - 1]
        first = expenses.drop_duplicates("cleaned_description", keep="first")
        first = first[(first["date"] > start) & ~first["cleaned_description"].isin(self.known)]
        return [
            self._alert(AlertKind.NEW_MERCHANT_LARGE, row, threshold=round(threshold))
            for row in first[first["amount"] >= threshold].itertuples(index=False)
        ]


class PriceIncreaseRule(AlertRule):
    """A regular payment suddenly costs more than it used to."""

    def __init__(self, recurring_names: Iterable[str], min_increase: float = 0.15,
                 variable_categories: Iterable[str] = ("Utilities",)) -> None:
        self.names = set(recurring_names)
        self.min_increase = min_increase
        self.variable_categories = set(variable_categories)

    def check(self, expenses: pd.DataFrame) -> list[Alert]:
        alerts = []
        rows = expenses[expenses["cleaned_description"].isin(self.names)]
        for _, group in rows.groupby("cleaned_description"):
            group = group.sort_values("date")
            if len(group) < 3 or group["category"].iloc[-1] in self.variable_categories:
                continue
            before = float(group["amount"].iloc[:-1].median())
            last = group.iloc[-1]
            if before > 0 and last["amount"] >= before * (1 + self.min_increase):
                row = next(group.tail(1).itertuples(index=False))
                alerts.append(self._alert(
                    AlertKind.PRICE_INCREASE, row,
                    before=round(before), pct=round((last["amount"] / before - 1) * 100),
                ))
        return alerts


class NightPaymentRule(AlertRule):
    """Larger card payments in the middle of the night (needs a time in the statement)."""

    def __init__(self, start_hour: int = 1, end_hour: int = 5, min_amount: float = 10000.0) -> None:
        self.start_hour, self.end_hour, self.min_amount = start_hour, end_hour, min_amount

    def check(self, expenses: pd.DataFrame) -> list[Alert]:
        return [
            self._alert(AlertKind.NIGHT_PAYMENT, row, time=row.date.strftime("%H:%M"))
            for row in expenses.itertuples(index=False)
            if row.amount >= self.min_amount and has_time(row.date)
            and self.start_hour <= row.date.hour < self.end_hour
        ]


class SuspiciousActivityDetector:
    def __init__(self, rules: Optional[list[AlertRule]] = None, recent_days: int = 90) -> None:
        """recent_days: only report alerts this close to the latest transaction."""
        self.rules = rules
        self.recent_days = recent_days

    def detect(self, df: Optional[pd.DataFrame], recurring_names: Iterable[str] = ()) -> list[Alert]:
        if df is None or df.empty:
            return []
        expenses = df[
            (df["transaction_type"] == TransactionType.EXPENSE.value)
            & ~df["category"].isin(INCOME_CATEGORIES)
        ].sort_values("date")
        names = set(recurring_names)
        rules = self.rules if self.rules is not None else [
            DuplicateChargeRule(),
            NewMerchantLargeRule(names),
            PriceIncreaseRule(names),
            NightPaymentRule(),
        ]
        since = df["date"].max() - pd.Timedelta(days=self.recent_days)
        alerts = [a for rule in rules for a in rule.check(expenses) if a.date >= since]
        return sorted(alerts, key=lambda a: a.date, reverse=True)


def alerts_frame(alerts: Iterable[Alert], describe) -> pd.DataFrame:
    """Table for the UI; ``describe(alert)`` turns an alert into a sentence."""
    return pd.DataFrame(
        [
            {"date": a.date, "kind": a.kind.value, "description": a.description,
             "category": a.category, "amount": a.amount, "detail": describe(a)}
            for a in alerts
        ],
        columns=ALERT_COLUMNS,
    )
