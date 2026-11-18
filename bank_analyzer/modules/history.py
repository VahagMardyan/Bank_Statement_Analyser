"""Stored transaction history: the repository interface and row fingerprints."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

def make_fingerprints(df: pd.DataFrame) -> pd.Series:
    """SHA-256 per row from date, amount, type and description.

    Two identical rows in one statement (same coffee twice in the same minute)
    get different fingerprints through their occurrence number, so both are kept.
    Uploading the same statement again gives the same numbers, so neither is added twice.
    """

    key = (
        df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
        + "|" + df['amount'].map("{:.2f}".format)
        + "|" + df['transaction_type'].astype(str)
        + "|" + df['description'].astype(str).str.strip()
    )

    occurrence = key.groupby(key).cumcount().astype(str)
    return (key + '|' + occurrence).map(
        lambda s: hashlib.sha256(s.encode('utf-8')).hexdigest()
    )

class TransactionRepository(ABC):
    @abstractmethod
    def save(self, df: pd.DataFrame) -> int:
        """Store rows not stored yet; return how many were new."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """All stored rows, in the same columns the classifier produces."""

    @abstractmethod
    def clear(self) -> None:
        """Delete every stored transaction of this user."""

    @abstractmethod
    def update_category(
        self, fingerprints: Iterable[str], category: str, method: str
    ) -> None:
        pass

