"""Export strategies for classified transaction tables.

New formats register themselves with ``@ExportStrategyFactory.register(...)``,
so adding one never requires editing the factory (Open/Closed principle).
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import pandas as pd


class ExportStrategy(ABC):
    """Strategy: serialize a DataFrame to downloadable bytes."""

    file_extension: str = ""
    mime_type: str = "application/octet-stream"

    def export(self, df: pd.DataFrame, columns: list[str] | None = None) -> bytes:
        """Template method: select columns, then serialize."""
        return self._serialize(df[columns] if columns is not None else df)

    @abstractmethod
    def _serialize(self, df: pd.DataFrame) -> bytes:
        """Return file bytes for ``df``."""


S = TypeVar("S", bound=type[ExportStrategy])


class ExportStrategyFactory:
    """Registry-based factory mapping a UI label to an export strategy."""

    _registry: dict[str, type[ExportStrategy]] = {}

    @classmethod
    def register(cls, label: str) -> Callable[[S], S]:
        """Class decorator that adds a strategy under ``label``."""

        def decorator(strategy_cls: S) -> S:
            if label in cls._registry:
                raise ValueError(f"Export format already registered: {label}")
            cls._registry[label] = strategy_cls
            return strategy_cls

        return decorator

    @classmethod
    def labels(cls) -> list[str]:
        return list(cls._registry)

    @classmethod
    def create(cls, label: str) -> ExportStrategy:
        strategy_cls = cls._registry.get(label)
        if strategy_cls is None:
            raise ValueError(f"Unknown export format: {label}")
        return strategy_cls()


@ExportStrategyFactory.register("CSV")
class CsvExportStrategy(ExportStrategy):
    """UTF-8-SIG CSV so Excel on Windows opens Armenian text correctly."""

    file_extension = ".csv"
    mime_type = "text/csv"

    def _serialize(self, df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8-sig")


@ExportStrategyFactory.register("Excel (.xlsx)")
class ExcelExportStrategy(ExportStrategy):
    """xlsx export via openpyxl."""

    file_extension = ".xlsx"
    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def _serialize(self, df: pd.DataFrame) -> bytes:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Transactions")
        return output.getvalue()
