"""Export strategies for classified transaction tables."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod

import pandas as pd


class ExportStrategy(ABC):
    """Strategy: serialize a DataFrame to downloadable bytes."""

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Filename suffix including the leading dot."""

    @property
    @abstractmethod
    def mime_type(self) -> str:
        """HTTP/content type for downloads."""

    @abstractmethod
    def export(self, df: pd.DataFrame, columns: list[str] | None = None) -> bytes:
        """Return file bytes for ``df`` (optionally restricted to ``columns``)."""


class CsvExportStrategy(ExportStrategy):
    """UTF-8-SIG CSV so Excel on Windows opens Armenian text correctly."""

    @property
    def file_extension(self) -> str:
        return ".csv"

    @property
    def mime_type(self) -> str:
        return "text/csv"

    def export(self, df: pd.DataFrame, columns: list[str] | None = None) -> bytes:
        target = df[columns] if columns is not None else df
        return target.to_csv(index=False).encode("utf-8-sig")


class ExcelExportStrategy(ExportStrategy):
    """xlsx export via openpyxl."""

    @property
    def file_extension(self) -> str:
        return ".xlsx"

    @property
    def mime_type(self) -> str:
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def export(self, df: pd.DataFrame, columns: list[str] | None = None) -> bytes:
        target = df[columns] if columns is not None else df
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            target.to_excel(writer, index=False, sheet_name="Transactions")
        return output.getvalue()


class ExportStrategyFactory:
    """Simple factory mapping a UI label to an export strategy."""

    _REGISTRY: dict[str, type[ExportStrategy]] = {
        "CSV": CsvExportStrategy,
        "Excel (.xlsx)": ExcelExportStrategy,
    }

    @classmethod
    def labels(cls) -> list[str]:
        return list(cls._REGISTRY.keys())

    @classmethod
    def create(cls, label: str) -> ExportStrategy:
        strategy_cls = cls._REGISTRY.get(label)
        if strategy_cls is None:
            raise ValueError(f"Unknown export format: {label}")
        return strategy_cls()
