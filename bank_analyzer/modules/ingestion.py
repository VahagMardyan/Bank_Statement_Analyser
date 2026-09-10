"""CSV, Excel, and PDF ingestion and standardization for bank statements."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .readers import (
    DATE_PATTERN,
    WHITESPACE,
    StatementReaderFactory,
    is_excel_file,
    is_pdf_file,
    normalize_amount_string,
    normalize_column_name,
)

NOISE_TOKENS: frozenset[str] = frozenset(
    {
        "am",
        "arm",
        "amd",
        "yer",
        "yerevan",
        "pos",
        "card",
        "visa",
        "mastercard",
        "purchase",
        "payment",
        "transaction",
        "ref",
        "txn",
        "atm",
        "fee",
        "com",
        "commission",
        "bank",
        "transfer",
        "online",
        "mobile",
        "terminal",
    }
)

LONG_NUMERIC_ID = re.compile(r"\b\d{6,}\b")
SIGNED_AMOUNT_PATTERN = re.compile(r"^\s*[+\-]\s*[\d\s]+(?:[.,]\d+)?")

BALANCE_ROW_KEYWORDS = (
    "մնացորդ",
    "balance",
    "opening balance",
    "closing balance",
    "օրվա վերջին",
)


class BankStatementLoader:
    """Load and normalize bank statement files into a unified schema.

    File-format parsing is delegated to :class:`StatementReaderFactory`
    (Strategy + Factory). This class owns bank detection, column mapping,
    and schema standardization (Template Method in :meth:`load`).
    """

    STANDARD_COLUMNS = (
        "date",
        "description",
        "cleaned_description",
        "amount",
        "transaction_type",
    )

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        reader_factory: StatementReaderFactory | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.config_path = (
            Path(config_path)
            if config_path
            else base_dir / "config" / "bank_configs.json"
        )
        self._configs = self._load_configs()
        self._reader_factory = reader_factory or StatementReaderFactory()

    def _load_configs(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"auto_detect": {}}
        with open(self.config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("banks", {})

    def list_banks(self) -> list[str]:
        """Return configured bank keys excluding auto-detect."""
        return [k for k in self._configs if k != "auto_detect"]

    def bank_display_name(self, bank_key: str) -> str:
        """Public accessor so UIs do not reach into ``_configs``."""
        cfg = self._configs.get(bank_key, {})
        if isinstance(cfg, dict):
            return str(cfg.get("display_name", bank_key))
        return bank_key

    def bank_labels(self) -> dict[str, str]:
        return {
            "auto_detect": "Auto Detect",
            **{key: self.bank_display_name(key) for key in self.list_banks()},
        }

    @staticmethod
    def clean_description(text: str) -> str:
        """Normalize transaction text without removing merchant information."""
        if not isinstance(text, str) or not text.strip():
            return ""
        cleaned = unicodedata.normalize("NFKC", text).lower()
        cleaned = DATE_PATTERN.sub(" ", cleaned)
        cleaned = LONG_NUMERIC_ID.sub(" ", cleaned)
        cleaned = re.sub(r"[^\w\s%]", " ", cleaned, flags=re.UNICODE)
        return WHITESPACE.sub(" ", cleaned).strip()

    def _detect_bank_from_content(self, raw_df: pd.DataFrame) -> str:
        text = " ".join(
            str(value).lower() for value in raw_df.values.flatten() if pd.notna(value)
        )

        if "ameriabank" in text or "ameria" in text or "ամերիա" in text:
            return "ameriabank"
        if "evocabank" in text or "evoca" in text:
            return "evocabank"
        if "acba" in text or "ակբա" in text:
            return "acba"
        if "ineco" in text or "ինեկո" in text:
            return "inecobank"
        return "auto_detect"

    def detect_bank(self, file_path: str | Path) -> str:
        path = Path(file_path)
        try:
            if is_pdf_file(path):
                rows = self._reader_factory.pdf_reader.extract_preview_rows(path)
                if rows:
                    raw_df = pd.DataFrame(rows)
                    return self._detect_bank_from_content(raw_df)
                return "auto_detect"
            if is_excel_file(path):
                raw_df = self._reader_factory.excel_reader.preview(path, nrows=25)
                return self._detect_bank_from_content(raw_df)

            for encoding in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
                try:
                    preview = pd.read_csv(
                        path, nrows=10, encoding=encoding, on_bad_lines="skip"
                    )
                    text = (
                        " ".join(
                            str(val).lower()
                            for val in preview.values.flatten()
                            if pd.notna(val)
                        )
                        + " "
                        + " ".join(str(col).lower() for col in preview.columns)
                    )
                    if "ameriabank" in text or "ameria" in text:
                        return "ameriabank"
                    if "evocabank" in text or "evoca" in text:
                        return "evocabank"
                    if "acba" in text:
                        return "acba"
                    if "ineco" in text:
                        return "inecobank"
                except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
                    continue
        except (OSError, ValueError):
            pass
        return "auto_detect"

    def _resolve_column(
        self, df: pd.DataFrame, candidates: list[str]
    ) -> Optional[str]:
        normalized = {
            normalize_column_name(column).lower(): column for column in df.columns
        }

        for candidate in candidates:
            key = candidate.lower()
            if key in normalized:
                return normalized[key]

        for col_lower, original in normalized.items():
            for candidate in candidates:
                if candidate.lower() in col_lower:
                    return original
        return None

    def _resolve_description_column(
        self,
        df: pd.DataFrame,
        mapping: dict[str, list[str]],
        date_col: str,
        amount_col: str,
    ) -> Optional[str]:
        excluded = {
            normalize_column_name(date_col).lower(),
            normalize_column_name(amount_col).lower(),
        }

        debit_col = self._resolve_column(
            df, mapping.get("debit", ["debit", "դեբետ", "ելք", "out"])
        )
        credit_col = self._resolve_column(
            df, mapping.get("credit", ["credit", "կրեդիտ", "մուտք", "in"])
        )

        if debit_col:
            excluded.add(normalize_column_name(debit_col).lower())
        if credit_col:
            excluded.add(normalize_column_name(credit_col).lower())

        candidates = mapping.get(
            "description",
            [
                "նկարագրություն",
                "description",
                "details",
                "purpose",
                "narrative",
                "merchant",
                "comment",
            ],
        )

        for candidate in candidates:
            for col in df.columns:
                col_norm = normalize_column_name(col).lower()
                if col_norm not in excluded and candidate.lower() in col_norm:
                    return col

        for col in df.columns:
            col_norm = normalize_column_name(col).lower()
            if col_norm in excluded:
                continue
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue

            text_ratio = sample.str.contains(r"[A-Za-zԱ-ֆ]{2,}", regex=True, na=False)
            if text_ratio.mean() >= 0.3:
                return col

        return None

    def _parse_amount_and_type(
        self, df: pd.DataFrame, mapping: dict[str, list[str]], bank_cfg: dict[str, Any]
    ) -> pd.DataFrame:
        del bank_cfg
        debit_col = self._resolve_column(
            df,
            mapping.get(
                "debit", ["debit", "դեբետ", "ելք", "ելքեր", "out", "withdrawal"]
            ),
        )
        credit_col = self._resolve_column(
            df,
            mapping.get(
                "credit", ["credit", "կրեդիտ", "մուտք", "մուտքեր", "in", "deposit"]
            ),
        )

        if debit_col and credit_col:
            debit = df[debit_col].map(normalize_amount_string).fillna(0).abs()
            credit = df[credit_col].map(normalize_amount_string).fillna(0).abs()

            final_amount = np.where(credit > 0, credit, debit)
            transaction_type = np.where(credit > 0, "income", "expense")

            return pd.DataFrame(
                {"amount": final_amount, "transaction_type": transaction_type}
            )

        amount_col = self._resolve_column(
            df,
            mapping.get(
                "amount", ["գումար", "amount", "sum", "գործարքի գումար", "մուտք/ելք"]
            ),
        )
        if amount_col is None:
            for col in df.columns:
                vals = df[col].map(normalize_amount_string).dropna()
                if len(vals) > 0:
                    amount_col = col
                    break

        if amount_col is None:
            raise ValueError("Could not identify amount column in file.")

        raw_amounts = df[amount_col].map(normalize_amount_string).fillna(0)
        transaction_type = np.where(raw_amounts >= 0, "income", "expense")

        return pd.DataFrame(
            {"amount": raw_amounts.abs(), "transaction_type": transaction_type}
        )

    def _parse_dates(
        self, series: pd.Series, date_format: Optional[str]
    ) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        cleaned_series = series.astype(str).str.extract(
            r"(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4}(?:[,\s]+\d{1,2}:\d{2}(?::\d{2})?)?)"
        )[0]

        if date_format:
            parsed = pd.to_datetime(
                cleaned_series, format=date_format, errors="coerce"
            )
            if parsed.notna().sum() > 0:
                return parsed

        iso_mask = cleaned_series.str.match(r"^\d{4}[./\-]", na=False)
        result = pd.Series(
            pd.NaT, index=cleaned_series.index, dtype="datetime64[ns]"
        )

        if iso_mask.any():
            result.loc[iso_mask] = pd.to_datetime(
                cleaned_series[iso_mask],
                format="mixed",
                dayfirst=False,
                errors="coerce",
            )
        if (~iso_mask).any():
            result.loc[~iso_mask] = pd.to_datetime(
                cleaned_series[~iso_mask],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
        return result

    @staticmethod
    def _is_balance_row(description: str) -> bool:
        lowered = str(description).lower()
        return any(keyword in lowered for keyword in BALANCE_ROW_KEYWORDS)

    def _standardize(
        self, raw_df: pd.DataFrame, resolved_bank: str
    ) -> pd.DataFrame:
        bank_cfg = self._configs.get(resolved_bank, {})
        mapping = bank_cfg.get("column_mapping", {})

        date_col = self._resolve_column(
            raw_df,
            mapping.get(
                "date",
                [
                    "ամսաթիվ",
                    "date",
                    "trans date",
                    "ձևակերպման ամսաթիվ",
                    "գործարքի ամսաթիվ",
                ],
            ),
        )
        if date_col is None:
            for col in raw_df.columns:
                sample = raw_df[col].dropna().astype(str).head(10)
                if sample.str.contains(
                    r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}", regex=True
                ).any():
                    date_col = col
                    break

        if date_col is None:
            raise ValueError(
                f"Could not map date column. Available columns: {list(raw_df.columns)}"
            )

        amount_df = self._parse_amount_and_type(raw_df, mapping, bank_cfg)
        desc_col = self._resolve_description_column(
            raw_df, mapping, date_col, str(amount_df.columns[0])
        )

        descriptions = (
            raw_df[desc_col].fillna("").astype(str)
            if desc_col
            else pd.Series([""] * len(raw_df))
        )
        dates = self._parse_dates(raw_df[date_col], bank_cfg.get("date_format"))

        standardized = pd.DataFrame(
            {
                "date": dates,
                "description": descriptions,
                "amount": amount_df["amount"],
                "transaction_type": amount_df["transaction_type"],
            }
        )

        standardized["cleaned_description"] = standardized["description"].map(
            self.clean_description
        )
        standardized["amount"] = pd.to_numeric(standardized["amount"], errors="coerce")
        standardized = standardized.dropna(subset=["date", "amount"])
        standardized = standardized[standardized["amount"] > 0]
        standardized = standardized[
            ~standardized["description"].map(self._is_balance_row).astype(bool)
        ]
        standardized = standardized.reset_index(drop=True)
        standardized["transaction_id"] = standardized.index.astype(str)
        return standardized[[*self.STANDARD_COLUMNS, "transaction_id"]]

    def load(
        self,
        file_path: str | Path,
        bank_key: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a bank statement and return a standardized DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        resolved_bank = (
            bank_key
            if bank_key and bank_key != "auto_detect"
            else self.detect_bank(path)
        )
        reader = self._reader_factory.create(path)
        raw_df = reader.read(path, encoding)
        return self._standardize(raw_df, resolved_bank)
