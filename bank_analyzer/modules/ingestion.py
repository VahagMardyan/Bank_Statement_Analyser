"""Turning a statement file into the unified transaction schema.

Collaborators (each a small single-purpose class):
    BankConfigRegistry  - bank_configs.json: names, column hints, date format,
                          detection keywords
    BankDetector        - guesses the bank from a reader's preview
    ColumnResolver      - finds the date / description columns by header hints
    AmountParser        - debit+credit or signed-amount -> amount + type
    DateParser          - DD.MM vs ISO dates without silent day/month swaps
    DescriptionBuilder  - description + counterparty columns -> one text

:class:`StatementLoaderTemplate` fixes the order of the steps in ``load()``
(Template Method); :class:`BankStatementLoader` supplies the steps.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .models import TransactionType
from .readers import (
    WHITESPACE,
    StatementReaderFactory,
    normalize_amount_string,
    normalize_column_name,
)
from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

AUTO_DETECT = "auto_detect"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "bank_configs.json"

# Header hints used when a bank's config has no ``column_mapping`` entry.
DEFAULT_COLUMN_HINTS: dict[str, list[str]] = {
    "date": ["ամսաթիվ", "date", "trans date", "ձևակերպման ամսաթիվ", "գործարքի ամսաթիվ"],
    "debit": ["debit", "դեբետ", "ելք", "ելքեր", "out", "withdrawal"],
    "credit": ["credit", "կրեդիտ", "մուտք", "մուտքեր", "in", "deposit"],
    "amount": ["գումար", "amount", "sum", "գործարքի գումար", "մուտք/ելք"],
    "description": [
        "նկարագրություն", "description", "details", "purpose", "narrative",
        "merchant", "comment",
    ],
}

# Prefer the amount converted to the ACCOUNT currency. Ameriabank CSV exports
# have both "Գործարքի գումար" (-1.99 USD) and "Գործարքի գումար հաշվի
# արժույթով" (-738.29 AMD); the first one would record USD purchases as AMD.
ACCOUNT_CURRENCY_AMOUNT_HINTS = ["հաշվի արժույթով", "amount in account currency", "account currency"]

# Columns that carry merchant / counterparty information in addition to the
# main description. ACBA puts only "Վճարում" in the description and the real
# merchant in "Գործարքի վայրը"; Ameriabank CSV has a payee column.
COUNTERPARTY_COLUMN_HINTS = (
    "գործարքի վայրը", "շահառու", "վճարող", "payee", "beneficiary",
    "counterparty", "merchant", "place", "location",
)

# Clearing/processor names that appear on almost every card row - no signal.
GENERIC_COUNTERPARTY = re.compile(
    r"արմենիան\s*քարդ|visa\s*international|mastercard|միջն\.?/|փոխհաշվարկներ",
    re.IGNORECASE,
)
EMPTY_CELL_VALUES = frozenset({"nan", "կիրառելի չէ"})

BALANCE_ROW_KEYWORDS = ("մնացորդ", "balance", "opening balance", "closing balance", "օրվա վերջին")

# Order matters: the first bank whose keyword appears wins.
DEFAULT_DETECTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "ameriabank": ("ameriabank", "ameria", "ամերիա"),
    "evocabank": ("evocabank", "evoca"),
    "acba": ("acba", "ակբա"),
    "inecobank": ("ineco", "ինեկո"),
}

DATE_LIKE = re.compile(r"\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}")
DATE_CAPTURE = re.compile(r"(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4}(?:[,\s]+\d{1,2}:\d{2}(?::\d{2})?)?)")
TEXT_LIKE = r"[A-Za-zԱ-ֆ]{2,}"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class BankConfigRegistry:
    """Read-only access to ``bank_configs.json`` (``{"banks": {key: {...}}}``).

    Per bank: ``display_name``, ``column_mapping`` ({field: [hints]}),
    ``date_format`` and optional ``detect_keywords`` ([...]).
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONFIG_PATH
        self._banks: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        banks = data.get("banks", {}) if isinstance(data, dict) else {}
        return {k: v for k, v in banks.items() if isinstance(v, dict)}

    def bank_keys(self) -> list[str]:
        return [k for k in self._banks if k != AUTO_DETECT]

    def config(self, bank_key: str) -> dict[str, Any]:
        return self._banks.get(bank_key, {})

    def display_name(self, bank_key: str) -> str:
        return str(self.config(bank_key).get("display_name", bank_key))

    def column_hints(self, bank_key: str, field: str) -> list[str]:
        mapping = self.config(bank_key).get("column_mapping", {})
        return list(mapping.get(field, DEFAULT_COLUMN_HINTS.get(field, [])))

    def date_format(self, bank_key: str) -> Optional[str]:
        return self.config(bank_key).get("date_format")

    def detection_keywords(self) -> dict[str, tuple[str, ...]]:
        """Built-in keywords, overridden/extended by ``detect_keywords`` in config."""
        keywords = dict(DEFAULT_DETECTION_KEYWORDS)
        for key, cfg in self._banks.items():
            custom = cfg.get("detect_keywords")
            if key != AUTO_DETECT and isinstance(custom, list) and custom:
                keywords[key] = tuple(str(k).lower() for k in custom)
        return keywords


# --------------------------------------------------------------------------
# Bank detection
# --------------------------------------------------------------------------
class BankDetector:
    """Guesses the bank from the first rows of a statement."""

    def __init__(
        self, registry: BankConfigRegistry, reader_factory: StatementReaderFactory
    ) -> None:
        self._keywords = registry.detection_keywords()
        self._readers = reader_factory

    def detect(self, path: Path) -> str:
        try:
            preview = self._readers.create(path).preview(path)
        except (OSError, ValueError):
            return AUTO_DETECT
        return self.detect_from_frame(preview)

    def detect_from_frame(self, raw_df: pd.DataFrame) -> str:
        text = " ".join(
            str(v).lower() for v in raw_df.to_numpy().ravel() if pd.notna(v)
        )
        for bank, words in self._keywords.items():
            if any(w in text for w in words):
                return bank
        return AUTO_DETECT


# --------------------------------------------------------------------------
# Column / value parsing
# --------------------------------------------------------------------------
class ColumnResolver:
    """Finds a column by header hints: exact match first, then substring."""

    @staticmethod
    def key(column: Any) -> str:
        return normalize_column_name(column).lower()

    def resolve(self, df: pd.DataFrame, hints: Sequence[str]) -> Optional[str]:
        by_key = {self.key(c): c for c in df.columns}
        for hint in hints:
            if hint.lower() in by_key:
                return by_key[hint.lower()]
        for col_key, original in by_key.items():
            if any(hint.lower() in col_key for hint in hints):
                return original
        return None

    def resolve_date(self, df: pd.DataFrame, hints: Sequence[str]) -> str:
        col = self.resolve(df, hints)
        if col is not None:
            return col
        for col in df.columns:  # fallback: first column that holds dates
            sample = df[col].dropna().astype(str).head(10)
            if sample.str.contains(DATE_LIKE, regex=True).any():
                return col
        raise ValueError(f"Could not map date column. Available columns: {list(df.columns)}")

    def resolve_description(
        self, df: pd.DataFrame, hints: Sequence[str], exclude: Sequence[Any]
    ) -> Optional[str]:
        excluded = {self.key(c) for c in exclude if c is not None}
        for hint in hints:
            for col in df.columns:
                k = self.key(col)
                if k not in excluded and hint.lower() in k:
                    return col
        for col in df.columns:  # fallback: first mostly-text column
            if self.key(col) in excluded:
                continue
            sample = df[col].dropna().astype(str).head(20)
            if not sample.empty and sample.str.contains(TEXT_LIKE, regex=True, na=False).mean() >= 0.3:
                return col
        return None


@dataclass(frozen=True)
class ParsedAmounts:
    """Amounts plus the columns they were read from."""

    amount: pd.Series
    transaction_type: pd.Series
    source_columns: tuple[str, ...]


class AmountParser:
    """Debit/credit pair or one signed amount column -> positive amount + type."""

    def __init__(self, resolver: ColumnResolver) -> None:
        self._resolver = resolver

    def parse(
        self,
        df: pd.DataFrame,
        debit_hints: Sequence[str],
        credit_hints: Sequence[str],
        amount_hints: Sequence[str],
        exclude: Sequence[Any] = (),
    ) -> ParsedAmounts:
        """``exclude``: columns that must never be read as amounts (the date)."""
        debit_col = self._resolver.resolve(df, debit_hints)
        credit_col = self._resolver.resolve(df, credit_hints)
        if debit_col and credit_col:
            return self._from_debit_credit(df, debit_col, credit_col)

        amount_col = (
            self._resolver.resolve(df, ACCOUNT_CURRENCY_AMOUNT_HINTS)
            or self._resolver.resolve(df, amount_hints)
            or self._first_numeric_column(df, exclude)
        )
        if amount_col is None:
            raise ValueError("Could not identify amount column in file.")
        signed = df[amount_col].map(normalize_amount_string).fillna(0)
        kind = np.where(signed >= 0, TransactionType.INCOME.value, TransactionType.EXPENSE.value)
        return ParsedAmounts(
            amount=signed.abs(),
            transaction_type=pd.Series(kind, index=df.index),
            source_columns=tuple(c for c in (amount_col, debit_col, credit_col) if c),
        )

    @staticmethod
    def _from_debit_credit(df: pd.DataFrame, debit_col: str, credit_col: str) -> ParsedAmounts:
        debit = df[debit_col].map(normalize_amount_string).fillna(0).abs()
        credit = df[credit_col].map(normalize_amount_string).fillna(0).abs()
        is_income = credit > 0
        return ParsedAmounts(
            amount=pd.Series(np.where(is_income, credit, debit), index=df.index),
            transaction_type=pd.Series(
                np.where(is_income, TransactionType.INCOME.value, TransactionType.EXPENSE.value),
                index=df.index,
            ),
            source_columns=(debit_col, credit_col),
        )

    @staticmethod
    def _first_numeric_column(df: pd.DataFrame, exclude: Sequence[Any]) -> Optional[str]:
        # "01/04/2026" parses as 1042026, so the date column must be skipped.
        for col in df.columns:
            if col in exclude:
                continue
            if df[col].map(normalize_amount_string).notna().any():
                return col
        return None


class DateParser:
    """Parses statement dates.

    ISO dates (YYYY-MM-DD) and day-first dates (DD.MM.YYYY) are parsed
    separately: one ``dayfirst=True`` pass over both swaps day and month in
    ISO dates or drops them as NaT.
    """

    def parse(self, series: pd.Series, date_format: Optional[str] = None) -> pd.Series:
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        cleaned = series.astype(str).str.extract(DATE_CAPTURE)[0]

        if date_format:
            parsed = pd.to_datetime(cleaned, format=date_format, errors="coerce")
            if parsed.notna().sum() > 0:
                return parsed

        iso = cleaned.str.match(r"^\d{4}[./\-]", na=False)
        result = pd.Series(pd.NaT, index=cleaned.index, dtype="datetime64[ns]")
        if iso.any():
            result.loc[iso] = pd.to_datetime(
                cleaned[iso], format="mixed", dayfirst=False, errors="coerce"
            )
        if (~iso).any():
            result.loc[~iso] = pd.to_datetime(
                cleaned[~iso], format="mixed", dayfirst=True, errors="coerce"
            )
        return result


class DescriptionBuilder:
    """Joins the description with merchant/payee/place columns ("desc | payee")."""

    SEPARATOR = " | "

    def __init__(self, resolver: ColumnResolver) -> None:
        self._resolver = resolver

    def build(self, df: pd.DataFrame, desc_col: Optional[str]) -> pd.Series:
        base = (
            df[desc_col].fillna("").astype(str)
            if desc_col
            else pd.Series([""] * len(df), index=df.index)
        )
        extra = self._counterparty(df, desc_col)
        return pd.Series(
            [self._merge(d, c) for d, c in zip(base, extra)], index=df.index
        )

    def _counterparty(self, df: pd.DataFrame, desc_col: Optional[str]) -> pd.Series:
        cols = [
            c
            for c in df.columns
            if c != desc_col
            and any(h in self._resolver.key(c) for h in COUNTERPARTY_COLUMN_HINTS)
        ]
        if not cols:
            return pd.Series([""] * len(df), index=df.index)
        return df[cols].apply(
            lambda row: " ".join(p for p in map(self._clean_part, row) if p), axis=1
        ).astype(str)

    @staticmethod
    def _clean_part(value: Any) -> str:
        if pd.isna(value):
            return ""
        text = WHITESPACE.sub(" ", str(value)).strip(" ,;")
        if not text or text.lower() in EMPTY_CELL_VALUES or GENERIC_COUNTERPARTY.search(text):
            return ""
        return text

    @classmethod
    def _merge(cls, desc: str, extra: str) -> str:
        desc = WHITESPACE.sub(" ", str(desc)).strip()
        extra = str(extra).strip()
        if not extra or extra.lower() in desc.lower():
            return desc
        return f"{desc}{cls.SEPARATOR}{extra}" if desc else extra


# --------------------------------------------------------------------------
# Loader (Template Method)
# --------------------------------------------------------------------------
class StatementLoaderTemplate(ABC):
    """Fixed loading pipeline; subclasses implement the individual steps."""

    STANDARD_COLUMNS = (
        "date", "description", "cleaned_description", "amount", "transaction_type",
    )

    def load(
        self,
        file_path: str | Path,
        bank_key: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """Template method: validate -> resolve bank -> read -> standardize -> filter."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        bank = self.resolve_bank(path, bank_key)
        raw = self.read_raw(path, encoding)
        rows = self.standardize(raw, bank)
        rows = self.drop_invalid_rows(rows)
        return self.finalize(rows)

    @abstractmethod
    def resolve_bank(self, path: Path, bank_key: Optional[str]) -> str: ...

    @abstractmethod
    def read_raw(self, path: Path, encoding: Optional[str]) -> pd.DataFrame: ...

    @abstractmethod
    def standardize(self, raw: pd.DataFrame, bank: str) -> pd.DataFrame: ...

    @abstractmethod
    def drop_invalid_rows(self, rows: pd.DataFrame) -> pd.DataFrame: ...

    def finalize(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Hook: stable ids and column order."""
        rows = rows.reset_index(drop=True)
        rows["transaction_id"] = rows.index.astype(str)
        return rows[[*self.STANDARD_COLUMNS, "transaction_id"]]


class BankStatementLoader(StatementLoaderTemplate):
    """Loads CSV / Excel / PDF statements from Armenian banks."""

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        reader_factory: Optional[StatementReaderFactory] = None,
        *,
        registry: Optional[BankConfigRegistry] = None,
        detector: Optional[BankDetector] = None,
        normalizer: TextNormalizer = DEFAULT_NORMALIZER,
    ) -> None:
        self._registry = registry or BankConfigRegistry(config_path)
        self._readers = reader_factory or StatementReaderFactory()
        self._detector = detector or BankDetector(self._registry, self._readers)
        self._normalizer = normalizer
        self._columns = ColumnResolver()
        self._amounts = AmountParser(self._columns)
        self._dates = DateParser()
        self._descriptions = DescriptionBuilder(self._columns)

    # -------------------------------------------------------- bank metadata
    def list_banks(self) -> list[str]:
        return self._registry.bank_keys()

    def bank_display_name(self, bank_key: str) -> str:
        return self._registry.display_name(bank_key)

    def bank_labels(self) -> dict[str, str]:
        return {
            AUTO_DETECT: "Auto Detect",
            **{key: self.bank_display_name(key) for key in self.list_banks()},
        }

    def detect_bank(self, file_path: str | Path) -> str:
        return self._detector.detect(Path(file_path))

    def clean_description(self, text: str) -> str:
        return self._normalizer.clean(text)

    # ------------------------------------------------------- template steps
    def resolve_bank(self, path: Path, bank_key: Optional[str]) -> str:
        if bank_key and bank_key != AUTO_DETECT:
            return bank_key
        return self.detect_bank(path)

    def read_raw(self, path: Path, encoding: Optional[str]) -> pd.DataFrame:
        return self._readers.create(path).read(path, encoding)

    def standardize(self, raw: pd.DataFrame, bank: str) -> pd.DataFrame:
        def hints(field: str) -> list[str]:
            return self._registry.column_hints(bank, field)

        date_col = self._columns.resolve_date(raw, hints("date"))
        amounts = self._amounts.parse(
            raw, hints("debit"), hints("credit"), hints("amount"), exclude=(date_col,)
        )
        desc_col = self._columns.resolve_description(
            raw, hints("description"), exclude=(date_col, *amounts.source_columns)
        )
        descriptions = self._descriptions.build(raw, desc_col)

        return pd.DataFrame(
            {
                "date": self._dates.parse(raw[date_col], self._registry.date_format(bank)),
                "description": descriptions,
                "cleaned_description": descriptions.map(self._normalizer.clean),
                "amount": pd.to_numeric(amounts.amount, errors="coerce"),
                "transaction_type": amounts.transaction_type,
            }
        )

    def drop_invalid_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.dropna(subset=["date", "amount"])
        rows = rows[rows["amount"] > 0]
        return rows[~rows["description"].map(self._is_balance_row).astype(bool)]

    @staticmethod
    def _is_balance_row(description: str) -> bool:
        lowered = str(description).lower()
        return any(keyword in lowered for keyword in BALANCE_ROW_KEYWORDS)
