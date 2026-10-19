"""File-reading strategies for bank statements (CSV, Excel, PDF).

Patterns:
  * Strategy  - :class:`StatementReader` subclasses, one per file format;
                :class:`PdfLayoutStrategy` subclasses, one per PDF layout.
  * Factory   - :class:`StatementReaderFactory` picks the reader for a path.

Every reader offers the same two operations: ``read`` (full table with
headers applied) and ``preview`` (first raw rows, used for bank detection).
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import pdfplumber

WHITESPACE = re.compile(r"\s+")

EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".xlsm", ".xlsb"})
OLE_MAGIC = b"\xd0\xcf\x11\xe0"
ZIP_MAGIC = b"PK\x03\x04"
PDF_MAGIC = b"%PDF-"

HEADER_DATE_KEYWORDS = (
    "ամսաթիվ", "date", "trans date", "transaction date", "operation date", "value date",
)
HEADER_AMOUNT_KEYWORDS = (
    "գումար", "amount", "sum", "transaction amount", "մուտք", "ելք", "debit", "credit",
)

# pdfplumber's default x_tolerance (3) glues words in some PDFs (Idram:
# "ԼԻԱՆԱ ԴԱՎԹՅԱՆ" -> "ԼԻԱՆԱԴԱՎԹՅԱՆ", "24/05/2022 12:00" -> "24/05/202212:00"),
# which breaks keyword matching and loses the time. 1.5 keeps the spaces.
PDF_X_TOLERANCE = 1.5
# A complete date at the start of a cell (2- or 4-digit year, not a fragment
# such as "24/05/202" left over when a column split cuts the year).
FULL_DATE_START = re.compile(r"^\d{1,2}[./\-]\d{1,2}[./\-](?:\d{4}|\d{2})(?!\d)")

BORDERLESS_DATE_TOKEN = re.compile(r"^\d{2}/\d{2}/\d{2},?$")
BORDERLESS_TIME_TOKEN = re.compile(r"^\d{2}:\d{2}$")
BORDERLESS_AMOUNT_TOKEN = re.compile(r"^[+\-][\d,]+\.\d{2}$")
BORDERLESS_CURRENCY_TOKEN = re.compile(r"^[A-Z]{3}$")


# --------------------------------------------------------------------------
# Small pure helpers (used by readers and the loader)
# --------------------------------------------------------------------------
def normalize_column_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE.sub(" ", str(value).replace("\n", " ")).strip()


def normalize_amount_string(value: Any) -> Optional[float]:
    """Parse '1 234,56', '-1,234.56', '+12,000.00 AMD' ... into a float.

    The LAST separator decides which one is decimal; a lone comma is decimal
    only when exactly two digits follow it (thousands separator otherwise).
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    sign = -1.0 if text.lstrip().startswith("-") else 1.0
    text = re.sub(r"^\s*[+\-]\s*", "", text)
    text = re.sub(r"(?i)(amd|usd|eur|rub|֏)", "", text)
    text = text.replace(" ", "")

    if "," in text and "." in text:
        if text.rfind(".") > text.rfind(","):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif "," in text:
        tail_len = len(text.split(",")[-1])
        text = text.replace(",", ".") if tail_len == 2 else text.replace(",", "")

    text = re.sub(r"[^\d.]", "", text)
    if not text:
        return None
    try:
        return sign * float(text)
    except ValueError:
        return None


def _magic(path: Path, n: int) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(n)
    except OSError:
        return b""


def is_excel_file(path: Path) -> bool:
    return path.suffix.lower() in EXCEL_EXTENSIONS or _magic(path, 4) in (ZIP_MAGIC, OLE_MAGIC)


def is_pdf_file(path: Path) -> bool:
    return path.suffix.lower() == ".pdf" or _magic(path, 5) == PDF_MAGIC


def excel_engine(path: Path) -> str:
    return "xlrd" if _magic(path, 4) == OLE_MAGIC else "openpyxl"


# --------------------------------------------------------------------------
# Header detection
# --------------------------------------------------------------------------
class TableExtractor:
    """Locates the header row in a raw grid and returns a named DataFrame."""

    MAX_HEADER_SCAN = 50

    def extract(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        header_idx = self._find_header_row(raw_df)
        if header_idx is None:
            return raw_df

        header_rows = raw_df.iloc[header_idx : header_idx + 2]
        has_second_header = len(header_rows) > 1 and not any(
            self._looks_like_data(normalize_column_name(v))
            for v in header_rows.iloc[1]
            if pd.notna(v)
        )
        group_row = raw_df.iloc[header_idx - 1] if header_idx > 0 else None

        headers: list[str] = []
        for col_idx in range(raw_df.shape[1]):
            name = normalize_column_name(header_rows.iloc[0, col_idx])
            if has_second_header:
                second = normalize_column_name(header_rows.iloc[1, col_idx])
                name = f"{name} {second}".strip() if second else name
            if not name and group_row is not None:
                name = normalize_column_name(group_row.iloc[col_idx])
            headers.append(name)

        df = raw_df.iloc[header_idx + (2 if has_second_header else 1) :].copy()
        df.columns = headers
        df = df.loc[:, df.columns != ""]
        return df.reset_index(drop=True)

    def _find_header_row(self, raw_df: pd.DataFrame) -> Optional[int]:
        for idx in range(min(len(raw_df), self.MAX_HEADER_SCAN)):
            row_str = " ".join(
                normalize_column_name(v).lower() for v in raw_df.iloc[idx] if pd.notna(v)
            )
            if any(k in row_str for k in HEADER_DATE_KEYWORDS) and any(
                k in row_str for k in HEADER_AMOUNT_KEYWORDS
            ):
                return idx
        return None

    @staticmethod
    def _looks_like_data(value: str) -> bool:
        if not value:
            return False
        if re.match(r"^\d{1,2}[./\-]\d{1,2}|^\d{4}[./\-]\d{1,2}", value):
            return True
        return bool(re.match(r"^[+\-]?[\d.,\s]+$", value))


# --------------------------------------------------------------------------
# Reader strategies
# --------------------------------------------------------------------------
class StatementReader(ABC):
    """Strategy: turn a file on disk into a raw tabular DataFrame."""

    #: rows returned by :meth:`preview` when ``nrows`` is not given
    DEFAULT_PREVIEW_ROWS = 25

    def __init__(self, table_extractor: TableExtractor) -> None:
        self._table_extractor = table_extractor

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Whether this reader should handle ``path``."""

    @abstractmethod
    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        """Load the file and return a table (headers applied when possible)."""

    @abstractmethod
    def preview(self, path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        """First raw rows (no header detection), for bank detection."""


# ---------------------------------------------------------------- PDF layouts
class PdfLayoutStrategy(ABC):
    """Strategy for one way a PDF statement can be laid out."""

    @abstractmethod
    def extract(self, path: Path) -> Optional[pd.DataFrame]:
        """Return transactions as a table, or None if this layout doesn't fit."""


@dataclass(frozen=True)
class BorderlessColumns:
    """x-coordinate boundaries (PDF points) of a borderless statement layout.

    Defaults match the Idram/Ameriabank-style export this was built for; a
    bank with another layout gets its own instance instead of code changes.
    """

    first_date_max_x: float = 80
    date_time_max_x: float = 160
    type_max_x: float = 260
    counterparty_max_x: float = 380
    amount_min_x: float = 560
    default_row_gap: float = 40.0


class BorderlessLayoutStrategy(PdfLayoutStrategy):
    """Rows found by date positions; fields assigned by x-coordinate."""

    def __init__(self, columns: BorderlessColumns = BorderlessColumns()) -> None:
        self.columns = columns

    def extract(self, path: Path) -> Optional[pd.DataFrame]:
        records: list[dict[str, Any]] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(x_tolerance=PDF_X_TOLERANCE)
                if words:
                    records.extend(self._page_records(words))

        rows = [r for r in records if r["date"] and r["amount_acct"] is not None]
        if not rows:
            return None
        return pd.DataFrame(
            {
                "date": [f"{r['date']} {r['time'] or ''}".strip() for r in rows],
                "description": [r["description"] for r in rows],
                "amount": [r["amount_acct"] for r in rows],
            }
        )

    def _page_records(self, words: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Split a page into one word-group per first-column date."""
        anchors = sorted(
            (
                w
                for w in words
                if BORDERLESS_DATE_TOKEN.match(w["text"])
                and w["x0"] < self.columns.first_date_max_x
            ),
            key=lambda w: w["top"],
        )
        if not anchors:
            return []
        tops = [w["top"] for w in anchors]
        gaps = [b - a for a, b in zip(tops, tops[1:])]
        lead = gaps[0] if gaps else self.columns.default_row_gap
        trail = gaps[-1] if gaps else self.columns.default_row_gap
        bounds = (
            [tops[0] - lead]
            + [(a + b) / 2 for a, b in zip(tops, tops[1:])]
            + [tops[-1] + trail]
        )
        return [
            self._parse_record([w for w in words if lo <= w["top"] < hi])
            for lo, hi in zip(bounds, bounds[1:])
        ]

    def _parse_record(self, rec_words: list[dict[str, Any]]) -> dict[str, Any]:
        cols = self.columns
        rec_sorted = sorted(rec_words, key=lambda w: (round(w["top"]), w["x0"]))

        date = time = None
        amount_tokens: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        for w in rec_sorted:
            x0, text = w["x0"], w["text"]
            if BORDERLESS_DATE_TOKEN.match(text) and x0 < cols.date_time_max_x:
                if x0 < cols.first_date_max_x:  # second (value) date is ignored
                    date = text.rstrip(",")
            elif BORDERLESS_TIME_TOKEN.match(text) and x0 < cols.date_time_max_x:
                if x0 < cols.first_date_max_x:
                    time = text
            elif x0 >= cols.amount_min_x:
                amount_tokens.append(w)
            else:
                remaining.append(w)

        amount_acct = self._account_amount(amount_tokens)

        type_words, counterparty_words, description_words = [], [], []
        for w in remaining:
            if w["x0"] < cols.type_max_x:
                type_words.append(w)
            elif w["x0"] < cols.counterparty_max_x:
                counterparty_words.append(w)
            else:
                description_words.append(w)

        counterparty = self._join(counterparty_words)
        description_text = self._join(description_words)
        description = (
            " ".join(p for p in (description_text, counterparty) if p)
            or self._join(type_words)
        )
        return {
            "date": date,
            "time": time,
            "description": description,
            "amount_acct": (
                normalize_amount_string(amount_acct) if amount_acct is not None else None
            ),
        }

    @staticmethod
    def _account_amount(tokens: list[dict[str, Any]]) -> Optional[str]:
        """Amount in ACCOUNT currency = the last (amount, currency) pair."""
        texts = [w["text"] for w in sorted(tokens, key=lambda w: (round(w["top"]), w["x0"]))]
        pairs: list[str] = []
        i = 0
        while i < len(texts) - 1:
            if BORDERLESS_AMOUNT_TOKEN.match(texts[i]) and BORDERLESS_CURRENCY_TOKEN.match(
                texts[i + 1]
            ):
                pairs.append(texts[i])
                i += 2
            else:
                i += 1
        return pairs[-1] if pairs else None

    @staticmethod
    def _join(words: list[dict[str, Any]]) -> str:
        ordered = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
        return " ".join(w["text"] for w in ordered)


class TableLayoutStrategy(PdfLayoutStrategy):
    """pdfplumber table extraction; keeps the settings that find most dated rows."""

    SETTINGS: tuple[dict[str, Any], ...] = (
        {"text_x_tolerance": PDF_X_TOLERANCE},
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "text_x_tolerance": PDF_X_TOLERANCE,
        },
    )

    def __init__(self, table_extractor: TableExtractor) -> None:
        self._table_extractor = table_extractor

    def extract(self, path: Path) -> Optional[pd.DataFrame]:
        rows = self.raw_rows(path)
        if not rows:
            return None
        widths = [len(r) for r in rows]
        width = max(set(widths), key=widths.count)
        grid = pd.DataFrame([(r + [None] * (width - len(r)))[:width] for r in rows])
        extracted = self._table_extractor.extract(grid)
        if len(extracted) > 0 and all(isinstance(c, str) and c for c in extracted.columns):
            return extracted
        return None

    def raw_rows(self, path: Path) -> list[list[Any]]:
        best_rows: list[list[Any]] = []
        best_score = -1
        with pdfplumber.open(path) as pdf:
            for settings in self.SETTINGS:
                rows = [
                    row
                    for page in pdf.pages
                    for table in page.extract_tables(settings)
                    for row in table
                ]
                score = self._transaction_rows(rows)
                if score > best_score:
                    best_rows, best_score = rows, score
        return best_rows

    @staticmethod
    def _transaction_rows(rows: list[list[Any]]) -> int:
        """Rows that START with a full date (= real transactions).

        Raw row count would let the "text" strategy win with fragment rows.
        """
        return sum(
            1
            for row in rows
            if row
            and FULL_DATE_START.match(
                str(next((c for c in row if c not in (None, "")), "")).strip()
            )
        )


class PdfStatementReader(StatementReader):
    """Tries each PDF layout strategy in order; first non-empty result wins."""

    def __init__(
        self,
        table_extractor: TableExtractor,
        layouts: Optional[Sequence[PdfLayoutStrategy]] = None,
    ) -> None:
        super().__init__(table_extractor)
        self._tables = TableLayoutStrategy(table_extractor)
        self._layouts: list[PdfLayoutStrategy] = list(
            layouts if layouts is not None else (BorderlessLayoutStrategy(), self._tables)
        )

    def can_read(self, path: Path) -> bool:
        return is_pdf_file(path)

    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        del encoding
        for layout in self._layouts:
            df = layout.extract(path)
            if df is not None and not df.empty:
                return df
        raise ValueError(
            f"Could not find any table in {path.name}. This may be a scanned/"
            "image-only PDF, which isn't supported - try exporting the statement "
            "as CSV or Excel instead."
        )

    def preview(self, path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        rows = self._tables.raw_rows(path)[: nrows or self.DEFAULT_PREVIEW_ROWS]
        return pd.DataFrame(rows)


class ExcelStatementReader(StatementReader):
    """Excel strategy (.xlsx / .xls) using magic-byte engine selection."""

    def can_read(self, path: Path) -> bool:
        return is_excel_file(path)

    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        del encoding
        raw_df = pd.read_excel(path, header=None, engine=excel_engine(path))
        return self._table_extractor.extract(raw_df)

    def preview(self, path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        return pd.read_excel(
            path, header=None, engine=excel_engine(path),
            nrows=nrows or self.DEFAULT_PREVIEW_ROWS,
        )


class CsvStatementReader(StatementReader):
    """CSV strategy with encoding fallback (always used as last resort)."""

    DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1", "cp1252", "armscii8")
    # header row + 10 data rows, as the old CSV detection used
    DEFAULT_PREVIEW_ROWS = 11

    def can_read(self, path: Path) -> bool:
        return True

    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        return self._table_extractor.extract(self._read_raw(path, encoding))

    def preview(self, path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
        return self._read_raw(path, None, nrows or self.DEFAULT_PREVIEW_ROWS)

    def _read_raw(
        self, path: Path, encoding: Optional[str], nrows: Optional[int] = None
    ) -> pd.DataFrame:
        for enc in [encoding] if encoding else self.DEFAULT_ENCODINGS:
            if not enc:
                continue
            try:
                return pd.read_csv(
                    path, header=None, encoding=enc, on_bad_lines="skip",
                    engine="python", nrows=nrows,
                )
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
                continue
        raise ValueError(f"Could not read CSV file {path.name} with any supported encodings.")


class StatementReaderFactory:
    """Factory: pick the first reader whose ``can_read`` matches the path."""

    def __init__(
        self,
        table_extractor: Optional[TableExtractor] = None,
        readers: Optional[Sequence[StatementReader]] = None,
    ) -> None:
        extractor = table_extractor or TableExtractor()
        self._readers: list[StatementReader] = list(
            readers
            if readers is not None
            else (
                PdfStatementReader(extractor),
                ExcelStatementReader(extractor),
                CsvStatementReader(extractor),  # accepts anything: keep last
            )
        )

    def create(self, path: Path) -> StatementReader:
        for reader in self._readers:
            if reader.can_read(path):
                return reader
        raise ValueError(f"No statement reader available for {path.name}")
