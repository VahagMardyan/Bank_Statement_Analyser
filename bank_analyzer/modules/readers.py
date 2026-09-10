"""File-reading strategies for bank statements (CSV, Excel, PDF)."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pdfplumber

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b|\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b"
)
WHITESPACE = re.compile(r"\s+")

EXCEL_EXTENSIONS: frozenset[str] = frozenset({".xlsx", ".xls", ".xlsm", ".xlsb"})
OLE_MAGIC = b"\xd0\xcf\x11\xe0"
ZIP_MAGIC = b"PK\x03\x04"
PDF_MAGIC = b"%PDF-"

HEADER_DATE_KEYWORDS = (
    "ամսաթիվ",
    "date",
    "trans date",
    "transaction date",
    "operation date",
    "value date",
)
HEADER_AMOUNT_KEYWORDS = (
    "գումար",
    "amount",
    "sum",
    "transaction amount",
    "մուտք",
    "ելք",
    "debit",
    "credit",
)

BORDERLESS_DATE_TOKEN = re.compile(r"^\d{2}/\d{2}/\d{2},?$")
BORDERLESS_TIME_TOKEN = re.compile(r"^\d{2}:\d{2}$")
BORDERLESS_AMOUNT_TOKEN = re.compile(r"^[+\-][\d,]+\.\d{2}$")
BORDERLESS_CURRENCY_TOKEN = re.compile(r"^[A-Z]{3}$")


def normalize_column_name(value: Any) -> str:
    if pd.isna(value):
        return ""
    return WHITESPACE.sub(" ", str(value).replace("\n", " ")).strip()


def normalize_amount_string(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
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


def is_excel_file(path: Path) -> bool:
    if path.suffix.lower() in EXCEL_EXTENSIONS:
        return True
    try:
        magic = path.read_bytes()[:4]
        return magic == ZIP_MAGIC or magic == OLE_MAGIC
    except OSError:
        return False


def is_pdf_file(path: Path) -> bool:
    if path.suffix.lower() == ".pdf":
        return True
    try:
        return path.read_bytes()[:5] == PDF_MAGIC
    except OSError:
        return False


def excel_engine(path: Path) -> str:
    try:
        magic = path.read_bytes()[:4]
        if magic == ZIP_MAGIC:
            return "openpyxl"
        if magic == OLE_MAGIC:
            return "xlrd"
    except OSError:
        pass
    return "openpyxl"


class TableExtractor:
    """Locates the header row in a raw grid and returns a named DataFrame."""

    def extract(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        header_idx = None

        for idx in range(min(len(raw_df), 50)):
            row_vals = [
                normalize_column_name(v).lower()
                for v in raw_df.iloc[idx]
                if pd.notna(v)
            ]
            row_str = " ".join(row_vals)

            has_date = any(kw in row_str for kw in HEADER_DATE_KEYWORDS)
            has_amount = any(kw in row_str for kw in HEADER_AMOUNT_KEYWORDS)

            if has_date and has_amount:
                header_idx = idx
                break

        if header_idx is None:
            return raw_df

        header_rows = raw_df.iloc[header_idx : header_idx + 2]

        def _looks_like_data(value: str) -> bool:
            if not value:
                return False
            if re.match(r"^\d{1,2}[./\-]\d{1,2}|^\d{4}[./\-]\d{1,2}", value):
                return True
            return bool(re.match(r"^[+\-]?[\d.,\s]+$", value))

        row_below_is_data = False
        if len(header_rows) > 1:
            row_below_is_data = any(
                _looks_like_data(normalize_column_name(v))
                for v in header_rows.iloc[1]
                if pd.notna(v)
            )

        has_second_header = len(header_rows) > 1 and not row_below_is_data
        combined_headers: list[str] = []
        group_row = raw_df.iloc[header_idx - 1] if header_idx > 0 else None

        for col_idx in range(raw_df.shape[1]):
            val1 = normalize_column_name(header_rows.iloc[0, col_idx])

            if has_second_header:
                val2 = normalize_column_name(header_rows.iloc[1, col_idx])
                combined = f"{val1} {val2}".strip() if val2 else val1
            else:
                combined = val1

            if not combined and group_row is not None:
                combined = normalize_column_name(group_row.iloc[col_idx])

            combined_headers.append(combined)

        skip_offset = 2 if has_second_header else 1
        df = raw_df.iloc[header_idx + skip_offset :].copy()
        df.columns = combined_headers
        df = df.loc[:, df.columns != ""]
        return df.reset_index(drop=True)


class StatementReader(ABC):
    """Strategy: turn a file on disk into a raw tabular DataFrame."""

    def __init__(self, table_extractor: TableExtractor) -> None:
        self._table_extractor = table_extractor

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Whether this reader should handle ``path``."""

    @abstractmethod
    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        """Load the file and return a table (headers applied when possible)."""


class PdfStatementReader(StatementReader):
    """PDF strategy: borderless layout first, then pdfplumber tables."""

    def can_read(self, path: Path) -> bool:
        return is_pdf_file(path)

    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        del encoding
        positional_df = self._extract_borderless_pdf_transactions(path)
        if positional_df is not None and not positional_df.empty:
            return positional_df

        rows = self._extract_pdf_tables(path)
        if rows:
            widths = [len(r) for r in rows]
            target_width = max(set(widths), key=widths.count)
            normalized_rows = [
                (row + [None] * (target_width - len(row)))[:target_width]
                for row in rows
            ]
            candidate_df = pd.DataFrame(normalized_rows)
            extracted = self._table_extractor.extract(candidate_df)
            if len(extracted) > 0 and all(
                isinstance(c, str) and c for c in extracted.columns
            ):
                return extracted

        raise ValueError(
            f"Could not find any table in {path.name}. "
            "This may be a scanned/image-only PDF, which isn't "
            "supported - try exporting the statement as CSV or Excel "
            "instead."
        )

    def extract_preview_rows(self, path: Path, limit: int = 25) -> list[list[Any]]:
        return self._extract_pdf_tables(path)[:limit]

    @staticmethod
    def _extract_pdf_tables(path: Path) -> list[list[Any]]:
        strategies: list[dict[str, str]] = [
            {},
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
            },
        ]

        best_rows: list[list[Any]] = []
        with pdfplumber.open(path) as pdf:
            for settings in strategies:
                rows: list[list[Any]] = []
                for page in pdf.pages:
                    tables = (
                        page.extract_tables(settings)
                        if settings
                        else page.extract_tables()
                    )
                    for table in tables:
                        rows.extend(table)
                if len(rows) > len(best_rows):
                    best_rows = rows

        return best_rows

    def _extract_borderless_pdf_transactions(self, path: Path) -> Optional[pd.DataFrame]:
        """Recover transactions from a borderless/lineless PDF statement layout."""
        records: list[dict[str, Any]] = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words:
                    continue

                col1_dates = sorted(
                    (
                        w
                        for w in words
                        if BORDERLESS_DATE_TOKEN.match(w["text"]) and w["x0"] < 80
                    ),
                    key=lambda w: w["top"],
                )
                if not col1_dates:
                    continue

                tops = [w["top"] for w in col1_dates]
                gaps = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
                lead_gap = gaps[0] if gaps else 40.0
                trail_gap = gaps[-1] if gaps else 40.0
                bounds = (
                    [tops[0] - lead_gap]
                    + [(tops[i] + tops[i + 1]) / 2 for i in range(len(tops) - 1)]
                    + [tops[-1] + trail_gap]
                )

                for i in range(len(col1_dates)):
                    lo, hi = bounds[i], bounds[i + 1]
                    rec_words = [w for w in words if lo <= w["top"] < hi]
                    records.append(self._parse_borderless_record(rec_words))

        if not records:
            return None

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

    @staticmethod
    def _parse_borderless_record(rec_words: list[dict[str, Any]]) -> dict[str, Any]:
        rec_sorted = sorted(rec_words, key=lambda w: (round(w["top"]), w["x0"]))

        date1 = date2 = time1 = time2 = None
        amount_tokens: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []

        for w in rec_sorted:
            x0, text = w["x0"], w["text"]
            if BORDERLESS_DATE_TOKEN.match(text) and x0 < 160:
                if x0 < 80:
                    date1 = text.rstrip(",")
                else:
                    date2 = text.rstrip(",")
            elif BORDERLESS_TIME_TOKEN.match(text) and x0 < 160:
                if x0 < 80:
                    time1 = text
                else:
                    time2 = text
            elif x0 >= 560:
                amount_tokens.append(w)
            else:
                remaining.append(w)

        amount_tokens.sort(key=lambda w: (round(w["top"]), w["x0"]))
        pairs: list[tuple[str, str]] = []
        i = 0
        texts = [w["text"] for w in amount_tokens]
        while i < len(texts) - 1:
            if BORDERLESS_AMOUNT_TOKEN.match(texts[i]) and BORDERLESS_CURRENCY_TOKEN.match(
                texts[i + 1]
            ):
                pairs.append((texts[i], texts[i + 1]))
                i += 2
            else:
                i += 1

        amount_native, amount_acct = None, None
        if len(pairs) >= 2:
            amount_native = pairs[0][0]
            amount_acct = pairs[-1][0]
        elif len(pairs) == 1:
            amount_native = amount_acct = pairs[0][0]

        type_words: list[dict[str, Any]] = []
        counterparty_words: list[dict[str, Any]] = []
        description_words: list[dict[str, Any]] = []
        for w in remaining:
            x0 = w["x0"]
            if x0 < 260:
                type_words.append(w)
            elif x0 < 380:
                counterparty_words.append(w)
            else:
                description_words.append(w)

        def join(ws: list[dict[str, Any]]) -> str:
            ws_sorted = sorted(ws, key=lambda item: (round(item["top"]), item["x0"]))
            return " ".join(item["text"] for item in ws_sorted)

        txn_type = join(type_words)
        counterparty = join(counterparty_words)
        description_text = join(description_words)
        description = (
            " ".join(part for part in (description_text, counterparty) if part) or txn_type
        )

        del date2, time2, amount_native

        return {
            "date": date1,
            "time": time1,
            "description": description,
            "amount_acct": (
                normalize_amount_string(amount_acct) if amount_acct is not None else None
            ),
        }


class ExcelStatementReader(StatementReader):
    """Excel strategy (.xlsx / .xls) using magic-byte engine selection."""

    def can_read(self, path: Path) -> bool:
        return is_excel_file(path)

    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        del encoding
        engine = excel_engine(path)
        raw_df = pd.read_excel(path, header=None, engine=engine)
        return self._table_extractor.extract(raw_df)

    def preview(self, path: Path, nrows: int = 25) -> pd.DataFrame:
        engine = excel_engine(path)
        return pd.read_excel(path, header=None, engine=engine, nrows=nrows)


class CsvStatementReader(StatementReader):
    """CSV strategy with encoding fallback (always used as last resort)."""

    DEFAULT_ENCODINGS: tuple[str, ...] = (
        "utf-8-sig",
        "utf-8",
        "latin-1",
        "cp1252",
        "armscii8",
    )

    def can_read(self, path: Path) -> bool:
        return True

    def read(self, path: Path, encoding: Optional[str] = None) -> pd.DataFrame:
        encodings_to_try = [encoding] if encoding else list(self.DEFAULT_ENCODINGS)
        raw_df: Optional[pd.DataFrame] = None

        for enc in encodings_to_try:
            if not enc:
                continue
            try:
                raw_df = pd.read_csv(
                    path,
                    header=None,
                    encoding=enc,
                    on_bad_lines="skip",
                    engine="python",
                )
                break
            except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
                continue

        if raw_df is None:
            raise ValueError(
                f"Could not read CSV file {path.name} with any supported encodings."
            )

        return self._table_extractor.extract(raw_df)


class StatementReaderFactory:
    """Factory: pick the first reader whose ``can_read`` matches the path."""

    def __init__(self, table_extractor: TableExtractor | None = None) -> None:
        extractor = table_extractor or TableExtractor()
        self._pdf = PdfStatementReader(extractor)
        self._excel = ExcelStatementReader(extractor)
        self._csv = CsvStatementReader(extractor)
        self._readers: list[StatementReader] = [self._pdf, self._excel, self._csv]

    def create(self, path: Path) -> StatementReader:
        for reader in self._readers:
            if reader.can_read(path):
                return reader
        raise ValueError(f"No statement reader available for {path.name}")

    @property
    def pdf_reader(self) -> PdfStatementReader:
        return self._pdf

    @property
    def excel_reader(self) -> ExcelStatementReader:
        return self._excel
