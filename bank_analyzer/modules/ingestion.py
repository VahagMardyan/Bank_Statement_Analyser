"""CSV, Excel, and PDF ingestion and standardization for bank statements."""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import pdfplumber

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

DATE_PATTERN = re.compile(
    r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b|\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b"
)
LONG_NUMERIC_ID = re.compile(r"\b\d{6,}\b")
WHITESPACE = re.compile(r"\s+")
SIGNED_AMOUNT_PATTERN = re.compile(r"^\s*[+\-]\s*[\d\s]+(?:[.,]\d+)?")

# Used by the borderless/lineless PDF layout parser (see
# _extract_borderless_pdf_transactions), where records are recovered from
# word positions rather than pdfplumber's ruled-table detection.
BORDERLESS_DATE_TOKEN = re.compile(r"^\d{2}/\d{2}/\d{2},?$")
BORDERLESS_TIME_TOKEN = re.compile(r"^\d{2}:\d{2}$")
BORDERLESS_AMOUNT_TOKEN = re.compile(r"^[+\-][\d,]+\.\d{2}$")
BORDERLESS_CURRENCY_TOKEN = re.compile(r"^[A-Z]{3}$")

BALANCE_ROW_KEYWORDS = (
    "մնացորդ",
    "balance",
    "opening balance",
    "closing balance",
    "օրվա վերջին",
)
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


class BankStatementLoader:
    """Load and normalize bank statement CSV/Excel files into a unified schema."""

    STANDARD_COLUMNS = (
        "date",
        "description",
        "cleaned_description",
        "amount",
        "transaction_type",
    )
    EXCEL_EXTENSIONS = {".xlsx", ".xls", ".xlsm", ".xlsb"}

    def __init__(self, config_path: Optional[str | Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.config_path = (
            Path(config_path)
            if config_path
            else base_dir / "config" / "bank_configs.json"
        )
        self._configs = self._load_configs()

    def _load_configs(self) -> dict[str, Any]:
        if not self.config_path.exists():
            return {"auto_detect": {}}
        with open(self.config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("banks", {})

    def list_banks(self) -> list[str]:
        """Return configured bank keys excluding auto-detect."""
        return [k for k in self._configs if k != "auto_detect"]

    @staticmethod
    def clean_description(text: str) -> str:
        """Sanitize raw transaction description text."""
        if not isinstance(text, str) or not text.strip():
            return ""

        cleaned = text.lower()
        cleaned = DATE_PATTERN.sub(" ", cleaned)
        cleaned = LONG_NUMERIC_ID.sub(" ", cleaned)
        tokens = cleaned.split()
        filtered = [t for t in tokens if t not in NOISE_TOKENS and len(t) > 1]
        return WHITESPACE.sub(" ", " ".join(filtered)).strip()

    @staticmethod
    def _normalize_column_name(value: Any) -> str:
        if pd.isna(value):
            return ""
        return WHITESPACE.sub(" ", str(value).replace("\n", " ")).strip()

    @staticmethod
    def _is_excel_file(path: Path) -> bool:
        if path.suffix.lower() in BankStatementLoader.EXCEL_EXTENSIONS:
            return True
        try:
            magic = path.read_bytes()[:4]
            return magic == b"PK\x03\x04" or magic == b"\xd0\xcf\11\xe0"
        except Exception:
            return False

    @staticmethod
    def _is_pdf_file(path: Path) -> bool:
        if path.suffix.lower() == ".pdf":
            return True
        try:
            return path.read_bytes()[:5] == b"%PDF-"
        except Exception:
            return False

    @staticmethod
    def _excel_engine(path: Path) -> str:
        try:
            magic = path.read_bytes()[:4]
            if magic == b"PK\x03\x04":
                return "openpyxl"
            if magic == b"\xd0\xcf\11\xe0":
                return "xlrd"
        except Exception:
            pass
        return "openpyxl"

    @staticmethod
    def _normalize_amount_string(value: Any) -> Optional[float]:
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
            if self._is_pdf_file(path):
                rows = self._extract_pdf_tables(path)[:25]
                if rows:
                    raw_df = pd.DataFrame(rows)
                    return self._detect_bank_from_content(raw_df)
                return "auto_detect"
            if self._is_excel_file(path):
                engine = self._excel_engine(path)
                raw_df = pd.read_excel(path, header=None, engine=engine, nrows=25)
                return self._detect_bank_from_content(raw_df)
            else:
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
                            + " ".join(
                                str(col).lower() for col in preview.columns
                            )
                        )
                        if "ameriabank" in text or "ameria" in text:
                            return "ameriabank"
                        if "evocabank" in text or "evoca" in text:
                            return "evocabank"
                        if "acba" in text:
                            return "acba"
                        if "ineco" in text:
                            return "inecobank"
                    except Exception:
                        continue
        except Exception:
            pass
        return "auto_detect"

    def _find_and_extract_table(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        header_idx = None

        for idx in range(min(len(raw_df), 50)):
            row_vals = [
                self._normalize_column_name(v).lower()
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
                _looks_like_data(self._normalize_column_name(v))
                for v in header_rows.iloc[1]
                if pd.notna(v)
            )

        has_second_header = len(header_rows) > 1 and not row_below_is_data

        combined_headers = []

        group_row = raw_df.iloc[header_idx - 1] if header_idx > 0 else None

        for col_idx in range(raw_df.shape[1]):
            val1 = self._normalize_column_name(header_rows.iloc[0, col_idx])

            if has_second_header:
                val2 = self._normalize_column_name(header_rows.iloc[1, col_idx])
                combined = f"{val1} {val2}".strip() if val2 else val1
            else:
                combined = val1

            if not combined and group_row is not None:
                combined = self._normalize_column_name(group_row.iloc[col_idx])

            combined_headers.append(combined)

        skip_offset = 2 if has_second_header else 1
        df = raw_df.iloc[header_idx + skip_offset :].copy()
        df.columns = combined_headers

        df = df.loc[:, df.columns != ""]
        return df.reset_index(drop=True)
    
    def _resolve_column(
        self, df: pd.DataFrame, candidates: list[str]
    ) -> Optional[str]:
        normalized = {
            self._normalize_column_name(column).lower(): column
            for column in df.columns
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
            self._normalize_column_name(date_col).lower(),
            self._normalize_column_name(amount_col).lower(),
        }

        debit_col = self._resolve_column(df, mapping.get("debit", ["debit", "դեբետ", "ելք", "out"]))
        credit_col = self._resolve_column(df, mapping.get("credit", ["credit", "կրեդիտ", "մուտք", "in"]))

        if debit_col:
            excluded.add(self._normalize_column_name(debit_col).lower())
        if credit_col:
            excluded.add(self._normalize_column_name(credit_col).lower())

        candidates = mapping.get(
            "description",
            ["նկարագրություն", "description", "details", "purpose", "narrative", "merchant", "comment"],
        )

        for candidate in candidates:
            for col in df.columns:
                col_norm = self._normalize_column_name(col).lower()
                if col_norm not in excluded and candidate.lower() in col_norm:
                    return col

        for col in df.columns:
            col_norm = self._normalize_column_name(col).lower()
            if col_norm in excluded:
                continue
            sample = df[col].dropna().astype(str).head(20)
            if sample.empty:
                continue

            text_ratio = sample.str.contains(
                r"[A-Za-zԱ-ֆ]{2,}", regex=True, na=False
            )
            if text_ratio.mean() >= 0.3:
                return col

        return None
        
    def _parse_amount_and_type(
        self, df: pd.DataFrame, mapping: dict[str, list[str]], bank_cfg: dict[str, Any]
    ) -> pd.DataFrame:
        debit_col = self._resolve_column(
            df,
            mapping.get("debit", ["debit", "դեբետ", "ելք", "ելքեր", "out", "withdrawal"]),
        )
        credit_col = self._resolve_column(
            df, mapping.get("credit", ["credit", "կրեդիտ", "մուտք", "մուտքեր", "in", "deposit"])
        )

        if debit_col and credit_col:
            debit = (
                df[debit_col].map(self._normalize_amount_string).fillna(0).abs()
            )
            credit = (
                df[credit_col].map(self._normalize_amount_string).fillna(0).abs()
            )

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
                vals = df[col].map(self._normalize_amount_string).dropna()
                if len(vals) > 0:
                    amount_col = col
                    break

        if amount_col is None:
            raise ValueError("Could not identify amount column in file.")

        raw_amounts = df[amount_col].map(self._normalize_amount_string).fillna(0)
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
    def _extract_pdf_tables(path: Path) -> list[list[Any]]:
        if pdfplumber is None:
            raise ImportError(
                "pdfplumber is required to read PDF statements. "
                "Install it with: pip install pdfplumber"
            )

        strategies = [
            {},  # pdfplumber default: line-based detection
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

    @staticmethod
    def _extract_borderless_pdf_transactions(path: Path) -> Optional[pd.DataFrame]:
        """Recovers transactions from a borderless/lineless PDF statement
        layout (e.g. Ameriabank's account-statement export) where there are
        no ruling lines and each column wraps its text onto a different
        number of physical lines per record, so pdfplumber's line- or
        text-position table detection cannot reconstruct clean rows -
        adjacent records' wrapped lines get interleaved by pdfplumber
        because it groups words by absolute vertical position only,
        ignoring which column they belong to.

        Instead, this anchors on the two unambiguous, single-line, regex-
        identifiable tokens every record has exactly once - the "date,"
        column value (leftmost, e.g. "13/07/26,") marks where a record
        starts, and the trailing amount/currency tokens (rightmost, e.g.
        "-710.00 AMD -710.00 AMD") are extracted directly. Every other word
        in the vertical band between one record's date token and the next
        is bucketed into type/counterparty/description by x-position and
        joined back together, since those fields are free text and don't
        need to be unambiguous the way dates and amounts do.

        Returns a DataFrame with plain "date"/"description"/"amount"
        columns (already resolvable by the standard column-mapping logic)
        or None if this page layout isn't detected.
        """
        records: list[dict[str, Any]] = []

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                words = page.extract_words()
                if not words:
                    continue

                col1_dates = sorted(
                    (w for w in words if BORDERLESS_DATE_TOKEN.match(w["text"]) and w["x0"] < 80),
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
                    records.append(BankStatementLoader._parse_borderless_record(rec_words))

        if not records:
            return None

        rows = [
            r
            for r in records
            if r["date"] and r["amount_acct"] is not None
        ]
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
        """Splits one record's words (see _extract_borderless_pdf_transactions)
        into date/time/type/counterparty/description/amount fields using
        regex token matching plus x-position bucketing."""
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
            if BORDERLESS_AMOUNT_TOKEN.match(texts[i]) and BORDERLESS_CURRENCY_TOKEN.match(texts[i + 1]):
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

        type_words, counterparty_words, description_words = [], [], []
        for w in remaining:
            x0 = w["x0"]
            if x0 < 260:
                type_words.append(w)
            elif x0 < 380:
                counterparty_words.append(w)
            else:
                description_words.append(w)

        def join(ws: list[dict[str, Any]]) -> str:
            ws_sorted = sorted(ws, key=lambda w: (round(w["top"]), w["x0"]))
            return " ".join(w["text"] for w in ws_sorted)

        txn_type = join(type_words)
        counterparty = join(counterparty_words)
        description_text = join(description_words)
        description = " ".join(part for part in (description_text, counterparty) if part) or txn_type

        return {
            "date": date1,
            "time": time1,
            "description": description,
            "amount_native": amount_native,
            "amount_acct": (
                BankStatementLoader._normalize_amount_string(amount_acct)
                if amount_acct is not None
                else None
            ),
        }

    def _read_pdf_file(self, path: Path) -> pd.DataFrame:
        """Reads a PDF bank statement into a raw DataFrame and extracts
        table content, reusing the same header-detection/normalization
        pipeline used for CSV and Excel statements."""
        positional_df = self._extract_borderless_pdf_transactions(path)
        if positional_df is not None and not positional_df.empty:
            return positional_df

        rows = self._extract_pdf_tables(path)
        raw_df = None
        if rows:
            widths = [len(r) for r in rows]
            target_width = max(set(widths), key=widths.count)
            normalized_rows = [
                (row + [None] * (target_width - len(row)))[:target_width]
                for row in rows
            ]
            candidate_df = pd.DataFrame(normalized_rows)
            extracted = self._find_and_extract_table(candidate_df)
            if len(extracted) > 0 and all(
                isinstance(c, str) and c for c in extracted.columns
            ):
                raw_df = extracted

        if raw_df is None:
            raise ValueError(
                f"Could not find any table in {path.name}. "
                "This may be a scanned/image-only PDF, which isn't "
                "supported - try exporting the statement as CSV or Excel "
                "instead."
            )

        return raw_df

        widths = [len(r) for r in rows]
        target_width = max(set(widths), key=widths.count)
        normalized_rows = [
            (row + [None] * (target_width - len(row)))[:target_width]
            for row in rows
        ]

        raw_df = pd.DataFrame(normalized_rows)
        return self._find_and_extract_table(raw_df)

    def _read_raw_file(
            self,
            path : Path,
            bank_key : str,
            encoding : Optional[str] = None
    ) -> pd.DataFrame:
        """ Reads CSV, Excel, or PDF file into a raw DataFrame and extracts table content. """
        if self._is_pdf_file(path):
            return self._read_pdf_file(path)

        if self._is_excel_file(path):
            engine = self._excel_engine(path)
            raw_df = pd.read_excel(path, header=None, engine=engine)
            return self._find_and_extract_table(raw_df)

        # CSV handling with encoding fallback
        encodings_to_try = [encoding] if encoding else ["utf-8-sig", "utf-8", "latin-1", "cp1252", "armscii8"]
        raw_df = None

        for enc in encodings_to_try:
            if not enc:
                continue
            try:
                raw_df = pd.read_csv(
                    path,
                    header=None,
                    encoding=enc,
                    on_bad_lines="skip",
                    engine="python"
                )
                break
            except Exception:
                continue

        if raw_df is None:
            raise ValueError(f"Could not read CSV file {path.name} with any supported encodings.")

        return self._find_and_extract_table(raw_df)

    @staticmethod
    def _is_balance_row(description: str) -> bool:
        lowered = str(description).lower()
        return any(keyword in lowered for keyword in BALANCE_ROW_KEYWORDS)

    def load(
        self,
        file_path: str | Path,
        bank_key: Optional[str] = None,
        encoding: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a bank statement (CSV/Excel) and return a standardized DataFrame."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        resolved_bank = (
            bank_key
            if bank_key and bank_key != "auto_detect"
            else self.detect_bank(path)
        )
        raw_df = self._read_raw_file(path, resolved_bank, encoding)

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
