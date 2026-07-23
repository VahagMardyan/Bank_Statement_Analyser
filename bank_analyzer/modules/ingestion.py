"""CSV and Excel ingestion and standardization for bank statements."""

from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

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

        debit_col = self._resolve_column(df, mapping.get("debit", ["debit", "ելք", "out"]))
        credit_col = self._resolve_column(df, mapping.get("credit", ["credit", "մուտք", "in"]))

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
            mapping.get("debit", ["debit", "ելք", "ելքեր", "out", "withdrawal"]),
        )
        credit_col = self._resolve_column(
            df, mapping.get("credit", ["credit", "մուտք", "մուտքեր", "in", "deposit"])
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
            r"(\d{1,4}[./\-]\d{1,2}[./\-]\d{1,4})"
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

    def _read_raw_file(
            self,
            path : Path,
            bank_key : str,
            encoding : Optional[str] = None
    ) -> pd.DataFrame:
        """ Reads CSV or Excel file into a raw DataFrame and extracts table content. """
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