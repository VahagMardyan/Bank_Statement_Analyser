"""Statements in foreign currencies: find the currency, convert amounts to AMD.

Classes:
    CurrencyDetector     - finds the account currency of one statement
    CurrencyApiClient    - downloads the latest rates from currencyapi.com
    RateStore            - the saved snapshots in data/exchange_rates.json
    ExchangeRateService  - decides when to download and when to keep a snapshot
    CurrencyConverter    - converts ``amount`` to AMD on import, keeps the
                           original values, and converts AMD to any supported
                           currency for display and export

Rates are stored as "units of a currency per 1 USD", exactly as currencyapi
returns them, so ``amd = amount / rate[cur] * rate["AMD"]``.

A download happens at most every ``CURRENCY_REFRESH_DAYS`` days (default 3),
and only when a statement in another currency is loaded. A new dated snapshot
is kept only if some currency moved ``CURRENCY_CHANGE_PCT`` percent (default 3)
against AMD since the last snapshot; otherwise only ``last_checked`` changes.
Each row is converted with the newest snapshot on or before its date.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

BASE_CURRENCY = "AMD"
# AED = UAE dirham (Dubai), CNY = Chinese yuan
SUPPORTED_CURRENCIES: tuple[str, ...] = ("AMD", "USD", "EUR", "RUB", "GBP", "GEL", "AED", "CNY")
DEFAULT_RATES_PATH = Path(__file__).resolve().parent.parent / "data" / "exchange_rates.json"
API_URL = "https://api.currencyapi.com/v3/latest"

DEFAULT_REFRESH_DAYS = 3.0
DEFAULT_CHANGE_PCT = 3.0
# After a failed download, wait this long before trying again
# when the saved rates already cover what is needed. When a needed currency is
# missing, try again sooner, but not on every Streamlit rerun.
RETRY_AFTER = timedelta(hours=6)
RETRY_MISSING_AFTER = timedelta(minutes=5)


# Reading a currency out of text
_CODES = r"AMD|USD|EUR|RUB|RUR|GBP|GEL|AED|CNY|RMB"
_ALIASES = {"RUR": "RUB", "RMB": "CNY"}
_CODE = re.compile(rf"(?<![A-Za-z])({_CODES})(?![A-Za-z])", re.IGNORECASE)
_NAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"դրամ|֏|драм", re.IGNORECASE), "AMD"),
    (re.compile(r"\$|դոլար|доллар|dollar", re.IGNORECASE), "USD"),
    (re.compile(r"€|եվրո|евро|euro", re.IGNORECASE), "EUR"),
    (re.compile(r"₽|ռուբլի|рубл|ruble|rouble", re.IGNORECASE), "RUB"),
    (re.compile(r"£|ֆունտ|фунт|pound", re.IGNORECASE), "GBP"),
    (re.compile(r"₾|լարի|лари|lari", re.IGNORECASE), "GEL"),
    (re.compile(r"د\.?إ|դիրհամ|дирхам|dirham", re.IGNORECASE), "AED"),
    (re.compile(r"¥|元|յուան|юан|yuan|renminbi", re.IGNORECASE), "CNY"),
)
_SYMBOLS = r"֏|\$|€|₽|£|₾|¥"
_AMOUNT_SUFFIX = re.compile(
    rf"(?:(?<![A-Za-z])({_CODES})|({_SYMBOLS}))\s*$", re.IGNORECASE
)
_AMOUNT_PREFIX = re.compile(rf"^\s*[+\-]?\s*({_SYMBOLS})")
_HEADING_CODE = re.compile(rf"\(\s*({_CODES})\s*\)", re.IGNORECASE)

# "Account currency: USD" and the same in Armenian / Russian. Anchored to the
# start of a cell, so a column title like "Գործարքի գումար քարտի արժույթով"
# (amount in card currency) is not taken for a label.
ACCOUNT_LABEL = re.compile(
    r"^\s*(?:հաշվի\s+արժույթ|քարտի\s+արժույթ|account\s+currency|валюта\s+(?:сч[её]та|карты))"
    r"[\s:`'՝,.\-]*",
    re.IGNORECASE,
)
GENERIC_LABEL = re.compile(r"^\s*(?:արժույթ|currency|валюта)(?!\w)[\s:`'՝,.\-]*", re.IGNORECASE)


def parse_currency(text: Any) -> Optional[str]:
    """'USD', 'ՀՀ դրամ', '€' ... -> ISO code, or None."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return None
    value = str(text).strip()
    if not value:
        return None
    match = _CODE.search(value)
    if match:
        code = match.group(1).upper()
        return _ALIASES.get(code, code)
    for pattern, code in _NAMES:
        if pattern.search(value):
            return code
    return None


def amount_currency(value: Any) -> Optional[str]:
    """Currency written next to an amount: '-300.00 AMD', '$12.50' ..."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    match = _AMOUNT_SUFFIX.search(text) or _AMOUNT_PREFIX.search(text)
    if not match:
        return None
    return parse_currency(next(g for g in match.groups() if g))


# Detection
@dataclass(frozen=True)
class DetectedCurrency:
    code: str
    source: str  # "header", "amounts", "column" or "default"


class CurrencyDetector:
    """Finds the account currency of a statement.

    Order: the header block above the table ("Հաշվի արժույթ` USD"), then a code
    written next to the amounts, then a currency column with one single value,
    then AMD. The per-row "Արժույթ" column in Ameriabank xlsx files is the
    currency of each purchase, not of the account, so it comes late and only
    counts when every row has the same value.
    """

    def __init__(self, default: str = BASE_CURRENCY) -> None:
        self.default = default

    def detect(
        self,
        table: pd.DataFrame,
        amount_columns: Sequence[Any] = (),
        preamble: Optional[Sequence[Sequence[str]]] = None,
    ) -> DetectedCurrency:
        rows = [list(r) for r in (preamble if preamble is not None else table.attrs.get("preamble", []))]
        code = self.from_preamble(rows)
        if code:
            return DetectedCurrency(code, "header")
        code = self.from_amounts(table, amount_columns)
        if code:
            return DetectedCurrency(code, "amounts")
        code = self.from_currency_column(table)
        if code:
            return DetectedCurrency(code, "column")
        return DetectedCurrency(self.default, "default")

    def from_preamble(self, rows: list[list[str]]) -> Optional[str]:
        for label in (ACCOUNT_LABEL, GENERIC_LABEL):
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    match = label.match(str(cell))
                    if not match:
                        continue
                    code = parse_currency(str(cell)[match.end():]) or self._neighbour(rows, r, c)
                    if code:
                        return code
        # Summary headings such as "ԳՈՐԾԱՐՔՆԵՐԻ ԱՄՓՈՓ ՏԵՂԵԿԱՏՎՈՒԹՅՈՒՆ (USD)"
        for row in rows:
            for cell in row:
                match = _HEADING_CODE.search(str(cell))
                if match:
                    return parse_currency(match.group(1))
        # Totals and balances: "Ընդհանուր ելքեր 123,456.00 AMD" (Ameria PDF)
        codes = [
            amount_currency(cell)
            for row in rows
            for cell in row
            if re.search(r"\d", str(cell))
        ]
        codes = [c for c in codes if c]
        if codes:
            return max(set(codes), key=codes.count)
        return None

    @staticmethod
    def _neighbour(rows: list[list[str]], r: int, c: int) -> Optional[str]:
        """Value of a label: the next filled cell to the right, else the cell below."""
        right = next((v for v in rows[r][c + 1:] if str(v).strip()), None)
        code = parse_currency(right)
        if code:
            return code
        if r + 1 < len(rows) and c < len(rows[r + 1]):
            return parse_currency(rows[r + 1][c])
        return None

    @staticmethod
    def from_amounts(table: pd.DataFrame, columns: Sequence[Any]) -> Optional[str]:
        for col in columns:
            if col not in table.columns:
                continue
            codes = table[col].map(amount_currency).dropna()
            if not codes.empty:
                return str(codes.mode().iloc[0])
        return None

    @staticmethod
    def from_currency_column(table: pd.DataFrame) -> Optional[str]:
        for col in table.columns:
            if not (GENERIC_LABEL.match(str(col)) or ACCOUNT_LABEL.match(str(col))):
                continue
            codes = set(table[col].map(parse_currency).dropna())
            if len(codes) == 1:
                return codes.pop()
        return None


# Rates
class RateSourceError(Exception):
    """The rates could not be downloaded (no key, network, bad answer)."""


class RatesUnavailableError(Exception):
    """A statement needs a currency we have no rate for, and none could be fetched."""

    def __init__(self, currency: str) -> None:
        super().__init__(f"No exchange rate for {currency}")
        self.currency = currency


@dataclass(frozen=True)
class RateSnapshot:
    """Rates of one day, as units of each currency per 1 USD."""

    day: date
    rates: dict[str, float]

    def has(self, currency: str) -> bool:
        return currency in self.rates and BASE_CURRENCY in self.rates

    def amd_price(self, currency: str) -> float:
        """How many AMD one unit of ``currency`` costs."""
        return self.rates[BASE_CURRENCY] / self.rates[currency]

    def to_amd(self, amount: float, currency: str) -> float:
        return amount * self.amd_price(currency)

    def moved_pct(self, older: "RateSnapshot") -> float:
        """Largest change, in percent, of any currency's AMD price since ``older``."""
        moves = [
            abs(self.amd_price(c) / older.amd_price(c) - 1) * 100
            for c in self.rates
            if c != BASE_CURRENCY and older.has(c) and self.has(c)
        ]
        return max(moves, default=100.0)

    def to_json(self) -> dict[str, Any]:
        return {"date": self.day.isoformat(), "rates": self.rates}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RateSnapshot":
        return cls(
            date.fromisoformat(str(data["date"])),
            {str(k): float(v) for k, v in data["rates"].items()},
        )


@dataclass
class RateFile:
    snapshots: list[RateSnapshot] = field(default_factory=list)
    last_checked: Optional[datetime] = None
    last_attempt: Optional[datetime] = None
    last_failed: Optional[datetime] = None  # set by a failed download, cleared by a good one


class RateStore:
    """Snapshots kept in a small JSON file (lost on Streamlit Cloud restarts, which is fine)."""

    def __init__(self, path: str | Path = DEFAULT_RATES_PATH) -> None:
        self.path = Path(path)

    def load(self) -> RateFile:
        if not self.path.exists():
            return RateFile()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return RateFile(
                snapshots=sorted(
                    (RateSnapshot.from_json(s) for s in data.get("snapshots", [])),
                    key=lambda s: s.day,
                ),
                last_checked=self._time(data.get("last_checked")),
                last_attempt=self._time(data.get("last_attempt")),
                last_failed=self._time(data.get("last_failed")),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Ignoring broken rates file %s: %s", self.path, exc)
            return RateFile()

    def save(self, file: RateFile) -> None:
        data = {
            "base": "USD",
            "last_checked": file.last_checked.isoformat(timespec="seconds") if file.last_checked else None,
            "last_attempt": file.last_attempt.isoformat(timespec="seconds") if file.last_attempt else None,
            "last_failed": file.last_failed.isoformat(timespec="seconds") if file.last_failed else None,
            "snapshots": [s.to_json() for s in file.snapshots],
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:  # a read-only disk must not break the upload
            logger.warning("Could not save rates to %s: %s", self.path, exc)

    @staticmethod
    def _time(value: Any) -> Optional[datetime]:
        return datetime.fromisoformat(value) if value else None


class RateSource(Protocol):
    def latest(self) -> RateSnapshot: ...


class CurrencyApiClient:
    """currencyapi.com, free plan: 300 requests a month, 10 a minute."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        currencies: Sequence[str] = SUPPORTED_CURRENCIES,
        timeout: float = 10.0,
    ) -> None:
        self._api_key = api_key
        self.currencies = tuple(currencies)
        self.timeout = timeout

    def latest(self) -> RateSnapshot:
        key = (self._api_key or os.getenv("CURRENCY_API_KEY", "")).strip()
        if not key:
            raise RateSourceError("CURRENCY_API_KEY is not set")
        query = urllib.parse.urlencode({"apikey": key, "currencies": ",".join(self.currencies)})
        try:
            with urllib.request.urlopen(f"{API_URL}?{query}", timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:  # the message never contains the URL (or the key)
            raise RateSourceError(f"currencyapi answered HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RateSourceError(f"currencyapi is not reachable: {getattr(exc, 'reason', exc)}") from None
        except ValueError:
            raise RateSourceError("currencyapi sent something that is not JSON") from None
        return self.parse(payload)

    def parse(self, payload: dict[str, Any]) -> RateSnapshot:
        try:
            data = payload["data"]
            rates = {c: float(data[c]["value"]) for c in self.currencies if c in data}
            rates.setdefault("USD", 1.0)
            day = datetime.fromisoformat(
                str(payload["meta"]["last_updated_at"]).replace("Z", "+00:00")
            ).date()
        except (KeyError, TypeError, ValueError) as exc:
            raise RateSourceError(f"Unexpected currencyapi answer: {exc}") from None
        if BASE_CURRENCY not in rates or any(v <= 0 for v in rates.values()):
            raise RateSourceError("currencyapi answer has no usable AMD rate")
        return RateSnapshot(day, rates)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


class ExchangeRateService:
    """Keeps the saved snapshots up to date without wasting API requests."""

    def __init__(
        self,
        store: Optional[RateStore] = None,
        source: Optional[RateSource] = None,
        refresh_days: Optional[float] = None,
        change_pct: Optional[float] = None,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._store = store or RateStore()
        self._source = source or CurrencyApiClient()
        self.refresh_days = (
            refresh_days if refresh_days is not None
            else _env_float("CURRENCY_REFRESH_DAYS", DEFAULT_REFRESH_DAYS)
        )
        self.change_pct = (
            change_pct if change_pct is not None
            else _env_float("CURRENCY_CHANGE_PCT", DEFAULT_CHANGE_PCT)
        )
        self._clock = clock

    def current(self, required: Sequence[str] = ()) -> tuple[list[RateSnapshot], bool]:
        """Saved snapshots after a refresh if one is due.

        ``required``: currencies the caller needs. If the saved rates miss one
        of them (for example AED, added after the file was first written), a
        refresh is due even inside the normal 3-day wait.

        The flag is True when a refresh was due but failed, so the caller can
        say the rates may be old.
        """
        file = self._store.load()
        now = self._clock()
        if not self._due(file, now, required):
            return file.snapshots, False

        file.last_attempt = now
        try:
            latest = self._source.latest()
        except RateSourceError as exc:
            logger.warning("Exchange rates not refreshed: %s", exc)
            file.last_failed = now
            self._store.save(file)
            return file.snapshots, True

        file.last_checked = now
        file.last_failed = None
        self._add(file, latest)
        self._store.save(file)
        return file.snapshots, False

    def _due(self, file: RateFile, now: datetime, required: Sequence[str] = ()) -> bool:
        have = set(file.snapshots[-1].rates) if file.snapshots else set()
        missing = any(c not in have for c in (*required, BASE_CURRENCY))
        wait = RETRY_MISSING_AFTER if missing else RETRY_AFTER
        if file.last_failed is not None and now - file.last_failed < wait:
            return False
        if missing:
            return True
        return file.last_checked is None or now - file.last_checked >= timedelta(days=self.refresh_days)

    def _add(self, file: RateFile, latest: RateSnapshot) -> None:
        if not file.snapshots:
            file.snapshots.append(latest)
            return
        last = file.snapshots[-1]
        new_currencies = set(latest.rates) - set(last.rates)
        if latest.day < last.day or (
            not new_currencies and latest.moved_pct(last) < self.change_pct
        ):
            return  # nothing moved enough: only last_checked changes
        if latest.day == last.day:
            file.snapshots[-1] = latest
        else:
            file.snapshots.append(latest)


# Conversion
@dataclass(frozen=True)
class ConversionInfo:
    """What happened to one statement; the UI turns it into a message."""

    currency: str = BASE_CURRENCY
    rate_dates: tuple[date, ...] = ()
    approximate_rows: int = 0
    stale: bool = False

    @property
    def converted(self) -> bool:
        return self.currency != BASE_CURRENCY


class CurrencyConverter:
    """Converts ``amount`` to AMD and keeps ``original_amount`` / ``original_currency``."""

    def __init__(self, rates: Optional[ExchangeRateService] = None) -> None:
        self._rates = rates or ExchangeRateService()

    def convert(self, df: pd.DataFrame) -> tuple[pd.DataFrame, ConversionInfo]:
        out = df.copy()
        if "original_amount" not in out.columns:
            out["original_amount"] = out["amount"]
        if "original_currency" not in out.columns:
            out["original_currency"] = BASE_CURRENCY

        foreign = out["original_currency"] != BASE_CURRENCY
        if not foreign.any():
            return out, ConversionInfo()

        currencies = sorted(set(out.loc[foreign, "original_currency"]))
        snapshots, stale = self._rates.current(required=currencies)
        for code in currencies:
            if not any(s.has(code) for s in snapshots):
                raise RatesUnavailableError(code)

        by_code = {c: [s for s in snapshots if s.has(c)] for c in currencies}
        used: set[date] = set()
        approximate = 0
        amounts = out["amount"].astype(float).copy()
        for idx in out.index[foreign]:
            code = out.at[idx, "original_currency"]
            snap, approx = self._pick(by_code[code], out.at[idx, "date"])
            approximate += approx
            amounts.at[idx] = round(snap.to_amd(float(out.at[idx, "original_amount"]), code), 2)
            used.add(snap.day)
        out["amount"] = amounts
        info = ConversionInfo(
            currency=currencies[0] if len(currencies) == 1 else ", ".join(currencies),
            rate_dates=tuple(sorted(used)),
            approximate_rows=approximate,
            stale=stale,
        )
        return out, info

    def to_currency(self, df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, ConversionInfo]:
        """Copy of ``df`` with ``amount`` shown in ``target`` instead of AMD.

        Used for the explorer table and its CSV / Excel export; stored data is
        never changed. Each row uses the snapshot of its own date, like the
        import does. Rows that were already in ``target`` get their original
        amount back exactly, with no rounding trip through AMD.
        """
        out = df.copy()
        if target == BASE_CURRENCY or out.empty:
            return out, ConversionInfo(target)
        snapshots, stale = self._rates.current(required=(target,))
        eligible = [s for s in snapshots if s.has(target)]
        if not eligible:
            raise RatesUnavailableError(target)

        has_original = {"original_amount", "original_currency"} <= set(out.columns)
        used: set[date] = set()
        approximate = 0
        amounts = []
        for idx in out.index:
            if has_original and out.at[idx, "original_currency"] == target:
                amounts.append(round(float(out.at[idx, "original_amount"]), 2))
                continue
            snap, approx = self._pick(eligible, out.at[idx, "date"])
            approximate += approx
            used.add(snap.day)
            amounts.append(round(float(out.at[idx, "amount"]) / snap.amd_price(target), 2))
        out["amount"] = pd.Series(amounts, index=out.index, dtype=float)
        return out, ConversionInfo(target, tuple(sorted(used)), approximate, stale)

    @staticmethod
    def _pick(eligible: list[RateSnapshot], when: Any) -> tuple[RateSnapshot, bool]:
        """Newest snapshot on or before ``when``; the earliest one (approximate) if none."""
        pos = bisect_right([s.day for s in eligible], pd.Timestamp(when).date()) - 1
        if pos < 0:
            return eligible[0], True
        return eligible[pos], False
