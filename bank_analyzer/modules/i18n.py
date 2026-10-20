"""Multi-language support for the UI.

Every language is ONE JSON file in ``locales/`` (e.g. ``hy.json``, ``en.json``):

    {
      "meta": {"name": "Հայերեն", "flag": "🇦🇲", "currency": "դրամ"},
      "app": {"title": "...", ...},          <- UI text, grouped by screen
      "columns": {"date": "Ամսաթիվ", ...},   <- table headers
      "categories": {"Transport": "Տրանսպորտ", ...},
      "transaction_types": {"income": "Մուտք", "expense": "Ելք"},
      "methods": {"Merchant rule": "...", ...}
    }

* Add a language: copy ``en.json`` to ``<code>.json``, translate the values.
  It appears in the language switch automatically.
* Delete a language: delete its file.
* Check for untranslated text: ``python -m modules.i18n``.

Anything missing in a language falls back to ``FALLBACK_LANGUAGE`` and then to
the key itself, so a half-translated file never crashes the app.

Patterns: Repository (``LocaleRepository`` hides the files) and a Translator
object injected into the UI instead of hard-coded strings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_LOCALES_DIR = Path(__file__).resolve().parent.parent / "locales"
DEFAULT_LANGUAGE = "hy"
FALLBACK_LANGUAGE = "en"

# Sections whose keys are DATA values (not UI text ids).
VALUE_SECTIONS = ("categories", "transaction_types", "methods", "columns")


@dataclass(frozen=True)
class Locale:
    """One language: its code (file name) and nested translation tree."""

    code: str
    data: dict[str, Any] = field(repr=False)

    @property
    def name(self) -> str:
        """Native name shown in the language switch, e.g. 'Հայերեն'."""
        return str(self.data.get("meta", {}).get("name", self.code))

    @property
    def flag(self) -> str:
        """Flag emoji for the language switch, e.g. '🇦🇲' (optional)."""
        return str(self.data.get("meta", {}).get("flag", ""))

    def get(self, key: str) -> Any:
        """Dotted lookup: ``get("sidebar.upload")``. Returns None if missing."""
        node: Any = self.data
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, str]:
        value = self.data.get(name, {})
        return value if isinstance(value, dict) else {}

    def leaf_keys(self) -> Iterator[str]:
        """Every dotted key that holds a value (for completeness checks)."""

        def walk(node: Any, prefix: str) -> Iterator[str]:
            if isinstance(node, dict):
                for k, v in node.items():
                    yield from walk(v, f"{prefix}.{k}" if prefix else k)
            else:
                yield prefix

        yield from walk(self.data, "")


class LocaleRepository:
    """Loads every ``*.json`` file in the locales folder."""

    def __init__(self, directory: Optional[str | Path] = None) -> None:
        self.directory = Path(directory) if directory else DEFAULT_LOCALES_DIR

    def load_all(self) -> dict[str, Locale]:
        locales: dict[str, Locale] = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError) as exc:
                logger.warning("Skipping broken locale file %s: %s", path.name, exc)
                continue
            if isinstance(data, dict):
                locales[path.stem] = Locale(path.stem, data)
        return locales

    def reverse_lookup(self, section: str) -> dict[str, str]:
        """``{translated label (lower-case): key}`` across ALL languages.

        Used to read back files exported in any language.
        """
        reverse: dict[str, str] = {}
        for locale in self.load_all().values():
            for key, label in locale.section(section).items():
                reverse[str(label).strip().lower()] = key
                reverse[str(key).strip().lower()] = key
        return reverse


class Translator:
    """Looks up UI text and data labels in the current language."""

    def __init__(
        self,
        locales: dict[str, Locale],
        language: str = DEFAULT_LANGUAGE,
        fallback: str = FALLBACK_LANGUAGE,
    ) -> None:
        if not locales:
            raise ValueError("No locale files found - add at least locales/en.json.")
        self._locales = locales
        self._fallback = locales.get(fallback)
        self._current = self._fallback or next(iter(locales.values()))
        self.language = language

    @classmethod
    def from_directory(
        cls, directory: Optional[str | Path] = None, language: str = DEFAULT_LANGUAGE
    ) -> "Translator":
        return cls(LocaleRepository(directory).load_all(), language)

    # language
    @property
    def language(self) -> str:
        return self._current.code

    @language.setter
    def language(self, code: str) -> None:
        if code in self._locales:
            self._current = self._locales[code]

    def languages(self) -> dict[str, str]:
        """``{code: "🇦🇲 Հայերեն"}`` for the language switch (flag is optional)."""
        return {code: f"{loc.flag} {loc.name}".strip() for code, loc in self._locales.items()}

    # lookup
    def _lookup(self, key: str) -> Any:
        value = self._current.get(key)
        if value is None and self._fallback is not None:
            value = self._fallback.get(key)
        return value

    def t(self, key: str, **params: Any) -> str:
        """UI text for ``key`` with ``{placeholders}`` filled in.

        ``{currency}`` is always available from ``meta.currency``.
        """
        value = self._lookup(key)
        if value is None:
            return key
        text = str(value)
        params.setdefault("currency", self._lookup("meta.currency") or "")
        try:
            return text.format(**params)
        except (KeyError, IndexError, ValueError):
            return text

    def t_list(self, key: str) -> list[str]:
        value = self._lookup(key)
        return [str(v) for v in value] if isinstance(value, list) else []

    def _value_label(self, section: str, value: Any) -> str:
        text = str(value)
        label = self._current.section(section).get(text)
        if label is None and self._fallback is not None:
            label = self._fallback.section(section).get(text)
        return str(label) if label is not None else text

    def category(self, key: str) -> str:
        return self._value_label("categories", key)

    def transaction_type(self, value: str) -> str:
        return self._value_label("transaction_types", value)

    def method(self, value: str) -> str:
        return self._value_label("methods", value)

    def column(self, name: str) -> str:
        return self._value_label("columns", name)

    def yes_no(self, value: Any) -> str:
        return self.t("common.yes") if bool(value) else self.t("common.no")

    # DataFrames
    def localize(self, df: pd.DataFrame, headers: bool = True) -> pd.DataFrame:
        """Display copy: translated categories/types/methods, yes/no for
        booleans, and (optionally) translated column headers."""
        out = df.copy()
        mappers = {
            "category": self.category,
            "transaction_type": self.transaction_type,
            "method": self.method,
        }
        for col, fn in mappers.items():
            if col in out.columns:
                out[col] = out[col].map(lambda v, fn=fn: fn(v) if pd.notna(v) else v)
        for col in out.columns:
            if pd.api.types.is_bool_dtype(out[col]):
                out[col] = out[col].map(self.yes_no)
        if headers:
            out = out.rename(columns={c: self.column(c) for c in out.columns})
        return out

# Completeness check: python -m modules.i18n
def missing_keys(locale: Locale, reference: Locale) -> list[str]:
    have = set(locale.leaf_keys())
    return [k for k in reference.leaf_keys() if k not in have]


def main() -> None:
    locales = LocaleRepository().load_all()
    reference = locales.get(FALLBACK_LANGUAGE)
    if reference is None:
        raise SystemExit(f"Reference locale {FALLBACK_LANGUAGE}.json not found.")
    for code, locale in locales.items():
        if code == reference.code:
            continue
        missing = missing_keys(locale, reference)
        extra = missing_keys(reference, locale)
        print(f"{code} ({locale.name}): {len(missing)} missing, {len(extra)} unknown")
        for k in missing:
            print(f"  missing: {k}")
        for k in extra:
            print(f"  unknown: {k}")


if __name__ == "__main__":
    main()
