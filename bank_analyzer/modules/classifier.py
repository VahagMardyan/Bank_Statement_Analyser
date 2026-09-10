"""Transaction classification using merchant rules + keyword rules + TF-IDF fallback."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .classification_handlers import RuleIndex, build_classification_chain
from .models import ClassificationResult


class TransactionClassifier:
    """Classify bank transactions with a hybrid rule-based + TF-IDF approach.

    Rules file schema:

    {
        "Transport": {
            "keywords": ["taxi", "տաքսի"],
            "merchants": [
                {
                    "name": "Yandex Go",
                    "aliases": ["yandex go", "yandexgo", "յանդեքս"]
                }
            ]
        }
    }

    Matching order (Chain of Responsibility):
        1. Merchant alias match (highest priority)
        2. Keyword match
        3. TF-IDF similarity fallback
        4. Other
    """

    def __init__(self, rules_path: Optional[str | Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.rules_path = (
            Path(rules_path)
            if rules_path
            else base_dir / "config" / "category_rules.json"
        )

        self.rules: dict[str, dict[str, Any]] = self._load_rules()
        self._index = RuleIndex(self.rules)
        self._chain = build_classification_chain(self._index)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize Unicode and common statement formatting noise."""
        if not isinstance(text, str) or not text.strip():
            return ""

        normalized = unicodedata.normalize("NFKC", text).lower()
        normalized = normalized.replace("’", "'").replace("`", "'")
        cleaned = re.sub(r"[^\w\s%]+", " ", normalized, flags=re.UNICODE)
        return " ".join(cleaned.split())

    @classmethod
    def _normalize_alias(cls, text: str) -> str:
        return cls._clean_text(text)

    @staticmethod
    def _empty_category_rule() -> dict[str, Any]:
        return {"keywords": [], "merchants": []}

    def _load_rules(self) -> dict[str, dict[str, Any]]:
        if not self.rules_path.exists():
            return {}

        with open(self.rules_path, encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, dict):
            raise ValueError("category_rules.json must contain a JSON object.")

        normalized_rules: dict[str, dict[str, Any]] = {}

        for category, raw_rule in data.items():
            # Backward compatibility with the old schema:
            # "Transport": ["taxi", "yandex"]
            if isinstance(raw_rule, list):
                keywords = [self._normalize_alias(str(x)) for x in raw_rule if x]
                normalized_rules[str(category)] = {
                    "keywords": [x for x in keywords if x],
                    "merchants": [],
                }
                continue

            if not isinstance(raw_rule, dict):
                normalized_rules[str(category)] = self._empty_category_rule()
                continue

            keywords_raw = raw_rule.get("keywords", [])
            merchants_raw = raw_rule.get("merchants", [])

            keywords: list[str] = []
            if isinstance(keywords_raw, list):
                for value in keywords_raw:
                    cleaned = self._normalize_alias(str(value))
                    if cleaned:
                        keywords.append(cleaned)

            merchants: list[dict[str, Any]] = []
            if isinstance(merchants_raw, list):
                for merchant in merchants_raw:
                    if isinstance(merchant, str):
                        aliases = [self._normalize_alias(merchant)]
                        merchants.append(
                            {
                                "name": merchant,
                                "aliases": [a for a in aliases if a],
                            }
                        )
                        continue

                    if not isinstance(merchant, dict):
                        continue

                    name = str(merchant.get("name", "")).strip()
                    aliases_raw = merchant.get("aliases", [])
                    aliases: list[str] = []

                    if name:
                        aliases.append(self._normalize_alias(name))

                    if isinstance(aliases_raw, list):
                        aliases.extend(
                            self._normalize_alias(str(alias))
                            for alias in aliases_raw
                            if alias
                        )

                    deduped = list(dict.fromkeys(a for a in aliases if a))
                    if deduped:
                        merchants.append(
                            {
                                "name": name or deduped[0],
                                "aliases": deduped,
                            }
                        )

            normalized_rules[str(category)] = {
                "keywords": list(dict.fromkeys(keywords)),
                "merchants": merchants,
            }

        return normalized_rules

    def get_categories(self) -> list[str]:
        return sorted(self.rules.keys())

    def classify_description_details(self, cleaned_desc: str) -> tuple[str, float, str]:
        """Return ``(category, confidence, method)`` for one description."""
        return self.classify_description_result(cleaned_desc).as_tuple()

    def classify_description_result(self, cleaned_desc: str) -> ClassificationResult:
        if not cleaned_desc or not str(cleaned_desc).strip():
            return ClassificationResult("Other", 0.0, "None")

        cleaned = self._clean_text(str(cleaned_desc))
        if not cleaned:
            return ClassificationResult("Other", 0.0, "None")

        return self._chain.handle(cleaned)

    def classify_description(self, cleaned_desc: str) -> str:
        return self.classify_description_result(cleaned_desc).category

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``category``, ``confidence`` and ``method`` columns."""
        out = df.copy()

        if "description" in out.columns:
            source_col = "description"
        elif "cleaned_description" in out.columns:
            source_col = "cleaned_description"
        else:
            out["category"] = "Other"
            out["confidence"] = 0.0
            out["method"] = "None"
            return out

        details = out[source_col].fillna("").astype(str).map(
            self.classify_description_details
        )

        out["category"] = [item[0] for item in details]
        out["confidence"] = [item[1] for item in details]
        out["method"] = [item[2] for item in details]
        return out

    def manual_override(
        self,
        transaction_id: str,
        new_category: str,
        cleaned_description: str,
        persist: bool = True,
    ) -> None:
        """Persist a manual correction as a category keyword.

        ``transaction_id`` is retained for UI/API compatibility. The current
        rules remain description-based, so the override applies to matching
        descriptions rather than to one transaction row only.
        """
        del transaction_id

        keyword = self._normalize_alias(cleaned_description)
        if not keyword:
            return

        rule = self.rules.setdefault(new_category, self._empty_category_rule())
        keywords = rule.setdefault("keywords", [])

        if keyword not in keywords:
            keywords.append(keyword)
            self._index.rebuild()

        if persist:
            self._save_rules()

    def _save_rules(self) -> None:
        """Write current rules back to the configured JSON file."""
        try:
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rules_path, "w", encoding="utf-8") as fh:
                json.dump(self.rules, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def classify(
        self,
        target: Union[str, pd.DataFrame],
    ) -> Union[str, pd.DataFrame]:
        """Classify either a single description or a DataFrame."""
        if isinstance(target, pd.DataFrame):
            return self.classify_dataframe(target)
        return self.classify_description(str(target))
