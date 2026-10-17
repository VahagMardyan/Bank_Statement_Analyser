"""Transaction classification using merchant rules + keyword rules + TF-IDF fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

from .classification_handlers import RuleIndex, build_classification_chain
from .constants import INCOME_CATEGORIES
from .ml_model import LearnedCategoryModel
from .models import ClassificationMethod, ClassificationResult, TransactionType
from .text_utils import clean_text

# Rows labelled by these methods are trusted enough to train on.
TRUSTED_METHODS = frozenset(
    {ClassificationMethod.MERCHANT_RULE.value, ClassificationMethod.KEYWORD_RULE.value}
)


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
        3. Learned model (TF-IDF + logistic regression, trained on the rules
           plus your own rule-labelled rows and manual overrides)
        4. Fuzzy TF-IDF match on short description windows
        5. Other
    """

    def __init__(
        self,
        rules_path: Optional[str | Path] = None,
        model: Optional[LearnedCategoryModel] = None,
        use_model: bool = True,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.rules_path = (
            Path(rules_path)
            if rules_path
            else base_dir / "config" / "category_rules.json"
        )

        self.rules: dict[str, dict[str, Any]] = self._load_rules()
        self._index = RuleIndex(self.rules)
        self.model: Optional[LearnedCategoryModel] = None
        if use_model:
            self.model = model or LearnedCategoryModel()
            self.model.set_seed_from_rules(self.rules)
            self.model.fit()
        self._chain = build_classification_chain(self._index, self.model)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize Unicode, drop dates/times/long IDs and punctuation.

        Dates and IDs must be stripped here too: manual overrides are saved
        from ``cleaned_description`` (which has them removed), so if they
        stayed in the text the saved keyword would never match again.
        """
        return clean_text(text)

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

    def classify_description_result(
        self, cleaned_desc: str, exclude: frozenset[str] = frozenset()
    ) -> ClassificationResult:
        if not cleaned_desc or not str(cleaned_desc).strip():
            return ClassificationResult("Other", 0.0, "None")

        text = str(cleaned_desc)
        cleaned = self._clean_text(text)
        if not cleaned:
            return ClassificationResult("Other", 0.0, "None")

        # The loader appends merchant/payee/place after " | ". The bank's own
        # description is more specific about WHAT the payment is ("Commission",
        # "Ուսման վարձ", "Կրթաթոշակ"); the payee only says WHO. So the rules are
        # tried on the description alone first, and the payee is used only when
        # the description decides nothing (e.g. ACBA's bare "Վճարում").
        if " | " in text:
            primary = self._clean_text(text.split(" | ", 1)[0])
            if primary:
                first = self._index.find_merchant_match(primary)
                method = ClassificationMethod.MERCHANT_RULE.value
                if first is None:
                    first = self._index.find_keyword_match(primary)
                    method = ClassificationMethod.KEYWORD_RULE.value
                if first is not None and first[0] not in exclude:
                    confidence = 1.0 if method == ClassificationMethod.MERCHANT_RULE.value else 0.95
                    return ClassificationResult(first[0], confidence, method)

        return self._chain.handle(cleaned, exclude)

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

        texts = out[source_col].fillna("").astype(str)
        if "transaction_type" in out.columns:
            # An expense can never be Income/Salary.
            is_expense = out["transaction_type"] == TransactionType.EXPENSE.value
        else:
            is_expense = pd.Series(False, index=out.index)
        details = [
            self.classify_description_result(
                text, INCOME_CATEGORIES if expense else frozenset()
            ).as_tuple()
            for text, expense in zip(texts, is_expense)
        ]

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

        if self.model is not None:
            self.model.add_examples(
                [(cleaned_description, new_category)], source="manual", persist=persist
            )
            self.model.set_seed_from_rules(self.rules)
            self.model.fit()

        if persist:
            self._save_rules()

    def learn_from_dataframe(self, classified: pd.DataFrame, persist: bool = True) -> int:
        """Self-training: store rows the rules labelled with certainty, retrain.

        Returns the number of new training examples added.
        """
        if self.model is None or classified.empty or "method" not in classified:
            return 0
        source_col = "description" if "description" in classified else "cleaned_description"
        trusted = classified[classified["method"].isin(TRUSTED_METHODS)]
        added = self.model.add_examples(
            zip(trusted[source_col].astype(str), trusted["category"].astype(str)),
            source="rule",
            persist=persist,
        )
        if added:
            self.model.fit()
        return added

    def reclassify_uncertain(self, classified: pd.DataFrame) -> pd.DataFrame:
        """Re-run the chain only on rows that were not decided by a rule/manual."""
        out = classified.copy()
        keep = out["method"].isin(TRUSTED_METHODS | {ClassificationMethod.MANUAL.value})
        if keep.all():
            return out
        redo = self.classify_dataframe(out.loc[~keep, out.columns.difference(
            ["category", "confidence", "method"])])
        out.loc[~keep, ["category", "confidence", "method"]] = redo[
            ["category", "confidence", "method"]
        ].to_numpy()
        return out

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
