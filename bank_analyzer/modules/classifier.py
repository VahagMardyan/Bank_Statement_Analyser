"""Transaction classification logic using Hybrid (Rule-Based + Vector Embedding) approach."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_categories_from_json(file_path = "../config/category_rules.json") -> list:
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return list(data.keys())

DEFAULT_CATEGORIES = get_categories_from_json()

class TransactionClassifier:
    """Classifies transactions using exact keyword matching + Vector Similarity fallback."""

    def __init__(self, rules_path: Optional[str | Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.rules_path = (
            Path(rules_path)
            if rules_path
            else base_dir / "config" / "category_rules.json"
        )
        self.rules: dict[str, list[str]] = self._load_rules()

        self._vectorizer: Optional[TfidfVectorizer] = None
        self._keyword_vectors = None
        self._keyword_categories: list[str] = []
        self._build_vector_index()

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalizes Unicode characters and strips common Armenian bank statement noise."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Unicode Normalization (converts special Armenian dots/symbols to standard ASCII)
        normalized = unicodedata.normalize("NFKC", text).lower()

        # 2. Strip common bank noise (Ameriabank, etc.)
        cleaned = re.sub(
            r"ամերիաբանկ\s*փբը.*", "", normalized, flags=re.IGNORECASE
        )
        cleaned = re.sub(
            r"գլխամասային\s*գրասենյակ.*", "", cleaned, flags=re.IGNORECASE
        )
        cleaned = re.sub(r"^\s*ք:\s*", "", cleaned, flags=re.IGNORECASE)

        # 3. Replace punctuation with spaces for cleaner keyword matching
        cleaned = re.sub(r"[^\w\s%]", " ", cleaned)

        return " ".join(cleaned.split())

    def _load_rules(self) -> dict[str, list[str]]:
        if not self.rules_path.exists():
            return {}
        with open(self.rules_path, encoding="utf-8") as fh:
            data = json.load(fh)
        
        # Apply clean_text to all keywords in JSON
        return {
            cat: [self._clean_text(str(kw)) for kw in kws if kw]
            for cat, kws in data.items()
        }

    def _build_vector_index(self) -> None:
        """Builds a TF-IDF vector index with one row per keyword.

        Each keyword gets its own vector (tagged with its owning category),
        rather than blending every keyword in a category into a single
        combined string. Blending dilutes the signal: a transaction that
        closely matches one keyword (e.g. "dental clinic") would otherwise
        get averaged against every unrelated keyword in that category
        (e.g. "pharmacy", "hospital", "doctor"...), often pulling the
        similarity score below threshold. Matching against individual
        keywords keeps a strong match strong.
        """
        corpus = []
        self._keyword_categories = []

        for cat, keywords in self.rules.items():
            for kw in keywords:
                if kw.strip():
                    corpus.append(kw)
                    self._keyword_categories.append(cat)

        if corpus:
            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4)
            )
            self._keyword_vectors = self._vectorizer.fit_transform(corpus)

    def get_categories(self) -> list[str]:
        """Returns a list of all available categories for the Streamlit UI."""
        categories = set(self.rules.keys())
        categories.update(DEFAULT_CATEGORIES)
        return sorted(list(categories))

    def _predict_vector_category_details(
        self, text: str, threshold: float = 0.45
    ) -> tuple[str, float]:
        """Calculates Cosine Similarity against the closest single keyword
        and returns (category, score) if score >= threshold."""
        if (
            not self._vectorizer
            or self._keyword_vectors is None
            or not text.strip()
        ):
            return "Other", 0.0

        text_vec = self._vectorizer.transform([text])
        similarities = cosine_similarity(text_vec, self._keyword_vectors)[0]

        max_idx = similarities.argmax()
        max_score = float(similarities[max_idx])

        if max_score >= threshold:
            return self._keyword_categories[max_idx], max_score
        return "Other", max_score

    def classify_description_details(
        self, cleaned_desc: str
    ) -> tuple[str, float, str]:
        """Returns the (category, confidence, method) triple for a single description."""
        if not cleaned_desc or not str(cleaned_desc).strip():
            return "Other", 0.0, "None"

        cleaned = self._clean_text(cleaned_desc)

        best_match: tuple[str, str] | None = None  # (category, keyword)
        for category, keywords in self.rules.items():
            for kw in keywords:
                if kw and kw in cleaned:
                    if best_match is None or len(kw) > len(best_match[1]):
                        best_match = (category, kw)
        if best_match is not None:
            return best_match[0], 1.0, "Rule-based"

        # Phase 2: Vector Embedding Match (Threshold: 0.35)
        cat, score = self._predict_vector_category_details(cleaned)
        if cat != "Other":
            return cat, round(score, 2), "Vector"

        return "Other", 0.0, "Vector"

    def classify_description(self, cleaned_desc: str) -> str:
        """Returns only the category name."""
        cat, _, _ = self.classify_description_details(cleaned_desc)
        return cat

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'category', 'confidence', and 'method' columns to the DataFrame."""
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

        details = out[source_col].astype(str).map(self.classify_description_details)

        out["category"] = [d[0] for d in details]
        out["confidence"] = [d[1] for d in details]
        out["method"] = [d[2] for d in details]

        return out

    def manual_override(
        self,
        transaction_id: str,
        new_category: str,
        cleaned_description: str,
        persist: bool = True,
    ) -> None:
        """Records a manual category correction.

        Adds `cleaned_description` as a new rule-based keyword under
        `new_category`, so this transaction (and any future transaction with
        a matching description) is classified consistently going forward.
        Rebuilds the vector index and, by default, persists the updated
        rules back to `self.rules_path` so the correction survives restarts.

        `transaction_id` isn't used to alter the rules themselves (rules are
        keyed by description text, not by row), but is accepted to match the
        call signature already used by the UI, and to allow future per-row
        bookkeeping (e.g. an override log) if needed.
        """
        keyword = self._clean_text(cleaned_description)
        if not keyword:
            return

        self.rules.setdefault(new_category, [])
        if keyword not in self.rules[new_category]:
            self.rules[new_category].append(keyword)
            self._build_vector_index()

        if persist:
            self._save_rules()

    def _save_rules(self) -> None:
        """Writes the current in-memory rules back to self.rules_path."""
        try:
            self.rules_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.rules_path, "w", encoding="utf-8") as fh:
                json.dump(self.rules, fh, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def classify(
        self, target: Union[str, pd.DataFrame]
    ) -> Union[str, pd.DataFrame]:
        """Universal classify method for string or DataFrame."""
        if isinstance(target, pd.DataFrame):
            return self.classify_dataframe(target)
        return self.classify_description(str(target))
