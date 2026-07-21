"""Transaction classification logic using Hybrid (Rule-Based + Vector Embedding) approach."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_CATEGORIES = [
    "Transport",
    "Supermarket",
    "Restaurants",
    "Utilities",
    "Healthcare",
    "Entertainment",
    "Shopping",
    "Transfer",
    "Salary",
    "ATM",
    "Fee",
    "Other",
]

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
        self._category_vectors = None
        self._category_names: list[str] = []
        self._build_vector_index()

    def _load_rules(self) -> dict[str, list[str]]:
        if not self.rules_path.exists():
            return {}
        with open(self.rules_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return {cat: [str(kw).lower() for kw in kws] for cat, kws in data.items()}

    def _build_vector_index(self) -> None:
        """Builds a TF-IDF vector index based on categories."""
        corpus = []
        self._category_names = []

        for cat, keywords in self.rules.items():
            if keywords:
                cat_text = " ".join(keywords)
                corpus.append(cat_text)
                self._category_names.append(cat)

        if corpus:
            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            self._category_vectors = self._vectorizer.fit_transform(corpus)

    def get_categories(self) -> list[str]:
        """Returns a list of all available categories for the Streamlit UI."""
        categories = set(self.rules.keys())
        categories.update(DEFAULT_CATEGORIES)
        return sorted(list(categories))

    def _predict_vector_category_details(
        self, text: str, threshold: float = 0.15
    ) -> tuple[str, float]:
        """Calculates the Cosine Similarity of the text and returns (category, score):"""
        if not self._vectorizer or self._category_vectors is None or not text.strip():
            return "Other", 0.0

        text_vec = self._vectorizer.transform([text])
        similarities = cosine_similarity(text_vec, self._category_vectors)[0]

        max_idx = similarities.argmax()
        max_score = float(similarities[max_idx])

        if max_score >= threshold:
            return self._category_names[max_idx], max_score
        return "Other", max_score

    def classify_description_details(
        self, cleaned_desc: str
    ) -> tuple[str, float, str]:
        """Returns the (category, confidence, method) triple for a single description."""
        if not cleaned_desc or not str(cleaned_desc).strip():
            return "Other", 0.0, "None"

        lowered = str(cleaned_desc).lower()

        # Phase 1: Exact Keyword Match
        for category, keywords in self.rules.items():
            for kw in keywords:
                if kw in lowered:
                    return category, 1.0, "Rule-based"

        # Phase 2: Vector Embedding Match
        cat, score = self._predict_vector_category_details(lowered)
        if cat != "Other":
            return cat, round(score, 2), "Vector"

        return "Other", 0.0, "Vector"

    def classify_description(self, cleaned_desc: str) -> str:
        """Returns only the category name."""
        cat, _, _ = self.classify_description_details(cleaned_desc)
        return cat

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds the 'category', 'confidence', and 'method' columns to the DataFrame."""
        out = df.copy()
        if "cleaned_description" in out.columns:
            source_col = "cleaned_description"
        elif "description" in out.columns:
            source_col = "description"
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

    def classify(
        self, target: Union[str, pd.DataFrame]
    ) -> Union[str, pd.DataFrame]:
        """Universal classify method for string or DataFrame."""
        if isinstance(target, pd.DataFrame):
            return self.classify_dataframe(target)
        return self.classify_description(str(target))