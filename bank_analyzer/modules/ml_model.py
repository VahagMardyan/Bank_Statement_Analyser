"""Trainable transaction classifier (TF-IDF + Logistic Regression).

Where the training data comes from:
  1. Seed rows built from category_rules.json (every keyword / merchant alias),
     so the model works on a cold start.
  2. Rows from the user's own statements that the merchant/keyword rules
     already labelled with certainty (weak supervision / self-training).
  3. Manual overrides from the UI - the strongest signal, weighted highest.

Sources 2 and 3 live in ``data/training_data.csv`` and grow with every
statement you load, so the model learns context the JSON doesn't contain
(e.g. merchant names that appear next to a known one, Armenian spellings,
typos, terminal names).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text_utils import ml_text

SOURCE_WEIGHTS: dict[str, float] = {"seed": 1.0, "rule": 2.0, "manual": 6.0}
TRAINING_COLUMNS = ["text", "category", "source"]


@dataclass(frozen=True)
class Prediction:
    category: str
    probability: float
    margin: float  # gap between best and second-best class


class LearnedCategoryModel:
    """Char + word TF-IDF features fed to a class-balanced logistic regression."""

    def __init__(
        self,
        store_path: Optional[str | Path] = None,
        min_probability: float = 0.25,
        min_margin: float = 0.20,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.store_path = (
            Path(store_path) if store_path else base_dir / "data" / "training_data.csv"
        )
        self.min_probability = min_probability
        self.min_margin = min_margin

        self._char_vec: Optional[TfidfVectorizer] = None
        self._word_vec: Optional[TfidfVectorizer] = None
        self._clf: Optional[LogisticRegression] = None
        self._seed_rows: list[dict[str, str]] = []
        self.user_rows = self._load_store()

    # ------------------------------------------------------------------ data
    def _load_store(self) -> pd.DataFrame:
        if self.store_path.exists():
            try:
                df = pd.read_csv(self.store_path, encoding="utf-8")
                if set(TRAINING_COLUMNS) <= set(df.columns):
                    return df[TRAINING_COLUMNS].dropna().astype(str)
            except (OSError, ValueError, pd.errors.ParserError):
                pass
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    def _save_store(self) -> None:
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            self.user_rows.to_csv(self.store_path, index=False, encoding="utf-8")
        except OSError:
            pass

    def set_seed_from_rules(self, rules: dict[str, dict[str, Any]]) -> None:
        rows: list[dict[str, str]] = []
        for category, rule in rules.items():
            for kw in rule.get("keywords", []):
                rows.append({"text": kw, "category": category, "source": "seed"})
            for merchant in rule.get("merchants", []):
                for alias in merchant.get("aliases", []):
                    rows.append({"text": alias, "category": category, "source": "seed"})
        self._seed_rows = rows

    def add_examples(
        self, examples: Iterable[tuple[str, str]], source: str, persist: bool = True
    ) -> int:
        """Add (raw_text, category) pairs. Manual labels replace older labels."""
        new = pd.DataFrame(
            [
                {"text": ml_text(t), "category": c, "source": source}
                for t, c in examples
            ],
            columns=TRAINING_COLUMNS,
        )
        new = new[new["text"].str.len() > 0]
        if new.empty:
            return 0

        combined = pd.concat([self.user_rows, new], ignore_index=True)
        # Priority: manual > rule; for the same text keep the strongest/latest.
        combined["_prio"] = combined["source"].map({"manual": 1, "rule": 0}).fillna(0)
        combined = (
            combined.reset_index()
            .sort_values(["_prio", "index"])
            .drop_duplicates(subset="text", keep="last")
            .drop(columns=["_prio", "index"])
            .reset_index(drop=True)
        )
        added = len(combined) - len(self.user_rows)
        self.user_rows = combined
        if persist:
            self._save_store()
        return added

    def training_frame(self) -> pd.DataFrame:
        seed = pd.DataFrame(self._seed_rows, columns=TRAINING_COLUMNS)
        seed["text"] = seed["text"].map(ml_text)
        frame = pd.concat([seed, self.user_rows], ignore_index=True)
        return frame[frame["text"].str.len() > 0].reset_index(drop=True)

    # ----------------------------------------------------------------- model
    def _features(self, texts: list[str], fit: bool = False) -> Any:
        assert self._char_vec is not None and self._word_vec is not None
        if fit:
            return hstack(
                [self._char_vec.fit_transform(texts), self._word_vec.fit_transform(texts)]
            ).tocsr()
        return hstack(
            [self._char_vec.transform(texts), self._word_vec.transform(texts)]
        ).tocsr()

    def fit(self, frame: Optional[pd.DataFrame] = None) -> "LearnedCategoryModel":
        frame = self.training_frame() if frame is None else frame
        if frame.empty or frame["category"].nunique() < 2:
            self._clf = None
            return self

        # char_wb n-grams: robust to typos and transliteration (starbuks ~ starbucks)
        self._char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1
        )
        # word uni/bi-grams: picks up context words from real statements
        self._word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b", min_df=1,
        )
        X = self._features(frame["text"].tolist(), fit=True)
        weights = frame["source"].map(SOURCE_WEIGHTS).fillna(1.0).to_numpy()

        self._clf = LogisticRegression(
            C=8.0, class_weight="balanced", max_iter=300, tol=1e-3
        )
        self._clf.fit(X, frame["category"].to_numpy(), sample_weight=weights)
        return self

    @property
    def is_ready(self) -> bool:
        return self._clf is not None

    @property
    def classes(self) -> list[str]:
        return list(self._clf.classes_) if self._clf is not None else []

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self._clf is not None
        return self._clf.predict_proba(self._features([ml_text(t) for t in texts]))

    def predict(self, text: str, exclude: frozenset[str] = frozenset()) -> Optional[Prediction]:
        """Return the best class, or None if the model is not confident enough."""
        if self._clf is None:
            return None
        cleaned = ml_text(text)
        if not cleaned:
            return None
        probs = self._clf.predict_proba(self._features([cleaned]))[0]
        classes = self._clf.classes_
        order = [i for i in np.argsort(probs)[::-1] if classes[i] not in exclude]
        if not order:
            return None
        best = float(probs[order[0]])
        second = float(probs[order[1]]) if len(order) > 1 else 0.0
        if best < self.min_probability or best - second < self.min_margin:
            return None
        return Prediction(str(classes[order[0]]), best, best - second)