"""Trainable transaction classifier (TF-IDF + Logistic Regression).

Where the training data comes from:
  1. Seed rows built from the rules (every keyword / merchant alias), so the
     model works on a cold start. Refreshed automatically: the model is a
     :class:`~modules.rules.RuleSet` observer.
  2. Rows from the user's statements that the merchant/keyword rules already
     labelled with certainty (weak supervision / self-training).
  3. Manual overrides from the UI - the strongest signal, weighted highest.

Sources 2 and 3 are kept by a :class:`TrainingExampleStore` (CSV by default)
and grow with every statement you load.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

if TYPE_CHECKING:
    from .rules import RuleSet

logger = logging.getLogger(__name__)

TRAINING_COLUMNS = ["text", "category", "source"]
DEFAULT_STORE_PATH = Path(__file__).resolve().parent.parent / "data" / "training_data.csv"


class TrainingSource(str, Enum):
    """Where a training row came from; decides its weight and priority."""

    SEED = "seed"
    RULE = "rule"
    MANUAL = "manual"

    @property
    def weight(self) -> float:
        return {"seed": 1.0, "rule": 2.0, "manual": 6.0}[self.value]

    @property
    def priority(self) -> int:
        """Higher wins when the same text has two labels."""
        return 1 if self is TrainingSource.MANUAL else 0


SOURCE_WEIGHTS: dict[str, float] = {s.value: s.weight for s in TrainingSource}


@dataclass(frozen=True)
class Prediction:
    category: str
    probability: float
    margin: float  # gap between best and second-best class


# Repository for user training rows
class TrainingExampleStore:
    """Persists rule-labelled rows and manual overrides as CSV."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else DEFAULT_STORE_PATH

    def load(self) -> pd.DataFrame:
        if self.path.exists():
            try:
                df = pd.read_csv(self.path, encoding="utf-8")
                if set(TRAINING_COLUMNS) <= set(df.columns):
                    return df[TRAINING_COLUMNS].dropna().astype(str)
            except (OSError, ValueError, pd.errors.ParserError) as exc:
                logger.warning("Ignoring unreadable training store %s: %s", self.path, exc)
        return pd.DataFrame(columns=TRAINING_COLUMNS)

    def save(self, rows: pd.DataFrame) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rows.to_csv(self.path, index=False, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save training store %s: %s", self.path, exc)


# Model
class LearnedCategoryModel:
    """Char + word TF-IDF features fed to a class-balanced logistic regression."""

    def __init__(
        self,
        store_path: Optional[str | Path] = None,
        min_probability: float = 0.25,
        min_margin: float = 0.20,
        store: Optional[TrainingExampleStore] = None,
        normalizer: TextNormalizer = DEFAULT_NORMALIZER,
    ) -> None:
        self._store = store or TrainingExampleStore(store_path)
        self._normalizer = normalizer
        self.min_probability = min_probability
        self.min_margin = min_margin

        self._char_vec: Optional[TfidfVectorizer] = None
        self._word_vec: Optional[TfidfVectorizer] = None
        self._clf: Optional[LogisticRegression] = None
        self._seed_rows: list[dict[str, str]] = []
        self.user_rows = self._store.load()

    @property
    def store_path(self) -> Path:
        return self._store.path

    # RuleSet observer
    def on_rules_changed(self, rules: "RuleSet") -> None:
        """Rules changed -> refresh the seed rows and retrain."""
        self.set_seed_from_rules(rules)
        self.fit()

    def set_seed_from_rules(self, rules: "RuleSet") -> None:
        seed = TrainingSource.SEED.value
        self._seed_rows = [
            {"text": text, "category": category, "source": seed}
            for category, text in rules.iter_rule_texts()
        ]

    # data
    def add_examples(
        self,
        examples: Iterable[tuple[str, str]],
        source: str | TrainingSource,
        persist: bool = True,
    ) -> int:
        """Add (raw_text, category) pairs. Manual labels replace older labels.

        Returns how many NEW texts were added (relabels don't count).
        """
        source = TrainingSource(source)
        new = pd.DataFrame(
            [
                {"text": self._normalizer.ml_text(t), "category": c, "source": source.value}
                for t, c in examples
            ],
            columns=TRAINING_COLUMNS,
        )
        new = new[new["text"].str.len() > 0]
        if new.empty:
            return 0

        combined = pd.concat([self.user_rows, new], ignore_index=True)
        # For the same text keep the strongest source, then the latest row.
        combined["_prio"] = combined["source"].map(
            {s.value: s.priority for s in TrainingSource}
        ).fillna(0)
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
            self._store.save(self.user_rows)
        return added

    def training_frame(self) -> pd.DataFrame:
        seed = pd.DataFrame(self._seed_rows, columns=TRAINING_COLUMNS)
        seed["text"] = seed["text"].map(self._normalizer.ml_text)
        frame = pd.concat([seed, self.user_rows], ignore_index=True)
        return frame[frame["text"].str.len() > 0].reset_index(drop=True)

    # model
    def _features(self, texts: list[str], fit: bool = False) -> Any:
        if self._char_vec is None or self._word_vec is None:
            raise RuntimeError("Model is not fitted.")
        if fit:
            parts = [self._char_vec.fit_transform(texts), self._word_vec.fit_transform(texts)]
        else:
            parts = [self._char_vec.transform(texts), self._word_vec.transform(texts)]
        return hstack(parts).tocsr()

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
        if self._clf is None:
            raise RuntimeError("Model is not fitted.")
        return self._clf.predict_proba(
            self._features([self._normalizer.ml_text(t) for t in texts])
        )

    def predict(self, text: str, exclude: frozenset[str] = frozenset()) -> Optional[Prediction]:
        """Return the best class, or None if the model is not confident enough."""
        if self._clf is None:
            return None
        cleaned = self._normalizer.ml_text(text)
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
