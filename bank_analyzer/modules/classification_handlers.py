"""Chain of Responsibility handlers for hybrid transaction classification."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import ClassificationMethod, ClassificationResult


class ClassificationHandler(ABC):
    """Base handler: try this strategy, otherwise forward to the next."""

    def __init__(self) -> None:
        self._next: Optional[ClassificationHandler] = None

    def set_next(self, handler: ClassificationHandler) -> ClassificationHandler:
        """Link the next handler and return it so callers can chain fluently."""
        self._next = handler
        return handler

    def handle(self, text: str) -> ClassificationResult:
        result = self.try_classify(text)
        if result is not None:
            return result
        if self._next is not None:
            return self._next.handle(text)
        return ClassificationResult("Other", 0.0, ClassificationMethod.NONE.value)

    @abstractmethod
    def try_classify(self, text: str) -> Optional[ClassificationResult]:
        """Return a result if this handler can classify ``text``, else None."""


class RuleIndex:
    """Shared keyword/merchant corpus and TF-IDF index used by handlers."""

    def __init__(self, rules: dict[str, dict[str, Any]]) -> None:
        self.rules = rules
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._rule_vectors: Any = None
        self._rule_categories: list[str] = []
        self._rule_types: list[str] = []
        self._rule_values: list[str] = []
        self.rebuild()

    def iter_keyword_rules(self) -> Iterator[tuple[str, str]]:
        for category, rule in self.rules.items():
            for keyword in rule.get("keywords", []):
                if keyword:
                    yield category, keyword

    def iter_merchant_rules(self) -> Iterator[tuple[str, str, str]]:
        for category, rule in self.rules.items():
            for merchant in rule.get("merchants", []):
                merchant_name = str(merchant.get("name", ""))
                for alias in merchant.get("aliases", []):
                    if alias:
                        yield category, merchant_name, alias

    @staticmethod
    def alias_matches(alias: str, text: str) -> bool:
        """Match short aliases as tokens; longer aliases as phrases."""
        if not alias or not text:
            return False
        if len(alias.replace(" ", "")) <= 3:
            return bool(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text))
        return alias in text

    def find_merchant_match(self, text: str) -> tuple[str, str] | None:
        best_match: tuple[str, str, int] | None = None
        for category, merchant_name, alias in self.iter_merchant_rules():
            if self.alias_matches(alias, text):
                score = len(alias.replace(" ", ""))
                if best_match is None or score > best_match[2]:
                    best_match = (category, merchant_name or alias, score)
        if best_match is None:
            return None
        return best_match[0], best_match[1]

    def find_keyword_match(self, text: str) -> tuple[str, str] | None:
        best_match: tuple[str, str, int] | None = None
        for category, keyword in self.iter_keyword_rules():
            if self.alias_matches(keyword, text):
                score = len(keyword.replace(" ", ""))
                if best_match is None or score > best_match[2]:
                    best_match = (category, keyword, score)
        if best_match is None:
            return None
        return best_match[0], best_match[1]

    def rebuild(self) -> None:
        corpus: list[str] = []
        self._rule_categories = []
        self._rule_types = []
        self._rule_values = []

        for category, keyword in self.iter_keyword_rules():
            corpus.append(keyword)
            self._rule_categories.append(category)
            self._rule_types.append("keyword")
            self._rule_values.append(keyword)

        for category, merchant_name, alias in self.iter_merchant_rules():
            corpus.append(alias)
            self._rule_categories.append(category)
            self._rule_types.append("merchant")
            self._rule_values.append(merchant_name or alias)

        if not corpus:
            self._vectorizer = None
            self._rule_vectors = None
            return

        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            lowercase=False,
            min_df=1,
        )
        self._rule_vectors = self._vectorizer.fit_transform(corpus)

    def predict_vector(
        self, text: str, threshold: float = 0.45
    ) -> tuple[str, float]:
        if self._vectorizer is None or self._rule_vectors is None or not text.strip():
            return "Other", 0.0

        text_vec = self._vectorizer.transform([text])
        similarities = cosine_similarity(text_vec, self._rule_vectors)[0]
        max_idx = int(similarities.argmax())
        max_score = float(similarities[max_idx])
        if max_score >= threshold:
            return self._rule_categories[max_idx], max_score
        return "Other", max_score


class MerchantRuleHandler(ClassificationHandler):
    """Highest-priority: named merchant aliases."""

    def __init__(self, index: RuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(self, text: str) -> Optional[ClassificationResult]:
        match = self._index.find_merchant_match(text)
        if match is None:
            return None
        return ClassificationResult(match[0], 1.0, ClassificationMethod.MERCHANT_RULE.value)


class KeywordRuleHandler(ClassificationHandler):
    """Generic category keywords (after merchants)."""

    def __init__(self, index: RuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(self, text: str) -> Optional[ClassificationResult]:
        match = self._index.find_keyword_match(text)
        if match is None:
            return None
        return ClassificationResult(match[0], 0.95, ClassificationMethod.KEYWORD_RULE.value)


class VectorSimilarityHandler(ClassificationHandler):
    """TF-IDF char-ngram fallback; always produces a result (including Other)."""

    def __init__(self, index: RuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(self, text: str) -> Optional[ClassificationResult]:
        category, score = self._index.predict_vector(text)
        return ClassificationResult(
            category, round(score, 2), ClassificationMethod.VECTOR.value
        )


def build_classification_chain(index: RuleIndex) -> ClassificationHandler:
    """Assemble merchant → keyword → vector (Factory Method for the chain)."""
    head = MerchantRuleHandler(index)
    head.set_next(KeywordRuleHandler(index)).set_next(VectorSimilarityHandler(index))
    return head
