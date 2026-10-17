"""Chain of Responsibility handlers for hybrid transaction classification."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from abc import ABC, abstractmethod
from typing import Any, Iterator, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ml_model import LearnedCategoryModel
from .models import ClassificationMethod, ClassificationResult
from .text_utils import ml_text, token_windows

# Categories a handler may not pick (e.g. Income/Salary for an expense row).
Exclude = frozenset[str]


class ClassificationHandler(ABC):
    """Base handler: try this strategy, otherwise forward to the next."""

    def __init__(self) -> None:
        self._next: Optional[ClassificationHandler] = None

    def set_next(self, handler: ClassificationHandler) -> ClassificationHandler:
        """Link the next handler and return it so callers can chain fluently."""
        self._next = handler
        return handler

    def handle(self, text: str, exclude: Exclude = frozenset()) -> ClassificationResult:
        result = self.try_classify(text, exclude)
        if result is not None:
            return result
        if self._next is not None:
            return self._next.handle(text, exclude)
        return ClassificationResult("Other", 0.0, ClassificationMethod.NONE.value)

    @abstractmethod
    def try_classify(
        self, text: str, exclude: Exclude = frozenset()
    ) -> Optional[ClassificationResult]:
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
        self._rule_values_raw: list[str] = []
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
        self._rule_values_raw = corpus

    def predict_vector(
        self,
        text: str,
        threshold: float = 0.80,
        exclude: Exclude = frozenset(),
        shortlist: int = 15,
    ) -> tuple[str, float]:
        """Fuzzy-match short spans of the description against each rule.

        1. Strip bank boilerplate and split the description into 1..3-token
           windows (whole-description cosine is swamped by boilerplate).
        2. Shortlist (window, rule) pairs by char-ngram TF-IDF cosine.
        3. Verify: window and rule must have the same number of words and
           every word pair must be similar (SequenceMatcher). This accepts
           "starbuks"~"starbucks" and "yandx go"~"yandex go" but rejects
           "go service"~"service fee".
        Returns (category, score) where score is the weakest word-pair ratio.
        """
        if self._vectorizer is None or self._rule_vectors is None:
            return "Other", 0.0
        windows = [w for w in token_windows(ml_text(text)) if len(w) >= 3]
        if not windows:
            return "Other", 0.0

        sims = cosine_similarity(self._vectorizer.transform(windows), self._rule_vectors)
        if exclude:
            mask = np.array([c in exclude for c in self._rule_categories])
            sims[:, mask] = 0.0

        flat = np.argsort(sims, axis=None)[::-1][:shortlist]
        best_cat, best_score = "Other", 0.0
        for flat_idx in flat:
            w_idx, r_idx = np.unravel_index(int(flat_idx), sims.shape)
            if sims[w_idx, r_idx] < 0.35:
                break
            score = self._word_alignment(windows[w_idx], self._rule_values_raw[r_idx])
            if score > best_score:
                best_cat, best_score = self._rule_categories[r_idx], score
        if best_score >= threshold:
            return best_cat, best_score
        return "Other", best_score

    @staticmethod
    def _word_alignment(window: str, rule: str) -> float:
        w_tokens, r_tokens = window.split(), rule.split()
        if len(w_tokens) != len(r_tokens):
            return 0.0
        ratios = []
        for w, r in zip(w_tokens, r_tokens):
            # very short words must match exactly ("go", "sas", "gg")
            if min(len(w), len(r)) <= 3:
                ratios.append(1.0 if w == r else 0.0)
            else:
                ratios.append(SequenceMatcher(None, w, r).ratio())
        return float(min(ratios))


class MerchantRuleHandler(ClassificationHandler):
    """Highest-priority: named merchant aliases."""

    def __init__(self, index: RuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(
        self, text: str, exclude: Exclude = frozenset()
    ) -> Optional[ClassificationResult]:
        match = self._index.find_merchant_match(text)
        if match is None or match[0] in exclude:
            return None
        return ClassificationResult(match[0], 1.0, ClassificationMethod.MERCHANT_RULE.value)


class KeywordRuleHandler(ClassificationHandler):
    """Generic category keywords (after merchants)."""

    def __init__(self, index: RuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(
        self, text: str, exclude: Exclude = frozenset()
    ) -> Optional[ClassificationResult]:
        match = self._index.find_keyword_match(text)
        if match is None or match[0] in exclude:
            return None
        return ClassificationResult(match[0], 0.95, ClassificationMethod.KEYWORD_RULE.value)


INDIVIDUAL_ENTREPRENEUR = "Individual Entrepreneurs"

# Legal-form markers: ԱՁ (Armenian), IE/PE (Latin transliterations), ИП (Russian).
_IE_MARKERS = frozenset({"աձ", "ie", "pe", "ип"})
# Typical Armenian surname endings, Latin and Armenian script.
_SURNAME_RE = re.compile(r"(yan|ian|yants|ants|unts|uni|յան|յանց|ունի|ян)$")
# Words that mean the counterparty is a business, not a person's name.
_BUSINESS_WORDS = frozenset({
    "market", "supermarket", "mart", "shop", "store", "pharm", "pharmacy",
    "house", "city", "club", "center", "centre", "clothes", "cafe", "bar",
    "fitness", "salon", "studio", "service", "group", "trade", "plus",
})
# Card-payment prefixes after cleaning: ACBA "Վճարում | ...", Ameriabank "Ք: ...".
_CARD_PREFIXES = ("վճարում ", "ք ")
_TAIL_NOISE = frozenset({
    "yerevan", "yerev", "yereva", "am", "arm", "armenia", "ամերիաբանկ", "փբը",
    "գլխամասային", "գրասենյակ", "երևան",
})


def looks_like_individual_entrepreneur(text: str) -> bool:
    """True if a card payment goes to a person-named business (ԱՁ / IE / PE).

    Examples: "Վճարում | K. A. MELKONYAN", "Ք: ARTAK BEGYAN IE Yerevan AM".
    Personal transfers ("Անձնական փոխանցում | ...") never reach this check,
    because the Transfer keyword rule fires first.
    """
    tokens = text.split()
    if any(tok in _IE_MARKERS for tok in tokens):
        return True
    if not text.startswith(_CARD_PREFIXES):
        return False
    name = [
        t for t in tokens[1:]
        if t not in _TAIL_NOISE and not t.isdigit()
    ]
    if not 2 <= len(name) <= 4:
        return False
    if any(t in _BUSINESS_WORDS or not t.isalpha() for t in name):
        return False
    return any(len(t) > 4 and _SURNAME_RE.search(t) for t in name)


class IndividualEntrepreneurHandler(ClassificationHandler):
    """Card payments to businesses registered under the owner's name.

    In Armenia many small shops, salons and kiosks are individual
    entrepreneurs, so the statement shows only a name. The real category is
    unknown, so they get their own bucket instead of a wrong guess.
    """

    def try_classify(
        self, text: str, exclude: Exclude = frozenset()
    ) -> Optional[ClassificationResult]:
        if INDIVIDUAL_ENTREPRENEUR in exclude:
            return None
        if looks_like_individual_entrepreneur(text):
            return ClassificationResult(
                INDIVIDUAL_ENTREPRENEUR, 0.7, ClassificationMethod.NAME_HEURISTIC.value
            )
        return None


class LearnedModelHandler(ClassificationHandler):
    """Supervised model trained on rules + the user's own labelled history."""

    def __init__(self, model: LearnedCategoryModel) -> None:
        super().__init__()
        self._model = model

    def try_classify(
        self, text: str, exclude: Exclude = frozenset()
    ) -> Optional[ClassificationResult]:
        pred = self._model.predict(text, exclude=exclude)
        if pred is None:
            return None
        return ClassificationResult(
            pred.category, round(pred.probability, 2), ClassificationMethod.ML_MODEL.value
        )


class VectorSimilarityHandler(ClassificationHandler):
    """Fuzzy char-ngram match on description windows; always returns a result."""

    def __init__(self, index: RuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(
        self, text: str, exclude: Exclude = frozenset()
    ) -> Optional[ClassificationResult]:
        category, score = self._index.predict_vector(text, exclude=exclude)
        method = (
            ClassificationMethod.VECTOR.value
            if category != "Other"
            else ClassificationMethod.NONE.value
        )
        return ClassificationResult(category, round(score, 2), method)


def build_classification_chain(
    index: RuleIndex, model: Optional[LearnedCategoryModel] = None
) -> ClassificationHandler:
    """Assemble merchant -> keyword -> entrepreneur name -> learned model ->
    fuzzy vector -> Other."""
    head = MerchantRuleHandler(index)
    tail = head.set_next(KeywordRuleHandler(index))
    tail = tail.set_next(IndividualEntrepreneurHandler())
    if model is not None:
        tail = tail.set_next(LearnedModelHandler(model))
    tail.set_next(VectorSimilarityHandler(index))
    return head
