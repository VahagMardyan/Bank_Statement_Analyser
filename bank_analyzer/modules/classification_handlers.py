"""Chain of Responsibility for hybrid transaction classification.

Order (assembled by :class:`ClassificationChainBuilder`):
    1. PrimaryDescriptionHandler  - rules on the bank's own description only
    2. MerchantRuleHandler        - merchant aliases on the full text
    3. KeywordRuleHandler         - category keywords on the full text
    4. IndividualEntrepreneurHandler - person-named card payees (ԱՁ / IE)
    5. LearnedModelHandler        - TF-IDF + logistic regression
    6. VectorSimilarityHandler    - fuzzy char-ngram match (terminal)
    -> "Other"

Each handler receives a :class:`ClassificationRequest`, so cleaning and
splitting the text happen once, not in every handler.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ml_model import LearnedCategoryModel
from .models import (
    OTHER_CATEGORY,
    ClassificationMethod,
    ClassificationRequest,
    ClassificationResult,
)
from .rules import RuleSet
from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

MERCHANT_CONFIDENCE = 1.0
KEYWORD_CONFIDENCE = 0.95
NAME_HEURISTIC_CONFIDENCE = 0.7


# Rule indexes (both are RuleSet observers)
class RuleMatcher:
    """Exact longest-match lookup of merchant aliases and keywords."""

    def __init__(self) -> None:
        self._merchants: list[tuple[str, str, str]] = []
        self._keywords: list[tuple[str, str, str]] = []

    def on_rules_changed(self, rules: RuleSet) -> None:
        # Stored as (category, label, alias) so one lookup routine serves both.
        self._merchants = [
            (c, name or alias, alias) for c, name, alias in rules.iter_merchant_rules()
        ]
        self._keywords = [(c, kw, kw) for c, kw in rules.iter_keyword_rules()]

    @staticmethod
    def alias_matches(alias: str, text: str) -> bool:
        """Short aliases (<=3 chars) must match as whole tokens; longer as phrases."""
        if not alias or not text:
            return False
        if len(alias.replace(" ", "")) <= 3:
            return bool(re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", text))
        return alias in text

    @classmethod
    def _longest(
        cls, candidates: list[tuple[str, str, str]], text: str
    ) -> Optional[tuple[str, str]]:
        """``candidates`` = (category, label, alias). Longest matching alias wins."""
        best: Optional[tuple[str, str, int]] = None
        for category, label, alias in candidates:
            if cls.alias_matches(alias, text):
                score = len(alias.replace(" ", ""))
                if best is None or score > best[2]:
                    best = (category, label, score)
        return None if best is None else (best[0], best[1])

    def find_merchant_match(self, text: str) -> Optional[tuple[str, str]]:
        """Return ``(category, merchant_name)`` or None."""
        return self._longest(self._merchants, text)

    def find_keyword_match(self, text: str) -> Optional[tuple[str, str]]:
        """Return ``(category, keyword)`` or None."""
        return self._longest(self._keywords, text)


class FuzzyRuleIndex:
    """Char-ngram TF-IDF over every keyword/alias, verified word by word."""

    SHORTLIST_MIN_COSINE = 0.35

    def __init__(self, normalizer: TextNormalizer = DEFAULT_NORMALIZER) -> None:
        self._normalizer = normalizer
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._vectors: Any = None
        self._categories: list[str] = []
        self._corpus: list[str] = []

    def on_rules_changed(self, rules: RuleSet) -> None:
        corpus: list[str] = []
        categories: list[str] = []
        for category, keyword in rules.iter_keyword_rules():
            corpus.append(keyword)
            categories.append(category)
        for category, _name, alias in rules.iter_merchant_rules():
            corpus.append(alias)
            categories.append(category)

        self._corpus, self._categories = corpus, categories
        if not corpus:
            self._vectorizer = self._vectors = None
            return
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), lowercase=False, min_df=1
        )
        self._vectors = self._vectorizer.fit_transform(corpus)

    def predict(
        self,
        text: str,
        threshold: float = 0.80,
        exclude: frozenset[str] = frozenset(),
        shortlist: int = 15,
    ) -> tuple[str, float]:
        """Fuzzy-match short spans of the description against each rule.

        1. Strip bank boilerplate and split into 1..3-token windows
           (whole-description cosine is swamped by boilerplate).
        2. Shortlist (window, rule) pairs by char-ngram TF-IDF cosine.
        3. Verify: same word count and every word pair similar. Accepts
           "starbuks"~"starbucks", rejects "go service"~"service fee".
        Returns (category, score) where score is the weakest word-pair ratio.
        """
        if self._vectorizer is None:
            return OTHER_CATEGORY, 0.0
        windows = [
            w
            for w in self._normalizer.token_windows(self._normalizer.ml_text(text))
            if len(w) >= 3
        ]
        if not windows:
            return OTHER_CATEGORY, 0.0

        sims = cosine_similarity(self._vectorizer.transform(windows), self._vectors)
        if exclude:
            sims[:, np.array([c in exclude for c in self._categories])] = 0.0

        best_cat, best_score = OTHER_CATEGORY, 0.0
        for flat_idx in np.argsort(sims, axis=None)[::-1][:shortlist]:
            w_idx, r_idx = np.unravel_index(int(flat_idx), sims.shape)
            if sims[w_idx, r_idx] < self.SHORTLIST_MIN_COSINE:
                break
            score = self._word_alignment(windows[w_idx], self._corpus[r_idx])
            if score > best_score:
                best_cat, best_score = self._categories[r_idx], score
        if best_score >= threshold:
            return best_cat, best_score
        return OTHER_CATEGORY, best_score

    @staticmethod
    def _word_alignment(window: str, rule: str) -> float:
        w_tokens, r_tokens = window.split(), rule.split()
        if len(w_tokens) != len(r_tokens):
            return 0.0
        ratios = [
            # very short words must match exactly ("go", "sas", "gg")
            (1.0 if w == r else 0.0)
            if min(len(w), len(r)) <= 3
            else SequenceMatcher(None, w, r).ratio()
            for w, r in zip(w_tokens, r_tokens)
        ]
        return float(min(ratios))


# Handlers
class ClassificationHandler(ABC):
    """Base handler: try this strategy, otherwise forward to the next."""

    def __init__(self) -> None:
        self._next: Optional[ClassificationHandler] = None

    def set_next(self, handler: "ClassificationHandler") -> "ClassificationHandler":
        """Link the next handler and return it so callers can chain fluently."""
        self._next = handler
        return handler

    def handle(self, request: ClassificationRequest) -> ClassificationResult:
        result = self.try_classify(request)
        if result is not None:
            return result
        if self._next is not None:
            return self._next.handle(request)
        return ClassificationResult.unclassified()

    @abstractmethod
    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        """Return a result if this handler can classify the request, else None."""


class PrimaryDescriptionHandler(ClassificationHandler):
    """Rules on the bank's description BEFORE the appended payee.

    The description says WHAT ("Commission", "Ուսման վարձ"); the payee only
    says WHO. So it gets the first word. If the description decides nothing
    (e.g. ACBA's bare "Վճարում"), the rest of the chain sees the full text.
    """

    def __init__(self, matcher: RuleMatcher) -> None:
        super().__init__()
        self._matcher = matcher

    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        primary = request.primary
        if not primary:
            return None
        match = self._matcher.find_merchant_match(primary)
        method, confidence = ClassificationMethod.MERCHANT_RULE, MERCHANT_CONFIDENCE
        if match is None:
            match = self._matcher.find_keyword_match(primary)
            method, confidence = ClassificationMethod.KEYWORD_RULE, KEYWORD_CONFIDENCE
        if match is None or not request.allows(match[0]):
            return None
        return ClassificationResult(match[0], confidence, method.value)


class MerchantRuleHandler(ClassificationHandler):
    """Named merchant aliases on the full text."""

    def __init__(self, matcher: RuleMatcher) -> None:
        super().__init__()
        self._matcher = matcher

    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        match = self._matcher.find_merchant_match(request.cleaned)
        if match is None or not request.allows(match[0]):
            return None
        return ClassificationResult(
            match[0], MERCHANT_CONFIDENCE, ClassificationMethod.MERCHANT_RULE.value
        )


class KeywordRuleHandler(ClassificationHandler):
    """Generic category keywords (after merchants)."""

    def __init__(self, matcher: RuleMatcher) -> None:
        super().__init__()
        self._matcher = matcher

    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        match = self._matcher.find_keyword_match(request.cleaned)
        if match is None or not request.allows(match[0]):
            return None
        return ClassificationResult(
            match[0], KEYWORD_CONFIDENCE, ClassificationMethod.KEYWORD_RULE.value
        )


INDIVIDUAL_ENTREPRENEUR = "Individual Entrepreneurs"


class IndividualEntrepreneurDetector:
    """Recognises card payments to businesses registered under a person's name.

    Examples: "Վճարում | K. A. MELKONYAN", "Ք: ARTAK BEGYAN IE Yerevan AM".
    Personal transfers ("Անձնական փոխանցում | ...") never reach this check,
    because the Transfer keyword rule fires first.
    """

    # Legal-form markers: ԱՁ (Armenian), IE/PE (Latin), ИП (Russian).
    IE_MARKERS = frozenset({"աձ", "ie", "pe", "ип"})
    # Typical Armenian surname endings, Latin and Armenian script.
    SURNAME_RE = re.compile(r"(yan|ian|yants|ants|unts|uni|յան|յանց|ունի|ян)$")
    # Words that mean the counterparty is a business, not a person.
    BUSINESS_WORDS = frozenset({
        "market", "supermarket", "mart", "shop", "store", "pharm", "pharmacy",
        "house", "city", "club", "center", "centre", "clothes", "cafe", "bar",
        "fitness", "salon", "studio", "service", "group", "trade", "plus",
    })
    # Card-payment prefixes after cleaning: ACBA "Վճարում | ...", Ameriabank "Ք: ...".
    CARD_PREFIXES = ("վճարում ", "ք ")
    TAIL_NOISE = frozenset({
        "yerevan", "yerev", "yereva", "am", "arm", "armenia", "ամերիաբանկ", "փբը",
        "գլխամասային", "գրասենյակ", "երևան",
    })

    def matches(self, text: str) -> bool:
        tokens = text.split()
        if any(tok in self.IE_MARKERS for tok in tokens):
            return True
        if not text.startswith(self.CARD_PREFIXES):
            return False
        name = [t for t in tokens[1:] if t not in self.TAIL_NOISE and not t.isdigit()]
        if not 2 <= len(name) <= 4:
            return False
        if any(t in self.BUSINESS_WORDS or not t.isalpha() for t in name):
            return False
        return any(len(t) > 4 and self.SURNAME_RE.search(t) for t in name)


def looks_like_individual_entrepreneur(text: str) -> bool:
    """Function alias kept for existing callers."""
    return IndividualEntrepreneurDetector().matches(text)


class IndividualEntrepreneurHandler(ClassificationHandler):
    """Person-named card payees get their own bucket instead of a wrong guess.

    In Armenia many small shops, salons and kiosks are individual
    entrepreneurs, so the statement shows only a name.
    """

    def __init__(self, detector: Optional[IndividualEntrepreneurDetector] = None) -> None:
        super().__init__()
        self._detector = detector or IndividualEntrepreneurDetector()

    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        if not request.allows(INDIVIDUAL_ENTREPRENEUR):
            return None
        if self._detector.matches(request.cleaned):
            return ClassificationResult(
                INDIVIDUAL_ENTREPRENEUR,
                NAME_HEURISTIC_CONFIDENCE,
                ClassificationMethod.NAME_HEURISTIC.value,
            )
        return None


class LearnedModelHandler(ClassificationHandler):
    """Supervised model trained on rules + the user's own labelled history."""

    def __init__(self, model: LearnedCategoryModel) -> None:
        super().__init__()
        self._model = model

    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        pred = self._model.predict(request.cleaned, exclude=request.exclude)
        if pred is None:
            return None
        return ClassificationResult(
            pred.category, round(pred.probability, 2), ClassificationMethod.ML_MODEL.value
        )


class VectorSimilarityHandler(ClassificationHandler):
    """Terminal handler: fuzzy match; always returns a result (maybe 'Other')."""

    def __init__(self, index: FuzzyRuleIndex) -> None:
        super().__init__()
        self._index = index

    def try_classify(
        self, request: ClassificationRequest
    ) -> Optional[ClassificationResult]:
        category, score = self._index.predict(request.cleaned, exclude=request.exclude)
        if category == OTHER_CATEGORY:
            return ClassificationResult.unclassified(round(score, 2))
        return ClassificationResult(
            category, round(score, 2), ClassificationMethod.VECTOR.value
        )


# Builder
class ClassificationChainBuilder:
    """Builder: assembles the handler chain in one readable place."""

    def __init__(self) -> None:
        self._handlers: list[ClassificationHandler] = []

    def then(self, handler: Optional[ClassificationHandler]) -> "ClassificationChainBuilder":
        """Append a handler; ``None`` is skipped (for optional steps)."""
        if handler is not None:
            self._handlers.append(handler)
        return self

    def build(self) -> ClassificationHandler:
        if not self._handlers:
            raise ValueError("A classification chain needs at least one handler.")
        for current, nxt in zip(self._handlers, self._handlers[1:]):
            current.set_next(nxt)
        return self._handlers[0]


def build_classification_chain(
    matcher: RuleMatcher,
    fuzzy_index: FuzzyRuleIndex,
    model: Optional[LearnedCategoryModel] = None,
) -> ClassificationHandler:
    """Default chain: primary description -> merchant -> keyword ->
    entrepreneur name -> learned model -> fuzzy vector -> Other."""
    return (
        ClassificationChainBuilder()
        .then(PrimaryDescriptionHandler(matcher))
        .then(MerchantRuleHandler(matcher))
        .then(KeywordRuleHandler(matcher))
        .then(IndividualEntrepreneurHandler())
        .then(LearnedModelHandler(model) if model is not None else None)
        .then(VectorSimilarityHandler(fuzzy_index))
        .build()
    )
