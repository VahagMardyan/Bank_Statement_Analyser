"""Transaction classifier: wires rules, indexes, ML model and the handler chain.

The classifier itself no longer parses JSON, matches text or trains anything.
It composes collaborators (each replaceable through the constructor):

    JsonRuleRepository --load--> RuleSet --notifies--> RuleMatcher
                                         --notifies--> FuzzyRuleIndex
                                         --notifies--> LearnedCategoryModel
    ClassificationChainBuilder -> handler chain (see classification_handlers)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .classification_handlers import (
    ClassificationHandler,
    FuzzyRuleIndex,
    RuleMatcher,
    build_classification_chain,
)
from .constants import INCOME_CATEGORIES
from .ml_model import LearnedCategoryModel, TrainingSource
from .models import (
    TRUSTED_METHODS,
    ClassificationMethod,
    ClassificationRequest,
    ClassificationResult,
    TransactionType,
)
from .rules import JsonRuleRepository, RuleRepository, RuleSet
from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

RESULT_COLUMNS = ["category", "confidence", "method"]


class TransactionClassifier:
    """Hybrid rule + ML classifier. Implements ``ClassifierProtocol`` and
    ``SelfTrainingClassifier``."""

    def __init__(
        self,
        rules_path: Optional[str | Path] = None,
        model: Optional[LearnedCategoryModel] = None,
        use_model: bool = True,
        *,
        repository: Optional[RuleRepository] = None,
        normalizer: TextNormalizer = DEFAULT_NORMALIZER,
    ) -> None:
        self._normalizer = normalizer
        self._repository = repository or JsonRuleRepository(rules_path, normalizer)
        self._rules: RuleSet = self._repository.load()

        self._matcher = RuleMatcher()
        self._fuzzy = FuzzyRuleIndex(normalizer)
        self.model: Optional[LearnedCategoryModel] = (
            (model or LearnedCategoryModel(normalizer=normalizer)) if use_model else None
        )

        # Observers: each builds its index now and rebuilds on every change.
        self._rules.subscribe(self._matcher)
        self._rules.subscribe(self._fuzzy)
        if self.model is not None:
            self._rules.subscribe(self.model)

        self._chain: ClassificationHandler = build_classification_chain(
            self._matcher, self._fuzzy, self.model
        )

    # accessors
    @property
    def rules(self) -> RuleSet:
        return self._rules

    @property
    def rules_path(self) -> Optional[Path]:
        return getattr(self._repository, "path", None)

    @property
    def normalizer(self) -> TextNormalizer:
        return self._normalizer

    @property
    def matcher(self) -> RuleMatcher:
        return self._matcher

    def get_categories(self) -> list[str]:
        return self._rules.categories()

    # classification
    def classify_description_result(
        self, description: str, exclude: frozenset[str] = frozenset()
    ) -> ClassificationResult:
        if not description or not str(description).strip():
            return ClassificationResult.unclassified()
        request = ClassificationRequest(str(description), self._normalizer, exclude)
        if not request.cleaned:
            return ClassificationResult.unclassified()
        return self._chain.handle(request)

    def classify_description_details(self, description: str) -> tuple[str, float, str]:
        """Return ``(category, confidence, method)`` for one description."""
        return self.classify_description_result(description).as_tuple()

    def classify_description(self, description: str) -> str:
        return self.classify_description_result(description).category

    def classify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``category``, ``confidence`` and ``method`` columns."""
        out = df.copy()
        source_col = self._text_column(out)
        if source_col is None:
            unclassified = ClassificationResult.unclassified()
            for col, value in zip(RESULT_COLUMNS, unclassified.as_tuple()):
                out[col] = value
            return out

        texts = out[source_col].fillna("").astype(str)
        if "transaction_type" in out.columns:
            # An expense can never be Income/Salary.
            is_expense = out["transaction_type"] == TransactionType.EXPENSE.value
        else:
            is_expense = pd.Series(False, index=out.index)

        results = [
            self.classify_description_result(
                text, INCOME_CATEGORIES if expense else frozenset()
            )
            for text, expense in zip(texts, is_expense)
        ]
        out["category"] = [r.category for r in results]
        out["confidence"] = [r.confidence for r in results]
        out["method"] = [r.method for r in results]
        return out

    def classify(self, target: Union[str, pd.DataFrame]) -> Union[str, pd.DataFrame]:
        """Classify either a single description or a DataFrame."""
        if isinstance(target, pd.DataFrame):
            return self.classify_dataframe(target)
        return self.classify_description(str(target))

    @staticmethod
    def _text_column(df: pd.DataFrame) -> Optional[str]:
        # The raw description carries more signal than the cleaned one.
        for col in ("description", "cleaned_description"):
            if col in df.columns:
                return col
        return None

    # learning
    def manual_override(
        self,
        transaction_id: str,
        new_category: str,
        cleaned_description: str,
        persist: bool = True,
    ) -> None:
        """Save a correction as a keyword rule and as a strong training example.

        ``transaction_id`` is kept for interface compatibility: rules are
        description-based, so the override applies to every matching text.
        """
        del transaction_id
        keyword = self._normalizer.clean(cleaned_description)
        if not keyword:
            return

        if self.model is not None:
            # Store the example first; the rule change below triggers the refit.
            self.model.add_examples(
                [(cleaned_description, new_category)],
                source=TrainingSource.MANUAL,
                persist=persist,
            )
        rule_added = self._rules.add_keyword(new_category, keyword)
        if not rule_added and self.model is not None:
            self.model.fit()  # rules unchanged, so no notification - refit here

        if persist:
            self._repository.save(self._rules)

    def learn_from_dataframe(self, classified: pd.DataFrame, persist: bool = True) -> int:
        """Self-training: store rows the rules labelled with certainty, retrain.

        Returns the number of new training examples added.
        """
        if self.model is None or classified.empty or "method" not in classified:
            return 0
        source_col = self._text_column(classified)
        if source_col is None:
            return 0
        trusted = classified[classified["method"].isin(TRUSTED_METHODS)]
        added = self.model.add_examples(
            zip(trusted[source_col].astype(str), trusted["category"].astype(str)),
            source=TrainingSource.RULE,
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
        redo = self.classify_dataframe(
            out.loc[~keep, out.columns.difference(RESULT_COLUMNS)]
        )
        out.loc[~keep, RESULT_COLUMNS] = redo[RESULT_COLUMNS].to_numpy()
        return out
