"""Category rules: domain objects, an observable rule set, and its repository.

Patterns used here:
  * Repository  - :class:`JsonRuleRepository` hides how rules are stored
                  (JSON today; a DB later) from the classifier.
  * Observer    - :class:`RuleSet` notifies subscribers (the exact matcher,
                  the fuzzy index, the ML model) whenever rules change, so
                  nobody has to remember to call ``rebuild()`` / ``fit()``.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Protocol

from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "category_rules.json"


# --------------------------------------------------------------------------
# Domain objects
# --------------------------------------------------------------------------
@dataclass
class Merchant:
    """A named merchant and the (already normalized) aliases that identify it."""

    name: str
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "aliases": list(self.aliases)}


@dataclass
class CategoryRule:
    """Everything that maps text to one category."""

    keywords: list[str] = field(default_factory=list)
    merchants: list[Merchant] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "merchants": [m.to_dict() for m in self.merchants],
        }


class RuleSetObserver(Protocol):
    """Anything that must react when the rules change."""

    def on_rules_changed(self, rules: "RuleSet") -> None: ...


class RuleSet:
    """Observable collection of :class:`CategoryRule` keyed by category name."""

    def __init__(self, rules: Optional[dict[str, CategoryRule]] = None) -> None:
        self._rules: dict[str, CategoryRule] = dict(rules or {})
        self._observers: list[RuleSetObserver] = []

    # ---------------------------------------------------------- observer API
    def subscribe(self, observer: RuleSetObserver, notify_now: bool = True) -> None:
        self._observers.append(observer)
        if notify_now:
            observer.on_rules_changed(self)

    def _notify(self) -> None:
        for observer in self._observers:
            observer.on_rules_changed(self)

    # ---------------------------------------------------------------- queries
    def categories(self) -> list[str]:
        return sorted(self._rules)

    def __contains__(self, category: object) -> bool:
        return category in self._rules

    def __len__(self) -> int:
        return len(self._rules)

    def iter_keyword_rules(self) -> Iterator[tuple[str, str]]:
        """Yield ``(category, keyword)``."""
        for category, rule in self._rules.items():
            for keyword in rule.keywords:
                if keyword:
                    yield category, keyword

    def iter_merchant_rules(self) -> Iterator[tuple[str, str, str]]:
        """Yield ``(category, merchant_name, alias)``."""
        for category, rule in self._rules.items():
            for merchant in rule.merchants:
                for alias in merchant.aliases:
                    if alias:
                        yield category, merchant.name, alias

    def iter_rule_texts(self) -> Iterator[tuple[str, str]]:
        """Yield ``(category, text)`` per category: its keywords, then its aliases."""
        for category, rule in self._rules.items():
            for keyword in rule.keywords:
                yield category, keyword
            for merchant in rule.merchants:
                for alias in merchant.aliases:
                    yield category, alias

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {category: rule.to_dict() for category, rule in self._rules.items()}

    # -------------------------------------------------------------- commands
    def add_keyword(self, category: str, keyword: str) -> bool:
        """Add a keyword; returns True (and notifies) only if it was new."""
        if not keyword:
            return False
        rule = self._rules.setdefault(category, CategoryRule())
        if keyword in rule.keywords:
            return False
        rule.keywords.append(keyword)
        self._notify()
        return True

    def prune(self, should_remove: Callable[[str], bool]) -> int:
        """Remove keywords, and merchants with any alias, matching the predicate.

        Notifies once at the end. Returns the number of entries removed.
        """
        removed = 0
        for rule in self._rules.values():
            kept_keywords = [k for k in rule.keywords if not should_remove(k)]
            removed += len(rule.keywords) - len(kept_keywords)
            rule.keywords = kept_keywords

            kept_merchants = [
                m for m in rule.merchants if not any(should_remove(a) for a in m.aliases)
            ]
            removed += len(rule.merchants) - len(kept_merchants)
            rule.merchants = kept_merchants
        if removed:
            self._notify()
        return removed


# --------------------------------------------------------------------------
# Repository
# --------------------------------------------------------------------------
class RuleRepository(ABC):
    """Where rules come from and go back to."""

    @abstractmethod
    def load(self) -> RuleSet: ...

    @abstractmethod
    def save(self, rules: RuleSet) -> None: ...


class JsonRuleRepository(RuleRepository):
    """Reads/writes ``category_rules.json``.

    Accepted schema per category::

        "Transport": {
            "keywords": ["taxi", "տաքսի"],
            "merchants": [{"name": "Yandex Go", "aliases": ["yandex go", "յանդեքս"]}]
        }

    The old list form (``"Transport": ["taxi", "yandex"]``) and bare string
    merchants are still accepted. All keywords and aliases are normalized with
    the same :class:`TextNormalizer` the classifier uses.
    """

    def __init__(
        self,
        path: Optional[str | Path] = None,
        normalizer: TextNormalizer = DEFAULT_NORMALIZER,
    ) -> None:
        self.path = Path(path) if path else DEFAULT_RULES_PATH
        self._normalizer = normalizer

    # ------------------------------------------------------------------ load
    def load(self) -> RuleSet:
        if not self.path.exists():
            return RuleSet()
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{self.path.name} must contain a JSON object.")
        return RuleSet({str(cat): self._parse_rule(raw) for cat, raw in data.items()})

    def _norm(self, value: Any) -> str:
        return self._normalizer.clean(str(value))

    def _parse_rule(self, raw: Any) -> CategoryRule:
        if isinstance(raw, list):  # old schema
            return CategoryRule(keywords=self._unique(self._norm(x) for x in raw if x))
        if not isinstance(raw, dict):
            return CategoryRule()

        keywords_raw = raw.get("keywords", [])
        keywords = (
            self._unique(self._norm(v) for v in keywords_raw)
            if isinstance(keywords_raw, list)
            else []
        )
        merchants_raw = raw.get("merchants", [])
        merchants = [
            m
            for m in (self._parse_merchant(x) for x in merchants_raw)
            if m is not None
        ] if isinstance(merchants_raw, list) else []
        return CategoryRule(keywords=keywords, merchants=merchants)

    def _parse_merchant(self, raw: Any) -> Optional[Merchant]:
        if isinstance(raw, str):
            alias = self._norm(raw)
            return Merchant(name=raw, aliases=[alias] if alias else [])
        if not isinstance(raw, dict):
            return None
        name = str(raw.get("name", "")).strip()
        aliases = [self._norm(name)] if name else []
        aliases_raw = raw.get("aliases", [])
        if isinstance(aliases_raw, list):
            aliases.extend(self._norm(a) for a in aliases_raw if a)
        deduped = self._unique(aliases)
        if not deduped:
            return None
        return Merchant(name=name or deduped[0], aliases=deduped)

    @staticmethod
    def _unique(values: Any) -> list[str]:
        return list(dict.fromkeys(v for v in values if v))

    # ------------------------------------------------------------------ save
    def save(self, rules: RuleSet) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(rules.to_dict(), fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            # Classification keeps working in memory; the user should still know.
            logger.warning("Could not save rules to %s: %s", self.path, exc)


class InMemoryRuleRepository(RuleRepository):
    """Repository that never touches disk (tests, evaluation runs)."""

    def __init__(self, rules: Optional[RuleSet] = None) -> None:
        self._rules = rules or RuleSet()

    def load(self) -> RuleSet:
        return self._rules

    def save(self, rules: RuleSet) -> None:
        self._rules = rules
