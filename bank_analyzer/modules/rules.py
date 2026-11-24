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
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol

from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

logger = logging.getLogger(__name__)

# Old files used "Transport | Տրանսպորտ" keys; only the English part is the key.
LABEL_SEPARATOR = " | "

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "category_rules.json"

# Domain objects
@dataclass
class Merchant:
    """A named merchant and the (already normalized) aliases that identify it."""

    name: str
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "aliases": list(self.aliases)}


@dataclass
class CategoryRule:
    """Everything that maps text to one category.

    The category KEY (e.g. "Transport") is what the code, budgets and the
    training data use. Display names live in ``locales/<language>.json``.
    """

    keywords: list[str] = field(default_factory=list)
    merchants: list[Merchant] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "merchants": [m.to_dict() for m in self.merchants],
        }


@dataclass(frozen=True)
class CommunityRule:
    """A keyword enough users mapped to the same category."""

    keyword: str
    category: str
    users: int        # users who chose this category for the keyword
    total_users: int  # users who corrected this keyword at all

    @property
    def share(self) -> float:
        return self.users / self.total_users if self.total_users else 0.0


def agreed_rules(
    counts: Iterable[tuple[str, str, int]],
    min_users: int = 3,
    min_share: float = 0.75,
) -> list[CommunityRule]:
    """Pick the corrections users agree on.

    ``counts``: ``(keyword, category, distinct users)`` rows, e.g. from
    ``user_rules`` grouped by keyword and category. A keyword becomes a
    community rule when its most chosen category has at least ``min_users``
    users AND at least ``min_share`` of everyone who corrected that keyword.
    So 3 users saying Shopping and 2 saying Cafes (60%) gives no rule.
    """
    by_keyword: dict[str, dict[str, int]] = {}
    for keyword, category, users in counts:
        if keyword and category:
            cats = by_keyword.setdefault(keyword, {})
            cats[category] = cats.get(category, 0) + int(users)
    agreed: list[CommunityRule] = []
    for keyword, cats in by_keyword.items():
        total = sum(cats.values())
        category, users = max(cats.items(), key=lambda kv: (kv[1], kv[0]))
        if users >= min_users and users / total >= min_share:
            agreed.append(CommunityRule(keyword, category, users, total))
    return sorted(agreed, key=lambda r: (r.category, r.keyword))


class RuleSetObserver(Protocol):
    """Anything that must react when the rules change."""

    def on_rules_changed(self, rules: "RuleSet") -> None: ...


# Where a learned keyword came from (see RuleSet.set_learned).
LEARNED_USER = "user"            # this user's own override
LEARNED_COMMUNITY = "community"  # enough other users agreed on it


class RuleSet:
    """Observable collection of :class:`CategoryRule` keyed by category name.

    Keywords learned from overrides are also listed in ``_learned`` with their
    source. They are checked before merchant aliases, so a correction can
    fix a wrong merchant rule from category_rules.json.
    """

    def __init__(self, rules: Optional[dict[str, CategoryRule]] = None) -> None:
        self._rules: dict[str, CategoryRule] = dict(rules or {})
        self._observers: list[RuleSetObserver] = []
        self._learned: dict[str, tuple[str, str]] = {}  # keyword -> (category, source)

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

    def iter_learned(self) -> Iterator[tuple[str, str, str]]:
        """Yield ``(keyword, category, source)`` for keywords learned from overrides."""
        for keyword, (category, source) in self._learned.items():
            yield keyword, category, source

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {category: rule.to_dict() for category, rule in self._rules.items()}

    # commands
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

    def set_learned(
        self, category: str, keyword: str, source: str = LEARNED_USER
    ) -> bool:
        """Point ``keyword`` at ``category`` (and only there), marked as learned.

        The keyword is removed from every other category, so one text never
        matches two categories with the same keyword. Returns True (and
        notifies) only if something changed.
        """
        if not keyword:
            return False
        rule = self._rules.setdefault(category, CategoryRule())
        unchanged = (
            self._learned.get(keyword) == (category, source)
            and keyword in rule.keywords
            and not any(
                keyword in other.keywords
                for name, other in self._rules.items()
                if name != category
            )
        )
        if unchanged:
            return False
        for name, other in self._rules.items():
            if name != category and keyword in other.keywords:
                other.keywords.remove(keyword)
        if keyword not in rule.keywords:
            rule.keywords.append(keyword)
        self._learned[keyword] = (category, source)
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
        self._learned = {k: v for k, v in self._learned.items() if not should_remove(k)}
        if removed:
            self._notify()
        return removed

# Repository
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

    # load
    def load(self) -> RuleSet:
        if not self.path.exists():
            return RuleSet()
        with open(self.path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"{self.path.name} must contain a JSON object.")
        return RuleSet(
            {self._key(str(raw_key)): self._parse_rule(raw) for raw_key, raw in data.items()}
        )

    @staticmethod
    def _key(raw_key: str) -> str:
        """Accept "Transport | Տրանսպորտ" keys too: the key is the English part."""
        return raw_key.split(LABEL_SEPARATOR, 1)[0].strip()

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

    # save
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
