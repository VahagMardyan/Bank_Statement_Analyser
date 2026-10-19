"""Text normalization shared by the loader, the rules and the ML path.

There is exactly ONE cleaner in the project (``TextNormalizer.clean``). The
loader's ``cleaned_description`` column, the rule matcher and manual overrides
all go through it, so a keyword saved from an override always matches the text
the classifier sees next time.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

DATE_RE = re.compile(
    r"\b\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}\b|\b\d{4}[./\-]\d{1,2}[./\-]\d{1,2}\b"
)
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
LONG_ID_RE = re.compile(r"\b\d{5,}\b")
MASKED_CARD_RE = re.compile(r"\b\d{2,6}[x*]{2,}\d{0,4}\b", re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s%]+", re.UNICODE)

# Bank boilerplate that appears in almost every row and carries no category
# signal. Removed ONLY on the ML path - rules still see the full text, so
# keywords such as "atm" or "fee" keep working there.
ML_NOISE_TOKENS: frozenset[str] = frozenset(
    {
        # currencies / geography
        "am", "arm", "amd", "usd", "eur", "rub", "yer", "yerev", "yereva",
        "yerevan", "armenia", "երևան",
        # card / payment boilerplate
        "pos", "card", "visa", "mastercard", "mc", "arca", "purchase", "payment",
        "transaction", "ref", "txn", "terminal", "operation", "online", "mobile",
        "գործարք", "վճարում", "վճարումներ", "քարտ", "քարտով",
        "оплата", "покупка", "карта",
        # legal forms and filler words
        "the", "for", "of", "and", "llc", "ltd", "cjsc", "ooo", "sp", "ie", "pe",
        "po", "branch", "n", "no", "nr", "a", "b", "v", "սպը",
        # Ameriabank / ACBA boilerplate seen in real statements
        "ք", "ամերիաբանկ", "ամերիաբ", "ամերիաբան", "փբը", "բբը", "գլխամասային",
        "գրասենյակ", "մասնաճյուղ", "քոչար", "հծ", "վիր", "վիրտու",
    }
)


class TextNormalizer:
    """Turns raw statement text into the forms the pipeline needs.

    * :meth:`clean` - for rules, ``cleaned_description`` and overrides.
    * :meth:`ml_text` - :meth:`clean` minus boilerplate tokens and bare numbers.
    * :meth:`token_windows` - short spans for the fuzzy matcher.
    """

    def __init__(self, noise_tokens: Iterable[str] = ML_NOISE_TOKENS) -> None:
        # Lower-case the noise set: clean() lower-cases, so an upper-case
        # entry (the old "ՍՊԸ") could never match.
        self._noise = frozenset(t.lower() for t in noise_tokens)

    @property
    def noise_tokens(self) -> frozenset[str]:
        return self._noise

    def clean(self, text: str) -> str:
        """Lowercase, NFKC-normalize, drop dates/times/IDs/card masks/punctuation."""
        if not isinstance(text, str) or not text.strip():
            return ""
        t = unicodedata.normalize("NFKC", text).lower()
        t = t.replace("’", "'").replace("`", "'")
        for pattern in (DATE_RE, TIME_RE, MASKED_CARD_RE, LONG_ID_RE, NON_WORD_RE):
            t = pattern.sub(" ", t)
        return " ".join(t.split())

    def ml_text(self, text: str) -> str:
        """Cleaned text with boilerplate tokens and bare numbers removed."""
        return " ".join(
            tok
            for tok in self.clean(text).split()
            if tok not in self._noise and not tok.isdigit()
        )

    @staticmethod
    def token_windows(text: str, max_len: int = 3) -> list[str]:
        """All contiguous 1..max_len token spans: 'a b c' -> a, b, c, a b, b c, a b c."""
        tokens = text.split()
        return [
            " ".join(tokens[i : i + size])
            for size in range(1, max_len + 1)
            for i in range(len(tokens) - size + 1)
        ]


# Shared default instance + function aliases kept for existing callers/scripts.
DEFAULT_NORMALIZER = TextNormalizer()


def clean_text(text: str) -> str:
    return DEFAULT_NORMALIZER.clean(text)


def ml_text(text: str) -> str:
    return DEFAULT_NORMALIZER.ml_text(text)


def token_windows(text: str, max_len: int = 3) -> list[str]:
    return TextNormalizer.token_windows(text, max_len)
