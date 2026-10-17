"""Text normalization shared by rules and the ML path."""

from __future__ import annotations

import re
import unicodedata

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
        "am", "arm", "amd", "usd", "eur", "rub", "yer", "yerevan", "armenia",
        "pos", "card", "visa", "mastercard", "mc", "arca", "purchase",
        "payment", "transaction", "ref", "txn", "terminal", "operation",
        "online", "mobile", "the", "for", "of", "and", "llc", "ltd", "cjsc",
        "ooo", "sp", "ie", "branch", "n", "no", "nr",
        "գործարք", "վճարում", "քարտ", "քարտով", "ՍՊԸ", "սպը", "երևան",
        "оплата", "покупка", "карта",
        # Ameriabank / ACBA boilerplate seen in real statements
        "ք", "ամերիաբանկ", "ամերիաբ", "ամերիաբան", "փբը", "բբը", "գլխամասային",
        "գրասենյակ", "մասնաճյուղ", "քոչար", "հծ", "վճարումներ", "pe", "ie", "po",
        "yerev", "yereva", "pos", "a", "b", "v", "վիր", "վիրտու", "online",
    }
)


def clean_text(text: str) -> str:
    """Lowercase, NFKC-normalize, drop dates/times/long numeric IDs/punctuation."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = unicodedata.normalize("NFKC", text).lower()
    t = t.replace("’", "'").replace("`", "'")
    t = DATE_RE.sub(" ", t)
    t = TIME_RE.sub(" ", t)
    t = MASKED_CARD_RE.sub(" ", t)
    t = LONG_ID_RE.sub(" ", t)
    t = NON_WORD_RE.sub(" ", t)
    return " ".join(t.split())


def ml_text(text: str) -> str:
    """Cleaned text with boilerplate tokens and bare numbers removed (ML input)."""
    tokens = [
        tok
        for tok in clean_text(text).split()
        if tok not in ML_NOISE_TOKENS and not tok.isdigit()
    ]
    return " ".join(tokens)


def token_windows(text: str, max_len: int = 3) -> list[str]:
    """All contiguous 1..max_len token spans, e.g. 'a b c' -> a, b, c, a b, b c, a b c."""
    tokens = text.split()
    spans: list[str] = []
    for size in range(1, max_len + 1):
        for i in range(len(tokens) - size + 1):
            spans.append(" ".join(tokens[i : i + size]))
    return spans
