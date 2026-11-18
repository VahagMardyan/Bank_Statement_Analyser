"""'Keep me signed in': long-lived login tokens stored in a browser cookie.

The cookie value is ``selector:validator``, two random strings:
* ``selector`` finds the row in ``auth_tokens`` (stored as is),
* ``validator`` proves the cookie is genuine; only its SHA-256 is stored.

So a leaked database doesn't give working cookies, and the cookie itself
says nothing about which user it belongs to.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .db import get_engine

REMEMBER_DAYS = 30


def _hash(validator: str) -> str:
    return hashlib.sha256(validator.encode("ascii")).hexdigest()


class TokenRepository(ABC):
    @abstractmethod
    def add(self, user_id: int, selector: str, token_hash: str, expires_at: datetime) -> None: ...

    @abstractmethod
    def find(self, selector: str) -> Optional[tuple[int, str]]:
        """(user_id, token_hash) of a token that hasn't expired, or None."""

    @abstractmethod
    def delete(self, selector: str) -> None: ...

    @abstractmethod
    def delete_for_user(self, user_id: int) -> None: ...


class MySqlTokenRepository(TokenRepository):
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()

    def add(self, user_id: int, selector: str, token_hash: str, expires_at: datetime) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM auth_tokens WHERE user_id = :uid AND expires_at < NOW()"),
                {"uid": user_id},
            )
            conn.execute(
                text(
                    "INSERT INTO auth_tokens (user_id, selector, token_hash, expires_at) "
                    "VALUES (:uid, :sel, :hash, :exp)"
                ),
                {"uid": user_id, "sel": selector, "hash": token_hash, "exp": expires_at},
            )

    def find(self, selector: str) -> Optional[tuple[int, str]]:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT user_id, token_hash FROM auth_tokens "
                    "WHERE selector = :sel AND expires_at > NOW()"
                ),
                {"sel": selector},
            ).first()
        return (row[0], row[1]) if row else None

    def delete(self, selector: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM auth_tokens WHERE selector = :sel"), {"sel": selector})

    def delete_for_user(self, user_id: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("DELETE FROM auth_tokens WHERE user_id = :uid"), {"uid": user_id})


class RememberMeService:
    def __init__(self, tokens: TokenRepository | None = None) -> None:
        self._tokens = tokens or MySqlTokenRepository()

    def issue(self, user_id: int) -> str:
        """Create a token; return the cookie value."""
        selector = secrets.token_hex(12)
        validator = secrets.token_hex(32)
        expires = datetime.now() + timedelta(days=REMEMBER_DAYS)
        self._tokens.add(user_id, selector, _hash(validator), expires)
        return f"{selector}:{validator}"

    def resolve(self, cookie_value: Optional[str]) -> Optional[int]:
        """User id for a valid cookie, otherwise None."""
        parsed = self._parse(cookie_value)
        if parsed is None:
            return None
        selector, validator = parsed
        found = self._tokens.find(selector)
        if found is None:
            return None
        user_id, token_hash = found
        if not hmac.compare_digest(token_hash, _hash(validator)):
            self._tokens.delete(selector)  # right selector, wrong validator: forged
            return None
        return user_id

    def revoke(self, cookie_value: Optional[str]) -> None:
        parsed = self._parse(cookie_value)
        if parsed is not None:
            self._tokens.delete(parsed[0])

    def revoke_all(self, user_id: int) -> None:
        """Sign the user out on every device (e.g. after a password change)."""
        self._tokens.delete_for_user(user_id)

    @staticmethod
    def _parse(cookie_value: Optional[str]) -> Optional[tuple[str, str]]:
        if not cookie_value or cookie_value.count(":") != 1:
            return None
        selector, validator = cookie_value.split(":")
        if len(selector) != 24 or len(validator) != 64:
            return None
        return selector, validator
