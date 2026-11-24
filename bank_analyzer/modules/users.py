"""Users: storage (repository), password hashing and the auth service."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Optional

import bcrypt
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from .db import get_engine
from .models import User

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_BYTES = 72 # bcrypt's hard limit

@dataclass(frozen=True)
class NotificationSettings:
    enabled: bool
    days_before: int   # remind this many days before a recurring payment
    language: str      # account language; emails are sent in it


class AuthError(Exception):
    """Carries a locale key; the UI translates it."""
    def __init__(self, key: str) -> None:
        super().__init__(key)
        self.key = key

class UserRepository(ABC):
    @abstractmethod
    def find_by_email(self, email: str) -> Optional[tuple[User, str]]:
        """Return (user, password_hash) or None"""

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[tuple[User, str]]:
        """Return (user, password_hash) or None"""

    @abstractmethod
    def create(
        self, email: str, password_hash: str, display_name: str, language: str = "hy"
    ) -> User:
        pass

    @abstractmethod
    def update_display_name(self, user_id: int, display_name: str) -> None: ...

    @abstractmethod
    def update_language(self, user_id: int, language: str) -> None: ...

    @abstractmethod
    def update_password_hash(self, user_id: int, password_hash: str) -> None: ...

    @abstractmethod
    def get_notification_settings(self, user_id: int) -> NotificationSettings: ...

    @abstractmethod
    def update_notification_settings(self, user_id: int, settings: NotificationSettings) -> None: ...

    @abstractmethod
    def delete(self, user_id: int) -> None:
        """Remove the user; ON DELETE CASCADE removes everything they own."""

class MySqlUserRepository(UserRepository):
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()

    def find_by_email(self, email) -> Optional[tuple[User, str]]:
        with self._engine.connect() as conn:
            row = conn.execute(
            text(
                "SELECT id, email, display_name, language, password_hash "
                "FROM users WHERE email = :email"
            ),
                {"email": email},
            ).mappings().first()
            if row is None:
                return None

            user = User(
                id = row['id'],
                email = row['email'],
                display_name=row['display_name'],
                language=row['language']
            )
            return user, row['password_hash']

    def create(self, email, password_hash, display_name, language="hy") -> User:
        try:
            with self._engine.begin() as conn:
                result = conn.execute(
                    text(
                        "INSERT INTO users (email, password_hash, display_name, language) "
                        "VALUES (:email, :password_hash, :display_name, :language)"
                    ),
                    {'email' : email, "password_hash": password_hash,
                     "display_name": display_name, "language": language}
                )

                new_id = result.lastrowid
        except IntegrityError:
            raise AuthError('auth.error_email_taken') from None

        return User(id=new_id, email=email, display_name=display_name, language=language)

    def find_by_id(self, user_id: int) -> Optional[tuple[User, str]]:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT id, email, display_name, language, password_hash "
                    "FROM users WHERE id = :id"
                ),
                {"id": user_id},
            ).mappings().first()
        if row is None:
            return None
        user = User(
            id=row["id"],
            email=row["email"],
            display_name=row["display_name"],
            language=row["language"],
        )
        return user, row["password_hash"]

    def update_display_name(self, user_id: int, display_name: str) -> None:
        self._execute(
            "UPDATE users SET display_name = :name WHERE id = :id",
            {"name": display_name, "id": user_id},
        )

    def update_language(self, user_id: int, language: str) -> None:
        self._execute(
            "UPDATE users SET language = :lang WHERE id = :id",
            {"lang": language, "id": user_id},
        )

    def update_password_hash(self, user_id: int, password_hash: str) -> None:
        self._execute(
            "UPDATE users SET password_hash = :hash WHERE id = :id",
            {"hash": password_hash, "id": user_id},
        )

    def delete(self, user_id: int) -> None:
        self._execute("DELETE FROM users WHERE id = :id", {"id": user_id})

    def get_notification_settings(self, user_id: int) -> NotificationSettings:
        with self._engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT notification_enabled, remind_days_before, language "
                    "FROM users WHERE id = :id"
                ),
                {"id": user_id},
            ).first()
        if row is None:
            return NotificationSettings(False, 3, "hy")
        return NotificationSettings(bool(row[0]), int(row[1]), str(row[2]))

    def update_notification_settings(self, user_id: int, settings: NotificationSettings) -> None:
        self._execute(
            "UPDATE users SET notification_enabled = :on, remind_days_before = :days, "
            "language = :lang WHERE id = :id",
            {"on": settings.enabled, "days": settings.days_before,
             "lang": settings.language, "id": user_id},
        )

    def _execute(self, sql: str, params: dict) -> None:
        with self._engine.begin() as conn:
            conn.execute(text(sql), params)

class PasswordHasher:
    def hash(self, password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('ascii')

    def verify(self, password: str, password_hash: str) -> bool:
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('ascii'))
        except ValueError:
            return False

class AuthService:
    def __init__(
        self,
        users: UserRepository | None = None,
        hasher: PasswordHasher | None = None
    ) -> None:
        self._users = users or MySqlUserRepository()
        self._hasher = hasher or PasswordHasher()

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    def register(
        self, email: str, password: str, display_name: str, language: str = "hy"
    ) -> User:
        email = self._normalize_email(email)
        display_name = display_name.strip()

        if "@" not in email or not display_name:
            raise AuthError('auth.error_invalid_input')

        self._validate_password(password)
        return self._users.create(
            email, self._hasher.hash(password), display_name, self._check_language(language)
        )

    def login(self, email: str, password: str) -> User:
        found = self._users.find_by_email(self._normalize_email(email))

        if found is None or not self._hasher.verify(password, found[1]):
            raise AuthError("auth.error_wrong_credentials")

        return found[0]

    def user_by_id(self, user_id: int) -> Optional[User]:
        found = self._users.find_by_id(user_id)
        return found[0] if found else None

    def rename(self, user: User, new_name: str) -> User:
        name = new_name.strip()
        if not name:
            raise AuthError("auth.error_empty_name")
        self._users.update_display_name(user.id, name)
        return replace(user, display_name=name)

    def set_language(self, user: User, language: str) -> User:
        """Account language: the app opens in it after signing in, emails use it."""
        language = self._check_language(language)
        self._users.update_language(user.id, language)
        return replace(user, language=language)

    @staticmethod
    def _check_language(language: str) -> str:
        # users.language is VARCHAR(5): locale file names like "hy", "en", "ru".
        language = (language or "").strip()
        if not language or len(language) > 5:
            raise AuthError("auth.error_invalid_input")
        return language

    def change_password(self, user: User, current: str, new: str) -> None:
        self._check_password(user, current)
        self._validate_password(new)
        self._users.update_password_hash(user.id, self._hasher.hash(new))

    def delete_account(self, user: User, password: str) -> None:
        self._check_password(user, password)
        self._users.delete(user.id)

    def _check_password(self, user: User, password: str) -> None:
        """Sensitive actions need the current password, not just a session."""
        found = self._users.find_by_id(user.id)
        if found is None or not self._hasher.verify(password, found[1]):
            raise AuthError("auth.error_wrong_password")

    @staticmethod
    def _validate_password(password: str) -> None:
        if len(password) < MIN_PASSWORD_LENGTH:
            raise AuthError("auth.error_password_short")
        if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise AuthError("auth.error_password_long")

    def notification_settings(self, user: User) -> NotificationSettings:
        return self._users.get_notification_settings(user.id)

    def save_notification_settings(self, user: User, settings: NotificationSettings) -> None:
        days = max(0, min(int(settings.days_before), 30))
        self._users.update_notification_settings(user.id, replace(settings, days_before=days))

