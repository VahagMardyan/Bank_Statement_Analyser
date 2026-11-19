"""Email reminders for upcoming recurring payments, plus budget warnings.

Run once a day by ``send_reminders.py``. For every user who has reminders on:
  * recurring payments due within their "days before" setting
  * advisor warnings of the over-budget kinds for the current month
are put into one email. ``notifications_log`` remembers what was sent, so the
same reminder never goes out twice, even if the script runs several times a day.

How the email is sent is a Strategy (``NotificationChannel``): SMTP for real,
the console for a dry run.
"""

from __future__ import annotations

import os
import smtplib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from email.message import EmailMessage
from typing import Callable, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from .advisor import AdviceKind
from .db import get_engine
from .i18n import Translator

REMINDER, ADVICE = "reminder", "advice"   # notifications_log.kind values
WARN_KINDS = (AdviceKind.OVER_BUDGET, AdviceKind.BUDGET_PACE)


# delivery
class NotificationChannel(ABC):
    @abstractmethod
    def send(self, to: str, subject: str, body: str) -> None: ...


class EmailChannel(NotificationChannel):
    """SMTP with STARTTLS (port 587) or SSL (port 465)."""

    def __init__(self, host: str, port: int, user: str, password: str,
                 sender: Optional[str] = None) -> None:
        self.host, self.port, self.user, self.password = host, port, user, password
        self.sender = sender or user

    @classmethod
    def from_env(cls) -> "EmailChannel":
        """SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD and optional SMTP_FROM (.env)."""
        missing = [k for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD") if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing in .env: {', '.join(missing)}")
        return cls(
            host=os.environ["SMTP_HOST"],
            port=int(os.getenv("SMTP_PORT", "587")),
            user=os.environ["SMTP_USER"],
            password=os.environ["SMTP_PASSWORD"],
            sender=os.getenv("SMTP_FROM"),
        )

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        if self.port == 465:
            with smtplib.SMTP_SSL(self.host, self.port, timeout=30) as smtp:
                smtp.login(self.user, self.password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(self.host, self.port, timeout=30) as smtp:
                smtp.starttls()
                smtp.login(self.user, self.password)
                smtp.send_message(msg)


class ConsoleChannel(NotificationChannel):
    """Dry run: print instead of sending."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append((to, subject, body))
        print(f"--- To: {to}\nSubject: {subject}\n\n{body}\n")


# storage
@dataclass(frozen=True)
class Recipient:
    user_id: int
    email: str
    name: str
    language: str
    days_before: int


class NotificationStore(ABC):
    @abstractmethod
    def recipients(self) -> list[Recipient]:
        """Users who have reminders turned on."""

    @abstractmethod
    def already_sent(self, user_id: int, kind: str, keys: list[str]) -> set[str]: ...

    @abstractmethod
    def mark_sent(self, user_id: int, kind: str, keys: list[str]) -> None: ...


class MySqlNotificationStore(NotificationStore):
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or get_engine()

    def recipients(self) -> list[Recipient]:
        with self._engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, email, display_name, language, remind_days_before "
                "FROM users WHERE notification_enabled"
            )).all()
        return [Recipient(r[0], r[1], r[2], r[3], int(r[4])) for r in rows]

    def already_sent(self, user_id: int, kind: str, keys: list[str]) -> set[str]:
        if not keys:
            return set()
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT reference_key FROM notifications_log "
                     "WHERE user_id = :uid AND kind = :kind"),
                {"uid": user_id, "kind": kind},
            ).scalars()
            return set(rows) & set(keys)

    def mark_sent(self, user_id: int, kind: str, keys: list[str]) -> None:
        if not keys:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text("INSERT IGNORE INTO notifications_log (user_id, kind, reference_key) "
                     "VALUES (:uid, :kind, :key)"),
                [{"uid": user_id, "kind": kind, "key": k[:100]} for k in keys],
            )


# what goes into one email
@dataclass
class Digest:
    recipient: Recipient
    reminders: list[tuple[str, str]] = field(default_factory=list)   # (key, line)
    warnings: list[tuple[str, str]] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.reminders and not self.warnings


def money(value: float) -> str:
    return f"{value:,.0f}"


class ReminderService:
    def __init__(
        self,
        store: NotificationStore,
        channel: NotificationChannel,
        translator: Translator,
        facade_for_user: Callable[[int], object],
        today: Optional[date] = None,
    ) -> None:
        self.store = store
        self.channel = channel
        self.tr = translator
        self.facade_for_user = facade_for_user
        self.today = today or date.today()

    def run(self) -> list[Digest]:
        """Send what's due; return the digests that were sent."""
        sent = []
        for recipient in self.store.recipients():
            digest = self.build_digest(recipient)
            if digest.empty:
                continue
            subject, body = self.render(digest)
            self.channel.send(recipient.email, subject, body)
            self.store.mark_sent(recipient.user_id, REMINDER, [k for k, _ in digest.reminders])
            self.store.mark_sent(recipient.user_id, ADVICE, [k for k, _ in digest.warnings])
            sent.append(digest)
        return sent

    def build_digest(self, recipient: Recipient) -> Digest:
        self.tr.language = recipient.language
        facade = self.facade_for_user(recipient.user_id)
        digest = Digest(recipient)
        currency = self.tr.t("meta.currency")

        payments = facade.upcoming_payments(None, self.today)
        for row in payments.itertuples(index=False):
            days = (row.next_due - self.today).days
            if 0 <= days <= recipient.days_before and pd.notna(row.expected_amount):
                key = f"recurring:{row.id}:{row.next_due.isoformat()}"
                line = self.tr.t(
                    "email.reminder_line",
                    name=str(row.name).capitalize(),
                    amount=money(row.expected_amount),
                    currency=currency,
                    date=row.next_due.isoformat(),
                    when=self._when(days),
                )
                digest.reminders.append((key, line))

        forecast = facade.month_forecast(None)
        if forecast is not None:
            month = forecast.month.strftime("%Y-%m")
            for a in facade.spending_advice(None):
                if a.kind not in WARN_KINDS:
                    continue
                key = f"advice:{a.kind.value}:{a.category}:{month}"
                line = self.tr.t(
                    f"advisor.{a.kind.value}",
                    category=self.tr.category(a.category), spent=money(a.spent),
                    projected=money(a.projected), reference=money(a.reference),
                    excess=money(a.excess), pct=int(a.pct_over),
                ).replace("**", "")
                digest.warnings.append((key, line))

        uid = recipient.user_id
        done = self.store.already_sent(uid, REMINDER, [k for k, _ in digest.reminders])
        digest.reminders = [(k, l) for k, l in digest.reminders if k not in done]
        done = self.store.already_sent(uid, ADVICE, [k for k, _ in digest.warnings])
        digest.warnings = [(k, l) for k, l in digest.warnings if k not in done]
        return digest

    def _when(self, days: int) -> str:
        if days == 0:
            return self.tr.t("email.today")
        if days == 1:
            return self.tr.t("email.tomorrow")
        return self.tr.t("email.in_days", days=days)

    def render(self, digest: Digest) -> tuple[str, str]:
        self.tr.language = digest.recipient.language
        parts = [self.tr.t("email.greeting", name=digest.recipient.name), ""]
        if digest.reminders:
            parts += [self.tr.t("email.reminders_header")]
            parts += [f"- {line}" for _, line in digest.reminders] + [""]
        if digest.warnings:
            parts += [self.tr.t("email.warnings_header")]
            parts += [f"- {line}" for _, line in digest.warnings] + [""]
        parts += [self.tr.t("email.footer")]
        return self.tr.t("email.subject"), "\n".join(parts)
