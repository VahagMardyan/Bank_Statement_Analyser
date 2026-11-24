"""MySQL backend repositories: per-user rules and training rows."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from typing import Iterable

from .history import BASE_CURRENCY, TransactionRepository, make_fingerprints
from .text_utils import DEFAULT_NORMALIZER, TextNormalizer

from .db import get_engine
from .ml_model import TRAINING_COLUMNS, TrainingStore
from .rules import (
    LEARNED_COMMUNITY,
    LEARNED_USER,
    CommunityRule,
    JsonRuleRepository,
    RuleRepository,
    RuleSet,
    agreed_rules,
)
from .budgets import BudgetRepository

from .recurring import (
    PAYMENT_COLUMNS,
    PERIOD_BY_DAYS,
    Period,
    RecurringPayment,
    RecurringRepository
)

class MySqlCommunityRules:
    """Corrections many users agree on, computed from ``user_rules``.

    No extra table: every user's overrides are already there. See
    :func:`modules.rules.agreed_rules` for the agreement test. Defaults can be
    changed in .env with COMMUNITY_MIN_USERS and COMMUNITY_MIN_SHARE.
    """

    def __init__(
        self,
        engine: Engine | None = None,
        min_users: int | None = None,
        min_share: float | None = None,
    ) -> None:
        self._engine = engine or get_engine()
        self.min_users = min_users if min_users is not None else _env_int("COMMUNITY_MIN_USERS", 3)
        self.min_share = (
            min_share if min_share is not None else _env_float("COMMUNITY_MIN_SHARE", 0.75)
        )

    def load(self) -> list[CommunityRule]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT keyword, category, COUNT(DISTINCT user_id) "
                    "FROM user_rules GROUP BY keyword, category"
                )
            ).all()
        return agreed_rules(
            ((kw, cat, n) for kw, cat, n in rows), self.min_users, self.min_share
        )


class MySqlRuleRepository(RuleRepository):
    """Shared rules from JSON + community rules + this user's own keywords.

    Later layers win for the same keyword: JSON < community < the user's own.
    """

    def __init__(
            self,
            user_id: int,
            shared: RuleRepository | None = None,
            engine: Engine | None = None,
            community: MySqlCommunityRules | None | bool = True,
            ):
        self.user_id = user_id
        self._shared = shared or JsonRuleRepository()
        self._engine = engine or get_engine()
        if community is True:
            community = MySqlCommunityRules(self._engine)
        self._community = community or None

    def load(self) -> RuleSet:
        rules = self._shared.load()
        known = set(rules.categories())
        if self._community is not None:
            for rule in self._community.load():
                if rule.category in known:  # ignore categories that no longer exist
                    rules.set_learned(rule.category, rule.keyword, LEARNED_COMMUNITY)

        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT category, keyword FROM user_rules WHERE user_id = :uid"),
                {"uid" : self.user_id},
            ).all()

        for category, keyword in rows:
            rules.set_learned(category, keyword, LEARNED_USER)

        return rules

    def save(self, rules: RuleSet) -> None:
        # Only the user's own overrides. Community rules are loaded into the
        # same RuleSet; saving them here would count them as this user's vote.
        personal = [
            {"uid": self.user_id, "cat": category, "kw": keyword}
            for keyword, category, source in rules.iter_learned()
            if source == LEARNED_USER
        ]
        if not personal:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO user_rules (user_id, category, keyword) "
                    "VALUES (:uid, :cat, :kw) "
                    "ON DUPLICATE KEY UPDATE category = VALUES(category)"
                ), personal
            )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default

class MySqlTrainingExampleStore(TrainingStore):
    """The ML model's learned rows for one user."""
    def __init__(self, user_id: int, engine: Engine | None = None) -> None:
        self.user_id = user_id
        self._engine = engine or get_engine()

    def load(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT text, category, source FROM training_examples "
                    "WHERE user_id = :uid"
                ),
                conn,
                params={"uid": self.user_id},
            )
        return df[TRAINING_COLUMNS].astype(str)

    def save(self, rows: pd.DataFrame) -> None:
        records = [
            {'uid' : self.user_id, 'text' : r.text, 'cat' : r.category, 'src' : r.source}
            for r in rows[TRAINING_COLUMNS].itertuples(index = False)
        ]
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM training_examples WHERE user_id = :uid"),
                {"uid" : self.user_id}
            )

            if records:
                conn.execute(
                    text(
                        'INSERT INTO training_examples (user_id, text, category, source) '
                        'VALUES (:uid, :text, :cat, :src)'
                    ),
                    records
                )

class MySqlTransactionRepository(TransactionRepository):
    """One user's transaction history in MySQL."""

    def __init__(
        self,
        user_id: int,
        engine: Engine | None = None,
        normalizer: TextNormalizer = DEFAULT_NORMALIZER,
    ) -> None:
        self.user_id = user_id
        self._engine = engine or get_engine()
        self._normalizer = normalizer

    def save(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        rows = df.assign(fingerprint=make_fingerprints(df))
        with self._engine.begin() as conn:
            existing = set(
                conn.execute(
                    text("SELECT fingerprint FROM transactions WHERE user_id = :uid"),
                    {"uid": self.user_id},
                ).scalars()
            )
            new = rows[~rows["fingerprint"].isin(existing)]
            if new.empty:
                return 0
            new = new.assign(
                original_amount=(
                    new["original_amount"].fillna(new["amount"])
                    if "original_amount" in new.columns else new["amount"]
                ),
                original_currency=(
                    new["original_currency"].fillna(BASE_CURRENCY)
                    if "original_currency" in new.columns else BASE_CURRENCY
                ),
            )
            records = [
                {
                    "uid": self.user_id,
                    "dt": r.date.to_pydatetime(),
                    "descr": str(r.description),
                    "amount": round(float(r.amount), 2),
                    "oamount": round(float(r.original_amount), 2),
                    "ocur": str(r.original_currency),
                    "ttype": r.transaction_type,
                    "cat": r.category,
                    "method": r.method,
                    "conf": None if pd.isna(r.confidence) else round(float(r.confidence), 3),
                    "fp": r.fingerprint,
                }
                for r in new.itertuples(index=False)
            ]
            conn.execute(
                text(
                    "INSERT INTO transactions (user_id, txn_date, description, amount, "
                    "original_amount, original_currency, "
                    "transaction_type, category, method, confidence, fingerprint) "
                    "VALUES (:uid, :dt, :descr, :amount, :oamount, :ocur, "
                    ":ttype, :cat, :method, :conf, :fp)"
                ),
                records,
            )
        return len(records)

    def load(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT id, txn_date, description, amount, original_amount, "
                    "original_currency, transaction_type, "
                    "category, method, confidence, fingerprint "
                    "FROM transactions WHERE user_id = :uid ORDER BY txn_date"
                ),
                conn,
                params={"uid": self.user_id},
            )
        df = df.rename(columns={"txn_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df["amount"] = df["amount"].astype(float)
        # Rows saved before currency support have no original values: they were AMD.
        df["original_amount"] = df["original_amount"].astype(float).fillna(df["amount"])
        df["original_currency"] = df["original_currency"].fillna(BASE_CURRENCY).astype(str)
        df["confidence"] = df["confidence"].astype(float)
        df["cleaned_description"] = df["description"].map(self._normalizer.clean)
        df["transaction_id"] = df.pop("id").astype(str)
        return df

    def clear(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text("DELETE FROM transactions WHERE user_id = :uid"),
                {"uid": self.user_id},
            )

    def update_category(
        self, fingerprints: Iterable[str], category: str, method: str
    ) -> None:
        records = [
            {"uid": self.user_id, "fp": fp, "cat": category, "method": method}
            for fp in fingerprints
        ]
        if not records:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE transactions SET category = :cat, method = :method, "
                    "confidence = 1.0 WHERE user_id = :uid AND fingerprint = :fp"
                ),
                records,
            )

class MySqlRecurringRepository(RecurringRepository):
    """One user's recurring payments in MySQL."""

    def __init__(self, user_id: int, engine: Engine | None = None) -> None:
        self.user_id = user_id
        self._engine = engine or get_engine()

    def sync(self, detected: list[RecurringPayment]) -> None:
        records = [
            {
                "uid": self.user_id,
                "name": p.name,
                "cat": p.category,
                "amount": p.expected_amount,
                "interval": p.period.days,
                "last_paid": p.last_paid,
                "next_due": p.next_due,
            }
            for p in detected
        ]
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE recurring_payments SET active = FALSE "
                    "WHERE user_id = :uid AND source = 'detected'"
                ),
                {"uid": self.user_id},
            )
            if records:
                conn.execute(
                    text(
                        "INSERT INTO recurring_payments (user_id, name, category, "
                        "expected_amount, interval_days, last_paid, next_due, source, active) "
                        "VALUES (:uid, :name, :cat, :amount, :interval, :last_paid, "
                        ":next_due, 'detected', TRUE) "
                        "ON DUPLICATE KEY UPDATE category = VALUES(category), "
                        "expected_amount = VALUES(expected_amount), "
                        "interval_days = VALUES(interval_days), last_paid = VALUES(last_paid), "
                        "next_due = VALUES(next_due), active = TRUE"
                    ),
                    records,
                )

    def load(self) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(
                text(
                    "SELECT id, name, category, expected_amount, interval_days, "
                    "last_paid, next_due, source FROM recurring_payments "
                    "WHERE user_id = :uid AND active AND NOT dismissed "
                    "ORDER BY next_due"
                ),
                conn,
                params={"uid": self.user_id},
            )
        df["expected_amount"] = df["expected_amount"].astype(float)
        df["period"] = df.pop("interval_days").map(
            lambda d: PERIOD_BY_DAYS[d].key if d in PERIOD_BY_DAYS else f"{d}d"
        )
        for col in ("last_paid", "next_due"):
            df[col] = pd.to_datetime(df[col]).dt.date
        return df[PAYMENT_COLUMNS]

    def clear(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "DELETE FROM recurring_payments "
                    "WHERE user_id = :uid AND source = 'detected'"
                ),
                {"uid": self.user_id},
            )

    def add_manual(
        self, name: str, category: str, amount: float, period: Period, next_due: date
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO recurring_payments (user_id, name, category, "
                    "expected_amount, interval_days, last_paid, next_due, source, active) "
                    "VALUES (:uid, :name, :cat, :amount, :interval, NULL, :next_due, "
                    "'manual', TRUE) "
                    "ON DUPLICATE KEY UPDATE category = VALUES(category), "
                    "expected_amount = VALUES(expected_amount), interval_days = VALUES(interval_days), "
                    "next_due = VALUES(next_due), source = 'manual', active = TRUE, dismissed = FALSE"
                ),
                {
                    "uid": self.user_id, "name": name[:255], "cat": category,
                    "amount": round(float(amount), 2), "interval": period.days,
                    "next_due": next_due,
                },
            )

    def dismiss(self, payment_id: int) -> None:
        with self._engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE recurring_payments SET dismissed = TRUE "
                    "WHERE id = :id AND user_id = :uid"
                ),
                {"id": payment_id, "uid": self.user_id},
            )

class MySqlBudgetRepository(BudgetRepository):
    """One user's monthly limits in MySQL."""

    def __init__(self, user_id: int, engine: Engine | None = None) -> None:
        self.user_id = user_id
        self._engine = engine or get_engine()

    def load(self) -> dict[str, float]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text('SELECT category, monthly_limit FROM budgets WHERE user_id=:uid;'),
                {'uid': self.user_id}
            ).all()

        return { category: float(limit) for category, limit in rows }

    def save(self, budgets: dict[str, float]) -> None:
        records = [
            { 'uid': self.user_id, 'cat': category, 'limit': round(float(value), 2) }
            for category, value in budgets.items()
        ]

        if not records:
            return

        with self._engine.begin() as conn:
            conn.execute(
                text("INSERT INTO budgets (user_id, category, monthly_limit) "
                     "VALUES (:uid, :cat, :limit) "
                     "ON DUPLICATE KEY UPDATE monthly_limit = VALUES(monthly_limit)" 
                    ),
                    records
            )

