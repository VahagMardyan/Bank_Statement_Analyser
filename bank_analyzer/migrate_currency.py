"""Adds the currency columns to a database created before currency support.

Run once against each database (local MySQL, then TiDB with its .env values):

    python migrate_currency.py

Running it again does nothing. New databases made from db/schema.sql already
have the columns.
"""

from __future__ import annotations

from sqlalchemy import text

from modules.db import get_engine

# Separate ALTERs, one column each: TiDB handles that everywhere.
COLUMNS = {
    "original_amount": "DECIMAL(14, 2) NULL AFTER amount",
    "original_currency": "CHAR(3) NOT NULL DEFAULT 'AMD' AFTER original_amount",
}


def main() -> None:
    engine = get_engine()
    with engine.begin() as conn:
        existing = set(
            conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
                    "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'transactions'"
                )
            ).scalars()
        )
        if not existing:
            raise SystemExit("Table 'transactions' not found. Check DB_NAME in .env.")
        for name, ddl in COLUMNS.items():
            if name in existing:
                print(f"{name}: already there")
                continue
            conn.execute(text(f"ALTER TABLE transactions ADD COLUMN {name} {ddl}"))
            print(f"{name}: added")
        filled = conn.execute(
            text("UPDATE transactions SET original_amount = amount WHERE original_amount IS NULL")
        ).rowcount
        print(f"Old rows filled with their AMD amount: {filled}")


if __name__ == "__main__":
    main()
