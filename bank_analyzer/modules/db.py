"""Database connection: one shared engine for the whole app."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, URL

load_dotenv()

def _build_url() -> URL:
    return URL.create(
        drivername='mysql+pymysql',
        username=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '3306')),
        database=os.environ['DB_NAME'],
        query={"charset": 'utf8mb4'},
    )

def _connect_args() -> dict:
    """Encrypted connection for hosted databases (TiDB Cloud requires it).

    DB_SSL=true turns it on; the server certificate is checked against
    DB_SSL_CA if given, otherwise against the certifi bundle.
    """
    if os.getenv("DB_SSL", "").strip().lower() not in ("1", "true", "yes"):
        return {}
    ca = os.getenv("DB_SSL_CA")
    if not ca:
        import certifi
        ca = certifi.where()
    return {"ssl": {"ca": ca}}


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        _build_url(),
        connect_args=_connect_args(),
        pool_pre_ping=True,
        pool_recycle=300,  # hosted servers drop idle connections sooner than a local MySQL
    )

if __name__ == "__main__":
    with get_engine().connect() as conn:
        tables = conn.execute(text("SHOW TABLES")).scalars().all()
        print("Connected. Tables:", tables)

