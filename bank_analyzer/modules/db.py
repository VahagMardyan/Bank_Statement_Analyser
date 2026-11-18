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

@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        _build_url(),
        pool_pre_ping=True,
        pool_recycle=3600
    )

if __name__ == "__main__":
    with get_engine().connect() as conn:
        tables = conn.execute(text("SHOW TABLES")).scalars().all()
        print("Connected. Tables:", tables)

