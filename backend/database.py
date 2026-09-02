"""Database engine and session setup (SQLAlchemy 2.0 style)."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import settings
from backend.models import Base

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


@event.listens_for(Engine, "connect")
def _enable_sqlite_fk(dbapi_conn, _record) -> None:
    """Enforce foreign-key constraints in SQLite (off by default)."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a scoped session."""
    with SessionLocal() as session:
        yield session
