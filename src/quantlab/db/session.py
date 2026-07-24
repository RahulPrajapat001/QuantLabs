"""Engine and session construction.

The database URL comes from :class:`quantlab.config.Settings` (SQLite in
development, PostgreSQL in production) — never hardcoded. ``create_all`` exists
for tests and first-run bootstrapping; production schema changes go through
Alembic migrations.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from quantlab.config import get_settings

from .models import Base


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create an :class:`~sqlalchemy.Engine` for ``database_url``.

    Defaults to the configured ``QUANTLAB_DATABASE_URL``. Enforces SQLite
    foreign keys (off by default in SQLite) so ``ON DELETE CASCADE`` behaves the
    same on dev SQLite and prod PostgreSQL.
    """

    url = database_url or get_settings().database_url
    engine = create_engine(url, future=True)

    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_connection, _record) -> None:  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a configured :class:`~sqlalchemy.orm.sessionmaker`."""

    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def create_all(engine: Engine) -> None:
    """Create every table on ``engine`` (test / bootstrap convenience)."""

    Base.metadata.create_all(engine)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional session context: commit on success, roll back on error."""

    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
