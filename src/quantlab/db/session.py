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
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from quantlab.config import get_settings

from .models import Base


def create_db_engine(database_url: str | None = None) -> Engine:
    """Create an :class:`~sqlalchemy.Engine` for ``database_url``.

    Defaults to the configured ``QUANTLAB_DATABASE_URL``. Both backends are
    supported deliberately: SQLite for local development (zero setup, one file
    to delete) and PostgreSQL for production (M4). The per-dialect tuning below
    is the only place the difference is visible — every model, query and
    migration is written once against SQLAlchemy Core.

    * **SQLite** — foreign keys are *off* by default, so ``ON DELETE CASCADE``
      would silently not cascade. The pragma makes dev behave like prod.
    * **PostgreSQL** — ``pool_pre_ping`` because a pooled connection that a
      container restart or an idle timeout has already closed otherwise surfaces
      as a mid-request ``OperationalError``. ``pool_recycle`` keeps connections
      under the typical proxy idle limit.
    """

    url = database_url or get_settings().database_url
    dialect = make_url(url).get_backend_name()

    if dialect.startswith("postgresql"):
        engine = create_engine(url, future=True, pool_pre_ping=True, pool_recycle=1800)
    else:
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
