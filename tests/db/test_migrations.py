"""Alembic migrations must run clean from empty on **both** backends.

SQLite is exercised for real: a temporary file database is migrated from nothing
to ``head``, and the resulting schema is compared against the ORM metadata, so a
model added without a migration fails here rather than at deploy time.

PostgreSQL is exercised in *offline* mode — Alembic renders the full DDL against
the PostgreSQL dialect without a server. That is deliberate rather than a
shortcut: it runs in any CI job with no service container, and it catches the
failure mode that actually bites, which is a migration containing SQLite-only
constructs (``AUTOINCREMENT``, a bare ``DATETIME``, a pragma) that no amount of
SQLite testing would reveal. The compose stack runs the real thing against a real
Postgres on startup.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from quantlab.db.models import Base

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Constructs that are valid on SQLite and wrong (or absent) on PostgreSQL.
SQLITE_ONLY = re.compile(r"\bAUTOINCREMENT\b|\bPRAGMA\b", re.IGNORECASE)


@pytest.fixture
def alembic_at(monkeypatch: pytest.MonkeyPatch):
    """Build an Alembic config pointed at ``url`` — via the environment.

    ``migrations/env.py`` deliberately overrides ``sqlalchemy.url`` with
    ``get_settings().database_url`` so the URL has exactly one source of truth.
    That means ``config.set_main_option("sqlalchemy.url", ...)`` is silently
    ignored, and a test that used it would migrate — and on downgrade, *drop* —
    whatever database the developer has configured. Setting the environment
    variable and clearing the settings cache is the only way to redirect it.
    """

    from quantlab.config import get_settings

    def build(url: str) -> Config:
        monkeypatch.setenv("QUANTLAB_DATABASE_URL", url)
        get_settings.cache_clear()
        config = Config(str(REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", "src/quantlab/db/migrations")
        return config

    yield build
    get_settings.cache_clear()


def test_migrations_run_clean_from_empty_on_sqlite(tmp_path: Path, alembic_at) -> None:
    """Empty file -> head, then the schema matches the ORM metadata exactly."""

    db = tmp_path / "migrated.db"
    url = f"sqlite:///{db.as_posix()}"

    command.upgrade(alembic_at(url), "head")

    engine = create_engine(url)
    try:
        actual = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()

    expected = set(Base.metadata.tables)
    assert actual == expected, (
        "migrated schema differs from the models — a model was added or renamed "
        f"without a migration.\nonly in DB: {sorted(actual - expected)}\n"
        f"only in models: {sorted(expected - actual)}"
    )


def test_migrations_render_for_postgres(
    alembic_at, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same migrations emit valid PostgreSQL DDL, with no SQLite-isms."""

    command.upgrade(
        alembic_at("postgresql+psycopg://quantlab:quantlab@localhost:5432/quantlab"),
        "head",
        sql=True,  # offline: render DDL, never connect
    )
    sql = capsys.readouterr().out

    for table in Base.metadata.tables:
        assert f"CREATE TABLE {table}" in sql, f"{table} missing from PostgreSQL DDL"

    # Autoincrement primary keys must render as SERIAL, and tz-aware columns as
    # TIMESTAMP WITH TIME ZONE — the two places a SQLite-shaped migration
    # silently produces the wrong Postgres column.
    assert "SERIAL" in sql
    assert "TIMESTAMP WITH TIME ZONE" in sql

    leaked = SQLITE_ONLY.findall(sql)
    assert not leaked, f"SQLite-only DDL leaked into the PostgreSQL script: {leaked}"


def test_downgrade_returns_to_empty_on_sqlite(tmp_path: Path, alembic_at) -> None:
    """`base` leaves nothing behind, so a migration is reversible in dev."""

    db = tmp_path / "reversible.db"
    url = f"sqlite:///{db.as_posix()}"
    config = alembic_at(url)

    command.upgrade(config, "head")
    command.downgrade(config, "base")

    engine = create_engine(url)
    try:
        remaining = set(inspect(engine).get_table_names()) - {"alembic_version"}
    finally:
        engine.dispose()

    assert remaining == set(), f"downgrade left tables behind: {sorted(remaining)}"
