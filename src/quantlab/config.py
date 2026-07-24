"""Central configuration for QuantLab.

All configuration flows through Pydantic Settings — there are no hardcoded paths
or credentials anywhere else in the codebase. The canonical data-split boundaries
(including the locked ``HOLDOUT_START``) are defined here exactly once.

See ``CLAUDE.md`` §3 (Research Integrity) and ``docs/PRD.md`` §4.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Fixed research-integrity boundaries (do not revisit; see CLAUDE.md §3) ---

#: Development history begins here (PRD §2.1: minimum 10 years).
DATA_START: date = date(2015, 1, 1)
#: Train split: 2015-01-01 → 2021-12-31.
TRAIN_END: date = date(2021, 12, 31)
#: Validation split: 2022-01-01 → 2023-12-31.
VALIDATION_END: date = date(2023, 12, 31)
#: Holdout is LOCKED from this date onward. Evaluated exactly once, at project end.
#: The DataHandler refuses to serve data at/past this date without an explicit
#: ``holdout=True`` and the ``QUANTLAB_UNLOCK_HOLDOUT=1`` environment variable.
HOLDOUT_START: date = date(2024, 1, 1)


class Settings(BaseSettings):
    """Runtime settings, populated from environment variables and ``.env``.

    Prefix every environment variable with ``QUANTLAB_`` (e.g.
    ``QUANTLAB_DATABASE_URL``).
    """

    model_config = SettingsConfigDict(
        env_prefix="QUANTLAB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    data_dir: Path = Field(default=Path("data"))
    raw_dir: Path = Field(default=Path("data/raw"))
    features_dir: Path = Field(default=Path("data/features"))
    database_url: str = Field(default="sqlite:///quantlab.db")

    # Data-provider credentials (blank by default; set locally, never commit)
    alpha_vantage_api_key: str | None = Field(default=None)
    polygon_api_key: str | None = Field(default=None)

    # Research-integrity flags
    #: Deliberately absent from ``.env.example``. Must be explicitly set to unlock
    #: holdout data, and even then only in combination with ``holdout=True``.
    unlock_holdout: bool = Field(default=False, alias="QUANTLAB_UNLOCK_HOLDOUT")

    # Observability
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
