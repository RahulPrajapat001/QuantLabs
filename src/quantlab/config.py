"""Central configuration for QuantLab.

All configuration flows through Pydantic Settings — there are no hardcoded paths
or credentials anywhere else in the codebase. The canonical data-split boundaries
(including the locked ``HOLDOUT_START``) are defined here exactly once.

See ``CLAUDE.md`` §3 (Research Integrity) and ``docs/PRD.md`` §4.
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
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
    #: Cached primary sources for universe construction — reconstitution PDFs
    #: (NSE) and changes-table HTML (S&P 500) — so results are reproducible
    #: without re-downloading (PRD Module 2).
    reference_dir: Path = Field(default=Path("data/raw/reference"))
    features_dir: Path = Field(default=Path("data/features"))
    database_url: str = Field(default="sqlite:///quantlab.db")
    #: Pre-registration document (PRD §4.5). Resolved relative to the working
    #: directory, then to the git repository root, so the CLI works from any
    #: subdirectory. Never referenced as a literal at a call site.
    hypotheses_file: Path = Field(default=Path("docs/hypotheses.md"))

    # Universe / backtest
    #: Recovery rate applied when a security is delisted for bankruptcy: the
    #: liquidation receives ``last_price * recovery_rate``. Default 0 — a
    #: bankruptcy is assumed a total loss unless configured otherwise
    #: (PRD Module 2, "Delisting in backtest").
    bankruptcy_recovery_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Alpha research (PRD Module 6)
    #: A factor must clear this absolute mean information coefficient AND the
    #: t-stat gate below before it is eligible for the strategy builder. Enforced
    #: in code (see ``quantlab.factors.gate``), not documentation.
    factor_min_abs_ic: float = Field(default=0.02, ge=0.0)
    #: Minimum absolute Newey-West t-statistic on the mean IC for eligibility.
    factor_min_ic_tstat: float = Field(default=2.0, ge=0.0)

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


# --- Holdout lockout (CLAUDE.md §3, PRD §4.2) --------------------------------
#
# The lock lives here, beside ``HOLDOUT_START``, so that every enforcement point
# — the backtester's ``DataHandler``, factor evaluation, the research log — shares
# one error type and one definition of "unlocked". Two guards that disagree about
# what is locked are worse than no guard at all.


class HoldoutViolationError(RuntimeError):
    """Raised when locked holdout data (2024-01-01 onward) is touched.

    Shared by every enforcement point so ``except HoldoutViolationError`` catches
    the violation wherever it originates.
    """


def holdout_is_unlocked() -> bool:
    """Whether ``QUANTLAB_UNLOCK_HOLDOUT=1`` is set in the environment.

    This is necessary but never sufficient: each call site must *also* be handed
    an explicit opt-in (``holdout=True`` / ``allow_holdout=True``). Requiring two
    independent switches means neither a stray environment variable nor a
    copy-pasted keyword argument can unlock the holdout on its own.
    """

    return get_settings().unlock_holdout


def assert_holdout_allowed(
    latest: date,
    *,
    holdout: bool,
    context: str = "this operation",
) -> None:
    """Raise :class:`HoldoutViolationError` unless ``latest`` is legitimately allowed.

    ``latest`` is the most recent date the caller intends to touch. Data strictly
    before :data:`HOLDOUT_START` always passes. At or past it, the caller must
    have passed ``holdout=True`` *and* set ``QUANTLAB_UNLOCK_HOLDOUT=1``.
    """

    if latest < HOLDOUT_START:
        return
    if holdout and holdout_is_unlocked():
        return

    reason = (
        "the explicit holdout opt-in was given but QUANTLAB_UNLOCK_HOLDOUT=1 is "
        "not set in the environment"
        if holdout
        else "the holdout is locked and no explicit holdout opt-in was given"
    )
    raise HoldoutViolationError(
        f"{context} reaches {latest.isoformat()}, at/past the locked holdout start "
        f"{HOLDOUT_START.isoformat()}: {reason}. The holdout is evaluated exactly "
        "once, at the end of the project; no result computed on it may inform any "
        "strategy, parameter or feature decision (CLAUDE.md §3)."
    )


class Split(StrEnum):
    """A canonical data split (PRD §4.1)."""

    TRAIN = "train"
    VALIDATION = "validation"
    HOLDOUT = "holdout"


#: Half-open bounds ``[start, end)`` of each split, in chronological order.
_SPLIT_BOUNDS: tuple[tuple[Split, date, date | None], ...] = (
    (Split.TRAIN, DATA_START, date(TRAIN_END.year, 12, 31)),
    (Split.VALIDATION, date(TRAIN_END.year + 1, 1, 1), VALIDATION_END),
    (Split.HOLDOUT, HOLDOUT_START, None),
)


def splits_touched(start: date, end: date) -> tuple[Split, ...]:
    """Return every split the inclusive range ``[start, end]`` overlaps."""

    if end < start:
        raise ValueError(f"end {end.isoformat()} precedes start {start.isoformat()}")
    touched = [
        split
        for split, lo, hi in _SPLIT_BOUNDS
        if start <= (hi if hi is not None else end) and end >= lo
    ]
    return tuple(touched)


def resolve_split(start: date, end: date) -> str:
    """Label the split(s) covered by ``[start, end]`` for the research log.

    A range confined to one split gets that split's name; one that straddles
    several is labelled by every split it touches (e.g. ``"train+validation"``),
    because a run whose window silently spans a boundary must not be filed under
    the cleaner of the two. Data before :data:`DATA_START` is ``"pre-sample"``.
    """

    touched = splits_touched(start, end)
    if not touched:
        return "pre-sample"
    label = "+".join(split.value for split in touched)
    if start < DATA_START:
        return f"pre-sample+{label}"
    return label


def touches_holdout(end: date) -> bool:
    """Whether a run ending on ``end`` reaches into the locked holdout."""

    return end >= HOLDOUT_START
