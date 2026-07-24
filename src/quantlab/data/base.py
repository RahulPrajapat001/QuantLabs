"""Provider abstraction and the canonical OHLCV schema.

Every vendor is reached through :class:`DataProvider`. No other module imports a
vendor SDK — that boundary is the whole point of the abstraction (see
``CLAUDE.md`` §2 and ``docs/PRD.md`` §3).

The public :meth:`DataProvider.fetch_ohlcv` is a *template method*: it delegates
the actual download to a subclass hook and then validates the result against
:data:`OHLCV_SCHEMA` before returning. A provider therefore cannot return a
frame that violates the schema — the validation is centralised, not left to each
subclass to remember.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError

from .errors import PermanentProviderError


class Frequency(StrEnum):
    """Bar frequency. Only daily is in scope for research (PRD §2)."""

    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"


class CorporateActionType(StrEnum):
    """The corporate-action kinds we ingest (PRD Module 14)."""

    SPLIT = "split"
    DIVIDEND = "dividend"
    BONUS = "bonus"


@dataclass(frozen=True, slots=True)
class Security:
    """A security as seen by a provider.

    ``security_id`` is QuantLab's stable internal identity (never a ticker);
    ``symbol`` is the vendor-specific string used to query the provider
    (e.g. ``RELIANCE.NS``). Providers query by :attr:`symbol`; the rest of the
    platform joins on :attr:`security_id`.
    """

    security_id: str
    symbol: str
    exchange: str | None = None


@dataclass(frozen=True, slots=True)
class DelistingInfo:
    """Delisting status for a security.

    A delisted security keeps its full price history and a final price; it is
    never silently dropped (PRD Module 1).
    """

    delisted: bool
    delisting_date: date | None = None
    reason: str | None = None


# --- Canonical OHLCV schema --------------------------------------------------
#
# The single frame shape every provider must satisfy. Columns are ordered and
# strict (no extras): the ingest layer stamps ``security_id``, ``source`` and
# ``as_of`` afterwards — those are storage concerns, not provider output.

OHLCV_COLUMNS: tuple[str, ...] = (
    "date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


def _price(*, name: str) -> pa.Column:
    # Strictly positive: a non-positive traded price is a data fault, and later
    # log-return maths requires > 0. Outlier / circuit checks live in the
    # validation module, not here — this schema is structural only.
    return pa.Column(
        float,
        checks=pa.Check.gt(0, error=f"{name} must be > 0"),
        nullable=False,
        coerce=True,
    )


OHLCV_SCHEMA = pa.DataFrameSchema(
    columns={
        "date": pa.Column(
            "datetime64[ns]",
            nullable=False,
            unique=True,
            coerce=True,
            title="session date",
        ),
        "open": _price(name="open"),
        "high": _price(name="high"),
        "low": _price(name="low"),
        "close": _price(name="close"),
        "adj_close": _price(name="adj_close"),
        # Volume may legitimately be zero (a halted / no-trade session) but never
        # negative or fractional-with-loss; coerce to float to tolerate NaN-free
        # integer or float vendor payloads uniformly.
        "volume": pa.Column(
            float,
            checks=pa.Check.ge(0, error="volume must be >= 0"),
            nullable=False,
            coerce=True,
        ),
    },
    checks=[
        pa.Check(
            lambda df: df["high"] >= df["low"],
            error="high must be >= low",
        ),
        pa.Check(
            lambda df: df["high"] >= df[["open", "close"]].max(axis=1),
            error="high must be >= max(open, close)",
        ),
        pa.Check(
            lambda df: df["low"] <= df[["open", "close"]].min(axis=1),
            error="low must be <= min(open, close)",
        ),
    ],
    strict=True,
    ordered=True,
    coerce=True,
    name="ohlcv",
)


CORPORATE_ACTION_SCHEMA = pa.DataFrameSchema(
    columns={
        "date": pa.Column("datetime64[ns]", nullable=False, coerce=True),
        "action_type": pa.Column(
            str,
            checks=pa.Check.isin([a.value for a in CorporateActionType]),
            nullable=False,
        ),
        # Split ratio (e.g. 2.0 for a 2:1) or per-share cash amount for a
        # dividend. Always positive.
        "value": pa.Column(
            float,
            checks=pa.Check.gt(0, error="corporate-action value must be > 0"),
            nullable=False,
            coerce=True,
        ),
    },
    strict=True,
    ordered=True,
    coerce=True,
    name="corporate_actions",
)


def validate_ohlcv(df: pd.DataFrame, *, source: str, symbol: str) -> pd.DataFrame:
    """Validate and normalise a provider OHLCV frame.

    Returns a fresh frame with canonical column order, sorted by date. A schema
    violation is a data fault, not a network hiccup, so it is raised as a
    :class:`PermanentProviderError` — retrying the same request cannot fix it.
    """

    try:
        validated = OHLCV_SCHEMA.validate(df, lazy=True)
    except SchemaError as exc:  # pragma: no cover - lazy raises SchemaErrors
        raise PermanentProviderError(
            f"OHLCV schema violation: {exc}", source=source, symbol=symbol
        ) from exc
    except Exception as exc:  # pandera SchemaErrors (plural) under lazy=True
        raise PermanentProviderError(
            f"OHLCV schema violation: {exc}", source=source, symbol=symbol
        ) from exc
    return (
        validated.loc[:, list(OHLCV_COLUMNS)].sort_values("date").reset_index(drop=True)
    )


def validate_corporate_actions(
    df: pd.DataFrame, *, source: str, symbol: str
) -> pd.DataFrame:
    """Validate a corporate-actions frame, raising a typed error on violation."""

    try:
        validated = CORPORATE_ACTION_SCHEMA.validate(df, lazy=True)
    except Exception as exc:
        raise PermanentProviderError(
            f"corporate-action schema violation: {exc}",
            source=source,
            symbol=symbol,
        ) from exc
    return validated.sort_values("date").reset_index(drop=True)


class DataProvider(ABC):
    """Abstract base for every market-data vendor.

    Subclasses implement the ``_fetch_*`` hooks and never expose vendor types.
    The public ``fetch_*`` methods validate on the way out so that the rest of
    the platform only ever sees schema-conforming frames.
    """

    #: Short, stable identifier used in raw-parquet paths and the ``source``
    #: column (e.g. ``"yfinance"``). Set by each subclass.
    source: str

    # -- public template methods ---------------------------------------------

    def fetch_ohlcv(
        self,
        security: Security,
        start: date,
        end: date,
        frequency: Frequency = Frequency.DAILY,
    ) -> pd.DataFrame:
        """Return validated OHLCV for ``security`` over ``[start, end]``.

        The frame conforms to :data:`OHLCV_SCHEMA`. Empty ranges return an
        empty, correctly-typed frame rather than raising.
        """

        raw = self._fetch_ohlcv(security, start, end, frequency)
        return validate_ohlcv(raw, source=self.source, symbol=security.symbol)

    def fetch_corporate_actions(self, security: Security) -> pd.DataFrame:
        """Return validated corporate actions (splits, dividends, bonuses)."""

        raw = self._fetch_corporate_actions(security)
        return validate_corporate_actions(
            raw, source=self.source, symbol=security.symbol
        )

    def fetch_delisting_info(self, security: Security) -> DelistingInfo:
        """Return delisting status; ``delisted=False`` for a live security."""

        return self._fetch_delisting_info(security)

    # -- subclass hooks ------------------------------------------------------

    @abstractmethod
    def _fetch_ohlcv(
        self,
        security: Security,
        start: date,
        end: date,
        frequency: Frequency,
    ) -> pd.DataFrame:
        """Download raw OHLCV. May raise :class:`ProviderError` subclasses."""

    @abstractmethod
    def _fetch_corporate_actions(self, security: Security) -> pd.DataFrame:
        """Download raw corporate actions."""

    @abstractmethod
    def _fetch_delisting_info(self, security: Security) -> DelistingInfo:
        """Determine delisting status."""


def empty_ohlcv() -> pd.DataFrame:
    """An empty, correctly-typed OHLCV frame (for empty ranges)."""

    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "open": pd.Series([], dtype=float),
            "high": pd.Series([], dtype=float),
            "low": pd.Series([], dtype=float),
            "close": pd.Series([], dtype=float),
            "adj_close": pd.Series([], dtype=float),
            "volume": pd.Series([], dtype=float),
        }
    )


def empty_corporate_actions() -> pd.DataFrame:
    """An empty, correctly-typed corporate-actions frame."""

    return pd.DataFrame(
        {
            "date": pd.Series([], dtype="datetime64[ns]"),
            "action_type": pd.Series([], dtype=str),
            "value": pd.Series([], dtype=float),
        }
    )
