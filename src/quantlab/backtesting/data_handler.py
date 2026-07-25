"""``DataHandler`` — streams bars chronologically and enforces the holdout lock.

Two failure modes are structurally excluded here rather than tested for later.

**Look-ahead.** The full price panel is private. Every accessor slices at the
cursor, so there is no method on this object that returns a bar the strategy has
not yet been shown. A vectorised backtest computes a signal from a column that
already contains tomorrow's price and the bug is invisible; here, asking for
tomorrow raises.

**Holdout leakage.** The handler refuses to serve data at or past
:data:`~quantlab.config.HOLDOUT_START` unless it was constructed with
``holdout=True`` *and* ``QUANTLAB_UNLOCK_HOLDOUT=1`` is set in the environment.
That variable is deliberately absent from ``.env.example``: unlocking the holdout
requires a conscious act in a shell, not a config file edit. Two independent
switches means neither a stray environment variable nor a copy-pasted keyword
argument unlocks it alone.

The lock raises rather than silently clipping, because a caller who did not
realise their window crossed 2024-01-01 needs to find out. ``truncate_to_lock=True``
is the explicit opt-in for clipping instead — safe by construction, since it can
only ever remove holdout data.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import structlog

from quantlab.config import (
    HOLDOUT_START,
    HoldoutViolationError,
    assert_holdout_allowed,
    resolve_split,
)

log = structlog.get_logger(__name__)

#: Columns a bar panel must provide. ``adj_close`` and ``close`` are both
#: required and kept separate: signals use adjusted prices, fills and costs use
#: the raw traded price.
REQUIRED_COLUMNS = (
    "date",
    "security_id",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)


class LookAheadError(RuntimeError):
    """Raised when data beyond the current bar is requested."""


@dataclass(frozen=True, slots=True)
class BarBatch:
    """Every security's bar for one date — the unit the event loop advances on.

    Cross-sectional strategies need the whole cross-section of a date at once, so
    the stream yields one batch per date rather than one bar per security.
    """

    date: date
    bars: pd.DataFrame

    @property
    def security_ids(self) -> list[str]:
        return [str(s) for s in self.bars["security_id"]]

    def __len__(self) -> int:
        return len(self.bars)


class DataHandler:
    """Chronological bar stream with look-ahead and holdout guards.

    Parameters
    ----------
    bars:
        Tidy panel with :data:`REQUIRED_COLUMNS`. Copied and sorted on
        construction; the caller's frame is never mutated, and mutating it
        afterwards cannot affect a run in flight.
    start, end:
        Inclusive window. ``end`` defaults to the last date in ``bars`` — and
        note that defaulting does not soften the lock: an unbounded window over a
        panel that contains 2024 data is still a holdout access.
    holdout:
        Explicit opt-in to holdout data. Honoured only with
        ``QUANTLAB_UNLOCK_HOLDOUT=1`` also set.
    truncate_to_lock:
        Clip the window at the day before :data:`HOLDOUT_START` instead of
        raising. Only ever removes data, never adds it.
    """

    def __init__(
        self,
        bars: pd.DataFrame,
        *,
        start: date | None = None,
        end: date | None = None,
        holdout: bool = False,
        truncate_to_lock: bool = False,
    ) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in bars.columns]
        if missing:
            raise ValueError(
                f"bar panel is missing required column(s): {', '.join(missing)}"
            )
        if bars.empty:
            raise ValueError("bar panel is empty; there is nothing to stream")

        frame = bars.loc[:, list(REQUIRED_COLUMNS)].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["security_id"] = frame["security_id"].astype(str)

        data_start = frame["date"].min().date()
        data_end = frame["date"].max().date()
        requested_start = start or data_start
        requested_end = end or data_end

        if requested_end < requested_start:
            raise ValueError(
                f"end {requested_end.isoformat()} precedes start "
                f"{requested_start.isoformat()}"
            )

        self._holdout = holdout
        if (
            truncate_to_lock
            and requested_end >= HOLDOUT_START
            and not (holdout and _unlocked())
        ):
            requested_end = HOLDOUT_START - timedelta(days=1)
            log.warning(
                "data_handler.truncated_to_holdout_lock",
                requested_end=str(end or data_end),
                effective_end=requested_end.isoformat(),
            )
            if requested_end < requested_start:
                raise HoldoutViolationError(
                    f"the requested window [{requested_start.isoformat()}, "
                    f"{(end or data_end).isoformat()}] lies entirely inside the "
                    f"locked holdout (from {HOLDOUT_START.isoformat()}); there is "
                    "nothing left to serve after truncation"
                )

        # The lock. Checked against the *requested* window, before any data is
        # exposed — not lazily when a holdout bar happens to be reached.
        assert_holdout_allowed(
            requested_end,
            holdout=holdout,
            context=(
                f"DataHandler window [{requested_start.isoformat()}, "
                f"{requested_end.isoformat()}]"
            ),
        )

        mask = (frame["date"] >= pd.Timestamp(requested_start)) & (
            frame["date"] <= pd.Timestamp(requested_end)
        )
        frame = frame.loc[mask]
        if frame.empty:
            raise ValueError(
                f"no bars in the window [{requested_start.isoformat()}, "
                f"{requested_end.isoformat()}]"
            )

        self._frame = frame.sort_values(
            ["date", "security_id"], kind="stable"
        ).reset_index(drop=True)
        self._dates: list[date] = [
            ts.date() for ts in self._frame["date"].drop_duplicates()
        ]
        # Row offsets per date, so advancing the cursor is a slice, not a filter.
        self._offsets = (
            self._frame["date"].searchsorted(
                pd.to_datetime(pd.Series(self._dates)), side="left"
            )
        ).tolist()
        self._offsets.append(len(self._frame))

        self._start = requested_start
        self._end = requested_end
        self._cursor = -1
        self._served_holdout = False

        if holdout:
            log.warning(
                "data_handler.holdout_unlocked",
                start=self._start.isoformat(),
                end=self._end.isoformat(),
                message=(
                    "serving LOCKED holdout data — this evaluation happens exactly "
                    "once and every run is recorded in research_log with its git "
                    "commit"
                ),
            )

    # --- Window metadata ---

    @property
    def start(self) -> date:
        return self._start

    @property
    def end(self) -> date:
        return self._end

    @property
    def split(self) -> str:
        """The split label for this window, for the research log."""

        return resolve_split(self._start, self._end)

    @property
    def holdout(self) -> bool:
        """Whether this handler was constructed with the holdout opt-in."""

        return self._holdout

    @property
    def holdout_touched(self) -> bool:
        """Whether a bar at or past :data:`HOLDOUT_START` has actually been served.

        Distinct from :attr:`holdout`: a handler may be unlocked and still never
        reach a holdout bar. The research log records what was *touched*.
        """

        return self._served_holdout

    @property
    def dates(self) -> Sequence[date]:
        """Every date in the window, in order. Dates, not prices — no leakage."""

        return tuple(self._dates)

    @property
    def current_date(self) -> date | None:
        """The date of the most recently served batch, or ``None`` before streaming."""

        if self._cursor < 0:
            return None
        return self._dates[self._cursor]

    @property
    def finished(self) -> bool:
        return self._cursor >= len(self._dates) - 1

    def __len__(self) -> int:
        return len(self._dates)

    # --- Streaming ---

    def stream(self) -> Iterator[BarBatch]:
        """Yield one :class:`BarBatch` per date, in order, advancing the cursor.

        The cursor moves *before* the batch is yielded, so during handling of date
        ``t`` the accessors below expose data through ``t`` inclusive and nothing
        after it. An order emitted here is filled at ``t+1`` open by the execution
        handler, which is why look-ahead has to be impossible rather than merely
        discouraged.
        """

        while not self.finished:
            self._cursor += 1
            current = self._dates[self._cursor]
            if current >= HOLDOUT_START:
                self._served_holdout = True
            yield BarBatch(date=current, bars=self._current_slice())

    def reset(self) -> None:
        """Rewind to before the first bar, so one handler can drive several runs."""

        self._cursor = -1
        self._served_holdout = False

    def _current_slice(self) -> pd.DataFrame:
        lo = self._offsets[self._cursor]
        hi = self._offsets[self._cursor + 1]
        return self._frame.iloc[lo:hi].reset_index(drop=True)

    def _visible(self) -> pd.DataFrame:
        """Every row up to and including the current bar."""

        if self._cursor < 0:
            return self._frame.iloc[0:0]
        return self._frame.iloc[: self._offsets[self._cursor + 1]]

    # --- Accessors (never past the cursor) ---

    def current_bars(self) -> pd.DataFrame:
        """The current date's cross-section. Empty before streaming starts."""

        if self._cursor < 0:
            return self._frame.iloc[0:0].reset_index(drop=True)
        return self._current_slice()

    def history(
        self,
        security_id: str | None = None,
        *,
        lookback: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Visible history, optionally for one security and a bounded lookback.

        ``lookback`` counts *dates*, not rows, so it means the same thing for a
        single security and for the whole cross-section.
        """

        visible = self._visible()
        if security_id is not None:
            visible = visible.loc[visible["security_id"] == security_id]
        if lookback is not None:
            if lookback < 1:
                raise ValueError(f"lookback must be at least 1, got {lookback}")
            first = max(0, self._cursor - lookback + 1)
            cutoff = pd.Timestamp(self._dates[first]) if self._dates else None
            if cutoff is not None:
                visible = visible.loc[visible["date"] >= cutoff]
        if columns is not None:
            keep = [
                "date",
                "security_id",
                *[c for c in columns if c not in ("date", "security_id")],
            ]
            visible = visible.loc[:, keep]
        return visible.reset_index(drop=True)

    def history_as_of(
        self, as_of: date, security_id: str | None = None
    ) -> pd.DataFrame:
        """History through ``as_of``, which must not be past the current bar.

        The explicit form for code that carries its own notion of "now": if the
        two disagree, that is a look-ahead bug and it raises here instead of
        quietly returning the future.
        """

        current = self.current_date
        if current is None:
            raise LookAheadError(
                "no bar has been served yet; history_as_of has nothing to return"
            )
        if as_of > current:
            raise LookAheadError(
                f"requested history as of {as_of.isoformat()} but the current bar is "
                f"{current.isoformat()}. The backtester cannot see past the bar it "
                "is processing — this is a look-ahead bug, not a missing feature."
            )
        visible = self._visible()
        visible = visible.loc[visible["date"] <= pd.Timestamp(as_of)]
        if security_id is not None:
            visible = visible.loc[visible["security_id"] == security_id]
        return visible.reset_index(drop=True)

    def latest_price(self, security_id: str, *, column: str = "close") -> float | None:
        """The most recent visible price for one security, or ``None``."""

        if column not in REQUIRED_COLUMNS:
            raise ValueError(f"unknown price column {column!r}")
        visible = self._visible()
        rows = visible.loc[visible["security_id"] == security_id, column]
        if rows.empty:
            return None
        return float(rows.iloc[-1])


def _unlocked() -> bool:
    from quantlab.config import holdout_is_unlocked

    return holdout_is_unlocked()
