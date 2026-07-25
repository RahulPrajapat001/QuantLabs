"""Holdout enforcement for factor evaluation (PRD Module 6, CLAUDE.md §3).

Factor evaluation runs on the **train split**. The holdout period (2024-01-01
onward) is locked: no result computed on it may inform a factor decision. This
guard raises if evaluation is handed data at or past ``HOLDOUT_START`` unless it
is *explicitly* unlocked — mirroring the ``DataHandler`` rule, an ``allow_holdout``
flag is not enough on its own; the ``QUANTLAB_UNLOCK_HOLDOUT`` environment
variable must also be set (it is deliberately absent from ``.env.example``).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pandas as pd

from quantlab.config import HOLDOUT_START, HoldoutViolationError, holdout_is_unlocked

__all__ = ["HoldoutViolationError", "guard_holdout"]


def _to_date(value: object) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def guard_holdout(dates: Iterable[object], *, allow_holdout: bool = False) -> None:
    """Raise :class:`HoldoutViolationError` if any date is at/past ``HOLDOUT_START``.

    Passing ``allow_holdout=True`` is honoured **only** when
    ``QUANTLAB_UNLOCK_HOLDOUT=1`` is also set in the environment; otherwise the
    holdout stays locked and this still raises.
    """

    parsed = [_to_date(d) for d in dates]
    if not parsed:
        return
    latest = max(parsed)
    if latest < HOLDOUT_START:
        return

    if allow_holdout and holdout_is_unlocked():
        return

    reason = (
        "requires allow_holdout=True and QUANTLAB_UNLOCK_HOLDOUT=1"
        if allow_holdout
        else "the holdout is locked (allow_holdout not set)"
    )
    raise HoldoutViolationError(
        f"factor evaluation received data through {latest.isoformat()}, at/past "
        f"the locked holdout start {HOLDOUT_START.isoformat()}; {reason}. "
        "No result computed on holdout data may inform a factor decision."
    )
