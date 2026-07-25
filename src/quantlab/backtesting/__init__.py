"""Event-driven backtesting (PRD Module 8).

The components land across M2. Present here:

* :class:`~quantlab.backtesting.data_handler.DataHandler` — chronological bar
  stream that structurally prevents access beyond the current bar and enforces the
  holdout lockout.
* :class:`~quantlab.backtesting.logged.LoggedBacktestEngine` — the base every
  engine inherits, whose ``run()`` is final and writes to ``research_log``.
"""

from quantlab.backtesting.data_handler import (
    REQUIRED_COLUMNS,
    BarBatch,
    DataHandler,
    LookAheadError,
)
from quantlab.backtesting.logged import LoggedBacktest, LoggedBacktestEngine

__all__ = [
    "REQUIRED_COLUMNS",
    "BarBatch",
    "DataHandler",
    "LoggedBacktest",
    "LoggedBacktestEngine",
    "LookAheadError",
]
