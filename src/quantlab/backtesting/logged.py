"""The base every backtest engine inherits: running implies logging.

CLAUDE.md §3 requires that a backtest write to ``research_log`` "from inside the
engine, not via an omittable decorator". A decorator is omittable; so is a call at
the top of a method that someone later refactors. This base class makes the log
write part of the control flow a subclass cannot reach around:

* :meth:`LoggedBacktestEngine.run` is :func:`~typing.final` and does the logging
  itself. A subclass implements :meth:`execute`, which is called *inside* the log
  context manager.
* ``__init_subclass__`` rejects at class-creation time any subclass that defines
  its own ``run`` — mypy catches the override statically, this catches it at
  runtime, including in code that never gets type-checked.
* :meth:`run` returns a :class:`LoggedBacktest`, whose performance is reachable
  only as a :class:`~quantlab.research.reporting.SharpeReport` — a Sharpe bound to
  its trials count and deflation. There is no path from a run to a bare Sharpe.

The concrete event-driven engine (M2, PRD Module 8) subclasses this: its
``execute`` streams the :class:`~quantlab.backtesting.data_handler.DataHandler`,
dispatches signals to the portfolio, applies costs and builds the ledger. What it
cannot do is return a result without a row in ``research_log``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, final

import structlog

from quantlab.backtesting.data_handler import DataHandler
from quantlab.db.models import ResearchLogEntry
from quantlab.research.log import (
    LoggedRun,
    ResearchLog,
    ResearchLogError,
    RunResult,
    RunSpec,
)
from quantlab.research.reporting import SharpeReport

log = structlog.get_logger(__name__)


class LoggedBacktest:
    """A completed, logged backtest.

    Deliberately exposes no bare ``sharpe`` attribute. The Sharpe is available
    through :meth:`sharpe_report`, which reads the trials count from the log and
    deflates — so the number and its context cannot be separated by an accessor.
    """

    __slots__ = ("_log", "result", "run_id", "spec")

    def __init__(
        self,
        research_log: ResearchLog,
        run_id: str,
        spec: RunSpec,
        result: RunResult,
    ) -> None:
        self._log = research_log
        self.run_id = run_id
        self.spec = spec
        self.result = result

    @property
    def metrics(self) -> dict[str, Any]:
        """The run's output metrics as logged."""

        return dict(self.result.metrics)

    def entry(self) -> ResearchLogEntry:
        """The ``research_log`` row for this run."""

        return self._log.entry(self.run_id)

    def sharpe_report(
        self, *, pbo: float | None = None, scope_strategy: str | None = None
    ) -> SharpeReport:
        """The Sharpe, with its trials count and deflated Sharpe attached."""

        return SharpeReport.from_log(
            self._log, self.run_id, pbo=pbo, scope_strategy=scope_strategy
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<LoggedBacktest {self.run_id[:8]} {self.spec.strategy_name!r} "
            f"{self.spec.split}>"
        )


class LoggedBacktestEngine(ABC):
    """Abstract engine whose :meth:`run` always writes to ``research_log``."""

    def __init__(self, research_log: ResearchLog) -> None:
        self._research_log = research_log

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "run" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} may not override run(): the research-log write "
                "lives there and every backtest run must be logged "
                "(CLAUDE.md §3). Implement execute() instead — it is called from "
                "inside the log's run context."
            )

    @property
    def research_log(self) -> ResearchLog:
        return self._research_log

    @property
    def data_handler(self) -> DataHandler | None:
        """The handler this engine streams, when it has one.

        Override it and :meth:`run` cross-checks the logged window against the
        data actually served — catching a run logged as ``train`` while streaming
        holdout bars.
        """

        return None

    @abstractmethod
    def build_spec(self) -> RunSpec:
        """Describe the run *before* it executes: strategy, universe, window, costs."""

    @abstractmethod
    def execute(self, run: LoggedRun) -> RunResult:
        """Run the backtest and return its metrics.

        Called inside the log's run context: the ``running`` row already exists,
        so a crash in here leaves an auditable failed row rather than nothing.
        Build the return value with
        :meth:`~quantlab.research.log.RunResult.from_returns` so the Sharpe and
        the return moments the deflation needs are captured together.
        """

    @final
    def run(self) -> LoggedBacktest:
        """Execute the backtest inside the research log. Cannot be overridden."""

        spec = self.build_spec()
        self._check_handler_agrees(spec)
        with self._research_log.run(spec) as handle:
            result = self.execute(handle)
            if not isinstance(result, RunResult):  # pragma: no cover - defensive
                raise ResearchLogError(
                    f"{type(self).__name__}.execute must return a RunResult, got "
                    f"{type(result).__name__}; the log cannot record an unknown shape"
                )
            handle.record(result)
            run_id = handle.run_id
        return LoggedBacktest(self._research_log, run_id, spec, result)

    def _check_handler_agrees(self, spec: RunSpec) -> None:
        handler = self.data_handler
        if handler is None:
            return
        if handler.end != spec.end or handler.start != spec.start:
            raise ResearchLogError(
                f"the run spec window [{spec.start.isoformat()}, "
                f"{spec.end.isoformat()}] does not match the DataHandler window "
                f"[{handler.start.isoformat()}, {handler.end.isoformat()}]. The log "
                "would record a window the backtest did not run on."
            )
        if handler.holdout != spec.holdout:
            raise ResearchLogError(
                f"holdout opt-in disagrees: DataHandler holdout={handler.holdout}, "
                f"RunSpec holdout={spec.holdout}. Both must state the same "
                "intention for the lock to mean anything."
            )
