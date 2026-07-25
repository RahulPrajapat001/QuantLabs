"""Hyper-parameter search, with the size of the search reported alongside it.

The result of a model search is not the best configuration. It is the best
configuration *together with how many were tried*, because those two numbers only
mean something as a pair. A rank IC of 0.012 from a single pre-registered
configuration and a rank IC of 0.012 selected from ninety are different claims,
and only one of them is evidence.

PRD Module 11 therefore requires the search space and the iteration count to be
reported next to the results, and this module makes that structural:

* :class:`SearchSpace` is data, not a loop written at the call site, so it can be
  printed exactly as it was executed.
* :meth:`SearchResult.render` leads with the space and the iteration count,
  before any performance number appears.
* Every configuration is evaluated through
  :func:`~quantlab.ml_models.evaluate.evaluate_model`, so every one of them is a
  row in ``research_log``. :attr:`SearchResult.trials_before` and
  :attr:`SearchResult.trials_after` are read back from the log, which means the
  report states what the search cost the *whole project's* trials count, not
  just its own.

Selection runs on the validation split when one is supplied, per the split
discipline in ``CLAUDE.md`` §3. Without one, selection falls back to the pooled
purged-CV score within the training split — still out-of-sample, but chosen on
the same data the folds came from, which the report says plainly.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import structlog

from quantlab.ml_models.baselines import Baseline
from quantlab.ml_models.dataset import MLDataset
from quantlab.ml_models.evaluate import ModelEvaluation, evaluate_model
from quantlab.ml_models.models import (
    DEFAULT_SEARCH_SPACE,
    MODEL_FAMILIES,
    ModelSpec,
    Task,
)
from quantlab.ml_models.trading import TradingRule
from quantlab.ml_models.validation import Splitter
from quantlab.research.log import ResearchLog
from quantlab.research.reporting import SharpeReport

log = structlog.get_logger(__name__)

__all__ = [
    "SearchResult",
    "SearchSpace",
    "default_space",
    "run_search",
]


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """One family's hyper-parameter grid — the thing that gets reported."""

    family: str
    grid: Mapping[str, Sequence[Any]]

    def __post_init__(self) -> None:
        if self.family not in MODEL_FAMILIES:
            raise ValueError(
                f"unknown model family {self.family!r}; permitted families are "
                f"{sorted(MODEL_FAMILIES)}"
            )
        empty = [k for k, v in self.grid.items() if not v]
        if empty:
            raise ValueError(f"grid axes with no values: {empty}")

    @property
    def size(self) -> int:
        """Number of configurations this grid expands to."""

        total = 1
        for values in self.grid.values():
            total *= len(values)
        return total

    def configurations(self) -> list[dict[str, Any]]:
        """The full Cartesian product, in a stable order.

        Stable because a search whose order changes between runs produces a
        different ``search_iteration`` for the same configuration, and the
        research log would no longer line up with a rerun.
        """

        if not self.grid:
            return [{}]
        keys = sorted(self.grid)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*(self.grid[k] for k in keys))
        ]

    def render(self) -> str:
        if not self.grid:
            return f"  {self.family}: defaults only (1 configuration)"
        axes = "\n".join(
            f"      {key}: {list(self.grid[key])}" for key in sorted(self.grid)
        )
        return f"  {self.family}: {self.size} configuration(s)\n{axes}"


def default_space(family: str) -> SearchSpace:
    """The declared default grid for ``family`` (see :data:`DEFAULT_SEARCH_SPACE`)."""

    return SearchSpace(family=family, grid=DEFAULT_SEARCH_SPACE.get(family, {}))


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Every configuration tried, the winner, and what the search cost in trials."""

    task: Task
    spaces: tuple[SearchSpace, ...]
    evaluations: tuple[ModelEvaluation, ...]
    #: ``(configuration label, error)`` for configurations that raised. Their
    #: ``research_log`` rows exist and are marked failed.
    failures: tuple[tuple[str, str], ...]
    #: Completed trials in the whole log immediately before the search started.
    trials_before: int
    #: …and immediately after. The difference is what this search added.
    trials_after: int
    selection_basis: str

    @property
    def n_configurations(self) -> int:
        """Configurations the search space defines — the iteration count."""

        return sum(space.size for space in self.spaces)

    @property
    def n_completed(self) -> int:
        return len(self.evaluations)

    @property
    def best(self) -> ModelEvaluation:
        """The configuration with the highest selection metric."""

        if not self.evaluations:
            raise ValueError(
                "no configuration completed, so there is no best model. The failed "
                "rows are in research_log."
            )
        return max(self.evaluations, key=lambda e: e.selection_metric)

    def ranked(self) -> list[ModelEvaluation]:
        return sorted(self.evaluations, key=lambda e: -e.selection_metric)

    def sharpe_report(
        self, research_log: ResearchLog, *, pbo: float | None = None
    ) -> SharpeReport:
        """The winner's Sharpe, deflated by the **whole log's** trials count.

        Not by ``n_configurations``: the model search is one part of a larger
        selection process that also includes every strategy backtest run on this
        project, and deflating by the smaller number would understate the
        correction. This is PRD Module 11's "deflated by the full trials count
        including model search", read in the direction that costs the researcher
        something.
        """

        return self.best.sharpe_report(research_log, pbo=pbo)

    def render(self, research_log: ResearchLog | None = None, top: int = 10) -> str:
        lines = [
            "Model search",
            "============",
            f"Task:                {self.task.value}",
            f"Search space ({self.n_configurations} configuration(s) total):",
        ]
        lines.extend(space.render() for space in self.spaces)
        lines += [
            "",
            f"Iterations run:      {self.n_completed} completed, "
            f"{len(self.failures)} failed",
            f"Selection metric:    {self.selection_basis} "
            f"({'rank IC' if self.task is Task.REGRESSION else 'ROC-AUC'})",
            f"Trials in research_log: {self.trials_before} before -> "
            f"{self.trials_after} after "
            f"(+{self.trials_after - self.trials_before})",
            "",
        ]
        if not self.evaluations:
            lines.append("No configuration completed. Failures:")
            lines.extend(f"  {label}: {error}" for label, error in self.failures)
            return "\n".join(lines)

        header = "rank IC" if self.task is Task.REGRESSION else "ROC-AUC"
        lines.append(
            f"{'#':>3s} {'configuration':38s} {header:>9s} {'IS-OOS':>8s} "
            f"{'Sharpe':>8s} {'beats bl':>9s}"
        )
        for i, evaluation in enumerate(self.ranked()[:top], start=1):
            lines.append(
                f"{i:3d} {evaluation.model.describe()[:38]:38s} "
                f"{evaluation.selection_metric:9.4f} "
                f"{evaluation.overfitting_gap:8.4f} "
                f"{evaluation.headline_traded.annualised_sharpe:8.3f} "
                f"{'yes' if evaluation.beats_baselines else 'NO':>9s}"
            )
        remaining = self.n_completed - min(top, self.n_completed)
        if remaining > 0:
            lines.append(f"    … and {remaining} further configuration(s)")

        if self.failures:
            lines.append("")
            lines.append("Failed configurations (logged as failed rows):")
            lines.extend(f"  {label}: {error}" for label, error in self.failures)

        best = self.best
        lines += [
            "",
            "Winner",
            "------",
            best.model.describe(),
            "",
            best.headline_comparison.render(),
            "",
        ]
        if research_log is not None:
            lines.append(self.sharpe_report(research_log).render())
        lines.append(
            f"The winner was selected from {self.n_configurations} configuration(s); "
            f"the log now holds {self.trials_after} completed trials, and that is "
            "the number its Sharpe is deflated by."
        )
        return "\n".join(lines)


def run_search(
    research_log: ResearchLog,
    dataset: MLDataset,
    *,
    families: Sequence[str] | None = None,
    spaces: Sequence[SearchSpace] | None = None,
    splitter: Splitter | None = None,
    rule: TradingRule | None = None,
    validation: MLDataset | None = None,
    baselines: Sequence[Baseline] | None = None,
    hypothesis_id: str | None = None,
    random_state: int = 0,
    compute_shap: bool = False,
    stop_on_error: bool = False,
) -> SearchResult:
    """Evaluate every configuration in the space, logging each as its own trial.

    Failures are recorded and the search continues by default: a configuration
    that fails to converge is information, and abandoning ninety completed
    evaluations because the ninety-first raised loses more than it protects. Each
    failure leaves a ``failed`` row in ``research_log``, so nothing disappears.
    """

    task = dataset.task
    resolved: list[SearchSpace] = (
        list(spaces)
        if spaces is not None
        else [default_space(f) for f in (families or list(MODEL_FAMILIES))]
    )
    if not resolved:
        raise ValueError("a search needs at least one model family")

    trials_before = research_log.trials()
    evaluations: list[ModelEvaluation] = []
    failures: list[tuple[str, str]] = []
    iteration = 0

    for space in resolved:
        for params in space.configurations():
            iteration += 1
            model = ModelSpec(
                family=space.family,
                task=task,
                params=params,
                random_state=random_state,
            )
            try:
                evaluations.append(
                    evaluate_model(
                        research_log,
                        dataset,
                        model,
                        splitter=splitter,
                        rule=rule,
                        validation=validation,
                        baselines=baselines,
                        hypothesis_id=hypothesis_id,
                        search_iteration=iteration,
                        compute_shap=compute_shap,
                        notes=(
                            f"model search iteration {iteration} of "
                            f"{sum(s.size for s in resolved)}"
                        ),
                    )
                )
            except Exception as exc:
                if stop_on_error:
                    raise
                failures.append((model.describe(), f"{type(exc).__name__}: {exc}"))
                log.warning(
                    "ml.search.configuration_failed",
                    family=space.family,
                    params=params,
                    error=str(exc),
                )

    result = SearchResult(
        task=task,
        spaces=tuple(resolved),
        evaluations=tuple(evaluations),
        failures=tuple(failures),
        trials_before=trials_before,
        trials_after=research_log.trials(),
        selection_basis=(
            "validation split" if validation is not None else "purged-CV out-of-sample"
        ),
    )
    log.info(
        "ml.search.complete",
        configurations=result.n_configurations,
        completed=result.n_completed,
        failed=len(result.failures),
        trials_before=trials_before,
        trials_after=result.trials_after,
    )
    return result
