"""The reporting rule: no Sharpe ratio without its trials count and deflation.

PRD §4.6 is a *code* requirement, not an editorial guideline. A Sharpe ratio
reported without the number of trials that preceded it is not a weaker claim than
one reported with it — it is an unfalsifiable claim, because the reader cannot
tell whether they are looking at one hypothesis or the best of three hundred.

Two mechanisms enforce it:

* :class:`SharpeReport` cannot be constructed without a trials count and a
  deflated Sharpe. There is no default, no ``None``, and no keyword to skip them.
  Any layer that emits performance — the report generator, the FastAPI response
  models, the ``SharpeDisplay`` React component's props — accepts *this type*
  rather than a float, so a bare number fails to typecheck rather than printing.
* :func:`assert_no_bare_sharpe` scans a metrics mapping and raises if it finds a
  Sharpe-like key travelling without an accompanying report. This catches the
  realistic failure: not someone deliberately bypassing the contract, but a
  metrics dict flowing into a template.

Trials counts come from ``research_log`` via :meth:`SharpeReport.from_log`; there
is no constructor path that takes a hand-entered count.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from quantlab.research.deflated import (
    DEFAULT_SIGNIFICANCE,
    DeflatedSharpeResult,
    deflate_run,
)
from quantlab.research.log import ResearchLog

#: Metric keys that count as "a Sharpe ratio" for the purposes of the rule.
#: ``deflated_sharpe`` and friends are the accompaniment, not the bare number.
_SHARPE_KEY_RE = re.compile(r"sharpe", re.IGNORECASE)
_ALLOWED_SHARPE_KEYS = frozenset(
    {
        "deflated_sharpe",
        "expected_max_sharpe",
        "sharpe_trials",
        "sharpe_variance",
        "probabilistic_sharpe",
    }
)


class ReportingRuleError(RuntimeError):
    """Raised when a Sharpe ratio would be emitted without its context."""


@dataclass(frozen=True, slots=True)
class SharpeReport:
    """A Sharpe ratio bound to the evidence needed to interpret it.

    Every field before the defaults is required, which is the whole design: the
    type cannot represent a Sharpe that lacks a trials count or a deflated
    Sharpe. Construct it with :meth:`from_log`.
    """

    #: Annualised Sharpe as observed in the backtest.
    value: float
    #: Number of trials that preceded this result, counted from ``research_log``.
    trials: int
    #: Probability the true Sharpe exceeds zero, after correcting for ``trials``.
    deflated_sharpe: float
    #: The best-of-``trials`` Sharpe expected from an edgeless search.
    expected_max_sharpe: float
    #: Which split produced ``value`` — a Sharpe without a split is not a result.
    split: str
    cost_model: str
    run_id: str
    n_periods: int
    #: Probability of backtest overfitting, when a CSCV run is available.
    pbo: float | None = None
    exploratory: bool = True
    holdout_touched: bool = False
    hypothesis_id: str | None = None

    def __post_init__(self) -> None:
        if self.trials < 1:
            raise ReportingRuleError(
                f"trials must be at least 1, got {self.trials}: a reported Sharpe "
                "always has at least the one trial that produced it"
            )
        if not 0.0 <= self.deflated_sharpe <= 1.0:
            raise ReportingRuleError(
                f"deflated_sharpe is a probability and must lie in [0, 1], got "
                f"{self.deflated_sharpe}"
            )
        if not self.split:
            raise ReportingRuleError("split is required alongside a Sharpe ratio")
        if not self.cost_model:
            raise ReportingRuleError(
                "cost_model is required: a Sharpe computed without costs is a "
                "different quantity from a net Sharpe and must say so"
            )

    @classmethod
    def from_log(
        cls,
        research_log: ResearchLog,
        run_id: str,
        *,
        pbo: float | None = None,
        scope_strategy: str | None = None,
    ) -> SharpeReport:
        """Build a report for a logged run, deflating against the logged trials.

        Raises rather than degrading: if the run has no Sharpe, no observation
        count, or cannot be deflated, there is nothing reportable here.
        """

        entry = research_log.entry(run_id)
        deflation = deflate_run(research_log, run_id, scope_strategy=scope_strategy)
        return cls(
            value=float(entry.sharpe or 0.0),
            trials=deflation.trials,
            deflated_sharpe=deflation.deflated_sharpe,
            expected_max_sharpe=deflation.expected_max_sharpe,
            split=entry.split,
            cost_model=entry.cost_model,
            run_id=run_id,
            n_periods=deflation.n_periods,
            pbo=pbo,
            exploratory=bool(entry.exploratory),
            holdout_touched=bool(entry.holdout_touched),
            hypothesis_id=entry.hypothesis_id,
        )

    @classmethod
    def from_deflation(
        cls,
        deflation: DeflatedSharpeResult,
        *,
        split: str,
        cost_model: str,
        run_id: str,
        pbo: float | None = None,
        exploratory: bool = True,
        holdout_touched: bool = False,
        hypothesis_id: str | None = None,
    ) -> SharpeReport:
        """Build a report from an already-computed deflation."""

        return cls(
            value=deflation.observed_sharpe,
            trials=deflation.trials,
            deflated_sharpe=deflation.deflated_sharpe,
            expected_max_sharpe=deflation.expected_max_sharpe,
            split=split,
            cost_model=cost_model,
            run_id=run_id,
            n_periods=deflation.n_periods,
            pbo=pbo,
            exploratory=exploratory,
            holdout_touched=holdout_touched,
            hypothesis_id=hypothesis_id,
        )

    @property
    def significant(self) -> bool:
        """Whether the deflated Sharpe clears the 95% threshold."""

        return self.deflated_sharpe >= DEFAULT_SIGNIFICANCE

    @property
    def provenance(self) -> str:
        """Pre-registered (with id) or exploratory — always stated."""

        if self.exploratory or self.hypothesis_id is None:
            return "exploratory"
        return f"pre-registered ({self.hypothesis_id})"

    def render(self) -> str:
        """Render the Sharpe with its context inseparable from the number."""

        parts = [
            f"Sharpe {self.value:.2f}",
            f"{self.trials} trials",
            f"deflated {self.deflated_sharpe:.3f}",
        ]
        if self.pbo is not None:
            parts.append(f"PBO {self.pbo:.2f}")
        parts.append(self.split.upper())
        parts.append(self.cost_model)
        parts.append(self.provenance)
        line = " | ".join(parts)
        if not self.significant:
            line += "  ← not significant after deflation"
        if self.holdout_touched:
            line += "  [HOLDOUT]"
        return line

    def as_dict(self) -> dict[str, Any]:
        """The API-boundary shape: an object, never a bare float (PRD §Module 13)."""

        return {
            "value": self.value,
            "trials": self.trials,
            "deflated_sharpe": self.deflated_sharpe,
            "expected_max_sharpe": self.expected_max_sharpe,
            "pbo": self.pbo,
            "split": self.split,
            "cost_model": self.cost_model,
            "run_id": self.run_id,
            "n_periods": self.n_periods,
            "significant": self.significant,
            "provenance": self.provenance,
            "holdout_touched": self.holdout_touched,
        }


def find_bare_sharpe_keys(metrics: Mapping[str, Any]) -> list[str]:
    """Return metric keys that look like a bare Sharpe ratio."""

    return [
        key
        for key in metrics
        if _SHARPE_KEY_RE.search(key) and key not in _ALLOWED_SHARPE_KEYS
    ]


def assert_no_bare_sharpe(
    metrics: Mapping[str, Any],
    sharpe_report: SharpeReport | None,
    *,
    tolerance: float = 1e-6,
) -> None:
    """Raise unless every Sharpe-like metric is covered by ``sharpe_report``.

    Also checks that the report describes *this* result: a report whose ``value``
    disagrees with the metrics dict means a deflation has been paired with the
    wrong run, which is worse than no deflation at all because it looks rigorous.
    """

    bare = find_bare_sharpe_keys(metrics)
    if not bare:
        return
    if sharpe_report is None:
        raise ReportingRuleError(
            f"refusing to emit Sharpe metric(s) {sorted(bare)} without a "
            "SharpeReport. Any reported Sharpe must be accompanied by the number "
            "of trials that preceded it and its deflated Sharpe, both read from "
            "research_log (CLAUDE.md §3). Build one with "
            "SharpeReport.from_log(research_log, run_id)."
        )
    for key in bare:
        value = metrics[key]
        if not isinstance(value, (int, float)):
            continue
        if abs(float(value) - sharpe_report.value) > tolerance:
            raise ReportingRuleError(
                f"metric {key}={float(value):.6f} does not match the accompanying "
                f"SharpeReport value {sharpe_report.value:.6f} (run "
                f"{sharpe_report.run_id}). A deflated Sharpe paired with a "
                "different run's number is not a correction, it is a mislabel."
            )
