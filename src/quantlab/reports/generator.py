"""Report generation, subject to the reporting rule.

This is the enforcement point PRD §4.6 describes: the generator refuses to emit a
Sharpe ratio without its trials count and deflated Sharpe. It does not print a
caveat, and it does not fall back to the bare number — it raises.

The mechanism is :class:`~quantlab.research.reporting.SharpeReport`, which cannot
be constructed without both, plus
:func:`~quantlab.research.reporting.assert_no_bare_sharpe`, which catches a Sharpe
arriving through the metrics mapping. Later milestones render PDF tearsheets from
the same contract (M4); the format changes, the rule does not.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from quantlab.research.deflated import NullMaxSharpeDistribution, PBOResult
from quantlab.research.log import ResearchLog
from quantlab.research.reporting import (
    SharpeReport,
    assert_no_bare_sharpe,
    find_bare_sharpe_keys,
)


def render_metrics_table(
    metrics: Mapping[str, Any], sharpe_report: SharpeReport | None = None
) -> str:
    """Render a metrics mapping as a markdown table.

    Raises :class:`~quantlab.research.reporting.ReportingRuleError` if ``metrics``
    contains a Sharpe ratio and ``sharpe_report`` is absent — including when the
    Sharpe is one row among twenty and nobody was thinking about it.
    """

    assert_no_bare_sharpe(metrics, sharpe_report)
    bare = set(find_bare_sharpe_keys(metrics))
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in metrics.items():
        if key in bare:
            # The Sharpe is rendered from the report, never from the raw number,
            # so its trials count and deflation travel with it.
            continue
        lines.append(f"| {key} | {_format(value)} |")
    if sharpe_report is not None:
        lines.append(f"| sharpe | {sharpe_report.render()} |")
    return "\n".join(lines)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_run_report(
    research_log: ResearchLog,
    run_id: str,
    *,
    pbo: PBOResult | None = None,
    null_distribution: NullMaxSharpeDistribution | None = None,
    scope_strategy: str | None = None,
) -> str:
    """A markdown report for one logged run.

    Every number here is traceable to ``run_id``, which is the property the
    research paper depends on (PRD §Module 15: "every number traceable to a
    ``research_log`` run_id").
    """

    entry = research_log.entry(run_id)
    report = SharpeReport.from_log(
        research_log,
        run_id,
        pbo=pbo.pbo if pbo is not None else None,
        scope_strategy=scope_strategy,
    )

    provenance = entry.git_commit_hash or "not a git work tree"
    if entry.git_dirty:
        provenance += " (dirty work tree — not reproducible from this commit alone)"

    lines = [
        f"# Backtest report — {entry.strategy_name}",
        "",
        f"**run_id** `{run_id}`  ",
        f"**commit** `{provenance}`  ",
        f"**window** {entry.start_date.isoformat()} → {entry.end_date.isoformat()} "
        f"(`{entry.split}`)  ",
        f"**universe** {entry.index_name}, {entry.universe}  ",
        f"**costs** {entry.cost_model}  ",
        f"**provenance** {report.provenance}",
        "",
        "## Performance",
        "",
        report.render(),
        "",
        f"- Trials preceding this result: **{report.trials}** (from `research_log`)",
        f"- Expected max Sharpe from an edgeless search of that size: "
        f"**{report.expected_max_sharpe:.3f}**",
        f"- Deflated Sharpe (probability the true Sharpe exceeds zero): "
        f"**{report.deflated_sharpe:.4f}**",
    ]
    if not report.significant:
        lines.append(
            "- **This result is not significant after deflation.** It is consistent "
            "with the best of an equally large search over strategies with no edge."
        )
    if entry.holdout_touched:
        lines.append(
            "- **HOLDOUT**: computed on locked holdout data. Evaluated once; no "
            "decision downstream of this number may feed back into the strategy."
        )
    if report.exploratory:
        lines.append(
            "- **Exploratory**: no pre-registered hypothesis. The prediction was not "
            "written down before the test."
        )

    lines += ["", "## Metrics", "", render_metrics_table(entry.metrics or {}, report)]

    if null_distribution is not None:
        lines += [
            "",
            "## Expected maximum Sharpe under the null",
            "",
            "```",
            null_distribution.render(report.value),
            "```",
        ]
    if pbo is not None:
        lines += [
            "",
            "## Probability of backtest overfitting",
            "",
            "```",
            pbo.render(),
            "```",
        ]

    lines += [
        "",
        "---",
        "",
        "Generated by QuantLab. Any Sharpe ratio above is accompanied by the number "
        "of trials that preceded it and its deflated Sharpe; the report generator "
        "refuses to emit one otherwise (PRD §4.6).",
    ]
    return "\n".join(lines)
