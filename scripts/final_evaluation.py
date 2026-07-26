"""The holdout evaluation. Runs **once**, at the very end of the project.

    QUANTLAB_UNLOCK_HOLDOUT=1 python scripts/final_evaluation.py \\
        --strategy momentum_12_1 --hypothesis H1 --index NIFTY500

This script exists to be committed *before* it is run. That ordering is the whole
point: a holdout evaluation whose script was written after seeing the result is
not an out-of-sample test, it is a fitted one. The git history is the evidence,
which is why the script refuses to run from a dirty work tree.

**What it does**

1. Resolves the single pre-registered strategy named on the command line and
   verifies the hypothesis exists in ``docs/hypotheses.md``.
2. Runs it over ``HOLDOUT_START`` onward with ``IndianEquityCostModel`` — the only
   cost model permitted here, because a holdout result under approximated costs
   answers a question nobody asked.
3. Emits full metrics plus the deflated Sharpe, using the **total trials count
   read from research_log** across the whole project.
4. Prints a report to paste verbatim into ``docs/research/holdout_results.md``.

**What it deliberately does not do**

It does not sweep parameters, try several strategies, or pick the best of
anything. It takes one strategy and reports what happened to it. If the result is
bad, that is the finding.

**Afterwards**

Do not modify the strategy and re-run. The holdout is consumed. Iterating on it is
a new project with a new holdout, and the second run's number would be a
selection, not a test.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Only this cost model is accepted. A holdout number under `GenericBpsModel` or
#: `ZeroCostModel` is not the result the project set out to produce.
REQUIRED_COST_MODEL = "IndianEquityCostModel"


class PreflightError(RuntimeError):
    """A precondition for the one-time run is not met."""


def _preflight(*, allow_dirty: bool) -> tuple[str, bool]:
    """Refuse unless this is a legitimate, reproducible, once-only run.

    Returns ``(commit_hash, dirty)``.
    """

    if os.environ.get("QUANTLAB_UNLOCK_HOLDOUT") != "1":
        raise PreflightError(
            "QUANTLAB_UNLOCK_HOLDOUT=1 is not set.\n"
            "The holdout is locked (CLAUDE.md §3). Unlocking is a conscious act; "
            "this script will not do it for you."
        )

    from quantlab.research.log import capture_git_provenance

    provenance = capture_git_provenance()
    commit = provenance.commit_hash
    dirty = bool(provenance.dirty)

    if commit is None:
        raise PreflightError(
            "Not running inside a git work tree, so this run has no commit hash.\n"
            "A holdout result that cannot be traced to a commit is not evidence. "
            "(Note this also means the Docker image cannot run the holdout: .git "
            "is excluded from the build context by design.)"
        )
    if dirty and not allow_dirty:
        raise PreflightError(
            "The work tree is dirty.\n"
            "This script must be committed BEFORE it runs — that ordering is what "
            "makes the result out-of-sample rather than fitted. Commit first, then "
            "run."
        )
    return commit, dirty


def _verify_preregistration(hypothesis_id: str) -> Any:
    """Confirm the hypothesis was written down, and return it."""

    from quantlab.research.preregister import load_hypotheses

    hypotheses = load_hypotheses()
    if hypothesis_id not in hypotheses:
        raise PreflightError(
            f"{hypothesis_id!r} is not in docs/hypotheses.md. "
            f"Available: {sorted(hypotheses)}.\n"
            "An unregistered strategy evaluated on the holdout is exploratory work "
            "wearing a prediction's clothes."
        )
    return hypotheses[hypothesis_id]


def run(
    *,
    strategy_name: str,
    hypothesis_id: str,
    index_name: str,
    end: date | None,
    initial_capital: float,
    max_positions: int,
    max_position_weight: float,
    allow_dirty: bool,
) -> int:
    from quantlab.api.deps import load_price_panel
    from quantlab.backtesting.costs import default_cost_models
    from quantlab.backtesting.data_handler import DataHandler
    from quantlab.backtesting.engine import BacktestConfig, BacktestEngine
    from quantlab.backtesting.portfolio import Portfolio, RiskLimits
    from quantlab.config import HOLDOUT_START
    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.research.log import ResearchLog
    from quantlab.strategies import default_strategies
    from quantlab.utils.logging import configure_logging, request_context

    configure_logging()

    commit, dirty = _preflight(allow_dirty=allow_dirty)
    hypothesis = _verify_preregistration(hypothesis_id)

    strategies = default_strategies()
    if strategy_name not in strategies:
        raise PreflightError(
            f"unknown strategy {strategy_name!r}; available: {sorted(strategies)}"
        )

    factory = create_session_factory(create_db_engine())

    with request_context() as request_id, factory() as session:
        research_log = ResearchLog(session)
        trials_before = research_log.trials()

        if trials_before == 0:
            raise PreflightError(
                "research_log is empty.\n"
                "The deflated Sharpe is meaningless without a trials count, and a "
                "holdout run is only interpretable against the search that "
                "preceded it. Do the validation-split work first."
            )

        print(_banner(), flush=True)
        print(
            f"strategy       {strategy_name}\n"
            f"hypothesis     {hypothesis_id} — {hypothesis.title}\n"
            f"prediction     {hypothesis.prediction}\n"
            f"index          {index_name}\n"
            f"window         {HOLDOUT_START.isoformat()} → "
            f"{end.isoformat() if end else 'latest available'}\n"
            f"cost model     {REQUIRED_COST_MODEL}\n"
            f"trials before  {trials_before}\n"
            f"commit         {commit}{' (DIRTY)' if dirty else ''}\n"
            f"request id     {request_id}\n"
            f"started        {datetime.now(UTC).isoformat()}\n",
            flush=True,
        )

        bars = load_price_panel(session, index_name, start=HOLDOUT_START, end=end)
        if bars.empty:
            raise PreflightError(
                f"no price history for {index_name} at/after "
                f"{HOLDOUT_START.isoformat()}. There is nothing to evaluate."
            )

        # holdout=True is the caller's half of the lock contract; the DataHandler
        # checks QUANTLAB_UNLOCK_HOLDOUT itself, independently of the preflight.
        handler = DataHandler(bars, start=HOLDOUT_START, end=end, holdout=True)

        engine = BacktestEngine(
            research_log,
            strategy=strategies[strategy_name],
            data_handler=handler,
            portfolio=Portfolio(
                initial_capital=initial_capital,
                limits=RiskLimits(
                    max_position_weight=max_position_weight,
                    max_positions=max_positions,
                ),
            ),
            config=BacktestConfig(
                initial_capital=initial_capital,
                cost_model=default_cost_models()[REQUIRED_COST_MODEL],
                universe="point-in-time",
                index_name=index_name,
                hypothesis_id=hypothesis_id,
                notes=(
                    "FINAL HOLDOUT EVALUATION — one-time, irreversible. "
                    f"Script committed at {commit}."
                ),
            ),
        )

        backtest = engine.run()
        report = backtest.sharpe_report()
        rendered = _render(
            strategy_name=strategy_name,
            hypothesis_id=hypothesis_id,
            index_name=index_name,
            commit=commit,
            entry=backtest.entry,
            metrics=backtest.metrics,
            report=report,
        )

    print(rendered)
    print(
        "\nWrite the block above VERBATIM to docs/research/holdout_results.md.\n"
        "Do not edit the numbers. Do not modify the strategy and re-run: the "
        "holdout is now consumed.",
        flush=True,
    )
    return 0


def _banner() -> str:
    return (
        "=" * 78
        + "\n  FINAL HOLDOUT EVALUATION — ONE TIME ONLY\n"
        + "  This consumes the holdout. Whatever it reports is the result.\n"
        + "=" * 78
    )


def _num(value: Any, spec: str = ".4f") -> str:
    """Format defensively: a missing metric prints as a dash, never as 0.0."""

    if value is None:
        return "—"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def _render(
    *,
    strategy_name: str,
    hypothesis_id: str,
    index_name: str,
    commit: str,
    entry: Any,
    metrics: dict[str, Any],
    report: Any,
) -> str:
    """Plain text, pasted verbatim. No formatting cleverness to argue with."""

    pbo = _num(report.pbo) if report.pbo is not None else "—"
    lines = [
        "",
        "-" * 78,
        "HOLDOUT RESULT",
        "-" * 78,
        f"strategy        {strategy_name}",
        f"hypothesis      {hypothesis_id}",
        f"index           {index_name}",
        f"run_id          {entry.run_id}",
        f"commit          {commit}",
        f"split           {entry.split}",
        f"holdout_touched {entry.holdout_touched}",
        f"evaluated_at    {datetime.now(UTC).isoformat()}",
        f"window          {entry.start_date} → {entry.end_date}",
        "",
        "SHARPE (never reported without the three lines that follow it)",
        f"  sharpe                {_num(report.value)}",
        f"  trials                {report.trials}",
        f"  deflated_sharpe       {_num(report.deflated_sharpe)}",
        f"  expected_max_sharpe   {_num(report.expected_max_sharpe)}",
        f"  significant           {report.significant}",
        f"  pbo                   {pbo}",
        f"  provenance            {report.provenance}",
        "",
        "METRICS",
    ]
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, dict):
            lines.append(f"  {key}:")
            lines += [f"    {k:24} {_num(v, '.2f')}" for k, v in sorted(value.items())]
        else:
            lines.append(f"  {key:26} {_num(value)}")

    lines += [
        "",
        f"summary         {report.summary}",
        "-" * 78,
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--strategy",
        required=True,
        help="The ONE pre-registered strategy chosen from validation results.",
    )
    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Its pre-registered hypothesis id (e.g. H1), from docs/hypotheses.md.",
    )
    parser.add_argument("--index", default="NIFTY500")
    parser.add_argument(
        "--end",
        default=None,
        help="Inclusive end date (YYYY-MM-DD). Omit for all available holdout data.",
    )
    parser.add_argument("--initial-capital", type=float, default=1_000_000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--max-position-weight", type=float, default=0.15)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help=argparse.SUPPRESS,  # rehearsal only; never for the real run
    )
    args = parser.parse_args(argv)

    end = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else None

    try:
        return run(
            strategy_name=args.strategy,
            hypothesis_id=args.hypothesis,
            index_name=args.index,
            end=end,
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            max_position_weight=args.max_position_weight,
            allow_dirty=args.allow_dirty,
        )
    except PreflightError as exc:
        print(f"\nREFUSED\n\n{exc}\n", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
