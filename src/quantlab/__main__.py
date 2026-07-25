"""QuantLab command-line entry point.

A Typer application. Commands are implemented per-milestone; ``ingest`` is the
first (M1). See ``docs/PRD.md`` §5.
"""

from __future__ import annotations

from datetime import date, datetime

import structlog
import typer

app = typer.Typer(help="QuantLab — quantitative research platform for Indian equities.")

log = structlog.get_logger(__name__)


@app.command()
def version() -> None:
    """Print the installed QuantLab version."""
    from quantlab import __version__

    typer.echo(__version__)


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - Typer surfaces the message
        raise typer.BadParameter(f"expected YYYY-MM-DD, got {value!r}") from exc


@app.command()
def ingest(
    tickers: str = typer.Option(
        "",
        "--tickers",
        help="Comma-separated vendor tickers, e.g. RELIANCE.NS,TCS.NS.",
    ),
    start: str = typer.Option(
        "2015-01-01", "--start", help="Inclusive start date (YYYY-MM-DD)."
    ),
    end: str = typer.Option(
        None, "--end", help="Inclusive end date (YYYY-MM-DD). Defaults to today."
    ),
    source: str = typer.Option(
        "yfinance", "--source", help="Data source: yfinance | alphavantage."
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Fetch only data since each security's last successful ingest.",
    ),
) -> None:
    """Ingest OHLCV, corporate actions and delisting status into the platform.

    Examples::

        quantlab ingest --tickers RELIANCE.NS,TCS.NS --start 2015-01-01 \
            --end 2025-12-31 --source yfinance
        quantlab ingest --incremental
    """

    from quantlab.data import get_provider
    from quantlab.data.ingest import ingest_incremental, ingest_ticker
    from quantlab.db.session import (
        create_all,
        create_db_engine,
        create_session_factory,
        session_scope,
    )

    provider = get_provider(source)
    engine = create_db_engine()
    create_all(engine)  # no-op once Alembic has run; convenient for first use
    factory = create_session_factory(engine)

    if incremental:
        with session_scope(factory) as session:
            results = ingest_incremental(session, provider)
        for r in results:
            typer.echo(
                f"{r.ticker} [{r.security_id}] {r.start}..{r.end}: "
                f"+{r.rows_written} rows"
            )
        if not results:
            typer.echo("Nothing to do: all securities are current.")
        return

    symbols = [t.strip() for t in tickers.split(",") if t.strip()]
    if not symbols:
        raise typer.BadParameter("provide --tickers or use --incremental")

    start_date = _parse_date(start)
    end_date = _parse_date(end) if end else date.today()

    for symbol in symbols:
        with session_scope(factory) as session:
            try:
                result = ingest_ticker(session, provider, symbol, start_date, end_date)
            except Exception as exc:
                typer.echo(f"{symbol}: FAILED — {exc}", err=True)
                continue
        typer.echo(
            f"{result.ticker} [{result.security_id}] "
            f"{result.start}..{result.end}: +{result.rows_written} rows "
            f"({result.rows_total} fetched, "
            f"{result.corporate_actions_written} corporate actions)"
        )


universe_app = typer.Typer(help="Point-in-time universe construction (PRD Module 2).")
app.add_typer(universe_app, name="universe")


@universe_app.command("build")
def universe_build(
    index: str = typer.Option(
        ..., "--index", help="Index name, e.g. NIFTY500, NIFTY50, SP500."
    ),
    source: str = typer.Option(
        "auto",
        "--source",
        help="auto | nse | sp500. 'auto' infers NSE for NIFTY*, Wikipedia for SP500.",
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Re-download S&P 500 changes instead of using cache."
    ),
) -> None:
    """Ingest index reconstitution history into ``index_membership``.

    NIFTY indices are built from NSE reconstitution PDFs cached under
    ``reference_dir`` (name each ``{INDEX}_{YYYY-MM-DD}.pdf``). SP500 is built
    from Wikipedia's changes table (cached HTML). The resolution rate — the share
    of reconstitution entries mapped to a ``security_id`` — is reported; anything
    unresolved is recorded in ``unresolved_membership_change``, never dropped.
    """

    from quantlab.db.session import (
        create_all,
        create_db_engine,
        create_session_factory,
        session_scope,
    )
    from quantlab.universe.reconstitution import (
        ReconstitutionEvent,
        collect_nifty_events,
        fetch_sp500_changes,
        ingest_reconstitution,
        parse_sp500_changes,
    )

    resolved_source = source.lower()
    if resolved_source == "auto":
        resolved_source = "sp500" if index.upper().startswith("SP") else "nse"

    events: list[ReconstitutionEvent]
    if resolved_source == "sp500":
        html, source_document = fetch_sp500_changes(refresh=refresh)
        events = parse_sp500_changes(
            html, source_document=source_document, index_name=index
        )
    elif resolved_source == "nse":
        events = collect_nifty_events(index)
    else:
        raise typer.BadParameter(f"unknown source {source!r}; use auto | nse | sp500")

    if not events:
        typer.echo(
            f"No reconstitution events found for {index} via {resolved_source}. "
            "For NSE, place cached PDFs named '{INDEX}_{YYYY-MM-DD}.pdf' in the "
            "reference directory.",
            err=True,
        )
        raise typer.Exit(code=1)

    engine = create_db_engine()
    create_all(engine)  # no-op once Alembic has run; convenient for first use
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        report = ingest_reconstitution(session, index, events)

    typer.echo(report.summary())
    if report.unresolved:
        typer.echo("Unresolved entries (recorded, not dropped):")
        for entry in report.unresolved:
            typer.echo(f"  - {entry}")


@universe_app.command("as-of")
def universe_as_of_cmd(
    index: str = typer.Option(..., "--index", help="Index name, e.g. NIFTY500."),
    date_: str = typer.Option(..., "--date", help="Point-in-time date (YYYY-MM-DD)."),
    universe_mode: str = typer.Option(
        "point-in-time",
        "--universe-mode",
        help=(
            "point-in-time | fixed. 'fixed' freezes constituents at --as-of and "
            "holds them constant: this removes selection bias but NOT delisting "
            "attrition — a name that delists must still be liquidated, not held."
        ),
    ),
    as_of: str = typer.Option(
        None,
        "--as-of",
        help="Freeze date for --universe-mode fixed (YYYY-MM-DD). Defaults to --date.",
    ),
) -> None:
    """Print the constituents of ``index`` on a given date.

    In ``point-in-time`` mode this is the honest, date-aware universe. In
    ``fixed`` mode the constituents are frozen at ``--as-of`` and returned
    regardless of ``--date`` — the fallback that removes selection bias only.
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.universe.membership import (
        UniverseMode,
        freeze_fixed_universe,
        universe_as_of,
    )

    query_date = _parse_date(date_)
    engine = create_db_engine()
    factory = create_session_factory(engine)
    session = factory()
    try:
        if universe_mode == UniverseMode.FIXED.value:
            frozen_at = _parse_date(as_of) if as_of else query_date
            resolver = freeze_fixed_universe(session, index, frozen_at)
            members = resolver.constituents(query_date)
            typer.echo(
                f"# {index} (fixed @ {frozen_at}) — {len(members)} constituents "
                f"[selection bias removed; delisting attrition NOT removed]"
            )
        elif universe_mode == UniverseMode.POINT_IN_TIME.value:
            members = universe_as_of(session, index, query_date)
            typer.echo(
                f"# {index} @ {query_date} (point-in-time) — "
                f"{len(members)} constituents"
            )
        else:
            raise typer.BadParameter(
                f"unknown --universe-mode {universe_mode!r}; use point-in-time | fixed"
            )
        for security_id in sorted(members):
            typer.echo(security_id)
    finally:
        session.close()


features_app = typer.Typer(
    help="Feature engineering: DAG registry build + benchmark (PRD Module 3)."
)
app.add_typer(features_app, name="features")


@features_app.command("build")
def features_build(
    universe: str = typer.Option(
        ..., "--universe", help="Index whose members to build features for."
    ),
    features: str = typer.Option(
        "all",
        "--features",
        help="'all', a category name, or a comma-separated list of feature names.",
    ),
    start: str = typer.Option(None, "--start", help="Inclusive start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Inclusive end (YYYY-MM-DD)."),
    n_jobs: int = typer.Option(
        -1, "--jobs", help="Parallel workers for per-security compute (-1 = all cores)."
    ),
) -> None:
    """Build features for a universe and write partitioned parquet to data/features/.

    Example::

        quantlab features build --universe NIFTY500 --features all
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.feature_engineering.build import build_and_write

    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None

    engine = create_db_engine()
    factory = create_session_factory(engine)
    session = factory()
    try:
        rows, root = build_and_write(
            session,
            universe,
            features=features,
            start=start_date,
            end=end_date,
            n_jobs=n_jobs,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        session.close()

    typer.echo(f"Wrote {rows} feature rows to {root} (partitioned by year).")


@features_app.command("benchmark")
def features_benchmark(
    securities: int = typer.Option(
        500, "--securities", help="Synthetic securities to benchmark."
    ),
    days: int = typer.Option(
        2520, "--days", help="Trading days per security (2520 ~= 10 years)."
    ),
    write_doc: bool = typer.Option(
        True, "--write-doc/--no-write-doc", help="Write docs/performance.md."
    ),
) -> None:
    """Benchmark a full feature build, profile it, and write docs/performance.md."""

    from quantlab.feature_engineering.benchmark import (
        profile_build,
        run_benchmark,
        write_performance_doc,
    )

    typer.echo(f"Benchmarking {securities} securities x {days} days...")
    result = run_benchmark(n_securities=securities, n_days=days)
    typer.echo(
        f"  caching speed-up:       {result.caching_speedup:.1f}x "
        f"({result.naive_seconds:.2f}s -> {result.cached_seconds:.2f}s)"
    )
    typer.echo(f"  vectorisation speed-up: {result.vectorisation_speedup:.1f}x")
    typer.echo(
        f"  parallel speed-up:      {result.parallel_speedup:.1f}x "
        f"({result.serial_seconds:.2f}s -> {result.parallel_seconds:.2f}s)"
    )
    if write_doc:
        profile_text = profile_build()
        path = write_performance_doc(result, profile_text)
        typer.echo(f"Wrote {path}")


research_app = typer.Typer(
    help="Research integrity: the log, multiple-testing corrections, pre-registration."
)
app.add_typer(research_app, name="research")


@research_app.command("log")
def research_log_cmd(
    summary: bool = typer.Option(
        False,
        "--summary",
        help="Aggregate view: trials count, holdout runs, best Sharpe.",
    ),
    strategy: str = typer.Option(
        None, "--strategy", help="Restrict the listing to one strategy name."
    ),
    limit: int = typer.Option(
        20, "--limit", help="Rows to list (ignored with --summary)."
    ),
    all_statuses: bool = typer.Option(
        False, "--all", help="Include running and failed rows, not just completed ones."
    ),
) -> None:
    """Query the research log.

    Examples::

        quantlab research log --summary
        quantlab research log --strategy momentum_12_1 --limit 50
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.research.log import ResearchLog

    engine = create_db_engine()
    factory = create_session_factory(engine)
    session = factory()
    try:
        research_log = ResearchLog(session)
        if summary:
            typer.echo(research_log.summary().render())
            return

        entries = research_log.entries(
            strategy_name=strategy,
            status=None if all_statuses else "complete",
            limit=limit,
        )
        if not entries:
            typer.echo("No runs logged yet.")
            return
        typer.echo(
            f"{'run_id':10} {'when':20} {'strategy':24} {'split':18} "
            f"{'sharpe':>8} {'status':9} flags"
        )
        for entry in entries:
            flags = []
            if entry.holdout_touched:
                flags.append("HOLDOUT")
            flags.append(
                "exploratory" if entry.exploratory else f"prereg:{entry.hypothesis_id}"
            )
            if entry.git_dirty:
                flags.append("dirty")
            sharpe = "-" if entry.sharpe is None else f"{entry.sharpe:8.3f}"
            when = entry.created_at.isoformat(sep=" ", timespec="seconds")[:19]
            typer.echo(
                f"{entry.run_id[:10]} {when:20} {entry.strategy_name[:24]:24} "
                f"{entry.split[:18]:18} {sharpe:>8} {entry.status:9} {' '.join(flags)}"
            )
        typer.echo("")
        typer.echo(
            f"Trials in log (all strategies, completed): {research_log.trials()}. "
            "Any Sharpe above is undeflated — see 'quantlab research deflate'."
        )
    finally:
        session.close()


@research_app.command("deflate")
def research_deflate_cmd(
    strategy: str = typer.Option(
        None, "--strategy", help="Deflate this strategy's best logged run."
    ),
    run_id: str = typer.Option(None, "--run-id", help="Deflate one specific run."),
    scope_strategy: bool = typer.Option(
        False,
        "--scope-strategy",
        help=(
            "Count trials only within this strategy name instead of the whole log. "
            "Less conservative: strategy and model search are one selection process."
        ),
    ),
    simulate_null: bool = typer.Option(
        True,
        "--simulate-null/--no-simulate-null",
        help="Also simulate the expected maximum Sharpe of an edgeless search.",
    ),
    simulations: int = typer.Option(
        1000, "--simulations", help="Monte Carlo paths for the null distribution."
    ),
) -> None:
    """Deflate a logged Sharpe against the trials count in ``research_log``.

    Example::

        quantlab research deflate --strategy momentum_12_1
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.research.deflated import (
        DeflationError,
        deflate_run,
        deflate_strategy,
        null_max_sharpe_for_run,
    )
    from quantlab.research.log import ResearchLogError

    if not strategy and not run_id:
        raise typer.BadParameter("provide --strategy or --run-id")

    engine = create_db_engine()
    factory = create_session_factory(engine)
    session = factory()
    try:
        from quantlab.research.log import ResearchLog

        research_log = ResearchLog(session)
        scope = strategy if (scope_strategy and strategy) else None
        try:
            if run_id:
                result = deflate_run(research_log, run_id, scope_strategy=scope)
            else:
                result = deflate_strategy(research_log, strategy, scope_strategy=scope)
        except (ResearchLogError, DeflationError) as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        typer.echo(result.render())
        if simulate_null and result.run_id is not None:
            typer.echo("")
            null = null_max_sharpe_for_run(
                research_log,
                result.run_id,
                scope_strategy=scope,
                n_simulations=simulations,
            )
            typer.echo(null.render(result.observed_sharpe))
    finally:
        session.close()


@research_app.command("prereg")
def research_prereg_cmd(
    validate_only: bool = typer.Option(
        False,
        "--validate",
        help="Validate docs/hypotheses.md and exit non-zero on error.",
    ),
    check_log: bool = typer.Option(
        True,
        "--check-log/--no-check-log",
        help="Also check logged runs against the hypotheses they claim to test.",
    ),
    no_git: bool = typer.Option(
        False, "--no-git", help="Skip git-history checks (faster; less evidence)."
    ),
) -> None:
    """Validate the pre-registration document.

    Example::

        quantlab research prereg --validate
    """

    from quantlab.research.preregister import validate, validate_against_log

    report = validate(check_git=not no_git)
    typer.echo(report.render())

    issues = []
    if check_log:
        try:
            from quantlab.db.session import create_db_engine, create_session_factory
            from quantlab.research.log import ResearchLog

            engine = create_db_engine()
            factory = create_session_factory(engine)
            session = factory()
            try:
                entries = ResearchLog(session).entries(status=None)
                issues = validate_against_log(entries)
            finally:
                session.close()
        except Exception as exc:
            typer.echo(f"\n(could not cross-check research_log: {exc})", err=True)
        if issues:
            typer.echo("")
            typer.echo("Logged runs vs. registered hypotheses:")
            for issue in issues:
                typer.echo(f"  {issue.render()}")

    failed = not report.ok or any(i.severity == "error" for i in issues)
    if validate_only and failed:
        raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
