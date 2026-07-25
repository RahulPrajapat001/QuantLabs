"""QuantLab command-line entry point.

A Typer application. Commands are implemented per-milestone; ``ingest`` is the
first (M1). See ``docs/PRD.md`` §5.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import structlog
import typer

from quantlab.config import get_settings

if TYPE_CHECKING:  # pandas is imported lazily inside commands to keep CLI startup fast
    import pandas as pd

    from quantlab.ml_models.dataset import MLDataset
    from quantlab.ml_models.validation import Splitter

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


backtest_app = typer.Typer(help="Event-driven backtesting (PRD Modules 7 and 8).")
app.add_typer(backtest_app, name="backtest")


def _load_bars(
    universe: str, start: date | None, end: date | None
) -> tuple[pd.DataFrame, str]:
    """Load a bar panel for a universe from the database.

    Returns ``(frame, description)``. Raises ``typer.BadParameter`` when the
    universe has no members or no price data — a backtest on an empty panel would
    otherwise report a flat equity curve as though it were a result.
    """

    import pandas as pd
    from sqlalchemy import select

    from quantlab.db.models import OHLCVDaily
    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.universe.membership import universe_as_of

    engine = create_db_engine()
    factory = create_session_factory(engine)
    session = factory()
    try:
        as_of = end or date.today()
        members = universe_as_of(session, universe, as_of)
        if not members:
            raise typer.BadParameter(
                f"no point-in-time members for {universe} as of {as_of}. "
                f"Build the universe first: quantlab universe build --index {universe}"
            )
        stmt = select(
            OHLCVDaily.date,
            OHLCVDaily.security_id,
            OHLCVDaily.open,
            OHLCVDaily.high,
            OHLCVDaily.low,
            OHLCVDaily.close,
            OHLCVDaily.adj_close,
            OHLCVDaily.volume,
        ).where(OHLCVDaily.security_id.in_(sorted(members)))
        if start is not None:
            stmt = stmt.where(OHLCVDaily.date >= start)
        if end is not None:
            stmt = stmt.where(OHLCVDaily.date <= end)
        rows = session.execute(stmt).all()
    finally:
        session.close()

    if not rows:
        raise typer.BadParameter(
            f"no OHLCV rows for {universe} in the requested window. Ingest data "
            "first: quantlab ingest --tickers ..."
        )
    frame = pd.DataFrame(
        rows,
        columns=[
            "date",
            "security_id",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ],
    )
    return frame, f"{len(members)} securities, {len(frame)} bars"


@backtest_app.command("run")
def backtest_run(
    strategy: str = typer.Option(
        ..., "--strategy", help="Strategy name (see 'quantlab backtest strategies')."
    ),
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to trade."),
    start: str = typer.Option(None, "--start", help="Inclusive start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Inclusive end (YYYY-MM-DD)."),
    cost_model: str = typer.Option(
        "IndianEquityCostModel",
        "--cost-model",
        help="ZeroCostModel | GenericBpsModel | IndianEquityCostModel.",
    ),
    capital: float = typer.Option(1_000_000.0, "--capital", help="Initial capital."),
    hypothesis: str = typer.Option(
        None,
        "--hypothesis",
        help="Pre-registered hypothesis id. Omit and the run is logged exploratory.",
    ),
) -> None:
    """Run one backtest. Writes to ``research_log`` automatically.

    Example::

        quantlab backtest run --strategy momentum_12_1 --universe NIFTY500 \
            --start 2015-01-01 --end 2021-12-31
    """

    from quantlab.backtesting.costs import default_cost_models
    from quantlab.backtesting.data_handler import DataHandler
    from quantlab.backtesting.engine import BacktestConfig, BacktestEngine
    from quantlab.backtesting.portfolio import Portfolio, RiskLimits
    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.research.log import ResearchLog
    from quantlab.strategies import default_strategies

    strategies = default_strategies()
    if strategy not in strategies:
        raise typer.BadParameter(
            f"unknown strategy {strategy!r}; available: {', '.join(sorted(strategies))}"
        )
    models = default_cost_models()
    if cost_model not in models:
        raise typer.BadParameter(
            f"unknown cost model {cost_model!r}; available: {', '.join(models)}"
        )

    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None
    bars, description = _load_bars(universe, start_date, end_date)
    typer.echo(f"Loaded {description}.")

    engine_db = create_db_engine()
    session = create_session_factory(engine_db)()
    try:
        engine = BacktestEngine(
            ResearchLog(session),
            strategy=strategies[strategy],
            data_handler=DataHandler(bars, start=start_date, end=end_date),
            portfolio=Portfolio(
                initial_capital=capital,
                limits=RiskLimits(max_positions=10, max_position_weight=0.15),
                target_positions=10,
            ),
            config=BacktestConfig(
                initial_capital=capital,
                cost_model=models[cost_model],
                universe="point-in-time",
                index_name=universe,
                hypothesis_id=hypothesis,
            ),
        )
        backtest = engine.run()
        assert engine.metrics is not None
        typer.echo("")
        typer.echo(engine.metrics.render())
        typer.echo("")
        typer.echo(f"run_id: {backtest.run_id}")
        typer.echo(
            "The Sharpe above is UNDEFLATED. To report it: "
            f"quantlab research deflate --run-id {backtest.run_id}"
        )
    finally:
        session.close()


@backtest_app.command("strategies")
def backtest_strategies() -> None:
    """List available strategies and their parameters."""

    from quantlab.strategies import default_strategies

    for name, strategy in sorted(default_strategies().items()):
        parameters = ", ".join(f"{k}={v}" for k, v in strategy.parameters().items())
        typer.echo(f"{name:22s} warmup={strategy.warmup:4d}  {parameters}")


@backtest_app.command("cost-sensitivity")
def backtest_cost_sensitivity(
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to trade."),
    start: str = typer.Option(None, "--start", help="Inclusive start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Inclusive end (YYYY-MM-DD)."),
    capital: float = typer.Option(1_000_000.0, "--capital", help="Initial capital."),
    output: str = typer.Option(
        None, "--output", help="Defaults to docs/research/cost_sensitivity.md."
    ),
) -> None:
    """Run every strategy under all three cost models and write the report.

    A required deliverable (PRD Module 8). Strategies that die under realistic
    Indian costs are reported as findings, not dropped.

    Example::

        quantlab backtest cost-sensitivity --universe NIFTY500 --end 2021-12-31
    """

    from pathlib import Path

    from quantlab.backtesting.cost_sensitivity import (
        run_cost_sensitivity,
        write_report,
    )
    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.research.log import ResearchLog
    from quantlab.strategies import default_strategies

    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None
    bars, description = _load_bars(universe, start_date, end_date)
    typer.echo(f"Loaded {description}. Running every strategy x 3 cost models...")

    engine_db = create_db_engine()
    session = create_session_factory(engine_db)()
    try:
        report = run_cost_sensitivity(
            ResearchLog(session),
            default_strategies(),
            bars,
            initial_capital=capital,
            index_name=universe,
        )
    finally:
        session.close()

    path = write_report(report, Path(output) if output else None)
    typer.echo("")
    typer.echo(report.pivot("sharpe").to_string())
    typer.echo("")
    survivors = report.survivors()
    if survivors:
        typer.echo(f"Survive realistic costs: {', '.join(survivors)}")
    else:
        typer.echo(
            "No strategy clears the survival threshold under Indian statutory "
            "costs. That is the finding, and it is reported as one."
        )
    if report.casualties():
        typer.echo(f"Do not survive: {', '.join(report.casualties())}")
    typer.echo("")
    typer.echo(f"Wrote {path}")
    typer.echo(
        f"research_log trials now: {report.trials_after} — every Sharpe above must "
        "be deflated against that count."
    )


risk_app = typer.Typer(
    help="Risk analytics: VaR, stress testing, exposure (PRD Module 9)."
)
app.add_typer(risk_app, name="risk")


@risk_app.command("stress-plan")
def risk_stress_plan(
    padding: int = typer.Option(
        30, "--padding", help="Calendar days of padding either side."
    ),
) -> None:
    """Print the date ranges to ingest so every stress window can be replayed.

    These ranges are retained regardless of any backtest window. The GFC starts in
    2007, seven years before ``DATA_START``, so ordinary ingestion never fetches it.
    """

    from quantlab.risk.stress import stress_window_ingest_plan

    typer.echo(stress_window_ingest_plan(padding_days=padding).render())


@risk_app.command("stress-coverage")
def risk_stress_coverage() -> None:
    """Report which stress windows have ingested data.

    Run this before trusting a stress test.
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.risk.stress import stress_coverage

    engine = create_db_engine()
    session = create_session_factory(engine)()
    try:
        coverages = stress_coverage(session)
    finally:
        session.close()

    typer.echo("Stress window coverage:")
    for coverage in coverages:
        typer.echo(coverage.render())
    missing = [c.window.slug for c in coverages if not c.available]
    if missing:
        typer.echo("")
        typer.echo(
            f"{len(missing)} of {len(coverages)} windows have no data: "
            f"{', '.join(missing)}. Stress testing will refuse to run on them rather "
            "than substitute a synthetic shock — see 'quantlab risk stress-plan'."
        )


@risk_app.command("stress")
def risk_stress(
    universe: str = typer.Option(
        None, "--universe", help="Equal-weight the point-in-time members of this index."
    ),
    as_of: str = typer.Option(
        None, "--as-of", help="Membership date for --universe (YYYY-MM-DD)."
    ),
) -> None:
    """Replay a portfolio through every stress window that has real ingested data.

    Windows without data are reported as untested, never quietly skipped.
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.risk.stress import stress_test

    engine = create_db_engine()
    session = create_session_factory(engine)()
    try:
        weights = None
        if universe:
            from quantlab.universe.membership import universe_as_of

            when = _parse_date(as_of) if as_of else date.today()
            members = sorted(universe_as_of(session, universe, when))
            if not members:
                raise typer.BadParameter(
                    f"{universe} has no point-in-time members as of {when}"
                )
            weights = dict.fromkeys(members, 1.0 / len(members))
            typer.echo(
                f"Equal-weighting {len(members)} members of {universe} @ {when}."
            )
        report = stress_test(session, weights=weights)
    finally:
        session.close()

    typer.echo("")
    typer.echo(report.render())
    if not report.complete:
        raise typer.Exit(code=1)


@risk_app.command("var")
def risk_var(
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to analyse."),
    as_of: str = typer.Option(None, "--as-of", help="Point-in-time date (YYYY-MM-DD)."),
    lookback: int = typer.Option(
        504, "--lookback", help="Trading days of history to estimate from."
    ),
) -> None:
    """VaR and Expected Shortfall for an equal-weighted point-in-time universe.

    All three estimators are shown, because their disagreement is the information: a
    Gaussian VaR materially below the other two means the portfolio's tail is worse
    than a normal approximation admits.

    Example::

        quantlab risk var --universe NIFTY500 --as-of 2021-12-31
    """

    from quantlab.risk.var import var_comparison

    when = _parse_date(as_of) if as_of else date.today()
    bars, description = _load_bars(universe, None, when)
    typer.echo(f"Loaded {description}.")

    wide = bars.pivot(
        index="date", columns="security_id", values="adj_close"
    ).sort_index()
    returns = wide.pct_change().dropna(how="all").tail(lookback)
    portfolio = returns.mean(axis=1).dropna()
    if len(portfolio) < 30:
        raise typer.BadParameter(
            f"only {len(portfolio)} usable return observations; VaR needs at least 30"
        )

    typer.echo(
        f"Equal-weighted across {returns.shape[1]} securities, "
        f"{len(portfolio)} days to {when}."
    )
    typer.echo("")
    typer.echo(var_comparison(portfolio).render())


portfolio_app = typer.Typer(help="Portfolio optimisation (PRD Module 10).")
app.add_typer(portfolio_app, name="portfolio")


@portfolio_app.command("optimise")
def portfolio_optimise(
    universe: str = typer.Option(
        "NIFTY500", "--universe", help="Index to optimise over."
    ),
    method: str = typer.Option(
        "minimum_variance",
        "--method",
        help=(
            "equal_weight | inverse_volatility | risk_parity | minimum_variance | "
            "maximum_sharpe"
        ),
    ),
    as_of: str = typer.Option(None, "--as-of", help="Point-in-time date (YYYY-MM-DD)."),
    lookback: int = typer.Option(
        252, "--lookback", help="Trading days of history to use."
    ),
    covariance: str = typer.Option(
        "ledoit_wolf",
        "--covariance",
        help="sample | ledoit_wolf | constant_correlation. Shrunk by default.",
    ),
    max_weight: float = typer.Option(0.10, "--max-weight", help="Per-position cap."),
) -> None:
    """Optimise weights over the point-in-time universe.

    Example::

        quantlab portfolio optimise --universe NIFTY500 --method risk_parity \
            --as-of 2021-12-31
    """

    from quantlab.portfolio.covariance import estimate_covariance
    from quantlab.portfolio.optimisers import (
        PortfolioConstraints,
        equal_weight,
        inverse_volatility,
        maximum_sharpe,
        minimum_variance,
        risk_parity,
    )

    when = _parse_date(as_of) if as_of else date.today()
    bars, description = _load_bars(universe, None, when)
    typer.echo(f"Loaded {description}.")

    import pandas as pd

    wide = bars.pivot(
        index="date", columns="security_id", values="adj_close"
    ).sort_index()
    returns = wide.pct_change().dropna(how="all").tail(lookback)
    returns = returns.dropna(axis=1, thresh=max(20, lookback // 2))
    if returns.shape[1] < 2:
        raise typer.BadParameter(
            f"only {returns.shape[1]} securities have enough history in the last "
            f"{lookback} days"
        )
    typer.echo(f"Optimising over {returns.shape[1]} securities, {len(returns)} days.")

    estimate = estimate_covariance(returns, method=covariance)  # type: ignore[arg-type]
    typer.echo("")
    typer.echo(estimate.render())

    constraints = PortfolioConstraints(max_weight=max_weight)
    if method == "equal_weight":
        result = equal_weight(returns.columns, covariance=estimate)
    elif method == "inverse_volatility":
        result = inverse_volatility(estimate, constraints=constraints)
    elif method == "risk_parity":
        result = risk_parity(estimate, max_weight=max_weight)
    elif method == "minimum_variance":
        result = minimum_variance(estimate, constraints=constraints)
    elif method == "maximum_sharpe":
        result = maximum_sharpe(returns.mean(), estimate, constraints=constraints)
    else:
        raise typer.BadParameter(f"unknown method {method!r}")

    typer.echo("")
    typer.echo(result.render())
    assert isinstance(returns, pd.DataFrame)


@portfolio_app.command("rebalance")
def portfolio_rebalance(
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to rebalance."),
    as_of: str = typer.Option(..., "--as-of", help="Rebalance date (YYYY-MM-DD)."),
) -> None:
    """Show the rebalance plan against the point-in-time universe.

    Securities that have left the index are exited explicitly and reported — the
    point being that they are never held silently.
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.portfolio.rebalance import rebalance_plan

    when = _parse_date(as_of)
    engine = create_db_engine()
    session = create_session_factory(engine)()
    try:
        plan = rebalance_plan(session, universe, when)
    finally:
        session.close()
    typer.echo(plan.render())


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


ml_app = typer.Typer(help="Machine learning under purged validation (PRD Module 11).")
app.add_typer(ml_app, name="ml")


def _load_ml_dataset(
    universe: str,
    *,
    features: str | None,
    start: date | None,
    end: date | None,
    target: str,
    horizon: int,
) -> MLDataset:
    """Assemble a dataset from the feature parquet, prices, and index membership.

    Kept in one place so ``train``, ``search`` and ``folds`` all build the dataset
    identically — three subtly different loaders would make their numbers
    incomparable.
    """

    import pandas as pd

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.feature_engineering.build import (
        load_ohlcv_panel,
        read_feature_panel,
        universe_by_date,
    )
    from quantlab.ml_models.dataset import TargetKind, TargetSpec, build_dataset

    engine = create_db_engine()
    session = create_session_factory(engine)()
    try:
        prices = load_ohlcv_panel(session, universe, start=start, end=end)
        if prices.empty:
            raise typer.BadParameter(
                f"no OHLCV rows for any member of {universe}. Ingest data and "
                "build the universe first."
            )
        panel = read_feature_panel(start=start, end=end)
        if panel.empty:
            raise typer.BadParameter(
                "the feature dataset is empty. Build it first: "
                f"quantlab features build --index {universe}"
            )
        merged = panel.merge(
            prices[["date", "security_id", "adj_close"]],
            on=["date", "security_id"],
            how="inner",
        )
        dates = sorted({d.date() for d in pd.to_datetime(merged["date"])})
        members = universe_by_date(session, universe, dates)
    finally:
        session.close()

    reserved = {"date", "security_id", "adj_close", "year"}
    names = (
        [f.strip() for f in features.split(",") if f.strip()]
        if features
        else [c for c in merged.columns if c not in reserved]
    )
    spec = TargetSpec(kind=TargetKind(target), horizon=horizon)
    return build_dataset(
        merged,
        features=names,
        universe=members,
        index_name=universe,
        target=spec,
    )


def _build_splitter(
    kind: str, *, n_splits: int, embargo: int, train_days: int | None
) -> Splitter:
    from quantlab.ml_models.validation import PurgedKFold, WalkForward

    if kind == "purged-kfold":
        return PurgedKFold(n_splits=n_splits, embargo_days=embargo)
    if kind == "walk-forward-expanding":
        return WalkForward(mode="expanding", n_splits=n_splits, gap_days=embargo)
    if kind == "walk-forward-rolling":
        if train_days is None:
            raise typer.BadParameter(
                "--train-days is required for a rolling walk-forward: the whole "
                "difference from the expanding variant is a fixed window length"
            )
        return WalkForward(
            mode="rolling",
            n_splits=n_splits,
            train_days=train_days,
            gap_days=embargo,
        )
    raise typer.BadParameter(
        f"unknown validation scheme {kind!r}: expected purged-kfold, "
        "walk-forward-expanding or walk-forward-rolling"
    )


def _split_for_selection(dataset: MLDataset) -> tuple[MLDataset, MLDataset | None]:
    """``(train, validation)`` when the range covers both, else ``(dataset, None)``.

    Model selection belongs on the validation split (``CLAUDE.md`` §3). When the
    requested range does not reach 2022 there is nothing to select on, and the
    caller falls back to the purged-CV score — which the report states rather
    than presenting a CV number as though it were a held-back one.
    """

    from quantlab.config import Split

    train_mask = dataset.split_mask(Split.TRAIN)
    validation_mask = dataset.split_mask(Split.VALIDATION)
    if not train_mask.any():
        return dataset, None
    if not validation_mask.any():
        return dataset.subset(train_mask), None
    return dataset.subset(train_mask), dataset.subset(validation_mask)


@ml_app.command("models")
def ml_models_cmd() -> None:
    """List the permitted model families and their declared search spaces."""

    from quantlab.ml_models.models import DEFAULT_SEARCH_SPACE, MODEL_FAMILIES

    typer.echo("Permitted model families (PRD Module 11):")
    for family in sorted(MODEL_FAMILIES):
        grid = DEFAULT_SEARCH_SPACE.get(family, {})
        size = 1
        for values in grid.values():
            size *= len(values)
        typer.echo(f"  {family:16s} default grid: {size} configuration(s)")
        for axis, values in sorted(grid.items()):
            typer.echo(f"      {axis}: {list(values)}")
    typer.echo("")
    typer.echo(
        "Deep sequence models (LSTM, GRU, Transformer) are a deliberate scope cut "
        "(CLAUDE.md §4) and cannot be built."
    )


@ml_app.command("folds")
def ml_folds(
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to use."),
    start: str = typer.Option(None, "--start", help="Inclusive start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Inclusive end (YYYY-MM-DD)."),
    validation: str = typer.Option(
        "purged-kfold",
        "--validation",
        help="purged-kfold | walk-forward-expanding | walk-forward-rolling",
    ),
    n_splits: int = typer.Option(5, "--n-splits", help="Number of folds."),
    embargo: int = typer.Option(
        10, "--embargo", help="Embargo (or walk-forward gap) in trading days."
    ),
    train_days: int = typer.Option(
        None, "--train-days", help="Rolling training window length."
    ),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days."),
) -> None:
    """Show the fold structure and how much each fold purged.

    Run this before trusting any result: a purged CV that removed nothing has
    protected nothing, and the counts are how that becomes visible.

    Example::

        quantlab ml folds --universe NIFTY500 --validation purged-kfold --embargo 10
    """

    from quantlab.ml_models.validation import render_folds

    dataset = _load_ml_dataset(
        universe,
        features=None,
        start=_parse_date(start) if start else None,
        end=_parse_date(end) if end else None,
        target="return",
        horizon=horizon,
    )
    splitter = _build_splitter(
        validation, n_splits=n_splits, embargo=embargo, train_days=train_days
    )
    typer.echo(dataset.describe())
    typer.echo(splitter.describe())
    typer.echo("")
    typer.echo(render_folds(splitter.split(dataset.times)))


@ml_app.command("train")
def ml_train(
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to use."),
    model: str = typer.Option(
        "ridge", "--model", help="ridge | lasso | random_forest | xgboost | lightgbm"
    ),
    target: str = typer.Option(
        "return", "--target", help="return (regression) | direction (classification)"
    ),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days."),
    features: str = typer.Option(
        None, "--features", help="Comma-separated features. Defaults to all."
    ),
    start: str = typer.Option(None, "--start", help="Inclusive start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Inclusive end (YYYY-MM-DD)."),
    validation: str = typer.Option(
        "purged-kfold",
        "--validation",
        help="purged-kfold | walk-forward-expanding | walk-forward-rolling",
    ),
    n_splits: int = typer.Option(5, "--n-splits", help="Number of folds."),
    embargo: int = typer.Option(10, "--embargo", help="Embargo in trading days."),
    train_days: int = typer.Option(
        None, "--train-days", help="Rolling training window length."
    ),
    gross: bool = typer.Option(
        False, "--gross", help="Report gross of costs as well as net."
    ),
    shap: bool = typer.Option(False, "--shap", help="Compute SHAP attribution."),
    hypothesis: str = typer.Option(
        None, "--hypothesis", help="Pre-registered hypothesis id (docs/hypotheses.md)."
    ),
) -> None:
    """Evaluate one model configuration, against both required baselines.

    Writes a ``research_log`` row and counts toward the trials total, whatever the
    result. Selection uses the validation split when the date range covers it.

    Example::

        quantlab ml train --universe NIFTY500 --model lightgbm --target direction \
            --validation walk-forward-expanding --embargo 10
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.ml_models.evaluate import evaluate_model
    from quantlab.ml_models.models import ModelSpec, Task
    from quantlab.ml_models.trading import TradingRule
    from quantlab.research.log import ResearchLog

    dataset = _load_ml_dataset(
        universe,
        features=features,
        start=_parse_date(start) if start else None,
        end=_parse_date(end) if end else None,
        target=target,
        horizon=horizon,
    )
    splitter = _build_splitter(
        validation, n_splits=n_splits, embargo=embargo, train_days=train_days
    )
    train, held_back = _split_for_selection(dataset)

    spec = ModelSpec(
        family=model,
        task=Task.REGRESSION if target == "return" else Task.CLASSIFICATION,
    )
    rule = TradingRule(cost_per_unit_traded=0.0 if gross else None)

    session = create_session_factory(create_db_engine())()
    try:
        research_log = ResearchLog(session)
        evaluation = evaluate_model(
            research_log,
            train,
            spec,
            splitter=splitter,
            rule=rule,
            validation=held_back,
            hypothesis_id=hypothesis,
            compute_shap=shap,
        )
        typer.echo(evaluation.render(research_log))
    finally:
        session.close()


@ml_app.command("search")
def ml_search(
    universe: str = typer.Option("NIFTY500", "--universe", help="Index to use."),
    families: str = typer.Option(
        None,
        "--families",
        help="Comma-separated model families. Defaults to all five.",
    ),
    target: str = typer.Option(
        "return", "--target", help="return (regression) | direction (classification)"
    ),
    horizon: int = typer.Option(1, "--horizon", help="Target horizon in days."),
    features: str = typer.Option(
        None, "--features", help="Comma-separated features. Defaults to all."
    ),
    start: str = typer.Option(None, "--start", help="Inclusive start (YYYY-MM-DD)."),
    end: str = typer.Option(None, "--end", help="Inclusive end (YYYY-MM-DD)."),
    validation: str = typer.Option(
        "purged-kfold",
        "--validation",
        help="purged-kfold | walk-forward-expanding | walk-forward-rolling",
    ),
    n_splits: int = typer.Option(5, "--n-splits", help="Number of folds."),
    embargo: int = typer.Option(10, "--embargo", help="Embargo in trading days."),
    train_days: int = typer.Option(
        None, "--train-days", help="Rolling training window length."
    ),
    top: int = typer.Option(10, "--top", help="Configurations to list."),
) -> None:
    """Search the declared hyper-parameter grids, logging every configuration.

    The report leads with the search space and the iteration count, because a
    winning score means nothing without the number of attempts behind it.

    Example::

        quantlab ml search --universe NIFTY500 --families xgboost,lightgbm \
            --target direction
    """

    from quantlab.db.session import create_db_engine, create_session_factory
    from quantlab.ml_models.search import run_search
    from quantlab.research.log import ResearchLog

    dataset = _load_ml_dataset(
        universe,
        features=features,
        start=_parse_date(start) if start else None,
        end=_parse_date(end) if end else None,
        target=target,
        horizon=horizon,
    )
    splitter = _build_splitter(
        validation, n_splits=n_splits, embargo=embargo, train_days=train_days
    )
    train, held_back = _split_for_selection(dataset)
    chosen = [f.strip() for f in families.split(",")] if families else None

    session = create_session_factory(create_db_engine())()
    try:
        research_log = ResearchLog(session)
        result = run_search(
            research_log,
            train,
            families=chosen,
            splitter=splitter,
            validation=held_back,
        )
        typer.echo(result.render(research_log, top=top))
    finally:
        session.close()


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind address."),
    port: int = typer.Option(8000, "--port", help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on change."),
) -> None:
    """Serve the API — the sole data source for both frontends (PRD Module 12).

    Binds to localhost by default. This API exposes an unauthenticated research
    database, so making it reachable from the network is a deliberate act, not a
    default.

    Example::

        quantlab serve --reload
        # OpenAPI docs at http://127.0.0.1:8000/docs
    """

    import uvicorn

    typer.echo(f"QuantLab API on http://{host}:{port}  (docs at /docs)")
    uvicorn.run(
        "quantlab.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=get_settings().log_level.lower(),
    )


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
