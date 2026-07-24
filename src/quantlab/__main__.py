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


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
