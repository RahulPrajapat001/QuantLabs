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


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
