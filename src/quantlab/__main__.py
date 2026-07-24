"""QuantLab command-line entry point.

A minimal Typer application skeleton. Commands (``ingest``, ``validate``, …) are
implemented per-milestone; see ``docs/PRD.md`` §5.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="QuantLab — quantitative research platform for Indian equities.")


@app.command()
def version() -> None:
    """Print the installed QuantLab version."""
    from quantlab import __version__

    typer.echo(__version__)


def main() -> None:
    """Console-script entry point (see ``[project.scripts]`` in pyproject.toml)."""
    app()


if __name__ == "__main__":
    main()
