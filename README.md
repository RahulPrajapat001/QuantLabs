# QuantLab

A quantitative research platform for **Indian equities (NSE)**. QuantLab ingests
historical daily OHLCV, constructs point-in-time universes (NIFTY 50/100/500),
engineers alpha factors as a DAG registry, evaluates statistical relationships,
backtests strategies under realistic Indian transaction costs, analyses portfolio
risk, and serves results through a FastAPI-backed interface.

The deliverable is a **defensible research finding**, not a feature demo. See
[`docs/PRD.md`](docs/PRD.md) for the full specification and [`CLAUDE.md`](CLAUDE.md)
for the fixed architectural decisions and binding research-integrity rules.

## Quickstart (development)

Requires **Python 3.11** and [`uv`](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

pytest            # run the test suite
ruff check .      # lint
mypy              # strict type-check on src/
```

Copy `.env.example` to `.env` before running any data commands.

## Status

Scaffold only — package structure, tooling, and CI. Business logic is implemented
per milestone (M1 Data Platform → M4 Production).

## Limitations

Stated up front (see PRD §11): no intraday data, no point-in-time fundamentals,
no borrow costs, universe resolution below 100%, self-built NSE adjustment logic,
single market / single asset class.
