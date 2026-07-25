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

## The two interfaces

QuantLab ships **two** frontends over one API. This is not duplication — it
mirrors how quant shops actually operate, where the person who wrote the
strategy and the person reading its results need different things.

| | **Streamlit** (`dashboard/`) | **Next.js** (`frontend/`) |
|---|---|---|
| Audience | **Researcher-facing** — the person who ran it | **Product-facing** — everyone else |
| Optimised for | Speed of iteration | Speed of comprehension |
| Shows | Raw numbers, full tables, every column | The finding, with its caveat attached |
| Cost of a new view | Minutes | Hours |
| Styling | Deliberately plain | Designed against `docs/DESIGN_SYSTEM.md` |

A researcher checking whether an IC survived a window change wants the number
and the table immediately; building that in React would cost an afternoon and
still be the worse tool. A reader who did not run the backtest needs the Sharpe,
its 312 trials and its deflated value to land as one statement — which takes
deliberate design that Streamlit cannot express.

Both consume the **same API** and the **same generated design tokens**, so a
number cannot differ between them, only its presentation. The three
statistical-honesty rules (Sharpe never bare, split badge always present,
unrealistic-cost marker) are enforced in **both**.

```bash
# API — required by both frontends
uv pip install -e ".[api,data,research]"
uvicorn quantlab.api.app:create_app --factory --reload

# Researcher dashboard
uv pip install -e ".[dashboard]"
streamlit run dashboard/Overview.py                 # http://localhost:8501

# Product UI
cd frontend && npm install && npm run dev      # http://localhost:3000
```

Set `QUANTLAB_API_URL` (Streamlit) or `NEXT_PUBLIC_API_URL` (Next.js) if the API
is not on `http://127.0.0.1:8000`.

### Design tokens

Colour, spacing and type are defined once in
[`scripts/build_design_tokens.py`](scripts/build_design_tokens.py) and generated
into `config/`, which both frontends import. Tailwind's default palette and
spacing scale are **replaced** rather than extended, so an ad-hoc colour or
spacing value is a build error rather than a review comment.

```bash
python scripts/build_design_tokens.py --check   # CI: fail on drift
```

## Status

M1–M3 implemented: data platform, research engine, API, product UI and
researcher dashboard. M4 (PostgreSQL, Docker, CI/CD, holdout evaluation, paper)
is outstanding.

## Limitations

Stated up front (see PRD §11): no intraday data, no point-in-time fundamentals,
no borrow costs, universe resolution below 100%, self-built NSE adjustment logic,
single market / single asset class.
