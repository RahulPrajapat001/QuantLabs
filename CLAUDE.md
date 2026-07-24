# CLAUDE.md

Guidance for Claude Code (and any contributor) working in this repository. Read this before writing code. The authoritative specification is [`docs/PRD.md`](docs/PRD.md); this file distills the parts that constrain day-to-day implementation.

---

## 1. PROJECT

**QuantLab** is a quantitative research platform for **Indian equities (NSE)**. It ingests historical daily OHLCV, constructs point-in-time universes (NIFTY 50/100/500), engineers 70+ alpha factors, evaluates statistical relationships, backtests strategies under realistic Indian transaction costs, analyses portfolio risk, and serves results through a FastAPI-backed web interface (Next.js product UI + Streamlit researcher dashboard).

The platform exists to answer a research question, not to demonstrate features. Its deliverable is a **defensible research finding** — the software is the instrument that produces it. A documented negative result is a success; an impressive dashboard reporting an unverifiable Sharpe is not.

It is built to structurally defeat the four failure modes that inflate retail quant results: **survivorship bias** (point-in-time index membership from reconstitution filings), **look-ahead bias** (event-driven backtester, truncated-series feature tests, `as_of` timestamps), **multiple testing** (automatic `research_log`, deflated Sharpe, PBO), and **unrealistic costs** (itemised STT / stamp duty / GST / circuit limits). US providers exist only to prove the `DataProvider` abstraction — **no research runs on US data**. History spans 2015-01-01 onward. Scope: linear, regularised linear, and tree-ensemble models only.

---

## 2. ARCHITECTURE DECISIONS (fixed — do not revisit)

These are settled. Implementations conform; they are not reopened for discussion.

- **Immutable raw landing.** Raw vendor data lands immutable at `data/raw/{source}/{ticker}/{year}.parquet`. A result is **never** reproduced by re-downloading.
- **Cleaned storage.** Cleaned data goes to **SQLite** (dev) → **PostgreSQL** (M4) via **SQLAlchemy** (Alembic migrations). Store `close` and `adj_close` **separately**, plus a `corporate_actions` table. Every row carries an **`as_of` timestamp** for point-in-time correctness.
- **Provider abstraction.** All data access goes through an abstract **`DataProvider`** base class. **No module imports `yfinance`, `polygon`, or any vendor SDK directly.**
- **Stable identity.** Securities are identified by a stable internal **`security_id`**, never by ticker. A **`ticker_history`** table maps tickers to `security_id`s over time. **All joins use `security_id`.**
- **Point-in-time universe.** Universe membership is point-in-time, sourced from index reconstitution history. **Never** use current index constituents for historical backtests.
- **Feature registry (DAG).** Features are a **registry of declared objects** (name, dependencies, lookback, category, function) resolved as a **DAG** in topological order. Not a script of pandas calls.
- **Event-driven backtester.** The backtester is **event-driven, not vectorized**. Default **T+1 open** execution. Vectorized backtests hide look-ahead bugs.
- **Alpha evaluation is separate.** Alpha evaluation (IC, rank IC, decay, turnover) runs **before and separately from** strategy backtesting. A factor that fails IC decay never reaches the strategy builder.

---

## 3. RESEARCH INTEGRITY RULES (binding)

This section takes precedence over feature completeness. A platform that produces indefensible numbers has negative value.

- **HOLDOUT.** `2024-01-01` onward is **locked**. No result computed on holdout data may inform any strategy, parameter, or feature decision. Holdout is evaluated **exactly once, at the end**. Enforced in code: `DataHandler` refuses to serve data at/past `HOLDOUT_START` unless constructed `holdout=True`, which additionally requires `QUANTLAB_UNLOCK_HOLDOUT=1` (deliberately absent from `.env.example`).
- **Splits.**
  - Train: `2015-01-01` → `2021-12-31`
  - Validation: `2022-01-01` → `2023-12-31`
  - Holdout: `2024-01-01` onward
- **Research log.** Every backtest run writes to a **`research_log`** table **automatically** — from inside the engine, not via an omittable decorator. A run that fails to log **raises**. No exceptions.
- **Sharpe reporting.** Any reported Sharpe ratio **must** be accompanied by the **number of trials** that preceded it and its **deflated Sharpe ratio**. Trial counts are read from `research_log`, never hand-entered. This is enforced at the API boundary and the UI component contract (a `SharpeDisplay` accepting a bare number must not typecheck).
- **Pre-registration.** Strategies are pre-registered in [`docs/hypotheses.md`](docs/hypotheses.md) **before** testing. Anything found afterward is labelled **exploratory** in the research log.

---

## 4. SCOPE CUTS (do not build)

Deliberate cuts, not omissions:

- **LSTM / GRU / Transformer** (any deep sequence model)
- **Black-Litterman** optimisation
- **News / sentiment** data

---

## 5. ENGINEERING STANDARDS

- **Python 3.11**, full type hints, **mypy strict on `src/`**.
- **pytest** for testing.
- **Pydantic Settings** for all configuration — **no hardcoded paths or credentials anywhere**.
- **structlog** for structured logging.
- **ruff** for lint/format; **pre-commit** enforced.
- CI (GitHub Actions) runs **ruff + mypy + pytest** on every PR.

---

## 6. MILESTONES

| Milestone | Scope | Phases |
|---|---|---|
| **M1 — Data Platform** | Ingestion, universe construction, validation, feature registry | P1–P3 |
| **M2 — Research Engine** | Statistics, alpha research, integrity layer, backtester, risk, portfolio, ML | P4–P6 |
| **M3 — Interface** | Design system, API, frontend, Streamlit, design review | P7–P8d |
| **M4 — Production** | PostgreSQL, Docker, CI/CD, docs, holdout evaluation, research paper | P9 |

Each milestone ships something that works end to end. Nothing is built "for later."

---

## Repository layout

```
src/quantlab/
├── config.py            # Pydantic settings, HOLDOUT_START
├── data/                # providers, storage, validation
├── universe/            # point-in-time membership
├── db/                  # SQLAlchemy models, Alembic
├── feature_engineering/ # registry, DAG, feature definitions
├── stats/               # statistical tests
├── factors/             # alpha research, IC evaluation
├── strategies/          # strategy implementations
├── backtesting/         # engine, cost models, execution
├── risk/                # VaR, stress tests, Monte Carlo
├── portfolio/           # optimisers
├── ml_models/           # purged CV, models, SHAP
├── research/            # log, deflated Sharpe, pre-registration
├── api/                 # FastAPI
├── reports/             # tearsheet generation
└── utils/
```
