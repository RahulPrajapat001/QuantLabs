# QuantLab — Product Requirements Document

**Version:** 1.0
**Status:** Approved for build
**Primary market:** Indian equities (NSE)
**Repository:** `D:\QUANTLABS`

---

## 1. Overview

### 1.1 Elevator pitch

QuantLab is a quantitative research platform for Indian equities. It ingests historical market data, constructs point-in-time universes, engineers alpha factors, evaluates statistical relationships, backtests trading strategies under realistic transaction costs, analyses portfolio risk, and presents results through an API-backed web interface.

The platform exists to answer research questions, not to demonstrate features. Its output is a defensible research finding; the software is the instrument that produces it.

### 1.2 What distinguishes this from a typical retail quant project

Most portfolio-grade quant projects fail on four counts, all of which inflate reported performance:

1. **Survivorship bias** — backtesting on today's index constituents, which excludes every company that failed.
2. **Look-ahead bias** — features or signals that use information unavailable at the decision point.
3. **Multiple testing** — reporting the best of hundreds of untracked trials as if it were a single hypothesis.
4. **Unrealistic costs** — generic basis-point assumptions that ignore STT, stamp duty, GST, circuit limits, and liquidity.

QuantLab addresses each structurally, in code, not by documentation or convention. These mechanisms are described in Section 4 and are the core requirement of this specification.

### 1.3 Target users

- Quantitative researchers evaluating factor hypotheses
- Data analysts working with financial time series
- Finance students learning empirical methodology
- Technical reviewers assessing research rigour

---

## 2. Scope

### 2.1 In scope

| Area | Included |
|---|---|
| Market | Indian equities (NSE) — primary research market |
| Instruments | Equities, ETFs, index constituents |
| Frequency | Daily OHLCV |
| History | 2015-01-01 onward (minimum 10 years) |
| Universe | NIFTY 50, NIFTY 100, NIFTY 500 with point-in-time membership |
| Models | Linear, regularised linear, tree ensembles (RF, XGBoost, LightGBM) |

### 2.2 Explicitly out of scope

These are deliberate cuts, not omissions. Do not build them.

| Excluded | Reason |
|---|---|
| LSTM / GRU / Transformer models | Deep sequence models on daily OHLCV do not beat momentum baselines and signal inexperience to informed reviewers |
| Black-Litterman optimisation | Requires a views framework outside project scope |
| News / sentiment data | Documented as future work |
| Intraday data and execution | No reliable free source; fill assumptions would be unverifiable |
| Point-in-time fundamentals | Not freely available for NSE at required quality |
| Live or paper trading | Research platform, not an execution system |

### 2.3 Secondary market support

US equity providers (yfinance, Alpha Vantage, Polygon) are implemented to validate that the `DataProvider` abstraction genuinely abstracts. **No research runs on US data.** All reported findings use the Indian universe.

---

## 3. System architecture

### 3.1 Pipeline

```
Market Data Sources
        ↓
Ingestion  (immutable raw parquet)
        ↓
Validation  (quality findings, regression guard)
        ↓
Universe Construction  (point-in-time index membership)
        ↓
Feature Engineering  (DAG registry, 70+ features)
        ↓
Statistical Analysis
        ↓
Alpha Research  (IC, decay, turnover)
        ↓
Strategy Builder
        ↓
Backtesting Engine  (event-driven, realistic costs)
        ↓
Risk Analytics  →  Portfolio Optimisation
        ↓
API  →  Dashboard / Reports / Research Paper
```

### 3.2 Fixed architectural decisions

These are settled. Implementations must conform; do not revisit.

**Storage**
Raw vendor responses land immutable at `data/raw/{source}/{ticker}/{year}.parquet`. A result is never reproduced by re-downloading. Cleaned data goes to SQLite during development, PostgreSQL in production, via SQLAlchemy with Alembic migrations.

**Point-in-time correctness**
`close` and `adj_close` are stored separately alongside a `corporate_actions` table. Adjustment factors change retroactively when a split occurs; overwriting history makes yesterday's backtest irreproducible. Every row carries an `as_of` timestamp.

**Identity**
Securities are keyed by a stable internal `security_id`, never by ticker. A `ticker_history` table maps tickers to security_ids over time so renames resolve correctly. All joins use `security_id`.

**Provider abstraction**
All data access passes through an abstract `DataProvider` base class. No module imports `yfinance` or any vendor SDK directly.

**Universe**
Membership is point-in-time, derived from index reconstitution history. Current constituents are never used for historical backtests.

**Features**
A registry of declared objects — name, dependencies, lookback, category, function — resolved as a DAG in topological order. Not a script of pandas calls.

**Backtester**
Event-driven, not vectorised. Default T+1 open execution. Vectorised backtests silently permit trading on information from the bar being traded.

**Separation of concerns**
Alpha evaluation (IC, rank IC, decay, turnover) runs before and independently of strategy backtesting. A factor that fails IC decay never reaches the strategy builder.

---

## 4. Research integrity requirements

This section is binding and takes precedence over feature completeness. A platform that produces indefensible numbers has negative value.

### 4.1 Data splits

| Split | Range | Use |
|---|---|---|
| Train | 2015-01-01 → 2021-12-31 | Development, factor discovery, feature engineering |
| Validation | 2022-01-01 → 2023-12-31 | Walk-forward, model selection, strategy selection |
| Holdout | 2024-01-01 onward | Evaluated exactly once, at project end |

### 4.2 Holdout lockout

`HOLDOUT_START` is defined once in `src/quantlab/config.py`. The `DataHandler` refuses to serve data at or past that date unless constructed with `holdout=True`, which additionally requires the environment variable `QUANTLAB_UNLOCK_HOLDOUT=1`. That variable is deliberately absent from `.env.example`.

Every backtest records `holdout_touched`. Any holdout run is logged with its git commit hash so the sequence of events is auditable after the fact.

No result computed on holdout data may inform any strategy, parameter, or feature decision.

### 4.3 Research log

Every backtest run writes to a `research_log` table automatically — from inside the engine, not via a decorator the caller can omit. A run that fails to log raises.

Recorded fields: `run_id`, timestamp, git commit hash, strategy name, parameter dict, universe, index, date range, split, cost model, all output metrics, `holdout_touched`, and `hypothesis_id` where applicable.

### 4.4 Multiple testing correction

Required implementations:

- **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014) — given trials count, observed Sharpe, variance of Sharpes across trials, and return skew/kurtosis, returns the probability that true Sharpe exceeds zero.
- **Expected maximum Sharpe under the null** — simulate N zero-edge strategies matched on length and volatility; a real strategy failing to clear this distribution is noise.
- **Probability of Backtest Overfitting** via combinatorially symmetric cross-validation.

All three read the trials count directly from `research_log`. Hand-entered trial counts are not permitted.

### 4.5 Pre-registration

Hypotheses are written to `docs/hypotheses.md` before testing: id, economic rationale, factor or strategy, prediction, metric, threshold. Registered hypotheses are hashed and committed. Backtests link to a hypothesis id; anything else is marked exploratory in the research log.

### 4.6 Reporting rule

The report generator refuses to emit a Sharpe ratio without its accompanying trials count and deflated Sharpe. Absent either, it raises rather than printing a bare number. This is enforced at the API boundary and in the UI component contract.

### 4.7 Survivorship bias measurement

A required deliverable, not an optional check. Every headline strategy is run twice — once on the point-in-time universe, once on current constituents — and the delta in annual return, Sharpe, and max drawdown is reported in `docs/research/survivorship_bias.md`.

---

## 5. Functional modules

### Module 1 — Data ingestion

**Sources.** NSE bhavcopy archives (authoritative, unadjusted), yfinance with `.NS` suffix (convenient, unofficial), Alpha Vantage (backup). US providers implemented for abstraction validation only.

**Stored fields.** Open, High, Low, Close, Adjusted Close, Volume, plus `as_of`.

**Identity handling.** `securities` table holds `security_id`, ISIN where available, company name, exchange, listing date, delisting date, delisting reason. `ticker_history` maps tickers to security_ids with date ranges. Ingestion resolves ticker to security_id as of the requested date, never as of today.

**Delisted securities** retain full price history and a final price. They are never silently dropped.

**Requirements.** Rate limiting with exponential backoff, retry on transient failure, typed `ProviderError` on permanent failure, idempotent writes, incremental mode fetching only since last successful ingest.

**CLI.**
```
quantlab ingest --tickers RELIANCE.NS,TCS.NS --start 2015-01-01 --end 2025-12-31
quantlab ingest --incremental
```

---

### Module 2 — Universe construction

The single most consequential module for result validity.

**Schema.** `index_membership(security_id, index_name, start_date, end_date, source_document)`. The query `universe_as_of(index_name, date)` returns exactly the constituents on that date and must never consult present-day membership.

**Source.** NSE publishes semi-annual index reconstitution press releases (NIFTY 50 / 100 / 500, effective March and September) as PDFs. These are parsed into the membership schema, with raw PDFs cached in `data/raw/reference/` for reproducibility.

**Unresolved changes.** Where a reconstitution entry cannot be mapped to a `security_id`, it is written to `unresolved_membership_change` rather than dropped. The resolution rate is a reported metric.

**Fallback mode.** `--universe-mode fixed --as-of 2015-01-01` freezes constituents at a date. This removes selection bias but not delisting attrition; the distinction must be documented wherever the mode is used.

**Delisting in backtest.** A security delisted mid-backtest is liquidated at its last available price on the delisting date. Where `delisting_reason` indicates bankruptcy, a configurable recovery rate applies (default 0). Delisted prices are never forward-filled.

---

### Module 3 — Data validation

Automated checks producing structured `Finding` objects (severity, description, affected rows) persisted to `data_quality_findings`.

| Check | Detects |
|---|---|
| Missing values | Incomplete vendor responses |
| Calendar gaps | Sessions missing vs `exchange_calendars` NSE calendar |
| Duplicate dates | Ingestion faults |
| OHLC integrity | high < low, close outside [low, high], negative volume |
| Price outliers | Returns beyond N median absolute deviations |
| Unadjusted splits | Single-day return near a common split ratio with no matching corporate action |
| Stale prices | Identical close across consecutive sessions |
| Circuit-limit days | Moves equal to the applicable band — non-tradeable |

**Outputs.** `DataQualityReport` rendering to HTML and dict with a completeness percentage.

**Regression guard.** `quantlab validate --fail-on-regression` compares against the previous run and exits non-zero if severity counts worsen. This is what the scheduled refresh invokes.

---

### Module 4 — Feature engineering

A registry, not a script:

```python
@register_feature(name="rsi_14", dependencies=["close"], lookback=14, category="momentum")
def rsi_14(df: pd.DataFrame) -> pd.Series: ...
```

Dependencies resolve as a DAG; features may depend on other features.

**Look-ahead protection — non-negotiable.** For every registered feature, a parametrised test computes the value at time *t* on the full series and on the series truncated at *t*, asserting equality. Any feature failing this fails the build. This runs over the entire registry, not a spot check.

**Categories (target 70+ features).**

| Category | Features |
|---|---|
| Returns | Simple, log, rolling, cumulative |
| Momentum | 5D, 10D, 20D, 50D, 100D, 200D |
| Moving averages | SMA, EMA, WMA |
| Volatility | Rolling std, ATR, Parkinson, Bollinger width |
| Technicals | RSI, MACD, ADX, OBV, VWAP |
| Risk-adjusted | Rolling Sharpe, Sortino, Calmar, beta, alpha |
| Cross-sectional | Z-score, rank, rolling rank |
| Microstructure | Volume shock, gap %, lag features |

Cross-sectional features are computed **within the point-in-time universe on each date**, never across the full ticker set.

RSI, MACD, ATR and ADX are hand-verified against a reference implementation in tests.

**Performance — required deliverable.** Benchmark a full build across 500 securities × 10 years, profile it, optimise the top three bottlenecks, and report before/after timings with profile output in `docs/performance.md`. Expected wins: caching shared intermediates in the DAG (many features need EMA), vectorising rolling operations, parallelising per-security work.

---

### Module 5 — Statistical analysis

Wrapping `statsmodels`; not reimplemented. Each test returns a typed result carrying statistic, p-value, and a plain-English interpretation string.

ADF, KPSS, Jarque-Bera, Ljung-Box, ACF/PACF, Engle-Granger cointegration, Johansen cointegration, Variance Inflation Factor, PCA, correlation and covariance matrices.

---

### Module 6 — Alpha research

**Factor protocol.** Takes a feature panel, returns cross-sectional scores per (date, security), computed within the point-in-time universe for that date.

**Evaluation suite.**

- Pearson IC and Spearman rank IC
- IC time series with t-statistic and Newey-West standard errors
- IC decay across forward horizons 1D / 5D / 10D / 20D
- Quantile bucket returns with long-short spread
- Turnover and factor autocorrelation
- Fitness metric

**Baseline factors.** 12-1 momentum, short-term reversal, low volatility, volume-based.

**Gate.** A factor must clear an IC threshold before becoming eligible for the strategy builder. Enforced in code.

**Split enforcement.** Factor evaluation runs on train split only. The evaluator raises if handed data past `HOLDOUT_START` without the holdout flag.

---

### Module 7 — Strategy builder

Available strategies: moving average cross, RSI, MACD, Bollinger, breakout, momentum, mean reversion, pairs trading, dollar-cost averaging, plus a custom-rule interface.

Rule components: buy condition, sell condition, exit, stop loss, take profit, position size.

---

### Module 8 — Backtesting engine

Event-driven, strictly separated components:

| Component | Responsibility |
|---|---|
| `DataHandler` | Streams bars chronologically; structurally prevents access beyond current bar; enforces holdout lockout |
| `Strategy` | Consumes bars, emits Signals |
| `Portfolio` | Consumes Signals, applies sizing and risk limits, emits Orders; knows nothing about strategies |
| `ExecutionHandler` | Consumes Orders, applies costs and slippage, emits Fills; default T+1 open |
| `Ledger` | Records every fill, position, daily mark |

**Cost models — three required, pluggable.**

1. `ZeroCostModel` — validation tests only
2. `GenericBpsModel` — configurable bps per side plus fixed
3. `IndianEquityCostModel` — itemised:

| Component | Rate |
|---|---|
| STT (delivery) | 0.1% both sides |
| STT (intraday) | 0.025% sell side only |
| NSE transaction charge | ~0.00297% of turnover |
| SEBI turnover fee | ₹10 per crore |
| Stamp duty | 0.015% buy side only |
| GST | 18% on (brokerage + exchange charges) |
| Brokerage | Parameterised — zero-delivery discount broker, or 0.03% / ₹20 per order |

Each component is itemised separately in the ledger so cost attribution is inspectable.

**Execution realism.**

- **Circuit limits** — an order on a day the security hit its price band receives no fill; logged as a rejected order with reason. Momentum signals cluster on exactly these days.
- **Volume participation cap** — orders may not exceed X% (default 1%) of that bar's volume; excess rejected or split across days, configurable.
- **T+1 settlement** — affects capital availability, not merely execution timing.
- **Delisting** — position liquidated at last available price per Module 2 rules.

**Metrics.** Annual return, Sharpe, Sortino, Calmar, max drawdown and duration, win rate, profit factor, average trade, expectancy, turnover, equity curve, benchmark comparison, itemised total costs paid.

**Cost sensitivity — required deliverable.** Every strategy runs under all three cost models; results written to `docs/research/cost_sensitivity.md`. Strategies that die under realistic costs are a finding and must be reported as such.

**Required tests, in priority order.**

1. Buy-and-hold under `ZeroCostModel` reproduces benchmark return to floating-point tolerance
2. A strategy attempting to access future bars raises rather than silently succeeding
3. Golden regression — one strategy's full metric output pinned to a committed fixture
4. Cost model — hand-computed charge on a known Indian trade matches to the paisa
5. Circuit limit — a signal on a band-hit day produces no fill

---

### Module 9 — Risk analytics

Historical / parametric / Cornish-Fisher VaR, Expected Shortfall, rolling beta and alpha vs benchmark, rolling volatility, correlation matrix with hierarchical clustering, factor exposure via regression on factor returns, Monte Carlo with bootstrapped returns.

**Stress testing uses real ingested data, not synthetic shocks.** These windows are explicitly ingested and retained regardless of backtest range:

| Window | Period |
|---|---|
| Global Financial Crisis | 2007-10 → 2009-03 |
| COVID crash | 2020-02 → 2020-04 |
| Rate shock | 2022-01 → 2022-10 |
| Demonetisation | Nov 2016 |
| IL&FS credit event | Sep–Oct 2018 |

Each replays the portfolio and reports drawdown, worst day, and recovery time.

---

### Module 10 — Portfolio optimisation

Equal weight, inverse volatility, risk parity, minimum variance, maximum Sharpe, efficient frontier. Implemented with `cvxpy`. Supports weight bounds, long-only or long-short, turnover penalty.

**Covariance estimation must offer Ledoit-Wolf shrinkage.** Sample covariance is unstable when assets outnumber observations.

**Rebalancing runs against the point-in-time universe** — a security that left the index is exited at the rebalance date, not held silently.

**Test.** Two-asset case with known correlation; min-variance weights verified against the closed-form analytical solution.

---

### Module 11 — Machine learning

Restricted per Section 2.2. No sequence models.

**Targets.** Next-day return (regression), price direction (classification).
**Models.** Ridge, lasso, random forest, XGBoost, LightGBM.

**Validation methodology is the entire point of this module.**

- **PurgedKFold with embargo** — standard k-fold leaks in time series; purging removes training samples whose label windows overlap the test window
- Walk-forward with expanding and rolling windows
- Training on train split, model selection on validation split, holdout untouched
- Every reported metric out-of-sample; in-sample metrics logged separately and explicitly labelled

Cross-sectional features computed within the point-in-time universe — a model must never see a security absent from the index on that date.

**Metrics.** MAE, RMSE, directional accuracy, precision, recall, ROC-AUC, confusion matrix, SHAP attribution.

**Trials accounting.** Every model configuration writes to `research_log` and counts toward the trials total. A model's traded Sharpe is deflated by the full trials count including model search.

**Baseline honesty — required.** Every model reported against buy-and-hold and a naive momentum rule. Where a model fails to beat the baselines, the report states so in plain language. Model search space and iteration count reported alongside results.

---

### Module 12 — API

FastAPI. Sole data source for all frontends; no computation client-side.

**Endpoints.** Securities, universe membership as-of, OHLCV, features, data quality findings, factor evaluation, backtest submission and polling, research log queries, deflated Sharpe, risk metrics, portfolio optimisation, stress tests.

**Sharpe contract.** Any endpoint returning a Sharpe ratio returns an object — `{value, trials, deflated_sharpe, pbo, split, cost_model}` — never a bare float. This makes the reporting rule of Section 4.6 structurally enforceable.

Long-running backtests return a job id and are polled. Pydantic response models throughout. OpenAPI schema generates a TypeScript client into `frontend/src/lib/api/`.

Every endpoint returns display-ready formatted values alongside raw ones so formatting cannot diverge between frontends.

---

### Module 13 — Interface

**Design system first.** `docs/DESIGN_SYSTEM.md` is authored before any UI code and is binding. Reference points: Bloomberg Terminal information density, Linear's restraint, FT/Economist chart conventions. Not a generic dark SaaS dashboard.

Key rules: all numbers in tabular-nums monospace so digits align in columns; colour encodes meaning only, never decoration; 4px spacing base, dense layout; metric cards under 120px; tables showing 20+ rows without scrolling.

**Statistical honesty is a design requirement.** Any displayed Sharpe renders with trials count and deflated Sharpe adjacent — a `SharpeDisplay` component accepting a bare number must not typecheck. Backtest results carry a visible TRAIN / VALIDATION / HOLDOUT split badge. Results without realistic costs carry a warning marker. Exploratory results are visually distinguished from pre-registered ones.

**Frontend.** Next.js 14 App Router, TypeScript, Tailwind, shadcn/ui. Plotly for statistical charts; `lightweight-charts` for price series and equity curves.

**Pages.** Overview, Data Quality, Universe (point-in-time membership viewer with date scrubber and survivorship comparison), Feature Explorer, Factor Research, Backtest, Research Log, Risk, Portfolio.

**Streamlit dashboard** built alongside in `dashboard/`, consuming the same API, deliberately plain. The README documents Streamlit as the researcher-facing tool and Next.js as the product-facing one — mirroring how quant desks actually operate.

---

### Module 14 — Database

| Table | Contents |
|---|---|
| `securities` | security_id, ISIN, name, exchange, listing/delisting dates and reason |
| `ticker_history` | security_id ↔ ticker mappings over time |
| `ohlcv_daily` | Price and volume, unique on (security_id, date) |
| `corporate_actions` | Splits, dividends, bonuses |
| `index_membership` | Point-in-time constituents |
| `unresolved_membership_change` | Reconstitution entries that failed to resolve |
| `data_quality_findings` | Validation output |
| `ingest_log` | Ingestion history for incremental mode |
| `research_log` | Every backtest run, with git commit and holdout flag |
| `strategies`, `signals`, `trades`, `backtests`, `metrics` | Strategy and result storage |

SQLite in development, PostgreSQL in production. Alembic migrations run clean from empty on both.

---

### Module 15 — Reporting

**PDF tearsheet** via WeasyPrint: performance, charts, risk, trades, summary, and a **mandatory methodology page** stating universe construction, cost model, splits, trials count, and limitations.

**Research paper** — `docs/research/paper.md`, 2000–3000 words, the primary deliverable of the project. Structure: hypothesis and economic rationale (written before results), data and universe construction with survivorship measurement, methodology, results across train/validation/holdout with trials count and deflated Sharpe, robustness, limitations, conclusion.

Every number traceable to a `research_log` run_id. A negative result is written as a negative result.

---

## 6. Technology stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| API | FastAPI |
| Database | PostgreSQL (prod), SQLite (dev), SQLAlchemy, Alembic |
| Analysis | pandas, NumPy, SciPy, statsmodels |
| Optimisation | cvxpy |
| Machine learning | scikit-learn, XGBoost, LightGBM, SHAP |
| Visualisation | Plotly, lightweight-charts |
| Frontend | Next.js 14, TypeScript, Tailwind, shadcn/ui |
| Internal dashboard | Streamlit |
| Calendars | exchange_calendars |
| Validation | pandera, Pydantic |
| CLI | typer |
| Logging | structlog |
| Testing | pytest |
| Quality | ruff, mypy (strict on `src/`), pre-commit |
| Deployment | Docker, docker-compose, GitHub Actions |

---

## 7. Engineering standards

- Python 3.11, full type hints, mypy strict on `src/`
- All configuration through Pydantic Settings; no hardcoded paths or credentials anywhere
- Structured JSON logging via structlog with request ids
- pytest with 80% coverage gate on `src/`
- ruff and pre-commit enforced in CI
- GitHub Actions running ruff, mypy, pytest, the full look-ahead suite, and docker build on every PR
- Scheduled nightly `ingest --incremental` followed by `validate --fail-on-regression`; failure opens an issue
- `/health` and `/ready` endpoints including a data freshness check

---

## 8. Repository structure

```
QuantLab/
├── src/quantlab/
│   ├── config.py            # Pydantic settings, HOLDOUT_START
│   ├── data/                # providers, storage, validation
│   ├── universe/            # point-in-time membership
│   ├── db/                  # SQLAlchemy models, Alembic
│   ├── feature_engineering/ # registry, DAG, feature definitions
│   ├── stats/               # statistical tests
│   ├── factors/             # alpha research, IC evaluation
│   ├── strategies/          # strategy implementations
│   ├── backtesting/         # engine, cost models, execution
│   ├── risk/                # VaR, stress tests, Monte Carlo
│   ├── portfolio/           # optimisers
│   ├── ml_models/           # purged CV, models, SHAP
│   ├── research/            # log, deflated Sharpe, pre-registration
│   ├── api/                 # FastAPI
│   ├── reports/             # tearsheet generation
│   └── utils/
├── frontend/                # Next.js
├── dashboard/               # Streamlit
├── notebooks/
├── tests/
├── data/
│   ├── raw/                 # immutable vendor responses
│   ├── features/            # partitioned parquet
│   └── raw/reference/       # cached reconstitution PDFs
├── docs/
│   ├── PRD.md
│   ├── DESIGN_SYSTEM.md
│   ├── hypotheses.md
│   ├── methodology.md
│   ├── data_dictionary.md
│   ├── performance.md
│   └── research/
│       ├── paper.md
│       ├── survivorship_bias.md
│       ├── cost_sensitivity.md
│       └── holdout_results.md
├── scripts/
│   └── final_evaluation.py
├── config/
└── pyproject.toml
```

---

## 9. Milestones

| Milestone | Scope | Tag |
|---|---|---|
| **M1 — Data Platform** | Ingestion, universe construction, validation, feature registry | `v0.1-data` |
| **M2 — Research Engine** | Statistics, alpha research, research integrity layer, backtester, risk, portfolio, ML | `v0.2-research` |
| **M3 — Interface** | Design system, API, frontend, Streamlit, design review | `v0.3-ui` |
| **M4 — Production** | PostgreSQL, Docker, CI/CD, documentation, holdout evaluation, research paper | `v1.0` |

Nothing is built "for later" — each milestone ships something that works end to end.

---

## 10. Acceptance criteria

The project is complete when all of the following hold.

**Data integrity**
- [ ] Point-in-time index membership built from NSE reconstitution filings, with resolution rate reported
- [ ] Survivorship bias measured and documented with a specific number
- [ ] Ticker renames resolve to a single security_id; delisted securities retain history
- [ ] Every registered feature passes the truncated-series look-ahead test
- [ ] Data validation covers all checks in Module 3, each with a test injecting the defect

**Research integrity**
- [ ] Holdout lockout enforced in code; `DataHandler` raises without the environment variable
- [ ] Every backtest writes to `research_log` automatically; total trials count is queryable
- [ ] Deflated Sharpe, expected-max-Sharpe-under-null, and PBO implemented and reading from the log
- [ ] Pre-registered hypotheses committed before their tests were run, verifiable in git history
- [ ] No Sharpe ratio appears anywhere — API, UI, PDF, paper — without trials count and deflated Sharpe

**Backtest correctness**
- [ ] Buy-and-hold under zero costs reproduces the benchmark to floating-point tolerance
- [ ] A strategy accessing future bars raises
- [ ] Golden regression fixture committed
- [ ] Indian cost model matches a hand-computed trade to the paisa
- [ ] Circuit-limit days produce no fill
- [ ] Cost sensitivity across all three models reported; strategies that die under realistic costs are named

**Engineering**
- [ ] mypy strict passes on `src/`
- [ ] Test coverage ≥ 80% on `src/`
- [ ] CI green: ruff, mypy, pytest, look-ahead suite, docker build
- [ ] Alembic migrations run clean from empty on SQLite and PostgreSQL
- [ ] No hardcoded paths or committed credentials
- [ ] Feature build performance benchmarked, optimised, and documented before/after

**Deliverables**
- [ ] README with architecture diagram, quickstart, honest RESULTS section including the worst drawdown period
- [ ] LIMITATIONS section stating plainly what the platform cannot do
- [ ] `docs/methodology.md` covering universe construction, look-ahead prevention, purged CV, cost modelling, multiple-testing correction
- [ ] `docs/data_dictionary.md` covering every feature
- [ ] Holdout evaluated exactly once, script committed before the run, output written verbatim
- [ ] Research paper complete, every number traceable to a run_id

---

## 11. Known limitations

Stated up front, and to be repeated in the README. A platform whose limits are documented is more credible than one claiming completeness.

- **No intraday data** — fills are modelled at T+1 open with assumed slippage, not observed
- **No point-in-time fundamentals** — value and quality factors cannot be constructed without look-ahead
- **No borrow costs** — short-side results are optimistic
- **Universe resolution below 100%** — some historical reconstitution entries will not map to a security_id; the rate is reported, not hidden
- **yfinance is an unofficial source** — used for convenience; NSE bhavcopy is authoritative where discrepancies arise
- **Adjustment logic is self-built** for NSE, since bhavcopy ships unadjusted; errors here propagate to every downstream result
- **Single market, single asset class** — no cross-asset or international generalisation is claimed

---

## 12. Definition of done

The platform is not the deliverable. The deliverable is one research question, investigated honestly enough that its answer — positive or negative — would survive scrutiny from someone who knows the field.

A documented negative result meeting the acceptance criteria above is a successful outcome. An impressive dashboard reporting an unverifiable Sharpe ratio is not.