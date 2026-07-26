# Methodology

How QuantLab defends against the four failure modes that inflate retail quant
results, and **where in the code each defence is enforced**. Every section names
the module that makes the claim true and the test that would fail if it stopped
being true, because a methodology document that only describes intentions is
indistinguishable from one describing intentions that were never implemented.

The four failure modes, from [`CLAUDE.md`](../CLAUDE.md) §1:

| Failure mode | Defence | Enforced in |
|---|---|---|
| Survivorship bias | Point-in-time membership from reconstitution filings | [`universe/`](../src/quantlab/universe/) |
| Look-ahead bias | Event-driven backtester, truncated-series feature tests | [`backtesting/`](../src/quantlab/backtesting/), [`feature_engineering/`](../src/quantlab/feature_engineering/) |
| Multiple testing | Automatic `research_log`, deflated Sharpe, PBO | [`research/`](../src/quantlab/research/) |
| Unrealistic costs | Itemised STT / stamp duty / GST / circuit limits | [`backtesting/costs.py`](../src/quantlab/backtesting/costs.py) |

---

## 1. Universe construction

**The claim.** Index membership is reconstructed as it stood on each historical
date, from NSE reconstitution filings — never from today's constituent list.

**Why it matters.** A backtest run on today's NIFTY 500 has been handed the answer
to a question it is supposed to be asking. Every company that was removed for
underperformance, delisted, or merged away is absent, so the universe is a
survivor cohort selected partly on the outcome being measured.

**How it is enforced.**

- [`universe/reconstitution.py`](../src/quantlab/universe/reconstitution.py)
  parses additions and removals from filings and resolves each to a stable
  `security_id`. `resolve_event()` matches through `ticker_history`, so a company
  that renamed does not become two companies.
- [`universe/membership.py`](../src/quantlab/universe/membership.py) —
  `universe_as_of(session, index, date)` is the only way membership is read.
  Nothing in the codebase queries "current constituents".
- **Unresolved changes are recorded, not dropped.** An entry whose ticker cannot
  be resolved goes to `unresolved_membership_change` with a reason, and the
  resolution rate is a *reported* metric (`ResolutionReport`). Silently dropping
  an unresolvable removal would quietly reintroduce the bias this module exists
  to remove.
- The `data_quality` checks surface unresolved entries as `error`-severity
  findings, because a universe with unresolved changes has incomplete
  point-in-time membership.

**The bias, quantified.** `GET /universe/{index}/survivorship` returns the
point-in-time constituent count, today's count, and the overlap for any date. The
Universe page charts it. See [`docs/research/survivorship_bias.md`](research/survivorship_bias.md).

**Known limit.** Resolution is below 100%. That number is reported rather than
rounded up — see [LIMITATIONS](../README.md#limitations).

---

## 2. Look-ahead prevention

Four independent mechanisms, because look-ahead enters through more than one door.

### 2.1 Every feature is tested against its own truncated series

[`tests/feature_engineering/test_lookahead.py`](../tests/feature_engineering/test_lookahead.py)
computes each registered feature twice: once on the full history, once on the
history truncated at *t*. The value at *t* must be identical. A feature that peeks
forward — a centred rolling window, a `shift(-1)`, a `bfill` — produces different
values and fails.

This runs over the **entire registry** (75 features), not a spot check, and it is
a named step in CI. It is parametrised per feature so a failure names the culprit.

### 2.2 The backtester is event-driven

[`backtesting/engine.py`](../src/quantlab/backtesting/engine.py) walks bars in
order and hands a strategy only what was knowable at that moment.
[`backtesting/data_handler.py`](../src/quantlab/backtesting/data_handler.py) —
`history_as_of()` refuses to return a bar at or after the current timestamp
(`test_history_as_of_refuses_the_future`). Default execution is **T+1 open**: a
signal computed from today's close fills at tomorrow's open, because filling at
the close that generated the signal is not a trade anyone could have made.

A vectorised backtest was rejected for this reason (`CLAUDE.md` §2): vectorised
code hides look-ahead in an off-by-one that produces plausible results.

### 2.3 `as_of` timestamps on every price row

`ohlcv_daily` stores `close` and `adj_close` separately and stamps every row with
`as_of`. Adjustment factors change retroactively when a split occurs, so
overwriting history in place makes yesterday's backtest irreproducible. The split
is visible rather than merged.

### 2.4 The holdout lock

`2024-01-01` onward is inaccessible by default.
[`config.py`](../src/quantlab/config.py) — `assert_holdout_allowed()` raises
`HoldoutViolationError` unless the caller passed `holdout=True` **and**
`QUANTLAB_UNLOCK_HOLDOUT=1` is set in the environment. The variable is
deliberately absent from `.env.example`, so unlocking is a conscious act that
cannot happen by copying a template.

The lock is checked at three independent points: the `DataHandler`, the backtest
submission endpoint, and the research log. `research_log.holdout_touched` records
any run that crossed the boundary, so an accidental access is auditable after the
fact rather than deniable.

---

## 3. Purged cross-validation

**The claim.** ML models are validated with purged, embargoed, contiguous-block
cross-validation — never a random `KFold`.

**Why it matters.** Financial labels overlap. A 20-day forward return computed on
day *t* uses prices through *t+20*, so a training sample at *t* and a test sample
at *t+5* share most of their outcome. Random k-fold puts them in different folds
and calls the result out-of-sample; it is not, and the leakage inflates every
metric derived from it.

**How it is enforced.**
[`ml_models/validation.py`](../src/quantlab/ml_models/validation.py):

- Folds are **contiguous date blocks**, not shuffled rows.
- **Purging** drops training samples whose label window overlaps the test block.
- **Embargo** drops training samples immediately after the test block, so
  serial correlation does not carry information backwards across the boundary.
- Every fold reports `n_purged` and `n_embargoed`. A purged CV that removed zero
  samples is not purged, and the counts make that visible instead of implied.

**Reporting discipline.** `ml_models/evaluate.py` labels the basis of every
headline number explicitly: `selection_basis` returns the literal string
`cv_out_of_sample` when the number came from pooled CV folds rather than the
validation split. CV out-of-sample is **not** the holdout, and the code does not
let the two be confused. In-sample figures are reported alongside, because the
gap between them is the overfitting measurement.

---

## 4. Cost modelling

**The claim.** Indian statutory transaction costs are itemised, not approximated
by a single basis-point figure.

**Why it matters.** STT alone is 0.1% per side on delivery equity. A strategy
turning over 400% annually pays roughly 0.8% a year in STT before brokerage,
stamp duty, GST, exchange charges or slippage. A "10 bps round trip" assumption —
the default in most retail backtests — understates the true cost by a factor that
is frequently larger than the alpha being claimed.

**How it is enforced.**
[`backtesting/costs.py`](../src/quantlab/backtesting/costs.py) —
`IndianEquityCostModel` charges each component separately and reports them
separately (`CostBreakdown`: `brokerage`, `stt`, `exchange_transaction`,
`sebi_turnover_fee`, `stamp_duty`, `gst`). The Backtest page charts the
attribution, so a strategy dying of STT is visibly dying of STT.

- **Hand-verified.** A test matches a single trade against a hand-computed figure
  to the paisa. An arithmetic error in a cost model is invisible in aggregate.
- **Circuit limits.** `CircuitLimitPolicy` produces **no fill** on a day the
  price moved the full band. A backtest that fills on a locked-limit day is
  trading against a counterparty who did not exist.
- **Participation cap.** Orders are capped at a share of the day's volume, so a
  backtest cannot buy more than the market traded.
- **Three cost models, and the difference is reported.**
  `backtesting/cost_sensitivity.py` runs a strategy under `ZeroCostModel`,
  `GenericBpsModel` and `IndianEquityCostModel`. Strategies that survive gross
  and die net are named — see
  [`docs/research/cost_sensitivity.md`](research/cost_sensitivity.md).

**Reporting discipline.** Any result whose cost model is not exactly
`IndianEquityCostModel` carries a warning marker in the UI, the dashboard and the
PDF (`api/models.py` → `CostModelStatus.realistic`). A `ZeroCostModel` Sharpe
additionally renders in muted type: a gross Sharpe is a validation instrument, not
a result.

---

## 5. Multiple-testing correction

**The claim.** No Sharpe ratio is reported without the number of trials that
preceded it and its deflated value.

**Why it matters.** Search enough strategies and the best one looks good by
construction. With 300 trials, the highest Sharpe you find on noise alone is
substantially above zero — so an undeflated Sharpe from a search is not evidence,
it is the maximum of a sample. The trials count is the denominator that makes the
number interpretable.

**How it is enforced.**

- **The log is automatic.** [`research/log.py`](../src/quantlab/research/log.py)
  writes one `research_log` row per run **from inside the engine**
  ([`backtesting/logged.py`](../src/quantlab/backtesting/logged.py)), not via a
  decorator a caller could omit. The row is inserted and committed *before* the
  run executes, so a crashed run leaves evidence instead of vanishing. A run that
  fails to log raises.
- **Trials are read, never entered.**
  [`research/deflated.py`](../src/quantlab/research/deflated.py) counts rows in
  `research_log`. There is no parameter for a caller to supply a trials count,
  because a researcher who has run 300 variants will not remember 300.
- **Deflated Sharpe.** `deflated_sharpe_ratio()` implements Bailey & López de
  Prado: the probabilistic Sharpe ratio against a benchmark raised by the
  expected maximum Sharpe under the null, given the trial count and the observed
  skew and kurtosis of the returns. `expected_max_sharpe_analytic()` is that
  benchmark.
- **PBO.** Probability of backtest overfitting, via combinatorially symmetric
  cross-validation, reported where enough trials exist to compute it.
- **The contract is structural.** `SharpeObject` carries `value`, `trials`,
  `deflated_sharpe`, `pbo` and `significant` as required fields. A schema test
  (`tests/api/test_contract.py`) asserts no response model anywhere exposes a
  Sharpe as a bare number, `assert_no_bare_sharpe()` guards the PDF path, and the
  UI's `SharpeDisplay` will not typecheck if handed a float. A ratio to which
  deflation genuinely does not apply — a benchmark's own Sharpe, an optimiser's
  ex-ante forecast — is a `ReferenceRatio` and must carry a written reason
  (`why_not_deflated`).

**Pre-registration.** Hypotheses are written to
[`docs/hypotheses.md`](hypotheses.md) **before** testing, with a content hash
stored on the run. Anything found afterwards is labelled `exploratory` in the log
and in every view. Most research is exploratory and that is legitimate; the rule
exists so exploratory work is never presented with the authority of a prediction.

---

## 6. What none of this fixes

These mechanisms address bias in *method*. They do not repair the data, and they
do not make a negative result positive. Read
[LIMITATIONS](../README.md#limitations) alongside this document — in particular
that fundamentals are not point-in-time, shorts carry no borrow cost, fills are
modelled at daily granularity, and the NSE adjustment logic is self-built rather
than vendor-certified.
