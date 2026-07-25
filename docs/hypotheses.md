# Pre-registered hypotheses

Every hypothesis QuantLab tests is written here **before** it is tested, with its
economic rationale, the factor or strategy under test, the prediction, the metric,
and the threshold that decides the outcome. This is the discipline that separates
a research finding from a data-mining artefact.

## How this file is used

Each hypothesis is hashed over its substantive content (`id`, title, `subject`,
`rationale`, `prediction`, `metric`, `threshold`). When a backtest links to a
hypothesis id, the hash of that hypothesis is written into the run's
`research_log` row. Three consequences:

- A backtest that names no hypothesis is logged as **exploratory**. That is
  legitimate — most research is exploratory — but it is labelled, and an
  exploratory result is never presented as a confirmed prediction.
- If a hypothesis is edited after a run tested it, the recorded hash no longer
  matches and `quantlab research prereg --validate` reports it as an error. A
  threshold cannot be quietly relaxed to fit a disappointing result.
- Registration is verifiable in git history: the commit that introduced each
  hypothesis is discoverable from the file's history, so "we predicted this in
  advance" is checkable rather than asserted.

Rules for editing this file:

- **Add** a hypothesis before running its test. Never after.
- **Never delete** a tested hypothesis. Set `status: withdrawn` and say why in
  `notes`. A file containing only the hypotheses that worked is worse than no
  file at all.
- The `metric` must name the split it applies to. Without that, a failed test on
  validation can be re-run on train until it passes.
- The `threshold` must be an operator and a number. "Sharpe should be good" is
  not a threshold; `>= 0.50` is.
- Thresholds refer to **net** performance under `IndianEquityCostModel` unless
  stated otherwise, and to the **validation** split. The holdout
  (2024-01-01 onward) is evaluated exactly once, at the end of the project, and
  no hypothesis may be added or amended after that evaluation.

Format:

```markdown
## H<n> — Short title
- **subject:** the factor or strategy under test
- **rationale:** why this should work economically
- **prediction:** what will be observed if it does
- **metric:** the number that decides it, and on which split
- **threshold:** >= 0.50
- **registered:** YYYY-MM-DD
- **status:** registered | tested | withdrawn
```

---

## H1 — 12-1 cross-sectional momentum in the NIFTY 500

- **subject:** strategy `momentum_12_1` — long the top decile, short the bottom
  decile, ranked on 12-month cumulative return skipping the most recent month,
  monthly rebalance, on the point-in-time NIFTY 500 universe.
- **rationale:** Cross-sectional momentum is the most widely replicated equity
  anomaly outside the United States, and the mechanisms proposed for it —
  under-reaction to firm-specific news, slow diffusion of information across
  investors, and institutional flows that chase recent winners — are if anything
  stronger in Indian equities, where analyst coverage below the NIFTY 100 is thin
  and retail participation is high. The one-month skip exists because the most
  recent month carries short-horizon reversal (see H2), which works against the
  twelve-month signal and contaminates it.
- **prediction:** The long-short spread earns a positive risk-adjusted return
  after realistic Indian transaction costs. Monthly rebalancing at decile
  boundaries keeps annual turnover low enough that STT and stamp duty do not
  consume the spread — which is the specific reason this is tested at monthly and
  not weekly frequency.
- **metric:** annualised Sharpe of the long-short portfolio, net of
  `IndianEquityCostModel`, on the validation split (2022-01-01 → 2023-12-31)
- **threshold:** >= 0.50 net of costs
- **registered:** 2026-07-25
- **status:** registered

## H2 — Short-horizon reversal is a cost illusion after Indian transaction taxes

- **subject:** strategy `reversal_5d` — long the bottom decile, short the top
  decile of trailing 5-day return (`return_5d`), weekly rebalance, on the
  point-in-time NIFTY 500 universe.
- **rationale:** Short-horizon reversal is well documented and usually attributed
  to liquidity provision: a stock that fell sharply over a week has, on average,
  paid a premium to whoever absorbed the selling. The premium is real but small
  per trade, and the strategy requires high turnover to harvest it. Indian
  transaction taxes are levied on turnover rather than profit — STT at 0.1% per
  side on delivery, stamp duty at 0.015% on the buy side — so the cost of
  harvesting scales with the same quantity that generates the return. The
  hypothesis is that this anomaly is gross-profitable and net-unprofitable, and
  that the crossover is measurable.
- **prediction:** The gross Sharpe is clearly positive while the net Sharpe under
  `IndianEquityCostModel` is not distinguishable from zero. This is registered as
  an expected **negative** result, and will be reported as one: the interesting
  quantity is the cost level at which the edge disappears, not a headline Sharpe.
- **metric:** annualised Sharpe net of `IndianEquityCostModel` on the validation
  split (2022-01-01 → 2023-12-31), reported beside the gross Sharpe from
  `ZeroCostModel` on the same split
- **threshold:** < 0.20 net of costs
- **registered:** 2026-07-25
- **status:** registered

## H3 — Low-volatility outperformance survives costs because turnover is low

- **subject:** factor `parkinson_20` (inverted) — long the lowest-volatility
  quintile of the point-in-time NIFTY 500, quarterly rebalance, long-only.
- **rationale:** The low-volatility anomaly is attributed to leverage
  constraints: investors who want high returns and cannot borrow bid up
  high-beta stocks instead, leaving low-volatility stocks structurally cheap.
  India's retail-heavy, leverage-constrained investor base and the popularity of
  high-beta small caps make this mechanism plausible here. Unlike H2, the signal
  is slow-moving: realised volatility quintiles are persistent quarter to
  quarter, so turnover — and therefore cost drag — is low. The Parkinson
  estimator is used rather than close-to-close standard deviation because it uses
  the high-low range and is materially less noisy at a 20-day window.
- **prediction:** The lowest-volatility quintile earns a higher Sharpe than the
  cap-weighted NIFTY 500 over the same period, with a smaller maximum drawdown,
  and the advantage is not eliminated by transaction costs because quarterly
  rebalancing keeps annual turnover modest.
- **metric:** annualised Sharpe of the low-volatility quintile minus the
  annualised Sharpe of the point-in-time NIFTY 500 benchmark, net of
  `IndianEquityCostModel`, on the validation split (2022-01-01 → 2023-12-31)
- **threshold:** >= 0.25 Sharpe advantage over the benchmark
- **registered:** 2026-07-25
- **status:** registered

## H4 — Survivorship bias inflates a momentum backtest by a measurable margin

- **subject:** strategy `momentum_12_1` (as defined in H1), run twice: once on
  the point-in-time NIFTY 500 universe and once on the index's current
  constituents held fixed across all of history.
- **rationale:** A universe built from today's index members excludes every
  company that fell out of the index or delisted, and inclusion in today's index
  is itself a consequence of good past performance. For a momentum strategy the
  distortion should be directionally predictable: momentum's losses come from
  crashes in high-momentum names, and the current-constituent universe has
  quietly removed the names that crashed hardest. This is not an incidental data
  check but a measurable quantity, and reporting it is a project deliverable
  (`docs/research/survivorship_bias.md`).
- **prediction:** The current-constituent backtest reports a higher annualised
  return, a higher Sharpe and a shallower maximum drawdown than the
  point-in-time backtest over the same period. The Sharpe gap is the headline
  number.
- **metric:** annualised Sharpe on current constituents minus annualised Sharpe
  on the point-in-time universe, both net of `IndianEquityCostModel`, on the
  train split (2015-01-01 → 2021-12-31)
- **threshold:** >= 0.20 Sharpe inflation
- **registered:** 2026-07-25
- **status:** registered
- **notes:** Measured on train deliberately: this is a property of the data
  construction, not a strategy result, so it does not consume validation.

## H5 — A gradient-boosted combination of registered factors beats its best single input

- **subject:** model `gbm_factor_blend` — gradient-boosted trees over the
  momentum, reversal, volatility and liquidity feature families, trained with
  purged, embargoed walk-forward cross-validation, traded as a decile long-short
  portfolio on the point-in-time NIFTY 500.
- **rationale:** The factors in H1–H3 capture different mechanisms —
  under-reaction, liquidity provision, and leverage constraints — and their
  return streams are unlikely to be perfectly correlated. A non-linear model can
  additionally condition one signal on another, for example muting momentum in
  the highest-volatility names where it is known to crash. The economic claim is
  narrow: the benefit comes from combination and conditioning, not from the model
  discovering an unnamed effect.
- **prediction:** The blended model's net Sharpe exceeds the best single-factor
  strategy from H1–H3 on the same split. Because this hypothesis involves a model
  search, its trials count includes every configuration tried, and its reported
  Sharpe is deflated against that full count — a raw improvement over the best
  single factor is not sufficient.
- **metric:** annualised Sharpe net of `IndianEquityCostModel` on the validation
  split (2022-01-01 → 2023-12-31), **and** its deflated Sharpe ratio computed
  against the full `research_log` trials count
- **threshold:** >= 0.95 deflated Sharpe (probability the true Sharpe exceeds
  zero), with a net Sharpe above the best of H1–H3
- **registered:** 2026-07-25
- **status:** registered
- **notes:** The threshold is set on the deflated Sharpe rather than the raw
  Sharpe because a model search over dozens of configurations is precisely the
  situation the deflation exists to correct.
