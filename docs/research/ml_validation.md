# Machine learning: validation methodology

*PRD Module 11. This document describes the method; results tables are appended by `quantlab ml search` runs against ingested data.*

The models in this module are ordinary. Ridge, lasso, random forest, XGBoost and LightGBM against a next-day return and its direction — nothing here is a contribution to the literature. **The validation harness is the deliverable.** A cross-validated equity model that reports a positive out-of-sample R² is worth exactly as much as the discipline that produced the "out-of-sample" claim, and that discipline is what this document specifies.

Deep sequence models (LSTM, GRU, Transformer) are a deliberate scope cut (`CLAUDE.md` §4). `MODEL_FAMILIES` is the only construction path in the codebase, so the cut is enforced rather than documented: `build_model("lstm", ...)` raises.

## Why standard k-fold is not merely optimistic here

It is wrong in two independent ways, and fixing one does not fix the other.

**Shuffling.** Assigning rows to folds at random trains on Thursday and tests on Wednesday. Every splitter in `quantlab.ml_models.validation` cuts contiguous blocks of **dates**, never rows. On a panel that additionally means a single date never appears on both sides of a split — otherwise the model learns Reliance's Tuesday and is scored on Infosys's Tuesday, in a market whose cross-section is correlated by construction.

**Label overlap.** A sample dated *t* carries a label that resolves at *t + h*. If *t + h* falls inside the test block, that training sample and the test block share a price. Contiguous splitting does nothing about this. **Purging** removes any training sample whose label window `[t0, t1]` intersects the test window (López de Prado, *Advances in Financial Machine Learning*, ch. 7). It applies on both sides of the block: samples before it whose labels reach in, and samples inside its label tail.

Every sample in `MLDataset` therefore carries two timestamps, `t0` and `t1`, not one. That is what makes purging expressible at all.

**Embargo.** Purging cannot reach what serial correlation carries. A sample drawn immediately *after* a test block still contains information from inside it even though its label window does not overlap. A fixed number of dates following each test block is excluded from training as well. `embargo_pct` states it as a fraction of the date grid (the conventional form); `embargo_days` states it in trading days.

Note the asymmetry in `WalkForward`: its training data lies entirely before the test block, so there is nothing after it to embargo. The equivalent protection is the boundary purge — which is mandatory and not configurable — plus an optional `gap_days` buffer. Calling that buffer an "embargo" would be borrowing the word for a different mechanism.

### Verification, not assertion

A purged CV that silently fails to purge produces exactly the results a working one produces, only better. Two mechanisms guard against that:

- Every `Fold` reports `n_purged` and `n_embargoed`. A fold that dropped nothing protected nothing, and the counts make that visible instead of assumed.
- Every splitter re-checks its own output before returning it. If any training sample's label window overlaps its test window, `split()` raises `LeakageError` rather than handing back a fold that leaks.

Run `quantlab ml folds` before trusting any result:

```
quantlab ml folds --universe NIFTY500 --validation purged-kfold --embargo 10
```

## The three validation schemes

| Scheme | What it answers | When it is the right one |
| --- | --- | --- |
| `purged-kfold` | Does the signal generalise across the sample? | Short samples, where reserving half the history for training wastes what little there is. Trains on data that *follows* its test block, so it is not a simulation of live trading. |
| `walk-forward-expanding` | Would this have worked, refitting on everything known so far? | The natural simulation when the data generating process is assumed stable. |
| `walk-forward-rolling` | …and if it is not stable? | A model fitted through 2015–2019 and applied in 2021 must generalise across a regime change a rolling window would have forgotten. |

Which is correct is an empirical question about the market, so PRD Module 11 requires all of them to be run rather than picking one.

## Split discipline

| Split | Range | Use |
| --- | --- | --- |
| Train | 2015-01-01 → 2021-12-31 | Fitting, and the purged CV within it |
| Validation | 2022-01-01 → 2023-12-31 | **Model selection only** |
| Holdout | 2024-01-01 onward | Locked. Evaluated once, at the end |

A sample whose features fall on the last training day and whose label resolves on the first validation day belongs to **neither** split. `MLDataset.split_mask` requires both `t0` and `t1` to lie inside a split's bounds. Assigning the straddling sample to train would leak a validation price into training; assigning it to validation would score the model on a feature date it was trained on. It costs one cross-section per boundary.

The holdout lock is checked against the **label** date, not the feature date. A dataset whose features stop on 2023-12-29 with a five-day horizon reads prices well inside the locked period, and `build_dataset` refuses to assemble it. The research log re-checks the window independently before writing a row.

## Point-in-time universe

`universe` is a required argument to `build_dataset`; there is no `None`. A date with no membership record raises rather than defaulting to "keep everything", because a missing record and an empty index are indistinguishable and the permissive reading is the one that produces survivorship bias.

Forward returns are computed on the **unrestricted** price panel and the rows are restricted afterwards. A security's return is a property of its price series; restricting first would compute a "next day" return across the gap left by its absence from the index, which is a different number from the one it actually returned.

## Metrics

| Target | Reported |
| --- | --- |
| Next-day return (regression) | MAE, RMSE, R², directional accuracy, rank IC |
| Direction (classification) | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, base rate, lift over base rate |

Both carry SHAP attribution: `TreeExplainer` for the ensembles, `LinearExplainer` on the scaled features for ridge and lasso. What is informative is the *shape* of the attribution — concentrated in one or two features that duplicate the momentum baseline means the model has rediscovered the baseline at greater cost; spread evenly across seventy means it has found nothing.

**Every metric object states which sample it came from.** `Sample` is a required field, set where the metrics are computed, and it prefixes every key the object emits (`oos_mae` / `is_mae`). An in-sample metrics dict cannot be mistaken for a result downstream, and rendering one prints `IN-SAMPLE (diagnostic only — not a result)`. In-sample figures are kept rather than discarded because the **gap** between them is the overfitting measurement: an ensemble at in-sample R² 0.85 and out-of-sample R² 0.001 has memorised the panel, and only the pair reveals it.

`lift_over_base_rate` deserves its own mention. Indian equity direction is not 50/50, so a classifier reporting 53% accuracy against a 54% base rate is worse than a constant. That subtraction is a property on the metrics object rather than something the reader is left to perform.

## Trials accounting

**Every model configuration writes a `research_log` row.** `ModelEvaluationRun` subclasses `LoggedBacktestEngine`, whose `run()` is `@final` and performs the write itself; the class implements `execute()`, which is called from inside the log's context manager. A crash leaves a `failed` row. Returning without recording metrics raises. There is no unlogged evaluation path, and `evaluate_model` takes the log as its first required argument.

A hyper-parameter sweep is a search over strategies by another name. A deflated Sharpe that counts the twelve backtests but not the two hundred model fits is not a correction — it is the problem rounded down to a comfortable size. So a model's traded Sharpe is deflated by the **whole log's** trials count, spanning every strategy backtest and every model configuration, not by the size of its own grid.

`quantlab ml search` reports the search space and the iteration count **before** any performance number, and states what the search cost the project's trials count:

```
Search space (12 configuration(s) total):
  lightgbm: 12 configuration(s)
      learning_rate: [0.03, 0.1]
      n_estimators: [200, 500]
      num_leaves: [7, 15, 31]

Iterations run:      12 completed, 0 failed
Trials in research_log: 27 before -> 39 after (+12)
```

## Baseline honesty

Required, not courteous. Every model is reported against:

- **Buy-and-hold** — the point-in-time universe held equally weighted. As a predictor it makes the constant prediction carrying no cross-sectional information (the training mean, or the training base rate), so its directional accuracy is the honest floor: the rate at which the market went up anyway.
- **Naive momentum** — rank on a trailing momentum feature, hold the top slice. Nothing is fitted. For the regression target the raw feature is multiplied by the ratio of training standard deviations so MAE and RMSE compare like with like; it is a positive scaling computed on training data only, and it cannot change the rule's ranking, sign, rank IC or directional accuracy.

Both are calibrated on training data and scored on the test fold, through the same metric functions and the same trading rule as the model. Both are required: `compare_to_baselines` raises if either is missing, and `default_baselines` raises rather than quietly returning one when the panel has no momentum feature.

The verdict is strict — the model must beat every baseline on **both** the predictive axis (rank IC or ROC-AUC) and the traded axis (net Sharpe). When it does not, the report says so in plain language and names which baseline won on which axis:

```
VERDICT: ml:xgboost:regression did NOT beat the required baselines.
  - lost to naive_momentum on: rank IC 0.0031 vs 0.0089
  - lost to buy_and_hold on: net Sharpe 0.21 vs 0.64
  On this evidence the model does not earn its complexity over a rule that
  requires no fitting. This is the result; it is reported as the result rather
  than buried under a chart of the one fold where it won.
```

And when it does beat them, the text still refuses to overclaim: beating the baselines is necessary, not sufficient, and the deflated Sharpe decides whether the result survives the search that produced it.

## What the traded comparison is not

`quantlab.ml_models.trading` builds the return series the baseline comparison runs on. It is **not the backtester**: no order book, no circuit-limit rejection, no volume participation cap, no T+1 settlement — all of which `quantlab.backtesting` models and all of which move the answer. Costs are a single turnover-proportional charge derived from `IndianEquityCostModel.round_trip_cost_fraction()`, and the cost model is recorded in the log as `IndianEquityCostModel(turnover-approx)` so it cannot be mistaken in the log for the itemised ledger.

It is the right instrument for ranking configurations against each other and against baselines under identical assumptions. It is the wrong instrument for a headline Sharpe. A model that survives here goes to the event-driven engine next.

Return observations do not overlap: with a horizon of *h* days, rebalancing happens every *h* dates and the annualisation factor is `252 / h`. Daily rebalancing on a five-day label would produce consecutive observations sharing four days of price movement, understating the standard deviation and inflating the Sharpe.

## Commands

```
quantlab ml models                       # permitted families and their declared grids
quantlab ml folds   --validation purged-kfold --embargo 10
quantlab ml train   --model lightgbm --target direction --validation walk-forward-expanding
quantlab ml search  --families xgboost,lightgbm --target return
quantlab research log --summary          # what the search cost the trials count
quantlab research deflate --run-id <id>  # the number that decides anything
```

## Results

No results are recorded yet. This section is populated by `quantlab ml search` against ingested NSE data, and — per the rule above — will state plainly whether the models beat buy-and-hold and naive momentum, including if they do not.
