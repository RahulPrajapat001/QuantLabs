# Cost sensitivity

*Generated 2026-07-25 05:42 UTC.*

Every strategy run under all three cost models. This is a required deliverable (PRD Module 8), and its purpose is to establish which strategies survive realistic Indian transaction costs — **not** to find the best Sharpe.

## How to read this

- `ZeroCostModel` is a **validation instrument, not a result.** Its column exists to show what the strategy would earn in a frictionless world, which is the gross edge. No number in that column is tradeable.
- `GenericBpsModel` applies a flat 10 bps per side — the conventional simplification, shown for comparison.
- `IndianEquityCostModel` is the honest one: itemised STT, stamp duty, exchange and SEBI charges, and GST. A round trip costs roughly **0.22% of notional** before slippage.
- **A strategy that dies under realistic costs is a finding, not a failure.** The gap between the gross and net columns measures how much of the apparent edge was actually a subsidy from ignoring costs.

**Every Sharpe below is undeflated.** These runs contributed to a `research_log` trials count now standing at **27**. Deflate before drawing any conclusion:

```
quantlab research deflate --strategy <name>
```

## Run specification

| Universe | NIFTY500 (synthetic), point-in-time |
| --- | --- |
| Period | 2016-01-04 → 2017-12-01 |
| Survival threshold | net Sharpe ≥ 0.5 under `IndianEquityCostModel` |
| Combinations | 27 (strategy × cost model) |

## Sharpe by cost model

| Strategy | ZeroCostModel | GenericBpsModel | IndianEquityCostModel | Gross → net gap |
| --- | --- | --- | --- | --- |
| `bollinger_reversion` | -0.131 | -0.318 | -0.340 | -0.209 |
| `breakout` | 0.097 | -0.117 | -0.141 | -0.238 |
| `buy_and_hold` | 0.621 | 0.616 | 0.615 | -0.006 |
| `dca` | 0.648 | 0.643 | 0.643 | -0.006 |
| `ma_cross` | -0.101 | -0.197 | -0.209 | -0.108 |
| `macd` | 0.610 | 0.276 | 0.238 | -0.372 |
| `momentum_12_1` | 0.642 | 0.608 | 0.604 | -0.037 |
| `reversal_5d` | -0.742 | -1.036 | -1.068 | -0.326 |
| `rsi_reversion` | 0.549 | 0.443 | 0.431 | -0.118 |

## Annualised return by cost model

| Strategy | ZeroCostModel | GenericBpsModel | IndianEquityCostModel | Gross → net gap |
| --- | --- | --- | --- | --- |
| `bollinger_reversion` | -0.54% | -1.20% | -1.28% | -0.74% |
| `breakout` | 0.31% | -0.55% | -0.65% | -0.97% |
| `buy_and_hold` | 4.88% | 4.83% | 4.83% | -0.05% |
| `dca` | 6.18% | 6.12% | 6.12% | -0.06% |
| `ma_cross` | -0.55% | -0.98% | -1.03% | -0.48% |
| `macd` | 2.49% | 1.07% | 0.91% | -1.58% |
| `momentum_12_1` | 1.48% | 1.40% | 1.39% | -0.09% |
| `reversal_5d` | -1.94% | -2.69% | -2.77% | -0.83% |
| `rsi_reversion` | 1.58% | 1.27% | 1.23% | -0.35% |

## Cost drag under Indian costs

Cost drag is total charges as a fraction of initial capital. Turnover is annualised two-way turnover — the quantity Indian taxes are levied on, and therefore the best single predictor of whether a strategy survives.

| Strategy | Trades | Turnover (x/yr) | Costs paid | Cost drag | Net Sharpe | Survives |
| --- | --- | --- | --- | --- | --- | --- |
| `macd` | 222 | 13.93 | ₹30,185 | 3.02% | 0.238 | **no** |
| `breakout` | 127 | 8.69 | ₹18,704 | 1.87% | -0.141 | **no** |
| `reversal_5d` | 85 | 7.67 | ₹16,726 | 1.67% | -1.068 | **no** |
| `bollinger_reversion` | 86 | 6.76 | ₹15,002 | 1.50% | -0.340 | **no** |
| `ma_cross` | 64 | 4.32 | ₹9,496 | 0.95% | -0.209 | **no** |
| `rsi_reversion` | 37 | 3.15 | ₹7,071 | 0.71% | 0.431 | **no** |
| `momentum_12_1` | 8 | 0.78 | ₹1,744 | 0.17% | 0.604 | yes |
| `dca` | 0 | 0.47 | ₹1,152 | 0.12% | 0.643 | yes |
| `buy_and_hold` | 0 | 0.38 | ₹928 | 0.09% | 0.615 | yes |

## Verdict

**Survive realistic costs** (net Sharpe ≥ 0.5): `buy_and_hold`, `dca`, `momentum_12_1`

**Do not survive realistic costs**: `bollinger_reversion`, `breakout`, `ma_cross`, `macd`, `reversal_5d`, `rsi_reversion`

These are reported rather than discarded. A strategy whose gross edge is real and whose net edge is not has told you something specific: the edge is smaller than the cost of harvesting it at this turnover. The response is lower turnover, not a different cost model.

## What this table cannot tell you

- It is a **cost** sweep, not a validation of any strategy. A strategy that survives here has cleared one hurdle out of several; it still has to clear deflation against the full trials count, and it still has to hold up out of sample.
- Slippage is not modelled in this sweep (zero slippage), so the Indian column is a **floor** on costs rather than a full estimate. Real execution adds market impact on top.
- Every number here comes from the train and/or validation split. The holdout (2024-01-01 onward) is locked and is not touched by this sweep.

---

Every row above is traceable to a `research_log` run_id:

```
quantlab research log --strategy <name>
```
