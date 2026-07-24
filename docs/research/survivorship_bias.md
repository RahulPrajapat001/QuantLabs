# Survivorship bias measurement

> **Status: awaiting real data.** This file is the *methodology and contract* for
> QuantLab's survivorship-bias measurement. It does **not** yet contain a research
> result — no numbers here are real, because point-in-time membership and prices
> for the NSE universe have not been ingested at the time of writing. Running the
> measurement on real data (see below) overwrites this file with the actual
> figures via `quantlab.universe.survivorship.write_survivorship_report`. Do not
> cite this document as a finding until that has happened.

## Why this is a required deliverable

Survivorship bias is the first of the four failure modes QuantLab is built to
defeat (PRD §1.2, §4.7). A backtest run on **today's** index constituents cannot
see any company that was dropped from the index — and dropped names are
disproportionately the failures (delistings, bankruptcies, chronic
under-performers that fell out of the large-cap band). Excluding them silently
inflates return and Sharpe and hides drawdowns. The size of that inflation is not
assumed; it is **measured**, and the measurement is reported here with a specific
number.

## How it is measured

The same strategy is run twice over the same period, differing only in how the
investable universe is resolved on each date:

| Run | Universe | Resolver |
|---|---|---|
| **Point-in-time** | The honest, date-aware constituents reconstructed from index reconstitution history | `PointInTimeUniverse` |
| **Current constituents** | Today's index members applied to *every* historical date (the biased baseline) | `CurrentUniverse` |

Both resolvers implement the same `UniverseResolver` protocol, so the strategy is
identical across runs — only the survivorship treatment differs. The harness then
reports the **delta** in three headline metrics:

- **Annualised return** (geometric)
- **Sharpe ratio** (plain, un-deflated — used only to size the gap; see the note
  below)
- **Maximum drawdown**

A positive return/Sharpe delta, or a shallower (less negative) drawdown on the
current-constituents run, is survivorship bias inflating the biased backtest by
that amount.

## Results

_To be filled in by the measurement run. The table below is the template the
renderer emits; the values are placeholders (`—`) until real data is ingested._

| Metric | Point-in-time | Current constituents | Delta (current minus PIT) |
|---|---:|---:|---:|
| Annualised return | — | — | — |
| Sharpe ratio | — | — | — |
| Max drawdown | — | — | — |

## Reproducing this measurement

1. Ingest OHLCV for the full universe **including delisted names** (`quantlab
   ingest …`). Delisted securities keep their price history and are liquidated at
   their last available price per `quantlab.universe.delisting` — a name that left
   the index is exited, never held or forward-filled.
2. Build point-in-time membership from reconstitution filings:
   `quantlab universe build --index NIFTY500`. The resolution rate (share of
   reconstitution entries mapped to a `security_id`) is reported; unresolved
   entries are recorded in `unresolved_membership_change`, not dropped.
3. Run `measure_survivorship_bias(strategy, …)` with a `PointInTimeUniverse` and a
   `CurrentUniverse` over the same index, then
   `write_survivorship_report(comparison)` to regenerate this file.

## A note on the fixed-universe fallback

`--universe-mode fixed --as-of <date>` freezes constituents at a single date and
holds them constant. This removes **selection bias** (the frozen set is a real
historical universe, not today's winners) but it does **not** remove **delisting
attrition**: a name that delists during the backtest must still be liquidated at
its last price, not held. Fixed mode is a fallback for periods where full
reconstitution history is unavailable, and this caveat must accompany any result
that uses it.

## A note on Sharpe

The Sharpe figures in the results table are plain, un-deflated ratios used **only**
to quantify the survivorship gap. Any Sharpe reported as a headline research
result elsewhere in QuantLab must be accompanied by its trial count and deflated
Sharpe ratio, read from the `research_log` (CLAUDE.md §3). The survivorship delta
is a diagnostic, not a performance claim.
