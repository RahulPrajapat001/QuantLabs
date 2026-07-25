/**
 * Type-level tests for the honesty contracts.
 *
 * These assert things that must fail to compile. `tsc --noEmit` is the test
 * runner: each `@ts-expect-error` passes only if the line below it really is a
 * type error, so if the API ever widened `SharpeObject` to accept a bare number,
 * the expect-error would itself become an error and the build would fail.
 *
 * This is the mechanism behind DESIGN_SYSTEM.md §7.1: *"a SharpeDisplay
 * component accepting a bare number must not typecheck."*
 *
 * Assertions use `const x: T = …` rather than a multi-line call, because
 * `@ts-expect-error` only suppresses the line immediately after it and a
 * multi-line call reports its error on an inner line.
 *
 * Run: npx tsc --noEmit -p frontend/tsconfig.json
 */

import type { CostModelStatus, Num, SharpeObject, SplitBadge } from "./client";

// --- A Sharpe cannot be a number --------------------------------------------

/** The component contract the design system mandates. */
declare function SharpeDisplay(props: { report: SharpeObject }): unknown;

declare const properReport: SharpeObject;
SharpeDisplay({ report: properReport }); // ok

// @ts-expect-error a bare number is not a SharpeObject
const bareNumber: SharpeObject = 1.8;

// @ts-expect-error a Sharpe without its trials count is not reportable
const noTrials: SharpeObject = { value: 1.8, deflated_sharpe: 0.31 };

// @ts-expect-error a Sharpe without its deflated value is not reportable
const noDeflation: SharpeObject = { value: 1.8, trials: 312 };

// @ts-expect-error the mandated shape is still incomplete without the rest
const partialShape: SharpeObject = {
  value: 1.8,
  trials: 312,
  deflated_sharpe: 0.31,
  split: "train",
  cost_model: "IndianEquityCostModel",
};

// A trials count is required, so it can never be inferred as optional.
type TrialsIsRequired = undefined extends SharpeObject["trials"] ? never : true;
const trialsRequired: TrialsIsRequired = true;

// --- Numbers arrive pre-formatted -------------------------------------------

declare function MetricCell(props: { value: Num }): unknown;

declare const properNum: Num;
MetricCell({ value: properNum }); // ok

// @ts-expect-error a raw number has no formatted string to render
const rawNumber: Num = 0.124;

// @ts-expect-error a formatted string alone loses the value charts need
const stringOnly: Num = "+12.4%";

// @ts-expect-error `formatted` is required — absence has no display form
const valueOnly: Num = { value: 0.124 };

// --- A split badge is never optional ----------------------------------------

// @ts-expect-error a bare label loses the decomposed compound-split segments
const bareLabel: SplitBadge = "train";

// @ts-expect-error compound splits must arrive already decomposed
const noParts: SplitBadge = { label: "train+validation", holdout_touched: false };

// --- The cost warning is decided server-side --------------------------------

// @ts-expect-error `realistic` is required; a UI must not infer it from the name
const noVerdict: CostModelStatus = { name: "ZeroCostModel" };

export type {
  bareNumber,
  noTrials,
  noDeflation,
  partialShape,
  rawNumber,
  stringOnly,
  valueOnly,
  bareLabel,
  noParts,
  noVerdict,
  trialsRequired,
};
