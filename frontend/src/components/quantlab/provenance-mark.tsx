/**
 * Exploratory vs pre-registered, as a table cell (DESIGN_SYSTEM.md §7.4).
 *
 * This exists because Overview and Research Log had drifted: the log gave
 * exploratory runs the dashed left rule the design system requires, and Overview
 * rendered the same fact as plain grey text. Same concept, two treatments, and
 * nothing structural stopping a third. One component now, so a page cannot
 * invent its own.
 *
 * `Provenance` in `sharpe-display.tsx` is the prose form, parsed out of a
 * `SharpeReport.provenance` string. This is the tabular form, driven by the
 * `exploratory` boolean and `hypothesis_id` a `RunOut` carries directly.
 */

import { cn } from "@/lib/utils";

export function ProvenanceMark({
  exploratory,
  hypothesisId,
  className,
}: {
  exploratory: boolean;
  hypothesisId?: string | null;
  className?: string;
}) {
  if (exploratory) {
    return (
      <span
        className={cn(
          // Dashed rule, and the literal word — never an icon, never hidden.
          "border-l border-dashed border-border-strong pl-2 text-xs uppercase tracking-wide text-text-muted",
          className,
        )}
      >
        Exploratory
      </span>
    );
  }

  return (
    <span
      className={cn(
        "border-l border-solid border-border-strong pl-2 text-xs text-text-secondary",
        className,
      )}
    >
      <a
        className="ql-num text-text-primary underline underline-offset-2"
        href="https://github.com/RahulPrajapat001/QuantLabs/blob/main/docs/hypotheses.md"
        target="_blank"
        rel="noreferrer"
      >
        {hypothesisId ?? "PRE-REGISTERED"}
      </a>
    </span>
  );
}
