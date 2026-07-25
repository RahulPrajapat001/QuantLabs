/**
 * A Sharpe ratio cannot be rendered as a number (DESIGN_SYSTEM.md §7.1).
 *
 * The props type is the enforcement. There is no `value?: number` overload, no
 * `trials?: number`, and no widening cast anywhere in this file — passing a bare
 * number is a compile error:
 *
 *   <SharpeDisplay value={1.8} />              // ✗ does not typecheck
 *   <SharpeDisplay report={result.sharpe} />   // ✓ the only form
 *
 * The trials count and the deflated value render adjacent to the Sharpe at equal
 * visual weight, never behind a tooltip. A reader who sees `1.82` must see
 * `312 trials · deflated 0.31` without an interaction, because with 312 trials
 * behind it the deflated value *is* the finding.
 */

import type { SharpeObject } from "@/lib/api/client";
import { cn } from "@/lib/utils";

import { CostWarning } from "./cost-warning";
import { SplitBadge } from "./split-badge";

export interface SharpeDisplayProps {
  /** The full report. A number is not accepted; see the module docstring. */
  report: SharpeObject;
  /** `xl` for the single headline metric on a page; `md` everywhere else. */
  size?: "md" | "xl";
  /** Hide the split badge when the surrounding card already carries one. */
  showBadge?: boolean;
  className?: string;
}

export function SharpeDisplay({
  report,
  size = "md",
  showBadge = true,
  className,
}: SharpeDisplayProps) {
  // A gross Sharpe is a validation instrument, not a result, and should not look
  // like one (§7.3).
  const gross = report.cost_model === "ZeroCostModel";

  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <div className="flex items-baseline gap-2">
        <span
          className={cn(
            "ql-num font-medium",
            size === "xl" ? "text-xl" : "text-md",
            gross ? "text-text-muted" : "text-text-primary",
          )}
        >
          {report.value_display.formatted}
        </span>
        {showBadge && <SplitBadge badge={report.badge} />}
      </div>

      {/* Equal visual weight, adjacent, no interaction required. */}
      <div className="flex flex-wrap items-baseline gap-1 text-sm text-text-secondary">
        <span className="ql-num">{report.trials_display.formatted}</span>
        <span>trials</span>
        <span className="text-text-muted">·</span>
        <span>deflated</span>
        <span className="ql-num text-text-primary">
          {report.deflated_display.formatted}
        </span>
        {report.pbo !== null && (
          <>
            <span className="text-text-muted">·</span>
            <span>PBO</span>
            <span className="ql-num">{report.pbo_display.formatted}</span>
          </>
        )}
      </div>

      {/*
       * When the deflated Sharpe fails the threshold, the reason is the result.
       * It is stated, not implied by the absence of a tick.
       */}
      {!report.significant && (
        <p className="text-xs text-warning">
          Not significant after {report.trials_display.formatted} trials — the
          deflated Sharpe is below the 0.95 threshold. {report.summary}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Provenance provenance={report.provenance} />
        <CostWarning costs={report.costs} />
      </div>
    </div>
  );
}

/**
 * Exploratory vs pre-registered (§7.4). Most research is exploratory and that is
 * legitimate; the rule exists so exploratory work is never presented with the
 * authority of a prediction. Never abbreviated to an icon.
 */
export function Provenance({ provenance }: { provenance: string }) {
  const preregistered = !provenance.toLowerCase().startsWith("exploratory");
  const hypothesis = /\(([^)]+)\)/.exec(provenance)?.[1];

  if (!preregistered) {
    return (
      <span className="border-l border-dashed border-border-strong pl-2 text-xs uppercase tracking-wide text-text-muted">
        Exploratory
      </span>
    );
  }
  return (
    <span className="border-l border-solid border-border-strong pl-2 text-xs text-text-secondary">
      Pre-registered{" "}
      {hypothesis && (
        <a
          className="ql-num text-text-primary underline underline-offset-2"
          href="https://github.com/RahulPrajapat001/QuantLabs/blob/main/docs/hypotheses.md"
          target="_blank"
          rel="noreferrer"
        >
          {hypothesis}
        </a>
      )}
    </span>
  );
}
