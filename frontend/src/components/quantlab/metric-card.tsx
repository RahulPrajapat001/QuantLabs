/**
 * Value, label, benchmark comparison. Nothing else.
 *
 * No icon beside the metric, no sparkline, no animated counter, and no
 * background colour. The card is under 120px including its context line, which
 * is what keeps a row of them scannable rather than a set of hero panels.
 *
 * `value` is a `Num` — `{ value, formatted }` from the API. The frontend never
 * formats a number: that would be a second implementation to drift from the PDF
 * tearsheet and the Streamlit dashboard.
 */

import type { Num } from "@/lib/api/client";
import { cn } from "@/lib/utils";

export interface MetricCardProps {
  label: string;
  value: Num;
  /** The one line of context: a benchmark, a threshold, a count. */
  benchmark?: string;
  /**
   * Colour the value by sign. Only for quantities where the sign is the point
   * (returns, drawdowns, alpha) — never for a count or a neutral ratio.
   */
  signed?: boolean;
  /** Render muted: a gross number that should not look like a result. */
  muted?: boolean;
  className?: string;
}

export function MetricCard({
  label,
  value,
  benchmark,
  signed = false,
  muted = false,
  className,
}: MetricCardProps) {
  const raw = value.value;
  const tone =
    muted || raw === null
      ? "text-text-muted"
      : signed && raw < 0
        ? "text-negative"
        : signed && raw > 0
          ? "text-positive"
          : "text-text-primary";

  return (
    <div
      className={cn(
        "flex max-h-metric-card flex-col justify-between gap-1 rounded border border-border bg-surface p-3",
        className,
      )}
    >
      <span className="text-xs uppercase tracking-wide text-text-secondary">
        {label}
      </span>
      <span className={cn("ql-num text-md font-medium", tone)}>
        {value.formatted}
      </span>
      {/*
       * Reserved whether or not there is a benchmark, so a row of cards with
       * mixed context lines does not have ragged heights.
       */}
      <span className="min-h-4 text-xs text-text-muted">{benchmark ?? ""}</span>
    </div>
  );
}

/** The 8px gap between cards, as a grid. */
export function MetricRow({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "grid grid-cols-2 gap-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6",
        className,
      )}
    >
      {children}
    </div>
  );
}
