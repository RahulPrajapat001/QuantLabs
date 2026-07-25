/**
 * The split badge (DESIGN_SYSTEM.md §7.2). There is no "no badge" state.
 *
 * `resolve_split` returns compound labels when a window spans a boundary
 * (`train+validation`). Those render as distinct segments — never collapsed to
 * the cleaner of the two, and never falling through to a default. An
 * unrecognised split renders `UNKNOWN` in `negative`, because an unlabelled
 * result is worse than a wrongly labelled one: it looks fine.
 */

import type { SplitBadge as SplitBadgeData } from "@/lib/api/client";
import { cn } from "@/lib/utils";

const TONE: Record<string, string> = {
  TRAIN: "border-info text-info",
  VALIDATION: "border-warning text-warning",
  "PRE-SAMPLE": "border-border text-text-muted",
};

/** HOLDOUT is filled — the only filled badge in the system, and deliberately
 * the loudest element on any screen it appears on. The holdout is evaluated
 * once, at the end, so a screen showing holdout numbers should look unusual. */
const HOLDOUT = "border-negative bg-negative text-bg";
const UNKNOWN = "border-negative text-negative";

function segmentTone(part: string): string {
  if (part === "HOLDOUT") return HOLDOUT;
  return TONE[part] ?? UNKNOWN;
}

export function SplitBadge({
  badge,
  className,
}: {
  badge: SplitBadgeData;
  className?: string;
}) {
  const parts = badge.parts.length > 0 ? badge.parts : ["UNKNOWN"];

  return (
    <span
      className={cn("inline-flex items-center gap-1", className)}
      title={`Split: ${badge.label}`}
    >
      {parts.map((part, index) => (
        <span key={`${part}-${index}`} className="inline-flex items-center gap-1">
          {index > 0 && <span className="text-xs text-text-muted">+</span>}
          <span
            className={cn(
              "rounded border px-1 text-xs font-medium uppercase tracking-wide",
              segmentTone(part.toUpperCase()),
            )}
          >
            {part.toUpperCase()}
          </span>
        </span>
      ))}
    </span>
  );
}
