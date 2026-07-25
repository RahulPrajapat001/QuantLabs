/**
 * The unrealistic-cost marker (DESIGN_SYSTEM.md §7.3).
 *
 * Any result whose cost model is not exactly `IndianEquityCostModel` carries an
 * icon *and* the model name — never colour alone, because colour alone excludes
 * the reader who cannot distinguish it. The API decides what counts as
 * unrealistic (`CostModelStatus.realistic`); this renders the verdict.
 */

import type { CostModelStatus } from "@/lib/api/client";
import { cn } from "@/lib/utils";

export function CostWarning({
  costs,
  className,
}: {
  costs: CostModelStatus;
  className?: string;
}) {
  if (costs.realistic) return null;

  return (
    <span
      className={cn("inline-flex items-baseline gap-1 text-xs text-warning", className)}
      title={costs.name}
    >
      <span aria-hidden>⚠</span>
      <span>{costs.warning ?? costs.name}</span>
    </span>
  );
}
