/**
 * Empty and error states. Every chart and table has both (DESIGN_SYSTEM.md §9).
 *
 * The distinction is not cosmetic. "No data" and "the request failed" are
 * different claims about the world, and collapsing them into one grey box is
 * how a broken ingest looks like an empty universe for a week.
 */

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

function Frame({
  height,
  className,
  children,
}: {
  height?: number;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      // Same box the content would have occupied: swapping a state for the real
      // thing must not move the page.
      style={height ? { height } : undefined}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded border border-dashed border-border p-4 text-center",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function EmptyState({
  message,
  hint,
  height,
  className,
}: {
  message: string;
  /** What the reader should do to make data appear. */
  hint?: string;
  height?: number;
  className?: string;
}) {
  return (
    <Frame height={height} className={className}>
      <p className="text-sm text-text-secondary">{message}</p>
      {hint && <p className="max-w-prose text-xs text-text-muted">{hint}</p>}
    </Frame>
  );
}

export function ErrorState({
  message,
  onRetry,
  height,
  className,
}: {
  message: string;
  onRetry?: () => void;
  height?: number;
  className?: string;
}) {
  return (
    <Frame height={height} className={cn("border-negative", className)}>
      <p className="text-sm text-negative">Could not load this.</p>
      <p className="max-w-prose text-xs text-text-secondary">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          Retry
        </Button>
      )}
    </Frame>
  );
}
