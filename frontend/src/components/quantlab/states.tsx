/**
 * Empty and error states. Every chart and table has both (DESIGN_SYSTEM.md §9).
 *
 * The distinction is not cosmetic. "No data" and "the request failed" are
 * different claims about the world, and collapsing them into one grey box is
 * how a broken ingest looks like an empty universe for a week.
 */

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

/**
 * `height` reserves the box the real content will occupy so nothing moves when
 * data lands. The *message* inside is pinned to the top rather than centred,
 * and capped in width: a 420px-tall chart slot with its "no data" text floating
 * dead-centre turns absence into the loudest thing on the screen. Data should be
 * the focus when it arrives, and the placeholder should not compete before then.
 */
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
      style={height ? { height } : undefined}
      className={cn(
        "flex flex-col items-start gap-1 rounded border border-dashed border-border px-3 py-2",
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

/**
 * One line, no box. For a slot inside a card that already has a border — a
 * second dashed rectangle inside a bordered card is two frames around nothing.
 */
export function EmptyLine({
  message,
  className,
}: {
  message: string;
  className?: string;
}) {
  return <p className={cn("text-sm text-text-muted", className)}>{message}</p>;
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
