import { cn } from "@/lib/utils";

/**
 * The loading state for everything. **Never a spinner.**
 *
 * A spinner says "something is happening"; a skeleton says "a table of this
 * shape is arriving", and because it occupies the final layout exactly, nothing
 * moves when the data lands. That is the whole point — a research screen that
 * reflows under the cursor loses the row the reader was looking at.
 *
 * Callers must pass the height the real content will occupy. There is no
 * default height, because a default would be wrong on every second use and the
 * layout shift would be invisible in review.
 */
export function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      aria-hidden
      className={cn("animate-ql-pulse rounded bg-surface-raised", className)}
      {...props}
    />
  );
}

/** Rows of a table skeleton, at the exact 28px row height tables use. */
export function SkeletonRows({ rows = 20, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="flex flex-col" aria-hidden>
      {Array.from({ length: rows }, (_, r) => (
        <div key={r} className="flex h-table-row items-center gap-3 px-2">
          {Array.from({ length: cols }, (_, c) => (
            <Skeleton
              key={c}
              className="h-2"
              // Varying widths so the placeholder reads as a table of values
              // rather than a grid of identical bars.
              style={{ width: c === 0 ? "18%" : `${10 + ((r + c) % 4) * 4}%` }}
            />
          ))}
        </div>
      ))}
    </div>
  );
}

/** A chart-shaped placeholder. `height` must match the chart it replaces. */
export function SkeletonChart({ height }: { height: number }) {
  return <Skeleton className="w-full" style={{ height }} />;
}
