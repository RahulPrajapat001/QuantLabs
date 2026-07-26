"use client";

/**
 * The frame every chart sits in, and the reason the design system's "every
 * chart has a table view" rule is cheap to obey rather than a per-page chore.
 *
 * It owns the four states so no page reimplements them: loading renders a
 * skeleton of the chart's exact height, error renders the message with a retry,
 * empty renders what is missing and what to do, and data renders the chart.
 * Because all four occupy the same box, nothing on the page moves as a request
 * settles.
 *
 * The table view is the accessibility fallback and it is also how a researcher
 * gets the number instead of the picture — which is usually what they wanted.
 */

import { useState } from "react";

import { EmptyState, ErrorState } from "@/components/quantlab/states";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { SkeletonChart } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

export interface TableView {
  columns: string[];
  /** Pre-formatted strings. The frontend does not format numbers. */
  rows: string[][];
  /** Column indices to render mono and right-aligned. Defaults to all but 0. */
  numericFrom?: number;
}

export interface ChartCardProps {
  title: string;
  /**
   * States the finding, not the axes — the FT convention. "Momentum decays
   * beyond 10 days", not "IC by horizon".
   */
  caption?: string;
  height: number;
  loading?: boolean;
  error?: string | null;
  /** True when the request succeeded and there is genuinely nothing to plot. */
  empty?: boolean;
  emptyMessage?: string;
  emptyHint?: string;
  onRetry?: () => void;
  table?: TableView;
  /** Rendered in the header, right-aligned: a badge, a toggle, a count. */
  actions?: React.ReactNode;
  /**
   * A multi-line report from the API, rendered below the chart with its line
   * breaks intact. Captions are one sentence; putting a rendered report in one
   * collapses its newlines into a run-on.
   */
  footnote?: string;
  children: React.ReactNode;
  className?: string;
}

export function ChartCard({
  title,
  caption,
  height,
  loading = false,
  error = null,
  empty = false,
  emptyMessage = "No data for this selection.",
  emptyHint,
  onRetry,
  table,
  actions,
  footnote,
  children,
  className,
}: ChartCardProps) {
  const [view, setView] = useState<"chart" | "table">("chart");

  return (
    <Card className={className}>
      <CardHeader>
        <div className="flex min-w-0 flex-col gap-1">
          <CardTitle>{title}</CardTitle>
          {caption && (
            <p className="max-w-prose text-sm text-text-secondary">{caption}</p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-2">
          {/* A legend or a summary describes data; with none, it describes
              nothing and just adds ink. */}
          {!empty && !loading && !error && actions}
          {table && !loading && !error && !empty && (
            <button
              type="button"
              onClick={() => setView(view === "chart" ? "table" : "chart")}
              className="rounded border border-border px-1 text-xs text-text-secondary hover:border-border-strong hover:text-text-primary ql-fade"
              aria-pressed={view === "table"}
            >
              {view === "chart" ? "Table" : "Chart"}
            </button>
          )}
        </div>
      </CardHeader>

      <CardContent>
        {loading ? (
          <SkeletonChart height={height} />
        ) : error ? (
          <ErrorState message={error} onRetry={onRetry} height={height} />
        ) : empty ? (
          /*
           * Deliberately NOT `height={height}`. The no-layout-shift rule governs
           * loading → data, and the skeleton above reserves the full box for
           * that. An empty state is terminal: nothing arrives until the reader
           * changes a control, so reserving 420px to say "no data" makes absence
           * the largest element on the page.
           */
          <EmptyState message={emptyMessage} hint={emptyHint} />
        ) : view === "table" && table ? (
          <div className="overflow-auto" style={{ height }}>
            <Table>
              <TableHeader>
                <TableRow>
                  {table.columns.map((column, i) => (
                    <TableHead key={column} numeric={i >= (table.numericFrom ?? 1)}>
                      {column}
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {table.rows.map((row, r) => (
                  <TableRow key={r}>
                    {row.map((cell, c) => (
                      <TableCell key={c} numeric={c >= (table.numericFrom ?? 1)}>
                        {cell}
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        ) : (
          <div className={cn("animate-ql-fade-in")}>{children}</div>
        )}

        {footnote && !loading && !error && !empty && (
          <p className="mt-2 max-w-prose whitespace-pre-line border-t border-border pt-2 text-xs text-text-muted">
            {footnote}
          </p>
        )}
      </CardContent>
    </Card>
  );
}
