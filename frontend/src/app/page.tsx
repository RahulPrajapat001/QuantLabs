"use client";

import Link from "next/link";

import { MetricCard, MetricRow } from "@/components/quantlab/metric-card";
import { PageHeader } from "@/components/quantlab/page-header";
import { SharpeDisplay } from "@/components/quantlab/sharpe-display";
import { SplitBadge } from "@/components/quantlab/split-badge";
import { EmptyState, ErrorState } from "@/components/quantlab/states";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton, SkeletonRows } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { api, useApi } from "@/lib/api/use-api";

const INDEX = "NIFTY500";
const TODAY = new Date().toISOString().slice(0, 10);

export default function OverviewPage() {
  const summary = useApi(() => api().researchSummary(), []);
  const runs = useApi(() => api().researchLog({ limit: 10 }), []);
  const universe = useApi(() => api().universeAsOf(INDEX, TODAY), []);
  const quality = useApi(() => api().qualityFindings({ limit: 1 }), []);
  const securities = useApi(() => api().securities({ limit: 1 }), []);

  return (
    <>
      <PageHeader
        title="Overview"
        description="Universe coverage, data health, and every trial the research log has recorded. The trials counter is the number every Sharpe on this platform is deflated against."
      />

      {/* --- Trials counter: the headline metric on this page. --- */}
      <Card className="mb-2">
        <CardHeader>
          <div className="flex flex-col gap-1">
            <CardTitle>Research log</CardTitle>
            <p className="max-w-prose text-sm text-text-secondary">
              Every backtest writes here from inside the engine. A run that fails
              to log raises — so this count is the true denominator for multiple
              testing, not a number anyone remembered to increment.
            </p>
          </div>
          <Link
            href="/research-log"
            className="text-sm text-text-secondary underline underline-offset-2 hover:text-text-primary ql-fade"
          >
            Open log
          </Link>
        </CardHeader>
        <CardContent>
          {summary.loading ? (
            <Skeleton className="h-8 w-full" />
          ) : summary.error ? (
            <ErrorState
              message={summary.error}
              onRetry={summary.refetch}
              height={96}
            />
          ) : summary.data ? (
            <div className="flex flex-wrap items-start gap-6">
              <div className="flex flex-col gap-1">
                <span className="text-xs uppercase tracking-wide text-text-secondary">
                  Total trials
                </span>
                {/*
                 * From the log's own count, not from the page of runs above it:
                 * the trials denominator must not depend on how many rows a
                 * client happened to request.
                 */}
                <span className="ql-num text-xl font-medium text-text-primary">
                  {runs.data ? runs.data.total_trials.formatted : "—"}
                </span>
                <span className="text-xs text-text-muted">
                  {summary.data.first_run_at
                    ? `since ${summary.data.first_run_at}`
                    : "no runs yet"}
                </span>
              </div>

              {summary.data.best ? (
                <div className="flex flex-col gap-1">
                  <span className="text-xs uppercase tracking-wide text-text-secondary">
                    Best Sharpe recorded
                  </span>
                  <SharpeDisplay report={summary.data.best} size="xl" />
                </div>
              ) : (
                <EmptyState
                  message="No completed run yet."
                  hint="Run a backtest and it will appear here with its trials count."
                />
              )}
            </div>
          ) : null}
        </CardContent>
      </Card>

      {summary.data && (
        <MetricRow className="mb-2">
          <MetricCard
            label="Complete runs"
            value={summary.data.complete_runs}
            benchmark={`${summary.data.failed_runs.formatted} failed`}
          />
          <MetricCard
            label="Pre-registered"
            value={summary.data.preregistered_runs}
            benchmark={`${summary.data.exploratory_runs.formatted} exploratory`}
          />
          <MetricCard
            label="Holdout runs"
            value={summary.data.holdout_runs}
            benchmark="evaluated once, at the end"
          />
          <MetricCard
            label="Dirty-tree runs"
            value={summary.data.dirty_runs}
            benchmark="not reproducible from a commit alone"
          />
          <MetricCard
            label="Securities"
            value={{
              value: securities.data?.total ?? null,
              formatted: securities.data ? String(securities.data.total) : "—",
            }}
            benchmark="ingested, including delisted"
          />
          <MetricCard
            label={`${INDEX} today`}
            value={
              universe.data?.count ?? { value: null, formatted: "—" }
            }
            benchmark={universe.data ? `as of ${universe.data.as_of_display}` : ""}
          />
        </MetricRow>
      )}

      <div className="grid gap-2 lg:grid-cols-2">
        {/* --- Data coverage --- */}
        <Card>
          <CardHeader>
            <CardTitle>Data quality</CardTitle>
            <Link
              href="/data-quality"
              className="text-sm text-text-secondary underline underline-offset-2 hover:text-text-primary ql-fade"
            >
              Drill down
            </Link>
          </CardHeader>
          <CardContent>
            {quality.loading ? (
              <Skeleton className="h-7 w-full" />
            ) : quality.error ? (
              <ErrorState message={quality.error} onRetry={quality.refetch} height={112} />
            ) : quality.data ? (
              <div className="flex flex-col gap-2">
                <div className="flex flex-wrap gap-2">
                  {Object.entries(quality.data.counts).length === 0 ? (
                    <EmptyState
                      message="No findings."
                      hint="Either the data is clean or `quantlab validate` has not run."
                    />
                  ) : (
                    Object.entries(quality.data.counts).map(([check, count]) => (
                      <span
                        key={check}
                        className="rounded border border-border px-2 py-1 text-xs text-text-secondary"
                      >
                        {check.replace(/_/g, " ")}{" "}
                        <span className="ql-num text-text-primary">{count}</span>
                      </span>
                    ))
                  )}
                </div>
                <p className="text-xs text-text-muted">
                  {quality.data.rows_examined.formatted} rows examined ·{" "}
                  {quality.data.generated_on_demand
                    ? "computed on demand (no persisted run)"
                    : `persisted run ${quality.data.run_id?.slice(0, 8)}`}
                </p>
              </div>
            ) : null}
          </CardContent>
        </Card>

        {/* --- Universe summary --- */}
        <Card>
          <CardHeader>
            <CardTitle>Universe</CardTitle>
            <Link
              href="/universe"
              className="text-sm text-text-secondary underline underline-offset-2 hover:text-text-primary ql-fade"
            >
              Membership viewer
            </Link>
          </CardHeader>
          <CardContent>
            {universe.loading ? (
              <Skeleton className="h-7 w-full" />
            ) : universe.error ? (
              <ErrorState
                message={universe.error}
                onRetry={universe.refetch}
                height={112}
              />
            ) : universe.data ? (
              <div className="flex flex-col gap-2">
                <p className="text-sm text-text-secondary">
                  <span className="ql-num text-md text-text-primary">
                    {universe.data.count.formatted}
                  </span>{" "}
                  constituents of {universe.data.index_name} on{" "}
                  <span className="ql-num">{universe.data.as_of_display}</span>,
                  resolved from reconstitution filings rather than today&apos;s
                  index page.
                </p>
                <p className="text-xs text-text-muted">
                  Membership is point-in-time. A backtest that used today&apos;s
                  list would silently exclude everything that has since been
                  removed.
                </p>
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>

      {/* --- Recent runs --- */}
      <Card className="mt-2">
        <CardHeader>
          <CardTitle>Recent backtest runs</CardTitle>
          <Link
            href="/research-log"
            className="text-sm text-text-secondary underline underline-offset-2 hover:text-text-primary ql-fade"
          >
            All runs
          </Link>
        </CardHeader>
        <CardContent>
          {runs.loading ? (
            <SkeletonRows rows={10} cols={6} />
          ) : runs.error ? (
            <ErrorState message={runs.error} onRetry={runs.refetch} height={320} />
          ) : !runs.data || runs.data.items.length === 0 ? (
            <EmptyState
              message="No runs recorded."
              hint="Submit one from the Backtest page; it will be logged automatically."
              height={320}
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Strategy</TableHead>
                  <TableHead>Window</TableHead>
                  <TableHead>Split</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead numeric>Sharpe</TableHead>
                  <TableHead numeric>Deflated</TableHead>
                  <TableHead numeric>Trials</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {runs.data.items.map((run) => (
                  <TableRow key={run.run_id}>
                    <TableCell>
                      <span className="text-text-primary">{run.strategy_name}</span>
                      {run.exploratory && (
                        <span className="ml-2 text-xs uppercase text-text-muted">
                          exploratory
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="ql-num text-text-secondary">
                      {run.window_display}
                    </TableCell>
                    <TableCell>
                      <SplitBadge badge={run.badge} />
                    </TableCell>
                    <TableCell className="text-text-secondary">{run.status}</TableCell>
                    <TableCell numeric>
                      {run.sharpe ? run.sharpe.value_display.formatted : "—"}
                    </TableCell>
                    <TableCell numeric>
                      {run.sharpe ? run.sharpe.deflated_display.formatted : "—"}
                    </TableCell>
                    <TableCell numeric>
                      {run.sharpe ? run.sharpe.trials_display.formatted : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </>
  );
}
