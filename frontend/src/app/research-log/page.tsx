"use client";

import { useMemo, useState } from "react";

import { MetricCard, MetricRow } from "@/components/quantlab/metric-card";
import { PageHeader } from "@/components/quantlab/page-header";
import { Control, SidebarPortal } from "@/components/quantlab/sidebar-context";
import { SplitBadge } from "@/components/quantlab/split-badge";
import { EmptyState, ErrorState } from "@/components/quantlab/states";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { SkeletonRows } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { RunOut } from "@/lib/api/client";
import { api, useApi } from "@/lib/api/use-api";
import { cn } from "@/lib/utils";

type SortKey =
  | "created_at"
  | "strategy_name"
  | "sharpe"
  | "deflated"
  | "trials"
  | "status";

type Provenance = "all" | "preregistered" | "exploratory";

export default function ResearchLogPage() {
  const [search, setSearch] = useState("");
  const [provenance, setProvenance] = useState<Provenance>("all");
  const [sort, setSort] = useState<SortKey>("created_at");
  const [descending, setDescending] = useState(true);

  const log = useApi(() => api().researchLog({ limit: 500 }), []);
  const summary = useApi(() => api().researchSummary(), []);

  const rows = useMemo(() => {
    let items = [...(log.data?.items ?? [])];

    if (provenance !== "all") {
      items = items.filter((run) =>
        provenance === "exploratory" ? run.exploratory : !run.exploratory,
      );
    }
    if (search) {
      const needle = search.toLowerCase();
      items = items.filter(
        (run) =>
          run.strategy_name.toLowerCase().includes(needle) ||
          run.run_id.toLowerCase().includes(needle) ||
          (run.hypothesis_id ?? "").toLowerCase().includes(needle),
      );
    }

    const value = (run: RunOut): string | number => {
      switch (sort) {
        case "strategy_name":
          return run.strategy_name;
        case "sharpe":
          return run.sharpe?.value ?? Number.NEGATIVE_INFINITY;
        case "deflated":
          return run.sharpe?.deflated_sharpe ?? Number.NEGATIVE_INFINITY;
        case "trials":
          return run.sharpe?.trials ?? 0;
        case "status":
          return run.status;
        default:
          return run.created_at ?? "";
      }
    };

    items.sort((a, b) => {
      const left = value(a);
      const right = value(b);
      const order =
        typeof left === "string" && typeof right === "string"
          ? left.localeCompare(right)
          : Number(left) - Number(right);
      return descending ? -order : order;
    });
    return items;
  }, [log.data, provenance, search, sort, descending]);

  function toggle(key: SortKey) {
    if (sort === key) setDescending((current) => !current);
    else {
      setSort(key);
      setDescending(true);
    }
  }

  return (
    <>
      <SidebarPortal>
        <Control label="Search" hint="Press / to focus">
          {(id) => (
            <Input
              id={id}
              data-page-search
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="strategy, run id, H1"
            />
          )}
        </Control>

        <Control label="Provenance">
          {(id) => (
            <select
              id={id}
              value={provenance}
              onChange={(event) => setProvenance(event.target.value as Provenance)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              <option value="all">All runs</option>
              <option value="preregistered">Pre-registered only</option>
              <option value="exploratory">Exploratory only</option>
            </select>
          )}
        </Control>
      </SidebarPortal>

      <PageHeader
        title="Research Log"
        description="Every backtest ever run, written from inside the engine. This table is the denominator for multiple testing: a Sharpe selected from this many trials is not the same claim as a Sharpe from one."
      />

      <MetricRow className="mb-2">
        <MetricCard
          label="Total trials"
          value={log.data?.total_trials ?? { value: null, formatted: "—" }}
          benchmark="what every Sharpe is deflated against"
        />
        <MetricCard
          label="Complete"
          value={summary.data?.complete_runs ?? { value: null, formatted: "—" }}
          benchmark={`${summary.data?.failed_runs.formatted ?? "—"} failed, ${summary.data?.running_runs.formatted ?? "—"} running`}
        />
        <MetricCard
          label="Pre-registered"
          value={summary.data?.preregistered_runs ?? { value: null, formatted: "—" }}
          benchmark="predicted before the test"
        />
        <MetricCard
          label="Exploratory"
          value={summary.data?.exploratory_runs ?? { value: null, formatted: "—" }}
          benchmark="legitimate, but labelled"
        />
        <MetricCard
          label="Holdout touched"
          value={summary.data?.holdout_runs ?? { value: null, formatted: "—" }}
          benchmark="evaluated once, at the end"
        />
        <MetricCard
          label="Not reproducible"
          value={summary.data?.dirty_runs ?? { value: null, formatted: "—" }}
          benchmark="ran on a dirty work tree"
        />
      </MetricRow>

      <Card>
        <CardHeader>
          <div className="flex flex-col gap-1">
            <CardTitle>Runs</CardTitle>
            <p className="text-sm text-text-secondary">
              Sortable. Deflated Sharpe is its own column, not a tooltip.
            </p>
          </div>
          <span className="ql-num text-xs text-text-muted">{rows.length} shown</span>
        </CardHeader>
        <CardContent>
          {log.loading ? (
            <SkeletonRows rows={20} cols={8} />
          ) : log.error ? (
            <ErrorState message={log.error} onRetry={log.refetch} height={592} />
          ) : rows.length === 0 ? (
            <EmptyState
              message="No runs match."
              hint={
                log.data?.items.length === 0
                  ? "Run a backtest and it will be logged automatically."
                  : "Clear the search or widen the provenance filter."
              }
              height={592}
            />
          ) : (
            <div className="max-h-screen overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <SortableHead
                      label="Run"
                      active={sort === "created_at"}
                      descending={descending}
                      onClick={() => toggle("created_at")}
                    />
                    <SortableHead
                      label="Strategy"
                      active={sort === "strategy_name"}
                      descending={descending}
                      onClick={() => toggle("strategy_name")}
                    />
                    <TableHead>Window</TableHead>
                    <TableHead>Split</TableHead>
                    <TableHead>Provenance</TableHead>
                    <SortableHead
                      label="Status"
                      active={sort === "status"}
                      descending={descending}
                      onClick={() => toggle("status")}
                    />
                    <SortableHead
                      label="Sharpe"
                      numeric
                      active={sort === "sharpe"}
                      descending={descending}
                      onClick={() => toggle("sharpe")}
                    />
                    <SortableHead
                      label="Deflated"
                      numeric
                      active={sort === "deflated"}
                      descending={descending}
                      onClick={() => toggle("deflated")}
                    />
                    <SortableHead
                      label="Trials"
                      numeric
                      active={sort === "trials"}
                      descending={descending}
                      onClick={() => toggle("trials")}
                    />
                    <TableHead>Repro</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((run) => (
                    <TableRow key={run.run_id}>
                      <TableCell className="ql-num text-text-secondary">
                        {run.run_id.slice(0, 8)}
                      </TableCell>
                      <TableCell className="text-text-primary">
                        {run.strategy_name}
                      </TableCell>
                      <TableCell className="ql-num text-text-secondary">
                        {run.window_display}
                      </TableCell>
                      <TableCell>
                        <SplitBadge badge={run.badge} />
                      </TableCell>
                      <TableCell>
                        {run.exploratory ? (
                          <span className="border-l border-dashed border-border-strong pl-2 text-xs uppercase tracking-wide text-text-muted">
                            Exploratory
                          </span>
                        ) : (
                          <span className="border-l border-solid border-border-strong pl-2 text-xs text-text-secondary">
                            <span className="ql-num text-text-primary">
                              {run.hypothesis_id ?? "—"}
                            </span>
                          </span>
                        )}
                      </TableCell>
                      <TableCell
                        className={cn(
                          "text-text-secondary",
                          run.status === "failed" && "text-negative",
                        )}
                      >
                        {run.status}
                      </TableCell>
                      <TableCell
                        numeric
                        className={
                          run.costs.realistic ? undefined : "text-text-muted"
                        }
                      >
                        {run.sharpe?.value_display.formatted ?? "—"}
                      </TableCell>
                      <TableCell numeric>
                        {run.sharpe?.deflated_display.formatted ?? "—"}
                      </TableCell>
                      <TableCell numeric>
                        {run.sharpe?.trials_display.formatted ?? "—"}
                      </TableCell>
                      <TableCell
                        className={
                          run.reproducible ? "text-text-muted" : "text-warning"
                        }
                      >
                        {run.reproducible ? "clean" : "⚠ dirty"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}

function SortableHead({
  label,
  numeric,
  active,
  descending,
  onClick,
}: {
  label: string;
  numeric?: boolean;
  active: boolean;
  descending: boolean;
  onClick: () => void;
}) {
  return (
    <TableHead numeric={numeric}>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "inline-flex items-center gap-1 hover:text-text-primary ql-fade",
          active && "text-text-primary",
        )}
      >
        {label}
        <span aria-hidden className="text-text-muted">
          {active ? (descending ? "↓" : "↑") : ""}
        </span>
      </button>
    </TableHead>
  );
}
