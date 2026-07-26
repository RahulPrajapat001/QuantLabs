"use client";

import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/chart-card";
import { PlotlyChart } from "@/components/charts/plotly-chart";
import { readTokens, useTheme } from "@/components/charts/theme";
import { useSelectedSecurity } from "@/components/quantlab/command-palette";
import { MetricCard, MetricRow } from "@/components/quantlab/metric-card";
import { PageHeader } from "@/components/quantlab/page-header";
import { Control, SidebarPortal } from "@/components/quantlab/sidebar-context";
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
import type { CompletenessGrid, Finding } from "@/lib/api/client";
import { api, useApi } from "@/lib/api/use-api";

const SEVERITIES = ["error", "warning", "info"] as const;
type Severity = (typeof SEVERITIES)[number];

export default function DataQualityPage() {
  const [selected] = useSelectedSecurity();
  const [severity, setSeverity] = useState<Severity | "all">("all");
  const [search, setSearch] = useState("");
  const [drillSecurity, setDrillSecurity] = useState<string | null>(null);

  const scope = drillSecurity ?? selected ?? null;

  const report = useApi(
    () => api().qualityFindings({ security_id: scope ?? undefined, limit: 2000 }),
    [scope],
  );
  const grid = useApi(() => api().completeness({ limit: 40 }), []);

  const findings = useMemo(() => {
    const items = report.data?.findings ?? [];
    return items.filter((finding) => {
      if (severity !== "all" && finding.severity !== severity) return false;
      if (!search) return true;
      const needle = search.toLowerCase();
      return (
        finding.check.toLowerCase().includes(needle) ||
        finding.detail.toLowerCase().includes(needle) ||
        (finding.security_id ?? "").toLowerCase().includes(needle)
      );
    });
  }, [report.data, severity, search]);

  const counts = report.data?.counts ?? {};
  const bySeverity = useMemo(() => {
    const totals: Record<string, number> = { error: 0, warning: 0, info: 0 };
    for (const finding of report.data?.findings ?? []) {
      totals[finding.severity] = (totals[finding.severity] ?? 0) + 1;
    }
    return totals;
  }, [report.data]);

  return (
    <>
      <SidebarPortal>
        <Control label="Severity">
          {(id) => (
            <select
              id={id}
              value={severity}
              onChange={(event) => setSeverity(event.target.value as Severity | "all")}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              <option value="all">All</option>
              {SEVERITIES.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          )}
        </Control>

        <Control label="Search findings" hint="Press / to focus">
          {(id) => (
            <Input
              id={id}
              data-page-search
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="check, detail, security"
            />
          )}
        </Control>

        {scope && (
          <Control label="Drill-down">
            {() => (
              <button
                type="button"
                onClick={() => setDrillSecurity(null)}
                className="h-6 rounded border border-border px-2 text-left text-sm text-text-secondary hover:border-border-strong hover:text-text-primary ql-fade"
              >
                <span className="ql-num">{scope}</span> — clear
              </button>
            )}
          </Control>
        )}
      </SidebarPortal>

      <PageHeader
        title="Data Quality"
        description={
          report.data?.generated_on_demand
            ? "Computed for this request — no validation run has been persisted. Run `quantlab validate` to record findings with an id and a history."
            : `Persisted run ${report.data?.run_id?.slice(0, 8) ?? ""}, executed ${report.data?.generated_at ?? "—"}.`
        }
      />

      <MetricRow className="mb-2">
        <MetricCard
          label="Errors"
          value={{ value: bySeverity.error ?? 0, formatted: String(bySeverity.error ?? 0) }}
          benchmark="a defect that invalidates derived values"
        />
        <MetricCard
          label="Warnings"
          value={{
            value: bySeverity.warning ?? 0,
            formatted: String(bySeverity.warning ?? 0),
          }}
          benchmark="worth a human looking"
        />
        <MetricCard
          label="Rows examined"
          value={report.data?.rows_examined ?? { value: null, formatted: "—" }}
          benchmark="ohlcv_daily rows in scope"
        />
        <MetricCard
          label="Overall completeness"
          value={grid.data?.overall ?? { value: null, formatted: "—" }}
          benchmark="bars present / sessions expected"
        />
        <MetricCard
          label="Checks run"
          value={{
            value: report.data?.checks_run.length ?? null,
            formatted: report.data ? String(report.data.checks_run.length) : "—",
          }}
          benchmark="see docs/PRD.md Module 3"
        />
        <MetricCard
          label="Distinct checks firing"
          value={{
            value: Object.keys(counts).length,
            formatted: String(Object.keys(counts).length),
          }}
        />
      </MetricRow>

      <CompletenessHeatmap
        grid={grid.data}
        loading={grid.loading}
        error={grid.error}
        onRetry={grid.refetch}
      />

      <Card className="mt-2">
        <CardHeader>
          <div className="flex flex-col gap-1">
            <CardTitle>Findings</CardTitle>
            <p className="text-sm text-text-secondary">
              Every finding carries a remedy, not just a complaint. Click a row to
              scope this page to that security.
            </p>
          </div>
          <span className="ql-num text-xs text-text-muted">
            {findings.length} shown
          </span>
        </CardHeader>
        <CardContent>
          {report.loading ? (
            <SkeletonRows rows={20} cols={5} />
          ) : report.error ? (
            <ErrorState message={report.error} onRetry={report.refetch} height={592} />
          ) : findings.length === 0 ? (
            <EmptyState
              message="No findings match."
              hint={
                report.data?.findings.length === 0
                  ? "Either the data is clean or no validation run has been persisted."
                  : "Widen the severity filter or clear the search."
              }
              height={592}
            />
          ) : (
            <div className="max-h-screen overflow-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Severity</TableHead>
                    <TableHead>Check</TableHead>
                    <TableHead>Security</TableHead>
                    <TableHead>Date</TableHead>
                    <TableHead>Detail</TableHead>
                    <TableHead>Remedy</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {findings.slice(0, 500).map((finding, index) => (
                    <FindingRow
                      key={`${finding.check}-${finding.security_id}-${finding.date}-${index}`}
                      finding={finding}
                      onDrill={setDrillSecurity}
                    />
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

function FindingRow({
  finding,
  onDrill,
}: {
  finding: Finding;
  onDrill: (id: string) => void;
}) {
  const tone =
    finding.severity === "error"
      ? "text-negative"
      : finding.severity === "warning"
        ? "text-warning"
        : "text-info";

  return (
    <TableRow
      className={finding.security_id ? "cursor-pointer" : undefined}
      onClick={() => finding.security_id && onDrill(finding.security_id)}
    >
      <TableCell className={`text-xs uppercase ${tone}`}>{finding.severity}</TableCell>
      <TableCell className="text-text-primary">
        {finding.check.replace(/_/g, " ")}
      </TableCell>
      <TableCell className="ql-num text-text-secondary">
        {finding.security_id ?? "—"}
      </TableCell>
      <TableCell className="ql-num text-text-secondary">
        {finding.date_display}
      </TableCell>
      <TableCell className="text-text-secondary">{finding.detail}</TableCell>
      <TableCell className="text-text-muted">{finding.remedy}</TableCell>
    </TableRow>
  );
}

/**
 * Security × month, shaded by the share of expected sessions present.
 *
 * Sequential, not diverging: completeness is a magnitude with one direction, and
 * a diverging scale would imply a meaningful midpoint that does not exist. A
 * null cell (the security was not listed) is left unshaded rather than drawn as
 * 0%, so an unlisted month never reads as a data gap.
 */
function CompletenessHeatmap({
  grid,
  loading,
  error,
  onRetry,
}: {
  grid: CompletenessGrid | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  const plot = useMemo(() => {
    if (!grid || grid.cells.length === 0) return null;
    const tokens = readTokens();
    const index = new Map(grid.securities.map((s, i) => [s, i]));
    const monthIndex = new Map(grid.months.map((m, i) => [m, i]));

    const z: (number | null)[][] = grid.securities.map(() =>
      grid.months.map(() => null),
    );
    for (const cell of grid.cells) {
      const r = index.get(cell.security_id);
      const c = monthIndex.get(cell.month);
      if (r === undefined || c === undefined) continue;
      const row = z[r];
      if (row) row[c] = cell.completeness.value;
    }

    return {
      data: [
        {
          type: "heatmap" as const,
          z,
          x: grid.months,
          y: grid.securities,
          colorscale: tokens.sequential.map((colour, i) => [
            i / (tokens.sequential.length - 1),
            colour,
          ]) as [number, string][],
          zmin: 0,
          zmax: 1,
          hoverongaps: false,
          hovertemplate:
            "%{y} · %{x}<br>completeness %{z:.1%}<extra></extra>",
          colorbar: {
            thickness: 8,
            outlinewidth: 0,
            tickformat: ",.0%",
            tickfont: { size: 11 },
          },
        },
      ],
      layout: {
        // Every third month is labelled. With ~42 categories all of them
        // collide into unreadable diagonal mush; the cells still carry the exact
        // month in the hover and in the table view.
        xaxis: {
          showgrid: false,
          type: "category" as const,
          tickmode: "auto" as const,
          nticks: 14,
          tickangle: -45,
          // Rotated labels need room or they are clipped mid-string — "2016-01"
          // rendered as "16-01", which looks like a different date entirely.
          automargin: true,
        },
        yaxis: { showgrid: false, type: "category" as const, automargin: true },
      },
    };
  }, [grid, theme]);

  return (
    <ChartCard
      title="Completeness by security and month"
      caption="Bars present against sessions the exchange actually had. An unlisted month is left blank, never shaded as a gap."
      height={420}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={!grid || grid.cells.length === 0}
      emptyMessage="No price history ingested."
      emptyHint="Run `quantlab ingest` and this grid fills in."
      actions={
        grid?.truncated ? (
          <span className="text-xs text-warning">
            ⚠ showing first {grid.securities.length} securities
          </span>
        ) : null
      }
      table={
        grid
          ? {
              columns: ["Security", "Month", "Bars", "Expected", "Completeness"],
              rows: grid.cells
                .slice(0, 400)
                .map((cell) => [
                  cell.security_id,
                  cell.month,
                  cell.bars.formatted,
                  cell.expected.formatted,
                  cell.completeness.formatted,
                ]),
              numericFrom: 2,
            }
          : undefined
      }
    >
      {/*
       * Read-only by design: drill-down happens from the findings table below,
       * where the row already names the security. Wiring Plotly click events
       * through the wrapper to duplicate that path would add a second way to do
       * one thing.
       */}
      {plot && (
        <PlotlyChart
          data={plot.data}
          layout={plot.layout}
          height={420}
          ariaLabel="Completeness heatmap by security and month"
        />
      )}
    </ChartCard>
  );
}
