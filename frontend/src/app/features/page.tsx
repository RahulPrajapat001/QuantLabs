"use client";

import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/chart-card";
import { PlotlyChart } from "@/components/charts/plotly-chart";
import { readTokens, useTheme } from "@/components/charts/theme";
import { useSelectedSecurity } from "@/components/quantlab/command-palette";
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
import { api, useApi } from "@/lib/api/use-api";
import { cn } from "@/lib/utils";

export default function FeatureExplorerPage() {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState<string>("all");
  const [feature, setFeature] = useState<string | null>(null);
  const [security] = useSelectedSecurity();

  const catalog = useApi(
    () =>
      api().featureCatalog({
        search: search || undefined,
        category: category === "all" ? undefined : category,
        limit: 500,
      }),
    [search, category],
  );

  // The panel is only fetched once a feature *and* a security are chosen —
  // asking for every feature of every security would be a very expensive way to
  // render a chart nobody asked for yet.
  const panel = useApi(
    () =>
      api().features({
        security_id: security ?? undefined,
        names: feature ?? undefined,
        limit: 5000,
      }),
    [security, feature],
    { enabled: Boolean(security && feature) },
  );

  const categories = catalog.data?.categories ?? {};
  const selected = catalog.data?.items.find((item) => item.name === feature) ?? null;

  return (
    <>
      <SidebarPortal>
        <Control label="Search features" hint="Press / to focus">
          {(id) => (
            <Input
              id={id}
              data-page-search
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="rsi, momentum, volatility"
            />
          )}
        </Control>

        <Control label="Category">
          {(id) => (
            <select
              id={id}
              value={category}
              onChange={(event) => setCategory(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              <option value="all">
                All ({catalog.data?.total.formatted ?? "—"})
              </option>
              {Object.entries(categories)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([name, count]) => (
                  <option key={name} value={name}>
                    {name} ({count})
                  </option>
                ))}
            </select>
          )}
        </Control>

        <Control
          label="Security"
          hint={security ? undefined : "Press ⌘K to pick one for the chart."}
        >
          {() => (
            <span className="ql-num text-sm text-text-primary">
              {security ?? "—"}
            </span>
          )}
        </Control>
      </SidebarPortal>

      <PageHeader
        title="Feature Explorer"
        description="The registry, not the parquet panel: this lists what can be computed, its dependencies and the history it needs before its first valid value."
      />

      <div className="grid gap-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <div className="flex flex-col gap-1">
              <CardTitle>Catalog</CardTitle>
              <p className="text-sm text-text-secondary">
                Descriptions come from each compute function&apos;s docstring —
                one copy, so it cannot drift from the code.
              </p>
            </div>
            <span className="ql-num text-xs text-text-muted">
              {catalog.data?.returned.formatted ?? "—"} of{" "}
              {catalog.data?.total.formatted ?? "—"}
            </span>
          </CardHeader>
          <CardContent>
            {catalog.loading ? (
              <SkeletonRows rows={20} cols={4} />
            ) : catalog.error ? (
              <ErrorState
                message={catalog.error}
                onRetry={catalog.refetch}
                height={592}
              />
            ) : !catalog.data || catalog.data.items.length === 0 ? (
              <EmptyState
                message="No feature matches."
                hint="Clear the search, or widen the category filter."
                height={592}
              />
            ) : (
              <div className="max-h-screen overflow-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Feature</TableHead>
                      <TableHead>Category</TableHead>
                      <TableHead numeric>Lookback</TableHead>
                      <TableHead>Kind</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {catalog.data.items.map((item) => (
                      <TableRow
                        key={item.name}
                        onClick={() => setFeature(item.name)}
                        className={cn(
                          "cursor-pointer",
                          item.name === feature && "bg-surface-raised",
                        )}
                      >
                        <TableCell className="ql-num text-text-primary">
                          {item.name}
                        </TableCell>
                        <TableCell className="text-text-secondary">
                          {item.category}
                        </TableCell>
                        <TableCell numeric className="text-text-secondary">
                          {item.lookback.formatted}
                        </TableCell>
                        <TableCell className="text-text-muted">
                          {item.kind === "cross_sectional" ? "cross-sec" : "series"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-2">
          <Card>
            <CardHeader>
              <CardTitle>{selected ? selected.name : "No feature selected"}</CardTitle>
              {selected && (
                <span className="rounded border border-border px-1 text-xs text-text-secondary">
                  {selected.category}
                </span>
              )}
            </CardHeader>
            <CardContent>
              {!selected ? (
                <EmptyState
                  message="Pick a feature from the catalog."
                  hint="Its description, dependencies and lookback appear here."
                  height={120}
                />
              ) : (
                <div className="flex flex-col gap-2">
                  <p className="max-w-prose text-base text-text-secondary">
                    {selected.description}
                  </p>
                  <dl className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex flex-col">
                      <dt className="text-xs uppercase tracking-wide text-text-secondary">
                        Lookback
                      </dt>
                      <dd className="ql-num text-text-primary">
                        {selected.lookback.formatted} bars
                      </dd>
                    </div>
                    <div className="flex flex-col">
                      <dt className="text-xs uppercase tracking-wide text-text-secondary">
                        Kind
                      </dt>
                      <dd className="text-text-primary">{selected.kind}</dd>
                    </div>
                    <div className="col-span-2 flex flex-col">
                      <dt className="text-xs uppercase tracking-wide text-text-secondary">
                        Depends on
                      </dt>
                      <dd className="ql-num text-text-primary">
                        {selected.dependencies.length
                          ? selected.dependencies.join(", ")
                          : "—"}
                      </dd>
                    </div>
                  </dl>
                </div>
              )}
            </CardContent>
          </Card>

          <FeatureChart
            feature={feature}
            security={security}
            panel={panel.data}
            loading={panel.loading}
            error={panel.error}
            onRetry={panel.refetch}
            lookback={selected?.lookback.formatted ?? null}
          />
        </div>
      </div>
    </>
  );
}

function FeatureChart({
  feature,
  security,
  panel,
  loading,
  error,
  onRetry,
  lookback,
}: {
  feature: string | null;
  security: string | null;
  panel: Awaited<ReturnType<ReturnType<typeof api>["features"]>> | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
  lookback: string | null;
}) {
  const theme = useTheme();

  const points = useMemo(() => {
    if (!panel || !feature) return { x: [] as string[], y: [] as (number | null)[] };
    const rows = panel.rows.filter((row) => feature in row.values);
    return {
      x: rows.map((row) => row.date),
      y: rows.map((row) => row.values[feature] ?? null),
    };
  }, [panel, feature]);

  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "scatter" as const,
        mode: "lines" as const,
        x: points.x,
        y: points.y,
        line: { color: tokens.series[0], width: 2 },
        // A gap, not a bridge: connecting across the warm-up would draw a line
        // through values that do not exist.
        connectgaps: false,
        hovertemplate: "%{x}<br>%{y}<extra></extra>",
      },
    ];
  }, [points, theme]);

  const missing = !feature || !security;

  return (
    <ChartCard
      title={feature ? `${feature} on ${security ?? "—"}` : "Feature series"}
      caption={
        lookback
          ? `The leading gap is the ${lookback}-bar warm-up: the feature has no defined value until it has that much history.`
          : undefined
      }
      height={300}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={missing || points.x.length === 0}
      emptyMessage={
        missing
          ? "Select a feature and a security."
          : "The feature panel has no rows for this pair."
      }
      emptyHint={
        missing
          ? "Pick a feature on the left; press ⌘K to choose a security."
          : "Build the panel with `quantlab features build`."
      }
      table={{
        columns: ["Date", feature ?? "Value"],
        rows: points.x.map((date, i) => {
          const value = points.y[i];
          return [date, value === null || value === undefined ? "—" : String(value)];
        }),
      }}
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { showgrid: false, tickformat: "%Y-%m-%d" },
          yaxis: { title: { text: feature ?? "" } },
          hovermode: "x unified",
        }}
        height={300}
        ariaLabel={`${feature ?? "Feature"} time series`}
      />
    </ChartCard>
  );
}
