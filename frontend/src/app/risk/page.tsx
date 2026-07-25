"use client";

import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/chart-card";
import { PlotlyChart } from "@/components/charts/plotly-chart";
import { readTokens, useTheme } from "@/components/charts/theme";
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
import type { BetaOut, CorrelationOut, MonteCarloOut } from "@/lib/api/client";
import { api, useApi } from "@/lib/api/use-api";

export default function RiskPage() {
  const [index, setIndex] = useState("NIFTY500");
  const [start, setStart] = useState("2015-01-01");
  const [end, setEnd] = useState("2021-12-31");
  const [lookback, setLookback] = useState(504);

  const body = { index_name: index, start, end, lookback };

  const varReport = useApi(() => api().valueAtRisk(body), [index, start, end, lookback]);
  const beta = useApi(
    () => api().beta({ index_name: index, start, end, lookback: 756, window: 60 }),
    [index, start, end],
  );
  const correlation = useApi(
    () => api().correlation({ index_name: index, start, end, lookback: 252, limit: 40 }),
    [index, start, end],
  );
  const monteCarlo = useApi(
    () =>
      api().monteCarlo({
        index_name: index,
        start,
        end,
        lookback,
        n_simulations: 5000,
        horizon_days: 252,
        method: "block",
      }),
    [index, start, end, lookback],
  );
  const stress = useApi(() => api().stress({}), []);

  const worst = varReport.data?.entries.at(-1);

  return (
    <>
      <SidebarPortal>
        <Control label="Index">
          {(id) => (
            <Input
              id={id}
              value={index}
              onChange={(event) => setIndex(event.target.value)}
            />
          )}
        </Control>
        <Control label="Start">
          {(id) => (
            <Input
              id={id}
              type="date"
              value={start}
              onChange={(event) => setStart(event.target.value)}
            />
          )}
        </Control>
        <Control label="End" hint="2024-01-01 onward is locked.">
          {(id) => (
            <Input
              id={id}
              type="date"
              value={end}
              onChange={(event) => setEnd(event.target.value)}
            />
          )}
        </Control>
        <Control label="Lookback" hint="Observations in the estimate.">
          {(id) => (
            <Input
              id={id}
              type="number"
              min={30}
              max={5000}
              value={lookback}
              onChange={(event) => setLookback(Number(event.target.value))}
            />
          )}
        </Control>
      </SidebarPortal>

      <PageHeader
        title="Risk"
        description="Equal-weighted portfolio over the point-in-time universe. Stress windows replay real ingested history — a window with no data reports a coverage gap rather than an invented number."
      />

      <MetricRow className="mb-2">
        {(varReport.data?.entries ?? []).slice(0, 3).map((entry) => (
          <MetricCard
            key={entry.method}
            label={`VaR ${entry.confidence.formatted} · ${entry.method}`}
            value={entry.var}
            benchmark={`ES ${entry.expected_shortfall.formatted}`}
          />
        ))}
        <MetricCard
          label="Tail ratio"
          value={worst?.tail_ratio ?? { value: null, formatted: "—" }}
          benchmark="ES / VaR; above ~1.15 the tail is fat"
        />
        <MetricCard
          label="Mean beta"
          value={beta.data?.mean_beta ?? { value: null, formatted: "—" }}
          benchmark={
            beta.data
              ? `range ${beta.data.min_beta.formatted}–${beta.data.max_beta.formatted}`
              : ""
          }
        />
        <MetricCard
          label="Mean correlation"
          value={correlation.data?.mean_correlation ?? { value: null, formatted: "—" }}
          benchmark="pairwise, off-diagonal"
        />
      </MetricRow>

      <Card className="mb-2">
        <CardHeader>
          <div className="flex flex-col gap-1">
            <CardTitle>Value at risk and expected shortfall</CardTitle>
            <p className="max-w-prose text-sm text-text-secondary">
              {varReport.data?.interpretation ??
                "Three estimators. The point of the table is where they disagree."}
            </p>
          </div>
        </CardHeader>
        <CardContent>
          {varReport.loading ? (
            <SkeletonRows rows={9} cols={8} />
          ) : varReport.error ? (
            <ErrorState
              message={varReport.error}
              onRetry={varReport.refetch}
              height={280}
            />
          ) : !varReport.data || varReport.data.entries.length === 0 ? (
            <EmptyState
              message="No VaR estimate available."
              hint="Ingest price history for this index and window."
              height={280}
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Method</TableHead>
                  <TableHead numeric>Confidence</TableHead>
                  <TableHead numeric>VaR</TableHead>
                  <TableHead numeric>ES</TableHead>
                  <TableHead numeric>Tail ratio</TableHead>
                  <TableHead numeric>Volatility</TableHead>
                  <TableHead numeric>Skew</TableHead>
                  <TableHead numeric>Excess kurtosis</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {varReport.data.entries.map((entry) => (
                  <TableRow key={`${entry.method}-${entry.confidence.formatted}`}>
                    <TableCell className="text-text-primary">{entry.method}</TableCell>
                    <TableCell numeric>{entry.confidence.formatted}</TableCell>
                    <TableCell numeric>{entry.var.formatted}</TableCell>
                    <TableCell numeric>{entry.expected_shortfall.formatted}</TableCell>
                    <TableCell numeric>{entry.tail_ratio.formatted}</TableCell>
                    <TableCell numeric>{entry.volatility.formatted}</TableCell>
                    <TableCell numeric>{entry.skew.formatted}</TableCell>
                    <TableCell numeric>{entry.excess_kurtosis.formatted}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-2 lg:grid-cols-2">
        <RollingBetaChart
          data={beta.data}
          loading={beta.loading}
          error={beta.error}
          onRetry={beta.refetch}
        />
        <CorrelationMatrix
          data={correlation.data}
          loading={correlation.loading}
          error={correlation.error}
          onRetry={correlation.refetch}
        />
      </div>

      <MonteCarloFan
        data={monteCarlo.data}
        loading={monteCarlo.loading}
        error={monteCarlo.error}
        onRetry={monteCarlo.refetch}
      />

      <Card className="mt-2">
        <CardHeader>
          <div className="flex flex-col gap-1">
            <CardTitle>Stress windows</CardTitle>
            <p className="text-sm text-text-secondary">
              Replayed through real ingested history. Windows with no data are
              listed as skipped, never replaced with an estimate.
            </p>
          </div>
        </CardHeader>
        <CardContent>
          {stress.loading ? (
            <SkeletonRows rows={8} cols={7} />
          ) : stress.error ? (
            <ErrorState message={stress.error} onRetry={stress.refetch} height={320} />
          ) : !stress.data || stress.data.results.length === 0 ? (
            <EmptyState
              message="No stress window could be replayed."
              hint={
                stress.data?.skipped.length
                  ? `${stress.data.skipped.length} window(s) have no ingested data for their dates.`
                  : "Ingest history covering at least one configured window."
              }
              height={320}
            />
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Window</TableHead>
                    <TableHead>Dates</TableHead>
                    <TableHead numeric>Return</TableHead>
                    <TableHead numeric>Max DD</TableHead>
                    <TableHead numeric>Worst day</TableHead>
                    <TableHead numeric>Volatility</TableHead>
                    <TableHead numeric>Mean corr</TableHead>
                    <TableHead>Recovered</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {stress.data.results.map((entry) => (
                    <TableRow key={entry.slug}>
                      <TableCell className="text-text-primary">{entry.name}</TableCell>
                      <TableCell className="ql-num text-text-secondary">
                        {entry.window.start_display} → {entry.window.end_display}
                      </TableCell>
                      <TableCell
                        numeric
                        className={
                          (entry.total_return.value ?? 0) < 0
                            ? "text-negative"
                            : "text-positive"
                        }
                      >
                        {entry.total_return.formatted}
                      </TableCell>
                      <TableCell numeric className="text-negative">
                        {entry.max_drawdown.formatted}
                      </TableCell>
                      <TableCell numeric className="text-negative">
                        {entry.worst_day.formatted}
                      </TableCell>
                      <TableCell numeric>{entry.volatility.formatted}</TableCell>
                      <TableCell numeric>{entry.mean_correlation.formatted}</TableCell>
                      <TableCell
                        className={
                          entry.recovered ? "text-text-secondary" : "text-warning"
                        }
                      >
                        {entry.recovered
                          ? `${entry.recovery_days.formatted}d`
                          : "not by window end"}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {stress.data.skipped.length > 0 && (
                <p className="mt-2 text-xs text-warning">
                  ⚠ Skipped for want of data:{" "}
                  {stress.data.skipped.map((gap) => gap.name).join(", ")}
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </>
  );
}

function RollingBetaChart({
  data,
  loading,
  error,
  onRetry,
}: {
  data: BetaOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  const traces = useMemo(() => {
    if (!data) return [];
    const tokens = readTokens();
    return [
      {
        type: "scatter" as const,
        mode: "lines" as const,
        x: data.points.map((point) => point.date),
        y: data.points.map((point) => point.beta.value),
        line: { color: tokens.series[0], width: 2 },
        connectgaps: false,
        hovertemplate: "%{x}<br>beta %{y:.2f}<extra></extra>",
      },
    ];
  }, [data, theme]);

  return (
    <ChartCard
      title="Rolling beta"
      caption={
        data?.interpretation ??
        "Beta against the equal-weighted index, over a rolling window."
      }
      height={300}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={!data || data.points.length === 0}
      emptyMessage="Not enough overlapping history to regress."
      table={
        data
          ? {
              columns: ["Date", "Beta", "Alpha"],
              rows: data.points
                .filter((point) => point.beta.value !== null)
                .slice(-400)
                .map((point) => [
                  point.date_display,
                  point.beta.formatted,
                  point.alpha.formatted,
                ]),
            }
          : undefined
      }
    >
      <PlotlyChart
        data={traces}
        layout={{
          xaxis: { showgrid: false, tickformat: "%Y-%m-%d" },
          yaxis: { title: { text: "beta" } },
          hovermode: "x unified",
        }}
        height={300}
        ariaLabel="Rolling beta against the index"
      />
    </ChartCard>
  );
}

/**
 * Diverging scale with a neutral grey midpoint: correlation runs from −1 to +1
 * through an uninformative zero, and a hue at the middle would imply zero
 * correlation is itself a state worth noticing.
 */
function CorrelationMatrix({
  data,
  loading,
  error,
  onRetry,
}: {
  data: CorrelationOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  const traces = useMemo(() => {
    if (!data) return [];
    const tokens = readTokens();
    return [
      {
        type: "heatmap" as const,
        z: data.matrix,
        x: data.securities,
        y: data.securities,
        colorscale: tokens.diverging.map((colour, i) => [
          i / (tokens.diverging.length - 1),
          colour,
        ]) as [number, string][],
        zmin: -1,
        zmax: 1,
        zmid: 0,
        hovertemplate: "%{y} ↔ %{x}<br>ρ %{z:.2f}<extra></extra>",
        colorbar: { thickness: 8, outlinewidth: 0 },
      },
    ];
  }, [data, theme]);

  return (
    <ChartCard
      title="Correlation matrix"
      caption="Pairwise return correlation. A uniformly high matrix means diversification is thinner than the position count suggests."
      height={300}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={!data || data.securities.length === 0}
      emptyMessage="Need at least two securities with complete history."
      actions={
        data?.truncated ? (
          <span className="text-xs text-warning">
            ⚠ first {data.securities.length}
          </span>
        ) : null
      }
      table={
        data
          ? {
              columns: ["Security", ...data.securities.slice(0, 12)],
              rows: data.securities.map((name, r) => [
                name,
                ...(data.matrix[r] ?? []).slice(0, 12).map((v) => v.toFixed(2)),
              ]),
            }
          : undefined
      }
    >
      <PlotlyChart
        data={traces}
        layout={{
          xaxis: { showgrid: false, type: "category", automargin: true },
          yaxis: { showgrid: false, type: "category", automargin: true },
        }}
        height={300}
        ariaLabel="Pairwise return correlation matrix"
      />
    </ChartCard>
  );
}

/**
 * The fan chart. Each band is a per-day quantile *across* paths, not a path any
 * one simulation traced — the caption says so, because a fan chart read as five
 * scenarios is a fan chart read wrongly.
 */
function MonteCarloFan({
  data,
  loading,
  error,
  onRetry,
}: {
  data: MonteCarloOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  const traces = useMemo(() => {
    if (!data) return [];
    const tokens = readTokens();
    const days = data.bands[0]?.values.map((_, i) => i + 1) ?? [];

    return data.bands.map((band, i) => ({
      type: "scatter" as const,
      mode: "lines" as const,
      x: days,
      y: band.values,
      name: `p${band.percentile}`,
      line: {
        color: tokens.sequential[Math.min(i, tokens.sequential.length - 1)],
        width: band.percentile === 50 ? 2 : 1,
      },
      hovertemplate: `p${band.percentile}<br>day %{x}<br>%{y:.2f}×<extra></extra>`,
    }));
  }, [data, theme]);

  return (
    <ChartCard
      className="mt-2"
      title="Monte Carlo fan"
      caption={
        data?.interpretation ??
        "Bootstrapped forward paths from the sampled return distribution."
      }
      height={320}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={!data || data.bands.length === 0}
      emptyMessage="No simulation available."
      emptyHint="Needs a return history over the selected window."
      actions={
        data ? (
          <span className="text-xs text-text-secondary">
            terminal p05{" "}
            <span className="ql-num text-text-primary">
              {data.terminal_p05.formatted}
            </span>{" "}
            · p50{" "}
            <span className="ql-num text-text-primary">
              {data.terminal_p50.formatted}
            </span>{" "}
            · p95{" "}
            <span className="ql-num text-text-primary">
              {data.terminal_p95.formatted}
            </span>
          </span>
        ) : null
      }
      table={
        data
          ? {
              columns: ["Day", ...data.bands.map((band) => `p${band.percentile}`)],
              rows: (data.bands[0]?.values ?? [])
                .map((_, day) => [
                  String(day + 1),
                  ...data.bands.map(
                    (band) => band.values[day]?.toFixed(3) ?? "—",
                  ),
                ])
                .filter((_, day) => day % 5 === 0),
            }
          : undefined
      }
    >
      <PlotlyChart
        data={traces}
        layout={{
          xaxis: { title: { text: "trading days ahead" }, showgrid: false },
          yaxis: { title: { text: "equity multiple" } },
          hovermode: "x unified",
        }}
        height={320}
        ariaLabel="Monte Carlo simulated equity percentile bands"
      />
    </ChartCard>
  );
}
