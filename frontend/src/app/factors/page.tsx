"use client";

import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/chart-card";
import { PlotlyChart } from "@/components/charts/plotly-chart";
import { readTokens, useTheme } from "@/components/charts/theme";
import { MetricCard, MetricRow } from "@/components/quantlab/metric-card";
import { PageHeader } from "@/components/quantlab/page-header";
import { Control, SidebarPortal } from "@/components/quantlab/sidebar-context";
import { EmptyState, ErrorState } from "@/components/quantlab/states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { EvaluationOut } from "@/lib/api/client";
import { api, useApi } from "@/lib/api/use-api";

export default function FactorResearchPage() {
  const [factor, setFactor] = useState<string>("");
  const [index, setIndex] = useState("NIFTY500");
  const [start, setStart] = useState("2015-01-01");
  const [end, setEnd] = useState("2021-12-31");
  const [quantiles, setQuantiles] = useState(5);
  const [submitted, setSubmitted] = useState(0);

  const factors = useApi(() => api().factors(), []);

  const evaluation = useApi(
    () =>
      api().evaluateFactor({
        factor,
        index_name: index,
        start,
        end,
        n_quantiles: quantiles,
      }),
    [submitted],
    { enabled: submitted > 0 && factor.length > 0 },
  );

  const result = evaluation.data;

  return (
    <>
      <SidebarPortal>
        <Control label="Factor">
          {(id) => (
            <select
              id={id}
              value={factor}
              onChange={(event) => setFactor(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              <option value="">Select…</option>
              {(factors.data ?? []).map((item) => (
                <option key={item.name} value={item.name}>
                  {item.name}
                </option>
              ))}
            </select>
          )}
        </Control>

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

        <Control label="Quantiles">
          {(id) => (
            <Input
              id={id}
              type="number"
              min={2}
              max={20}
              value={quantiles}
              onChange={(event) => setQuantiles(Number(event.target.value))}
            />
          )}
        </Control>

        <Button onClick={() => setSubmitted((n) => n + 1)} disabled={!factor}>
          Evaluate
        </Button>
      </SidebarPortal>

      <PageHeader
        title="Factor Research"
        description="Alpha evaluation runs before and separately from strategy backtesting. A factor that fails IC decay never reaches the strategy builder — so the gate verdict is reported here, not left for the reader to apply."
      />

      {evaluation.error && (
        <ErrorState
          message={evaluation.error}
          onRetry={evaluation.refetch}
          className="mb-2"
        />
      )}

      {result && <EligibilityBanner result={result} />}

      <MetricRow className="mb-2">
        <MetricCard
          label="Mean IC"
          value={
            result?.pearson_ic[0]?.mean_ic ?? { value: null, formatted: "—" }
          }
          benchmark={`primary horizon ${result?.primary_horizon ?? "—"}d`}
          signed
        />
        <MetricCard
          label="IC IR"
          value={result?.pearson_ic[0]?.ic_ir ?? { value: null, formatted: "—" }}
          benchmark="mean IC / std IC"
          signed
        />
        <MetricCard
          label="t-statistic"
          value={result?.pearson_ic[0]?.t_stat ?? { value: null, formatted: "—" }}
          benchmark="Newey-West, overlapping returns"
          signed
        />
        <MetricCard
          label="p-value"
          value={result?.pearson_ic[0]?.p_value ?? { value: null, formatted: "—" }}
        />
        <MetricCard
          label="Long-short spread"
          value={result?.long_short_spread ?? { value: null, formatted: "—" }}
          benchmark="top minus bottom quantile"
          signed
        />
        <MetricCard
          label="Turnover"
          value={result?.turnover ?? { value: null, formatted: "—" }}
          benchmark="a costly factor is not an alpha"
        />
      </MetricRow>

      <div className="grid gap-2 lg:grid-cols-2">
        <ICSeriesChart
          result={result}
          loading={evaluation.loading}
          error={evaluation.error}
          onRetry={evaluation.refetch}
        />
        <ICDecayChart
          result={result}
          loading={evaluation.loading}
          error={evaluation.error}
          onRetry={evaluation.refetch}
        />
      </div>

      <QuantileChart
        result={result}
        loading={evaluation.loading}
        error={evaluation.error}
        onRetry={evaluation.refetch}
      />

      <Card className="mt-2">
        <CardHeader>
          <CardTitle>IC statistics by horizon</CardTitle>
          <span className="text-xs text-text-muted">
            Pearson and Spearman, side by side
          </span>
        </CardHeader>
        <CardContent>
          {!result ? (
            <EmptyState
              message="No evaluation yet."
              hint="Choose a factor in the sidebar and press Evaluate."
              height={200}
            />
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead numeric>Horizon</TableHead>
                  <TableHead>Method</TableHead>
                  <TableHead numeric>Mean IC</TableHead>
                  <TableHead numeric>Std IC</TableHead>
                  <TableHead numeric>IC IR</TableHead>
                  <TableHead numeric>t</TableHead>
                  <TableHead numeric>p</TableHead>
                  <TableHead numeric>Periods</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {[...result.pearson_ic, ...result.spearman_ic].map((stats, i) => (
                  <TableRow key={`${stats.method}-${stats.horizon}-${i}`}>
                    <TableCell numeric>{stats.horizon}</TableCell>
                    <TableCell className="text-text-secondary">
                      {stats.method}
                    </TableCell>
                    <TableCell numeric>{stats.mean_ic.formatted}</TableCell>
                    <TableCell numeric>{stats.std_ic.formatted}</TableCell>
                    <TableCell numeric>{stats.ic_ir.formatted}</TableCell>
                    <TableCell numeric>{stats.t_stat.formatted}</TableCell>
                    <TableCell numeric>{stats.p_value.formatted}</TableCell>
                    <TableCell numeric>{stats.n_periods.formatted}</TableCell>
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

/** The gate verdict, stated. An ineligible factor says why, in `warning`. */
function EligibilityBanner({ result }: { result: EvaluationOut }) {
  return (
    <div
      className={
        "mb-2 flex flex-wrap items-baseline gap-2 rounded border p-3 " +
        (result.eligible ? "border-border" : "border-warning")
      }
    >
      <span className="text-md text-text-primary">{result.factor_name}</span>
      <span
        className={
          "rounded border px-1 text-xs uppercase tracking-wide " +
          (result.eligible
            ? "border-positive text-positive"
            : "border-warning text-warning")
        }
      >
        {result.eligible ? "Eligible" : "Not eligible"}
      </span>
      <span className="ql-num text-xs text-text-muted">
        {result.window.start_display} → {result.window.end_display}
      </span>
      {!result.eligible && result.ineligibility_reason && (
        <p className="w-full text-sm text-warning">{result.ineligibility_reason}</p>
      )}
    </div>
  );
}

/** IC per period with its rolling mean. Zero is the reference, not the floor. */
function ICSeriesChart({
  result,
  loading,
  error,
  onRetry,
}: {
  result: EvaluationOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  // The API returns IC *statistics* per horizon rather than a per-date series,
  // so the honest chart here is IC by horizon with the primary horizon marked —
  // not a fabricated time series.
  const stats = result?.pearson_ic ?? [];

  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "scatter" as const,
        mode: "lines+markers" as const,
        x: stats.map((s) => s.horizon),
        y: stats.map((s) => s.mean_ic.value),
        line: { color: tokens.series[0], width: 2 },
        marker: { size: 8 },
        name: "Pearson",
        hovertemplate: "%{x}d horizon<br>mean IC %{y:.4f}<extra></extra>",
      },
      {
        type: "scatter" as const,
        mode: "lines+markers" as const,
        x: (result?.spearman_ic ?? []).map((s) => s.horizon),
        y: (result?.spearman_ic ?? []).map((s) => s.mean_ic.value),
        line: { color: tokens.series[2], width: 2, dash: "dot" as const },
        marker: { size: 8 },
        name: "Spearman",
        hovertemplate: "%{x}d horizon<br>rank IC %{y:.4f}<extra></extra>",
      },
    ];
  }, [stats, result, theme]);

  return (
    <ChartCard
      title="Information coefficient by horizon"
      caption="Pearson against Spearman. A large gap between them means the relationship is driven by a few extreme values rather than by rank."
      height={300}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={stats.length === 0}
      emptyMessage="No evaluation yet."
      emptyHint="Choose a factor in the sidebar and press Evaluate."
      table={{
        columns: ["Horizon", "Pearson IC", "Spearman IC"],
        rows: stats.map((s, i) => [
          `${s.horizon}d`,
          s.mean_ic.formatted,
          result?.spearman_ic[i]?.mean_ic.formatted ?? "—",
        ]),
      }}
      actions={
        // Two series that converge; a legend is justified here rather than
        // direct labels (§6).
        <span className="flex items-center gap-2 text-xs text-text-secondary">
          <Swatch token="--ql-series-1" /> Pearson
          <Swatch token="--ql-series-3" /> Spearman
        </span>
      }
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { title: { text: "horizon (days)" }, showgrid: false },
          yaxis: { title: { text: "mean IC" }, zeroline: true },
          hovermode: "x unified",
        }}
        height={300}
        ariaLabel="Information coefficient by forward-return horizon"
      />
    </ChartCard>
  );
}

/** IC decay: how fast the signal dies as the horizon lengthens. */
function ICDecayChart({
  result,
  loading,
  error,
  onRetry,
}: {
  result: EvaluationOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();
  const entries = useMemo(
    () =>
      Object.entries(result?.ic_decay ?? {}).sort(
        ([a], [b]) => Number(a) - Number(b),
      ),
    [result],
  );

  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "bar" as const,
        x: entries.map(([horizon]) => `${horizon}d`),
        y: entries.map(([, value]) => value.value),
        // Sequential: decay is a magnitude along one axis, so lightness carries
        // the ordering rather than five unrelated hues.
        marker: {
          color: entries.map(
            (_, i) => tokens.sequential[Math.min(i, tokens.sequential.length - 1)] ?? "",
          ),
        },
        hovertemplate: "%{x}<br>IC %{y:.4f}<extra></extra>",
      },
    ];
  }, [entries, theme]);

  return (
    <ChartCard
      title="IC decay"
      caption="How long the signal survives. A factor whose IC is gone by day 5 cannot fund a weekly rebalance after costs."
      height={300}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={entries.length === 0}
      emptyMessage="No evaluation yet."
      emptyHint="Choose a factor in the sidebar and press Evaluate."
      table={{
        columns: ["Horizon", "IC"],
        rows: entries.map(([horizon, value]) => [`${horizon}d`, value.formatted]),
      }}
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { showgrid: false },
          yaxis: { title: { text: "IC" }, zeroline: true },
        }}
        height={300}
        ariaLabel="Information coefficient decay by horizon"
      />
    </ChartCard>
  );
}

/** Quantile buckets with the long-short spread called out. */
function QuantileChart({
  result,
  loading,
  error,
  onRetry,
}: {
  result: EvaluationOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();
  const buckets = result?.quantile_returns ?? [];

  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "bar" as const,
        x: buckets.map((_, i) => `Q${i + 1}`),
        y: buckets.map((bucket) => bucket.value),
        // Signed bars carry the semantic colours: the sign is the point.
        marker: {
          color: buckets.map((bucket) =>
            (bucket.value ?? 0) < 0 ? tokens.negative : tokens.positive,
          ),
        },
        hovertemplate: "%{x}<br>%{y:.2%}<extra></extra>",
      },
    ];
  }, [buckets, theme]);

  return (
    <ChartCard
      className="mt-2"
      title="Quantile bucket returns"
      caption="Monotonicity across buckets is the evidence that the factor orders returns. A high top bucket with a jumbled middle is a story about one bucket, not a factor."
      height={300}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={buckets.length === 0}
      emptyMessage="No evaluation yet."
      emptyHint="Choose a factor in the sidebar and press Evaluate."
      actions={
        result ? (
          <span className="text-xs text-text-secondary">
            long-short spread{" "}
            <span className="ql-num text-text-primary">
              {result.long_short_spread.formatted}
            </span>
          </span>
        ) : null
      }
      table={{
        columns: ["Quantile", "Return"],
        rows: buckets.map((bucket, i) => [`Q${i + 1}`, bucket.formatted]),
      }}
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { showgrid: false },
          yaxis: { title: { text: "mean forward return" }, tickformat: ",.1%" },
        }}
        height={300}
        ariaLabel="Mean forward return by factor quantile"
      />
    </ChartCard>
  );
}

/** A colour chip for a legend. The label wears a text token, not the hue. */
function Swatch({ token }: { token: string }) {
  return (
    <span
      aria-hidden
      className="inline-block h-1 w-2 rounded"
      style={{ backgroundColor: `var(${token})` }}
    />
  );
}
