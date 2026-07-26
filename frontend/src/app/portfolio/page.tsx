"use client";

import { useEffect, useMemo, useState } from "react";

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
import { SkeletonRows } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { OptimiseOut } from "@/lib/api/client";
import { api, useApi } from "@/lib/api/use-api";

const COVARIANCE = ["ledoit_wolf", "sample", "oas"] as const;

export default function PortfolioPage() {
  const [index, setIndex] = useState("NIFTY500");
  // Null until the optimiser list arrives. A hardcoded default is how this page
  // shipped an error on first load: the guessed name `max_sharpe` is spelled
  // `maximum_sharpe` server-side, so the very first request 422'd. The API owns
  // the vocabulary; the UI does not get to assume it.
  const [method, setMethod] = useState<string | null>(null);
  const [covariance, setCovariance] = useState<string>(COVARIANCE[0]);
  const [lookback, setLookback] = useState(252);
  const [maxWeight, setMaxWeight] = useState(0.15);
  const [longOnly, setLongOnly] = useState(true);
  const [submitted, setSubmitted] = useState(1);

  const methods = useApi(() => api().portfolioMethods(), []);

  // Adopt the first method the server offers, once.
  useEffect(() => {
    if (method === null && methods.data?.length) {
      setMethod(methods.data[0] ?? null);
    }
  }, [method, methods.data]);

  const request = {
    index_name: index,
    method: method ?? "",
    covariance,
    lookback,
    max_weight: maxWeight,
    long_only: longOnly,
  };

  const result = useApi(() => api().optimise(request), [submitted, method], {
    enabled: method !== null,
  });

  /**
   * The frontier is built by asking the optimiser for several methods over the
   * same window: each is one achievable (risk, return) point. It is not a
   * synthetic curve interpolated between two numbers, which would draw
   * portfolios nobody can actually hold.
   */
  const frontier = useApi(
    async () => {
      const names = methods.data ?? [];
      const points = await Promise.all(
        names.map(async (name) => {
          try {
            const out = await api().optimise({ ...request, method: name });
            return {
              method: name,
              risk: out.volatility.value,
              ret: out.expected_return.value,
            };
          } catch {
            // A method that cannot solve on this window is omitted, not drawn
            // at the origin.
            return null;
          }
        }),
      );
      return points.filter((point): point is NonNullable<typeof point> =>
        Boolean(point && point.risk !== null && point.ret !== null),
      );
    },
    [submitted, methods.data],
    { enabled: Boolean(methods.data?.length) && method !== null },
  );

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

        <Control label="Method">
          {(id) => (
            <select
              id={id}
              value={method ?? ""}
              onChange={(event) => setMethod(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              {(methods.data ?? []).map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </Control>

        <Control label="Covariance">
          {(id) => (
            <select
              id={id}
              value={covariance}
              onChange={(event) => setCovariance(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              {COVARIANCE.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </Control>

        <Control label="Lookback">
          {(id) => (
            <Input
              id={id}
              type="number"
              min={30}
              value={lookback}
              onChange={(event) => setLookback(Number(event.target.value))}
            />
          )}
        </Control>

        <Control label="Max weight">
          {(id) => (
            <Input
              id={id}
              type="number"
              step="0.01"
              min={0.01}
              max={1}
              value={maxWeight}
              onChange={(event) => setMaxWeight(Number(event.target.value))}
            />
          )}
        </Control>

        <Control label="Long only">
          {(id) => (
            <input
              id={id}
              type="checkbox"
              checked={longOnly}
              onChange={(event) => setLongOnly(event.target.checked)}
              className="h-3 w-3 accent-border-strong"
            />
          )}
        </Control>

        <Button onClick={() => setSubmitted((n) => n + 1)}>Optimise</Button>
      </SidebarPortal>

      <PageHeader
        title="Portfolio"
        description="Mean-variance optimisation over the point-in-time universe. Black-Litterman is a deliberate scope cut, not an omission — see CLAUDE.md §4."
      />

      <MetricRow className="mb-2">
        <MetricCard
          label="Expected return"
          value={result.data?.expected_return ?? { value: null, formatted: "—" }}
          benchmark="annualised, in-sample"
          signed
        />
        <MetricCard
          label="Volatility"
          value={result.data?.volatility ?? { value: null, formatted: "—" }}
          benchmark="annualised"
        />
        <MetricCard
          label="Ex-ante Sharpe"
          value={
            result.data?.ex_ante_sharpe.display ?? { value: null, formatted: "—" }
          }
          benchmark={result.data?.ex_ante_sharpe.basis ?? ""}
          muted
        />
        <MetricCard
          label="Assets"
          value={result.data?.n_assets ?? { value: null, formatted: "—" }}
          benchmark={`${result.data?.effective_positions.formatted ?? "—"} effective`}
        />
        <MetricCard
          label="Turnover"
          value={result.data?.turnover ?? { value: null, formatted: "—" }}
          benchmark="against previous weights"
        />
        <MetricCard
          label="Condition number"
          value={result.data?.covariance.condition_number ?? { value: null, formatted: "—" }}
          benchmark={result.data?.covariance.method ?? "covariance estimator"}
        />
      </MetricRow>

      {result.error && (
        <ErrorState message={result.error} onRetry={result.refetch} className="mb-2" />
      )}

      <div className="grid gap-2 lg:grid-cols-2">
        <EfficientFrontier
          points={frontier.data ?? []}
          selected={method ?? ""}
          loading={frontier.loading}
          error={frontier.error}
          onRetry={frontier.refetch}
        />
        <WeightAllocation
          result={result.data}
          loading={result.loading}
          error={result.error}
          onRetry={result.refetch}
        />
      </div>

      <RebalanceTable
        result={result.data}
        loading={result.loading}
        error={result.error}
        onRetry={result.refetch}
      />
    </>
  );
}

function EfficientFrontier({
  points,
  selected,
  loading,
  error,
  onRetry,
}: {
  points: { method: string; risk: number | null; ret: number | null }[];
  selected: string;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  const traces = useMemo(() => {
    const tokens = readTokens();
    const others = points.filter((point) => point.method !== selected);
    const chosen = points.filter((point) => point.method === selected);

    return [
      {
        type: "scatter" as const,
        mode: "markers" as const,
        x: others.map((point) => point.risk),
        y: others.map((point) => point.ret),
        text: others.map((point) => point.method),
        // Markers only. Direct labels are the default (§6) but they are
        // "impossible" here in the sense §6 allows for: on a short window
        // several optimisers land within a pixel of each other and their labels
        // overlap into one unreadable blob. The selected point keeps its label,
        // hover names any point, and the table view carries all of them.
        marker: { size: 8, color: tokens.series[0] },
        hovertemplate: "%{text}<br>risk %{x:.2%}<br>return %{y:.2%}<extra></extra>",
      },
      {
        type: "scatter" as const,
        mode: "text+markers" as const,
        x: chosen.map((point) => point.risk),
        y: chosen.map((point) => point.ret),
        text: chosen.map((point) => point.method),
        textposition: "top center" as const,
        textfont: { size: 11, color: tokens.textPrimary },
        // The selected portfolio is marked by size and a strong outline, not by
        // a second hue competing with the others.
        marker: {
          size: 14,
          color: tokens.series[0],
          line: { width: 2, color: tokens.borderStrong },
        },
        hovertemplate: "%{text} (selected)<br>risk %{x:.2%}<br>return %{y:.2%}<extra></extra>",
      },
    ];
  }, [points, selected, theme]);

  return (
    <ChartCard
      title="Efficient frontier"
      caption="One point per optimiser over the same window — each is a portfolio you can actually hold, not an interpolated curve. The selected method is ringed."
      height={320}
      loading={loading}
      error={error}
      onRetry={onRetry}
      // Fewer than two points is not a frontier: Plotly auto-pads a lone dot to
      // a ±50% axis range, which looks like a broken chart rather than a thin
      // result. Say what happened instead.
      empty={points.length < 2}
      emptyMessage={
        points.length === 1
          ? "Only one optimiser produced a solution — not enough for a frontier."
          : "No optimiser produced a solution on this window."
      }
      emptyHint="Most optimisers need a positive expected return to solve. Try a longer lookback or a larger max weight."
      table={{
        columns: ["Method", "Risk", "Return"],
        rows: points.map((point) => [
          point.method,
          point.risk === null ? "—" : `${(point.risk * 100).toFixed(2)}%`,
          point.ret === null ? "—" : `${(point.ret * 100).toFixed(2)}%`,
        ]),
      }}
    >
      <PlotlyChart
        data={traces}
        layout={{
          xaxis: { title: { text: "annualised volatility" }, tickformat: ",.1%" },
          yaxis: { title: { text: "expected return" }, tickformat: ",.1%" },
        }}
        height={320}
        ariaLabel="Efficient frontier of optimiser solutions"
      />
    </ChartCard>
  );
}

function WeightAllocation({
  result,
  loading,
  error,
  onRetry,
}: {
  result: OptimiseOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();

  const holdings = useMemo(
    () =>
      [...(result?.holdings ?? [])]
        .sort((a, b) => (b.weight.value ?? 0) - (a.weight.value ?? 0))
        .slice(0, 25),
    [result],
  );

  const traces = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "bar" as const,
        orientation: "h" as const,
        y: holdings.map((holding) => holding.security_id),
        x: holdings.map((holding) => holding.weight.value),
        // Signed: a short is a different claim from a long, so the semantic
        // colours carry it.
        marker: {
          color: holdings.map((holding) =>
            (holding.weight.value ?? 0) < 0 ? tokens.negative : tokens.series[0],
          ),
        },
        hovertemplate: "%{y}<br>%{x:.2%}<extra></extra>",
      },
    ];
  }, [holdings, theme]);

  return (
    <ChartCard
      title="Weight allocation"
      caption="Largest 25 positions. A frontier point with two real holdings is a concentrated bet wearing a portfolio's name."
      height={320}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={holdings.length === 0}
      emptyMessage="No holdings in the solution."
      table={{
        columns: ["Security", "Weight"],
        rows: holdings.map((holding) => [
          holding.security_id,
          holding.weight.formatted,
        ]),
      }}
    >
      <PlotlyChart
        data={traces}
        layout={{
          xaxis: { title: { text: "weight" }, tickformat: ",.1%" },
          yaxis: { showgrid: false, automargin: true, autorange: "reversed" },
          margin: { l: 110, r: 16, t: 32, b: 40 },
        }}
        height={320}
        ariaLabel="Optimised portfolio weights by security"
      />
    </ChartCard>
  );
}

/**
 * Before and after. `previous_weights` is not sent by this page, so "before" is
 * the equal-weight portfolio the optimiser departed from — stated in the caption
 * rather than left for the reader to assume it was their live book.
 */
function RebalanceTable({
  result,
  loading,
  error,
  onRetry,
}: {
  result: OptimiseOut | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const rows = useMemo(() => {
    if (!result) return [];
    const equal = result.holdings.length > 0 ? 1 / result.holdings.length : 0;
    return [...result.holdings]
      .sort((a, b) => (b.weight.value ?? 0) - (a.weight.value ?? 0))
      .map((holding) => ({
        security_id: holding.security_id,
        before: equal,
        after: holding.weight.value ?? 0,
        change: (holding.weight.value ?? 0) - equal,
        formatted: holding.weight.formatted,
      }));
  }, [result]);

  return (
    <Card className="mt-2">
      <CardHeader>
        <div className="flex flex-col gap-1">
          <CardTitle>Before and after rebalance</CardTitle>
          <p className="max-w-prose text-sm text-text-secondary">
            &ldquo;Before&rdquo; is the equal-weight portfolio over the same
            names, which is what the optimiser is being judged against here — not
            a live book.
          </p>
        </div>
        <span className="ql-num text-xs text-text-muted">
          turnover {result?.turnover.formatted ?? "—"}
        </span>
      </CardHeader>
      <CardContent>
        {loading ? (
          <SkeletonRows rows={20} cols={4} />
        ) : error ? (
          <ErrorState message={error} onRetry={onRetry} height={400} />
        ) : rows.length === 0 ? (
          <EmptyState
            message="No solution to compare."
            hint="Press Optimise in the sidebar."
            height={400}
          />
        ) : (
          <div className="max-h-screen overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Security</TableHead>
                  <TableHead numeric>Before (equal)</TableHead>
                  <TableHead numeric>After</TableHead>
                  <TableHead numeric>Change</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row) => (
                  <TableRow key={row.security_id}>
                    <TableCell className="ql-num text-text-primary">
                      {row.security_id}
                    </TableCell>
                    <TableCell numeric className="text-text-muted">
                      {(row.before * 100).toFixed(1)}%
                    </TableCell>
                    <TableCell numeric>{row.formatted}</TableCell>
                    <TableCell
                      numeric
                      className={
                        row.change < 0 ? "text-negative" : "text-positive"
                      }
                    >
                      {row.change >= 0 ? "+" : "−"}
                      {Math.abs(row.change * 100).toFixed(1)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
