"use client";

import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/chart-card";
import { PlotlyChart } from "@/components/charts/plotly-chart";
import { PriceChart } from "@/components/charts/price-chart";
import { readTokens, useTheme } from "@/components/charts/theme";
import { CostWarning } from "@/components/quantlab/cost-warning";
import { MetricCard, MetricRow } from "@/components/quantlab/metric-card";
import { PageHeader } from "@/components/quantlab/page-header";
import { SharpeDisplay } from "@/components/quantlab/sharpe-display";
import { Control, SidebarPortal } from "@/components/quantlab/sidebar-context";
import { SplitBadge } from "@/components/quantlab/split-badge";
import { EmptyState, ErrorState } from "@/components/quantlab/states";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton, SkeletonRows } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { BacktestResult, EquityPoint } from "@/lib/api/client";
import { api, describe, useApi } from "@/lib/api/use-api";

const COST_MODELS = [
  "IndianEquityCostModel",
  "GenericBpsModel",
  "ZeroCostModel",
] as const;

export default function BacktestPage() {
  const [strategy, setStrategy] = useState("");
  const [index, setIndex] = useState("NIFTY500");
  const [start, setStart] = useState("2015-01-01");
  const [end, setEnd] = useState("2021-12-31");
  const [costModel, setCostModel] = useState<string>(COST_MODELS[0]);
  const [capital, setCapital] = useState(1_000_000);
  const [maxPositions, setMaxPositions] = useState(10);
  const [hypothesis, setHypothesis] = useState("");

  const [result, setResult] = useState<BacktestResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const strategies = useApi(() => api().strategies(), []);
  const hypotheses = useApi(() => api().hypotheses(), []);

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const outcome = await api().runBacktest({
        strategy,
        index_name: index,
        start,
        end,
        cost_model: costModel,
        initial_capital: capital,
        max_positions: maxPositions,
        max_position_weight: 0.15,
        hypothesis_id: hypothesis || null,
      });
      setResult(outcome);
    } catch (cause) {
      setError(describe(cause));
      setResult(null);
    } finally {
      setRunning(false);
    }
  }

  return (
    <>
      <SidebarPortal>
        <Control label="Strategy">
          {(id) => (
            <select
              id={id}
              value={strategy}
              onChange={(event) => setStrategy(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              <option value="">Select…</option>
              {(strategies.data ?? []).map((item) => (
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

        <Control
          label="Cost model"
          hint={
            costModel === COST_MODELS[0]
              ? undefined
              : "Anything but the Indian statutory model carries a warning on the result."
          }
        >
          {(id) => (
            <select
              id={id}
              value={costModel}
              onChange={(event) => setCostModel(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              {COST_MODELS.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </Control>

        <Control label="Initial capital">
          {(id) => (
            <Input
              id={id}
              type="number"
              value={capital}
              onChange={(event) => setCapital(Number(event.target.value))}
            />
          )}
        </Control>

        <Control label="Max positions">
          {(id) => (
            <Input
              id={id}
              type="number"
              min={1}
              max={200}
              value={maxPositions}
              onChange={(event) => setMaxPositions(Number(event.target.value))}
            />
          )}
        </Control>

        <Control
          label="Hypothesis"
          hint="Leave empty and the run is logged exploratory — legitimate, but labelled."
        >
          {(id) => (
            <select
              id={id}
              value={hypothesis}
              onChange={(event) => setHypothesis(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              <option value="">Exploratory</option>
              {(hypotheses.data ?? []).map((item) => (
                <option key={item.id} value={item.id}>
                  {item.id} — {item.title}
                </option>
              ))}
            </select>
          )}
        </Control>

        <Button onClick={run} disabled={!strategy || running}>
          {running ? "Running…" : "Run backtest"}
        </Button>
      </SidebarPortal>

      <PageHeader
        title="Backtest"
        description="Event-driven, T+1 open execution, itemised Indian statutory costs. Every run writes to the research log from inside the engine — including this one, before you see its Sharpe."
        actions={
          result ? (
            <div className="flex items-center gap-2">
              <SplitBadge badge={result.badge} />
              <a
                className="text-sm text-text-secondary underline underline-offset-2 hover:text-text-primary ql-fade"
                href={api().tearsheetUrl(result.run_id)}
                target="_blank"
                rel="noreferrer"
              >
                Tearsheet PDF
              </a>
            </div>
          ) : null
        }
      />

      {error && <ErrorState message={error} className="mb-2" />}

      {running && !result && (
        <div className="mb-2 flex flex-col gap-2">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      )}

      {!running && !result && !error && (
        <EmptyState
          message="No backtest run yet."
          hint="Configure a strategy in the sidebar and press Run backtest. The result will carry its split badge, its trials count and its deflated Sharpe."
          height={200}
          className="mb-2"
        />
      )}

      {result && (
        <>
          <Card className="mb-2">
            <CardHeader>
              <div className="flex flex-col gap-1">
                <CardTitle>{result.strategy}</CardTitle>
                <span className="ql-num text-sm text-text-secondary">
                  {result.window.start_display} → {result.window.end_display} ·{" "}
                  {result.trading_days.formatted} sessions
                </span>
              </div>
              <CostWarning costs={result.costs} />
            </CardHeader>
            <CardContent>
              <SharpeDisplay report={result.sharpe} size="xl" showBadge={false} />
            </CardContent>
          </Card>

          <MetricRow className="mb-2">
            <MetricCard
              label="Total return"
              value={result.total_return}
              benchmark="net of costs"
              signed
            />
            <MetricCard
              label="Annual return"
              value={result.annual_return}
              signed
            />
            <MetricCard
              label="Annual volatility"
              value={result.annual_volatility}
            />
            <MetricCard
              label="Max drawdown"
              value={result.drawdown.max_drawdown}
              benchmark={
                result.drawdown.recovered
                  ? `recovered in ${result.drawdown.recovery_days.formatted}d`
                  : "never recovered"
              }
              signed
            />
            <MetricCard
              label="Sortino"
              value={result.sortino}
              benchmark="downside deviation only"
            />
            <MetricCard
              label="Calmar"
              value={result.calmar}
              benchmark="return / max drawdown"
            />
            <MetricCard label="Trades" value={result.trades} />
            <MetricCard
              label="Win rate"
              value={result.win_rate}
              benchmark={`profit factor ${result.profit_factor.formatted}`}
            />
            <MetricCard
              label="Turnover"
              value={result.turnover}
              benchmark="annualised"
            />
            <MetricCard
              label="Total costs"
              value={result.total_costs}
              benchmark={`drag ${result.cost_drag.formatted}`}
            />
            <MetricCard
              label="Final equity"
              value={result.final_equity}
              benchmark={`from ${result.initial_capital.formatted}`}
            />
            {/*
             * The benchmark's own ratio is a reference, not a result: it is not
             * deflated because it was not selected over trials, and the API says
             * so rather than leaving the omission to be noticed.
             */}
            <MetricCard
              label="Benchmark Sharpe"
              value={result.benchmark_sharpe.display}
              benchmark={result.benchmark_sharpe.basis}
              muted
            />
          </MetricRow>

          <EquityCurveCard curve={result.equity_curve} />

          <div className="mt-2 grid gap-2 lg:grid-cols-2">
            <MonthlyReturnsCard curve={result.equity_curve} />
            <CostAttributionCard result={result} />
          </div>

          <TradeLogCard result={result} />
        </>
      )}
    </>
  );
}

/** Equity with drawdown shaded beneath it, on lightweight-charts. */
function EquityCurveCard({ curve }: { curve: EquityPoint[] }) {
  const series = useMemo(
    () => curve.map((point) => ({ date: point.date, value: point.equity })),
    [curve],
  );
  const drawdown = useMemo(
    () => curve.map((point) => ({ date: point.date, value: point.drawdown })),
    [curve],
  );

  return (
    <ChartCard
      title="Equity curve"
      caption="Drawdown is shaded beneath, filled to the zero line. The two share a time axis so a drawdown is read against the equity that produced it."
      height={360}
      empty={curve.length === 0}
      emptyMessage="The run produced no equity curve."
      table={{
        columns: ["Date", "Equity", "Drawdown"],
        rows: curve
          .slice(-400)
          .map((point) => [
            point.date,
            point.equity.toFixed(2),
            point.drawdown.toFixed(4),
          ]),
      }}
    >
      <PriceChart
        series={series}
        unit="currency"
        drawdown={drawdown}
        height={360}
        ariaLabel="Strategy equity curve with drawdown"
      />
    </ChartCard>
  );
}

/**
 * Monthly returns from the equity curve.
 *
 * Derived in the browser rather than added to the API: the equity curve is
 * already on the page and month-end compounding is not a research calculation
 * whose definition could drift.
 */
function MonthlyReturnsCard({ curve }: { curve: EquityPoint[] }) {
  const theme = useTheme();

  const grid = useMemo(() => {
    if (curve.length === 0) return null;

    const monthEnd = new Map<string, number>();
    for (const point of curve) {
      monthEnd.set(point.date.slice(0, 7), point.equity);
    }
    const months = [...monthEnd.keys()].sort();
    const years = [...new Set(months.map((m) => m.slice(0, 4)))].sort();

    // z[year][month] = return over that month.
    const z: (number | null)[][] = years.map(() => Array(12).fill(null));
    let previous: number | null = null;
    for (const month of months) {
      const equity = monthEnd.get(month);
      if (equity === undefined) continue;
      if (previous !== null && previous !== 0) {
        const yearIndex = years.indexOf(month.slice(0, 4));
        const monthIndex = Number(month.slice(5, 7)) - 1;
        const row = z[yearIndex];
        if (row) row[monthIndex] = equity / previous - 1;
      }
      previous = equity;
    }
    return { years, z };
  }, [curve]);

  const data = useMemo(() => {
    if (!grid) return [];
    const tokens = readTokens();
    return [
      {
        type: "heatmap" as const,
        z: grid.z,
        x: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
        y: grid.years,
        // Diverging with a neutral grey midpoint: zero return is the pole
        // between gain and loss, and a hue there would imply a third state.
        colorscale: tokens.diverging.map((colour, i) => [
          i / (tokens.diverging.length - 1),
          colour,
        ]) as [number, string][],
        zmid: 0,
        hoverongaps: false,
        hovertemplate: "%{y} %{x}<br>%{z:+.1%}<extra></extra>",
        colorbar: { thickness: 8, outlinewidth: 0, tickformat: "+,.0%" },
      },
    ];
  }, [grid, theme]);

  return (
    <ChartCard
      title="Monthly returns"
      caption="Compounded month over month from the equity curve."
      height={280}
      empty={!grid}
      emptyMessage="No equity curve to derive months from."
      table={
        grid
          ? {
              columns: ["Year", ...Array.from({ length: 12 }, (_, i) => String(i + 1))],
              rows: grid.years.map((year, y) => [
                year,
                ...(grid.z[y] ?? []).map((value) =>
                  value === null || value === undefined
                    ? "—"
                    : `${(value * 100).toFixed(1)}%`,
                ),
              ]),
            }
          : undefined
      }
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { showgrid: false, type: "category" },
          yaxis: { showgrid: false, type: "category", automargin: true },
        }}
        height={280}
        ariaLabel="Monthly returns heatmap"
      />
    </ChartCard>
  );
}

/** Where the money actually went. Itemised, not a single "costs" number. */
function CostAttributionCard({ result }: { result: BacktestResult }) {
  const theme = useTheme();
  const entries = useMemo(
    () =>
      Object.entries(result.cost_attribution).sort(
        ([, a], [, b]) => (b.value ?? 0) - (a.value ?? 0),
      ),
    [result],
  );

  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "bar" as const,
        orientation: "h" as const,
        y: entries.map(([name]) => name.replace(/_/g, " ")),
        x: entries.map(([, value]) => value.value),
        // Unsigned magnitudes, one series: slot 1.
        marker: { color: tokens.series[0] },
        hovertemplate: "%{y}<br>%{x:,.0f}<extra></extra>",
      },
    ];
  }, [entries, theme]);

  return (
    <ChartCard
      title="Cost attribution"
      caption="STT, stamp duty, GST, exchange and brokerage charges, itemised. A strategy that dies under realistic costs should be visible dying here."
      height={280}
      empty={entries.length === 0}
      emptyMessage="No costs recorded — this was a zero-cost run."
      actions={<CostWarning costs={result.costs} />}
      table={{
        columns: ["Component", "Amount"],
        rows: entries.map(([name, value]) => [
          name.replace(/_/g, " "),
          value.formatted,
        ]),
      }}
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { title: { text: "₹" }, tickformat: ",.0f" },
          yaxis: { showgrid: false, automargin: true },
          margin: { l: 120, r: 16, t: 32, b: 40 },
        }}
        height={280}
        ariaLabel="Transaction cost attribution by component"
      />
    </ChartCard>
  );
}

function TradeLogCard({ result }: { result: BacktestResult }) {
  return (
    <Card className="mt-2">
      <CardHeader>
        <div className="flex flex-col gap-1">
          <CardTitle>Trade log</CardTitle>
          <p className="text-sm text-text-secondary">
            Closed round trips, most recent first. Costs are charged across both
            legs and already deducted from P&amp;L.
          </p>
        </div>
        <span className="ql-num text-xs text-text-muted">
          {result.trade_log.length} of {result.trades.formatted}
          {result.trade_log_truncated && " (truncated)"}
        </span>
      </CardHeader>
      <CardContent>
        {result.trade_log.length === 0 ? (
          <EmptyState
            message="No closed trades."
            hint="The strategy either never entered, or never exited before the window ended."
            height={320}
          />
        ) : (
          <div className="max-h-screen overflow-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Security</TableHead>
                  <TableHead>Entry</TableHead>
                  <TableHead>Exit</TableHead>
                  <TableHead numeric>Days</TableHead>
                  <TableHead numeric>Qty</TableHead>
                  <TableHead numeric>Entry ₹</TableHead>
                  <TableHead numeric>Exit ₹</TableHead>
                  <TableHead numeric>Costs</TableHead>
                  <TableHead numeric>P&amp;L</TableHead>
                  <TableHead numeric>Return</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {result.trade_log.map((trade, index) => (
                  <TableRow key={`${trade.security_id}-${trade.entry_date}-${index}`}>
                    <TableCell className="ql-num text-text-primary">
                      {trade.security_id}
                    </TableCell>
                    <TableCell className="ql-num text-text-secondary">
                      {trade.entry_display}
                    </TableCell>
                    <TableCell className="ql-num text-text-secondary">
                      {trade.exit_display}
                    </TableCell>
                    <TableCell numeric>{trade.holding_days.formatted}</TableCell>
                    <TableCell numeric>{trade.quantity.formatted}</TableCell>
                    <TableCell numeric>{trade.entry_price.formatted}</TableCell>
                    <TableCell numeric>{trade.exit_price.formatted}</TableCell>
                    <TableCell numeric className="text-text-muted">
                      {trade.costs.formatted}
                    </TableCell>
                    <TableCell
                      numeric
                      className={
                        (trade.pnl.value ?? 0) < 0 ? "text-negative" : "text-positive"
                      }
                    >
                      {trade.pnl.formatted}
                    </TableCell>
                    <TableCell
                      numeric
                      className={
                        (trade.return_pct.value ?? 0) < 0
                          ? "text-negative"
                          : "text-positive"
                      }
                    >
                      {trade.return_pct.formatted}
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
