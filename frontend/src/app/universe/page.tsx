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
import type { IntervalOut } from "@/lib/api/client";
import { api, useApi } from "@/lib/api/use-api";

const INDICES = ["NIFTY500", "NIFTY100", "NIFTY50", "SP500"] as const;

export default function UniversePage() {
  const [index, setIndex] = useState<string>("NIFTY500");
  const [asOf, setAsOf] = useState("2021-12-31");

  const membership = useApi(() => api().universeAsOf(index, asOf), [index, asOf]);
  const survivorship = useApi(() => api().survivorship(index, asOf), [index, asOf]);
  const intervals = useApi(() => api().universeIntervals(index), [index]);

  const counts = useConstituentCounts(intervals.data);

  return (
    <>
      <SidebarPortal>
        <Control label="Index">
          {(id) => (
            <select
              id={id}
              value={index}
              onChange={(event) => setIndex(event.target.value)}
              className="h-6 w-full rounded border border-border bg-surface px-2 text-base text-text-primary"
            >
              {INDICES.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          )}
        </Control>

        <Control
          label="As of"
          hint="Membership is resolved from reconstitution filings on this date."
        >
          {(id) => (
            <Input
              id={id}
              type="date"
              value={asOf}
              onChange={(event) => setAsOf(event.target.value)}
            />
          )}
        </Control>

        {/*
         * The scrubber. A range input over the timeline gives the "drag through
         * history and watch membership change" reading that a date field cannot.
         */}
        <Control label="Scrub">
          {(id) => (
            <input
              id={id}
              type="range"
              min={0}
              max={counts.dates.length - 1 > 0 ? counts.dates.length - 1 : 0}
              value={Math.max(0, counts.dates.indexOf(asOf))}
              onChange={(event) => {
                const next = counts.dates[Number(event.target.value)];
                if (next) setAsOf(next);
              }}
              className="w-full accent-border-strong"
              disabled={counts.dates.length === 0}
            />
          )}
        </Control>
      </SidebarPortal>

      <PageHeader
        title="Universe"
        description="Point-in-time index membership, reconstructed from reconstitution history. Never today's constituent list projected backwards — that is survivorship bias, and it is quantified below."
      />

      <MetricRow className="mb-2">
        <MetricCard
          label="Constituents"
          value={membership.data?.count ?? { value: null, formatted: "—" }}
          benchmark={membership.data ? `on ${membership.data.as_of_display}` : ""}
        />
        <MetricCard
          label="Today's list"
          value={survivorship.data?.current_count ?? { value: null, formatted: "—" }}
          benchmark="what a naive backtest would use"
        />
        <MetricCard
          label="Overlap"
          value={survivorship.data?.overlap ?? { value: null, formatted: "—" }}
          benchmark="in both lists"
        />
        <MetricCard
          label="Survival rate"
          value={survivorship.data?.survival_rate ?? { value: null, formatted: "—" }}
          benchmark="of the point-in-time universe"
        />
        <MetricCard
          label="Dropped since"
          value={{
            value: survivorship.data?.dropped_since.length ?? null,
            formatted: survivorship.data
              ? String(survivorship.data.dropped_since.length)
              : "—",
          }}
          benchmark="removed after this date"
        />
        <MetricCard
          label="Survivorship-only"
          value={{
            value: survivorship.data?.survivorship_only.length ?? null,
            formatted: survivorship.data
              ? String(survivorship.data.survivorship_only.length)
              : "—",
          }}
          benchmark="in today's list but not then"
        />
      </MetricRow>

      <div className="grid gap-2 lg:grid-cols-2">
        <ConstituentCountChart
          dates={counts.dates}
          values={counts.values}
          loading={intervals.loading}
          error={intervals.error}
          onRetry={intervals.refetch}
        />

        <SurvivorshipChart
          pointInTime={survivorship.data?.point_in_time_count.value ?? null}
          current={survivorship.data?.current_count.value ?? null}
          overlap={survivorship.data?.overlap.value ?? null}
          loading={survivorship.loading}
          error={survivorship.error}
          onRetry={survivorship.refetch}
        />
      </div>

      <Card className="mt-2">
        <CardHeader>
          <div className="flex flex-col gap-1">
            <CardTitle>Constituents on {asOf}</CardTitle>
            <p className="text-sm text-text-secondary">
              Resolved by <span className="ql-num">security_id</span>, never by
              ticker — a rename does not create a second company.
            </p>
          </div>
        </CardHeader>
        <CardContent>
          {membership.loading ? (
            <SkeletonRows rows={20} cols={1} />
          ) : membership.error ? (
            <ErrorState
              message={membership.error}
              onRetry={membership.refetch}
              height={320}
            />
          ) : !membership.data || membership.data.security_ids.length === 0 ? (
            <EmptyState
              message={`No membership recorded for ${index} on ${asOf}.`}
              hint="Build it with `quantlab universe build --index <name>`."
              height={320}
            />
          ) : (
            <div className="grid max-h-screen grid-cols-2 gap-x-4 overflow-auto md:grid-cols-4 lg:grid-cols-6">
              {membership.data.security_ids.map((id) => (
                <span
                  key={id}
                  className="ql-num h-table-row border-b border-border text-sm leading-table-row text-text-secondary"
                  style={{ lineHeight: "28px" }}
                >
                  {id}
                </span>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </>
  );
}

/**
 * Constituent count over time, derived from the membership intervals.
 *
 * Computed here rather than asked for: the intervals are already loaded for the
 * scrubber, and a second endpoint returning a count per date would be a second
 * thing to keep consistent with the first.
 */
function useConstituentCounts(intervals: IntervalOut[] | null): {
  dates: string[];
  values: number[];
} {
  return useMemo(() => {
    if (!intervals || intervals.length === 0) return { dates: [], values: [] };

    // +1 on the day a security joins, -1 the day after it leaves.
    const delta = new Map<string, number>();
    for (const interval of intervals) {
      delta.set(interval.start, (delta.get(interval.start) ?? 0) + 1);
      if (interval.end) {
        delta.set(interval.end, (delta.get(interval.end) ?? 0) - 1);
      }
    }

    const dates = [...delta.keys()].sort();
    const values: number[] = [];
    let running = 0;
    for (const date of dates) {
      running += delta.get(date) ?? 0;
      values.push(running);
    }
    return { dates, values };
  }, [intervals]);
}

function ConstituentCountChart({
  dates,
  values,
  loading,
  error,
  onRetry,
}: {
  dates: string[];
  values: number[];
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();
  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "scatter" as const,
        mode: "lines" as const,
        x: dates,
        y: values,
        line: { color: tokens.series[0], width: 2, shape: "hv" as const },
        hovertemplate: "%{x}<br>%{y} constituents<extra></extra>",
      },
    ];
  }, [dates, values, theme]);

  return (
    <ChartCard
      title="Constituent count over time"
      caption="A step function: the index only changes on a reconstitution date."
      height={280}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={dates.length === 0}
      emptyMessage="No membership intervals."
      emptyHint="Run `quantlab universe build --index <name>` to ingest reconstitution history."
      table={{
        columns: ["Date", "Constituents"],
        rows: dates.map((date, i) => [date, String(values[i] ?? 0)]),
      }}
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { showgrid: false, tickformat: "%Y-%m-%d" },
          yaxis: {
            title: { text: "constituents" },
            // Explicit headroom: `rangemode: "tozero"` pins the floor but lets
            // the top land exactly on the series, which drew the step line along
            // the upper edge of the plot with no air above it.
            range: [0, Math.max(...values, 1) * 1.15],
          },
          hovermode: "x unified",
        }}
        height={280}
        ariaLabel="Index constituent count over time"
      />
    </ChartCard>
  );
}

/**
 * The bias, quantified. Three bars: what the universe actually was, what
 * today's list would have given you, and the overlap between them. The gap
 * between the first two is exactly the survivorship error a naive backtest
 * inherits.
 */
function SurvivorshipChart({
  pointInTime,
  current,
  overlap,
  loading,
  error,
  onRetry,
}: {
  pointInTime: number | null;
  current: number | null;
  overlap: number | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  const theme = useTheme();
  const empty = pointInTime === null && current === null;

  const data = useMemo(() => {
    const tokens = readTokens();
    return [
      {
        type: "bar" as const,
        x: ["Point-in-time", "Today's list", "In both"],
        y: [pointInTime ?? 0, current ?? 0, overlap ?? 0],
        // Slot 1 for all three: this is one measure at three scopes, not three
        // different series that need distinguishing by colour.
        marker: { color: tokens.series[0] },
        hovertemplate: "%{x}<br>%{y} securities<extra></extra>",
      },
    ];
  }, [pointInTime, current, overlap, theme]);

  return (
    <ChartCard
      title="Survivorship bias, quantified"
      caption="The difference between the first two bars is what a backtest on today's constituents would silently assume it had known all along."
      height={280}
      loading={loading}
      error={error}
      onRetry={onRetry}
      empty={empty}
      emptyMessage="No comparison available for this date."
      table={{
        columns: ["Scope", "Securities"],
        rows: [
          ["Point-in-time", pointInTime === null ? "—" : String(pointInTime)],
          ["Today's list", current === null ? "—" : String(current)],
          ["In both", overlap === null ? "—" : String(overlap)],
        ],
      }}
    >
      <PlotlyChart
        data={data}
        layout={{
          xaxis: { showgrid: false },
          yaxis: { title: { text: "securities" }, rangemode: "tozero" },
        }}
        height={280}
        ariaLabel="Point-in-time versus current index membership"
      />
    </ChartCard>
  );
}
