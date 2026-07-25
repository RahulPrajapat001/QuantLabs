"use client";

/**
 * The Plotly surface, for statistical charts: IC decay, quantile buckets, the
 * efficient frontier, correlation heatmaps, QQ plots. Price series and equity
 * curves go to lightweight-charts instead — see `price-chart.tsx`.
 *
 * Loaded with a dynamic `import()` so Plotly's ~3MB bundle never enters the
 * initial payload. Until it resolves the caller's skeleton stays up, which is
 * why this component renders nothing of its own while loading.
 */

import { useEffect, useRef, useState } from "react";

import { SkeletonChart } from "@/components/ui/skeleton";

import { BASE_LAYOUT, PLOT_CONFIG, plotlyTemplate, useTheme } from "./theme";

type PlotlyModule = typeof import("plotly.js-dist-min");

let plotlyPromise: Promise<PlotlyModule> | null = null;

function loadPlotly(): Promise<PlotlyModule> {
  // One module instance for the whole app, resolved once and shared.
  plotlyPromise ??= import("plotly.js-dist-min");
  return plotlyPromise;
}

export interface PlotlyChartProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  height: number;
  /** Announced to screen readers in place of the canvas. */
  ariaLabel: string;
}

export function PlotlyChart({ data, layout, height, ariaLabel }: PlotlyChartProps) {
  const node = useRef<HTMLDivElement | null>(null);
  const [ready, setReady] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const theme = useTheme();

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const Plotly = await loadPlotly();
        if (cancelled || !node.current) return;
        // `Promise.resolve` because `react()` is only documented to return a
        // promise: treating its return value as a thenable directly leaves the
        // chart drawn but permanently invisible if it ever returns undefined.
        await Promise.resolve(
          Plotly.react(
            node.current,
            data,
            {
              ...BASE_LAYOUT,
              ...layout,
              height,
              template: plotlyTemplate(theme),
            } as Partial<Plotly.Layout>,
            PLOT_CONFIG,
          ),
        );
        if (!cancelled) {
          setFailed(null);
          setReady(true);
        }
      } catch (cause) {
        // A chart that fails silently is worse than one that says so: a blank
        // card reads as "no data", which is a different claim entirely.
        if (!cancelled) setFailed(cause instanceof Error ? cause.message : String(cause));
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [data, layout, height, theme]);

  // Purge on unmount only. Purging in the redraw effect's cleanup races the
  // next redraw — the teardown is async, so it can land *after* the new chart
  // is drawn and wipe it.
  useEffect(() => {
    const element = node.current;
    return () => {
      if (element) void loadPlotly().then((Plotly) => Plotly.purge(element));
    };
  }, []);

  return (
    <div className="relative" style={{ height }}>
      {!ready && !failed && (
        <div className="absolute inset-0">
          <SkeletonChart height={height} />
        </div>
      )}
      {failed && (
        <div className="absolute inset-0 flex items-center justify-center p-4 text-center">
          <p className="text-sm text-negative">Could not draw this chart. {failed}</p>
        </div>
      )}
      <div
        ref={node}
        role="img"
        aria-label={ariaLabel}
        className="ql-fade"
        style={{ height, opacity: ready ? 1 : 0 }}
      />
    </div>
  );
}
