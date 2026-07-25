"use client";

/**
 * lightweight-charts, for price series and equity curves.
 *
 * This is what trading platforms actually use: it is canvas-based with its own
 * time-axis virtualisation, so a decade of daily bars pans and zooms at 60fps
 * where Plotly would re-layout the whole trace. Statistical charts stay on
 * Plotly — the split is by data shape, not by preference.
 *
 * Colours are read from the live CSS variables rather than passed in, so the
 * series obey the same tokens as everything else and re-theme with the page.
 */

import {
  ColorType,
  createChart,
  LineStyle,
  type IChartApi,
  type ISeriesApi,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef } from "react";

import { readTokens, useTheme } from "./theme";

export interface Point {
  /** ISO date, `YYYY-MM-DD`. */
  date: string;
  value: number;
}

export interface PriceChartProps {
  series: Point[];
  /** Muted, dashed, always grey. It is context, not a competitor. */
  benchmark?: Point[];
  /**
   * Shaded below zero, filled to the zero line in `negative`. The design system
   * fixes this: drawdown is the one series that carries a fill.
   */
  drawdown?: Point[];
  height: number;
  ariaLabel: string;
  /** Candles instead of a line, for raw OHLC price. */
  candles?: { date: string; open: number; high: number; low: number; close: number }[];
}

function toTime(iso: string): UTCTimestamp {
  return (Date.parse(`${iso}T00:00:00Z`) / 1000) as UTCTimestamp;
}

export function PriceChart({
  series,
  benchmark,
  drawdown,
  height,
  ariaLabel,
  candles,
}: PriceChartProps) {
  const node = useRef<HTMLDivElement | null>(null);
  const chart = useRef<IChartApi | null>(null);
  const theme = useTheme();

  useEffect(() => {
    if (!node.current) return;
    const tokens = readTokens();

    const instance = createChart(node.current, {
      height,
      autoSize: true,
      layout: {
        background: { type: ColorType.Solid, color: tokens.surface },
        textColor: tokens.textSecondary,
        fontSize: 11,
        fontFamily: '"JetBrains Mono", "SFMono-Regular", "Menlo", monospace',
      },
      grid: {
        // A subtle horizontal grid only. A vertical grid on a time axis adds
        // ink without enabling a comparison anyone makes.
        horzLines: { color: tokens.border },
        vertLines: { visible: false },
      },
      rightPriceScale: { borderColor: tokens.border },
      timeScale: { borderColor: tokens.border, timeVisible: false },
      crosshair: {
        vertLine: { color: tokens.borderStrong, labelBackgroundColor: tokens.surfaceRaised },
        horzLine: { color: tokens.borderStrong, labelBackgroundColor: tokens.surfaceRaised },
      },
    });
    chart.current = instance;

    if (candles?.length) {
      const candleSeries = instance.addCandlestickSeries({
        upColor: tokens.positive,
        downColor: tokens.negative,
        borderVisible: false,
        wickUpColor: tokens.positive,
        wickDownColor: tokens.negative,
      });
      candleSeries.setData(
        candles.map((c) => ({
          time: toTime(c.date) as Time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        })),
      );
    } else if (series.length) {
      const line = instance.addLineSeries({
        color: tokens.series[0],
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: true,
      });
      line.setData(series.map((p) => ({ time: toTime(p.date) as Time, value: p.value })));
    }

    if (benchmark?.length) {
      const line = instance.addLineSeries({
        color: tokens.benchmark,
        lineWidth: 2,
        lineStyle: LineStyle.Dashed,
        priceLineVisible: false,
      });
      line.setData(
        benchmark.map((p) => ({ time: toTime(p.date) as Time, value: p.value })),
      );
    }

    if (drawdown?.length) {
      // Its own scale, running from the trough to 0, pinned to the lower third
      // so it reads as a companion to the equity curve rather than competing
      // with it for vertical space.
      const area: ISeriesApi<"Area"> = instance.addAreaSeries({
        priceScaleId: "drawdown",
        lineColor: tokens.negative,
        topColor: `${tokens.negative}00`,
        bottomColor: `${tokens.negative}59`,
        lineWidth: 1,
        priceLineVisible: false,
        lastValueVisible: false,
      });
      area.setData(
        drawdown.map((p) => ({ time: toTime(p.date) as Time, value: p.value })),
      );
      instance.priceScale("drawdown").applyOptions({
        scaleMargins: { top: 0.7, bottom: 0 },
        visible: false,
      });
    }

    instance.timeScale().fitContent();

    return () => {
      instance.remove();
      chart.current = null;
    };
  }, [series, benchmark, drawdown, candles, height, theme]);

  return (
    <div
      ref={node}
      role="img"
      aria-label={ariaLabel}
      style={{ height }}
      className="w-full animate-ql-fade-in"
    />
  );
}
