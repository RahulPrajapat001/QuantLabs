"use client";

/**
 * Chart theming, read from the generated artifacts rather than restated here.
 *
 * The Plotly templates and the token CSS are both produced by
 * scripts/build_design_tokens.py. Importing the JSON means a colour cannot drift
 * between the chart layer and the rest of the UI: there is one definition and
 * two consumers (this and the Streamlit dashboard).
 */

import { useEffect, useState } from "react";

import darkTemplate from "../../../../config/plotly-template-dark.json";
import lightTemplate from "../../../../config/plotly-template-light.json";

export type Theme = "dark" | "light";

/** Colours a chart needs by *name*, resolved from the live CSS variables. */
export interface ChartTokens {
  series: string[];
  sequential: string[];
  diverging: string[];
  positive: string;
  negative: string;
  warning: string;
  info: string;
  benchmark: string;
  border: string;
  borderStrong: string;
  surface: string;
  surfaceRaised: string;
  textPrimary: string;
  textSecondary: string;
  textMuted: string;
}

function readVar(name: string): string {
  if (typeof window === "undefined") return "";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function readTokens(): ChartTokens {
  const scale = (prefix: string, n: number) =>
    Array.from({ length: n }, (_, i) => readVar(`--ql-${prefix}-${i + 1}`));
  return {
    series: scale("series", 5),
    sequential: scale("seq", 5),
    diverging: scale("div", 5),
    positive: readVar("--ql-positive"),
    negative: readVar("--ql-negative"),
    warning: readVar("--ql-warning"),
    info: readVar("--ql-info"),
    benchmark: readVar("--ql-benchmark"),
    border: readVar("--ql-border"),
    borderStrong: readVar("--ql-border-strong"),
    surface: readVar("--ql-surface"),
    surfaceRaised: readVar("--ql-surface-raised"),
    textPrimary: readVar("--ql-text-primary"),
    textSecondary: readVar("--ql-text-secondary"),
    textMuted: readVar("--ql-text-muted"),
  };
}

/**
 * The active theme, tracking the `data-theme` attribute the toggle sets.
 *
 * Charts are raster-rendered with baked-in colours, so unlike CSS they do not
 * re-theme for free — they have to be told, and redrawn.
 */
export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const read = () =>
      setTheme(
        (document.documentElement.getAttribute("data-theme") as Theme) ?? "dark",
      );
    read();
    const observer = new MutationObserver(read);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  return theme;
}

export function plotlyTemplate(theme: Theme): object {
  return theme === "light" ? lightTemplate : darkTemplate;
}

/** Every chart's base layout. Margins are tight: a chart is not a photo. */
export const BASE_LAYOUT = {
  margin: { l: 56, r: 16, t: 32, b: 40 },
  showlegend: false,
  autosize: true,
} as const;

/** No modebar, no logo, no "produced with Plotly". */
export const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  staticPlot: false,
} as const;
