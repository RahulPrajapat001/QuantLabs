import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";

import { AppShell } from "@/components/quantlab/app-shell";

import "./globals.css";

// Inter for all UI text; JetBrains Mono for every number without exception.
// The variables are bound to the design tokens in globals.css.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "QuantLab",
  description:
    "Quantitative research platform for Indian equities. Point-in-time universes, look-ahead-safe features, and Sharpe ratios reported with their trials count.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // `data-theme` is the switch the tokens and the chart layer both read. Dark
    // is the default: this is a working surface, not a marketing page.
    <html lang="en" data-theme="dark" className={`${inter.variable} ${mono.variable}`}>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
