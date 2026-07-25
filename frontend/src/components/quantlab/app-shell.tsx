"use client";

/**
 * Sidebar for controls, main area for data (DESIGN_SYSTEM.md §4).
 *
 * The sidebar is a fixed 260px and holds navigation plus whatever filters the
 * active page registers. The main area is never interrupted by a control that
 * could have lived here — that rule is what keeps a page scannable top to
 * bottom instead of alternating between chrome and content.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

import { CommandPalette } from "./command-palette";
import { SIDEBAR_HOST_ID } from "./sidebar-context";

const PAGES = [
  { href: "/", label: "Overview" },
  { href: "/data-quality", label: "Data Quality" },
  { href: "/universe", label: "Universe" },
  { href: "/features", label: "Feature Explorer" },
  { href: "/factors", label: "Factor Research" },
  { href: "/backtest", label: "Backtest" },
  { href: "/research-log", label: "Research Log" },
  { href: "/risk", label: "Risk" },
  { href: "/portfolio", label: "Portfolio" },
] as const;

export function AppShell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      const typing =
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable;

      // cmd/ctrl+k opens the palette from anywhere, including a text field.
      if (event.key === "k" && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        setPaletteOpen((open) => !open);
        return;
      }
      // `/` focuses the page's search. Not while already typing, or it would
      // swallow the character in a field the reader is mid-way through.
      if (event.key === "/" && !typing) {
        const search = document.querySelector<HTMLInputElement>("[data-page-search]");
        if (search) {
          event.preventDefault();
          search.focus();
          search.select();
        }
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 flex h-screen w-sidebar shrink-0 flex-col gap-4 border-r border-border bg-surface p-3">
        <div className="flex flex-col gap-1">
          <span className="text-md font-medium text-text-primary">QuantLab</span>
          <span className="text-xs text-text-muted">NSE equity research</span>
        </div>

        <nav className="flex flex-col gap-0">
          {PAGES.map((page) => {
            const active =
              page.href === "/" ? pathname === "/" : pathname.startsWith(page.href);
            return (
              <Link
                key={page.href}
                href={page.href}
                className={cn(
                  "border-l-2 px-2 py-1 text-base ql-fade",
                  active
                    ? "border-border-strong text-text-primary"
                    : "border-transparent text-text-secondary hover:text-text-primary",
                )}
              >
                {page.label}
              </Link>
            );
          })}
        </nav>

        {/* Page-registered controls portal in here (see SidebarPortal). */}
        <div id={SIDEBAR_HOST_ID} className="min-w-0 overflow-y-auto" />

        <div className="mt-auto flex flex-col gap-2 border-t border-border pt-2">
          <button
            type="button"
            onClick={() => setPaletteOpen(true)}
            className="flex items-center justify-between rounded border border-border px-2 py-1 text-xs text-text-secondary hover:border-border-strong hover:text-text-primary ql-fade"
          >
            <span>Find a security</span>
            <kbd className="ql-num text-text-muted">⌘K</kbd>
          </button>
          <ThemeToggle />
        </div>
      </aside>

      <main className="min-w-0 flex-1 p-4">{children}</main>

      <CommandPalette open={paletteOpen} onOpenChange={setPaletteOpen} />
    </div>
  );
}

function ThemeToggle() {
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <button
      type="button"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className="flex items-center justify-between rounded border border-border px-2 py-1 text-xs text-text-secondary hover:border-border-strong hover:text-text-primary ql-fade"
    >
      <span>Theme</span>
      <span className="text-text-muted">{theme === "dark" ? "Dark" : "Light"}</span>
    </button>
  );
}
