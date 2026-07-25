"use client";

/**
 * ⌘K / ctrl+K: switch security, or jump to a page.
 *
 * The security list is fetched from the API as the reader types rather than
 * filtered client-side — the NIFTY 500 universe plus every delisted security is
 * more than is worth shipping to the browser on load, and the API already has a
 * `search` parameter that does the work.
 *
 * The selected security is stored in `localStorage` and read by the pages that
 * take one (Feature Explorer, Data Quality drill-down), so switching here
 * changes what those pages show without a per-page picker.
 */

import { Command } from "cmdk";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import type { SecurityOut } from "@/lib/api/client";
import { api } from "@/lib/api/use-api";

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

const STORAGE_KEY = "quantlab.security_id";

/** The security the security-scoped pages should show. */
export function useSelectedSecurity(): [string | null, (id: string) => void] {
  const [id, setId] = useState<string | null>(null);

  useEffect(() => {
    setId(window.localStorage.getItem(STORAGE_KEY));
    const onChange = () => setId(window.localStorage.getItem(STORAGE_KEY));
    window.addEventListener("quantlab:security", onChange);
    return () => window.removeEventListener("quantlab:security", onChange);
  }, []);

  return [
    id,
    (next: string) => {
      window.localStorage.setItem(STORAGE_KEY, next);
      // Same-tab listeners; `storage` only fires across tabs.
      window.dispatchEvent(new Event("quantlab:security"));
    },
  ];
}

export function CommandPalette({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const [securities, setSecurities] = useState<SecurityOut[]>([]);
  const [loading, setLoading] = useState(false);
  const [, select] = useSelectedSecurity();

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    setLoading(true);
    api()
      .securities({ search: query || undefined, limit: 20 })
      .then((page) => {
        if (!cancelled) setSecurities(page.items);
      })
      .catch(() => {
        if (!cancelled) setSecurities([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, query]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-full sm:w-[520px]">
        <DialogTitle className="sr-only">Command palette</DialogTitle>
        <Command
          shouldFilter={false}
          className="flex flex-col"
          // The list is server-filtered; client filtering on top would hide
          // results the API deliberately returned.
        >
          <Command.Input
            value={query}
            onValueChange={setQuery}
            placeholder="Search securities, or jump to a page…"
            className="h-6 w-full border-b border-border bg-transparent px-3 text-base text-text-primary placeholder:text-text-muted focus:outline-none"
          />
          <Command.List className="max-h-screen overflow-y-auto p-1">
            <Command.Empty className="p-3 text-sm text-text-muted">
              {loading ? "Searching…" : "Nothing matches."}
            </Command.Empty>

            {securities.length > 0 && (
              <Command.Group
                heading="Securities"
                className="px-2 py-1 text-xs uppercase tracking-wide text-text-secondary"
              >
                {securities.map((security) => (
                  <Command.Item
                    key={security.security_id}
                    value={security.security_id}
                    onSelect={() => {
                      select(security.security_id);
                      onOpenChange(false);
                    }}
                    className="flex h-5 cursor-pointer items-center justify-between gap-3 rounded px-2 text-base text-text-primary data-[selected=true]:bg-surface-raised"
                  >
                    <span className="truncate">{security.name ?? "—"}</span>
                    <span className="ql-num shrink-0 text-xs text-text-muted">
                      {security.security_id}
                      {!security.is_active && " · delisted"}
                    </span>
                  </Command.Item>
                ))}
              </Command.Group>
            )}

            <Command.Group
              heading="Pages"
              className="px-2 py-1 text-xs uppercase tracking-wide text-text-secondary"
            >
              {PAGES.filter((page) =>
                page.label.toLowerCase().includes(query.toLowerCase()),
              ).map((page) => (
                <Command.Item
                  key={page.href}
                  value={`page:${page.href}`}
                  onSelect={() => {
                    router.push(page.href);
                    onOpenChange(false);
                  }}
                  className="flex h-5 cursor-pointer items-center rounded px-2 text-base text-text-secondary data-[selected=true]:bg-surface-raised data-[selected=true]:text-text-primary"
                >
                  {page.label}
                </Command.Item>
              ))}
            </Command.Group>
          </Command.List>
        </Command>
      </DialogContent>
    </Dialog>
  );
}
