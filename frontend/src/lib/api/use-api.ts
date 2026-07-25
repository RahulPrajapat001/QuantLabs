"use client";

/**
 * The one data-fetching hook in the product.
 *
 * Deliberately not React Query: every page here loads one or two documents from
 * a local API and re-fetches when a control changes. A cache layer would add a
 * dependency, a provider and a set of staleness rules to solve a problem this
 * product does not have.
 *
 * The three-state return is the contract the design system's chart and table
 * rules rest on — `loading` renders a skeleton of the same height as the
 * content, `error` renders the message, and `data` renders. There is no fourth
 * state where a component guesses.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, QuantLabClient } from "./client";

const client = new QuantLabClient({
  baseUrl: process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000",
});

/** The shared client. Components never construct their own. */
export function api(): QuantLabClient {
  return client;
}

export interface ApiState<T> {
  data: T | null;
  error: string | null;
  /** True only on the first load. A refetch keeps the old data on screen. */
  loading: boolean;
  /** True while a refetch is in flight over already-rendered data. */
  refreshing: boolean;
  refetch: () => void;
}

/**
 * Run `fetcher` on mount and whenever `deps` change.
 *
 * `enabled: false` holds the request — used by pages whose query is not yet
 * complete (no security picked, no strategy chosen). A held request reports
 * `loading: false` with `data: null`, which is the empty state, not a spinner.
 */
export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: readonly unknown[],
  options: { enabled?: boolean } = {},
): ApiState<T> {
  const enabled = options.enabled ?? true;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [refreshing, setRefreshing] = useState(false);
  const [nonce, setNonce] = useState(0);

  // The latest fetcher, so changing a closure does not itself trigger a fetch:
  // `deps` is the declared trigger and nothing else should be.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const hasData = data !== null;

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    let cancelled = false;
    if (hasData) setRefreshing(true);
    else setLoading(true);
    setError(null);

    fetcherRef
      .current()
      .then((result) => {
        if (cancelled) return;
        setData(result);
      })
      .catch((cause: unknown) => {
        if (cancelled) return;
        setData(null);
        setError(describe(cause));
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
        setRefreshing(false);
      });

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, nonce, ...deps]);

  const refetch = useCallback(() => setNonce((n) => n + 1), []);
  return { data, error, loading, refreshing, refetch };
}

/**
 * A message a researcher can act on.
 *
 * The holdout refusal is singled out because it is not a failure: it is the
 * lock in CLAUDE.md §3 working. Rendering it as "500 Internal Server Error"
 * would make a correctly functioning guard look like a broken API.
 */
export function describe(cause: unknown): string {
  if (cause instanceof ApiError) {
    if (cause.isHoldoutLocked) {
      return `Holdout locked — ${cause.detail}. This is policy, not a fault: holdout data is evaluated once, at the end.`;
    }
    return cause.detail;
  }
  if (cause instanceof TypeError) {
    return "Could not reach the API. Is `uvicorn quantlab.api.app:create_app --factory` running?";
  }
  return cause instanceof Error ? cause.message : String(cause);
}
