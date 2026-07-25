/**
 * Thin typed client over the QuantLab API.
 *
 * Hand-written and committed, unlike `schema.ts` which is generated. It is
 * deliberately small: every signature references a generated type, so when a
 * response model changes in Python this file stops compiling rather than
 * failing at runtime.
 *
 * Two rules this file exists to enforce on the frontend:
 *
 * 1. **Never format a number in the UI.** Every numeric field arrives as `Num`
 *    — `{ value, formatted }`. Render `formatted`; use `value` only for charts
 *    and sorting. `formatNum` below is the single accessor, and there is no
 *    helper here that turns a raw number into a string.
 *
 * 2. **Never render a Sharpe as a number.** `SharpeObject` carries its trials
 *    count and deflated value as required fields. A component typed to accept
 *    one cannot be handed `1.8`, and there is no widening cast in this file to
 *    let it through.
 */

import type { components, paths } from "./schema";

export type Schemas = components["schemas"];

export type Num = Schemas["Num"];
export type SharpeObject = Schemas["SharpeObject"];
export type ReferenceRatio = Schemas["ReferenceRatio"];
export type SplitBadge = Schemas["SplitBadge"];
export type CostModelStatus = Schemas["CostModelStatus"];
export type BacktestResult = Schemas["BacktestResult"];
export type BacktestRequest = Schemas["BacktestRequest"];
export type JobOut = Schemas["JobOut"];
export type RunOut = Schemas["RunOut"];
export type LogPage = Schemas["LogPage"];
export type LogSummaryOut = Schemas["LogSummaryOut"];
export type EvaluationOut = Schemas["EvaluationOut"];
export type OptimiseOut = Schemas["OptimiseOut"];
export type VaROut = Schemas["VaROut"];
export type StressReportOut = Schemas["StressReportOut"];
export type QualityReport = Schemas["QualityReport"];
export type SecurityPage = Schemas["SecurityPage"];
export type MembershipOut = Schemas["MembershipOut"];
export type SurvivorshipComparison = Schemas["SurvivorshipComparison"];

/** Every path the API serves, for callers that build URLs generically. */
export type ApiPaths = paths;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly code?: string,
  ) {
    super(`${status}: ${detail}`);
    this.name = "ApiError";
  }

  /** The holdout is locked. Not a bug — a policy refusal (see CLAUDE.md §3). */
  get isHoldoutLocked(): boolean {
    return this.code === "holdout_locked";
  }
}

export interface ClientOptions {
  baseUrl?: string;
  fetch?: typeof globalThis.fetch;
}

export class QuantLabClient {
  private readonly baseUrl: string;
  private readonly fetchImpl: typeof globalThis.fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? "http://127.0.0.1:8000").replace(/\/$/, "");
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  private async request<T>(
    path: string,
    init?: RequestInit & { query?: Record<string, unknown> },
  ): Promise<T> {
    const url = new URL(this.baseUrl + path);
    for (const [key, value] of Object.entries(init?.query ?? {})) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, String(value));
      }
    }
    const response = await this.fetchImpl(url.toString(), {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => ({}))) as {
        detail?: string;
        error?: string;
      };
      throw new ApiError(
        response.status,
        body.detail ?? response.statusText,
        body.error,
      );
    }
    return (await response.json()) as T;
  }

  private post<T>(path: string, body: unknown): Promise<T> {
    return this.request<T>(path, { method: "POST", body: JSON.stringify(body) });
  }

  // --- Securities and universe ---

  securities(query?: {
    search?: string;
    active_only?: boolean;
    limit?: number;
    offset?: number;
  }): Promise<SecurityPage> {
    return this.request<SecurityPage>("/securities", { query });
  }

  universeAsOf(indexName: string, on: string): Promise<MembershipOut> {
    return this.request<MembershipOut>(`/universe/${indexName}/as-of`, {
      query: { on },
    });
  }

  survivorship(indexName: string, on: string): Promise<SurvivorshipComparison> {
    return this.request<SurvivorshipComparison>(
      `/universe/${indexName}/survivorship`,
      { query: { on } },
    );
  }

  // --- Data ---

  ohlcv(query?: {
    security_id?: string;
    start?: string;
    end?: string;
    limit?: number;
  }) {
    return this.request<Schemas["BarSeries"]>("/market/ohlcv", { query });
  }

  features(query?: { security_id?: string; names?: string; limit?: number }) {
    return this.request<Schemas["FeaturePanel"]>("/market/features", { query });
  }

  qualityFindings(query?: { security_id?: string; limit?: number }) {
    return this.request<QualityReport>("/quality/findings", { query });
  }

  // --- Research ---

  evaluateFactor(body: Schemas["EvaluationRequest"]): Promise<EvaluationOut> {
    return this.post<EvaluationOut>("/factors/evaluate", body);
  }

  researchLog(query?: { strategy?: string; status?: string; limit?: number }) {
    return this.request<LogPage>("/research/log", { query });
  }

  researchSummary(): Promise<LogSummaryOut> {
    return this.request<LogSummaryOut>("/research/log/summary");
  }

  /** A run's Sharpe, deflated against the log's trials count. */
  deflated(runId: string, scopeStrategy?: string): Promise<SharpeObject> {
    return this.request<SharpeObject>(`/research/deflated/${runId}`, {
      query: { scope_strategy: scopeStrategy },
    });
  }

  // --- Backtests ---

  submitBacktest(body: BacktestRequest): Promise<JobOut> {
    return this.post<JobOut>("/backtests", body);
  }

  pollBacktest(jobId: string): Promise<JobOut> {
    return this.request<JobOut>(`/backtests/jobs/${jobId}`);
  }

  /**
   * Submit and poll until the job settles.
   *
   * Polling rather than a socket is the API's design (PRD Module 12); this is
   * the convenience wrapper, not a different mechanism.
   */
  async runBacktest(
    body: BacktestRequest,
    options: { intervalMs?: number; timeoutMs?: number } = {},
  ): Promise<BacktestResult> {
    const interval = options.intervalMs ?? 1000;
    const deadline = Date.now() + (options.timeoutMs ?? 600_000);
    let job = await this.submitBacktest(body);
    while (job.state === "queued" || job.state === "running") {
      if (Date.now() > deadline) {
        throw new ApiError(504, `backtest job ${job.job_id} did not finish in time`);
      }
      await new Promise((resolve) => setTimeout(resolve, interval));
      job = await this.pollBacktest(job.job_id);
    }
    if (job.state === "failed" || job.result === null) {
      throw new ApiError(500, job.error ?? "the backtest failed without a message");
    }
    return job.result;
  }

  // --- Risk and portfolio ---

  valueAtRisk(body: Schemas["VaRRequest"]): Promise<VaROut> {
    return this.post<VaROut>("/risk/var", body);
  }

  stress(body: Schemas["StressRequest"]): Promise<StressReportOut> {
    return this.post<StressReportOut>("/risk/stress", body);
  }

  optimise(body: Schemas["OptimiseRequest"]): Promise<OptimiseOut> {
    return this.post<OptimiseOut>("/portfolio/optimise", body);
  }

  // --- Reports ---

  tearsheetUrl(runId: string, format: "pdf" | "html" = "pdf"): string {
    return `${this.baseUrl}/reports/tearsheet/${runId}.${format}`;
  }
}

/**
 * The only way to get a display string out of a `Num`.
 *
 * There is deliberately no `formatNumber(value: number)` in this module. A
 * component that wants a formatted number asks the API for one; formatting in
 * the browser would create a second source of truth that drifts from the PDF
 * tearsheet and the Streamlit dashboard.
 */
export function formatNum(num: Num): string {
  return num.formatted;
}

/** True when a result's costs are not the full Indian statutory model. */
export function needsCostWarning(costs: CostModelStatus): boolean {
  return !costs.realistic;
}
