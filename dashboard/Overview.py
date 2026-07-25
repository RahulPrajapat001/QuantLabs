"""QuantLab research dashboard — the researcher-facing tool.

Run:

    uvicorn quantlab.api.app:create_app --factory --reload
    streamlit run dashboard/Overview.py

Consumes the same API and the same design tokens as the Next.js product UI, and
is deliberately plainer. See dashboard/README.md for why both exist.
"""

from __future__ import annotations

import streamlit as st

from _shared import ApiError, get, page, sharpe, show_error

page(
    "Overview",
    "Research log trials, universe coverage and data health. "
    "The trials count is the denominator every Sharpe here is deflated against.",
)

try:
    summary = get("/research/log/summary")
    log = get("/research/log", limit=25)
except ApiError as exc:
    show_error(exc)
    st.stop()

left, right = st.columns([1, 2])

with left:
    st.metric("Total trials", log["total_trials"]["formatted"])
    st.caption(
        f"since {summary['first_run_at'] or '—'} · "
        f"{summary['complete_runs']['formatted']} complete, "
        f"{summary['failed_runs']['formatted']} failed"
    )

with right:
    if summary["best"]:
        sharpe(summary["best"], label="Best Sharpe recorded")
    else:
        st.info("No completed run yet. Run a backtest and it appears here.")

st.divider()

counts = st.columns(5)
counts[0].metric("Pre-registered", summary["preregistered_runs"]["formatted"])
counts[1].metric("Exploratory", summary["exploratory_runs"]["formatted"])
counts[2].metric("Holdout touched", summary["holdout_runs"]["formatted"])
counts[3].metric("Running", summary["running_runs"]["formatted"])
counts[4].metric("Dirty tree", summary["dirty_runs"]["formatted"])
st.caption(
    "A dirty-tree run is not reproducible from its commit hash alone. "
    "Exploratory work is legitimate — it is labelled, not hidden."
)

st.divider()
st.subheader("Recent runs")

rows = [
    {
        "run_id": run["run_id"][:8],
        "strategy": run["strategy_name"],
        "window": run["window_display"],
        "split": " + ".join(p.upper() for p in run["badge"]["parts"]),
        "status": run["status"],
        "sharpe": run["sharpe"]["value_display"]["formatted"] if run["sharpe"] else "—",
        "deflated": (
            run["sharpe"]["deflated_display"]["formatted"] if run["sharpe"] else "—"
        ),
        "trials": (
            run["sharpe"]["trials_display"]["formatted"] if run["sharpe"] else "—"
        ),
        "provenance": "exploratory" if run["exploratory"] else (run["hypothesis_id"] or "—"),
        "reproducible": "clean" if run["reproducible"] else "dirty",
    }
    for run in log["items"]
]

if rows:
    st.dataframe(rows, use_container_width=True, hide_index=True)
else:
    st.info("The research log is empty.")

st.divider()
st.caption(
    "The plain-text summary below is exactly what `quantlab research summary` "
    "prints — same source, so the terminal and this page cannot disagree."
)
st.code(summary["rendered"], language=None)
