"""Every run, with its trials count and deflated Sharpe. The full table."""

from __future__ import annotations

import streamlit as st

from _shared import ApiError, get, page, sharpe, show_error

page(
    "Research Log",
    "The denominator for multiple testing. A Sharpe selected from 300 trials "
    "is not the same claim as a Sharpe from one.",
)

provenance = st.sidebar.selectbox(
    "Provenance", ["all", "pre-registered", "exploratory"]
)
strategy = st.sidebar.text_input("Strategy filter") or None

try:
    log = get("/research/log", limit=500, strategy=strategy)
    summary = get("/research/log/summary")
except ApiError as exc:
    show_error(exc)
    st.stop()

items = log["items"]
if provenance == "exploratory":
    items = [run for run in items if run["exploratory"]]
elif provenance == "pre-registered":
    items = [run for run in items if not run["exploratory"]]

cols = st.columns(5)
cols[0].metric("Total trials", log["total_trials"]["formatted"])
cols[1].metric("Complete", summary["complete_runs"]["formatted"])
cols[2].metric("Pre-registered", summary["preregistered_runs"]["formatted"])
cols[3].metric("Exploratory", summary["exploratory_runs"]["formatted"])
cols[4].metric("Holdout touched", summary["holdout_runs"]["formatted"])

st.dataframe(
    [
        {
            "run_id": run["run_id"][:8],
            "created": run["created_at"] or "—",
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
            "significant": (
                ("yes" if run["sharpe"]["significant"] else "no")
                if run["sharpe"]
                else "—"
            ),
            "provenance": (
                "exploratory" if run["exploratory"] else (run["hypothesis_id"] or "—")
            ),
            "cost_model": run["costs"]["name"],
            "reproducible": "clean" if run["reproducible"] else "dirty",
        }
        for run in items
    ],
    use_container_width=True,
    hide_index=True,
    height=600,
)

st.divider()
st.subheader("Inspect one run")
run_ids = [run["run_id"] for run in items]
if run_ids:
    chosen = st.selectbox("Run id", run_ids, format_func=lambda r: r[:8])
    try:
        detail = get(f"/research/log/{chosen}")
    except ApiError as exc:
        show_error(exc)
    else:
        sharpe(detail["sharpe"])
        st.json(detail["parameters"])
        st.caption(
            f"commit {detail['git_commit_hash'] or '—'} · "
            f"{'dirty tree' if detail['git_dirty'] else 'clean tree'}"
        )
else:
    st.info("No runs to inspect.")

st.divider()
st.code(summary["rendered"], language=None)
