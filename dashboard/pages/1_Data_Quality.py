"""Findings and completeness. The full table, unpaginated — that is the point."""

from __future__ import annotations

import streamlit as st

from _shared import ApiError, get, page, show_error

page(
    "Data Quality",
    "Structured findings from the Module 3 checks, and bar coverage per "
    "security per month.",
)

security = st.sidebar.text_input("Security id (optional)") or None
severity = st.sidebar.selectbox("Severity", ["all", "error", "warning", "info"])
refresh = st.sidebar.checkbox(
    "Recompute now", help="Skip the persisted table and run the checks."
)

try:
    report = get(
        "/quality/findings", security_id=security, limit=5000, refresh=refresh or None
    )
    grid = get("/quality/completeness", limit=60)
except ApiError as exc:
    show_error(exc)
    st.stop()

if report["generated_on_demand"]:
    st.info(
        "Computed for this request — no validation run has been persisted. "
        "Run `quantlab validate` to record findings with an id and a history."
    )
else:
    st.caption(
        f"Persisted run {report['run_id'][:8]} · executed {report['generated_at']}"
    )

findings = report["findings"]
if severity != "all":
    findings = [f for f in findings if f["severity"] == severity]

cols = st.columns(4)
cols[0].metric("Findings shown", len(findings))
cols[1].metric("Rows examined", report["rows_examined"]["formatted"])
cols[2].metric("Overall completeness", grid["overall"]["formatted"])
cols[3].metric("Checks run", len(report["checks_run"]))

st.subheader("Findings")
if findings:
    st.dataframe(
        [
            {
                "severity": f["severity"],
                "check": f["check"],
                "security_id": f["security_id"] or "—",
                "date": f["date_display"],
                "detail": f["detail"],
                "remedy": f["remedy"],
            }
            for f in findings
        ],
        use_container_width=True,
        hide_index=True,
        height=600,
    )
else:
    st.success("No findings match. Either the data is clean or nothing has run.")

st.subheader("Completeness by security and month")
if grid["cells"]:
    if grid["truncated"]:
        st.caption(f"Showing the first {len(grid['securities'])} securities.")
    # A pivot is the readable form here, and st.dataframe renders it without
    # any charting library. Null (not listed) stays null rather than becoming 0.
    pivot: dict[str, dict[str, str]] = {}
    for cell in grid["cells"]:
        pivot.setdefault(cell["security_id"], {})[cell["month"]] = cell[
            "completeness"
        ]["formatted"]
    st.dataframe(
        [{"security_id": sid, **months} for sid, months in pivot.items()],
        use_container_width=True,
        hide_index=True,
        height=600,
    )
else:
    st.info("No price history ingested. Run `quantlab ingest`.")
