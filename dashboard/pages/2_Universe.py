"""Point-in-time membership and the survivorship comparison."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from _shared import ApiError, get, page, show_error

page(
    "Universe",
    "Membership as of a date, reconstructed from reconstitution filings — "
    "never today's constituent list projected backwards.",
)

index = st.sidebar.text_input("Index", "NIFTY500")
as_of = st.sidebar.date_input("As of", dt.date(2021, 12, 31)).isoformat()

try:
    membership = get(f"/universe/{index}/as-of", on=as_of)
    comparison = get(f"/universe/{index}/survivorship", on=as_of)
except ApiError as exc:
    show_error(exc)
    st.stop()

cols = st.columns(5)
cols[0].metric("Point-in-time", membership["count"]["formatted"])
cols[1].metric("Today's list", comparison["current_count"]["formatted"])
cols[2].metric("Overlap", comparison["overlap"]["formatted"])
cols[3].metric("Survival rate", comparison["survival_rate"]["formatted"])
cols[4].metric("Dropped since", len(comparison["dropped_since"]))

st.caption(
    "The gap between the first two figures is the survivorship error a backtest "
    "on today's constituents inherits silently."
)

left, right = st.columns(2)

with left:
    st.subheader(f"Constituents on {membership['as_of_display']}")
    st.dataframe(
        [{"security_id": sid} for sid in membership["security_ids"]],
        use_container_width=True,
        hide_index=True,
        height=500,
    )

with right:
    st.subheader("Only in today's list")
    st.caption("These would be wrongly included by a naive backtest.")
    st.dataframe(
        [{"security_id": sid} for sid in comparison["survivorship_only"]],
        use_container_width=True,
        hide_index=True,
        height=220,
    )
    st.subheader("Dropped since this date")
    st.caption("These would be wrongly excluded.")
    st.dataframe(
        [{"security_id": sid} for sid in comparison["dropped_since"]],
        use_container_width=True,
        hide_index=True,
        height=220,
    )
