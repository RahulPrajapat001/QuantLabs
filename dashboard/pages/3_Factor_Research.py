"""Factor evaluation: IC, decay, quantiles, turnover, and the gate verdict."""

from __future__ import annotations

import datetime as dt

import streamlit as st

from _shared import ApiError, get, page, post, show_error

page(
    "Factor Research",
    "Alpha evaluation runs before and separately from strategy backtesting. "
    "A factor that fails the IC gate never reaches the strategy builder.",
)

try:
    factors = get("/factors")
except ApiError as exc:
    show_error(exc)
    st.stop()

names = [f["name"] for f in factors]
if not names:
    st.info("No factors registered.")
    st.stop()

factor = st.sidebar.selectbox("Factor", names)
index = st.sidebar.text_input("Index", "NIFTY500")
start = st.sidebar.date_input("Start", dt.date(2015, 1, 1)).isoformat()
end = st.sidebar.date_input("End", dt.date(2021, 12, 31)).isoformat()
quantiles = st.sidebar.slider("Quantiles", 2, 20, 5)

if not st.sidebar.button("Evaluate", type="primary"):
    st.info("Choose a factor and press Evaluate.")
    st.stop()

try:
    result = post(
        "/factors/evaluate",
        {
            "factor": factor,
            "index_name": index,
            "start": start,
            "end": end,
            "n_quantiles": quantiles,
        },
    )
except ApiError as exc:
    show_error(exc)
    st.stop()

if result["eligible"]:
    st.success(f"{result['factor_name']} passes the eligibility gate.")
else:
    st.warning(
        f"{result['factor_name']} is NOT eligible — {result['ineligibility_reason']}"
    )

cols = st.columns(5)
primary = result["pearson_ic"][0] if result["pearson_ic"] else None
cols[0].metric("Mean IC", primary["mean_ic"]["formatted"] if primary else "—")
cols[1].metric("IC IR", primary["ic_ir"]["formatted"] if primary else "—")
cols[2].metric("t-stat", primary["t_stat"]["formatted"] if primary else "—")
cols[3].metric("Long-short spread", result["long_short_spread"]["formatted"])
cols[4].metric("Turnover", result["turnover"]["formatted"])

st.subheader("IC statistics by horizon")
st.dataframe(
    [
        {
            "horizon": s["horizon"],
            "method": s["method"],
            "mean_ic": s["mean_ic"]["formatted"],
            "std_ic": s["std_ic"]["formatted"],
            "ic_ir": s["ic_ir"]["formatted"],
            "t_stat": s["t_stat"]["formatted"],
            "p_value": s["p_value"]["formatted"],
            "n_periods": s["n_periods"]["formatted"],
            "interpretation": s["interpretation"],
        }
        for s in [*result["pearson_ic"], *result["spearman_ic"]]
    ],
    use_container_width=True,
    hide_index=True,
)

left, right = st.columns(2)

with left:
    st.subheader("IC decay")
    decay = sorted(result["ic_decay"].items(), key=lambda kv: int(kv[0]))
    st.dataframe(
        [{"horizon": f"{h}d", "ic": v["formatted"]} for h, v in decay],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "A factor whose IC is gone by day 5 cannot fund a weekly rebalance "
        "after costs."
    )

with right:
    st.subheader("Quantile bucket returns")
    st.dataframe(
        [
            {"quantile": f"Q{i + 1}", "mean_forward_return": bucket["formatted"]}
            for i, bucket in enumerate(result["quantile_returns"])
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Monotonicity across buckets is the evidence the factor orders returns. "
        "A high top bucket with a jumbled middle is a story about one bucket."
    )
