"""VaR, stress replays, correlation, and the optimiser — one working surface.

Risk and portfolio share a page here where the product UI separates them. For a
researcher they are one question ("what does this book do when things go wrong,
and what would a different weighting have done"), and the extra navigation step
costs more than the separation is worth.
"""

from __future__ import annotations

import datetime as dt

import streamlit as st

from _shared import ApiError, get, page, post, show_error

page(
    "Risk and Portfolio",
    "Equal-weighted over the point-in-time universe unless weights are given. "
    "Stress windows replay real ingested history — a window with no data is "
    "reported as a gap, never estimated.",
)

index = st.sidebar.text_input("Index", "NIFTY500")
start = st.sidebar.date_input("Start", dt.date(2015, 1, 1)).isoformat()
end = st.sidebar.date_input("End", dt.date(2021, 12, 31)).isoformat()
lookback = st.sidebar.number_input("Lookback", 30, 5000, 504, step=42)

window = {"index_name": index, "start": start, "end": end, "lookback": int(lookback)}

var_tab, stress_tab, corr_tab, mc_tab, opt_tab = st.tabs(
    ["VaR / ES", "Stress", "Correlation", "Monte Carlo", "Optimiser"]
)

with var_tab:
    try:
        report = post("/risk/var", window)
    except ApiError as exc:
        show_error(exc)
    else:
        st.dataframe(
            [
                {
                    "method": e["method"],
                    "confidence": e["confidence"]["formatted"],
                    "VaR": e["var"]["formatted"],
                    "ES": e["expected_shortfall"]["formatted"],
                    "tail_ratio": e["tail_ratio"]["formatted"],
                    "volatility": e["volatility"]["formatted"],
                    "skew": e["skew"]["formatted"],
                    "excess_kurtosis": e["excess_kurtosis"]["formatted"],
                }
                for e in report["entries"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.info(report["interpretation"])

with stress_tab:
    try:
        stress = post("/risk/stress", {})
    except ApiError as exc:
        show_error(exc)
    else:
        st.dataframe(
            [
                {
                    "window": r["name"],
                    "dates": f"{r['window']['start_display']} → {r['window']['end_display']}",
                    "return": r["total_return"]["formatted"],
                    "max_drawdown": r["max_drawdown"]["formatted"],
                    "worst_day": r["worst_day"]["formatted"],
                    "volatility": r["volatility"]["formatted"],
                    "mean_correlation": r["mean_correlation"]["formatted"],
                    "recovered": (
                        r["recovery_days"]["formatted"]
                        if r["recovered"]
                        else "not by window end"
                    ),
                }
                for r in stress["results"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        if stress["skipped"]:
            st.warning(
                "Skipped for want of data: "
                + ", ".join(gap["name"] for gap in stress["skipped"])
            )
        st.code(stress["rendered"], language=None)

with corr_tab:
    try:
        corr = post(
            "/risk/correlation",
            {"index_name": index, "start": start, "end": end, "lookback": 252, "limit": 40},
        )
    except ApiError as exc:
        show_error(exc)
    else:
        cols = st.columns(3)
        cols[0].metric("Mean correlation", corr["mean_correlation"]["formatted"])
        cols[1].metric("Max correlation", corr["max_correlation"]["formatted"])
        cols[2].metric("Observations", corr["n_observations"]["formatted"])
        if corr["truncated"]:
            st.caption(f"First {len(corr['securities'])} securities.")
        st.dataframe(
            [
                {"security_id": name, **{
                    other: round(corr["matrix"][i][j], 2)
                    for j, other in enumerate(corr["securities"])
                }}
                for i, name in enumerate(corr["securities"])
            ],
            use_container_width=True,
            hide_index=True,
            height=500,
        )

with mc_tab:
    simulations = st.number_input("Simulations", 100, 50_000, 5_000, step=1_000)
    horizon = st.number_input("Horizon (days)", 5, 2_520, 252, step=21)
    try:
        mc = post(
            "/risk/monte-carlo",
            {**window, "n_simulations": int(simulations), "horizon_days": int(horizon)},
        )
    except ApiError as exc:
        show_error(exc)
    else:
        cols = st.columns(4)
        cols[0].metric("Terminal p05", mc["terminal_p05"]["formatted"])
        cols[1].metric("Terminal p50", mc["terminal_p50"]["formatted"])
        cols[2].metric("Terminal p95", mc["terminal_p95"]["formatted"])
        cols[3].metric("Max drawdown p95", mc["max_drawdown_p95"]["formatted"])
        st.line_chart(
            {f"p{band['percentile']:02d}": band["values"] for band in mc["bands"]},
            height=320,
        )
        st.info(mc["interpretation"])

with opt_tab:
    try:
        methods = get("/portfolio/methods")
    except ApiError as exc:
        show_error(exc)
        methods = []
    if methods:
        method = st.selectbox("Method", methods)
        covariance = st.selectbox("Covariance", ["ledoit_wolf", "sample", "oas"])
        max_weight = st.slider("Max weight", 0.01, 1.0, 0.15)
        long_only = st.checkbox("Long only", value=True)
        try:
            result = post(
                "/portfolio/optimise",
                {
                    "index_name": index,
                    "method": method,
                    "covariance": covariance,
                    "lookback": 252,
                    "max_weight": max_weight,
                    "long_only": long_only,
                },
            )
        except ApiError as exc:
            show_error(exc)
        else:
            cols = st.columns(5)
            cols[0].metric("Expected return", result["expected_return"]["formatted"])
            cols[1].metric("Volatility", result["volatility"]["formatted"])
            cols[2].metric(
                "Ex-ante Sharpe", result["ex_ante_sharpe"]["display"]["formatted"]
            )
            cols[3].metric("Assets", result["n_assets"]["formatted"])
            cols[4].metric(
                "Effective positions", result["effective_positions"]["formatted"]
            )
            st.caption(
                "Ex-ante Sharpe is a reference, not a result: "
                + result["ex_ante_sharpe"]["why_not_deflated"]
            )
            st.dataframe(
                [
                    {
                        "security_id": h["security_id"],
                        "weight": h["weight"]["formatted"],
                    }
                    for h in sorted(
                        result["holdings"],
                        key=lambda h: h["weight"]["value"] or 0,
                        reverse=True,
                    )
                ],
                use_container_width=True,
                hide_index=True,
                height=500,
            )
            st.caption(result["covariance"]["detail"])
