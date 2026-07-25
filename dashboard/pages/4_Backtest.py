"""Submit a backtest and read the result — including its trade log."""

from __future__ import annotations

import datetime as dt
import time

import streamlit as st

from _shared import ApiError, cost_warning, get, page, post, sharpe, show_error, split_badge

page(
    "Backtest",
    "Event-driven, T+1 open, itemised Indian statutory costs. Every run writes "
    "to the research log from inside the engine.",
)

try:
    strategies = get("/backtests/strategies")
    hypotheses = get("/research/hypotheses")
except ApiError as exc:
    show_error(exc)
    st.stop()

names = [s["name"] for s in strategies]
if not names:
    st.info("No strategies registered.")
    st.stop()

strategy = st.sidebar.selectbox("Strategy", names)
index = st.sidebar.text_input("Index", "NIFTY500")
start = st.sidebar.date_input("Start", dt.date(2015, 1, 1)).isoformat()
end = st.sidebar.date_input("End", dt.date(2021, 12, 31)).isoformat()
cost_model = st.sidebar.selectbox(
    "Cost model",
    ["IndianEquityCostModel", "GenericBpsModel", "ZeroCostModel"],
)
capital = st.sidebar.number_input("Initial capital", value=1_000_000, step=100_000)
max_positions = st.sidebar.slider("Max positions", 1, 50, 10)
hypothesis = st.sidebar.selectbox(
    "Hypothesis",
    ["(exploratory)"] + [h["id"] for h in hypotheses],
    help="Leave exploratory and the run is labelled as such — legitimate, but labelled.",
)

if not st.sidebar.button("Run backtest", type="primary"):
    st.info("Configure a run in the sidebar and press Run backtest.")
    st.stop()

try:
    job = post(
        "/backtests",
        {
            "strategy": strategy,
            "index_name": index,
            "start": start,
            "end": end,
            "cost_model": cost_model,
            "initial_capital": float(capital),
            "max_positions": max_positions,
            "max_position_weight": 0.15,
            "hypothesis_id": None if hypothesis == "(exploratory)" else hypothesis,
        },
    )
except ApiError as exc:
    show_error(exc)
    st.stop()

status = st.status(f"Job {job['job_id'][:8]} submitted…", expanded=True)
try:
    while job["state"] in ("queued", "running"):
        time.sleep(1.0)
        job = get(f"/backtests/jobs/{job['job_id']}")
        status.update(label=f"Job {job['job_id'][:8]} — {job['state']}…")
except ApiError as exc:
    status.update(label="Polling failed", state="error")
    show_error(exc)
    st.stop()

if job["state"] == "failed" or not job["result"]:
    status.update(label="The backtest failed", state="error")
    st.error(job["error"] or "The backtest failed without a message.")
    st.stop()

status.update(label=f"Complete in {job['duration_seconds']:.1f}s", state="complete")
result = job["result"]

st.subheader(f"{result['strategy']} · {split_badge(result['badge'])}")
st.caption(
    f"{result['window']['start_display']} → {result['window']['end_display']} · "
    f"{result['trading_days']['formatted']} sessions · run {result['run_id'][:8]}"
)

marker = cost_warning(result["costs"])
if marker:
    st.warning(marker)

sharpe(result["sharpe"])

st.divider()
row1 = st.columns(6)
row1[0].metric("Total return", result["total_return"]["formatted"])
row1[1].metric("Annual return", result["annual_return"]["formatted"])
row1[2].metric("Volatility", result["annual_volatility"]["formatted"])
row1[3].metric("Max drawdown", result["drawdown"]["max_drawdown"]["formatted"])
row1[4].metric("Sortino", result["sortino"]["formatted"])
row1[5].metric("Calmar", result["calmar"]["formatted"])

row2 = st.columns(6)
row2[0].metric("Trades", result["trades"]["formatted"])
row2[1].metric("Win rate", result["win_rate"]["formatted"])
row2[2].metric("Profit factor", result["profit_factor"]["formatted"])
row2[3].metric("Turnover", result["turnover"]["formatted"])
row2[4].metric("Total costs", result["total_costs"]["formatted"])
row2[5].metric("Cost drag", result["cost_drag"]["formatted"])

st.subheader("Equity curve")
# st.line_chart over the raw curve: plain, and it is the researcher's own
# working view rather than a presentation artefact.
st.line_chart(
    {
        "equity": [point["equity"] for point in result["equity_curve"]],
    },
    height=280,
)
st.caption("Drawdown, as a negative fraction from the running peak:")
st.area_chart(
    {"drawdown": [point["drawdown"] for point in result["equity_curve"]]},
    height=160,
)

st.subheader("Cost attribution")
st.dataframe(
    [
        {"component": name.replace("_", " "), "amount": value["formatted"]}
        for name, value in sorted(result["cost_attribution"].items())
    ],
    use_container_width=True,
    hide_index=True,
)

st.subheader("Trade log")
if result["trade_log_truncated"]:
    st.caption(
        f"Showing {len(result['trade_log'])} of {result['trades']['formatted']} trades."
    )
st.dataframe(
    [
        {
            "security_id": t["security_id"],
            "entry": t["entry_display"],
            "exit": t["exit_display"],
            "days": t["holding_days"]["formatted"],
            "quantity": t["quantity"]["formatted"],
            "entry_price": t["entry_price"]["formatted"],
            "exit_price": t["exit_price"]["formatted"],
            "costs": t["costs"]["formatted"],
            "pnl": t["pnl"]["formatted"],
            "return": t["return_pct"]["formatted"],
        }
        for t in result["trade_log"]
    ],
    use_container_width=True,
    hide_index=True,
    height=500,
)

st.subheader("Rejected orders")
st.caption("A finding, not a diagnostic — a strategy that cannot fill is not a strategy.")
st.dataframe(
    [{"reason": k, "count": v} for k, v in sorted(result["rejections"].items())],
    use_container_width=True,
    hide_index=True,
)
