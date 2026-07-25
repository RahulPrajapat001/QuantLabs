"""Shared plumbing for the Streamlit dashboard.

Deliberately small. The dashboard's value is that a new view costs minutes, and
a framework layer here would take that away. What lives in this module is only
what would otherwise be *copied*: the API client, the token CSS load, and the
three rendering rules that are binding rather than stylistic (Sharpe, split
badge, cost warning).

Nothing here formats a number. The API returns display strings alongside raw
values precisely so the dashboard, the product UI and the PDF tearsheet cannot
disagree — reimplementing formatting in Python would create a fourth answer.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st

API_URL = os.environ.get("QUANTLAB_API_URL", "http://127.0.0.1:8000").rstrip("/")

#: The generated token stylesheet, shared with the Next.js frontend.
TOKENS_CSS = Path(__file__).resolve().parents[1] / "config" / "design-tokens.css"

TIMEOUT = 120


class ApiError(RuntimeError):
    """A failed API call, carrying the server's own message."""

    def __init__(self, status: int, detail: str, code: str | None = None) -> None:
        self.status = status
        self.detail = detail
        self.code = code
        super().__init__(f"{status}: {detail}")

    @property
    def is_holdout_locked(self) -> bool:
        """Not a failure — the lock in CLAUDE.md §3 doing its job."""
        return self.code == "holdout_locked"


def _request(method: str, path: str, **kwargs: Any) -> Any:
    try:
        response = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(
            0,
            f"Could not reach the API at {API_URL}. Is uvicorn running? ({exc})",
        ) from exc

    if not response.ok:
        try:
            body = response.json()
        except ValueError:
            body = {}
        raise ApiError(
            response.status_code,
            body.get("detail", response.reason),
            body.get("error"),
        )
    return response.json()


@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, **params: Any) -> Any:
    """GET, cached for a minute so a widget change does not refetch everything."""

    clean = {k: v for k, v in params.items() if v is not None}
    return _request("GET", path, params=clean)


def post(path: str, body: dict[str, Any]) -> Any:
    """POST. Never cached — these compute, and a stale optimisation is a lie."""

    return _request("POST", path, json=body)


def page(title: str, description: str) -> None:
    """Standard page opener: config, tokens, heading."""

    st.set_page_config(page_title=f"QuantLab · {title}", layout="wide")
    _load_tokens()
    st.title(title)
    st.caption(description)


def _load_tokens() -> None:
    """Load the shared token CSS and apply the tabular-numbers rule.

    This is the whole of the styling. Streamlit's own widgets do the rest, on
    purpose — see dashboard/README.md.
    """

    if not TOKENS_CSS.exists():  # pragma: no cover - generated file
        return
    css = TOKENS_CSS.read_text(encoding="utf-8")
    st.markdown(
        f"""<style>
        {css}
        /* Every number mono and tabular, the one rule worth enforcing here. */
        [data-testid="stMetricValue"], .stDataFrame, code {{
            font-family: var(--ql-font-mono);
            font-variant-numeric: tabular-nums;
        }}
        </style>""",
        unsafe_allow_html=True,
    )


def show_error(exc: ApiError) -> None:
    """Render an API failure. A holdout refusal is explained, not alarmed."""

    if exc.is_holdout_locked:
        st.warning(
            f"Holdout locked — {exc.detail}\n\n"
            "This is policy, not a fault: holdout data is evaluated once, at the "
            "end of the project."
        )
    else:
        st.error(exc.detail)


# --- The three binding display rules -----------------------------------------


def split_badge(badge: dict[str, Any]) -> str:
    """Compound splits render as segments, never collapsed to one label."""

    parts = badge.get("parts") or ["UNKNOWN"]
    return " + ".join(part.upper() for part in parts)


def cost_warning(costs: dict[str, Any]) -> str | None:
    """The marker for anything that is not the full Indian statutory model."""

    if costs.get("realistic"):
        return None
    return f"⚠ {costs.get('warning') or costs.get('name')}"


def sharpe(report: dict[str, Any] | None, *, label: str = "Sharpe") -> None:
    """Render a Sharpe **with** its trials count and deflated value.

    The rule from DESIGN_SYSTEM.md §7.1 applies here exactly as it does in the
    product UI: a reader who sees 1.82 must see "312 trials · deflated 0.31"
    without an interaction. There is no code path in this dashboard that prints
    a bare Sharpe, and this function is the reason.
    """

    if report is None:
        st.metric(label, "—")
        st.caption("No Sharpe recorded for this run.")
        return

    st.metric(label, report["value_display"]["formatted"])
    st.caption(
        f"{report['trials_display']['formatted']} trials · "
        f"deflated {report['deflated_display']['formatted']} · "
        f"PBO {report['pbo_display']['formatted']} · "
        f"{split_badge(report['badge'])} · {report['provenance']}"
    )
    if not report.get("significant", True):
        st.warning(
            f"Not significant after {report['trials_display']['formatted']} trials — "
            f"the deflated Sharpe is below the 0.95 threshold. {report['summary']}"
        )
    marker = cost_warning(report["costs"])
    if marker:
        st.warning(marker)


def nums(rows: list[dict[str, Any]], *fields: str) -> list[dict[str, Any]]:
    """Flatten `Num` objects to their display strings for `st.dataframe`.

    Keeps the API's formatting rather than letting pandas re-render floats,
    which is how a `0.124` becomes `12.4%` in one view and `0.12` in another.
    """

    out = []
    for row in rows:
        flat = dict(row)
        for field in fields:
            value = flat.get(field)
            if isinstance(value, dict) and "formatted" in value:
                flat[field] = value["formatted"]
        out.append(flat)
    return out
