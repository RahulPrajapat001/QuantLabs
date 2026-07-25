"""The two contracts PRD Module 12 makes structural, checked against the schema.

These are schema-level tests rather than response-level ones on purpose. A
response that happens to carry a `SharpeObject` today proves nothing about the
endpoint added next week; the OpenAPI document is what the TypeScript client is
generated from, so it is what has to be right.
"""

from __future__ import annotations

import re

import pytest

from quantlab.api.app import create_app

MANDATED_SHARPE_FIELDS = {
    "value",
    "trials",
    "deflated_sharpe",
    "pbo",
    "split",
    "cost_model",
}
#: Types that may legitimately carry a Sharpe-shaped number.
CONTRACT_TYPES = {"SharpeObject", "ReferenceRatio"}


@pytest.fixture(scope="module")
def spec() -> dict:
    return create_app().openapi()


@pytest.fixture(scope="module")
def schemas(spec: dict) -> dict:
    return spec["components"]["schemas"]


# --- A Sharpe is never a bare float ------------------------------------------


def test_the_mandated_sharpe_shape_exists(schemas):
    sharpe = schemas["SharpeObject"]
    assert set(sharpe["properties"]) >= MANDATED_SHARPE_FIELDS


def test_trials_and_deflation_are_required_not_optional(schemas):
    """The whole contract: you cannot construct one without them."""

    required = set(schemas["SharpeObject"]["required"])
    assert "trials" in required
    assert "deflated_sharpe" in required
    assert "value" in required
    assert "split" in required
    assert "cost_model" in required


def test_no_schema_carries_a_sharpe_outside_a_contract_type(schemas):
    """Scan every property whose name mentions a Sharpe.

    Each must resolve to SharpeObject (deflated), ReferenceRatio (explicitly
    not deflatable, with a written reason) or Num (a display wrapper inside one
    of those). A raw `number` anywhere here would be a bare Sharpe.
    """

    offenders = []
    for name, schema in schemas.items():
        if name in CONTRACT_TYPES:
            continue
        for prop, definition in (schema.get("properties") or {}).items():
            if not re.search("sharpe", prop, re.IGNORECASE):
                continue
            rendered = str(definition)
            if not any(t in rendered for t in (*CONTRACT_TYPES, "Num")):
                offenders.append(f"{name}.{prop}: {definition}")
    assert not offenders, "bare Sharpe ratios in the schema:\n" + "\n".join(offenders)


def test_reference_ratio_demands_a_written_reason(schemas):
    """The escape hatch is not free — it costs a sentence."""

    ratio = schemas["ReferenceRatio"]
    assert "why_not_deflated" in ratio["required"]
    assert ratio["properties"]["why_not_deflated"].get("minLength", 0) >= 1


def test_the_backtest_result_carries_an_object_not_a_number(schemas):
    sharpe = schemas["BacktestResult"]["properties"]["sharpe"]
    assert "SharpeObject" in str(sharpe)
    assert "sharpe" in schemas["BacktestResult"]["required"]


def test_the_optimiser_does_not_pretend_its_forecast_is_deflatable(schemas):
    """An ex-ante ratio has no return series, so it is not a SharpeObject."""

    prop = schemas["OptimiseOut"]["properties"]["ex_ante_sharpe"]
    assert "ReferenceRatio" in str(prop)
    assert "SharpeObject" not in str(prop)


# --- Numbers travel with their rendering -------------------------------------


def test_num_carries_both_the_value_and_the_string(schemas):
    num = schemas["Num"]
    assert set(num["required"]) == {"value", "formatted"}
    # A missing value is null, never 0 — so the raw side must be nullable.
    assert "null" in str(num["properties"]["value"])


def test_metric_fields_are_wrapped_rather_than_raw(schemas):
    """Spot-check the tables a reader compares down a column."""

    for model, field in (
        ("BacktestResult", "total_return"),
        ("BacktestResult", "win_rate"),
        ("EvaluationOut", "long_short_spread"),
        ("VaREntry", "expected_shortfall"),
        ("OptimiseOut", "volatility"),
        ("LogSummaryOut", "complete_runs"),
    ):
        definition = str(schemas[model]["properties"][field])
        assert "Num" in definition, f"{model}.{field} is not a Num: {definition}"


def test_split_badges_arrive_decomposed(schemas):
    """Compound labels are split server-side; the UI computes nothing."""

    badge = schemas["SplitBadge"]
    assert {"label", "parts", "holdout_touched"} <= set(badge["required"])


def test_cost_status_decides_the_warning_server_side(schemas):
    costs = schemas["CostModelStatus"]
    assert "realistic" in costs["required"]
    assert "warning" in costs["properties"]


# --- The document itself -----------------------------------------------------


def test_every_operation_declares_a_response_model(spec):
    """An endpoint without a schema is an endpoint the client cannot type."""

    untyped = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            ok = operation.get("responses", {}).get("200") or operation.get(
                "responses", {}
            ).get("202")
            if not ok or "content" not in ok:
                untyped.append(f"{method.upper()} {path}")
    # The PDF endpoint streams bytes and declares its content type by hand.
    assert untyped == [] or untyped == ["get /reports/tearsheet/{run_id}.pdf"]


def test_all_endpoint_groups_are_present(spec):
    """Every group PRD Module 12 lists."""

    paths = set(spec["paths"])
    for required in (
        "/securities",
        "/universe/{index_name}/as-of",
        "/market/ohlcv",
        "/market/features",
        "/quality/findings",
        "/factors/evaluate",
        "/backtests",
        "/backtests/jobs/{job_id}",
        "/research/log",
        "/research/deflated/{run_id}",
        "/risk/var",
        "/risk/stress",
        "/portfolio/optimise",
    ):
        assert required in paths, f"missing endpoint: {required}"
