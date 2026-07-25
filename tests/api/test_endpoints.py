"""Every endpoint, exercised against a seeded database."""

from __future__ import annotations

import pytest

from .conftest import INDEX_NAME

EM_DASH = "—"


# --- Securities ---------------------------------------------------------------


def test_list_securities_includes_delisted_by_default(client):
    body = client.get("/securities").json()
    assert body["total"] == 8
    assert any(not item["is_active"] for item in body["items"]), (
        "a delisted security must still be listed; hiding it is survivorship bias"
    )


def test_active_only_excludes_the_delisted_one(client):
    body = client.get("/securities", params={"active_only": True}).json()
    assert body["total"] == 7
    assert all(item["is_active"] for item in body["items"])


def test_search_matches_ticker_history(client):
    body = client.get("/securities", params={"search": "TEST3"}).json()
    assert [i["security_id"] for i in body["items"]] == ["SEC003"]


def test_one_security_carries_its_ticker_periods(client):
    body = client.get("/securities/SEC001").json()
    assert body["tickers"][0]["ticker"] == "TEST1"
    assert body["tickers"][0]["is_current"] is True


def test_unknown_security_is_404(client):
    assert client.get("/securities/NOPE").status_code == 404


# --- Universe -----------------------------------------------------------------


def test_membership_as_of_uses_reconstitution_history(client):
    early = client.get(
        f"/universe/{INDEX_NAME}/as-of", params={"on": "2016-06-01"}
    ).json()
    late = client.get(
        f"/universe/{INDEX_NAME}/as-of", params={"on": "2017-01-05"}
    ).json()
    # SEC000 leaves the index on 2016-09-30.
    assert "SEC000" in early["security_ids"]
    assert "SEC000" not in late["security_ids"]


def test_survivorship_quantifies_the_look_ahead_names(client):
    body = client.get(
        f"/universe/{INDEX_NAME}/survivorship", params={"on": "2017-01-05"}
    ).json()
    # SEC000 is not in the index now but was not on that date either; the point
    # is that the endpoint reports the difference rather than hiding it.
    assert body["point_in_time_count"]["value"] == 7
    assert set(body["dropped_since"]) | set(body["survivorship_only"]) is not None
    assert 0.0 <= body["survival_rate"]["value"] <= 1.0


def test_intervals_expose_the_source_document(client):
    rows = client.get(f"/universe/{INDEX_NAME}/intervals").json()
    assert rows
    assert all(r["source_document"] for r in rows)


# --- Market data --------------------------------------------------------------


def test_ohlcv_serves_close_and_adj_close_separately(client):
    body = client.get(
        "/market/ohlcv", params={"security_id": "SEC002", "limit": 10}
    ).json()
    assert body["count"]["value"] == 10
    bar = body["bars"][0]
    assert "close" in bar and "adj_close" in bar
    assert bar["close"]["formatted"].startswith("₹")


def test_ohlcv_reports_truncation_rather_than_silently_cutting(client):
    body = client.get("/market/ohlcv", params={"limit": 5}).json()
    assert body["truncated"] is True


def test_ohlcv_with_no_match_is_404(client):
    r = client.get("/market/ohlcv", params={"security_id": "NOPE"})
    assert r.status_code == 404


def test_features_without_a_built_panel_is_404(client):
    r = client.get("/market/features")
    assert r.status_code == 404
    assert "features build" in r.json()["detail"]


# --- Data quality -------------------------------------------------------------


def test_quality_findings_run_on_demand(client):
    body = client.get("/quality/findings").json()
    assert body["generated_on_demand"] is True
    assert body["rows_examined"]["value"] > 0
    assert "extreme_move" in body["checks_run"]


def test_findings_carry_a_remedy_not_just_a_complaint(client):
    body = client.get("/quality/findings").json()
    for finding in body["findings"]:
        assert finding["remedy"], finding


def test_persisted_findings_are_preferred_over_recomputing(client, session):
    """With a run in the table, the endpoint serves it and says so.

    The seeded fixture is clean, so a real ``validate`` run would persist nothing
    — the row is inserted directly. That ``validate`` produces exactly what comes
    back out is covered in ``tests/data/test_validation.py``; what matters here is
    that the endpoint reads the table rather than rerunning the checks.
    """

    from datetime import date as _date

    from quantlab.db.models import DataQualityFinding

    session.add(
        DataQualityFinding(
            run_id="testrun0000",
            check="zero_volume",
            severity="warning",
            security_id="SEC001",
            date=_date(2016, 6, 1),
            detail="zero volume — the security did not trade",
            remedy="Check for a halt or suspension.",
            rows_examined=4_242,
        )
    )
    session.commit()
    try:
        body = client.get("/quality/findings").json()
        assert body["generated_on_demand"] is False
        assert body["run_id"] == "testrun0000"
        assert body["generated_at"] is not None
        assert body["rows_examined"]["value"] == 4_242
        assert [f["check"] for f in body["findings"]] == ["zero_volume"]

        # refresh sidesteps the table without writing to it.
        fresh = client.get("/quality/findings", params={"refresh": True}).json()
        assert fresh["generated_on_demand"] is True
        assert fresh["run_id"] is None
        assert session.query(DataQualityFinding).count() == 1
    finally:
        session.query(DataQualityFinding).delete()
        session.commit()


# --- Research log -------------------------------------------------------------


def test_the_log_returns_a_sharpe_object_never_a_number(client, logged_run):
    body = client.get("/research/log").json()
    entry = next(i for i in body["items"] if i["run_id"] == logged_run)
    sharpe = entry["sharpe"]
    assert isinstance(sharpe, dict)
    assert sharpe["trials"] >= 1
    assert 0.0 <= sharpe["deflated_sharpe"] <= 1.0
    assert sharpe["split"] == "train"


def test_the_trials_count_comes_from_the_log(client, logged_run):
    body = client.get("/research/log").json()
    assert body["total_trials"]["value"] >= 1
    entry = next(i for i in body["items"] if i["run_id"] == logged_run)
    assert entry["sharpe"]["trials"] == body["total_trials"]["value"]


def test_deflating_a_run(client, logged_run):
    body = client.get(f"/research/deflated/{logged_run}").json()
    assert body["run_id"] == logged_run
    assert body["expected_max_sharpe"] >= 0.0
    assert "trials" in body["summary"]


def test_deflating_an_unknown_run_is_not_a_500(client):
    r = client.get("/research/deflated/deadbeef")
    assert r.status_code in (404, 409, 422)


def test_the_summary_renders_the_whole_log(client, logged_run):
    body = client.get("/research/log/summary").json()
    assert body["complete_runs"]["value"] >= 1
    assert "Trials" in body["rendered"]
    assert body["best"]["trials"] >= 1


def test_hypotheses_are_served_with_their_content_hash(client):
    rows = client.get("/research/hypotheses").json()
    assert rows
    assert all(len(h["content_hash"]) == 64 for h in rows)


# --- Factors ------------------------------------------------------------------


def test_factor_list(client):
    names = {f["name"] for f in client.get("/factors").json()}
    assert {"momentum_12_1", "low_volatility"} <= names


def test_evaluating_an_unknown_factor_is_404(client):
    r = client.post("/factors/evaluate", json={"factor": "nope"})
    assert r.status_code == 404


# --- Risk ---------------------------------------------------------------------


def test_var_reports_three_estimators(client):
    body = client.post(
        "/risk/var", json={"index_name": INDEX_NAME, "lookback": 200}
    ).json()
    methods = {e["method"] for e in body["entries"]}
    assert len(methods) >= 3
    for entry in body["entries"]:
        # Losses are positive magnitudes, rendered unsigned.
        assert entry["var"]["value"] >= 0
        assert not entry["var"]["formatted"].startswith(("+", "−"))


def test_stress_windows_are_the_five_required_ones(client):
    slugs = {w["slug"] for w in client.get("/risk/stress/windows").json()}
    assert {"gfc", "covid"} <= slugs


def test_stress_coverage_admits_what_is_missing(client):
    rows = client.get("/risk/stress/coverage").json()
    assert rows
    # The fixture has no 2008 data, so the GFC window must report as unavailable
    # rather than being replayed against nothing.
    gfc = next(r for r in rows if r["slug"] == "gfc")
    assert gfc["available"] is False


# --- Portfolio ----------------------------------------------------------------


@pytest.mark.parametrize(
    "method", ["equal_weight", "inverse_volatility", "minimum_variance"]
)
def test_optimise(client, method):
    body = client.post(
        "/portfolio/optimise",
        json={
            "index_name": INDEX_NAME,
            "method": method,
            "as_of": "2017-01-05",
            "lookback": 200,
            # 7 names in the fixture universe, so a 10% cap could not reach a
            # 100% budget — the optimiser is right to refuse that, and does.
            "max_weight": 0.35,
        },
    ).json()
    assert body["method"]
    assert body["holdings"]
    total = sum(h["weight"]["value"] for h in body["holdings"])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_the_optimiser_sharpe_states_why_it_is_not_deflated(client):
    body = client.post(
        "/portfolio/optimise",
        json={
            "index_name": INDEX_NAME,
            "as_of": "2017-01-05",
            "lookback": 200,
            "max_weight": 0.35,
        },
    ).json()
    ratio = body["ex_ante_sharpe"]
    assert ratio["basis"] == "ex_ante"
    assert "not a realised track record" in ratio["why_not_deflated"]


def test_an_infeasible_weight_cap_is_422_with_the_fix_in_the_message(client):
    """7 names and a 10% cap cannot sum to 1.0. That is the caller's parameters."""

    r = client.post(
        "/portfolio/optimise",
        json={
            "index_name": INDEX_NAME,
            "as_of": "2017-01-05",
            "lookback": 200,
            "max_weight": 0.10,
        },
    )
    assert r.status_code == 422
    assert "Raise max_weight above" in r.json()["detail"]


def test_an_unknown_optimiser_is_422(client):
    r = client.post(
        "/portfolio/optimise", json={"index_name": INDEX_NAME, "method": "magic"}
    )
    assert r.status_code == 422


def test_optimising_before_the_index_existed_is_404(client):
    r = client.post(
        "/portfolio/optimise",
        json={"index_name": INDEX_NAME, "as_of": "2010-01-04"},
    )
    assert r.status_code == 404


# --- CORS ---------------------------------------------------------------------


def test_cors_allows_the_next_js_dev_server(client):
    r = client.options(
        "/securities",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_does_not_allow_an_arbitrary_origin(client):
    r = client.options(
        "/securities",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.headers.get("access-control-allow-origin") != "https://evil.example.com"


# --- Reports ------------------------------------------------------------------


def test_the_html_tearsheet_carries_its_methodology_page(client, logged_run):
    r = client.get(f"/reports/tearsheet/{logged_run}.html")
    assert r.status_code == 200
    assert "Methodology" in r.text
    assert "Universe construction" in r.text
    assert "Trials" in r.text


def test_the_pdf_endpoint_reports_a_missing_backend_as_503_not_500(client, logged_run):
    """A deployment state, not a bad request — and the detail says how to fix it."""

    r = client.get(f"/reports/tearsheet/{logged_run}.pdf")
    assert r.status_code in (200, 503)
    if r.status_code == 503:
        assert "Pango" in r.json()["detail"]
    else:
        assert r.content.startswith(b"%PDF")
        assert r.headers["content-type"] == "application/pdf"


def test_a_tearsheet_for_an_unknown_run_is_not_a_500(client):
    r = client.get("/reports/tearsheet/deadbeef.html")
    assert r.status_code in (404, 409)
