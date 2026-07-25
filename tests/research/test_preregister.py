"""Pre-registration: hypotheses parsed, hashed, and checked against the log."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quantlab.research.log import ResearchLog, RunResult
from quantlab.research.preregister import (
    PreRegistrationError,
    Threshold,
    load_hypotheses,
    parse_hypotheses,
    require_hypothesis,
    validate,
    validate_against_log,
)

from .conftest import make_spec, write_hypotheses

_H1 = """\
# Pre-registered hypotheses

Preamble prose that is not a hypothesis.

## H1 — 12-1 cross-sectional momentum
- **subject:** strategy `momentum_12_1` on the point-in-time NIFTY 500
- **rationale:** Cross-sectional momentum is the most widely replicated equity
  anomaly outside the United States, and thin analyst coverage below the NIFTY 100
  plausibly strengthens the under-reaction mechanism here.
- **prediction:** the long-short decile spread earns a positive Sharpe net of costs
- **metric:** annualised Sharpe net of IndianEquityCostModel on the validation split
- **threshold:** >= 0.50 net of costs
- **registered:** 2026-07-25
- **status:** registered
"""


def test_parses_fields_and_multiline_continuations() -> None:
    (hypothesis,) = parse_hypotheses(_H1)
    assert hypothesis.id == "H1"
    assert hypothesis.title == "12-1 cross-sectional momentum"
    assert "momentum_12_1" in hypothesis.subject
    # The rationale spans three source lines and must arrive as one paragraph.
    assert "United States" in hypothesis.rationale
    assert "under-reaction" in hypothesis.rationale
    assert "\n" not in hypothesis.rationale
    assert hypothesis.threshold == Threshold(">=", 0.5, "net of costs")
    assert hypothesis.registered_on == date(2026, 7, 25)
    assert hypothesis.status == "registered"


def test_threshold_is_machine_checkable() -> None:
    threshold = Threshold.parse(">= 0.50 net of costs")
    assert threshold.passes(0.5)
    assert threshold.passes(0.9)
    assert not threshold.passes(0.49)
    assert Threshold.parse("< 0.20").passes(0.1)
    assert not Threshold.parse("< 0.20").passes(0.2)
    assert Threshold.parse("> -0.05").passes(0.0)


def test_prose_threshold_is_rejected() -> None:
    with pytest.raises(PreRegistrationError, match="not machine-checkable"):
        Threshold.parse("Sharpe should be good")


def test_missing_required_field_raises(tmp_path: Path) -> None:
    body = """\
## H9 — Incomplete
- **subject:** strategy `x`
- **prediction:** it works
- **metric:** Sharpe on validation
- **threshold:** >= 1.0
"""
    path = write_hypotheses(tmp_path, body)
    with pytest.raises(
        PreRegistrationError, match=r"missing required field.*rationale"
    ):
        load_hypotheses(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = write_hypotheses(
        tmp_path, _H1 + _H1.split("# Pre-registered hypotheses")[-1]
    )
    with pytest.raises(PreRegistrationError, match="duplicate hypothesis id"):
        load_hypotheses(path)


def test_hash_is_stable_under_reflow_but_not_under_edits(tmp_path: Path) -> None:
    original = load_hypotheses(write_hypotheses(tmp_path / "a", _H1))["H1"]

    reflowed = _H1.replace(
        "anomaly outside the United States, and thin analyst coverage"
        " below the NIFTY 100\n  plausibly",
        "anomaly outside the United States,\n  and thin analyst coverage"
        " below the NIFTY 100 plausibly",
    )
    assert reflowed != _H1
    assert (
        load_hypotheses(write_hypotheses(tmp_path / "b", reflowed))["H1"].content_hash
        == original.content_hash
    )

    # Loosening the threshold after the fact must change the hash — this is the
    # check that catches a bar moved to fit a disappointing result.
    loosened = _H1.replace("**threshold:** >= 0.50", "**threshold:** >= 0.20")
    assert (
        load_hypotheses(write_hypotheses(tmp_path / "c", loosened))["H1"].content_hash
        != original.content_hash
    )


def test_require_hypothesis_rejects_unknown_and_withdrawn(tmp_path: Path) -> None:
    path = write_hypotheses(tmp_path, _H1)
    assert require_hypothesis("H1", path=path).id == "H1"
    with pytest.raises(PreRegistrationError, match="is not registered"):
        require_hypothesis("H99", path=path)

    withdrawn = write_hypotheses(
        tmp_path / "w", _H1.replace("**status:** registered", "**status:** withdrawn")
    )
    with pytest.raises(PreRegistrationError, match="withdrawn"):
        require_hypothesis("H1", path=withdrawn)


def test_metric_without_a_split_is_an_error(tmp_path: Path) -> None:
    body = _H1.replace(
        "- **metric:** annualised Sharpe net of IndianEquityCostModel"
        " on the validation split",
        "- **metric:** annualised Sharpe net of costs",
    )
    report = validate(path=write_hypotheses(tmp_path, body), check_git=False)
    assert not report.ok
    assert any("names no split" in issue.message for issue in report.errors)


def test_uncommitted_hypothesis_is_a_warning_not_an_error(tmp_path: Path) -> None:
    # A file outside the repository cannot be found in git history, which is
    # exactly the "written but not yet committed" state.
    report = validate(path=write_hypotheses(tmp_path, _H1), check_git=True)
    assert report.ok
    assert any("not yet committed" in issue.message for issue in report.warnings)


def test_project_hypotheses_document_is_valid() -> None:
    """docs/hypotheses.md itself must pass validation."""

    report = validate(check_git=False)
    assert report.hypotheses, "the project has no pre-registered hypotheses"
    assert report.ok, "\n".join(issue.render() for issue in report.errors)
    for hypothesis in report.hypotheses:
        assert hypothesis.registered_on is not None
        assert len(hypothesis.rationale) > 100


def test_project_hypotheses_are_committed_to_git() -> None:
    """Registration must be verifiable in git history, not merely asserted.

    Skipped while the document is still uncommitted in a working tree — the
    warning path is covered above; here the point is that once committed, the
    introducing commit is discoverable.
    """

    from quantlab.research.preregister import registration_commit

    hypotheses = load_hypotheses()
    records = {h.id: registration_commit(h) for h in hypotheses.values()}
    if all(record is None for record in records.values()):
        pytest.skip("docs/hypotheses.md is not committed yet")
    for hypothesis_id, record in records.items():
        assert record is not None, f"{hypothesis_id} is not in any commit of the file"
        assert len(record.commit_hash) == 40


def test_log_links_a_run_to_a_hypothesis(
    research_log: ResearchLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_hypotheses(tmp_path, _H1)
    monkeypatch.setenv("QUANTLAB_HYPOTHESES_FILE", str(path))
    from quantlab.config import get_settings

    get_settings.cache_clear()
    try:
        with research_log.run(make_spec("momentum_12_1", hypothesis_id="H1")) as run:
            run.record(RunResult.from_returns([0.01, -0.005, 0.02]))
        entry = research_log.entry(run.run_id)
        assert entry.hypothesis_id == "H1"
        assert entry.exploratory is False
        assert entry.hypothesis_hash == load_hypotheses(path)["H1"].content_hash
        assert validate_against_log([entry], path=path) == []

        # Edit the hypothesis after the run: the mismatch must be reported.
        edited = write_hypotheses(
            tmp_path / "edited",
            _H1.replace("**threshold:** >= 0.50", "**threshold:** >= 0.10"),
        )
        issues = validate_against_log([entry], path=edited)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "was edited after run" in issues[0].message
    finally:
        get_settings.cache_clear()


def test_run_claiming_an_unregistered_hypothesis_is_refused(
    research_log: ResearchLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QUANTLAB_HYPOTHESES_FILE", str(write_hypotheses(tmp_path, _H1)))
    from quantlab.config import get_settings

    get_settings.cache_clear()
    try:
        with (
            pytest.raises(PreRegistrationError, match="is not registered"),
            research_log.run(make_spec("x", hypothesis_id="H42")),
        ):
            pass
        assert research_log.entries(status=None) == []
    finally:
        get_settings.cache_clear()


def test_a_run_without_a_hypothesis_is_exploratory(research_log: ResearchLog) -> None:
    with research_log.run(make_spec("hunch")) as run:
        run.record(RunResult.from_returns([0.01, 0.02, -0.01]))
    entry = research_log.entry(run.run_id)
    assert entry.exploratory is True
    assert entry.hypothesis_id is None
    assert entry.hypothesis_hash is None
