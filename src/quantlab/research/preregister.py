"""Pre-registration: hypotheses written down before they are tested.

A prediction made after seeing the result is not a prediction. ``docs/hypotheses.md``
holds one entry per hypothesis — id, economic rationale, the factor or strategy,
the prediction, the metric, and the threshold — and each entry is **hashed**. The
hash travels into ``research_log.hypothesis_hash`` at run time, so three things
become checkable rather than trusted:

* A backtest linked to ``H1`` really was a test of what ``H1`` said, because the
  hash recorded with the run still matches the text of ``H1`` today. Quietly
  loosening a threshold after a disappointing result changes the hash and the
  mismatch is detectable (:func:`validate_against_log`).
* The hypothesis was committed to git **before** the run that tested it, because
  git history says which commit introduced it (:func:`registration_commit`).
* A run with no ``hypothesis_id`` is marked ``exploratory`` in the log. That is
  not a failure — most research is exploratory — but it is labelled, and an
  exploratory result never gets presented as a confirmed prediction.

The threshold is parsed into an operator and a number rather than left as prose,
so :meth:`Threshold.passes` can decide the outcome in code. "Sharpe should be
good" is not a pre-registered threshold.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import structlog

from quantlab.config import get_settings
from quantlab.utils.git import relative_to_repo, repo_root, run_git

log = structlog.get_logger(__name__)


class PreRegistrationError(RuntimeError):
    """Raised when a hypothesis is missing, malformed, or has been altered."""


#: ``## H1 — Title``, with an em dash, en dash, hyphen or colon separator.
_SECTION_RE = re.compile(r"^##\s+(?P<id>H\d+[a-z]?)\s*(?:[—–:-]\s*)(?P<title>.+?)\s*$")
#: ``- **key:** value``
_FIELD_RE = re.compile(
    r"^[-*]\s+\*\*(?P<key>[A-Za-z_ ]+?)\s*:\*\*\s*(?P<value>.*?)\s*$"
)
#: ``>= 0.50``, ``< -0.02``, ``>= 0.5 (net of costs)``
_THRESHOLD_RE = re.compile(
    r"^(?P<op>>=|<=|>|<)\s*(?P<value>[-+]?\d+(?:\.\d+)?)\s*(?P<qualifier>.*)$"
)

_REQUIRED_FIELDS = ("subject", "rationale", "prediction", "metric", "threshold")
_VALID_STATUSES = ("registered", "tested", "withdrawn")


@dataclass(frozen=True, slots=True)
class Threshold:
    """A machine-checkable pass/fail bar, e.g. ``>= 0.50``."""

    operator: str
    value: float
    #: Free text after the number, e.g. ``"net of Indian equity costs"``.
    qualifier: str = ""

    @classmethod
    def parse(cls, text: str) -> Threshold:
        match = _THRESHOLD_RE.match(text.strip())
        if match is None:
            raise PreRegistrationError(
                f"threshold {text!r} is not machine-checkable. Write it as an "
                "operator and a number, e.g. '>= 0.50 net of costs' — prose "
                "thresholds cannot decide anything."
            )
        return cls(
            operator=match.group("op"),
            value=float(match.group("value")),
            qualifier=match.group("qualifier").strip(),
        )

    def passes(self, observed: float) -> bool:
        """Whether ``observed`` clears this threshold."""

        if self.operator == ">=":
            return observed >= self.value
        if self.operator == ">":
            return observed > self.value
        if self.operator == "<=":
            return observed <= self.value
        return observed < self.value

    def render(self) -> str:
        base = f"{self.operator} {self.value:g}"
        return f"{base} {self.qualifier}".strip()


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One pre-registered hypothesis from ``docs/hypotheses.md``."""

    id: str
    title: str
    #: The factor or strategy under test, e.g. ``"strategy momentum_12_1"``.
    subject: str
    #: *Why* this should work, in economic terms. A hypothesis without a
    #: mechanism is a data-mining result waiting to happen.
    rationale: str
    prediction: str
    metric: str
    threshold: Threshold
    registered_on: date | None = None
    status: str = "registered"
    notes: str | None = None
    #: 1-based line of the ``##`` heading, for error messages.
    line: int = 0

    @property
    def content_hash(self) -> str:
        """SHA-256 over the substantive content of this hypothesis.

        Whitespace is normalised (reflowing a paragraph is not an edit); anything
        else — a reworded prediction, a loosened threshold, a changed metric —
        changes the hash. ``status`` and ``notes`` are excluded so that marking a
        hypothesis ``tested`` afterwards does not invalidate the runs that tested
        it.
        """

        canonical = "\n".join(
            (
                f"id={self.id}",
                f"title={_normalise(self.title)}",
                f"subject={_normalise(self.subject)}",
                f"rationale={_normalise(self.rationale)}",
                f"prediction={_normalise(self.prediction)}",
                f"metric={_normalise(self.metric)}",
                f"threshold={self.threshold.operator}{self.threshold.value:g}"
                f" {_normalise(self.threshold.qualifier)}".rstrip(),
            )
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def render(self) -> str:
        return (
            f"{self.id} [{self.status}] {self.title}\n"
            f"  subject:    {self.subject}\n"
            f"  metric:     {self.metric}\n"
            f"  threshold:  {self.threshold.render()}\n"
            f"  prediction: {self.prediction}\n"
            f"  hash:       {self.content_hash[:16]}…"
        )


def _normalise(text: str) -> str:
    """Collapse whitespace so reflowing a paragraph does not change the hash."""

    return re.sub(r"\s+", " ", text).strip()


# --- Locating and parsing ----------------------------------------------------


def hypotheses_path(path: Path | None = None) -> Path:
    """Resolve the hypotheses file (configurable; never hardcoded at a call site).

    Order: explicit argument, then ``QUANTLAB_HYPOTHESES_FILE`` as given (if
    absolute), then relative to the working directory, then relative to the git
    repository root — so the CLI works from any subdirectory of the project.
    """

    candidate = path or get_settings().hypotheses_file
    if candidate.is_absolute():
        return candidate
    cwd_relative = Path.cwd() / candidate
    if cwd_relative.exists():
        return cwd_relative
    root = repo_root()
    if root is not None:
        return root / candidate
    return cwd_relative


def parse_hypotheses(text: str) -> list[Hypothesis]:
    """Parse the markdown pre-registration document.

    Format, per hypothesis::

        ## H1 — 12-1 cross-sectional momentum
        - **subject:** strategy `momentum_12_1`
        - **rationale:** ...
        - **prediction:** ...
        - **metric:** annualised Sharpe, net of costs, validation split
        - **threshold:** >= 0.50
        - **registered:** 2026-07-25

    Field values may continue onto indented following lines. Anything outside a
    ``## H<n>`` section is prose and ignored.
    """

    hypotheses: list[Hypothesis] = []
    current_id: str | None = None
    current_title = ""
    current_line = 0
    fields: dict[str, str] = {}
    last_key: str | None = None

    def flush() -> None:
        nonlocal current_id, last_key
        if current_id is None:
            return
        hypotheses.append(_build(current_id, current_title, current_line, fields))
        current_id = None
        last_key = None

    for lineno, raw in enumerate(text.splitlines(), start=1):
        section = _SECTION_RE.match(raw)
        if section is not None:
            flush()
            current_id = section.group("id")
            current_title = section.group("title")
            current_line = lineno
            fields = {}
            last_key = None
            continue
        if current_id is None:
            continue
        if raw.startswith("## ") or raw.startswith("# "):
            # A non-hypothesis heading ends the current section.
            flush()
            continue
        field_match = _FIELD_RE.match(raw)
        if field_match is not None:
            key = field_match.group("key").strip().lower().replace(" ", "_")
            fields[key] = field_match.group("value")
            last_key = key
            continue
        if last_key is not None and raw.strip() and raw[:1].isspace():
            fields[last_key] = f"{fields[last_key]} {raw.strip()}".strip()

    flush()
    return hypotheses


def _build(
    hypothesis_id: str, title: str, line: int, fields: Mapping[str, str]
) -> Hypothesis:
    missing = [key for key in _REQUIRED_FIELDS if not fields.get(key, "").strip()]
    if missing:
        raise PreRegistrationError(
            f"hypothesis {hypothesis_id} (line {line}) is missing required "
            f"field(s): {', '.join(missing)}. A pre-registration without a "
            "rationale, prediction, metric and threshold pre-registers nothing."
        )
    registered_raw = fields.get("registered", "").strip()
    registered_on: date | None = None
    if registered_raw:
        try:
            registered_on = datetime.strptime(registered_raw, "%Y-%m-%d").date()
        except ValueError as exc:
            raise PreRegistrationError(
                f"hypothesis {hypothesis_id}: 'registered' must be YYYY-MM-DD, "
                f"got {registered_raw!r}"
            ) from exc
    status = (fields.get("status", "registered").strip() or "registered").lower()
    if status not in _VALID_STATUSES:
        raise PreRegistrationError(
            f"hypothesis {hypothesis_id}: status {status!r} is not one of "
            f"{', '.join(_VALID_STATUSES)}"
        )
    return Hypothesis(
        id=hypothesis_id,
        title=title,
        subject=fields["subject"].strip(),
        rationale=fields["rationale"].strip(),
        prediction=fields["prediction"].strip(),
        metric=fields["metric"].strip(),
        threshold=Threshold.parse(fields["threshold"]),
        registered_on=registered_on,
        status=status,
        notes=(fields.get("notes") or "").strip() or None,
        line=line,
    )


def load_hypotheses(path: Path | None = None) -> dict[str, Hypothesis]:
    """Load every registered hypothesis, keyed by id.

    Not cached: the file is small, and a stale cache is exactly the wrong
    behaviour for a document whose whole job is to be current.
    """

    resolved = hypotheses_path(path)
    if not resolved.exists():
        raise PreRegistrationError(
            f"no pre-registration document at {resolved}. Hypotheses are written "
            "before they are tested; create the file first."
        )
    hypotheses = parse_hypotheses(resolved.read_text(encoding="utf-8"))
    by_id: dict[str, Hypothesis] = {}
    for hypothesis in hypotheses:
        if hypothesis.id in by_id:
            raise PreRegistrationError(
                f"duplicate hypothesis id {hypothesis.id!r} in {resolved} "
                f"(lines {by_id[hypothesis.id].line} and {hypothesis.line}); ids "
                "must be unique for a run to link to one unambiguously"
            )
        by_id[hypothesis.id] = hypothesis
    return by_id


def require_hypothesis(hypothesis_id: str, *, path: Path | None = None) -> Hypothesis:
    """Return the hypothesis with this id, raising if it is not registered.

    Called by the research log when a run claims a ``hypothesis_id``: a run
    cannot link to a hypothesis nobody wrote down.
    """

    hypotheses = load_hypotheses(path)
    hypothesis = hypotheses.get(hypothesis_id)
    if hypothesis is None:
        known = ", ".join(sorted(hypotheses)) or "none"
        raise PreRegistrationError(
            f"hypothesis {hypothesis_id!r} is not registered in "
            f"{hypotheses_path(path)} (registered: {known}). Pre-register it "
            "before running the test, or run without a hypothesis_id and the run "
            "will be logged as exploratory."
        )
    if hypothesis.status == "withdrawn":
        raise PreRegistrationError(
            f"hypothesis {hypothesis_id} is withdrawn; a run cannot claim it"
        )
    return hypothesis


# --- Git-verifiable registration ---------------------------------------------


@dataclass(frozen=True, slots=True)
class RegistrationRecord:
    """The commit that introduced a hypothesis — proof of when it was registered."""

    hypothesis_id: str
    commit_hash: str
    committed_at: str

    def describe(self) -> str:
        return f"{self.commit_hash[:8]} @ {self.committed_at}"


def registration_commit(
    hypothesis: Hypothesis, *, path: Path | None = None
) -> RegistrationRecord | None:
    """Find the earliest commit whose ``hypotheses.md`` contains this exact hypothesis.

    Walks the file's history oldest-first and re-parses each revision, matching on
    :attr:`Hypothesis.content_hash`. Returns ``None`` if the hypothesis is not yet
    committed (or git is unavailable) — which means it is *not yet* pre-registered
    in any auditable sense, and :func:`validate` says so.
    """

    resolved = hypotheses_path(path)
    rel = relative_to_repo(resolved)
    if rel is None:
        return None
    history = run_git(["log", "--reverse", "--format=%H|%cI", "--", rel])
    if not history:
        return None
    target = hypothesis.content_hash
    for line in history.splitlines():
        commit_hash, _, committed_at = line.partition("|")
        blob = run_git(["show", f"{commit_hash}:{rel}"])
        if not blob:
            continue
        try:
            revision = parse_hypotheses(blob)
        except PreRegistrationError:
            # An older revision may predate the current format; skip it rather
            # than fail the whole audit.
            continue
        if any(h.content_hash == target for h in revision):
            return RegistrationRecord(
                hypothesis_id=hypothesis.id,
                commit_hash=commit_hash,
                committed_at=committed_at,
            )
    return None


# --- Validation --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """One problem found by :func:`validate`."""

    severity: str  # "error" | "warning"
    message: str
    hypothesis_id: str | None = None

    def render(self) -> str:
        where = f"[{self.hypothesis_id}] " if self.hypothesis_id else ""
        return f"{self.severity.upper():7s} {where}{self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Outcome of ``quantlab research prereg --validate``."""

    path: Path
    hypotheses: Sequence[Hypothesis]
    issues: Sequence[ValidationIssue]

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """Whether the document is valid. Warnings do not make it invalid."""

        return not self.errors

    def render(self) -> str:
        lines = [
            f"Pre-registration: {self.path}",
            f"{len(self.hypotheses)} hypotheses registered",
            "",
        ]
        for hypothesis in self.hypotheses:
            lines.append(hypothesis.render())
            lines.append("")
        if self.issues:
            lines.append("Findings:")
            lines.extend(f"  {issue.render()}" for issue in self.issues)
        else:
            lines.append("No findings: every hypothesis is complete, uniquely")
            lines.append("identified, machine-checkable, and committed to git.")
        lines.append("")
        lines.append(
            f"{'VALID' if self.ok else 'INVALID'} — {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        )
        return "\n".join(lines)


def validate(*, path: Path | None = None, check_git: bool = True) -> ValidationReport:
    """Validate the pre-registration document.

    Errors: unparseable file, duplicate ids, missing required fields, prose
    thresholds, a metric that names no split. Warnings: a hypothesis not yet
    committed to git (so its registration date is unverifiable), and a
    ``registered`` date later than the commit that introduced it.
    """

    resolved = hypotheses_path(path)
    issues: list[ValidationIssue] = []
    try:
        by_id = load_hypotheses(path)
    except PreRegistrationError as exc:
        return ValidationReport(
            path=resolved,
            hypotheses=(),
            issues=(ValidationIssue(severity="error", message=str(exc)),),
        )

    hypotheses = sorted(by_id.values(), key=lambda h: h.id)
    for hypothesis in hypotheses:
        if len(hypothesis.rationale) < 40:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    hypothesis_id=hypothesis.id,
                    message=(
                        "the economic rationale is one line long. A mechanism that "
                        "cannot be stated in a sentence or two is usually absent."
                    ),
                )
            )
        if not re.search(r"train|validation|holdout", hypothesis.metric, re.IGNORECASE):
            issues.append(
                ValidationIssue(
                    severity="error",
                    hypothesis_id=hypothesis.id,
                    message=(
                        f"metric {hypothesis.metric!r} names no split. State which "
                        "of train / validation / holdout the threshold applies to, "
                        "or the test can be re-run on whichever split passes."
                    ),
                )
            )
        if hypothesis.registered_on is None:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    hypothesis_id=hypothesis.id,
                    message="no 'registered' date given",
                )
            )
        if not check_git:
            continue
        record = registration_commit(hypothesis)
        if record is None:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    hypothesis_id=hypothesis.id,
                    message=(
                        "not found in any commit of this file: it is written down "
                        "but not yet committed, so its registration date is not "
                        "auditable. Commit it before running the test."
                    ),
                )
            )
        elif (
            hypothesis.registered_on is not None
            and record.committed_at[:10] < hypothesis.registered_on.isoformat()
        ):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    hypothesis_id=hypothesis.id,
                    message=(
                        f"declares registration on {hypothesis.registered_on} but "
                        f"was committed earlier ({record.describe()})"
                    ),
                )
            )
    return ValidationReport(path=resolved, hypotheses=hypotheses, issues=issues)


def validate_against_log(
    entries: Iterable[object], *, path: Path | None = None
) -> list[ValidationIssue]:
    """Check logged runs against the hypotheses they claim to test.

    Takes :class:`~quantlab.db.models.ResearchLogEntry` rows (typed loosely to
    keep this module free of a database dependency). Flags a run whose recorded
    ``hypothesis_hash`` no longer matches the hypothesis text — i.e. the
    hypothesis was edited after the run that tested it. That is the single most
    valuable check here: it catches a threshold quietly relaxed to fit a result.
    """

    issues: list[ValidationIssue] = []
    try:
        by_id = load_hypotheses(path)
    except PreRegistrationError as exc:
        return [ValidationIssue(severity="error", message=str(exc))]

    for entry in entries:
        hypothesis_id = getattr(entry, "hypothesis_id", None)
        if hypothesis_id is None:
            continue
        run_id = str(getattr(entry, "run_id", "?"))
        recorded_hash = getattr(entry, "hypothesis_hash", None)
        hypothesis = by_id.get(hypothesis_id)
        if hypothesis is None:
            issues.append(
                ValidationIssue(
                    severity="error",
                    hypothesis_id=hypothesis_id,
                    message=(
                        f"run {run_id[:8]} claims hypothesis {hypothesis_id}, which "
                        "is no longer in the document. A tested hypothesis is never "
                        "deleted — mark it 'withdrawn' instead."
                    ),
                )
            )
            continue
        if recorded_hash and recorded_hash != hypothesis.content_hash:
            issues.append(
                ValidationIssue(
                    severity="error",
                    hypothesis_id=hypothesis_id,
                    message=(
                        f"hypothesis {hypothesis_id} was edited after run "
                        f"{run_id[:8]} tested it (recorded hash "
                        f"{recorded_hash[:12]}…, current "
                        f"{hypothesis.content_hash[:12]}…). "
                        "The run did not test what the document now claims."
                    ),
                )
            )
    return issues
