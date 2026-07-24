"""Ingesting index reconstitution history into ``index_membership``.

Two primary sources, one schema:

* **US (S&P 500)** — the "Selected changes to the list of S&P 500 components"
  table on Wikipedia's *List of S&P 500 companies* page. Used only to prove the
  reconstitution pipeline generalises; no research runs on US data.
* **India (NIFTY 50 / 100 / 500)** — NSE's semi-annual reconstitution press
  releases (effective March and September), distributed as PDFs.

Both are parsed into :class:`ReconstitutionEvent` objects, then folded into
half-open membership intervals (:mod:`quantlab.universe.membership`). Every raw
source is cached under ``settings.reference_dir`` so a result is reproduced from
the cache, never by re-downloading (PRD §3, Immutable raw landing; Module 2).

The ``fetch_*`` helpers do network + caching and are not exercised in CI. The
``parse_*`` helpers are pure functions over already-fetched text/HTML and are
unit-tested against small fixtures — the fragile part (vendor formats change) is
isolated and testable offline.

**Resolution.** Each event's ticker is resolved to a stable ``security_id`` *as
of its effective date* via :mod:`quantlab.data.identity`. An entry that cannot be
resolved is written to ``unresolved_membership_change`` rather than dropped, and
the resolution rate is reported (:class:`ResolutionReport`) — the PRD requires
this to be surfaced, not hidden (§Limitations: "Universe resolution below 100%").
"""

from __future__ import annotations

import datetime as _dt
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import structlog
from sqlalchemy.orm import Session

from quantlab.config import get_settings
from quantlab.data.identity import resolve_security_id
from quantlab.db.models import UnresolvedMembershipChange

from .membership import (
    MembershipInterval,
    build_intervals_from_events,
    insert_membership_intervals,
)

log = structlog.get_logger(__name__)

#: The canonical Wikipedia source for the S&P 500 changes table.
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

_ADD = "add"
_REMOVE = "remove"


@dataclass(frozen=True, slots=True)
class ReconstitutionEvent:
    """A single index addition or removal, before ``security_id`` resolution."""

    index_name: str
    action: str  #: ``"add"`` or ``"remove"``
    effective_date: date
    source_document: str
    ticker: str | None = None
    company_name: str | None = None


@dataclass
class ResolutionReport:
    """Outcome of ingesting reconstitution events for one index."""

    index_name: str
    total_events: int = 0
    resolved_events: int = 0
    unresolved_events: int = 0
    intervals_inserted: int = 0
    #: Human-readable unresolved entries, for logging / the CLI.
    unresolved: list[str] = field(default_factory=list)

    @property
    def resolution_rate(self) -> float:
        """Share of events mapped to a ``security_id`` (0.0 to 1.0)."""

        if self.total_events == 0:
            return 1.0
        return self.resolved_events / self.total_events

    def summary(self) -> str:
        pct = self.resolution_rate * 100
        return (
            f"{self.index_name}: {self.resolved_events}/{self.total_events} events "
            f"resolved ({pct:.1f}%), {self.intervals_inserted} intervals inserted, "
            f"{self.unresolved_events} unresolved"
        )


# --- Ticker normalisation ----------------------------------------------------


def _candidate_tickers(ticker: str, index_name: str) -> list[str]:
    """Vendor-ticker candidates to try when resolving a reconstitution symbol.

    NSE filings list bare symbols (``RELIANCE``) but ``ticker_history`` stores
    vendor symbols (``RELIANCE.NS``). US symbols carry no suffix. We try the bare
    symbol and, for NIFTY indices, the ``.NS`` form.
    """

    bare = ticker.strip().upper()
    candidates = [bare]
    if index_name.upper().startswith("NIFTY") and not bare.endswith(".NS"):
        candidates.append(f"{bare}.NS")
    return candidates


def resolve_event(session: Session, event: ReconstitutionEvent) -> str | None:
    """Resolve one event's ticker to a ``security_id`` as of its effective date."""

    if not event.ticker:
        return None
    for candidate in _candidate_tickers(event.ticker, event.index_name):
        security_id = resolve_security_id(session, candidate, event.effective_date)
        if security_id is not None:
            return security_id
    return None


# --- Ingest ------------------------------------------------------------------


def ingest_reconstitution(
    session: Session,
    index_name: str,
    events: list[ReconstitutionEvent],
) -> ResolutionReport:
    """Resolve, fold into intervals, and persist reconstitution events.

    Resolved events are grouped per security, folded into half-open intervals
    (:func:`build_intervals_from_events`) and inserted idempotently. Unresolved
    events land in ``unresolved_membership_change`` with the reason. The returned
    :class:`ResolutionReport` carries the resolution rate — a reported metric.
    """

    report = ResolutionReport(index_name=index_name, total_events=len(events))
    resolved: list[tuple[str, str, date, str]] = []

    for event in events:
        security_id = resolve_event(session, event)
        if security_id is None:
            report.unresolved_events += 1
            label = event.ticker or event.company_name or "?"
            report.unresolved.append(f"{event.action} {label} @ {event.effective_date}")
            session.add(
                UnresolvedMembershipChange(
                    index_name=event.index_name,
                    action=event.action,
                    effective_date=event.effective_date,
                    ticker=event.ticker,
                    company_name=event.company_name,
                    source_document=event.source_document,
                    reason=(
                        "no ticker_history match as of "
                        f"{event.effective_date.isoformat()}"
                    ),
                )
            )
            continue
        report.resolved_events += 1
        resolved.append(
            (security_id, event.action, event.effective_date, event.source_document)
        )

    # Fold per security across all this index's resolved events into intervals.
    by_security: dict[str, list[tuple[str, str, date, str]]] = defaultdict(list)
    for row in resolved:
        by_security[row[0]].append(row)

    intervals: list[MembershipInterval] = []
    for rows in by_security.values():
        intervals.extend(build_intervals_from_events(rows))

    report.intervals_inserted = insert_membership_intervals(
        session, index_name, intervals
    )
    session.flush()

    log.info(
        "universe.reconstitution.ingested",
        index_name=index_name,
        resolution_rate=round(report.resolution_rate, 4),
        resolved=report.resolved_events,
        unresolved=report.unresolved_events,
        intervals=report.intervals_inserted,
    )
    return report


# --- S&P 500 (Wikipedia) parsing ---------------------------------------------


def parse_sp500_changes(
    html: str, *, source_document: str, index_name: str = "SP500"
) -> list[ReconstitutionEvent]:
    """Parse the S&P 500 "changes" table from the Wikipedia page HTML.

    The changes table has a two-level header: ``Date`` and grouped ``Added`` /
    ``Removed`` columns each with ``Ticker`` and ``Security`` sub-columns. We
    locate it by those columns, then emit an ``add`` per non-empty Added ticker
    and a ``remove`` per non-empty Removed ticker.

    Kept pure (takes HTML text) so it is testable offline; :func:`fetch_sp500_changes`
    handles the network and caching.
    """

    import io

    import pandas as pd

    # pandas requires a file-like for literal HTML (a bare str is read as a path).
    tables = pd.read_html(io.StringIO(html))  # requires lxml (see pyproject data)
    events: list[ReconstitutionEvent] = []

    for table in tables:
        columns = [
            "_".join(str(part) for part in col).lower()
            if isinstance(col, tuple)
            else str(col).lower()
            for col in table.columns
        ]
        if not any("date" in c for c in columns):
            continue
        has_add = any("add" in c for c in columns)
        has_remove = any("remov" in c for c in columns)
        if not (has_add and has_remove):
            continue

        table = table.copy()
        table.columns = columns
        date_col = next(c for c in columns if "date" in c)
        add_ticker_col = _find_col(columns, "add", "ticker")
        rem_ticker_col = _find_col(columns, "remov", "ticker")
        add_name_col = _find_col(columns, "add", "security")
        rem_name_col = _find_col(columns, "remov", "security")

        for _, raw in table.iterrows():
            effective = _parse_us_date(str(raw.get(date_col, "")))
            if effective is None:
                continue
            if add_ticker_col:
                ticker = _clean_cell(raw.get(add_ticker_col))
                if ticker:
                    events.append(
                        ReconstitutionEvent(
                            index_name=index_name,
                            action=_ADD,
                            effective_date=effective,
                            source_document=source_document,
                            ticker=ticker,
                            company_name=_clean_cell(raw.get(add_name_col)),
                        )
                    )
            if rem_ticker_col:
                ticker = _clean_cell(raw.get(rem_ticker_col))
                if ticker:
                    events.append(
                        ReconstitutionEvent(
                            index_name=index_name,
                            action=_REMOVE,
                            effective_date=effective,
                            source_document=source_document,
                            ticker=ticker,
                            company_name=_clean_cell(raw.get(rem_name_col)),
                        )
                    )
        break  # first matching table is the changes table

    return events


def _find_col(columns: list[str], *needles: str) -> str | None:
    for c in columns:
        if all(n in c for n in needles):
            return c
    return None


def _clean_cell(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "—", "-"}:
        return None
    return text


def _parse_us_date(value: str) -> date | None:
    text = value.strip()
    if not text or text.lower() == "nan":
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y"):
        try:
            return _dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def fetch_sp500_changes(
    *, reference_dir: Path | None = None, refresh: bool = False
) -> tuple[str, str]:
    """Download the S&P 500 page and cache the raw HTML; return ``(html, source)``.

    The HTML is cached at ``{reference_dir}/sp500_changes_{YYYY-MM-DD}.html`` so
    later runs reproduce from disk. Pass ``refresh=True`` to force a re-download.
    """

    import requests

    settings = get_settings()
    ref_dir = reference_dir or settings.reference_dir
    ref_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.date.today().isoformat()
    cache_path = ref_dir / f"sp500_changes_{stamp}.html"

    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8"), cache_path.name

    response = requests.get(
        SP500_WIKI_URL, timeout=30, headers={"User-Agent": "QuantLab/0.1 (research)"}
    )
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    log.info("universe.reconstitution.cached", source=cache_path.name)
    return response.text, cache_path.name


# --- NSE (NIFTY) PDF parsing -------------------------------------------------

_NSE_INCLUDE_RE = re.compile(
    r"(securities?\s+to\s+be\s+included|inclusion[s]?|included)", re.IGNORECASE
)
_NSE_EXCLUDE_RE = re.compile(
    r"(securities?\s+to\s+be\s+excluded|exclusion[s]?|excluded)", re.IGNORECASE
)
#: A plausible NSE symbol token: 2 to 20 uppercase alphanumerics (e.g. ``M&MFIN``).
_SYMBOL_RE = re.compile(r"^[A-Z0-9&\-]{2,20}$")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from an NSE reconstitution PDF (isolates the pypdf dependency)."""

    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


#: A ``YYYY-MM-DD`` or ``YYYY_MM_DD`` date embedded in a cached filename.
_FILENAME_DATE_RE = re.compile(r"(\d{4})[-_](\d{2})[-_](\d{2})")


def collect_nifty_events(
    index_name: str, *, reference_dir: Path | None = None
) -> list[ReconstitutionEvent]:
    """Parse every cached NSE reconstitution PDF for ``index_name``.

    Scans ``reference_dir`` for ``*.pdf`` files whose name contains the index
    token (case-insensitive) and an effective date (``YYYY-MM-DD``). Each is read
    with :func:`extract_pdf_text` and parsed with :func:`parse_nifty_reconstitution`.
    Files without a parseable date in the name are skipped with a warning — the
    effective date drives the half-open intervals and must not be guessed.
    """

    settings = get_settings()
    ref_dir = reference_dir or settings.reference_dir
    if not ref_dir.exists():
        return []

    token = index_name.lower().replace(" ", "")
    events: list[ReconstitutionEvent] = []
    for pdf_path in sorted(ref_dir.glob("*.pdf")):
        name = pdf_path.name.lower().replace(" ", "")
        if token not in name:
            continue
        match = _FILENAME_DATE_RE.search(pdf_path.name)
        if match is None:
            log.warning("universe.reconstitution.no_date_in_name", file=pdf_path.name)
            continue
        effective = date(int(match[1]), int(match[2]), int(match[3]))
        text = extract_pdf_text(pdf_path.read_bytes())
        events.extend(
            parse_nifty_reconstitution(
                text,
                index_name=index_name,
                effective_date=effective,
                source_document=pdf_path.name,
            )
        )
    return events


def parse_nifty_reconstitution(
    text: str,
    *,
    index_name: str,
    effective_date: date,
    source_document: str,
) -> list[ReconstitutionEvent]:
    """Parse included / excluded symbols from an NSE press-release's text.

    NSE releases vary in layout; this reads line-oriented text (from
    :func:`extract_pdf_text`) and switches between an *include* and *exclude*
    bucket on the section headings, collecting symbol-like tokens. It is
    intentionally conservative: anything it cannot confidently read is simply not
    emitted, so it never invents a membership change. Symbols it does emit still
    pass through :func:`resolve_event`, and any that fail to resolve are recorded
    as unresolved — so a mis-parse degrades the resolution rate rather than
    corrupting membership silently.
    """

    events: list[ReconstitutionEvent] = []
    bucket: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if _NSE_INCLUDE_RE.search(stripped):
            bucket = _ADD
            continue
        if _NSE_EXCLUDE_RE.search(stripped):
            bucket = _REMOVE
            continue
        if bucket is None:
            continue

        for token in re.split(r"[\s,;|]+", stripped):
            symbol = token.strip().upper()
            if _SYMBOL_RE.match(symbol) and not symbol.isdigit():
                events.append(
                    ReconstitutionEvent(
                        index_name=index_name,
                        action=bucket,
                        effective_date=effective_date,
                        source_document=source_document,
                        ticker=symbol,
                    )
                )
    return events
