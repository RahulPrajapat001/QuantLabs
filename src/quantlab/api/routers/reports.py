"""Tearsheet download (PRD Module 15).

Both representations are served: HTML for the browser and PDF for the artefact
that leaves the building. They render from the same
:class:`~quantlab.reports.tearsheet.TearsheetData` and the same template, so a
reader who prints the HTML and a reader who downloads the PDF are looking at the
same document.

The PDF endpoint returns **503**, not 500, when WeasyPrint's system libraries are
absent. That is a deployment state rather than a bug in the request, and the
detail says which libraries and how to install them — the HTML endpoint still
works meanwhile.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import HTMLResponse, Response

from quantlab.api.deps import SessionDep
from quantlab.reports.tearsheet import (
    MethodologyIncompleteError,
    TearsheetData,
    WeasyPrintUnavailableError,
    from_run,
    render_html,
    render_pdf,
)
from quantlab.research.log import ResearchLog

router = APIRouter(prefix="/reports", tags=["reports"])


def _build(session: SessionDep, run_id: str) -> TearsheetData:
    research_log = ResearchLog(session)
    try:
        return from_run(research_log, run_id)
    except MethodologyIncompleteError as exc:  # pragma: no cover - defensive
        # Refusing to serve is correct: a tearsheet without its methodology page
        # states a result while withholding what is needed to judge it.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
        ) from exc


@router.get(
    "/tearsheet/{run_id}.html",
    response_class=HTMLResponse,
    summary="Tearsheet as HTML.",
)
def tearsheet_html(run_id: str, session: SessionDep) -> HTMLResponse:
    return HTMLResponse(content=render_html(_build(session, run_id)))


@router.get(
    "/tearsheet/{run_id}.pdf",
    summary="Tearsheet as PDF, with its mandatory methodology page.",
    responses={
        200: {"content": {"application/pdf": {}}},
        503: {"description": "The PDF backend's system libraries are missing."},
    },
)
def tearsheet_pdf(
    run_id: str,
    session: SessionDep,
    download: bool = Query(
        default=True, description="Content-Disposition attachment vs inline."
    ),
) -> Response:
    data = _build(session, run_id)
    try:
        pdf = render_pdf(data)
    except WeasyPrintUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    disposition = "attachment" if download else "inline"
    filename = f"{data.strategy_name}-{run_id[:8]}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'{disposition}; filename="{filename}"',
        },
    )
