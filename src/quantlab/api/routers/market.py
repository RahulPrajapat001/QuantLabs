"""OHLCV bars and the engineered feature panel.

``close`` and ``adj_close`` are served **separately** and both are labelled,
because they answer different questions: ``close`` is what the tape printed and
``adj_close`` is what a holder earned. Returning one under the other's name is
how a split becomes a 50% one-day loss in somebody's risk model.
"""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from quantlab.api.deps import SessionDep
from quantlab.api.formatting import format_date
from quantlab.api.models import DateRange, Num
from quantlab.db.models import OHLCVDaily

router = APIRouter(prefix="/market", tags=["market"])

#: A hard ceiling on a single response. The UI paginates or narrows its window;
#: it does not get to pull the whole database into a browser tab.
MAX_ROWS = 20_000


class Bar(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    date_display: str
    security_id: str
    open: Num
    high: Num
    low: Num
    #: The traded price, unadjusted.
    close: Num
    #: Adjusted for splits, bonuses and dividends. Use this for returns.
    adj_close: Num
    volume: Num


class BarSeries(BaseModel):
    model_config = ConfigDict(frozen=True)

    security_id: str | None
    window: DateRange
    count: Num
    truncated: bool = Field(
        description=f"True when the window exceeded {MAX_ROWS} rows and was cut."
    )
    bars: list[Bar]


class FeatureRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    date_display: str
    security_id: str
    #: Feature name to value. Missing features are absent rather than zero.
    values: dict[str, float | None]


class FeaturePanel(BaseModel):
    model_config = ConfigDict(frozen=True)

    features: list[str]
    window: DateRange
    count: Num
    truncated: bool
    rows: list[FeatureRow]


@router.get("/ohlcv", response_model=BarSeries, summary="Daily OHLCV bars.")
def ohlcv(
    session: SessionDep,
    security_id: str | None = Query(
        default=None, description="Restrict to one security. Omit for all."
    ),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    limit: int = Query(default=5_000, ge=1, le=MAX_ROWS),
) -> BarSeries:
    stmt = select(OHLCVDaily)
    if security_id is not None:
        stmt = stmt.where(OHLCVDaily.security_id == security_id)
    if start is not None:
        stmt = stmt.where(OHLCVDaily.date >= start)
    if end is not None:
        stmt = stmt.where(OHLCVDaily.date <= end)
    stmt = stmt.order_by(OHLCVDaily.date, OHLCVDaily.security_id).limit(limit + 1)

    rows = list(session.scalars(stmt))
    truncated = len(rows) > limit
    rows = rows[:limit]
    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no OHLCV rows match that query; ingest data first",
        )

    return BarSeries(
        security_id=security_id,
        window=DateRange.of(rows[0].date, rows[-1].date),
        count=Num.count(len(rows)),
        truncated=truncated,
        bars=[
            Bar(
                date=row.date,
                date_display=format_date(row.date),
                security_id=row.security_id,
                open=Num.currency(row.open, abbreviate=False),
                high=Num.currency(row.high, abbreviate=False),
                low=Num.currency(row.low, abbreviate=False),
                close=Num.currency(row.close, abbreviate=False),
                adj_close=Num.currency(row.adj_close, abbreviate=False),
                volume=Num.count(row.volume),
            )
            for row in rows
        ],
    )


@router.get(
    "/features",
    response_model=FeaturePanel,
    summary="The engineered feature panel, from partitioned parquet.",
)
def features(
    security_id: str | None = Query(default=None),
    names: str | None = Query(
        default=None, description="Comma-separated feature names. Omit for all."
    ),
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    limit: int = Query(default=5_000, ge=1, le=MAX_ROWS),
) -> FeaturePanel:
    import pandas as pd

    from quantlab.feature_engineering.build import read_feature_panel

    requested = [n.strip() for n in names.split(",")] if names else None
    try:
        panel = read_feature_panel(
            start=start,
            end=end,
            securities=[security_id] if security_id else None,
            columns=(["date", "security_id", *requested] if requested else None),
        )
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "the feature dataset has not been built. Run: "
                "quantlab features build --index NIFTY500"
            ),
        ) from exc

    if panel.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "no feature rows match that query. If the dataset has not been "
                "built yet, run: quantlab features build --index NIFTY500"
            ),
        )

    panel = panel.sort_values(["date", "security_id"])
    truncated = len(panel) > limit
    panel = panel.head(limit)
    feature_names = [
        c for c in panel.columns if c not in ("date", "security_id", "year")
    ]

    rows: list[FeatureRow] = []
    for record in panel.to_dict("records"):
        when = pd.Timestamp(record["date"]).date()
        rows.append(
            FeatureRow(
                date=when,
                date_display=format_date(when),
                security_id=str(record["security_id"]),
                values={name: _finite(record.get(name)) for name in feature_names},
            )
        )

    return FeaturePanel(
        features=feature_names,
        window=DateRange.of(rows[0].date, rows[-1].date),
        count=Num.count(len(rows)),
        truncated=truncated,
        rows=rows,
    )


def _finite(value: object) -> float | None:
    import math

    if value is None:
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
