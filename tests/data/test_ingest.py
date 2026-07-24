"""End-to-end ingestion: idempotency, rename resolution, delisting, incremental."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session, sessionmaker

from quantlab.data.base import CorporateActionType, DelistingInfo
from quantlab.data.errors import PermanentProviderError
from quantlab.data.ingest import (
    ingest_incremental,
    ingest_ticker,
    last_successful_end,
)
from quantlab.db.models import (
    CorporateAction,
    IngestLog,
    OHLCVDaily,
    TickerHistory,
)
from quantlab.db.models import Security as SecurityRow

from .conftest import FakeProvider, make_ohlcv


def _count(session: Session, model: type) -> int:
    return session.query(model).count()


def test_ingest_writes_rows_and_logs(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    provider = FakeProvider(
        ohlcv={"ABC.NS": make_ohlcv(["2020-01-01", "2020-01-02", "2020-01-03"])}
    )
    with session_factory() as session:
        result = ingest_ticker(
            session,
            provider,
            "ABC.NS",
            date(2020, 1, 1),
            date(2020, 1, 3),
            raw_dir=tmp_path,
        )
        session.commit()

        assert result.rows_written == 3
        assert result.status == "success"
        assert _count(session, OHLCVDaily) == 3
        assert _count(session, IngestLog) == 1
        # Raw parquet landed too.
        assert (tmp_path / "fake" / "ABC.NS" / "2020.parquet").exists()


def test_ingest_twice_is_idempotent(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    provider = FakeProvider(
        ohlcv={"ABC.NS": make_ohlcv(["2020-01-01", "2020-01-02", "2020-01-03"])}
    )

    def run() -> None:
        with session_factory() as session:
            ingest_ticker(
                session,
                provider,
                "ABC.NS",
                date(2020, 1, 1),
                date(2020, 1, 3),
                raw_dir=tmp_path,
            )
            session.commit()

    run()
    with session_factory() as session:
        first_rows = _count(session, OHLCVDaily)
        first_secs = _count(session, SecurityRow)

    run()  # ingest the identical range again
    with session_factory() as session:
        assert _count(session, OHLCVDaily) == first_rows == 3
        assert _count(session, SecurityRow) == first_secs == 1
        # Second ingest wrote no new price rows.
        logs = session.query(IngestLog).order_by(IngestLog.id).all()
        assert [log.rows_ingested for log in logs] == [3, 0]


def test_rename_resolves_to_one_security_id_across_ingests(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    # Pre-seed the rename: OLD.NS (—2019-06-30) then NEW.NS (2019-07-01—).
    with session_factory() as session:
        session.add(SecurityRow(security_id="SEC-acme", name="Acme", exchange="NSE"))
        session.add_all(
            [
                TickerHistory(
                    security_id="SEC-acme",
                    ticker="OLD.NS",
                    start_date=date(2015, 1, 1),
                    end_date=date(2019, 6, 30),
                    source="fake",
                ),
                TickerHistory(
                    security_id="SEC-acme",
                    ticker="NEW.NS",
                    start_date=date(2019, 7, 1),
                    end_date=None,
                    source="fake",
                ),
            ]
        )
        session.commit()

    provider = FakeProvider(
        ohlcv={
            "OLD.NS": make_ohlcv(["2018-01-01", "2018-01-02"]),
            "NEW.NS": make_ohlcv(["2020-01-01", "2020-01-02"]),
        }
    )

    # Ingest the old ticker for its historical window ...
    with session_factory() as session:
        r_old = ingest_ticker(
            session,
            provider,
            "OLD.NS",
            date(2018, 1, 1),
            date(2018, 1, 2),
            raw_dir=tmp_path,
            as_of_date=date(2018, 1, 2),
        )
        session.commit()
    # ... and the new ticker for its later window.
    with session_factory() as session:
        r_new = ingest_ticker(
            session,
            provider,
            "NEW.NS",
            date(2020, 1, 1),
            date(2020, 1, 2),
            raw_dir=tmp_path,
            as_of_date=date(2020, 1, 2),
        )
        session.commit()

    assert r_old.security_id == r_new.security_id == "SEC-acme"

    with session_factory() as session:
        # Exactly one security; all four bars carry the single security_id.
        assert _count(session, SecurityRow) == 1
        sec_ids = {row.security_id for row in session.query(OHLCVDaily).all()}
        assert sec_ids == {"SEC-acme"}
        assert _count(session, OHLCVDaily) == 4


def test_corporate_actions_ingested(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    actions = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2020-01-02"),
                "action_type": CorporateActionType.SPLIT.value,
                "value": 2.0,
            },
            {
                "date": pd.Timestamp("2020-01-03"),
                "action_type": CorporateActionType.DIVIDEND.value,
                "value": 1.5,
            },
        ]
    )
    provider = FakeProvider(
        ohlcv={"ABC.NS": make_ohlcv(["2020-01-01", "2020-01-02", "2020-01-03"])},
        actions={"ABC.NS": actions},
    )
    with session_factory() as session:
        ingest_ticker(
            session,
            provider,
            "ABC.NS",
            date(2020, 1, 1),
            date(2020, 1, 3),
            raw_dir=tmp_path,
        )
        session.commit()
        assert _count(session, CorporateAction) == 2
        # Idempotent on re-ingest.
    with session_factory() as session:
        ingest_ticker(
            session,
            provider,
            "ABC.NS",
            date(2020, 1, 1),
            date(2020, 1, 3),
            raw_dir=tmp_path,
        )
        session.commit()
        assert _count(session, CorporateAction) == 2


def test_delisted_security_retains_history(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    provider = FakeProvider(
        ohlcv={"DEAD.NS": make_ohlcv(["2020-01-01", "2020-01-02"])},
        delisting={
            "DEAD.NS": DelistingInfo(
                delisted=True,
                delisting_date=date(2020, 1, 2),
                reason="bankruptcy",
            )
        },
    )
    with session_factory() as session:
        result = ingest_ticker(
            session,
            provider,
            "DEAD.NS",
            date(2020, 1, 1),
            date(2020, 1, 2),
            raw_dir=tmp_path,
        )
        session.commit()

        sec = session.get(SecurityRow, result.security_id)
        assert sec is not None
        assert sec.delisting_date == date(2020, 1, 2)
        assert sec.delisting_reason == "bankruptcy"
        # History is preserved, never dropped.
        assert _count(session, OHLCVDaily) == 2


def test_provider_failure_logs_and_raises(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    class FailingProvider(FakeProvider):
        def _fetch_ohlcv(self, security, start, end, frequency):  # type: ignore[no-untyped-def]
            raise PermanentProviderError(
                "unknown symbol", source=self.source, symbol=security.symbol
            )

    provider = FailingProvider()
    with session_factory() as session:
        try:
            ingest_ticker(
                session,
                provider,
                "BAD.NS",
                date(2020, 1, 1),
                date(2020, 1, 2),
                raw_dir=tmp_path,
            )
            raised = False
        except PermanentProviderError:
            raised = True
        session.commit()

    assert raised
    with session_factory() as session:
        logs = session.query(IngestLog).all()
        assert len(logs) == 1
        assert logs[0].status == "failed"
        assert logs[0].error is not None


def test_incremental_resumes_from_last_success(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    provider = FakeProvider(
        ohlcv={
            "ABC.NS": make_ohlcv(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            )
        }
    )
    # First ingest covers only the first two days.
    with session_factory() as session:
        ingest_ticker(
            session,
            provider,
            "ABC.NS",
            date(2020, 1, 1),
            date(2020, 1, 2),
            raw_dir=tmp_path,
        )
        session.commit()

    # Incremental should resume at 2020-01-03 and pull the remaining bars.
    with session_factory() as session:
        results = ingest_incremental(
            session, provider, until=date(2020, 1, 4), raw_dir=tmp_path
        )
        session.commit()

    assert len(results) == 1
    assert results[0].start == date(2020, 1, 3)
    assert results[0].rows_written == 2

    with session_factory() as session:
        assert _count(session, OHLCVDaily) == 4
        assert last_successful_end(
            session, results[0].security_id, provider.source
        ) == date(2020, 1, 4)


def test_incremental_skips_current_securities(
    session_factory: sessionmaker[Session], tmp_path: Path
) -> None:
    provider = FakeProvider(ohlcv={"ABC.NS": make_ohlcv(["2020-01-01", "2020-01-02"])})
    with session_factory() as session:
        ingest_ticker(
            session,
            provider,
            "ABC.NS",
            date(2020, 1, 1),
            date(2020, 1, 2),
            raw_dir=tmp_path,
        )
        session.commit()

    with session_factory() as session:
        results = ingest_incremental(
            session, provider, until=date(2020, 1, 2), raw_dir=tmp_path
        )
        session.commit()
    assert results == []
