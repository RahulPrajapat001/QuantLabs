"""DataHandler: the holdout lockout, and no access beyond the current bar."""

from __future__ import annotations

from datetime import date

import pytest

from quantlab.backtesting.data_handler import DataHandler, LookAheadError
from quantlab.config import HOLDOUT_START, HoldoutViolationError, get_settings

from .conftest import make_bars


@pytest.fixture(autouse=True)
def _locked_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from the locked state, whatever the developer's shell says."""

    get_settings.cache_clear()
    monkeypatch.delenv("QUANTLAB_UNLOCK_HOLDOUT", raising=False)
    yield
    get_settings.cache_clear()


# --- The holdout lockout -----------------------------------------------------


def test_raises_on_holdout_access_without_the_env_var() -> None:
    """The headline requirement: holdout data is refused without the env var."""

    bars = make_bars(start="2023-11-01", n_days=90)  # runs into 2024
    with pytest.raises(HoldoutViolationError, match="locked holdout start"):
        DataHandler(bars, start=date(2023, 11, 1), end=date(2024, 2, 1))


def test_holdout_flag_alone_is_not_enough() -> None:
    # Two independent switches: the keyword argument without the environment
    # variable must still refuse. A copy-pasted holdout=True unlocks nothing.
    bars = make_bars(start="2023-11-01", n_days=90)
    with pytest.raises(HoldoutViolationError, match="QUANTLAB_UNLOCK_HOLDOUT"):
        DataHandler(bars, start=date(2023, 11, 1), end=date(2024, 2, 1), holdout=True)


def test_env_var_alone_is_not_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    # And the reverse: the environment variable without the explicit opt-in.
    get_settings.cache_clear()
    monkeypatch.setenv("QUANTLAB_UNLOCK_HOLDOUT", "1")
    bars = make_bars(start="2023-11-01", n_days=90)
    with pytest.raises(HoldoutViolationError, match="no explicit holdout opt-in"):
        DataHandler(bars, start=date(2023, 11, 1), end=date(2024, 2, 1))


def test_both_switches_unlock_the_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("QUANTLAB_UNLOCK_HOLDOUT", "1")
    bars = make_bars(start="2023-11-01", n_days=90)
    handler = DataHandler(
        bars, start=date(2023, 11, 1), end=date(2024, 2, 1), holdout=True
    )
    assert handler.holdout is True
    assert handler.holdout_touched is False  # nothing served yet

    served = [batch.date for batch in handler.stream()]
    assert any(d >= HOLDOUT_START for d in served)
    assert handler.holdout_touched is True
    assert "holdout" in handler.split


def test_an_unbounded_window_over_holdout_data_still_raises() -> None:
    # Omitting `end` must not soften the lock: the effective end is the data's
    # last date, and that is what gets checked.
    bars = make_bars(start="2023-11-01", n_days=90)
    with pytest.raises(HoldoutViolationError):
        DataHandler(bars)


def test_truncate_to_lock_clips_instead_of_raising() -> None:
    bars = make_bars(start="2023-11-01", n_days=90)
    handler = DataHandler(bars, truncate_to_lock=True)
    assert handler.end < HOLDOUT_START
    assert handler.split == "validation"
    assert all(d < HOLDOUT_START for d in handler.dates)
    for _ in handler.stream():
        pass
    assert handler.holdout_touched is False


def test_truncation_raises_when_nothing_is_left() -> None:
    bars = make_bars(start="2024-02-01", n_days=30)
    with pytest.raises(HoldoutViolationError, match="entirely inside the locked"):
        DataHandler(bars, truncate_to_lock=True)


def test_pre_holdout_windows_need_no_opt_in() -> None:
    handler = DataHandler(make_bars(start="2016-01-04", n_days=40))
    assert handler.split == "train"
    assert handler.holdout is False
    assert len(handler) == 40


# --- Look-ahead prevention ---------------------------------------------------


def test_stream_is_chronological_and_batched_by_date() -> None:
    handler = DataHandler(make_bars(n_days=10, securities=4))
    batches = list(handler.stream())
    assert len(batches) == 10
    assert [b.date for b in batches] == sorted(b.date for b in batches)
    assert all(len(b) == 4 for b in batches)
    assert batches[0].security_ids == ["SEC-0000", "SEC-0001", "SEC-0002", "SEC-0003"]


def test_history_never_extends_past_the_current_bar() -> None:
    handler = DataHandler(make_bars(n_days=20, securities=2))
    for step, batch in enumerate(handler.stream(), start=1):
        history = handler.history()
        assert handler.current_date == batch.date
        assert history["date"].max().date() == batch.date
        # Exactly `step` dates visible, never more.
        assert history["date"].nunique() == step
        assert len(history) == step * 2


def test_history_before_streaming_is_empty() -> None:
    handler = DataHandler(make_bars(n_days=5))
    assert handler.current_date is None
    assert handler.history().empty
    assert handler.current_bars().empty


def test_history_as_of_refuses_the_future() -> None:
    handler = DataHandler(make_bars(n_days=20, securities=2))
    stream = handler.stream()
    next(stream)
    next(stream)
    current = handler.current_date
    assert current is not None

    # Asking for today is fine; asking for tomorrow is a look-ahead bug.
    assert not handler.history_as_of(current).empty
    with pytest.raises(LookAheadError, match="look-ahead bug"):
        handler.history_as_of(handler.dates[-1])


def test_history_as_of_before_streaming_raises() -> None:
    handler = DataHandler(make_bars(n_days=5))
    with pytest.raises(LookAheadError, match="no bar has been served"):
        handler.history_as_of(date(2015, 1, 5))


def test_lookback_counts_dates_not_rows() -> None:
    handler = DataHandler(make_bars(n_days=20, securities=3))
    for _ in handler.stream():
        pass
    window = handler.history(lookback=5)
    assert window["date"].nunique() == 5
    assert len(window) == 15
    single = handler.history("SEC-0001", lookback=5)
    assert len(single) == 5
    with pytest.raises(ValueError, match="lookback must be at least 1"):
        handler.history(lookback=0)


def test_latest_price_tracks_the_cursor() -> None:
    bars = make_bars(n_days=10, securities=2)
    handler = DataHandler(bars)
    prices = []
    for _ in handler.stream():
        prices.append(handler.latest_price("SEC-0000"))
    expected = bars.loc[bars["security_id"] == "SEC-0000", "close"].tolist()
    assert prices == pytest.approx(expected)
    assert handler.latest_price("NOT-A-SECURITY") is None


def test_reset_rewinds_the_cursor() -> None:
    handler = DataHandler(make_bars(n_days=6))
    assert len(list(handler.stream())) == 6
    assert handler.finished
    handler.reset()
    assert handler.current_date is None
    assert len(list(handler.stream())) == 6


def test_the_panel_is_copied_not_referenced() -> None:
    bars = make_bars(n_days=5, securities=1)
    handler = DataHandler(bars)
    bars.loc[0, "close"] = -999.0
    next(handler.stream())
    assert handler.latest_price("SEC-0000") != -999.0


# --- Input validation --------------------------------------------------------


def test_missing_columns_and_empty_panels_are_rejected() -> None:
    bars = make_bars(n_days=5)
    with pytest.raises(ValueError, match="missing required column"):
        DataHandler(bars.drop(columns=["adj_close"]))
    with pytest.raises(ValueError, match="empty"):
        DataHandler(bars.iloc[0:0])
    with pytest.raises(ValueError, match="precedes start"):
        DataHandler(bars, start=date(2016, 1, 1), end=date(2015, 1, 1))
    with pytest.raises(ValueError, match="no bars in the window"):
        DataHandler(bars, start=date(2019, 1, 1), end=date(2019, 6, 1))
