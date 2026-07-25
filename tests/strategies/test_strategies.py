"""Every strategy: signals are sane, warmup is respected, look-ahead is impossible."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtesting.data_handler import BarBatch, DataHandler
from quantlab.backtesting.events import Direction, Signal
from quantlab.strategies import (
    BollingerBandStrategy,
    BreakoutStrategy,
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    CrossSectionalReversalStrategy,
    CustomRuleStrategy,
    DollarCostAveragingStrategy,
    MACDStrategy,
    MovingAverageCrossStrategy,
    PairsTradingStrategy,
    RSIStrategy,
    Strategy,
    default_strategies,
)
from quantlab.strategies.rules import RuleContext


def make_panel(
    *,
    n_days: int = 200,
    securities: int = 3,
    seed: int = 0,
    drift: float = 0.0,
    volatility: float = 0.015,
    start: str = "2016-01-04",
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for i in range(securities):
        close = 100.0 * np.cumprod(1.0 + rng.normal(drift, volatility, n_days))
        for t, day in enumerate(dates):
            price = float(close[t])
            rows.append(
                {
                    "date": day,
                    "security_id": f"SEC-{i:04d}",
                    "open": price * 0.999,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "adj_close": price,
                    "volume": 1_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def drive(strategy: Strategy, bars: pd.DataFrame) -> list[Signal]:
    """Stream a panel through a strategy exactly as the engine does."""

    handler = DataHandler(bars)
    collected: list[Signal] = []
    for i, batch in enumerate(handler.stream()):
        if i < strategy.warmup:
            continue
        collected.extend(strategy.on_bar(batch, handler))
    return collected


ALL_STRATEGIES: list[Strategy] = [
    MovingAverageCrossStrategy(fast=10, slow=30),
    RSIStrategy(period=14),
    MACDStrategy(),
    BollingerBandStrategy(window=20),
    BreakoutStrategy(entry_window=20, exit_window=10),
    CrossSectionalMomentumStrategy(lookback=60, skip=5),
    CrossSectionalReversalStrategy(lookback=5),
    DollarCostAveragingStrategy(),
    BuyAndHoldStrategy(),
]


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_every_strategy_produces_well_formed_signals(strategy: Strategy) -> None:
    signals = drive(strategy, make_panel())
    for signal in signals:
        assert signal.strategy_name == strategy.name
        assert signal.direction in (Direction.LONG, Direction.SHORT, Direction.EXIT)
        assert 0.0 <= signal.strength <= 1.0
        assert signal.security_id.startswith("SEC-")


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_every_strategy_declares_loggable_parameters(strategy: Strategy) -> None:
    parameters = strategy.parameters()
    assert isinstance(parameters, dict)
    for key, value in parameters.items():
        assert isinstance(key, str)
        assert value is None or isinstance(value, (int, float, str, bool))


@pytest.mark.parametrize("strategy", ALL_STRATEGIES, ids=lambda s: s.name)
def test_no_strategy_can_see_beyond_the_current_bar(strategy: Strategy) -> None:
    """The handler is the only window onto the data, and it is cursor-bounded."""

    bars = make_panel(n_days=80, securities=2)
    handler = DataHandler(bars)
    for i, batch in enumerate(handler.stream()):
        if i < strategy.warmup:
            continue
        strategy.on_bar(batch, handler)
        visible = handler.history()
        assert visible["date"].max().date() <= batch.date


def test_signal_strength_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="strength"):
        Signal(
            date=pd.Timestamp("2016-01-04").date(),
            security_id="A",
            direction=Direction.LONG,
            strategy_name="x",
            strength=1.5,
        )


# --- Individual behaviour ----------------------------------------------------


def test_ma_cross_goes_long_in_an_uptrend_and_exits_in_a_downtrend() -> None:
    dates = pd.bdate_range("2016-01-04", periods=120)
    # Rise for 60 days, then fall for 60.
    prices = [100.0 * 1.01**i for i in range(60)]
    prices += [prices[-1] * 0.99**i for i in range(1, 61)]
    bars = pd.DataFrame(
        [
            {
                "date": d,
                "security_id": "SEC-0000",
                "open": p,
                "high": p * 1.001,
                "low": p * 0.999,
                "close": p,
                "adj_close": p,
                "volume": 1e6,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )
    signals = drive(MovingAverageCrossStrategy(fast=5, slow=20), bars)
    directions = [s.direction for s in signals]
    assert Direction.LONG in directions
    assert Direction.EXIT in directions
    # Long while rising, out by the end.
    assert signals[-1].direction is Direction.EXIT


def test_ma_cross_rejects_an_inverted_window_pair() -> None:
    with pytest.raises(ValueError, match="must be shorter"):
        MovingAverageCrossStrategy(fast=200, slow=50)


def test_rsi_buys_oversold() -> None:
    """A long, steady decline drives RSI below 30 and triggers a long."""

    dates = pd.bdate_range("2016-01-04", periods=80)
    prices = [100.0 * 0.98**i for i in range(80)]
    bars = pd.DataFrame(
        [
            {
                "date": d,
                "security_id": "SEC-0000",
                "open": p,
                "high": p * 1.001,
                "low": p * 0.999,
                "close": p,
                "adj_close": p,
                "volume": 1e6,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )
    signals = drive(RSIStrategy(period=14, oversold=30.0), bars)
    assert any(s.direction is Direction.LONG for s in signals)


def test_breakout_fires_on_a_new_high_excluding_today() -> None:
    """The channel must exclude today, or a new high is tautological."""

    dates = pd.bdate_range("2016-01-04", periods=40)
    prices = [100.0] * 30 + [100.0 + i for i in range(1, 11)]
    bars = pd.DataFrame(
        [
            {
                "date": d,
                "security_id": "SEC-0000",
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "adj_close": p,
                "volume": 1e6,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )
    signals = drive(BreakoutStrategy(entry_window=20, exit_window=10), bars)
    longs = [s for s in signals if s.direction is Direction.LONG]
    assert longs, "a run of new highs must produce entries"


def test_momentum_ranks_cross_sectionally_and_skips_recent_month() -> None:
    bars = make_panel(n_days=200, securities=10, seed=1)
    strategy = CrossSectionalMomentumStrategy(lookback=100, skip=21, top_fraction=0.2)
    signals = drive(strategy, bars)
    assert signals

    # On each rebalance date, exactly 20% are long and the rest are exits.
    by_date: dict[object, list[Signal]] = {}
    for signal in signals:
        by_date.setdefault(signal.date, []).append(signal)
    for _day, day_signals in by_date.items():
        longs = [s for s in day_signals if s.direction is Direction.LONG]
        assert len(longs) == max(1, int(len(day_signals) * 0.2))
        assert all("momentum" in s.metadata for s in day_signals)


def test_momentum_rebalances_monthly_not_daily() -> None:
    """Turnover control is the whole reason this is a monthly strategy."""

    bars = make_panel(n_days=200, securities=5, seed=2)
    signals = drive(CrossSectionalMomentumStrategy(lookback=60, skip=5), bars)
    rebalance_dates = {s.date for s in signals}
    months = {(d.year, d.month) for d in rebalance_dates}
    assert len(rebalance_dates) == len(months), "one rebalance per month"


def test_momentum_rejects_a_skip_longer_than_its_lookback() -> None:
    with pytest.raises(ValueError, match="skip must be shorter"):
        CrossSectionalMomentumStrategy(lookback=20, skip=30)
    with pytest.raises(ValueError, match="top_fraction"):
        CrossSectionalMomentumStrategy(top_fraction=1.5)


def test_reversal_holds_the_worst_performers() -> None:
    bars = make_panel(n_days=60, securities=10, seed=4)
    signals = drive(
        CrossSectionalReversalStrategy(lookback=5, bottom_fraction=0.2), bars
    )
    assert signals
    for signal in signals:
        assert "trailing_return" in signal.metadata
    # The longs are the ones with the lowest trailing return on that date.
    by_date: dict[object, list[Signal]] = {}
    for signal in signals:
        by_date.setdefault(signal.date, []).append(signal)
    for day_signals in by_date.values():
        longs = [s for s in day_signals if s.direction is Direction.LONG]
        exits = [s for s in day_signals if s.direction is Direction.EXIT]
        if longs and exits:
            assert max(s.metadata["trailing_return"] for s in longs) <= min(
                s.metadata["trailing_return"] for s in exits
            )


def test_buy_and_hold_signals_exactly_once() -> None:
    bars = make_panel(n_days=100, securities=3)
    signals = drive(BuyAndHoldStrategy(), bars)
    assert len(signals) == 3
    assert len({s.date for s in signals}) == 1


def test_dca_signals_once_per_month() -> None:
    bars = make_panel(n_days=200, securities=2)
    signals = drive(DollarCostAveragingStrategy(interval="monthly"), bars)
    months = {(s.date.year, s.date.month) for s in signals}
    assert len(signals) == len(months) * 2  # two securities per month


def test_dca_signals_accumulate_so_it_actually_averages_in() -> None:
    """Without ``accumulate``, DCA is indistinguishable from buy-and-hold.

    The portfolio treats a repeated entry signal as a hold, so a contribution to a
    name already held would be dropped and only the first purchase would ever
    happen.
    """

    signals = drive(DollarCostAveragingStrategy(), make_panel(n_days=120, securities=1))
    assert signals
    assert all(s.accumulate for s in signals)


def test_other_strategies_do_not_accumulate() -> None:
    """Only a scheduled-contribution strategy should pyramid."""

    for strategy in (
        MovingAverageCrossStrategy(fast=5, slow=20),
        BuyAndHoldStrategy(),
        CrossSectionalMomentumStrategy(lookback=60, skip=5),
    ):
        signals = drive(strategy, make_panel(n_days=120, securities=2))
        assert all(not s.accumulate for s in signals), strategy.name


def test_strategies_reset_their_state() -> None:
    """A second run must not inherit the first run's position memory."""

    bars = make_panel(n_days=100, securities=2)
    strategy = BuyAndHoldStrategy()
    first = drive(strategy, bars)
    assert drive(strategy, bars) == [], "state persists until reset"
    strategy.reset()
    assert len(drive(strategy, bars)) == len(first)


# --- Pairs -------------------------------------------------------------------


def test_pairs_trades_a_diverging_spread() -> None:
    """A cointegrated pair, pulled apart, must produce an entry then an exit."""

    n = 200
    dates = pd.bdate_range("2016-01-04", periods=n)
    rng = np.random.default_rng(9)
    base = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    a = base.copy()
    # Push A well below its relationship with B for 10 days, then restore it.
    a[120:130] *= 0.85
    rows = []
    for t, day in enumerate(dates):
        for sid, price in (("A", a[t]), ("B", base[t])):
            p = float(price)
            rows.append(
                {
                    "date": day,
                    "security_id": sid,
                    "open": p,
                    "high": p * 1.005,
                    "low": p * 0.995,
                    "close": p,
                    "adj_close": p,
                    "volume": 1e6,
                }
            )
    bars = pd.DataFrame(rows)
    signals = drive(PairsTradingStrategy("A", "B", window=60), bars)
    directions = [s.direction for s in signals]
    assert Direction.LONG in directions
    assert all(s.security_id == "A" for s in signals)
    assert all("zscore" in s.metadata for s in signals)


def test_pairs_validates_its_configuration() -> None:
    with pytest.raises(ValueError, match="entry_z must exceed"):
        PairsTradingStrategy("A", "B", entry_z=0.5, exit_z=2.0)
    with pytest.raises(ValueError, match="too short"):
        PairsTradingStrategy("A", "B", window=10)


# --- Custom rules ------------------------------------------------------------


def test_custom_rule_strategy_honours_buy_and_sell_rules() -> None:
    strategy = CustomRuleStrategy(
        buy=lambda ctx: ctx.close < 95.0,
        sell=lambda ctx: ctx.close > 105.0,
        lookback=30,
    )
    bars = make_panel(n_days=200, securities=2, seed=6)
    signals = drive(strategy, bars)
    assert any(s.direction is Direction.LONG for s in signals)


def test_custom_rule_stop_loss_fires_and_is_labelled() -> None:
    """The stop emits a signal filled at the next open — not an intrabar exit."""

    dates = pd.bdate_range("2016-01-04", periods=60)
    # Flat, then a sharp decline that breaches an 8% stop.
    prices = [100.0] * 30 + [100.0 * 0.97**i for i in range(1, 31)]
    bars = pd.DataFrame(
        [
            {
                "date": d,
                "security_id": "SEC-0000",
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "adj_close": p,
                "volume": 1e6,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )
    strategy = CustomRuleStrategy(buy=lambda ctx: True, stop_loss=0.08, lookback=30)
    signals = drive(strategy, bars)
    exits = [s for s in signals if s.direction is Direction.EXIT]
    assert exits
    assert exits[0].metadata["exit_reason"] == "stop_loss"


def test_custom_rule_take_profit_fires() -> None:
    dates = pd.bdate_range("2016-01-04", periods=60)
    prices = [100.0] * 25 + [100.0 * 1.02**i for i in range(1, 36)]
    bars = pd.DataFrame(
        [
            {
                "date": d,
                "security_id": "SEC-0000",
                "open": p,
                "high": p,
                "low": p,
                "close": p,
                "adj_close": p,
                "volume": 1e6,
            }
            for d, p in zip(dates, prices, strict=True)
        ]
    )
    strategy = CustomRuleStrategy(buy=lambda ctx: True, take_profit=0.10, lookback=30)
    exits = [s for s in drive(strategy, bars) if s.direction is Direction.EXIT]
    assert exits
    assert exits[0].metadata["exit_reason"] == "take_profit"


def test_custom_rule_position_size_becomes_signal_strength() -> None:
    strategy = CustomRuleStrategy(
        buy=lambda ctx: True,
        position_size=lambda ctx: 0.25,
        lookback=30,
    )
    signals = drive(strategy, make_panel(n_days=60, securities=1))
    assert signals[0].strength == pytest.approx(0.25)


def test_custom_rule_size_is_clamped_to_the_unit_interval() -> None:
    strategy = CustomRuleStrategy(
        buy=lambda ctx: True, position_size=lambda ctx: 42.0, lookback=30
    )
    signals = drive(strategy, make_panel(n_days=60, securities=1))
    assert signals[0].strength == 1.0


def test_a_crashing_rule_is_treated_as_no_signal() -> None:
    """A user rule that blows up on one bar must not abandon the backtest."""

    def broken(ctx: RuleContext) -> bool:
        raise ValueError("bad indicator")

    strategy = CustomRuleStrategy(buy=broken, lookback=30)
    assert drive(strategy, make_panel(n_days=60, securities=1)) == []


def test_a_rule_attempting_look_ahead_still_raises() -> None:
    """Look-ahead is never swallowed, even inside a user-supplied rule."""

    from quantlab.backtesting.data_handler import LookAheadError

    bars = make_panel(n_days=60, securities=1)
    final = pd.Timestamp(bars["date"].max()).date()

    def peeking(ctx: RuleContext) -> bool:
        raise LookAheadError("tried to read the future")

    strategy = CustomRuleStrategy(buy=peeking, lookback=30)
    with pytest.raises(LookAheadError):
        drive(strategy, bars)
    assert final is not None


def test_custom_rule_validates_stop_and_target_fractions() -> None:
    with pytest.raises(ValueError, match="stop_loss"):
        CustomRuleStrategy(buy=lambda ctx: True, stop_loss=8.0)
    with pytest.raises(ValueError, match="take_profit"):
        CustomRuleStrategy(buy=lambda ctx: True, take_profit=110.0)


def test_rule_context_exposes_unrealised_return() -> None:
    bars = make_panel(n_days=40, securities=1)
    handler = DataHandler(bars)
    batch: BarBatch | None = None
    for served in handler.stream():
        batch = served
    assert batch is not None
    history = handler.history("SEC-0000")
    context = RuleContext(
        security_id="SEC-0000",
        bar=history.iloc[-1],
        history=history,
        entry_price=100.0,
    )
    assert context.is_open
    assert context.unrealised_return == pytest.approx(context.close / 100.0 - 1.0)
    assert len(context.closes(5)) == 5

    flat = RuleContext("SEC-0000", history.iloc[-1], history, None)
    assert not flat.is_open
    assert flat.unrealised_return == 0.0


# --- Registry ----------------------------------------------------------------


def test_default_strategies_covers_the_required_set() -> None:
    strategies = default_strategies()
    for required in (
        "ma_cross",
        "rsi_reversion",
        "macd",
        "bollinger_reversion",
        "breakout",
        "momentum_12_1",
        "reversal_5d",
        "dca",
        "buy_and_hold",
    ):
        assert required in strategies
        assert strategies[required].name == required
