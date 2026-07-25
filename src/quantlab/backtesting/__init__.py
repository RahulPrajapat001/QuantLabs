"""Event-driven backtesting (PRD Module 8).

Strictly separated components, each with one responsibility:

* :class:`~quantlab.backtesting.data_handler.DataHandler` — streams bars
  chronologically; access beyond the current bar is structurally impossible, and the
  holdout lockout is enforced here.
* :class:`~quantlab.strategies.base.Strategy` — consumes bars, emits Signals.
* :class:`~quantlab.backtesting.portfolio.Portfolio` — consumes Signals, applies
  sizing and risk limits, emits Orders; knows nothing about strategies.
* :class:`~quantlab.backtesting.execution.ExecutionHandler` — consumes Orders,
  applies costs and slippage, emits Fills at the next bar's open.
* :class:`~quantlab.backtesting.ledger.Ledger` — records every fill, position and
  daily mark.

:class:`~quantlab.backtesting.engine.BacktestEngine` wires them together. Its
``run()`` is inherited from
:class:`~quantlab.backtesting.logged.LoggedBacktestEngine` and is final, so every
run writes to ``research_log`` — that is not optional and cannot be overridden.
"""

from quantlab.backtesting.costs import (
    CostBreakdown,
    CostModel,
    FixedBpsSlippage,
    GenericBpsModel,
    IndianEquityCostModel,
    PercentCappedBrokerage,
    Segment,
    SlippageModel,
    VolumeShareSlippage,
    ZeroCostModel,
    ZeroDeliveryBrokerage,
    ZeroSlippage,
    default_cost_models,
)
from quantlab.backtesting.data_handler import (
    REQUIRED_COLUMNS,
    BarBatch,
    DataHandler,
    LookAheadError,
)
from quantlab.backtesting.engine import (
    BacktestConfig,
    BacktestEngine,
    DelistingEvent,
)
from quantlab.backtesting.events import (
    Direction,
    Fill,
    Order,
    RejectedOrder,
    RejectionReason,
    Side,
    Signal,
)
from quantlab.backtesting.execution import (
    CircuitLimitPolicy,
    ExcessPolicy,
    ExecutionHandler,
)
from quantlab.backtesting.ledger import ClosedTrade, DailyMark, Ledger
from quantlab.backtesting.logged import LoggedBacktest, LoggedBacktestEngine
from quantlab.backtesting.metrics import BacktestMetrics, DrawdownInfo, compute_metrics
from quantlab.backtesting.portfolio import (
    Portfolio,
    Position,
    RiskLimits,
    SizingMethod,
)

__all__ = [
    "REQUIRED_COLUMNS",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestMetrics",
    "BarBatch",
    "CircuitLimitPolicy",
    "ClosedTrade",
    "CostBreakdown",
    "CostModel",
    "DailyMark",
    "DataHandler",
    "DelistingEvent",
    "Direction",
    "DrawdownInfo",
    "ExcessPolicy",
    "ExecutionHandler",
    "Fill",
    "FixedBpsSlippage",
    "GenericBpsModel",
    "IndianEquityCostModel",
    "Ledger",
    "LoggedBacktest",
    "LoggedBacktestEngine",
    "LookAheadError",
    "Order",
    "PercentCappedBrokerage",
    "Portfolio",
    "Position",
    "RejectedOrder",
    "RejectionReason",
    "RiskLimits",
    "Segment",
    "Side",
    "Signal",
    "SizingMethod",
    "SlippageModel",
    "VolumeShareSlippage",
    "ZeroCostModel",
    "ZeroDeliveryBrokerage",
    "ZeroSlippage",
    "compute_metrics",
    "default_cost_models",
]
