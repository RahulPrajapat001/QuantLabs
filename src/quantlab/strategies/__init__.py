"""Strategy implementations (PRD Module 7).

A strategy consumes bars and emits signals. It never sizes a position, never sees
the portfolio's capital, and never places an order — see
:mod:`quantlab.strategies.base` for why that boundary is load-bearing.

Available: moving-average cross, RSI, MACD, Bollinger, breakout, cross-sectional
momentum, cross-sectional reversal, pairs trading, dollar-cost averaging,
buy-and-hold, and a custom-rule interface.
"""

from quantlab.strategies.base import SingleNameStrategy, Strategy
from quantlab.strategies.pairs import PairsTradingStrategy
from quantlab.strategies.rules import CustomRuleStrategy, RuleContext
from quantlab.strategies.systematic import (
    BuyAndHoldStrategy,
    CrossSectionalMomentumStrategy,
    CrossSectionalReversalStrategy,
    DollarCostAveragingStrategy,
)
from quantlab.strategies.technical import (
    BollingerBandStrategy,
    BreakoutStrategy,
    MACDStrategy,
    MovingAverageCrossStrategy,
    RSIStrategy,
)

__all__ = [
    "BollingerBandStrategy",
    "BreakoutStrategy",
    "BuyAndHoldStrategy",
    "CrossSectionalMomentumStrategy",
    "CrossSectionalReversalStrategy",
    "CustomRuleStrategy",
    "DollarCostAveragingStrategy",
    "MACDStrategy",
    "MovingAverageCrossStrategy",
    "PairsTradingStrategy",
    "RSIStrategy",
    "RuleContext",
    "SingleNameStrategy",
    "Strategy",
]


def default_strategies() -> dict[str, Strategy]:
    """One instance of each strategy, for the cost-sensitivity sweep.

    Windows are shortened relative to their defaults so a sweep over a few years of
    data produces enough trades to say anything. These are *not* recommended
    parameters — and every one of them counts as a trial in ``research_log``.
    """

    return {
        "ma_cross": MovingAverageCrossStrategy(fast=20, slow=50),
        "rsi_reversion": RSIStrategy(),
        "macd": MACDStrategy(),
        "bollinger_reversion": BollingerBandStrategy(),
        "breakout": BreakoutStrategy(entry_window=20, exit_window=10),
        "momentum_12_1": CrossSectionalMomentumStrategy(lookback=126, skip=21),
        "reversal_5d": CrossSectionalReversalStrategy(),
        "dca": DollarCostAveragingStrategy(),
        "buy_and_hold": BuyAndHoldStrategy(),
    }
