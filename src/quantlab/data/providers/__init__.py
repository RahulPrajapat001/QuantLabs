"""Concrete data providers.

These modules are the *only* place vendor SDKs are imported. US providers exist
to prove the :class:`~quantlab.data.base.DataProvider` abstraction genuinely
abstracts — no research runs on US data (PRD §2).
"""

from __future__ import annotations

from .alpha_vantage import AlphaVantageProvider
from .yfinance import YFinanceProvider

__all__ = ["AlphaVantageProvider", "YFinanceProvider"]
