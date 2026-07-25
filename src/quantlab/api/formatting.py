"""Display formatting, implemented once so two frontends cannot disagree.

PRD Module 12: *"Every endpoint returns display-ready formatted values alongside
raw ones so formatting cannot diverge between frontends."* The Next.js product UI
and the Streamlit researcher dashboard both render numbers that a reader will
compare against each other and against the PDF tearsheet. If each formats its
own, they will drift — a 1dp percentage here and a 2dp one there — and the reader
cannot tell a formatting difference from a data difference.

So the API sends both: the raw number for charts and sorting, and the exact string
to print. This module is the only place that string is produced, and it implements
`docs/DESIGN_SYSTEM.md` §5 literally, including two rules that are easy to get
subtly wrong:

* **Negatives use U+2212 MINUS SIGN, not a hyphen**, and never parentheses.
  Accounting parentheses break digit alignment in a tabular-figures column, which
  is the whole point of the monospace rule.
* **Null renders as an em dash**, never ``0``, ``NaN``, ``null`` or blank. A
  missing Sharpe and a Sharpe of zero are different claims. ``NaN`` reaching a
  screen is a data bug wearing a number's clothes.
"""

from __future__ import annotations

import math
from datetime import date, datetime

__all__ = [
    "EM_DASH",
    "MINUS",
    "format_basis_points",
    "format_count",
    "format_currency",
    "format_date",
    "format_ic",
    "format_p_value",
    "format_percent",
    "format_ratio",
    "format_share",
    "format_t_stat",
]

#: U+2014. The only representation of a missing value in the product.
EM_DASH = "—"
#: U+2212. Typographic minus — aligns with digits; a hyphen does not.
MINUS = "−"

#: Indian equities, so the tearsheet and the UI both quote rupees.
CURRENCY_SYMBOL = "₹"


def _renderable(value: float | int | None) -> float | None:
    """The value as a float, or ``None`` when there is nothing to display.

    ``NaN`` and infinities count as nothing: they are not renderable quantities,
    and printing ``nan`` is strictly worse than admitting the value is absent.
    Returning the narrowed float also means every formatter below works with a
    real number instead of re-checking an ``Optional``.
    """

    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _sign(value: float, *, explicit: bool) -> str:
    if value < 0:
        return MINUS
    return "+" if explicit else ""


def format_percent(
    value: float | None, *, decimals: int = 1, explicit_sign: bool = True
) -> str:
    """A return or drawdown, as a percentage.

    ``value`` is a **fraction** (0.124 -> ``+12.4%``). Percentages are stored as
    fractions everywhere and formatted only here; a component that receives
    ``12.4`` and another that receives ``0.124`` is how a 100x error ships.

    ``explicit_sign=False`` is for a share of a whole (``34.2%``), which has no
    meaningful direction.
    """

    number = _renderable(value)
    if number is None:
        return EM_DASH
    number *= 100.0
    return f"{_sign(number, explicit=explicit_sign)}{abs(number):,.{decimals}f}%"


def format_share(value: float | None, *, decimals: int = 1) -> str:
    """A proportion of a whole — no sign (``34.2%``)."""

    return format_percent(value, decimals=decimals, explicit_sign=False)


def format_ratio(value: float | None, *, decimals: int = 2) -> str:
    """Sharpe, Sortino, Calmar, information ratio, profit factor. 2dp."""

    number = _renderable(value)
    if number is None:
        return EM_DASH
    return f"{_sign(number, explicit=False)}{abs(number):,.{decimals}f}"


def format_ic(value: float | None) -> str:
    """An information coefficient. 4dp — an IC of 0.01 is a real signal."""

    number = _renderable(value)
    if number is None:
        return EM_DASH
    return f"{_sign(number, explicit=False)}{abs(number):.4f}"


def format_t_stat(value: float | None) -> str:
    """A t-statistic. 2dp with an explicit sign — direction is the point."""

    number = _renderable(value)
    if number is None:
        return EM_DASH
    return f"{_sign(number, explicit=True)}{abs(number):.2f}"


def format_p_value(value: float | None) -> str:
    """A p-value. 3dp, or scientific below 0.001.

    ``0.000`` is not a p-value — it is a rounding artefact that overstates the
    result. Below the third decimal the exponent carries the information.
    """

    number = _renderable(value)
    if number is None:
        return EM_DASH
    if number < 0.001:
        return f"{number:.1e}"
    return f"{number:.3f}"


def format_currency(
    value: float | None, *, abbreviate: bool | None = None, decimals: int = 2
) -> str:
    """Rupees. Abbreviated above 100,000 by default, exact below.

    ``abbreviate=None`` picks by magnitude, which is what a table wants: a
    portfolio value reads ``₹1.2M`` and a per-trade cost reads ``₹184.20``.
    """

    number = _renderable(value)
    if number is None:
        return EM_DASH
    magnitude = abs(number)
    use_short = magnitude >= 100_000 if abbreviate is None else abbreviate
    sign = _sign(number, explicit=False)

    if not use_short:
        return f"{sign}{CURRENCY_SYMBOL}{magnitude:,.{decimals}f}"
    for threshold, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= threshold:
            return f"{sign}{CURRENCY_SYMBOL}{magnitude / threshold:,.1f}{suffix}"
    return f"{sign}{CURRENCY_SYMBOL}{magnitude:,.{decimals}f}"


def format_count(value: int | float | None) -> str:
    """A count of things. Thousands-separated, no decimals."""

    number = _renderable(value)
    if number is None:
        return EM_DASH
    return f"{int(number):,d}"


def format_basis_points(value: float | None) -> str:
    """A cost or spread in basis points, 1dp (``22.4 bps``)."""

    number = _renderable(value)
    if number is None:
        return EM_DASH
    return f"{number * 10_000:,.1f} bps"


def format_date(value: date | datetime | None) -> str:
    """ISO 8601. Never a locale-dependent format."""

    if value is None:
        return EM_DASH
    if isinstance(value, datetime):
        return value.date().isoformat()
    return value.isoformat()
