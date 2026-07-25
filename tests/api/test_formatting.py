"""Formatting, checked against DESIGN_SYSTEM.md §5 rule by rule.

The design system is binding, and its number-formatting table is the part two
frontends and a PDF all depend on. These assert the exact strings, because "looks
about right" is how a 1dp percentage becomes a 2dp one in the Streamlit view.
"""

from __future__ import annotations

import math

import pytest

from quantlab.api.formatting import (
    EM_DASH,
    MINUS,
    format_basis_points,
    format_count,
    format_currency,
    format_date,
    format_ic,
    format_p_value,
    format_percent,
    format_ratio,
    format_share,
    format_t_stat,
)
from quantlab.api.models import Num

# --- Returns: 1dp, explicit sign, typographic minus --------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.124, "+12.4%"),
        (-0.081, f"{MINUS}8.1%"),
        (0.0, "+0.0%"),
        (1.5, "+150.0%"),
        (-0.00004, f"{MINUS}0.0%"),
    ],
)
def test_returns_are_signed_percentages_of_a_fraction(value, expected):
    assert format_percent(value) == expected


def test_negatives_use_a_typographic_minus_not_a_hyphen():
    """A hyphen is narrower than a digit and breaks tabular alignment."""

    rendered = format_percent(-0.081)
    assert rendered.startswith(MINUS)
    assert not rendered.startswith("-")
    assert MINUS != "-"


def test_negatives_are_never_parenthesised():
    for renderer in (format_percent, format_ratio, format_currency):
        assert "(" not in renderer(-1.5)


def test_a_share_of_a_whole_carries_no_sign():
    assert format_share(0.342) == "34.2%"
    assert not format_share(0.342).startswith("+")


# --- Ratios, ICs, t-stats, p-values ------------------------------------------


def test_ratios_are_two_decimals():
    assert format_ratio(1.234) == "1.23"
    assert format_ratio(-0.5) == f"{MINUS}0.50"


def test_information_coefficients_keep_four_decimals():
    """An IC of 0.01 is a real signal; 2dp would round it to nothing."""

    assert format_ic(0.0087) == "0.0087"


def test_t_stats_are_signed():
    assert format_t_stat(2.41) == "+2.41"
    assert format_t_stat(-2.41) == f"{MINUS}2.41"


def test_p_values_go_scientific_below_a_thousandth():
    """`0.000` is a rounding artefact that overstates the result."""

    assert format_p_value(0.032) == "0.032"
    assert format_p_value(0.0000411) == "4.1e-05"
    assert format_p_value(0.0000411) != "0.000"


# --- Currency and counts ------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1_204_338.50, "₹1.2M"),
        (840_500.0, "₹840.5K"),
        (2_400_000_000.0, "₹2.4B"),
        (1_204.5, "₹1,204.50"),
    ],
)
def test_currency_abbreviates_by_magnitude(value, expected):
    assert format_currency(value) == expected


def test_currency_can_be_forced_exact():
    assert format_currency(1_204_338.50, abbreviate=False) == "₹1,204,338.50"


def test_counts_are_thousands_separated_with_no_decimals():
    assert format_count(12480) == "12,480"
    assert format_count(12480.0) == "12,480"


def test_basis_points():
    assert format_basis_points(0.00224) == "22.4 bps"


def test_dates_are_iso_8601():
    from datetime import date, datetime

    assert format_date(date(2021, 12, 31)) == "2021-12-31"
    assert format_date(datetime(2021, 12, 31, 15, 30)) == "2021-12-31"


# --- Nulls --------------------------------------------------------------------


@pytest.mark.parametrize(
    "renderer",
    [
        format_percent,
        format_share,
        format_ratio,
        format_ic,
        format_t_stat,
        format_p_value,
        format_currency,
        format_count,
        format_basis_points,
        format_date,
    ],
)
def test_none_renders_as_an_em_dash(renderer):
    """Never 0, never NaN, never blank — a missing value is not a zero."""

    assert renderer(None) == EM_DASH


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_render_as_missing_too(value):
    """NaN on a screen is a data bug wearing a number's clothes."""

    assert format_percent(value) == EM_DASH
    assert format_ratio(value) == EM_DASH


# --- The Num wrapper ----------------------------------------------------------


def test_num_carries_the_raw_value_and_the_string():
    num = Num.percent(0.124)
    assert num.value == pytest.approx(0.124)
    assert num.formatted == "+12.4%"


def test_num_maps_nan_to_null_because_json_has_no_nan():
    num = Num.ratio(float("nan"))
    assert num.value is None
    assert num.formatted == EM_DASH


def test_num_is_json_serialisable_even_when_missing():
    import json

    payload = json.dumps(Num.ratio(float("inf")).model_dump())
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert json.loads(payload)["value"] is None


def test_num_is_frozen():
    """A formatted number and its value must not drift apart after construction."""

    from pydantic import ValidationError

    num = Num.ratio(1.0)
    with pytest.raises(ValidationError):
        num.value = 2.0  # type: ignore[misc]


def test_every_num_factory_produces_a_string():
    for factory, argument in (
        (Num.percent, 0.1),
        (Num.share, 0.1),
        (Num.ratio, 1.0),
        (Num.ic, 0.01),
        (Num.t_stat, 2.0),
        (Num.p_value, 0.05),
        (Num.currency, 1000.0),
        (Num.count, 5),
        (Num.basis_points, 0.001),
    ):
        result = factory(argument)
        assert isinstance(result.formatted, str) and result.formatted
        assert result.value is None or math.isfinite(result.value)
