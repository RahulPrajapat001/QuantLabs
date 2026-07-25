"""Import-order regression: the packages must be acyclic in both directions.

``quantlab.strategies.base`` imports the ``DataHandler`` from
``quantlab.backtesting``, so if anything in ``quantlab.backtesting`` imports from
``quantlab.strategies`` at runtime, the two packages are circular — and which one
fails depends on which the caller imports first.

The rest of the suite never caught this, because every test module happens to import
``quantlab.backtesting`` before ``quantlab.strategies``. The CLI imported them the
other way round and broke immediately. These tests run each order in a fresh
subprocess, so import order is actually under test rather than inherited from
whatever pytest collected first.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

ORDERS = [
    # The order the CLI uses, which is the one that was broken.
    "import quantlab.strategies; import quantlab.backtesting",
    "import quantlab.backtesting; import quantlab.strategies",
    "from quantlab.strategies import default_strategies; "
    "from quantlab.backtesting import BacktestEngine",
    "from quantlab.backtesting import BacktestEngine; "
    "from quantlab.strategies import default_strategies",
    # Submodule-first, which triggers each package's __init__ in turn.
    "import quantlab.strategies.base",
    "import quantlab.backtesting.engine",
    "import quantlab.backtesting.cost_sensitivity",
]


@pytest.mark.parametrize("statement", ORDERS)
def test_packages_import_in_any_order(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"`{statement}` failed:\n{result.stderr}\n"
        "quantlab.backtesting must not import quantlab.strategies at runtime — "
        "use a TYPE_CHECKING import for the Strategy annotation."
    )


def test_the_cli_entry_point_loads() -> None:
    """The failure mode this file exists for: the CLI could not even start."""

    result = subprocess.run(
        [sys.executable, "-m", "quantlab", "backtest", "strategies"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "momentum_12_1" in result.stdout
    assert "buy_and_hold" in result.stdout
