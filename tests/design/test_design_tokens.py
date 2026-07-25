"""The design system is binding, so the build enforces it.

``docs/DESIGN_SYSTEM.md`` claims the tokens are verified at WCAG AA and that both
frontends are driven by identical values. Those are checkable claims, so they are
checked here rather than asserted in prose:

* every foreground/background pair the system permits clears its WCAG level;
* the four generated artifacts match what the generator produces right now, so a
  hand-edit to ``config/design-tokens.css`` fails instead of silently diverging
  from the Tailwind preset and the Plotly templates;
* the structural rules the document states about the palettes hold.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_generator() -> ModuleType:
    """Import the generator from ``scripts/``, which is not an installed package."""

    path = REPO_ROOT / "scripts" / "build_design_tokens.py"
    spec = importlib.util.spec_from_file_location("build_design_tokens", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tokens = _load_generator()


# --- The accessibility claim -------------------------------------------------


def test_every_permitted_pair_meets_its_wcag_level():
    """The claim in §2.3 is computed, not asserted."""

    failures = [c.render() for c in tokens.verify_contrast() if not c.passes]
    assert not failures, "\n".join(failures)


def test_the_contrast_gate_actually_fails_a_bad_colour():
    """A gate that cannot fail is not a gate."""

    assert tokens.contrast_ratio("#777777", "#808080") < tokens.AA_NORMAL
    assert tokens.contrast_ratio("#FFFFFF", "#000000") == pytest.approx(21.0)


def test_body_text_is_held_to_aa_normal_not_the_large_text_relaxation():
    checks = [c for c in tokens.verify_contrast() if c.foreground == "text-primary"]
    assert checks
    assert all(c.required == tokens.AA_NORMAL for c in checks)


def test_semantic_colours_are_held_to_body_text_contrast():
    """They always carry a number, so they are text — not a 3:1 UI component."""

    semantic = [c for c in tokens.verify_contrast() if c.foreground == "positive"]
    assert semantic
    assert all(c.required == tokens.AA_NORMAL for c in semantic)


# --- The single-source-of-truth claim ----------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "design-tokens.css",
        "tailwind.preset.js",
        "plotly-template-dark.json",
        "plotly-template-light.json",
    ],
)
def test_the_committed_artifact_matches_the_generator(name):
    path = REPO_ROOT / "config" / name
    assert path.exists(), f"{name} has not been generated"
    expected = tokens.artifacts()[path]
    assert path.read_text(encoding="utf-8") == expected, (
        f"config/{name} no longer matches scripts/build_design_tokens.py. "
        "It is generated — edit the tokens in the generator and re-run it."
    )


def test_check_mode_reports_success_when_in_sync():
    assert tokens.main(["--check"]) == 0


def test_every_generated_file_says_it_is_generated():
    for path, content in tokens.artifacts().items():
        assert "do not edit by hand" in content, path.name


# --- Structural rules the document states ------------------------------------


def test_both_modes_define_the_same_token_names():
    """A token that exists in one mode only would break the theme swap."""

    assert set(tokens.DARK.neutral) == set(tokens.LIGHT.neutral)
    assert set(tokens.DARK.semantic) == set(tokens.LIGHT.semantic)
    assert len(tokens.DARK.categorical) == len(tokens.LIGHT.categorical)


def test_the_categorical_palette_is_capped_at_five_slots():
    """§2.4: a sixth series folds into Other, small multiples, or a composite."""

    for mode in tokens.MODES:
        assert len(mode.categorical) == 5


def test_semantic_colours_are_never_reused_as_series_colours():
    """Green means gain everywhere; a green series would compete with that."""

    for mode in tokens.MODES:
        reserved = set(mode.semantic.values())
        assert not reserved & set(mode.categorical)


def test_the_diverging_ramp_has_a_neutral_grey_midpoint():
    """Never a hue at the middle of a diverging scale."""

    for mode in tokens.MODES:
        midpoint = mode.diverging[len(mode.diverging) // 2]
        raw = midpoint.lstrip("#")
        r, g, b = (int(raw[i : i + 2], 16) for i in (0, 2, 4))
        assert max(r, g, b) - min(r, g, b) < 24, f"{mode.name} midpoint {midpoint}"


def test_the_diverging_ramp_runs_negative_to_positive():
    for mode in tokens.MODES:
        assert mode.diverging[0] == mode.semantic["negative"]
        assert mode.diverging[-1] == mode.semantic["positive"]


def test_the_sequential_ramp_runs_monotonically_away_from_its_surface():
    """Low values sit near the surface, high values furthest from it.

    That means increasing luminance on a dark surface and decreasing luminance
    on a light one — not one ramp reused in both modes with the order flipped.
    """

    for mode in tokens.MODES:
        lums = [tokens.relative_luminance(c) for c in mode.sequential]
        surface = tokens.relative_luminance(mode.neutral["surface"])
        expected = sorted(lums, reverse=surface > 0.5)
        assert lums == expected, f"{mode.name} sequential ramp is not monotone"
        # …and the ramp genuinely departs from the surface rather than hugging it.
        assert abs(lums[-1] - surface) > abs(lums[0] - surface)


def test_spacing_is_a_four_pixel_scale():
    values = [int(v.removesuffix("px")) for v in tokens.SPACING.values()]
    assert values == sorted(values)
    assert all(v % 4 == 0 for v in values)
    assert max(values) == 64


def test_every_type_step_declares_a_line_height():
    for name, (size, height) in tokens.TYPE_SCALE.items():
        assert size.endswith("px") and height.endswith("px"), name
        assert int(height.removesuffix("px")) > int(size.removesuffix("px")), name


def test_the_density_constraints_hold_arithmetically():
    """§4 claims 20 rows fit without scrolling. Check the arithmetic."""

    row = int(tokens.DIMENSIONS["table-row-height"].removesuffix("px"))
    header = int(tokens.DIMENSIONS["table-header-height"].removesuffix("px"))
    assert 20 * row + header <= 600
    assert int(tokens.DIMENSIONS["metric-card-max-height"].removesuffix("px")) <= 120


# --- Chart template rules ----------------------------------------------------


@pytest.mark.parametrize("mode", list(tokens.MODES), ids=lambda m: m.name)
def test_the_plotly_template_has_no_vertical_grid_and_no_frame(mode):
    layout = tokens.build_plotly(mode)["layout"]
    assert layout["xaxis"]["showgrid"] is False, "no vertical grid"
    assert layout["yaxis"]["showgrid"] is True, "a subtle horizontal y grid"
    for axis in ("xaxis", "yaxis"):
        assert layout[axis]["showline"] is False, "no chart frame"
        assert layout[axis]["zeroline"] is False


@pytest.mark.parametrize("mode", list(tokens.MODES), ids=lambda m: m.name)
def test_chart_tick_labels_are_monospace(mode):
    """Axis numbers obey the tabular-figures rule exactly like table numbers."""

    layout = tokens.build_plotly(mode)["layout"]
    for axis in ("xaxis", "yaxis"):
        assert layout[axis]["tickfont"]["family"] == tokens.FONT_MONO


@pytest.mark.parametrize("mode", list(tokens.MODES), ids=lambda m: m.name)
def test_the_template_defaults_to_direct_labelling(mode):
    """showlegend defaults off so shipping a legend is a decision (§6)."""

    assert tokens.build_plotly(mode)["layout"]["showlegend"] is False


@pytest.mark.parametrize("mode", list(tokens.MODES), ids=lambda m: m.name)
def test_the_colorway_is_the_categorical_palette_in_order(mode):
    assert tokens.build_plotly(mode)["layout"]["colorway"] == list(mode.categorical)


def test_the_benchmark_colour_is_never_a_categorical_slot(mode=None):
    """§6: the benchmark is always muted grey, never a series colour.

    It lives in the token files rather than the Plotly template, because Plotly
    rejects invented layout keys — and does so by misparsing the surrounding
    layout rather than by naming the offender.
    """

    for m in tokens.MODES:
        assert m.semantic["benchmark"] not in m.categorical


@pytest.mark.parametrize("name", ["dark", "light"])
def test_the_template_is_a_valid_plotly_template(name):
    """Generated is not the same as loadable. Round-trip it through Plotly.

    This caught two keys that looked plausible and were not: a top-level
    ``_comment`` and an invented ``layout.colorway_meta``. The second is the
    dangerous one — Plotly reported it as a broken ``colorway``, so the template
    failed in a way that pointed at the wrong line.
    """

    go = pytest.importorskip("plotly.graph_objects")
    path = REPO_ROOT / "config" / f"plotly-template-{name}.json"
    template = go.layout.Template(json.loads(path.read_text(encoding="utf-8")))
    assert len(template.layout.colorway) == 5
    assert template.layout.yaxis.showgrid is True
    assert template.layout.xaxis.showgrid is False


# --- The split badge contract ------------------------------------------------


def test_every_split_the_config_can_produce_has_a_badge_colour():
    """resolve_split can emit compound labels; each component must be renderable."""

    from quantlab.config import Split

    for split in Split:
        assert split.value in tokens.SPLIT_BADGES
    assert "pre-sample" in tokens.SPLIT_BADGES


def test_holdout_is_the_loudest_badge():
    assert tokens.SPLIT_BADGES["holdout"] == "negative"


def test_compound_split_labels_decompose_into_known_badges():
    from quantlab.config import DATA_START, VALIDATION_END, resolve_split

    compound = resolve_split(DATA_START, VALIDATION_END)
    assert "+" in compound
    assert all(part in tokens.SPLIT_BADGES for part in compound.split("+"))
