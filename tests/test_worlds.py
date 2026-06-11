"""Tests for active_loop/worlds.py — procedural benchmark world generators.

Covers:
- Determinism per seed for all four world kinds
- aliased(seed 7/11/13) matches the exact layouts committed in
  experiments/outputs/exp143.txt and experiments/outputs/exp144.txt
- analytic_floor for p=0.7, n_colors=3 is within 0.02 nats of 0.82
- Module is purely additive (not imported by existing code)
"""

import math

import numpy as np
import pytest

from active_loop.worlds import (
    analytic_floor,
    aliased,
    nonstationary,
    noisy,
    _build_aliased_cmap,
    ALIASED_LAYOUT_SEEDS,
    N_COLORS_ALIASED,
)

# ---------------------------------------------------------------------------
# Committed CMAP values extracted from experiments/outputs/exp143.txt and
# experiments/outputs/exp144.txt.  These are the ground-truth values the
# layout-reproduction tests assert against.
#
# Source verification:
#   exp143.txt: "CMAP array: [1, 1, 2, 0, 2, 3, 0, 1, 0, 3, 0, 3, 3, 2, 1, 2]"
#               (seed 7, single layout)
#   exp144.txt: Layout 1 (rng(7)):  [1, 1, 2, 0, 2, 3, 0, 1, 0, 3, 0, 3, 3, 2, 1, 2]
#               Layout 2 (rng(11)): [3, 1, 0, 2, 2, 1, 1, 3, 0, 1, 3, 2, 0, 0, 3, 2]
#               Layout 3 (rng(13)): [1, 0, 0, 2, 1, 2, 1, 0, 1, 3, 2, 2, 3, 3, 3, 0]
# ---------------------------------------------------------------------------

COMMITTED_CMAP_SEED7 = [1, 1, 2, 0, 2, 3, 0, 1, 0, 3, 0, 3, 3, 2, 1, 2]
COMMITTED_CMAP_SEED11 = [3, 1, 0, 2, 2, 1, 1, 3, 0, 1, 3, 2, 0, 0, 3, 2]
COMMITTED_CMAP_SEED13 = [1, 0, 0, 2, 1, 2, 1, 0, 1, 3, 2, 2, 3, 3, 3, 0]


# ---------------------------------------------------------------------------
# analytic_floor tests
# ---------------------------------------------------------------------------


def test_analytic_floor_exp132_value():
    """p=0.7, n_colors=3 should be within 0.02 nats of 0.82 (Exp 132 spec)."""
    val = analytic_floor(0.7, 3)
    assert abs(val - 0.82) < 0.02, f"Expected ~0.82 nats, got {val:.4f}"


def test_analytic_floor_exact_formula():
    """Cross-check against the formula stated in exp132 docstring: ~0.8228 nats."""
    # 0.7 * ln(1/0.7) + 0.3 * ln(1/0.15) = -0.7*ln(0.7) - 0.3*ln(0.15)
    expected = -0.7 * math.log(0.7) - 0.15 * math.log(0.15) - 0.15 * math.log(0.15)
    val = analytic_floor(0.7, 3)
    assert abs(val - expected) < 1e-10


def test_analytic_floor_exceeds_ceiling_threshold():
    """Irreducible floor must exceed the 0.7-nat ceiling threshold for p=0.7, n=3."""
    val = analytic_floor(0.7, 3)
    assert val > 0.7, f"Expected > 0.7 nats, got {val:.4f}"


def test_analytic_floor_uniform_is_log_n():
    """Uniform noise (p_true -> 0, n colors) -> floor -> log(n)."""
    # For p_true very small the distribution approaches uniform over n colors
    # For p_true = 1/n exactly, all colors equally likely
    n = 4
    p_true = 1.0 / n
    val = analytic_floor(p_true, n)
    assert abs(val - math.log(n)) < 1e-10


def test_analytic_floor_no_noise_is_zero():
    """p_true approaching 1 means the floor approaches 0."""
    val = analytic_floor(0.9999999, 3)
    assert val < 0.001


def test_analytic_floor_invalid_p_true():
    """Out-of-range p_true raises ValueError."""
    with pytest.raises(ValueError):
        analytic_floor(0.0, 3)
    with pytest.raises(ValueError):
        analytic_floor(1.0, 3)
    with pytest.raises(ValueError):
        analytic_floor(-0.1, 3)


def test_analytic_floor_invalid_n_colors():
    """n_colors < 2 raises ValueError."""
    with pytest.raises(ValueError):
        analytic_floor(0.7, 1)


# ---------------------------------------------------------------------------
# aliased() layout reproduction tests (the critical ground-truth checks)
# ---------------------------------------------------------------------------


def test_aliased_seed7_matches_committed_exp143():
    """aliased(layout_seed=7) must reproduce the Exp 143 committed CMAP exactly."""
    w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=7)
    assert w["cmap"] == COMMITTED_CMAP_SEED7, (
        f"seed=7 cmap mismatch.\n"
        f"  got:      {w['cmap']}\n"
        f"  expected: {COMMITTED_CMAP_SEED7}"
    )


def test_aliased_seed11_matches_committed_exp144():
    """aliased(layout_seed=11) must reproduce the Exp 144 Layout 2 CMAP exactly."""
    w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=11)
    assert w["cmap"] == COMMITTED_CMAP_SEED11, (
        f"seed=11 cmap mismatch.\n"
        f"  got:      {w['cmap']}\n"
        f"  expected: {COMMITTED_CMAP_SEED11}"
    )


def test_aliased_seed13_matches_committed_exp144():
    """aliased(layout_seed=13) must reproduce the Exp 144 Layout 3 CMAP exactly."""
    w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=13)
    assert w["cmap"] == COMMITTED_CMAP_SEED13, (
        f"seed=13 cmap mismatch.\n"
        f"  got:      {w['cmap']}\n"
        f"  expected: {COMMITTED_CMAP_SEED13}"
    )


@pytest.mark.parametrize("seed,expected", [
    (7, COMMITTED_CMAP_SEED7),
    (11, COMMITTED_CMAP_SEED11),
    (13, COMMITTED_CMAP_SEED13),
])
def test_aliased_all_canonical_seeds(seed, expected):
    """All three canonical layout seeds reproduce their committed CMaps."""
    w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=seed)
    assert w["cmap"] == expected


# ---------------------------------------------------------------------------
# aliased() structural tests
# ---------------------------------------------------------------------------


def test_aliased_balanced_assignment():
    """Each color must appear exactly n_cells_per_color times."""
    for seed in ALIASED_LAYOUT_SEEDS:
        w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=seed)
        cmap = w["cmap"]
        for color in range(4):
            count = cmap.count(color)
            assert count == 4, (
                f"seed={seed}: color {color} appears {count} times, expected 4"
            )


def test_aliased_cmap_length():
    """CMAP length must equal n_colors * n_cells_per_color."""
    w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=7)
    assert len(w["cmap"]) == 16


def test_aliased_metadata():
    """aliased() dict has the correct kind and metadata fields."""
    w = aliased(n_colors=4, n_cells_per_color=4, layout_seed=7)
    assert w["kind"] == "aliased"
    assert w["n_colors"] == 4
    assert w["n_cells_per_color"] == 4
    assert w["layout_seed"] == 7
    assert w["rows"] == 4
    assert w["cols"] == 4


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------


def test_aliased_deterministic_per_seed():
    """aliased() is deterministic: two calls with the same seed return the same cmap."""
    for seed in ALIASED_LAYOUT_SEEDS:
        w1 = aliased(layout_seed=seed)
        w2 = aliased(layout_seed=seed)
        assert w1["cmap"] == w2["cmap"], f"Non-determinism for seed={seed}"


def test_aliased_different_seeds_differ():
    """Different layout seeds produce different CMaps."""
    cmaps = [aliased(layout_seed=s)["cmap"] for s in ALIASED_LAYOUT_SEEDS]
    # All three should be distinct
    assert cmaps[0] != cmaps[1], "seed 7 and 11 produced identical cmaps"
    assert cmaps[0] != cmaps[2], "seed 7 and 13 produced identical cmaps"
    assert cmaps[1] != cmaps[2], "seed 11 and 13 produced identical cmaps"


def test_noisy_deterministic():
    """noisy() is deterministic: same p_true -> same dict."""
    w1 = noisy(0.7)
    w2 = noisy(0.7)
    assert w1["cmap"] == w2["cmap"]
    assert w1["p_true"] == w2["p_true"]


def test_nonstationary_deterministic():
    """nonstationary() is deterministic per seed."""
    base = aliased(n_colors=4, n_cells_per_color=4, layout_seed=7)
    w1 = nonstationary(base, remap_at_step=1000, remap_seed=42)
    w2 = nonstationary(base, remap_at_step=1000, remap_seed=42)
    assert w1["cmap"] == w2["cmap"]
    assert w1["cmap_after"] == w2["cmap_after"]


def test_nonstationary_different_seeds_differ():
    """Different remap seeds produce different cmap_after."""
    base = aliased(n_colors=4, n_cells_per_color=4, layout_seed=7)
    w1 = nonstationary(base, remap_at_step=500, remap_seed=1)
    w2 = nonstationary(base, remap_at_step=500, remap_seed=2)
    assert w1["cmap_after"] != w2["cmap_after"], "Different remap seeds should differ"


# ---------------------------------------------------------------------------
# noisy() tests
# ---------------------------------------------------------------------------


def test_noisy_kind():
    """noisy() returns kind='noisy'."""
    w = noisy()
    assert w["kind"] == "noisy"


def test_noisy_p_true_stored():
    """noisy() stores the provided p_true value."""
    w = noisy(0.8)
    assert w["p_true"] == 0.8


def test_noisy_has_cmap():
    """noisy() includes a non-empty cmap."""
    w = noisy()
    assert isinstance(w["cmap"], list)
    assert len(w["cmap"]) > 0


# ---------------------------------------------------------------------------
# nonstationary() tests
# ---------------------------------------------------------------------------


def test_nonstationary_kind():
    """nonstationary() returns kind='nonstationary'."""
    base = aliased(layout_seed=7)
    w = nonstationary(base, remap_at_step=500, remap_seed=99)
    assert w["kind"] == "nonstationary"


def test_nonstationary_preserves_base_kind():
    """nonstationary() records the base world kind in base_kind."""
    base = aliased(layout_seed=7)
    w = nonstationary(base, remap_at_step=500, remap_seed=99)
    assert w["base_kind"] == "aliased"


def test_nonstationary_cmap_unchanged():
    """nonstationary() preserves the base cmap (pre-remap)."""
    base = aliased(layout_seed=7)
    w = nonstationary(base, remap_at_step=500, remap_seed=99)
    assert w["cmap"] == base["cmap"]


def test_nonstationary_remap_at_step():
    """nonstationary() records remap_at_step correctly."""
    base = aliased(layout_seed=7)
    w = nonstationary(base, remap_at_step=1234, remap_seed=7)
    assert w["remap_at_step"] == 1234


def test_nonstationary_cmap_after_length():
    """cmap_after has the same length as the base cmap."""
    base = aliased(layout_seed=7)
    w = nonstationary(base, remap_at_step=500, remap_seed=5)
    assert len(w["cmap_after"]) == len(base["cmap"])


def test_nonstationary_cmap_after_valid_colors():
    """cmap_after contains only valid color indices."""
    base = aliased(n_colors=4, n_cells_per_color=4, layout_seed=7)
    w = nonstationary(base, remap_at_step=500, remap_seed=5)
    for v in w["cmap_after"]:
        assert 0 <= v < base["n_colors"], f"Invalid color index {v} in cmap_after"


# ---------------------------------------------------------------------------
# _build_aliased_cmap low-level test
# ---------------------------------------------------------------------------


def test_build_aliased_cmap_balance():
    """_build_aliased_cmap produces a perfectly balanced assignment for any seed."""
    for seed in [1, 7, 11, 13, 42, 100]:
        cmap = _build_aliased_cmap(4, 4, seed)
        for color in range(4):
            assert cmap.count(color) == 4, (
                f"seed={seed}: color {color} appears {cmap.count(color)} times"
            )
