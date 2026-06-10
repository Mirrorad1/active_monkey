"""Exp 78 — aliasing isolation: clumping a color makes it less lovable, independent of count.

Exp 77 explained mirro's natural favorite-flip by gate asymmetry and READ the asymmetry as
spatial aliasing (color 2's nine cells form one contiguous block whose observation columns
stay soft). That reading rested on one world's incidental structure. This experiment
isolates the mechanism causally: fresh newborns (separate roots, zero history — the
legitimate baseline) raised in controlled worlds with IDENTICAL color counts (9/8/8) that
differ ONLY in the 9-cell color's layout — CLUMPED (3x3 contiguous block) vs SCATTERED
(nine checkerboard cells, no two orthogonally adjacent).

Layouts (declared exactly; color 2 is the 9-cell color in both):
  CLUMPED: color 2 at rows 2-4 x cols 0-2 (cells 10,11,12,15,16,17,20,21,22); the
  remaining 16 cells alternate colors 0 and 1 in cell-index order (8 each).
  SCATTERED: color 2 at cells 0,2,4,6,8,12,16,20,24 (nine cells with (r+c) even, pairwise
  non-adjacent); the remaining 16 cells alternate colors 0 and 1 in cell-index order
  (8 each).

Predeclared predictions (5 matched birth seeds per arm, 3000 raising steps, start cell 12):
  P1 (mechanism): mean gate of color 2, exp(-H(A_hat[:,s])) averaged over color-2 cells,
     is HIGHER in SCATTERED than CLUMPED for the same birth seed, in >= 4/5 seeds.
  P2 (accrual): color 2's value share is HIGHER in SCATTERED than CLUMPED for the same
     seed, in >= 4/5 seeds.
  P3 (the law's headline — clumping can invert abundance): in CLUMPED the favorite is NOT
     color 2 in >= 3/5 seeds, AND in SCATTERED the favorite IS color 2 in >= 3/5 seeds.
Falsifiers:
  F1 = P1 fails -> the aliasing reading of Exp 77 is WRONG (the gate asymmetry there has
     another cause); a correction entry is required.
  F2 = P1 holds but P2 fails -> gates do not drive accrual in fresh creatures; the
     accrual law (Exp 77) does not generalize off the spine; diagnose.
  F3 = P1+P2 hold but P3 fails on either side -> the inversion needs a different
     clump/abundance dose; reported as a dose question, the mechanism stands.
Provided priors declared: both layouts, the raising length, start cell, birth seeds
(400+s). Newborns are separate roots (no lineage); the spine and vela are untouched
(read-only nothing — they are not even loaded).
"""
from __future__ import annotations

import collections
import math
import sys

import numpy as np

# Add repo root to path so active_loop is importable when run directly
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEPS = 3000
SEEDS = list(range(5))          # 0..4
START = 12
N_COLORS = 3
ROWS, COLS = 5, 5
N_CELLS = ROWS * COLS           # 25

# ---------------------------------------------------------------------------
# Layout construction helpers
# ---------------------------------------------------------------------------

def build_clumped_cmap() -> list:
    """Color 2 at rows 2-4 x cols 0-2; remaining cells alternate 0,1 in index order."""
    color2_cells = set()
    for r in range(2, 5):          # rows 2,3,4
        for c in range(0, 3):      # cols 0,1,2
            color2_cells.add(r * COLS + c)
    # cells 10,11,12,15,16,17,20,21,22

    cmap = [None] * N_CELLS
    for cell in color2_cells:
        cmap[cell] = 2

    alt = 0
    for cell in range(N_CELLS):
        if cmap[cell] is None:
            cmap[cell] = alt
            alt = 1 - alt          # toggle between 0 and 1

    return cmap


def build_scattered_cmap() -> list:
    """Color 2 at (r+c) even cells; remaining cells alternate 0,1 in index order."""
    color2_cells = set()
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                color2_cells.add(r * COLS + c)
    # cells 0,2,4,6,8,10,12,14,16,18,20,22,24 — 13 cells with (r+c) even overall
    # But docstring says cells 0,2,4,6,8,12,16,20,24 (nine cells)
    # The docstring specifies exactly: cells 0,2,4,6,8,12,16,20,24 — let's use that.
    color2_cells = {0, 2, 4, 6, 8, 12, 16, 20, 24}

    cmap = [None] * N_CELLS
    for cell in color2_cells:
        cmap[cell] = 2

    alt = 0
    for cell in range(N_CELLS):
        if cmap[cell] is None:
            cmap[cell] = alt
            alt = 1 - alt

    return cmap


def print_cmap_grid(cmap: list, label: str) -> None:
    print(f"\n{label}:")
    for r in range(ROWS):
        row = [str(cmap[r * COLS + c]) for c in range(COLS)]
        print("  " + " ".join(row))


def verify_counts(cmap: list, label: str) -> None:
    counts = collections.Counter(cmap)
    assert counts == {0: 8, 1: 8, 2: 9}, (
        f"{label}: expected {{0:8, 1:8, 2:9}}, got {dict(counts)}"
    )


# ---------------------------------------------------------------------------
# Gate computation
# ---------------------------------------------------------------------------

def cell_gate(A_hat: np.ndarray, cell: int) -> float:
    """exp(-H(A_hat[:, cell])) — predictability weight for a single cell column."""
    col = A_hat[:, cell]
    h = -np.sum(col * np.log(col + 1e-12))
    return math.exp(-h)


def mean_gate_for_color(A_hat: np.ndarray, cmap: list, color: int) -> float:
    """Mean gate over all cells of the given color."""
    cells = [s for s in range(N_CELLS) if cmap[s] == color]
    if not cells:
        return 0.0
    return float(np.mean([cell_gate(A_hat, s) for s in cells]))


def value_share(value_counts: np.ndarray, color: int) -> float:
    total = value_counts.sum()
    if total == 0:
        return 0.0
    return float(value_counts[color] / total)


# ---------------------------------------------------------------------------
# Per-cell gate grid printer
# ---------------------------------------------------------------------------

def print_gate_grid(A_hat: np.ndarray, cmap: list, label: str) -> None:
    print(f"\n{label} per-cell gate (exp(-H)) — color shown as subscript:")
    for r in range(ROWS):
        row_parts = []
        for c in range(COLS):
            cell = r * COLS + c
            g = cell_gate(A_hat, cell)
            col_idx = cmap[cell]
            row_parts.append(f"{g:.3f}[{col_idx}]")
        print("  " + "  ".join(row_parts))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check(
    results_clumped: list,
    results_scattered: list,
) -> str:
    """
    Each results list is ordered by seed 0..4.
    Each entry: dict with keys mean_gate_c2, share_c2, favorite.
    Returns final verdict string.
    """
    # P1: scattered mean_gate_c2 > clumped mean_gate_c2, same seed, >= 4/5
    p1_wins = sum(
        1 for s in range(5)
        if results_scattered[s]["mean_gate_c2"] > results_clumped[s]["mean_gate_c2"]
    )
    p1_pass = p1_wins >= 4

    # P2: scattered share_c2 > clumped share_c2, same seed, >= 4/5
    p2_wins = sum(
        1 for s in range(5)
        if results_scattered[s]["share_c2"] > results_clumped[s]["share_c2"]
    )
    p2_pass = p2_wins >= 4

    # P3a: clumped favorite != 2 in >= 3/5
    p3a_count = sum(1 for r in results_clumped if r["favorite"] != 2)
    p3a_pass = p3a_count >= 3

    # P3b: scattered favorite == 2 in >= 3/5
    p3b_count = sum(1 for r in results_scattered if r["favorite"] == 2)
    p3b_pass = p3b_count >= 3

    p3_pass = p3a_pass and p3b_pass

    print("\n--- Prediction checks ---")
    print(f"P1 (scattered gate_c2 > clumped gate_c2): {p1_wins}/5 seeds  -> {'PASS' if p1_pass else 'FAIL'}")
    print(f"P2 (scattered share_c2 > clumped share_c2): {p2_wins}/5 seeds -> {'PASS' if p2_pass else 'FAIL'}")
    print(f"P3a (clumped fav != 2): {p3a_count}/5 seeds -> {'PASS' if p3a_pass else 'FAIL'}")
    print(f"P3b (scattered fav == 2): {p3b_count}/5 seeds -> {'PASS' if p3b_pass else 'FAIL'}")
    print(f"P3 combined: {'PASS' if p3_pass else 'FAIL'}")

    if p1_pass and p2_pass and p3_pass:
        return "EXP78: LAW GROUNDED (aliasing causally isolated)"
    elif not p1_pass:
        return "EXP78: F1 — aliasing reading WRONG (correction required)"
    elif not p2_pass:
        return "EXP78: F2 — gates do not drive accrual off-spine"
    else:
        details = (
            f"clumped_fav_not2={p3a_count}/5, scattered_fav_is2={p3b_count}/5"
        )
        return f"EXP78: F3 — mechanism stands, inversion is dose-dependent [{details}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Build and validate layouts
    clumped_cmap = build_clumped_cmap()
    scattered_cmap = build_scattered_cmap()

    verify_counts(clumped_cmap, "CLUMPED")
    verify_counts(scattered_cmap, "SCATTERED")

    print_cmap_grid(clumped_cmap, "CLUMPED cmap (5x5)")
    print_cmap_grid(scattered_cmap, "SCATTERED cmap (5x5)")

    world_clumped = World(rows=ROWS, cols=COLS, cmap=clumped_cmap, n_colors=N_COLORS)
    world_scattered = World(rows=ROWS, cols=COLS, cmap=scattered_cmap, n_colors=N_COLORS)

    # Table header
    print(
        "\n{:<12} {:>5} {:>14} {:>14} {:>14} {:>10} {:>8} {:>8}".format(
            "arm", "seed",
            "gate_c2", "gate_c0", "gate_c1",
            "share_c2", "fav", "map_acc",
        )
    )
    print("-" * 90)

    results: dict[str, list] = {"clumped": [], "scattered": []}

    # Keep A_hat for seed-0 diagnostics
    diag: dict[str, np.ndarray] = {}

    for arm_name, world in [("clumped", world_clumped), ("scattered", world_scattered)]:
        for seed_idx in SEEDS:
            birth_seed = 400 + seed_idx
            c = Creature.birth(
                f"exp78-{arm_name}-s{seed_idx}",
                world=world,
                seed=birth_seed,
            )
            c.true_pos = START
            c.live(STEPS)

            A_hat = c._A_hat()
            g2 = mean_gate_for_color(A_hat, world.cmap, 2)
            g0 = mean_gate_for_color(A_hat, world.cmap, 0)
            g1 = mean_gate_for_color(A_hat, world.cmap, 1)
            s2 = value_share(c.value_counts, 2)
            fav = c.favorite()
            acc = c.map_accuracy()

            print(
                "{:<12} {:>5} {:>14.6f} {:>14.6f} {:>14.6f} {:>10.6f} {:>8} {:>8.4f}".format(
                    arm_name, seed_idx, g2, g0, g1, s2, fav, acc
                )
            )

            results[arm_name].append(
                {"mean_gate_c2": g2, "share_c2": s2, "favorite": fav}
            )

            if seed_idx == 0:
                diag[arm_name] = (A_hat, list(world.cmap))

    # Diagnostic: seed-0 per-cell gate grids
    print("\n=== Diagnostic: seed-0 per-cell gate grids ===")
    A_hat_cl, cmap_cl = diag["clumped"]
    A_hat_sc, cmap_sc = diag["scattered"]
    print_gate_grid(A_hat_cl, cmap_cl, "CLUMPED  (seed 0)")
    print_gate_grid(A_hat_sc, cmap_sc, "SCATTERED (seed 0)")

    # Verdict
    verdict = check(results["clumped"], results["scattered"])
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
