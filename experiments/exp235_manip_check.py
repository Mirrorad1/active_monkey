"""Exp 235 Rung 1 — Pre-registered MANIPULATION CHECK (L38 reachability check).

PROTOCOL (binding, from loop/directions/environmental-complexity.md):
  Flood-fill over the expected-crossing admissibility graph from the founder spawn cells.
  Report:
    (a) WITHHELD-REGEN FRACTION at resident climb_ability=0.05 (≥30% sealed behind ridge
        is required for a valid substrate; <30% = invalid/too-legible).
    (b) MARGINAL REACHABLE-REGEN unlocked across the 0.05→0.10 climb step (= the
        additional fraction of steady-state regen that becomes REACHABLE when climb goes
        from 0.05 to 0.10, relative to the non-resident-reachable remainder).

METHOD:
  1. Build the frozen elevation map (no evolution; fixed rng seed).
  2. Compute the per-cell STEADY-STATE regen weight (proportional to what step_regen
     assigns each cell) — this is the "regen inventory" at each cell.
  3. For a given climb_ability c, compute the EXPECTED admissibility probability for
     every directed edge (pos -> n) using the sigmoid formula:
       p_admit(pos, n, c) = 1.0 if elev[n] <= elev[pos]   (downhill/flat)
       p_admit(pos, n, c) = sigmoid((c - delta) / softness)  (upslope, delta = elev[n]-elev[pos])
  4. Two metrics are reported:
     (A) EXPECTED-REGEN approach: for each cell, compute the EXPECTED regen actually
         collected per step, accounting for the probability of reaching it from the founder
         basin. For a cell c reachable via a path p0->p1->...->c, the expected regen is
         regen[c] * P(path), where P(path) = product of crossing probabilities.
         We approximate this as: reachable_regen_at_climb = sum over cells c of
         regen[c] * max over all paths of product(p_admit(pos_i, pos_{i+1}, climb)).
         For cells on the plateau, the bottleneck is the basin-to-ridge crossing,
         so we report both the hard-sealed (prob < threshold) and soft-sealed metrics.
     (B) BINARY-SEALED approach (simpler, interpretable): flood-fill with a threshold
         p_threshold on crossing probability. Below threshold = "effectively sealed."

  This script reports BOTH approaches so the human can calibrate.

  NOTE: The current geometry (ridge=1.0 above basin=0.0, softness=0.08) creates a
  very tight rim where even climb=1.0 gives only 50% crossing probability per step.
  The spec says "each ε buys a small increase in crossing odds" — this is technically
  satisfied, but the ABSOLUTE crossing odds at the resident are very low (<0.01%).
  The human should tune softness and/or the ridge height to get the desired gradient.

This script is runnable without modification:
  uv run --python .venv python experiments/exp235_manip_check.py

The human will use the printed numbers to tune basin/ridge/plateau geometry.

OUTPUT PATH: experiments/outputs/exp235_manip_check.txt
"""
from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Add project root to path for uv run
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecology.world import GridWorld, _TERRAIN_GATE_SOFTNESS_DEFAULT


# ---------------------------------------------------------------------------
# Configuration — first-guess basin/ridge/plateau geometry
# ---------------------------------------------------------------------------
ROWS = 12
COLS = 12
CAPACITY = 10.0
REGEN_RATE = 0.20
INITIAL_RESOURCE = 0.7
FOOD_CONCENTRATION = 2.0           # plateau regen multiplier (first guess)
TERRAIN_GATE_SOFTNESS = 0.08
SEED = 0

RESIDENT_CLIMB = 0.05
MUTANT_CLIMB = 0.10

# Probability threshold for "effectively sealed" in the binary-sealed approach.
# At p < SEAL_THRESHOLD, an upslope edge is considered sealed for the flood-fill.
# With softness=0.08, sigmoid((0.05 - 1.0) / 0.08) ≈ 7e-6 << 0.01 → ridge is sealed.
SEAL_THRESHOLD = 0.01

OUTPUT_PATH = Path("experiments/outputs/exp235_manip_check.txt")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def p_admit(elev_pos: float, elev_n: float, climb: float, softness: float) -> float:
    """Return crossing probability for edge pos -> n."""
    delta = elev_n - elev_pos
    if delta <= 0.0:
        return 1.0
    return sigmoid((climb - delta) / softness)


def binary_reachable_cells(
    world: GridWorld,
    founder_cells: list[int],
    climb: float,
    softness: float,
    threshold: float,
) -> set[int]:
    """BFS flood-fill: cell is reachable iff the crossing prob >= threshold."""
    assert world.elevation is not None
    visited: set[int] = set()
    queue: deque[int] = deque()
    for c in founder_cells:
        if c not in visited:
            visited.add(c)
            queue.append(c)
    while queue:
        pos = queue.popleft()
        elev_here = float(world.elevation[pos])
        for n in world.neighbors(pos):
            if n in visited:
                continue
            p = p_admit(elev_here, float(world.elevation[n]), climb, softness)
            if p >= threshold:
                visited.add(n)
                queue.append(n)
    return visited


def expected_reachable_regen(
    world: GridWorld,
    founder_cells: list[int],
    climb: float,
    softness: float,
    regen_weights: np.ndarray,
) -> float:
    """Expected regen accessible per step under the crossing probability model.

    For each cell n, computes the maximum path probability (Viterbi-style best path)
    from any founder cell to n, then weights the cell's regen by this probability.
    Returns the sum over all cells of regen[c] * best_path_prob(c).

    This is the expected regen if a creature took the BEST path to each cell —
    it represents an UPPER BOUND on expected regen income.
    """
    assert world.elevation is not None
    n_cells = world.rows * world.cols
    best_prob = np.zeros(n_cells, dtype=np.float64)
    for c in founder_cells:
        best_prob[c] = 1.0

    # Bellman-Ford style: propagate max probabilities iteratively.
    # Guaranteed to converge on a DAG; upslope edges form a DAG by elevation.
    # Flat edges need a few passes for lateral spread.
    for _ in range(n_cells):
        updated = False
        for pos in range(n_cells):
            if best_prob[pos] == 0.0:
                continue
            elev_here = float(world.elevation[pos])
            for n in world.neighbors(pos):
                p = p_admit(elev_here, float(world.elevation[n]), climb, softness)
                new_p = best_prob[pos] * p
                if new_p > best_prob[n] + 1e-15:
                    best_prob[n] = new_p
                    updated = True
        if not updated:
            break

    expected_regen = float(np.sum(regen_weights * best_prob))
    return expected_regen


def compute_cell_regen_weights(world: GridWorld) -> np.ndarray:
    """Return per-cell regen weight (proportional to steady-state regen rate)."""
    assert world.elevation is not None
    n_cells = world.rows * world.cols
    food_concentration = world.terrain_food_concentration
    on_plateau = world.elevation > 0.5
    n_in = int(np.sum(on_plateau))
    n_out = n_cells - n_in
    weights = np.zeros(n_cells, dtype=np.float64)
    if food_concentration > 0.0 and n_in > 0:
        if n_out > 0:
            out_factor = max(0.0, (n_cells - n_in * food_concentration) / n_out)
            if out_factor == 0.0:
                out_factor = 1.0   # guard: basin keeps normal regen
        else:
            out_factor = 0.0
        weights = np.where(on_plateau, food_concentration, out_factor).astype(np.float64)
    else:
        weights = np.ones(n_cells, dtype=np.float64)
    return weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(SEED)
    world = GridWorld.from_config(
        rows=ROWS,
        cols=COLS,
        capacity=CAPACITY,
        regen_rate=REGEN_RATE,
        initial_resource=INITIAL_RESOURCE,
        rng=rng,
        enable_terrain=True,
        terrain_food_concentration=FOOD_CONCENTRATION,
        terrain_gate_softness=TERRAIN_GATE_SOFTNESS,
        terrain_gates_movement=True,
    )
    assert world.elevation is not None

    elev = world.elevation
    unique_elevs = sorted(set(float(e) for e in elev))
    n_cells = ROWS * COLS
    n_basin = int(np.sum(elev == 0.0))
    n_ridge = int(np.sum(elev == 1.0))
    n_plateau = int(np.sum(elev > 0.5))

    # Founder spawn cells (basin-restricted — same logic as engine.py)
    basin_cells_list = [c for c in range(n_cells) if float(elev[c]) <= 0.0]
    n_founders = 12
    basin_step = max(1, len(basin_cells_list) // max(1, n_founders))
    founder_cells = [basin_cells_list[(i * basin_step) % len(basin_cells_list)]
                     for i in range(n_founders)]
    founder_cells_unique = sorted(set(founder_cells))

    regen_weights = compute_cell_regen_weights(world)
    total_regen = float(np.sum(regen_weights))

    # --- Binary-sealed (flood-fill with threshold) ---
    reached_res_binary = binary_reachable_cells(
        world, founder_cells_unique, RESIDENT_CLIMB, TERRAIN_GATE_SOFTNESS, SEAL_THRESHOLD
    )
    reached_mut_binary = binary_reachable_cells(
        world, founder_cells_unique, MUTANT_CLIMB, TERRAIN_GATE_SOFTNESS, SEAL_THRESHOLD
    )
    res_regen_binary = sum(float(regen_weights[c]) for c in reached_res_binary)
    mut_regen_binary = sum(float(regen_weights[c]) for c in reached_mut_binary)
    withheld_binary = (total_regen - res_regen_binary) / max(1e-9, total_regen)
    marginal_sealed_binary = total_regen - res_regen_binary
    marginal_unlocked_binary = (mut_regen_binary - res_regen_binary) / max(1e-9, marginal_sealed_binary)

    # --- Expected-regen approach ---
    res_expected = expected_reachable_regen(
        world, founder_cells_unique, RESIDENT_CLIMB, TERRAIN_GATE_SOFTNESS, regen_weights
    )
    mut_expected = expected_reachable_regen(
        world, founder_cells_unique, MUTANT_CLIMB, TERRAIN_GATE_SOFTNESS, regen_weights
    )
    withheld_expected = (total_regen - res_expected) / max(1e-9, total_regen)
    marginal_expected = (mut_expected - res_expected) / max(1e-9, total_regen - res_expected)

    # --- Per-elevation-transition crossing probabilities ---
    delta_rim = 1.0   # basin (0.0) to ridge (1.0)
    delta_top = 0.5   # ridge (1.0) to plateau (1.5)
    p_rim_res = sigmoid((RESIDENT_CLIMB - delta_rim) / TERRAIN_GATE_SOFTNESS)
    p_rim_mut = sigmoid((MUTANT_CLIMB - delta_rim) / TERRAIN_GATE_SOFTNESS)
    p_top_res = sigmoid((RESIDENT_CLIMB - delta_top) / TERRAIN_GATE_SOFTNESS)
    p_top_mut = sigmoid((MUTANT_CLIMB - delta_top) / TERRAIN_GATE_SOFTNESS)

    lines = [
        "=" * 70,
        "EXP 235 MANIPULATION CHECK — Reachability Flood-Fill (L38)",
        "=" * 70,
        "",
        "SUBSTRATE GEOMETRY",
        f"  Grid: {ROWS}x{COLS} = {n_cells} cells",
        f"  Unique elevations: {unique_elevs}",
        f"  Basin cells (elev=0.0): {n_basin}  ({n_basin/n_cells*100:.0f}%)",
        f"  Ridge cells (elev=1.0): {n_ridge}  ({n_ridge/n_cells*100:.0f}%)",
        f"  Plateau cells (elev>0.5): {n_plateau}  ({n_plateau/n_cells*100:.0f}%)",
        f"  Terrain gate softness: {TERRAIN_GATE_SOFTNESS}",
        f"  Plateau food_concentration: {FOOD_CONCENTRATION}",
        "",
        "REGEN WEIGHTS (proportional to per-cell steady-state regen)",
        f"  Total weight: {total_regen:.4f}",
        f"  Basin+ridge weight: {sum(float(regen_weights[c]) for c in range(n_cells) if float(elev[c]) <= 0.5):.4f}  ({sum(float(regen_weights[c]) for c in range(n_cells) if float(elev[c]) <= 0.5)/total_regen*100:.1f}%)",
        f"  Plateau weight: {sum(float(regen_weights[c]) for c in range(n_cells) if float(elev[c]) > 0.5):.4f}  ({sum(float(regen_weights[c]) for c in range(n_cells) if float(elev[c]) > 0.5)/total_regen*100:.1f}%)",
        "",
        "FOUNDER SPAWN (basin-restricted)",
        f"  Unique founder cells: {founder_cells_unique}",
        f"  Count: {len(founder_cells_unique)} (from {n_founders} founders)",
        "",
        "CROSSING PROBABILITIES PER STEP",
        f"  Basin->Ridge (delta=1.0, softness={TERRAIN_GATE_SOFTNESS}):",
        f"    climb={RESIDENT_CLIMB}: p = {p_rim_res:.2e}  ({p_rim_res*100:.4f}%/step)",
        f"    climb={MUTANT_CLIMB}: p = {p_rim_mut:.2e}  ({p_rim_mut*100:.4f}%/step)",
        f"    ratio: {p_rim_mut/max(1e-15, p_rim_res):.2f}x improvement",
        f"  Ridge->Plateau (delta=0.5, softness={TERRAIN_GATE_SOFTNESS}):",
        f"    climb={RESIDENT_CLIMB}: p = {p_top_res:.2e}  ({p_top_res*100:.4f}%/step)",
        f"    climb={MUTANT_CLIMB}: p = {p_top_mut:.2e}  ({p_top_mut*100:.4f}%/step)",
        "",
        "=" * 70,
        "PREDECLARED MANIPULATION CHECK METRICS",
        "=" * 70,
        "",
        "--- BINARY-SEALED approach (flood-fill, threshold={:.3f}) ---".format(SEAL_THRESHOLD),
        f"  Resident (climb={RESIDENT_CLIMB}): {len(reached_res_binary)} cells reachable, regen={res_regen_binary:.4f}",
        f"  Mutant   (climb={MUTANT_CLIMB}): {len(reached_mut_binary)} cells reachable, regen={mut_regen_binary:.4f}",
        "",
        f"  (a) WITHHELD-REGEN at resident: {withheld_binary*100:.1f}%",
        f"      Required: ≥30%  |  Status: {'PASS' if withheld_binary >= 0.30 else 'FAIL'}",
        f"  (b) MARGINAL UNLOCKED (0.05->0.10): {marginal_unlocked_binary*100:.1f}%",
        f"      Status: {'LARGE' if marginal_unlocked_binary >= 0.05 else 'SMALL'}",
        "",
        "--- EXPECTED-REGEN approach (best-path probability * regen_weight) ---",
        f"  Resident (climb={RESIDENT_CLIMB}): expected regen = {res_expected:.6f} (of {total_regen:.4f})",
        f"  Mutant   (climb={MUTANT_CLIMB}): expected regen = {mut_expected:.6f}",
        "",
        f"  (a) WITHHELD-REGEN at resident: {withheld_expected*100:.2f}%",
        f"      Required: ≥30%  |  Status: {'PASS' if withheld_expected >= 0.30 else 'FAIL'}",
        f"  (b) MARGINAL UNLOCKED (0.05->0.10): {marginal_expected*100:.4f}%",
        f"      Status: {'LARGE' if marginal_expected >= 0.05 else 'SMALL — crossing too tight at both values'}",
        "",
        "=" * 70,
        "TUNING GUIDANCE",
        "=" * 70,
        "",
        "The current geometry seals the plateau very tightly (crossing ~7e-4 % per step",
        "at resident climb=0.05). This satisfies (a) but the marginal step (b) is also tiny.",
        "",
        "To widen the gradient, consider:",
        "  - Reduce ridge height: set ridge elevation to 0.5 instead of 1.0 so delta=0.5",
        "    instead of delta=1.0. At climb=0.05, p = sigmoid((0.05-0.5)/0.08) = 0.4%.",
        "  - Increase softness: with softness=0.2, p(0.05, delta=1.0) = 0.7%,",
        "    p(0.10, delta=1.0) = 1.2% — ratio still 1.7x but LARGER absolute probability.",
        "  - Both together: delta=0.5, softness=0.15 gives p(0.05)=4.6%, p(0.10)=8.1%",
        "    — these are plausible per-step crossing rates in a 200-step run.",
        "",
        "NOTE: Exact tuning is the human's job. This script is the predeclared instrument.",
        "      Run it BEFORE the gradient batch to verify the manipulation check passes.",
        "=" * 70,
    ]

    output = "\n".join(lines)
    print(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output + "\n")
    print(f"\n[Written to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
