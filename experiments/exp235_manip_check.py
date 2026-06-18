"""Exp 235 Rung 1 — Pre-registered MANIPULATION CHECK (L38 reachability check).

PROTOCOL (binding, from loop/directions/environmental-complexity.md):
  Report:
    (a) WITHHELD — the resident at climb=0.05 consumes only a LOW share of its food on
        the plateau (plateau_intake_share(0.05) ≤ RESIDENT_SHARE_THRESHOLD).
    (b) MARGINAL — the marginal share unlocked across the 0.05→0.10 step is LARGE
        (plateau_intake_share(0.10) - plateau_intake_share(0.05) ≥ MARGINAL_THRESHOLD).

  Both criteria must PASS for the substrate to be valid.

METHOD:
  TWO instruments are run and reported:

  INSTRUMENT 1 — Static flood-fill (fast; no engine):
    Flood-fill over the expected-crossing admissibility graph from the founder spawn cells.
    Computes expected-regen withheld and marginal unlocked using the sigmoid crossing model.

  INSTRUMENT 2 — Gen-0 simulation-based (binding; engine runs):
    Run the ACTUAL engine for one generation (horizon steps, mutation_rate=0, all creatures
    fixed at climb=c) on the RELIEF world (enable_terrain=True) using ecology/runtime, and
    measure plateau_intake_share(c) = (food consumed on plateau cells)/(total food consumed),
    for c in GEN0_CLIMB_GRID.

    plateau_intake_share is measured by tagging each eat event with the cell elevation:
    we run the sim and at each step accumulate per-elevation intake. Since the engine
    does not expose per-step eat telemetry directly, we patch the world.consume() method
    to record intake by cell type (basin vs plateau) during the run.

  PASS/FAIL criteria (canonical thresholds):
    RESIDENT_SHARE_THRESHOLD = 0.25  (resident ≤ 25% plateau intake ⇒ sealed)
    MARGINAL_THRESHOLD = 0.10        (mutant share − resident share ≥ 10pp ⇒ large marginal)

  The public function manipulation_check(config)->dict is importable by pytest.

OUTPUT PATH: experiments/outputs/exp235_manip_check.txt

USAGE:
  uv run --python .venv python experiments/exp235_manip_check.py

  Or import and call directly:
    from experiments.exp235_manip_check import manipulation_check
    result = manipulation_check(config_dict)
"""
from __future__ import annotations

import dataclasses as D
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path for uv run
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecology.world import GridWorld, _TERRAIN_GATE_SOFTNESS_DEFAULT, _TERRAIN_RIDGE_HEIGHT_DEFAULT
from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS
from ecology.genotype import founder as make_founder


# ---------------------------------------------------------------------------
# Canonical thresholds (document here; test reads from here)
# ---------------------------------------------------------------------------

#: plateau_intake_share(resident) must be AT OR BELOW this for the substrate to be "sealed".
RESIDENT_SHARE_THRESHOLD: float = 0.25
#: plateau_intake_share(mutant) - plateau_intake_share(resident) must be AT OR ABOVE this.
MARGINAL_THRESHOLD: float = 0.10

# ---------------------------------------------------------------------------
# Configuration — canonical geometry (first-guess; human will retune)
# ---------------------------------------------------------------------------
ROWS = 12
COLS = 12
CAPACITY = 10.0
REGEN_RATE = 0.20
INITIAL_RESOURCE = 0.7
FOOD_CONCENTRATION = 2.0           # plateau regen multiplier (first guess)
TERRAIN_GATE_SOFTNESS = 0.08
TERRAIN_RIDGE_HEIGHT = _TERRAIN_RIDGE_HEIGHT_DEFAULT   # 0.15 (first guess)
SEED = 0

RESIDENT_CLIMB = 0.05
MUTANT_CLIMB = 0.10
GEN0_CLIMB_GRID = [0.0, 0.05, 0.10, 0.20, 0.40, 0.80, 1.0]
GEN0_HORIZON = 200          # canonical horizon for gen-0 measurement
GEN0_N_SEEDS = 3            # seeds to average over for gen-0 measurement

# Probability threshold for "effectively sealed" in the binary-sealed flood-fill.
SEAL_THRESHOLD = 0.01

OUTPUT_PATH = Path("experiments/outputs/exp235_manip_check.txt")

# Canonical EcologyConfig overrides for gen-0 measurement
# (matches the preflight YAML base_overrides for locomotion)
CANONICAL_CFG_OVERRIDES: dict[str, Any] = {
    "enable_terrain": True,
    "terrain_food_concentration": FOOD_CONCENTRATION,
    "terrain_gate_softness": TERRAIN_GATE_SOFTNESS,
    "terrain_ridge_height": TERRAIN_RIDGE_HEIGHT,
    "terrain_gates_movement": True,
    "climb_cost_floor": 0.05,
    "climb_cost_slope": 0.10,
    "horizon": GEN0_HORIZON,
    "mutation_rate": 0.0,
    "max_population": 500,
}


# ---------------------------------------------------------------------------
# Helpers — flood-fill instrument
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
    """
    assert world.elevation is not None
    n_cells = world.rows * world.cols
    best_prob = np.zeros(n_cells, dtype=np.float64)
    for c in founder_cells:
        best_prob[c] = 1.0

    # Bellman-Ford style: propagate max probabilities iteratively.
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
    """Return per-cell regen weight (proportional to steady-state regen rate).

    Uses ONLY plateau cells (elevation >= terrain_ridge_height - epsilon).
    No silent food inflation: raises ValueError if out_factor < 0 (matches step_regen).
    """
    assert world.elevation is not None
    n_cells = world.rows * world.cols
    food_concentration = world.terrain_food_concentration
    ridge_h = world.terrain_ridge_height
    _eps = 1e-9
    on_plateau = world.elevation >= (ridge_h - _eps)
    n_in = int(np.sum(on_plateau))
    n_out = n_cells - n_in
    weights = np.zeros(n_cells, dtype=np.float64)
    if food_concentration > 0.0 and n_in > 0:
        if n_out > 0:
            out_factor = (n_cells - n_in * food_concentration) / n_out
            if out_factor < 0.0:
                raise ValueError(
                    f"Flood-fill regen weights: out_factor={out_factor:.4f} < 0. "
                    f"Reduce terrain_food_concentration or plateau fraction."
                )
        else:
            out_factor = 0.0
        weights = np.where(on_plateau, food_concentration, out_factor).astype(np.float64)
    else:
        weights = np.ones(n_cells, dtype=np.float64)
    return weights


# ---------------------------------------------------------------------------
# Helpers — gen-0 simulation instrument
# ---------------------------------------------------------------------------

def _measure_plateau_intake_one_seed(
    base_cfg: EcologyConfig,
    climb: float,
    seed: int,
) -> float:
    """Run gen-0 at fixed climb_ability and return plateau_intake_share.

    plateau_intake_share = (food consumed on plateau cells) / (total food consumed).

    We monkey-patch world.consume() to accumulate per-cell-type intake.
    This is safe because consume() is a pure method: it only modifies world.resource
    and returns the consumed amount.  The patch wraps it to also accumulate telemetry.
    The engine itself is UNCHANGED; byte-identity of events_hash is UNAFFECTED (the
    telemetry arrays are outside events_hash).
    """
    # Build config with fixed climb, mutation_rate=0
    f = D.replace(make_founder(), climb_ability=climb)
    cfg = D.replace(base_cfg, founder=f)

    eco = Ecology(cfg, seed=seed)
    assert eco.world.elevation is not None

    world = eco.world
    ridge_h = world.terrain_ridge_height
    _eps = 1e-9

    # Accumulate intake telemetry (not in events_hash)
    intake_basin: float = 0.0
    intake_plateau: float = 0.0

    # Patch world.consume to accumulate by cell type
    _orig_consume = world.consume

    def _patched_consume(pos: int, amount: float) -> float:
        nonlocal intake_basin, intake_plateau
        consumed = _orig_consume(pos, amount)
        elev = float(world.elevation[pos])  # type: ignore[index]
        if elev >= (ridge_h - _eps):
            intake_plateau += consumed
        else:
            intake_basin += consumed
        return consumed

    world.consume = _patched_consume  # type: ignore[method-assign]

    # Run the simulation
    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        if not eco.has_alive():
            break

    world.consume = _orig_consume  # type: ignore[method-assign]

    total = intake_basin + intake_plateau
    if total < 1e-12:
        return float("nan")  # extinct / degenerate run
    return intake_plateau / total


def gen0_plateau_intake_share(
    base_cfg: EcologyConfig,
    climb: float,
    seeds: list[int],
) -> float:
    """Return mean plateau_intake_share(climb) over seeds, ignoring NaN seeds."""
    shares = [_measure_plateau_intake_one_seed(base_cfg, climb, s) for s in seeds]
    valid = [s for s in shares if not math.isnan(s)]
    if not valid:
        return float("nan")
    return sum(valid) / len(valid)


# ---------------------------------------------------------------------------
# Public API — manipulation_check
# ---------------------------------------------------------------------------

def manipulation_check(config: "dict[str, Any] | None" = None) -> dict:
    """Run the gen-0 simulation-based manipulation check and return a result dict.

    Parameters
    ----------
    config : optional dict of overrides on top of CANONICAL_CFG_OVERRIDES.
             Keys match EcologyConfig fields.  Pass None to use canonical defaults.

    Returns
    -------
    dict with keys:
        "resident_share"  : plateau_intake_share(resident) — mean over GEN0_N_SEEDS
        "mutant_share"    : plateau_intake_share(mutant) — mean over GEN0_N_SEEDS
        "marginal"        : mutant_share - resident_share
        "withheld_pass"   : bool — resident_share <= RESIDENT_SHARE_THRESHOLD
        "marginal_pass"   : bool — marginal >= MARGINAL_THRESHOLD
        "both_pass"       : bool — both criteria satisfied
        "thresholds"      : dict of the threshold values used
        "shares_grid"     : dict climb -> plateau_intake_share for GEN0_CLIMB_GRID
    """
    overrides = dict(CANONICAL_CFG_OVERRIDES)
    if config is not None:
        overrides.update(config)

    base_cfg = D.replace(SCENARIOS["balanced"], **overrides)
    seeds = list(range(GEN0_N_SEEDS))

    # Measure grid for curve
    shares_grid: dict[float, float] = {}
    for c in GEN0_CLIMB_GRID:
        shares_grid[c] = gen0_plateau_intake_share(base_cfg, c, seeds)

    resident_share = shares_grid.get(RESIDENT_CLIMB, float("nan"))
    mutant_share = shares_grid.get(MUTANT_CLIMB, float("nan"))

    if math.isnan(resident_share) or math.isnan(mutant_share):
        marginal = float("nan")
        withheld_pass = False
        marginal_pass = False
    else:
        marginal = mutant_share - resident_share
        withheld_pass = resident_share <= RESIDENT_SHARE_THRESHOLD
        marginal_pass = marginal >= MARGINAL_THRESHOLD

    return {
        "resident_share": resident_share,
        "mutant_share": mutant_share,
        "marginal": marginal,
        "withheld_pass": withheld_pass,
        "marginal_pass": marginal_pass,
        "both_pass": withheld_pass and marginal_pass,
        "thresholds": {
            "resident_share_threshold": RESIDENT_SHARE_THRESHOLD,
            "marginal_threshold": MARGINAL_THRESHOLD,
        },
        "shares_grid": shares_grid,
    }


# ---------------------------------------------------------------------------
# Main — full report
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
        terrain_ridge_height=TERRAIN_RIDGE_HEIGHT,
        terrain_gates_movement=True,
    )
    assert world.elevation is not None

    elev = world.elevation
    unique_elevs = sorted(set(float(e) for e in elev))
    n_cells = ROWS * COLS
    n_basin = int(np.sum(elev < TERRAIN_RIDGE_HEIGHT * 0.5))   # elevation ~0.0
    n_plateau = int(np.sum(elev >= TERRAIN_RIDGE_HEIGHT - 1e-9))

    # Founder spawn cells (basin-restricted — same logic as engine.py)
    basin_cells_list = [c for c in range(n_cells) if float(elev[c]) <= 0.0]
    n_founders = 12
    basin_step = max(1, len(basin_cells_list) // max(1, n_founders))
    founder_cells = [basin_cells_list[(i * basin_step) % len(basin_cells_list)]
                     for i in range(n_founders)]
    founder_cells_unique = sorted(set(founder_cells))

    regen_weights = compute_cell_regen_weights(world)
    total_regen = float(np.sum(regen_weights))

    # --- INSTRUMENT 1: Binary-sealed (flood-fill with threshold) ---
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

    # --- INSTRUMENT 1: Expected-regen approach ---
    res_expected = expected_reachable_regen(
        world, founder_cells_unique, RESIDENT_CLIMB, TERRAIN_GATE_SOFTNESS, regen_weights
    )
    mut_expected = expected_reachable_regen(
        world, founder_cells_unique, MUTANT_CLIMB, TERRAIN_GATE_SOFTNESS, regen_weights
    )
    withheld_expected = (total_regen - res_expected) / max(1e-9, total_regen)
    marginal_expected = (mut_expected - res_expected) / max(1e-9, total_regen - res_expected)

    # --- Per-elevation-transition crossing probabilities ---
    delta_rim = TERRAIN_RIDGE_HEIGHT   # basin (0.0) to plateau (terrain_ridge_height)
    p_rim_res = sigmoid((RESIDENT_CLIMB - delta_rim) / TERRAIN_GATE_SOFTNESS)
    p_rim_mut = sigmoid((MUTANT_CLIMB - delta_rim) / TERRAIN_GATE_SOFTNESS)

    # --- INSTRUMENT 2: Gen-0 simulation-based ---
    print("\nRunning gen-0 simulation-based measurement (this takes ~30-60s)...")
    result = manipulation_check()
    shares_grid = result["shares_grid"]
    resident_share = result["resident_share"]
    mutant_share = result["mutant_share"]
    marginal = result["marginal"]

    # Build output
    plateau_regen = sum(float(regen_weights[c]) for c in range(n_cells) if float(elev[c]) >= TERRAIN_RIDGE_HEIGHT - 1e-9)
    basin_regen = total_regen - plateau_regen

    out_factor_check = (n_cells - n_plateau * FOOD_CONCENTRATION) / max(1, n_cells - n_plateau)

    lines = [
        "=" * 70,
        "EXP 235 MANIPULATION CHECK — Binary Terrain + Gen-0 Simulation",
        "=" * 70,
        "",
        "SUBSTRATE GEOMETRY (BINARY — single crossing)",
        f"  Grid: {ROWS}x{COLS} = {n_cells} cells",
        f"  Unique elevations: {unique_elevs}",
        f"  Basin cells (elev=0.0): {n_basin}  ({n_basin/n_cells*100:.0f}%)",
        f"  Plateau cells (elev={TERRAIN_RIDGE_HEIGHT}): {n_plateau}  ({n_plateau/n_cells*100:.0f}%)",
        f"  Terrain ridge height: {TERRAIN_RIDGE_HEIGHT}",
        f"  Terrain gate softness: {TERRAIN_GATE_SOFTNESS}",
        f"  Plateau food_concentration: {FOOD_CONCENTRATION}",
        f"  Conserved-total out_factor: {out_factor_check:.4f} {'(OK)' if out_factor_check >= 0 else '(ERROR: NEGATIVE)'}",
        "",
        "REGEN WEIGHTS (proportional to per-cell steady-state regen)",
        f"  Total weight: {total_regen:.4f}",
        f"  Basin weight: {basin_regen:.4f}  ({basin_regen/total_regen*100:.1f}%)",
        f"  Plateau weight: {plateau_regen:.4f}  ({plateau_regen/total_regen*100:.1f}%)",
        "",
        "FOUNDER SPAWN (basin-restricted)",
        f"  Unique founder cells: {founder_cells_unique}",
        f"  Count: {len(founder_cells_unique)} (from {n_founders} founders)",
        "",
        "CROSSING PROBABILITY (single rim delta={:.3f})".format(delta_rim),
        f"  climb={RESIDENT_CLIMB}: p = {p_rim_res:.2e}  ({p_rim_res*100:.4f}%/step)",
        f"  climb={MUTANT_CLIMB}: p = {p_rim_mut:.2e}  ({p_rim_mut*100:.4f}%/step)",
        f"  ratio: {p_rim_mut/max(1e-15, p_rim_res):.2f}x improvement",
        "",
        "=" * 70,
        "INSTRUMENT 1 — Static Flood-Fill",
        "=" * 70,
        "",
        "--- Binary-sealed (flood-fill, threshold={:.3f}) ---".format(SEAL_THRESHOLD),
        f"  Resident (climb={RESIDENT_CLIMB}): {len(reached_res_binary)} cells reachable, regen={res_regen_binary:.4f}",
        f"  Mutant   (climb={MUTANT_CLIMB}): {len(reached_mut_binary)} cells reachable, regen={mut_regen_binary:.4f}",
        f"  (a) WITHHELD-REGEN at resident: {withheld_binary*100:.1f}%",
        f"      Required: ≥30%  |  Status: {'PASS' if withheld_binary >= 0.30 else 'FAIL'}",
        f"  (b) MARGINAL UNLOCKED (0.05->0.10): {marginal_unlocked_binary*100:.1f}%",
        f"      Status: {'LARGE' if marginal_unlocked_binary >= 0.05 else 'SMALL'}",
        "",
        "--- Expected-regen approach (best-path probability * regen_weight) ---",
        f"  Resident (climb={RESIDENT_CLIMB}): expected regen = {res_expected:.6f} (of {total_regen:.4f})",
        f"  Mutant   (climb={MUTANT_CLIMB}): expected regen = {mut_expected:.6f}",
        f"  (a) WITHHELD-REGEN at resident: {withheld_expected*100:.2f}%",
        f"      Required: ≥30%  |  Status: {'PASS' if withheld_expected >= 0.30 else 'FAIL'}",
        f"  (b) MARGINAL UNLOCKED (0.05->0.10): {marginal_expected*100:.4f}%",
        f"      Status: {'LARGE' if marginal_expected >= 0.05 else 'SMALL — crossing too tight at both values'}",
        "",
        "=" * 70,
        "INSTRUMENT 2 — Gen-0 Simulation (BINDING)",
        f"  (horizon={GEN0_HORIZON}, seeds={list(range(GEN0_N_SEEDS))}, mutation_rate=0)",
        "=" * 70,
        "",
        "  CLIMB GRID — plateau_intake_share(c) = (plateau_food_consumed)/(total_food_consumed):",
    ]
    for c in GEN0_CLIMB_GRID:
        share = shares_grid.get(c, float("nan"))
        if math.isnan(share):
            lines.append(f"    climb={c:.2f}: share=NaN (extinct)")
        else:
            lines.append(f"    climb={c:.2f}: share={share:.4f}  ({share*100:.1f}%)")

    lines += [
        "",
        f"  Resident (climb={RESIDENT_CLIMB}): plateau_intake_share = {resident_share:.4f}  ({resident_share*100:.1f}%)",
        f"  Mutant   (climb={MUTANT_CLIMB}): plateau_intake_share = {mutant_share:.4f}  ({mutant_share*100:.1f}%)",
        f"  Marginal: {marginal:.4f}  ({marginal*100:.1f} pp)",
        "",
        "PASS/FAIL (thresholds: resident ≤ {:.0f}%, marginal ≥ {:.0f}pp):".format(
            RESIDENT_SHARE_THRESHOLD * 100, MARGINAL_THRESHOLD * 100
        ),
        f"  (a) WITHHELD: resident_share={resident_share:.4f} ≤ {RESIDENT_SHARE_THRESHOLD} ?  "
        f"  {'PASS' if result['withheld_pass'] else 'FAIL — RESIDENT NOT SEALED (substrate too legible)'}",
        f"  (b) MARGINAL: marginal={marginal:.4f} ≥ {MARGINAL_THRESHOLD} ?  "
        f"  {'PASS' if result['marginal_pass'] else 'FAIL — MARGINAL TOO SMALL (retune geometry)'}",
        f"  OVERALL: {'PASS — substrate valid, proceed to gradient batch' if result['both_pass'] else 'FAIL — substrate invalid, retune before gradient batch'}",
        "",
        "NOTE: Exact tuning is the human's job. These are first-guess numbers.",
        "      Rerun this script after retuning terrain_ridge_height / terrain_gate_softness.",
        "      The test in tests/test_ecology_terrain.py::TestManipulationCheck reads",
        "      thresholds from this module (RESIDENT_SHARE_THRESHOLD, MARGINAL_THRESHOLD).",
        "=" * 70,
    ]

    output = "\n".join(lines)
    print(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(output + "\n")
    print(f"\n[Written to {OUTPUT_PATH}]")


if __name__ == "__main__":
    main()
