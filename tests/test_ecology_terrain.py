"""tests/test_ecology_terrain.py — Terrain substrate (Exp 235) tests.

Covers:
  1. OFF byte-identity: events_hash is UNCHANGED across {enable_terrain=False} x
     {climb_ability in {0.0, 0.05, 0.5, 1.0}} for the committed 'balanced' / 'lean' regimes.
  2. Cost-divergence (ON causally live): events_hash DIVERGES when enable_terrain=True
     with two different climb_ability founders (mirrors test_probe_cost_is_paid).
  3. One golden hash pin for an enable_terrain=True run (catch ON-path drift).
     NOTE: This hash tracks the CANONICAL geometry (binary basin/plateau with
     terrain_ridge_height=0.15, plateau=top 1/3 rows). It is EXPECTED to change when
     the human retunes geometry parameters; repin it at that point with a clear comment.
  4. Disconnect overrides byte-identity assertion (Gate-G Guard 1 style): with the axis
     disconnect overrides applied, the event hash is identical across climb_ability values.
  5. Manipulation check (gen-0 simulation): plateau_intake_share criteria from
     experiments/exp235_manip_check.py.

Design:
  - Mirrors tests/test_active_sensing.py + tests/test_ecology_thermosense.py structure.
  - The OFF byte-identity test is the HARD ANTI-CHEAT guard: if it fails, it is a real bug
    in the gating, NOT a hash to update.
"""
from __future__ import annotations

import dataclasses as D

import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS
from ecology.genotype import Genotype, founder, mutate
from ecology.evolvability.trait_axis import make_axis, LOCOMOTION_AXIS
from ecology.world import GridWorld
from experiments.exp235_manip_check import (
    manipulation_check,
    CANONICAL_CFG_OVERRIDES,
    RESIDENT_SHARE_THRESHOLD,
    MARGINAL_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terrain_cfg(**over) -> EcologyConfig:
    """Base terrain config (ON) with sensible defaults for fast tests.

    Uses terrain_food_concentration=2.0 (conserved-total OK: 48 plateau / 144 total,
    out_factor = (144-48*2)/96 = 0.5 >= 0).  The old value of 4.0 was invalid with
    the new binary geometry (out_factor = -0.5 < 0).
    """
    base = dict(
        enable_terrain=True,
        terrain_food_concentration=2.0,    # plateau carries 2x regen (conserved-total valid)
        terrain_gate_softness=0.08,
        terrain_ridge_height=0.15,
        climb_cost_floor=0.05,
        climb_cost_slope=0.10,
        terrain_gates_movement=True,
        horizon=150,
        max_population=500,
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


def _founder_with_climb(climb: float) -> Genotype:
    """Return a founder genotype with climb_ability set to climb."""
    return D.replace(founder(), climb_ability=climb)


# ---------------------------------------------------------------------------
# Test 1: OFF byte-identity across climb_ability values
# ---------------------------------------------------------------------------

class TestOffByteIdentity:
    """HARD ANTI-CHEAT: with enable_terrain=False, events_hash must be IDENTICAL
    across ALL climb_ability values.  Any deviation is a real gating bug, not a
    hash to update."""

    def _run_off(self, climb: float, scenario: str = "balanced", seed: int = 7) -> str:
        cfg = D.replace(
            SCENARIOS[scenario],
            enable_terrain=False,
            mutation_rate=0.0,
            horizon=120,
        )
        cfg = D.replace(cfg, founder=_founder_with_climb(climb))
        eco = Ecology(cfg, seed=seed)
        return eco.run()["events_hash"]

    def test_off_byte_identical_balanced_climb_values(self) -> None:
        """balanced, seed=7, OFF: hash must be the SAME for climb in {0.0, 0.05, 0.5, 1.0}."""
        climbs = [0.0, 0.05, 0.5, 1.0]
        hashes = [self._run_off(c, "balanced", 7) for c in climbs]
        assert len(set(hashes)) == 1, (
            f"OFF byte-identity BROKEN for 'balanced' regime! "
            f"Different hashes across climb_ability {climbs}:\n"
            + "\n".join(f"  climb={c}: {h}" for c, h in zip(climbs, hashes))
            + "\nThis is a REAL BUG in the gating — do NOT update hashes to fix this."
        )

    def test_off_byte_identical_scarce_climb_values(self) -> None:
        """scarce, seed=3, OFF: hash must be the SAME for climb in {0.0, 0.05, 0.5, 1.0}."""
        climbs = [0.0, 0.05, 0.5, 1.0]
        hashes = [self._run_off(c, "scarce", 3) for c in climbs]
        assert len(set(hashes)) == 1, (
            f"OFF byte-identity BROKEN for 'scarce' regime! "
            f"Different hashes across climb_ability {climbs}:\n"
            + "\n".join(f"  climb={c}: {h}" for c, h in zip(climbs, hashes))
            + "\nThis is a REAL BUG in the gating — do NOT update hashes to fix this."
        )

    def test_off_path_does_not_mutate_climb_ability(self) -> None:
        """With enable_terrain=False, climb_ability never gets drawn in mutate()."""
        # Run two otherwise-identical runs: one with climb=0.0, one with climb=1.0.
        # If mutate() draws for climb_ability when OFF, the rng streams diverge and
        # hashes differ — this is exactly what we're guarding against.
        h1 = self._run_off(0.0, "balanced", 42)
        h2 = self._run_off(1.0, "balanced", 42)
        assert h1 == h2, (
            f"OFF path rng stream DIVERGES between climb=0.0 and climb=1.0!\n"
            f"  climb=0.0: {h1}\n"
            f"  climb=1.0: {h2}\n"
            "The mutate() skip-guard for LOCOMOTION_TRAITS is broken."
        )


# ---------------------------------------------------------------------------
# Test 2: Cost-divergence — ON path is causally live
# ---------------------------------------------------------------------------

class TestCostDivergenceOnPath:
    """When enable_terrain=True, different climb_ability founders must produce
    DIFFERENT events_hash — proof the terrain mechanism is causally active."""

    def test_climb_divergence_on(self) -> None:
        """ON-path: events_hash DIVERGES between climb=0.0 and climb=1.0 founders."""
        def run(climb: float) -> str:
            cfg = _terrain_cfg(mutation_rate=0.0, horizon=100)
            cfg = D.replace(cfg, founder=_founder_with_climb(climb))
            return Ecology(cfg, seed=1).run()["events_hash"]

        h_low = run(0.0)
        h_high = run(1.0)
        assert h_low != h_high, (
            "Terrain mechanism is NOT causally active: events_hash is the SAME for "
            "climb_ability=0.0 and climb_ability=1.0 when enable_terrain=True.\n"
            f"  Both: {h_low}"
        )

    def test_climb_cost_divergence(self) -> None:
        """Changing climb_cost_slope (0 vs nonzero) with enable_terrain=True diverges hash."""
        def run(cost_slope: float) -> str:
            cfg = _terrain_cfg(mutation_rate=0.0, horizon=80, climb_cost_slope=cost_slope)
            cfg = D.replace(cfg, founder=_founder_with_climb(0.5))
            return Ecology(cfg, seed=2).run()["events_hash"]

        h_free = run(0.0)
        h_paid = run(0.5)
        assert h_free != h_paid, (
            "Climb cost is not causally active: changing climb_cost_slope 0->0.5 did NOT "
            "change the trajectory when enable_terrain=True."
        )


# ---------------------------------------------------------------------------
# Test 3: Golden hash pin for one ON-path run
# ---------------------------------------------------------------------------

# Golden hash for a fixed enable_terrain=True run (seed=5, horizon=100, no mutation,
# zero climb cost, climb=0.05 founder, food_concentration=2.0).
#
# TRACKS THE CANONICAL GEOMETRY: binary basin/plateau (terrain_ridge_height=0.15,
# plateau=top 1/3 of rows: rows 8-11 in 12x12 = 48 cells).
#
# This hash is EXPECTED to change when the human retunes geometry (e.g. terrain_ridge_height
# or terrain_gate_softness). At that point, repin it with a comment noting the new geometry.
# DO NOT update to fix a test caused by an OFF-path change — investigate the gating instead.
_TERRAIN_ON_GOLDEN_SEED = 5
# Pinned for canonical geometry: binary basin/plateau (terrain_ridge_height=0.15,
# plateau = top 1/3 of rows: rows 8-11 in 12x12 = 48 cells, basin = rows 0-7 = 96 cells).
# terrain_food_concentration=2.0 → out_factor=0.5. Seed=5, horizon=100, no mutation, no cost.
# This hash WILL change when the human retunes geometry; repin at that point with a new comment.
_TERRAIN_ON_GOLDEN_HASH = "1620e35d35bf4937da9a7f5e821d4a952ceaa544536752cf7004515a50f0a58e"


class TestGoldenHashTerrainOn:
    """Golden hash pin for a fixed enable_terrain=True run.

    The hash tracks the CANONICAL GEOMETRY (binary basin/plateau, terrain_ridge_height=0.15,
    plateau=top 1/3 rows). It will change when the human retunes geometry — at that point
    repin _TERRAIN_ON_GOLDEN_HASH with a comment noting the new geometry.

    Any change from a NON-geometry ON-path code change must be investigated.
    Do NOT update this hash to fix a test without understanding the cause.
    """

    def _run_terrain_on(self, seed: int = _TERRAIN_ON_GOLDEN_SEED) -> dict:
        # Zero climb cost + zero mutation so the hash is stable across population outcomes.
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=True,
            terrain_food_concentration=2.0,
            terrain_gate_softness=0.08,
            terrain_ridge_height=0.15,   # canonical geometry (binary, single crossing)
            climb_cost_floor=0.0,
            climb_cost_slope=0.0,
            terrain_gates_movement=True,
            horizon=100,
            max_population=500,
            mutation_rate=0.0,
        )
        cfg = D.replace(cfg, founder=_founder_with_climb(0.05))
        eco = Ecology(cfg, seed=seed)
        return eco.run()

    def test_terrain_on_golden_hash_stable(self) -> None:
        """Pin the ON-path events_hash for the canonical binary geometry.

        Hash tracks: binary basin/plateau, terrain_ridge_height=0.15, plateau=top 1/3 rows,
        terrain_food_concentration=2.0, seed=5, horizon=100, no mutation, no climb cost.

        This hash WILL change when the human retunes geometry (expected). At that point,
        repin _TERRAIN_ON_GOLDEN_HASH with a new comment documenting the geometry change.
        A change from a non-geometry ON-path code edit must be investigated.
        """
        result = self._run_terrain_on()
        h = result["events_hash"]
        assert h == _TERRAIN_ON_GOLDEN_HASH, (
            f"ON-path golden hash changed!\n"
            f"  Got:      {h}\n"
            f"  Expected: {_TERRAIN_ON_GOLDEN_HASH}\n"
            "If this change is due to geometry retuning, repin the hash with a comment.\n"
            "If this is an unexpected ON-path change, investigate — do not update blindly."
        )


# ---------------------------------------------------------------------------
# Test 4: Disconnect overrides byte-identity (Gate-G Guard 1 style)
# ---------------------------------------------------------------------------

class TestDisconnectOverridesByteIdentity:
    """With the LOCOMOTION_AXIS disconnect_overrides applied, the events_hash must be
    byte-identical across climb_ability values.  This is the anti-cheat guard verifying
    that ALL channels climb_ability feeds have been enumerated and can be turned off."""

    def _run_disconnected(self, climb: float, seed: int = 7) -> str:
        """Run with ALL disconnect_overrides from LOCOMOTION_AXIS applied."""
        overrides = LOCOMOTION_AXIS.disconnect_overrides
        cfg = D.replace(
            SCENARIOS["balanced"],
            horizon=120,
            mutation_rate=0.0,
            **overrides,
        )
        cfg = D.replace(cfg, founder=_founder_with_climb(climb))
        eco = Ecology(cfg, seed=seed)
        return eco.run()["events_hash"]

    def test_disconnect_overrides_byte_identical_across_climb(self) -> None:
        """Disconnect ALL climb_ability channels: hashes must be identical across climb values."""
        climbs = [0.0, 0.05, 0.5, 1.0]
        hashes = [self._run_disconnected(c, seed=7) for c in climbs]
        assert len(set(hashes)) == 1, (
            "Disconnect overrides byte-identity FAILED — a channel is missing from "
            "LOCOMOTION_AXIS.disconnect_overrides!\n"
            + "\n".join(f"  climb={c}: {h}" for c, h in zip(climbs, hashes))
            + "\nAdd the leaking channel to disconnect_overrides."
        )

    def test_disconnect_overrides_byte_identical_seed2(self) -> None:
        """Verify disconnect byte-identity holds on a second seed."""
        climbs = [0.0, 0.5, 1.0]
        hashes = [self._run_disconnected(c, seed=2) for c in climbs]
        assert len(set(hashes)) == 1, (
            "Disconnect overrides byte-identity FAILED on seed=2:\n"
            + "\n".join(f"  climb={c}: {h}" for c, h in zip(climbs, hashes))
        )


# ---------------------------------------------------------------------------
# Test 5: Terrain mechanics smoke tests
# ---------------------------------------------------------------------------

class TestTerrainMechanicsSmoke:
    """Quick smoke tests that the ON-path mechanics work without error."""

    def test_terrain_on_run_completes(self) -> None:
        """enable_terrain=True run completes and returns events_hash."""
        cfg = _terrain_cfg(horizon=50)
        eco = Ecology(cfg, seed=0)
        result = eco.run()
        assert "events_hash" in result, "events_hash missing from terrain ON result"
        assert len(result["events_hash"]) == 64, "events_hash is wrong length"

    def test_terrain_off_preserves_existing_hashes(self) -> None:
        """enable_terrain=False with default founder reproduces Exp 194 baseline hash.

        This verifies the OFF path is completely byte-identical to the pre-terrain code.
        The EXP194 hash is the canonical regression guard.
        """
        EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"
        cfg = D.replace(
            SCENARIOS["balanced"],
            complexity_cost_scale=0.0,
            enable_temperature=False,
            enable_thermosense=False,
            enable_terrain=False,
        )
        eco = Ecology(cfg, seed=0)
        result = eco.run()
        assert result["events_hash"] == EXP194_HASH, (
            f"Terrain OFF broke Exp 194 byte-identity!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {EXP194_HASH}\n"
            "This is a REAL BUG — the OFF path must be perfectly byte-identical."
        )

    def test_elevation_field_built_only_when_on(self) -> None:
        """Elevation field is None when enable_terrain=False, non-None when True."""
        cfg_off = D.replace(SCENARIOS["balanced"], enable_terrain=False)
        eco_off = Ecology(cfg_off, seed=0)
        assert eco_off.world.elevation is None, (
            "elevation field should be None when enable_terrain=False"
        )

        cfg_on = D.replace(SCENARIOS["balanced"], enable_terrain=True)
        eco_on = Ecology(cfg_on, seed=0)
        assert eco_on.world.elevation is not None, (
            "elevation field should be allocated when enable_terrain=True"
        )

    def test_climbable_neighbors_deterministic_downhill(self) -> None:
        """climbable_neighbors: downhill neighbors always admitted (no rng needed).

        With BINARY geometry: basin cells at 0.0, plateau at terrain_ridge_height.
        From a basin cell, flat basin neighbors are always admitted (delta=0, prob=1).
        From a plateau cell, downhill basin neighbors are always admitted (delta<0, prob=1).
        """
        cfg = D.replace(SCENARIOS["balanced"], enable_terrain=True)
        eco = Ecology(cfg, seed=0)
        world = eco.world
        assert world.elevation is not None

        rng = __import__("numpy").random.default_rng(42)
        # Basin: bottom 2/3 of rows (rows 0-7 for 12-row grid).
        # Cell at row=0, col=0 is basin (elevation=0.0).
        basin_pos = 0  # row=0, col=0, elevation=0.0
        # From a basin cell, flat basin neighbors (row=1, col=1) are admitted.
        nb = world.climbable_neighbors(basin_pos, 0.0, rng)
        # At least some neighbors should be admitted (flat/downhill ones).
        assert len(nb) >= 1, (
            "climbable_neighbors returned empty list for a basin cell with climb=0.0 "
            "— flat/downhill neighbors should always be admitted."
        )

    def test_binary_elevation_geometry(self) -> None:
        """Binary elevation: only two elevations (0.0 and terrain_ridge_height)."""
        cfg = D.replace(SCENARIOS["balanced"], enable_terrain=True, terrain_ridge_height=0.15)
        eco = Ecology(cfg, seed=0)
        world = eco.world
        assert world.elevation is not None

        unique_elevs = sorted(set(float(e) for e in world.elevation))
        assert len(unique_elevs) == 2, (
            f"Expected exactly 2 elevation levels (binary geometry), got {unique_elevs}"
        )
        assert abs(unique_elevs[0] - 0.0) < 1e-9, "Basin elevation should be 0.0"
        assert abs(unique_elevs[1] - 0.15) < 1e-9, (
            f"Plateau elevation should be terrain_ridge_height=0.15, got {unique_elevs[1]}"
        )

    def test_conserved_regen_total(self) -> None:
        """Terrain ON regen total matches terrain OFF (conserved-total guarantee)."""
        import numpy as np

        # Build two worlds: one ON, one OFF
        rng_on = __import__("numpy").random.default_rng(99)
        world_on = GridWorld.from_config(
            rows=12, cols=12, capacity=10.0, regen_rate=0.20,
            initial_resource=0.5, rng=rng_on,
            enable_terrain=True,
            terrain_food_concentration=2.0,
            terrain_ridge_height=0.15,
            terrain_gate_softness=0.08,
        )
        rng_off = __import__("numpy").random.default_rng(99)
        world_off = GridWorld.from_config(
            rows=12, cols=12, capacity=10.0, regen_rate=0.20,
            initial_resource=0.5, rng=rng_off,
            enable_terrain=False,
        )
        from ecology.world import GridWorld as _GW

        # Measure regen added in one step (start from empty to avoid capacity clamping)
        world_on.resource[:] = 0.0
        world_off.resource[:] = 0.0
        before_on = float(np.sum(world_on.resource))
        before_off = float(np.sum(world_off.resource))
        world_on.step_regen()
        world_off.step_regen()
        regen_on = float(np.sum(world_on.resource)) - before_on
        regen_off = float(np.sum(world_off.resource)) - before_off

        # They should be approximately equal (conserved-total)
        assert abs(regen_on - regen_off) < 0.01, (
            f"Terrain regen total NOT conserved!\n"
            f"  ON: {regen_on:.6f}\n"
            f"  OFF: {regen_off:.6f}\n"
            f"  Diff: {abs(regen_on - regen_off):.6f}"
        )

    def test_out_factor_negative_raises(self) -> None:
        """step_regen raises ValueError if out_factor < 0 (no silent food inflation)."""
        import numpy as np

        # Build a world where plateau fraction is too large for the concentration
        # plateau = top 1/3 (48 cells), concentration = 10.0 → out_factor < 0
        rng = __import__("numpy").random.default_rng(0)
        world = GridWorld.from_config(
            rows=12, cols=12, capacity=10.0, regen_rate=0.20,
            initial_resource=0.5, rng=rng,
            enable_terrain=True,
            terrain_food_concentration=10.0,  # way too high → out_factor < 0
            terrain_ridge_height=0.15,
        )
        world.resource[:] = 0.0
        with pytest.raises(ValueError, match="out_factor"):
            world.step_regen()


# ---------------------------------------------------------------------------
# Test 6: Manipulation check (gen-0 simulation)
# ---------------------------------------------------------------------------

class TestManipulationCheck:
    """Gen-0 simulation-based manipulation check (L38 reachability check, Exp 235 Rung 1).

    Reads thresholds and canonical config from experiments/exp235_manip_check.py so
    updating the geometry/thresholds there automatically updates this test.

    NOTE: With first-guess geometry (terrain_ridge_height=0.15, softness=0.08), the
    crossing probability per step is very low, so the resident and mutant shares may
    be nearly identical and both FAIL the criteria.  This is the EXPECTED outcome:
    the structure is enforced, not the calibration.  The human will retune the geometry.
    """

    @pytest.mark.slow
    def test_manipulation_check_canonical(self) -> None:
        """Run gen-0 manipulation check on the canonical config and assert criteria.

        STRUCTURE ENFORCED: the function runs without error and returns the right keys.
        CALIBRATION: the PASS/FAIL is logged, not hard-enforced here (geometry is first-guess).
        """
        result = manipulation_check()

        # Structural assertions (always required)
        assert "resident_share" in result
        assert "mutant_share" in result
        assert "marginal" in result
        assert "withheld_pass" in result
        assert "marginal_pass" in result
        assert "both_pass" in result
        assert "shares_grid" in result

        rs = result["resident_share"]
        ms = result["mutant_share"]
        mg = result["marginal"]

        # Shares must be in [0, 1]
        if rs == rs:  # not NaN
            assert 0.0 <= rs <= 1.0, f"resident_share out of range: {rs}"
        if ms == ms:
            assert 0.0 <= ms <= 1.0, f"mutant_share out of range: {ms}"

        # Log the result (the human will check and decide on geometry)
        import math
        print(
            f"\n[ManipulationCheck] resident_share={rs:.4f}, mutant_share={ms:.4f}, "
            f"marginal={mg:.4f}, "
            f"withheld_pass={result['withheld_pass']}, marginal_pass={result['marginal_pass']}"
        )
        print(
            f"  Thresholds: resident ≤ {RESIDENT_SHARE_THRESHOLD}, "
            f"marginal ≥ {MARGINAL_THRESHOLD}"
        )
