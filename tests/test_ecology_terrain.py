"""tests/test_ecology_terrain.py — Terrain substrate (Exp 235) tests.

Covers:
  1. OFF byte-identity: events_hash is UNCHANGED across {enable_terrain=False} x
     {climb_ability in {0.0, 0.05, 0.5, 1.0}} for the committed 'balanced' / 'lean' regimes.
  2. Cost-divergence (ON causally live): events_hash DIVERGES when enable_terrain=True
     with two different climb_ability founders (mirrors test_probe_cost_is_paid).
  3. One golden hash pin for an enable_terrain=True run (catch ON-path drift).
  4. Disconnect overrides byte-identity assertion (Gate-G Guard 1 style): with the axis
     disconnect overrides applied, the event hash is identical across climb_ability values.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _terrain_cfg(**over) -> EcologyConfig:
    """Base terrain config (ON) with sensible defaults for fast tests."""
    base = dict(
        enable_terrain=True,
        terrain_food_concentration=4.0,    # plateau carries 4x regen
        terrain_gate_softness=0.08,
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
# Pinned from the first correct implementation of the terrain substrate.
# DO NOT update this to fix a test — investigate the ON-path change instead.
_TERRAIN_ON_GOLDEN_SEED = 5
_TERRAIN_ON_GOLDEN_HASH = "9f105b95c544ee536911fdb857f0dff1a7927096b23b3574a03e9bda29cb17a5"


class TestGoldenHashTerrainOn:
    """Golden hash pin for a fixed enable_terrain=True run.

    The hash is pinned from the first correct implementation and must not change.
    Any change to the ON-path code will break this test — investigate, do not update.
    """

    def _run_terrain_on(self, seed: int = _TERRAIN_ON_GOLDEN_SEED) -> dict:
        # Zero climb cost + zero mutation so the hash is stable across population outcomes.
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=True,
            terrain_food_concentration=2.0,
            terrain_gate_softness=0.08,
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
        """Pin the ON-path events_hash for a fixed config/seed (regression guard)."""
        result = self._run_terrain_on()
        h = result["events_hash"]
        assert h == _TERRAIN_ON_GOLDEN_HASH, (
            f"ON-path golden hash changed!\n"
            f"  Got:      {h}\n"
            f"  Expected: {_TERRAIN_ON_GOLDEN_HASH}\n"
            "DO NOT update this hash to fix the test — investigate the ON-path change."
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
        """climbable_neighbors: downhill neighbors always admitted (no rng needed)."""
        cfg = D.replace(SCENARIOS["balanced"], enable_terrain=True)
        eco = Ecology(cfg, seed=0)
        world = eco.world
        assert world.elevation is not None

        rng = __import__("numpy").random.default_rng(42)
        # Find a plateau cell with basin neighbors (elevation transition row)
        # Row basin_rows = 4 (for 12 rows): cells at row 4 are ridge (elevation=1.0).
        # A cell on the plateau (row 5) looking downward toward row 4 is upslope from row 4.
        # A cell in the basin looking upward sees an upslope edge.
        # Cells at row 0, col 0..11 are basin (elevation=0.0).
        basin_pos = 0  # row=0, col=0, elevation=0.0
        # From a basin cell, neighbors at row=1 (also basin) are flat — always admitted.
        nb = world.climbable_neighbors(basin_pos, 0.0, rng)
        # At least some neighbors should be admitted (downhill/flat ones).
        # Right neighbor (col=1) and down neighbor (row=1) are both basin => admitted.
        assert len(nb) >= 1, (
            "climbable_neighbors returned empty list for a basin cell with climb=0.0 "
            "— flat/downhill neighbors should always be admitted."
        )
