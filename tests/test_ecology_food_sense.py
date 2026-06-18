"""tests/test_ecology_food_sense.py — Food-sense perception (Exp 237) tests.

Covers:
  (a) OFF byte-identity: enable_food_sense=False preserves the Exp 194 golden hash
      AND the Exp 235 terrain-ON golden hash. Any failure is a real gating bug.
  (b) Anti-gaming flat-world guard: on a flat-resource world (terrain_food_concentration=0,
      so the plateau is NOT richer than the basin), food-sense must NOT preferentially
      drive creatures to the high-index plateau region (per L40: food-driven, not index-driven).
  (c) Smoke: enable_food_sense=True run completes without error.
"""
from __future__ import annotations

import dataclasses as D

import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS
from ecology.genotype import Genotype, founder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _founder_with_climb(climb: float) -> Genotype:
    return D.replace(founder(), climb_ability=climb)


# ---------------------------------------------------------------------------
# (a) OFF byte-identity
# ---------------------------------------------------------------------------

class TestOffByteIdentity:
    """HARD ANTI-CHEAT: enable_food_sense=False must be byte-identical to Exp 194
    and Exp 235 terrain-ON golden hashes.  Any failure is a REAL gating bug."""

    EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"
    TERRAIN_ON_GOLDEN_HASH = "1620e35d35bf4937da9a7f5e821d4a952ceaa544536752cf7004515a50f0a58e"

    def test_food_sense_off_preserves_exp194_hash(self) -> None:
        """enable_food_sense=False with all terrain OFF must reproduce the Exp 194 hash."""
        cfg = D.replace(
            SCENARIOS["balanced"],
            complexity_cost_scale=0.0,
            enable_temperature=False,
            enable_thermosense=False,
            enable_terrain=False,
            enable_navigation=False,
            enable_food_sense=False,  # explicit OFF
        )
        eco = Ecology(cfg, seed=0)
        result = eco.run()
        assert result["events_hash"] == self.EXP194_HASH, (
            f"Food-sense OFF broke Exp 194 byte-identity!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {self.EXP194_HASH}\n"
            "This is a REAL BUG — enable_food_sense=False must be perfectly byte-identical."
        )

    def test_food_sense_off_preserves_terrain_on_golden_hash(self) -> None:
        """enable_food_sense=False with terrain ON must reproduce the Exp 235 terrain-ON hash."""
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=True,
            terrain_food_concentration=2.0,
            terrain_gate_softness=0.08,
            terrain_ridge_height=0.15,
            climb_cost_floor=0.0,
            climb_cost_slope=0.0,
            terrain_gates_movement=True,
            horizon=100,
            max_population=500,
            mutation_rate=0.0,
            enable_navigation=False,
            enable_food_sense=False,  # explicit OFF — must not change terrain-ON hash
        )
        cfg = D.replace(cfg, founder=_founder_with_climb(0.05))
        eco = Ecology(cfg, seed=5)
        result = eco.run()
        assert result["events_hash"] == self.TERRAIN_ON_GOLDEN_HASH, (
            f"Food-sense OFF broke Exp 235 terrain-ON byte-identity!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {self.TERRAIN_ON_GOLDEN_HASH}\n"
            "This is a REAL BUG — enable_food_sense=False must not alter terrain-ON hash."
        )

    def test_food_sense_default_is_off(self) -> None:
        """Verify the default (no enable_food_sense field) matches explicit False."""
        base = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=False,
            mutation_rate=0.0,
            horizon=80,
        )
        cfg_default = base
        cfg_explicit = D.replace(base, enable_food_sense=False)
        h1 = Ecology(cfg_default, seed=13).run()["events_hash"]
        h2 = Ecology(cfg_explicit, seed=13).run()["events_hash"]
        assert h1 == h2, (
            f"enable_food_sense=False (explicit) diverges from default!\n"
            f"  default: {h1}\n"
            f"  explicit False: {h2}"
        )


# ---------------------------------------------------------------------------
# (b) Anti-gaming flat-world guard (L40 binding)
# ---------------------------------------------------------------------------

class TestAntiGamingFlatWorld:
    """L40 anti-gaming guard: on a flat-resource world (terrain_food_concentration=0),
    food-sense must NOT preferentially drive creatures to the high-index plateau region.

    With uniform regen the plateau is NOT richer, so food-scent must not manufacture
    plateau-seeking by index/position artifact.  Plateau occupancy must be ~ uniform.

    PASS condition: food-sense plateau occupancy share < 0.45 on a flat world
    (the plateau is top 1/3 of rows = ~33% of cells; we allow up to 45% to
    account for noise, but not systematic over-representation driven by index).
    """

    def _measure_plateau_occupancy_share(
        self,
        n_seeds: int = 4,
        horizon: int = 150,
    ) -> float:
        """Return mean fraction of creature-steps spent on the plateau (flat regen world)."""
        shares: list[float] = []
        for seed in range(n_seeds):
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_terrain=True,
                terrain_food_concentration=0.0,   # FLAT — plateau NOT richer
                terrain_gate_softness=0.08,
                terrain_ridge_height=0.15,
                terrain_gates_movement=False,      # gate OPEN so access is unrestricted
                enable_navigation=False,
                enable_food_sense=True,
                food_sense_decay=0.5,
                mutation_rate=0.0,
                horizon=horizon,
                max_population=500,
            )
            cfg = D.replace(cfg, founder=_founder_with_climb(1.0))
            eco = Ecology(cfg, seed=seed)

            world = eco.world
            assert world.elevation is not None
            ridge_h = world.terrain_ridge_height
            _eps = 1e-9

            steps_basin: int = 0
            steps_plateau: int = 0

            # Patch world.resource_at to track cell visits (pos when returned from choose_action
            # is the cell after the move; we tally by the creature's position each step).
            # Simpler: patch consume() to track where eating happens.
            # Actually: just run and read phenotype.pos each step via monkey-patching step().
            # Easiest approach: count consume() calls per cell type.
            _orig_consume = world.consume
            intake_basin: float = 0.0
            intake_plateau: float = 0.0

            def _patched_consume(pos: int, amount: float) -> float:
                nonlocal intake_basin, intake_plateau
                consumed = _orig_consume(pos, amount)
                if float(world.elevation[pos]) >= (ridge_h - _eps):  # type: ignore[index]
                    intake_plateau += consumed
                else:
                    intake_basin += consumed
                return consumed

            world.consume = _patched_consume  # type: ignore[method-assign]

            while eco.t < cfg.horizon and not eco.exploded:
                eco.step()
                if not eco.has_alive():
                    break

            world.consume = _orig_consume  # type: ignore[method-assign]

            total = intake_basin + intake_plateau
            if total < 1e-12:
                continue  # extinct
            shares.append(intake_plateau / total)

        if not shares:
            return float("nan")
        return sum(shares) / len(shares)

    def test_flat_world_no_index_artifact(self) -> None:
        """On a flat-resource world (no plateau richness), food-sense must NOT
        preferentially drive creatures to the high-index plateau region.
        Plateau intake share must be < 0.45 (plateau is 33% of cells; < 45% = no artifact).
        """
        import math
        share = self._measure_plateau_occupancy_share()
        assert not math.isnan(share), "All seeds went extinct — check config."
        assert share < 0.45, (
            f"Food-sense is driving creatures to the plateau on a FLAT-resource world!\n"
            f"  Plateau intake share: {share*100:.1f}% (>= 45%)\n"
            f"  Plateau is 33% of cells; this level implies an index/position artifact.\n"
            f"  (L40: food-sense must navigate by FOOD, not by index/position.)"
        )


# ---------------------------------------------------------------------------
# (c) Smoke: enable_food_sense=True run completes
# ---------------------------------------------------------------------------

class TestFoodSenseSmoke:
    """With enable_food_sense=True, runs complete and the hash diverges from OFF."""

    def test_food_sense_on_run_completes(self) -> None:
        """enable_food_sense=True run completes without error."""
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=True,
            terrain_food_concentration=2.0,
            terrain_gate_softness=0.08,
            terrain_ridge_height=0.15,
            terrain_gates_movement=True,
            enable_navigation=False,
            enable_food_sense=True,
            food_sense_decay=0.5,
            horizon=50,
            max_population=500,
            mutation_rate=0.0,
        )
        cfg = D.replace(cfg, founder=_founder_with_climb(0.5))
        eco = Ecology(cfg, seed=0)
        result = eco.run()
        assert "events_hash" in result
        assert len(result["events_hash"]) == 64

    def test_food_sense_on_diverges_from_off(self) -> None:
        """enable_food_sense=True must produce a DIFFERENT events_hash than OFF
        (proof the food-sense branch is causally active)."""
        def run(fs: bool) -> str:
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_terrain=True,
                terrain_food_concentration=2.0,
                terrain_gate_softness=0.08,
                terrain_ridge_height=0.15,
                terrain_gates_movement=False,   # gate open so creatures move freely
                enable_navigation=False,
                enable_food_sense=fs,
                food_sense_decay=0.5,
                mutation_rate=0.0,
                horizon=80,
                max_population=500,
            )
            cfg = D.replace(cfg, founder=_founder_with_climb(1.0))
            return Ecology(cfg, seed=3).run()["events_hash"]

        h_off = run(False)
        h_on = run(True)
        assert h_off != h_on, (
            "Food-sense ON produces the SAME events_hash as OFF — "
            "the food-sense branch is not causally active!\n"
            f"  Both: {h_off}"
        )
