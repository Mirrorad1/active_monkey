"""tests/test_ecology_navigation.py — Navigation policy (Exp 236) tests.

Covers:
  (a) OFF byte-identity: enable_navigation=False is byte-identical to Exp 194 and Exp 235
      terrain-ON golden hashes across seeds.
  (b) Smoke test: with enable_navigation=True, creatures reach distant plateau cells they
      would not access under the local-greedy policy (plateau occupancy or intake materially > 0).
  (c) Navigation does not break the gate-open (terrain_gates_movement=False) path.

Design:
  - OFF byte-identity is the HARD ANTI-CHEAT guard; any failure is a gating bug.
  - Smoke test uses the same plateau consume-patching technique as exp235_manip_check.
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


def _nav_cfg(**over) -> EcologyConfig:
    """Terrain ON + navigation ON, sensible test defaults."""
    base = dict(
        enable_terrain=True,
        terrain_food_concentration=2.0,
        terrain_gate_softness=0.08,
        terrain_ridge_height=0.15,
        climb_cost_floor=0.0,
        climb_cost_slope=0.0,
        terrain_gates_movement=True,
        enable_navigation=True,
        horizon=150,
        max_population=500,
        mutation_rate=0.0,
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


# ---------------------------------------------------------------------------
# (a) OFF byte-identity
# ---------------------------------------------------------------------------

class TestOffByteIdentity:
    """HARD ANTI-CHEAT: enable_navigation=False must be byte-identical to Exp 194 and
    Exp 235 terrain-ON golden hashes.  Any failure is a REAL gating bug."""

    # Exp 194 golden hash (canonical regression anchor, all features OFF)
    EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"

    # Exp 235 terrain-ON golden hash (see tests/test_ecology_terrain.py)
    TERRAIN_ON_GOLDEN_HASH = "1620e35d35bf4937da9a7f5e821d4a952ceaa544536752cf7004515a50f0a58e"

    def test_nav_off_preserves_exp194_hash(self) -> None:
        """enable_navigation=False with all terrain OFF must reproduce the Exp 194 hash."""
        cfg = D.replace(
            SCENARIOS["balanced"],
            complexity_cost_scale=0.0,
            enable_temperature=False,
            enable_thermosense=False,
            enable_terrain=False,
            enable_navigation=False,  # explicit OFF
        )
        eco = Ecology(cfg, seed=0)
        result = eco.run()
        assert result["events_hash"] == self.EXP194_HASH, (
            f"Navigation OFF broke Exp 194 byte-identity!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {self.EXP194_HASH}\n"
            "This is a REAL BUG — the OFF path must be perfectly byte-identical."
        )

    def test_nav_off_preserves_terrain_on_golden_hash(self) -> None:
        """enable_navigation=False with terrain ON must reproduce the Exp 235 terrain-ON hash."""
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
            enable_navigation=False,  # explicit OFF — must not change terrain-ON hash
        )
        cfg = D.replace(cfg, founder=_founder_with_climb(0.05))
        eco = Ecology(cfg, seed=5)
        result = eco.run()
        assert result["events_hash"] == self.TERRAIN_ON_GOLDEN_HASH, (
            f"Navigation OFF broke Exp 235 terrain-ON byte-identity!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {self.TERRAIN_ON_GOLDEN_HASH}\n"
            "This is a REAL BUG — enable_navigation=False must not alter the terrain-ON hash."
        )

    def test_nav_off_identical_across_climb_values(self) -> None:
        """enable_navigation=False: hash is the SAME across climb_ability values (terrain OFF)."""
        def run(climb: float) -> str:
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_terrain=False,
                enable_navigation=False,
                mutation_rate=0.0,
                horizon=120,
            )
            cfg = D.replace(cfg, founder=_founder_with_climb(climb))
            return Ecology(cfg, seed=7).run()["events_hash"]

        climbs = [0.0, 0.05, 0.5, 1.0]
        hashes = [run(c) for c in climbs]
        assert len(set(hashes)) == 1, (
            "Navigation OFF broke terrain-OFF byte-identity across climb values!\n"
            + "\n".join(f"  climb={c}: {h}" for c, h in zip(climbs, hashes))
        )

    def test_nav_off_identical_to_default_no_nav_field(self) -> None:
        """Confirm that adding enable_navigation=False to terrain-OFF config is a no-op.

        Run the same config twice, with and without the explicit enable_navigation field;
        both must produce the same hash (since False is the default).
        """
        base = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=False,
            mutation_rate=0.0,
            horizon=80,
        )
        cfg_default = base
        cfg_explicit = D.replace(base, enable_navigation=False)
        h1 = Ecology(cfg_default, seed=13).run()["events_hash"]
        h2 = Ecology(cfg_explicit, seed=13).run()["events_hash"]
        assert h1 == h2, (
            f"enable_navigation=False (explicit) diverges from default!\n"
            f"  default: {h1}\n"
            f"  explicit False: {h2}"
        )


# ---------------------------------------------------------------------------
# (b) Navigation smoke test — creatures reach distant plateau cells
# ---------------------------------------------------------------------------

class TestNavigationSmoke:
    """With enable_navigation=True and a GATE-OPEN world, creatures must access the
    plateau materially more than ~1% (the local-greedy ceiling from Exp 235).

    PASS condition: nav-ON gate-open plateau intake share >= 15%.
    (The expressibility demonstration target is >=30%; the smoke test is deliberately
    more lenient to be fast and robust across seeds.)
    """

    def _measure_plateau_share(
        self,
        nav: bool,
        gates: bool = False,  # gate-open by default for expressibility
        climb: float = 1.0,
        horizon: int = 200,
        n_seeds: int = 3,
    ) -> float:
        """Return mean plateau_intake_share across seeds."""
        import math
        import numpy as np

        shares: list[float] = []
        for seed in range(n_seeds):
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_terrain=True,
                terrain_food_concentration=2.0,
                terrain_gate_softness=0.08,
                terrain_ridge_height=0.15,
                terrain_gates_movement=gates,
                enable_navigation=nav,
                mutation_rate=0.0,
                horizon=horizon,
                max_population=500,
            )
            cfg = D.replace(cfg, founder=_founder_with_climb(climb))
            eco = Ecology(cfg, seed=seed)
            world = eco.world
            assert world.elevation is not None
            ridge_h = world.terrain_ridge_height
            _eps = 1e-9

            intake_basin: float = 0.0
            intake_plateau: float = 0.0
            _orig = world.consume

            def _patched(pos: int, amount: float, _orig=_orig) -> float:
                nonlocal intake_basin, intake_plateau
                consumed = _orig(pos, amount)
                if float(world.elevation[pos]) >= (ridge_h - _eps):  # type: ignore[index]
                    intake_plateau += consumed
                else:
                    intake_basin += consumed
                return consumed

            world.consume = _patched  # type: ignore[method-assign]
            while eco.t < cfg.horizon and not eco.exploded:
                eco.step()
                if not eco.has_alive():
                    break
            world.consume = _orig  # type: ignore[method-assign]

            total = intake_basin + intake_plateau
            if total < 1e-12:
                continue  # extinct
            shares.append(intake_plateau / total)

        if not shares:
            return float("nan")
        return sum(shares) / len(shares)

    def test_nav_on_gate_open_plateau_share_rises(self) -> None:
        """Nav ON + gate open: plateau intake share must rise materially above ~1%."""
        share_nav_off = self._measure_plateau_share(nav=False, gates=False, climb=1.0)
        share_nav_on = self._measure_plateau_share(nav=True, gates=False, climb=1.0)

        # Nav OFF (local greedy) should be near 0–5% (the Exp 235 ceiling).
        # Nav ON should be materially higher.
        SMOKE_THRESHOLD = 0.15  # >=15% plateau intake with nav ON (gate open)
        assert share_nav_on >= SMOKE_THRESHOLD, (
            f"Navigation ON gate-open plateau share did NOT rise materially!\n"
            f"  nav=OFF gate-open: {share_nav_off:.4f}  ({share_nav_off*100:.1f}%)\n"
            f"  nav=ON  gate-open: {share_nav_on:.4f}  ({share_nav_on*100:.1f}%)\n"
            f"  Required: nav-ON >= {SMOKE_THRESHOLD:.0%}\n"
            "The navigation mechanic is too weak — iterate on the mechanic or distance penalty."
        )
        # Also confirm nav OFF is indeed low (verifies the Exp 235 finding)
        assert share_nav_off < 0.10, (
            f"Nav OFF gate-open share unexpectedly high: {share_nav_off:.4f} "
            f"— the local-greedy floor is no longer ~1%?"
        )

    def test_nav_on_run_completes(self) -> None:
        """enable_navigation=True run completes without error."""
        cfg = _nav_cfg(horizon=50)
        cfg = D.replace(cfg, founder=_founder_with_climb(1.0))
        eco = Ecology(cfg, seed=0)
        result = eco.run()
        assert "events_hash" in result
        assert len(result["events_hash"]) == 64

    def test_nav_on_diverges_from_nav_off(self) -> None:
        """enable_navigation=True must produce a DIFFERENT events_hash than OFF
        (proof the navigation path is causally active in the event stream)."""
        def run(nav: bool) -> str:
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_terrain=True,
                terrain_food_concentration=2.0,
                terrain_gate_softness=0.08,
                terrain_ridge_height=0.15,
                terrain_gates_movement=False,  # gate open so creatures actually move
                enable_navigation=nav,
                mutation_rate=0.0,
                horizon=80,
                max_population=500,
            )
            cfg = D.replace(cfg, founder=_founder_with_climb(1.0))
            return Ecology(cfg, seed=3).run()["events_hash"]

        h_off = run(False)
        h_on = run(True)
        assert h_off != h_on, (
            "Navigation ON produces the SAME events_hash as OFF — "
            "the navigation branch is not causally active!\n"
            f"  Both: {h_off}"
        )
