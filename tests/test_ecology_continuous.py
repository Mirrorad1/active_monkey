"""tests/test_ecology_continuous.py — Continuous-locomotion substrate (Exp 238) tests.

Covers:
  1. OFF byte-identity: events_hash is UNCHANGED across {enable_continuous_locomotion=False}
     x {locomotor_speed in {0.25, 1.0, 4.0}}, for both the Exp 194 hash and the
     terrain-ON golden hash (1620e35d…). Any deviation is a REAL gating bug.
  2. Determinism / bit-identity: two identical seeds of a continuous-ON run produce
     the same events_hash; pause/resume at step 10 is bit-identical to straight-through.
  3. Anti-cheat / disconnect_overrides byte-identity: with LOCOMOTION_CONTINUOUS_AXIS
     disconnect_overrides applied, events_hash is IDENTICAL across locomotor_speed values.
  4. Continuous mechanics smoke: the ON path completes without error and produces a
     64-char events_hash. A pinned golden hash detects ON-path drift.

Design:
  - Mirrors tests/test_ecology_terrain.py structure.
  - OFF byte-identity and disconnect_overrides tests are HARD ANTI-CHEAT guards.
    If they fail, investigate the gating — do NOT update hashes.
  - Pause/resume uses ecology.runtime.snapshot + restore, same as test_ecology_runtime.py.
"""
from __future__ import annotations

import dataclasses as D

import pytest

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS
from ecology.genotype import founder
from ecology.runtime import snapshot, restore
from ecology.evolvability.trait_axis import LOCOMOTION_CONTINUOUS_AXIS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _founder_with_speed(speed: float):
    return D.replace(founder(), locomotor_speed=speed)


def _cont_cfg(**over):
    """Minimal continuous-ON config for fast tests."""
    base = dict(
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        mutation_rate=0.0,
        horizon=50,
    )
    base.update(over)
    return D.replace(SCENARIOS["balanced"], **base)


# ---------------------------------------------------------------------------
# Test 1: OFF byte-identity across locomotor_speed values
# ---------------------------------------------------------------------------

class TestOffByteIdentity:
    """HARD ANTI-CHEAT: with enable_continuous_locomotion=False, events_hash must be
    IDENTICAL across ALL locomotor_speed values.  Any deviation is a REAL gating bug,
    not a hash to update."""

    SPEEDS = [0.25, 1.0, 4.0]

    def _run_off(self, speed: float, seed: int = 7) -> str:
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_continuous_locomotion=False,
            mutation_rate=0.0,
            horizon=80,
            founder=_founder_with_speed(speed),
        )
        return Ecology(cfg, seed=seed).run()["events_hash"]

    def test_off_byte_identical_across_speeds(self) -> None:
        """OFF: events_hash must be identical across locomotor_speed in {0.25, 1.0, 4.0}."""
        hashes = [self._run_off(s) for s in self.SPEEDS]
        assert len(set(hashes)) == 1, (
            "OFF byte-identity BROKEN! Different hashes across locomotor_speed values:\n"
            + "\n".join(f"  speed={s}: {h}" for s, h in zip(self.SPEEDS, hashes))
            + "\nThis is a REAL BUG in the gating — do NOT update hashes to fix this."
        )

    def test_off_byte_identical_second_seed(self) -> None:
        """OFF: byte-identity holds on a second seed."""
        hashes = [self._run_off(s, seed=3) for s in self.SPEEDS]
        assert len(set(hashes)) == 1, (
            "OFF byte-identity BROKEN on seed=3:\n"
            + "\n".join(f"  speed={s}: {h}" for s, h in zip(self.SPEEDS, hashes))
        )

    def test_off_rng_stream_not_perturbed_by_locomotor_speed(self) -> None:
        """OFF: changing locomotor_speed must NOT draw from rng (stream must be identical).

        This guards the LOCOMOTION_CONTINUOUS_TRAITS skip-guard in mutate().
        """
        h1 = self._run_off(0.25, seed=42)
        h4 = self._run_off(4.0, seed=42)
        assert h1 == h4, (
            f"OFF rng stream DIVERGES between locomotor_speed=0.25 and locomotor_speed=4.0!\n"
            f"  speed=0.25: {h1}\n"
            f"  speed=4.0:  {h4}\n"
            "The mutate() skip-guard for LOCOMOTION_CONTINUOUS_TRAITS is broken."
        )

    def test_exp194_hash_unchanged_by_continuous_locomotion_feature(self) -> None:
        """Adding the continuous-locomotion feature must NOT change the Exp 194 baseline hash.

        EXP194_HASH is the canonical regression guard for the OFF path.
        """
        EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"
        cfg = D.replace(
            SCENARIOS["balanced"],
            complexity_cost_scale=0.0,
            enable_temperature=False,
            enable_thermosense=False,
            enable_terrain=False,
            enable_continuous_locomotion=False,
        )
        result = Ecology(cfg, seed=0).run()
        assert result["events_hash"] == EXP194_HASH, (
            f"Continuous-locomotion feature BROKE Exp 194 byte-identity!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {EXP194_HASH}\n"
            "This is a REAL BUG — the OFF path must be perfectly byte-identical."
        )

    def test_exp194_hash_unchanged_with_varying_speed(self) -> None:
        """Exp 194 hash is stable across locomotor_speed values when continuous is OFF."""
        EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"
        for speed in self.SPEEDS:
            cfg = D.replace(
                SCENARIOS["balanced"],
                complexity_cost_scale=0.0,
                enable_temperature=False,
                enable_thermosense=False,
                enable_terrain=False,
                enable_continuous_locomotion=False,
                founder=_founder_with_speed(speed),
            )
            result = Ecology(cfg, seed=0).run()
            assert result["events_hash"] == EXP194_HASH, (
                f"Exp 194 hash changed with locomotor_speed={speed}!\n"
                f"  Got:      {result['events_hash']}\n"
                f"  Expected: {EXP194_HASH}\n"
                "The continuous locomotion OFF path is leaking — do NOT update the hash."
            )

    def test_terrain_on_hash_unchanged_by_continuous_locomotion_feature(self) -> None:
        """The terrain ON golden hash is NOT changed by adding continuous locomotion.

        Verifies the two features are independently gated.
        """
        TERRAIN_ON_HASH = "1620e35d35bf4937da9a7f5e821d4a952ceaa544536752cf7004515a50f0a58e"
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_terrain=True,
            terrain_food_concentration=2.0,
            terrain_gate_softness=0.08,
            terrain_ridge_height=0.15,
            climb_cost_floor=0.0,
            climb_cost_slope=0.0,
            terrain_gates_movement=True,
            mutation_rate=0.0,
            horizon=100,
            max_population=500,
            enable_continuous_locomotion=False,
            founder=D.replace(founder(), climb_ability=0.05),
        )
        result = Ecology(cfg, seed=5).run()
        assert result["events_hash"] == TERRAIN_ON_HASH, (
            f"Terrain ON hash changed — continuous locomotion feature may have broken gating!\n"
            f"  Got:      {result['events_hash']}\n"
            f"  Expected: {TERRAIN_ON_HASH}\n"
            "Investigate before updating this hash."
        )


# ---------------------------------------------------------------------------
# Test 2: Determinism and pause/resume bit-identity
# ---------------------------------------------------------------------------

class TestDeterminismAndPauseResume:
    """Continuous-ON runs must be fully deterministic (same seed → same hash) and
    pause/resume via snapshot/restore must be bit-identical to straight-through."""

    def _straight_through(self, seed: int, horizon: int = 50) -> str:
        cfg = D.replace(_cont_cfg(horizon=horizon), founder=_founder_with_speed(1.5))
        eco = Ecology(cfg, seed=seed)
        eco.run()
        return eco.events_hash()

    def test_continuous_on_deterministic(self) -> None:
        """Two identical-seed continuous-ON runs produce identical events_hash."""
        h1 = self._straight_through(seed=3)
        h2 = self._straight_through(seed=3)
        assert h1 == h2, (
            f"Continuous-ON runs are NOT deterministic!\n"
            f"  run1: {h1}\n"
            f"  run2: {h2}"
        )

    def test_continuous_on_seed_diverges(self) -> None:
        """Different seeds produce DIFFERENT events_hash (sanity check that hash is meaningful)."""
        h3 = self._straight_through(seed=3)
        h7 = self._straight_through(seed=7)
        assert h3 != h7, (
            "Different seeds produced IDENTICAL hashes — the continuous path may be degenerate."
        )

    def test_pause_resume_bit_identical(self) -> None:
        """Snapshot at step 10, restore, continue to horizon → identical to straight-through."""
        HORIZON = 50
        cfg = D.replace(_cont_cfg(horizon=HORIZON), founder=_founder_with_speed(1.5))

        # Straight-through reference
        eco_ref = Ecology(cfg, seed=3)
        eco_ref.run()
        straight_hash = eco_ref.events_hash()

        # Pause at step 10, resume
        eco_pause = Ecology(cfg, seed=3)
        for _ in range(10):
            eco_pause.step()
        snap = snapshot(eco_pause)
        resumed = restore(snap)
        while resumed.t < HORIZON and resumed._alive_list:
            resumed.step()

        assert resumed.events_hash() == straight_hash, (
            f"Pause/resume is NOT bit-identical for continuous-ON!\n"
            f"  Straight-through: {straight_hash}\n"
            f"  After resume:     {resumed.events_hash()}"
        )

    def test_snapshot_does_not_perturb_source(self) -> None:
        """Taking a snapshot must not alter the source run's future."""
        HORIZON = 50
        cfg = D.replace(_cont_cfg(horizon=HORIZON), founder=_founder_with_speed(1.5))
        straight_hash = self._straight_through(seed=3, horizon=HORIZON)

        eco = Ecology(cfg, seed=3)
        for _ in range(10):
            eco.step()
        _ = snapshot(eco)  # take snapshot — must not perturb eco
        while eco.t < HORIZON and eco._alive_list:
            eco.step()

        assert eco.events_hash() == straight_hash, (
            f"Snapshotting perturbed the source run!\n"
            f"  Expected: {straight_hash}\n"
            f"  Got:      {eco.events_hash()}"
        )

    def test_continuous_on_pos_cont_set_on_founders(self) -> None:
        """When continuous-ON, every creature should have pos_cont initialized."""
        cfg = D.replace(_cont_cfg(horizon=1), founder=_founder_with_speed(1.0))
        eco = Ecology(cfg, seed=0)
        eco.step()
        for c in eco._creatures:
            assert c.phenotype.pos_cont is not None, (
                f"Creature {c.genotype.id} has pos_cont=None when continuous-ON"
            )


# ---------------------------------------------------------------------------
# Test 3: Anti-cheat / disconnect_overrides byte-identity
# ---------------------------------------------------------------------------

class TestDisconnectOverridesByteIdentity:
    """With LOCOMOTION_CONTINUOUS_AXIS.disconnect_overrides applied, events_hash must be
    byte-identical across locomotor_speed values.  This is the Gate-G Guard 1 check:
    if it fails, a channel feeding locomotor_speed is missing from disconnect_overrides."""

    SPEEDS = [0.25, 1.0, 4.0]

    def _run_disconnected(self, speed: float, seed: int = 7) -> str:
        overrides = LOCOMOTION_CONTINUOUS_AXIS.disconnect_overrides
        cfg = D.replace(
            SCENARIOS["balanced"],
            horizon=80,
            mutation_rate=0.0,
            **overrides,
            founder=_founder_with_speed(speed),
        )
        return Ecology(cfg, seed=seed).run()["events_hash"]

    def test_disconnect_overrides_byte_identical_across_speeds(self) -> None:
        """Disconnect ALL locomotor_speed channels: hashes identical across speed values."""
        hashes = [self._run_disconnected(s) for s in self.SPEEDS]
        assert len(set(hashes)) == 1, (
            "Disconnect overrides byte-identity FAILED — a channel is missing from "
            "LOCOMOTION_CONTINUOUS_AXIS.disconnect_overrides!\n"
            + "\n".join(f"  speed={s}: {h}" for s, h in zip(self.SPEEDS, hashes))
            + "\nAdd the leaking channel to disconnect_overrides."
        )

    def test_disconnect_overrides_byte_identical_second_seed(self) -> None:
        """Disconnect byte-identity holds on a second seed."""
        hashes = [self._run_disconnected(s, seed=2) for s in self.SPEEDS]
        assert len(set(hashes)) == 1, (
            "Disconnect overrides byte-identity FAILED on seed=2:\n"
            + "\n".join(f"  speed={s}: {h}" for s, h in zip(self.SPEEDS, hashes))
        )


# ---------------------------------------------------------------------------
# Test 4: Continuous mechanics smoke + golden hash pin
# ---------------------------------------------------------------------------

# Golden hash for continuous-ON with:
#   locomotor_speed=1.5 founder, layout=bump, dt=1.0, zero cost, zero mutation,
#   horizon=50, seed=3.
# This hash WILL change if the continuous-locomotion ON-path mechanics change.
# Repin with a comment documenting the change when that happens.
# DO NOT update this hash to fix a test without understanding the root cause.
_CONTINUOUS_ON_GOLDEN_HASH = "2dd73836cf72b8d85a47157bba109c8abb7e0e3356aae3adb487a3a46ce4b31b"
_CONTINUOUS_ON_GOLDEN_SEED = 3


class TestContinuousOnMechanicsSmoke:
    """Quick smoke tests that the continuous-ON path works, plus a golden hash pin."""

    def _run_on(self, seed: int = _CONTINUOUS_ON_GOLDEN_SEED) -> dict:
        cfg = D.replace(_cont_cfg(horizon=50), founder=_founder_with_speed(1.5))
        return Ecology(cfg, seed=seed).run()

    def test_continuous_on_run_completes(self) -> None:
        """enable_continuous_locomotion=True run completes and returns events_hash."""
        result = self._run_on()
        assert "events_hash" in result
        assert len(result["events_hash"]) == 64

    def test_continuous_on_golden_hash_stable(self) -> None:
        """Pin the ON-path events_hash for the canonical bump layout.

        locomotor_speed=1.5, layout=bump, dt=1.0, zero cost, zero mutation,
        horizon=50, seed=3. Repin when ON-path mechanics intentionally change.
        """
        result = self._run_on()
        h = result["events_hash"]
        assert h == _CONTINUOUS_ON_GOLDEN_HASH, (
            f"Continuous ON golden hash changed — ON-path drift detected!\n"
            f"  Got:      {h}\n"
            f"  Expected: {_CONTINUOUS_ON_GOLDEN_HASH}\n"
            "Investigate before repinning. This is NOT an OFF-path issue."
        )

    def test_continuous_on_causally_live_speed(self) -> None:
        """ON-path: different locomotor_speed founders produce different events_hash."""
        def run(speed: float) -> str:
            cfg = D.replace(_cont_cfg(horizon=30), founder=_founder_with_speed(speed))
            return Ecology(cfg, seed=1).run()["events_hash"]

        h_slow = run(0.25)
        h_fast = run(4.0)
        assert h_slow != h_fast, (
            "Continuous locomotion is NOT causally active: events_hash identical for "
            "locomotor_speed=0.25 and locomotor_speed=4.0 when enable_continuous_locomotion=True."
        )

    def test_continuous_on_cost_seam_causally_live(self) -> None:
        """Speed cost seam is active: nonzero cost diverges hash from zero-cost run."""
        def run(slope: float) -> str:
            cfg = D.replace(
                _cont_cfg(horizon=30, speed_cost_slope=slope),
                founder=_founder_with_speed(2.0),
            )
            return Ecology(cfg, seed=2).run()["events_hash"]

        h_free = run(0.0)
        h_paid = run(0.5)
        assert h_free != h_paid, (
            "Speed cost seam is NOT causally active: changing speed_cost_slope 0→0.5 "
            "did NOT change the trajectory."
        )
