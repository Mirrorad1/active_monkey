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


# ---------------------------------------------------------------------------
# Test 5: Exp 240 continuous_logistic_regen knob
# ---------------------------------------------------------------------------

class TestLogisticRegen:
    """Tests for the Exp 240 continuous_logistic_regen density-dependent regulation.

    OFF (logistic_regen=False) must be BYTE-IDENTICAL to Exp 238-239 (the OFF hash must
    equal the existing continuous ON golden hash).  ON must be causally active (changes
    the trajectory vs OFF), and must be gated: when enable_continuous_locomotion=False
    the knob has NO effect (byte-identical to baseline OFF runs).
    """

    def test_logistic_regen_off_byte_identical_to_baseline(self) -> None:
        """continuous_logistic_regen=False must produce SAME hash as the plain ON run.

        Guarantees the OFF path is byte-identical to Exp 238-239 — no leakage.
        """
        def run(logistic: bool) -> str:
            cfg = D.replace(
                _cont_cfg(horizon=50, continuous_logistic_regen=logistic),
                founder=_founder_with_speed(1.5),
            )
            return Ecology(cfg, seed=_CONTINUOUS_ON_GOLDEN_SEED).run()["events_hash"]

        h_off = run(False)
        assert h_off == _CONTINUOUS_ON_GOLDEN_HASH, (
            f"continuous_logistic_regen=False broke OFF byte-identity!\n"
            f"  Got:      {h_off}\n"
            f"  Expected: {_CONTINUOUS_ON_GOLDEN_HASH}\n"
            "The logistic OFF path must be byte-identical to Exp 238-239. Do NOT update the"
            " existing hash — fix the gating instead."
        )

    def test_logistic_regen_on_depletes_cells_differently(self) -> None:
        """continuous_logistic_regen=True must produce a DIFFERENT resource state than OFF.

        Verifies the logistic branch is causally active (actually changes the resource field).
        At high population with heavy depletion: logistic regen cannot recover cells that hit 0
        (regen=0 at v=0), while linear regen adds regen_rate regardless. Run with more founders
        and higher max_pop so bump cells get depleted and the two branches diverge.
        The events_hash may be identical if creature starvation order is unchanged; resource
        state is the load-bearing proof of causal activity.
        """
        import dataclasses as dc2

        def run_get_resource_total(logistic: bool) -> float:
            # Use higher initial pop so bump cells actually deplete to 0 in the short horizon.
            cfg = D.replace(
                _cont_cfg(horizon=30, continuous_logistic_regen=logistic,
                          continuous_regen_rate=0.2, max_population=2000),
                founder=_founder_with_speed(1.5),
                initial_population=100,
            )
            eco = Ecology(cfg, seed=_CONTINUOUS_ON_GOLDEN_SEED)
            eco.run()
            assert eco.cont_world is not None
            return sum(
                eco.cont_world._resource[ri][ci]
                for ri in range(24) for ci in range(24)
            )

        r_off = run_get_resource_total(False)
        r_on = run_get_resource_total(True)
        # Logistic regen leaves depleted cells at 0 (logistic formula = 0 when v=0),
        # while linear regen bumps them back up. Logistic total resource should be LOWER.
        assert r_on < r_off, (
            f"continuous_logistic_regen=True did NOT produce lower total resource than OFF!\n"
            f"  linear (OFF):   total_resource={r_off:.2f}\n"
            f"  logistic (ON):  total_resource={r_on:.2f}\n"
            "At high population, logistic regen leaves depleted (v=0) cells at 0 while linear"
            " regen recovers them. If r_on >= r_off, the logistic branch is NOT active."
        )

    def test_logistic_regen_gated_by_enable_continuous_locomotion(self) -> None:
        """When enable_continuous_locomotion=False, logistic_regen setting has NO effect.

        The OFF-continuous path must never enter the logistic branch regardless of the
        continuous_logistic_regen flag (byte-identical to baseline OFF runs).
        """
        def run(logistic: bool) -> str:
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_continuous_locomotion=False,
                continuous_logistic_regen=logistic,
                mutation_rate=0.0,
                horizon=50,
                founder=_founder_with_speed(1.0),
            )
            return Ecology(cfg, seed=5).run()["events_hash"]

        h_false = run(False)
        h_true = run(True)
        assert h_false == h_true, (
            "continuous_logistic_regen=True changed the hash when enable_continuous_locomotion=False!\n"
            f"  logistic=False: {h_false}\n"
            f"  logistic=True:  {h_true}\n"
            "The logistic_regen flag must have zero effect when continuous locomotion is OFF."
        )

    def test_logistic_regen_deterministic(self) -> None:
        """Two identical-seed continuous-ON runs with logistic_regen=True are deterministic."""
        def run() -> str:
            cfg = D.replace(
                _cont_cfg(horizon=40, continuous_logistic_regen=True),
                founder=_founder_with_speed(1.0),
            )
            return Ecology(cfg, seed=7).run()["events_hash"]

        assert run() == run(), (
            "continuous_logistic_regen=True runs are NOT deterministic — different hashes!"
        )


# ---------------------------------------------------------------------------
# Test 6: Exp 242 enable_continuous_depletion_intake knob
# ---------------------------------------------------------------------------

class TestContinuousDepletionIntakeKnob:
    """Tests for the Exp 242 enable_continuous_depletion_intake knob.

    The Exp 238-241 substrate has a silent regulation bug: continuous intake is the line
    integral of the STRUCTURAL density field rho() (never depletes), so the depletable
    _resource grid has NO effect on intake.  The Exp 242 flag makes intake read the local
    availability (resource_cell / capacity), closing the density-dependent feedback.

    Guards:
      - OFF path is byte-identical (determinism) and the flag is gated by locomotion.
      - line_integral_intake ON a FULL field == OFF integral (reduces to OFF at zero depletion).
      - line_integral_intake ON a DEPLETED field is STRICTLY LESS than OFF (the feedback bites).
      - An ON run over a populated horizon produces a DIFFERENT events_hash than OFF (the
        mechanism actually changes intake → energy → reproduction/death events).
    """

    def _run(self, enable: bool, seed: int = 7) -> str:
        cfg = D.replace(
            SCENARIOS["balanced"],
            enable_continuous_locomotion=True,
            continuous_layout="bump",
            continuous_dt=1.0,
            speed_cost_floor=0.0,
            speed_cost_slope=0.2,
            mutation_rate=0.0,
            horizon=60,
            continuous_logistic_regen=False,
            enable_continuous_depletion_intake=enable,
            continuous_regen_rate=0.2,
            continuous_capacity=2.0,
            initial_population=40,
            max_population=2000,
        )
        return Ecology(cfg, seed=seed).run()["events_hash"]

    def test_depletion_intake_off_determinism(self) -> None:
        """enable_continuous_depletion_intake=False is deterministic (same hash twice)."""
        assert self._run(enable=False, seed=7) == self._run(enable=False, seed=7), (
            "enable_continuous_depletion_intake=False is not deterministic!"
        )

    def test_depletion_intake_on_determinism(self) -> None:
        """enable_continuous_depletion_intake=True is deterministic (same hash twice)."""
        assert self._run(enable=True, seed=11) == self._run(enable=True, seed=11), (
            "enable_continuous_depletion_intake=True runs are NOT deterministic!"
        )

    def test_depletion_intake_on_changes_events_hash(self) -> None:
        """ON must change the events_hash vs OFF — the mechanism actually bites.

        Depletion-aware intake lowers per-capita intake as the field is stripped, which
        changes energy → reproduction/death timings → the event stream.  If the hash were
        unchanged the flag would be a silent no-op (the very bug Exp 242 fixes).
        """
        h_off = self._run(enable=False, seed=7)
        h_on = self._run(enable=True, seed=7)
        assert h_off != h_on, (
            "enable_continuous_depletion_intake=True did NOT change the events_hash!\n"
            f"  OFF={h_off[:16]}  ON={h_on[:16]}\n"
            "The depletion-intake mechanism is a silent no-op — intake still ignores _resource."
        )

    def test_full_field_intake_equals_off(self) -> None:
        """On a FULL field, the depletion-aware integral EQUALS the structural OFF integral.

        availability == 1.0 everywhere ⇒ rho * 1.0 == rho ⇒ the ON path reduces to OFF
        physics at zero depletion.  This is the byte-identity-at-init property.
        """
        from ecology.continuous_world import ContinuousWorld

        w_off = ContinuousWorld(layout="bump", regen_rate=0.2, capacity=2.0,
                                enable_depletion_intake=False)
        w_on = ContinuousWorld(layout="bump", regen_rate=0.2, capacity=2.0,
                               enable_depletion_intake=True)
        # Both start full (capacity everywhere). Same segment.
        seg = (1.0, 1.0, 3.5, 3.5)
        i_off = w_off.line_integral_intake(*seg, 100.0)
        i_on = w_on.line_integral_intake(*seg, 100.0)
        assert abs(i_off - i_on) < 1e-12, (
            f"Full-field intake differs between OFF and ON: off={i_off}, on={i_on}\n"
            "ON must reduce to OFF when availability == 1.0 everywhere."
        )

    def test_depleted_field_intake_strictly_less(self) -> None:
        """On a DEPLETED field, the depletion-aware integral is STRICTLY LESS than OFF.

        This is the density-dependent feedback: a region the population has stripped
        yields proportionally less intake (rho * availability < rho).  OFF ignores
        depletion and integrates the full structural rho regardless.
        """
        from ecology.continuous_world import ContinuousWorld, _GRID_CELLS

        w_off = ContinuousWorld(layout="bump", regen_rate=0.2, capacity=2.0,
                                enable_depletion_intake=False)
        w_on = ContinuousWorld(layout="bump", regen_rate=0.2, capacity=2.0,
                               enable_depletion_intake=True)
        # Deplete the ON field to 25% availability everywhere.
        for ri in range(_GRID_CELLS):
            for ci in range(_GRID_CELLS):
                w_on._resource[ri][ci] = 0.5  # 0.5 / 2.0 = 25%
        seg = (1.0, 1.0, 5.0, 5.0)  # passes through the central bump
        i_off = w_off.line_integral_intake(*seg, 100.0)
        i_on = w_on.line_integral_intake(*seg, 100.0)
        assert i_on < i_off, (
            f"Depleted-field intake NOT less under ON: off={i_off:.4f}, on={i_on:.4f}\n"
            "The depletion-intake feedback is not biting."
        )
        # At uniform 25% availability the ON integral should be ~25% of OFF.
        assert abs(i_on - 0.25 * i_off) < 1e-9, (
            f"ON integral ({i_on:.6f}) is not 25% of OFF ({i_off:.6f}) at 25% availability."
        )

    def test_depletion_intake_gated_by_enable_locomotion(self) -> None:
        """When enable_continuous_locomotion=False, the depletion-intake knob has NO effect.

        cont_world is None when locomotion OFF, so enable_continuous_depletion_intake=True
        must be byte-identical to the plain OFF run.
        """
        def _run_off_locomotion(enable_depletion: bool) -> str:
            cfg = D.replace(
                SCENARIOS["balanced"],
                enable_continuous_locomotion=False,
                enable_continuous_depletion_intake=enable_depletion,
                mutation_rate=0.0,
                horizon=30,
            )
            return Ecology(cfg, seed=5).run()["events_hash"]

        h_false = _run_off_locomotion(False)
        h_true = _run_off_locomotion(True)
        assert h_false == h_true, (
            "enable_continuous_depletion_intake=True changed hash when locomotion is OFF!\n"
            f"  OFF depletion={h_false[:16]}  ON depletion={h_true[:16]}\n"
            "The knob must have zero effect when enable_continuous_locomotion=False."
        )
