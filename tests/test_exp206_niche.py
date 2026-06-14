"""Guard tests for the Exp 206 rotating-class niche mechanic (enable_niche).

Covers BOTH branches of the gated feature (L15/L16):
  - OFF path is byte-identical (reproduces the committed EXP194_HASH; niche_* params do not
    perturb the events_hash when enable_niche=False; no niche state allocated);
  - ON path is NON-TRIVIAL (class state allocated, occupancy accumulates, the class ROTATES);
  - the CORE mechanic works (precision changes routing ⇒ different realized intake) AND is
    h-BLIND given the percept (at niche_confusion=0 the realized intake is h-independent —
    the binding ANTI-CHEAT: h acts ONLY through the perceptual noise, never as a direct reward);
  - determinism under shuffle (same seed ⇒ identical events_hash with shuffle_creature_order).
"""
from __future__ import annotations

import dataclasses as D

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import founder as _founder
from ecology.scenarios import SCENARIOS

# The ONLY committed literal determinism anchor (balanced seed0; see
# experiments/outputs/exp194_n5_homeostatic_population/verdict.json and tests/test_ecology_*).
EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"


def _niche_cfg(h: float = 0.10, **kw) -> EcologyConfig:
    """A small compete-regime niche config (cost ON, shuffle ON) for fast tests."""
    f = D.replace(_founder(), thermosense_intensity=h, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    base = dict(
        rows=8, cols=8, horizon=300, initial_population=24, founder=f,
        mutation_rate=0.0, capacity=10.0, regen_rate=0.20, initial_resource=0.7,
        max_population=20000, min_survival_energy=4.0, name="exp206_test",
        enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05,
        thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.3, food_optimal_period=1500.0,
        food_concentration=14.0, food_band_width=0.08, enable_food_coupling=True,
        thermosense_forage_mode=True, shuffle_creature_order=True, freeze_thermosense=True,
        enable_niche=True, niche_classes=2, niche_rotation=0.013, niche_confusion=0.6,
        niche_crowding=1.0, niche_weight=4.0,
    )
    base.update(kw)
    return EcologyConfig(**base)


def test_niche_off_byte_identical_and_no_state():
    """enable_niche=False ⇒ reproduces EXP194_HASH, no niche state allocated, and niche_*
    params do not move the events_hash (the OFF path is byte-identical)."""
    eco = Ecology(SCENARIOS["balanced"], seed=0)
    assert eco.world.class_phase is None and eco.world.class_signal is None
    assert eco.run()["events_hash"] == EXP194_HASH

    # OFF with nonsense niche params must be identical to OFF without them.
    a = _niche_cfg(enable_niche=False)
    b = D.replace(a, niche_rotation=0.37, niche_confusion=0.9, niche_crowding=5.0,
                  niche_weight=9.0, niche_barcode_shuffle=True)
    assert Ecology(a, seed=7).run()["events_hash"] == Ecology(b, seed=7).run()["events_hash"]


def test_niche_on_nontrivial():
    """enable_niche=True ⇒ class state allocated, occupancy accumulates, and the class ROTATES
    (class_signal changes over time so a static map cannot memorise it)."""
    eco = Ecology(_niche_cfg(), seed=7)
    assert eco.world.class_phase is not None
    assert eco.world.class_signal.shape == (eco.cfg.rows * eco.cfg.cols,)
    sig_t0 = eco.world.class_signal.copy()
    for _ in range(40):
        eco.step()
    # the class signal has rotated (niche_rotation>0): not equal to t=0
    assert not np.allclose(sig_t0, eco.world.class_signal), "class signal did not rotate"
    # occupancy was tallied
    assert int(eco.world.class_occ_prev.sum()) > 0
    # at least one creature occupied each class over the run (non-degenerate niche)
    occ = np.zeros(2, dtype=np.int64)
    for c in eco._creatures:
        if c.policy is not None and c.policy.niche_occ is not None:
            occ += np.array(c.policy.niche_occ, dtype=np.int64)
    assert occ.min() > 0, f"a class was never occupied: {occ}"


def _mean_intake(h: float, confusion: float, seed: int, costoff: bool = False) -> float:
    cfg = _niche_cfg(h=h, niche_confusion=confusion,
                     enable_thermosense=(not costoff))
    eco = Ecology(cfg, seed=seed)
    eco.run()
    rates = [c.phenotype.resource_eaten / max(1, c.phenotype.age) for c in eco._creatures]
    return float(np.mean(rates)) if rates else float("nan")


def test_no_direct_h_reward_confusion_zero():
    """ANTI-CHEAT (binding): with niche_confusion=0 the routing percept is PERFECT for EVERY h,
    so the realized intake (kept) is h-INDEPENDENT at the FULL verdict params (real crowding,
    K=2, rotation>0) — certifying h acts ONLY through the perceptual noise, never as a reward.
    Compare two ACTIVE organs (both above the 0.05 activation threshold) cost-OFF, so the only
    possible h-channel is the percept (an inactive organ falls back to the base policy; cost-ON
    would differ via upkeep)."""
    seeds = [1, 2, 3]
    lo = float(np.mean([_mean_intake(0.10, 0.0, s, costoff=True) for s in seeds]))
    hi = float(np.mean([_mean_intake(0.90, 0.0, s, costoff=True) for s in seeds]))
    assert abs(hi - lo) < 1e-9, f"confusion=0 should make intake h-independent: lo={lo} hi={hi}"

    # And the events_hash is identical across h at confusion=0 cost-OFF (h does literally nothing).
    ha = Ecology(_niche_cfg(h=0.10, niche_confusion=0.0, enable_thermosense=False), seed=1).run()["events_hash"]
    hb = Ecology(_niche_cfg(h=0.90, niche_confusion=0.0, enable_thermosense=False), seed=1).run()["events_hash"]
    assert ha == hb, "at confusion=0 cost-OFF, h must not change the trajectory"


def test_crowding_mechanic_nontrivial_but_h_blind_percept():
    """The mechanic is NON-TRIVIAL (with confusion>0, precision CHANGES routing ⇒ different
    realized intake) yet h-BLIND given the percept (confusion=0 ⇒ identical). The contrast
    certifies h matters ONLY by sharpening the percept, not by a direct reward. Both organs are
    ACTIVE (>0.05 threshold) so the difference is percept sharpness, not activation."""
    seeds = [1, 2, 3]
    # confusion>0: a precise vs crude monomorphic pop realize DIFFERENT intake (routing differs)
    lo_c = float(np.mean([_mean_intake(0.10, 0.6, s, costoff=True) for s in seeds]))
    hi_c = float(np.mean([_mean_intake(0.90, 0.6, s, costoff=True) for s in seeds]))
    assert abs(hi_c - lo_c) > 1e-6, "with confusion>0 precision should change realized intake"
    # confusion=0: identical (h-blind percept)
    lo_0 = float(np.mean([_mean_intake(0.10, 0.0, s, costoff=True) for s in seeds]))
    hi_0 = float(np.mean([_mean_intake(0.90, 0.0, s, costoff=True) for s in seeds]))
    assert abs(hi_0 - lo_0) < 1e-9


def test_niche_on_determinism_under_shuffle():
    """Same seed ⇒ identical events_hash with shuffle_creature_order=True (the crowding discount
    reads the FROZEN class_occ_prev, never class_occ_cur mid-loop, so order cannot leak)."""
    cfg = _niche_cfg(shuffle_creature_order=True)
    h1 = Ecology(cfg, seed=11).run()["events_hash"]
    h2 = Ecology(cfg, seed=11).run()["events_hash"]
    assert h1 == h2
