"""Guard tests for the Exp 204 residue / false-positive discrimination mechanic.

Covers BOTH branches of the gated `enable_residue` feature (L15/L16):
  - OFF path is byte-identical (the feature flag + residue params do not perturb the
    events_hash when enable_residue=False, and no residue field is allocated);
  - ON path is NON-TRIVIAL (residue accumulates from eating and decays; the eat
    decision populates the discrimination counters);
  - the CORE mechanic works (higher thermosense precision h ⇒ fewer false positives);
  - the ANTI-CHEAT property holds (with residue_confusion=0 the percept is perfect,
    so h has NO effect on the eat outcome — h is never a direct reward, only a
    precision on the percept).
"""
from __future__ import annotations

import dataclasses as D

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import founder as _founder


def _compete_cfg(**kw) -> EcologyConfig:
    """A small depleting-band-ish compete config (cost ON, shuffle ON) for fast tests."""
    f = D.replace(_founder(), thermosense_intensity=0.10, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    base = dict(
        rows=8, cols=8, horizon=300, initial_population=24, founder=f,
        mutation_rate=0.05, capacity=10.0, regen_rate=0.20, initial_resource=0.7,
        max_population=20000, min_survival_energy=4.0, name="exp204_test",
        enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05,
        thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.3, food_optimal_period=60.0,
        food_concentration=8.0, food_band_width=0.12, enable_food_coupling=True,
        thermosense_forage_mode=True, enable_band_staleness=True, band_responsiveness=1.0,
        shuffle_creature_order=True, freeze_thermosense=True,
    )
    base.update(kw)
    return EcologyConfig(**base)


def test_residue_off_byte_identical_and_no_field():
    """enable_residue=False ⇒ residue field is None AND the residue params do not move
    the events_hash (the OFF path is byte-identical regardless of residue_* values)."""
    cfg_a = _compete_cfg(enable_residue=False)
    eco_a = Ecology(cfg_a, seed=7)
    assert eco_a.world.residue is None
    hash_a = eco_a.run()["events_hash"]

    # Same config, OFF, but with nonsense residue params — must not matter when OFF.
    cfg_b = _compete_cfg(enable_residue=False, residue_yield=5.0, residue_decay=0.9,
                         residue_loss=3.0, residue_confusion=1.0,
                         residue_eat_threshold=0.7, residue_fp_threshold=0.3)
    hash_b = Ecology(cfg_b, seed=7).run()["events_hash"]
    assert hash_a == hash_b, "residue params perturbed the OFF-path events_hash"


def test_residue_on_is_nontrivial():
    """enable_residue=True ⇒ a residue field is allocated, accumulates from eating, and
    the discrimination counters are populated (the feature actually does something)."""
    cfg = _compete_cfg(enable_residue=True, residue_yield=1.0, residue_decay=0.05,
                       residue_loss=0.5, residue_confusion=0.6)
    eco = Ecology(cfg, seed=7)
    assert eco.world.residue is not None
    assert eco.world.residue.shape == (cfg.rows * cfg.cols,)
    eco.run()
    # Residue was produced somewhere over the run (eating leaves a trace).
    assert float(np.max(eco.world.residue)) > 0.0
    # Some creature made eat decisions of each polarity at least in aggregate.
    tp = sum(c.phenotype.tp_count for c in eco._creatures)
    fp = sum(c.phenotype.fp_count for c in eco._creatures)
    decided = sum(c.phenotype.tp_count + c.phenotype.fp_count
                  + c.phenotype.fn_count + c.phenotype.tn_count for c in eco._creatures)
    assert decided > 0, "no eat decisions recorded under the residue mechanic"
    assert tp + fp > 0, "no creature ever chose to eat"


def _fp_rate_at_h(h: float, seed: int) -> float:
    """Monomorphic FP-rate = FP / (FP+TP) over all eat decisions at a clamped intensity h."""
    f = D.replace(_founder(), thermosense_intensity=h, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    cfg = _compete_cfg(enable_residue=True, residue_confusion=0.8, residue_loss=0.5,
                       founder=f)
    eco = Ecology(cfg, seed=seed)
    eco.run()
    fp = sum(c.phenotype.fp_count for c in eco._creatures)
    tp = sum(c.phenotype.tp_count for c in eco._creatures)
    return fp / max(1, fp + tp)


def test_higher_h_fewer_false_positives():
    """THE CORE MECHANIC: a high-precision sensor makes fewer false positives (eats less
    residue) than a primitive one, averaged over seeds.  Anti-cheat-clean: this is a
    consequence of a better PERCEPT (lower sigma), not a reward written on h."""
    seeds = [1, 2, 3, 4]
    lo = float(np.mean([_fp_rate_at_h(0.05, s) for s in seeds]))
    hi = float(np.mean([_fp_rate_at_h(0.90, s) for s in seeds]))
    assert hi < lo, f"high-h FP-rate {hi:.3f} not below low-h FP-rate {lo:.3f}"


def _fp_rate_zero_confusion(h: float, seed: int) -> float:
    """FP-rate at clamped h with residue_confusion=0 (perfect percept for every h)."""
    f = D.replace(_founder(), thermosense_intensity=h, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    cfg = _compete_cfg(enable_residue=True, residue_confusion=0.0, residue_loss=0.5,
                       founder=f)
    eco = Ecology(cfg, seed=seed)
    eco.run()
    fp = sum(c.phenotype.fp_count for c in eco._creatures)
    tp = sum(c.phenotype.tp_count for c in eco._creatures)
    return fp / max(1, fp + tp)


def test_no_direct_h_reward_confusion_zero():
    """ANTI-CHEAT: with residue_confusion=0 the percept is perfect for EVERY h, so h has
    NO effect on the eat outcome — the FP-rate is identical for a primitive and a precise
    sensor.  This certifies h acts ONLY through perceptual noise, never as a direct reward."""
    seeds = [1, 2, 3, 4]
    lo = float(np.mean([_fp_rate_zero_confusion(0.05, s) for s in seeds]))
    hi = float(np.mean([_fp_rate_zero_confusion(0.90, s) for s in seeds]))
    assert abs(hi - lo) < 1e-9, f"confusion=0 should make h irrelevant: lo={lo} hi={hi}"
