"""Exp 203 sense-gradient audit — engine + abstraction regression/validity guards.

The audit rests on two gated engine features (freeze_thermosense, founder_mix) and the
ecology/sense_axis.py instrument. These guards pin: (i) the OFF paths are byte-identical to
exp194-202 (no rng/seeding drift); (ii) freeze_thermosense breeds a clamp value TRUE while
keeping the rng stream intact and the upkeep charged; (iii) founder_mix seeds the explicit
polymorphic population without touching the no-hidden-evaluator invariant; (iv) the audit
never smuggles a direct-h-reward (the anti-cheat assert fires on a misconfigured cfg).
"""
from __future__ import annotations

import dataclasses as D

import numpy as np
import pytest

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import mutate
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology import sense_axis as SA


_BASE = dict(
    enable_thermosense=True, enable_temperature=True, temperature_stress_scale=0.0,
    thermosense_upkeep_floor=0.0, thermosense_active_threshold=0.05, thermosense_noise_base=0.5,
    thermal_avoidance_weight=4.0, food_optimal_base=0.5, food_optimal_amplitude=0.3,
    food_optimal_period=1500.0, food_concentration=14.0, food_band_width=0.08,
    enable_food_coupling=True, thermosense_forage_mode=True,
)


def _cfg(horizon=400, regen_rate=0.08, **kw):
    f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    return D.replace(SCENARIOS["balanced"], horizon=horizon, max_population=8000, founder=f,
                     regen_rate=regen_rate, shuffle_creature_order=True, **_BASE, **kw)


# --- byte-identical OFF paths (the regression guards) -----------------------

def test_freeze_thermosense_off_byte_identical():
    """freeze_thermosense=False (default) must equal a run without the flag — no rng drift."""
    base = Ecology(_cfg(), 38); base.run()
    off = Ecology(_cfg(freeze_thermosense=False), 38); off.run()
    assert off.events_hash() == base.events_hash()


def test_founder_mix_none_byte_identical():
    """founder_mix=None (default) must equal a run without it — existing single-founder seeding."""
    base = Ecology(_cfg(), 38); base.run()
    off = Ecology(_cfg(founder_mix=None), 38); off.run()
    assert off.events_hash() == base.events_hash()


def test_exp194_plain_byte_identical_with_new_flags_default():
    """A plain Exp 194 run (no thermosense) is unchanged by the new defaulted fields."""
    base = Ecology(D.replace(SCENARIOS["balanced"], horizon=600), 0); base.run()
    new = Ecology(D.replace(SCENARIOS["balanced"], horizon=600,
                            freeze_thermosense=False, founder_mix=None), 0); new.run()
    assert new.events_hash() == base.events_hash()


# --- freeze_thermosense breeds true + rng-stream parity ----------------------

def test_freeze_thermosense_pins_organ_and_preserves_stream():
    rng = np.random.default_rng(7)
    g0 = D.replace(FOUNDER, thermosense_intensity=0.45, thermosense_inefficiency=0.2)
    child = mutate(g0, rng, 0.1, mutate_thermosense=True, freeze_thermosense=True)
    assert child.thermosense_intensity == 0.45 and child.thermosense_inefficiency == 0.2
    # the perturbation is drawn-then-discarded: the NEXT draw is identical to a normal mutate
    ra = np.random.default_rng(7); ca = mutate(g0, ra, 0.1, mutate_thermosense=True, freeze_thermosense=False)
    rb = np.random.default_rng(7); cb = mutate(g0, rb, 0.1, mutate_thermosense=True, freeze_thermosense=True)
    assert float(ra.normal()) == float(rb.normal())          # stream parity
    assert ca.movement_cost == cb.movement_cost              # base traits unchanged by the freeze
    assert ca.energy_capacity == cb.energy_capacity


def test_freeze_thermosense_breeds_true_in_a_run():
    """With freeze_thermosense ON, EVERY creature carries exactly the founder's clamped intensity."""
    base_f = D.replace(FOUNDER, thermosense_intensity=0.30, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    cfg = D.replace(_cfg(horizon=600, regen_rate=0.20, freeze_thermosense=True), founder=base_f)
    eco = Ecology(cfg, 50); eco.run()
    assert all(abs(c.genotype.thermosense_intensity - 0.30) < 1e-9 for c in eco._creatures)


# --- founder_mix seeding ----------------------------------------------------

def test_founder_mix_seeds_explicit_polymorphic_population():
    base_f = D.replace(FOUNDER, temperature_tolerance=0.10)
    fm = SA.founder_mix_equal(base_f, grid=(0.0, 0.10, 0.30), per_clamp=4)   # 12 founders
    cfg = _cfg(horizon=10, regen_rate=0.20, freeze_thermosense=True, founder_mix=fm)
    eco = Ecology(cfg, 50)
    founders = [c for c in eco._creatures if c.generation == 0]
    assert len(founders) == 12
    from collections import Counter
    by_h = Counter(round(c.genotype.thermosense_intensity, 3) for c in founders)
    assert by_h[0.0] == 4 and by_h[0.10] == 4 and by_h[0.30] == 4


def test_audit_clamp_integrity_no_grid_escape():
    """In a freeze_thermosense common garden, NO creature's intensity escapes the clamp grid."""
    base_f = D.replace(FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2, temperature_tolerance=0.10)
    fm = SA.founder_mix_equal(base_f, per_clamp=6)
    cfg = _cfg(horizon=1000, regen_rate=0.20, founder_mix=fm)
    cfg = D.replace(cfg, freeze_thermosense=True)
    out = SA.run_gradient_audit(cfg, seed=50, window=(100, 700), checkpoint_stride=100, min_clamp_pop=4)
    assert out["other_intensities"] == 0


# --- anti-cheat guard fires -------------------------------------------------

def test_assert_no_direct_h_reward_requires_cost_on_and_freeze():
    base_f = D.replace(FOUNDER, temperature_tolerance=0.10)
    fm = SA.founder_mix_equal(base_f, per_clamp=2)
    good = _cfg(founder_mix=fm); good = D.replace(good, freeze_thermosense=True)
    SA.assert_no_direct_h_reward(good)                       # no raise
    with pytest.raises(AssertionError):                     # cost OFF -> not a valid verdict cfg
        SA.assert_no_direct_h_reward(D.replace(good, enable_thermosense=False))
    with pytest.raises(AssertionError):                     # not freezing -> clamp wouldn't breed true
        SA.assert_no_direct_h_reward(D.replace(good, freeze_thermosense=False))


def test_sense_axis_cost_is_floored_and_h_keyed():
    ax = SA.THERMOSENSE_AXIS
    assert ax.cost(0.0) == 0.0 and ax.cost(0.04) == 0.0      # below activation threshold = free
    assert ax.cost(0.10) == pytest.approx(0.20 * 0.10)       # floor 0 + inefficiency*h
    assert ax.cost(0.60) > ax.cost(0.10)                     # monotone in h (never free)
