"""Guard tests for the Exp 276a learnable-use (theta) mechanic (gate enable_learnable_use).

band_responsiveness (theta) becomes a HERITABLE genotype trait; the gate controls whether
the creature reads its OWN genotype theta (per-individual, mutated) + pays an L30 upkeep,
or the fixed CONFIG scalar (byte-identical to Exp 194-275).

Covers the binding gate:
  - OFF byte-identical (BINDING): two configs with enable_learnable_use=False that differ in
    ALL the new params (theta_upkeep_floor, theta_cost_slope, AND a mutated founder
    band_responsiveness) produce the IDENTICAL events_hash — the OFF path is byte-identical.
  - ON differs (L42, mechanism live): enable_learnable_use=True with a nonzero cost produces a
    DIFFERENT hash from OFF — the mechanism is live, not a no-op.

Mirrors tests/test_exp204_residue.py (the golden-hash / byte-identical-OFF template).
"""
from __future__ import annotations

import dataclasses as D

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import founder as _founder


def _theta_cfg(**kw) -> EcologyConfig:
    """A small band-staleness forage config (thermosense-forage regime) for fast tests.

    band_responsiveness is exercised through the Exp 201 tracker branch (enable_band_staleness
    + thermosense_forage_mode), so theta actually feeds the percept when the gate is ON.
    """
    f = D.replace(_founder(), thermosense_intensity=0.30, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10)
    base = dict(
        rows=8, cols=8, horizon=300, initial_population=24, founder=f,
        mutation_rate=0.05, capacity=10.0, regen_rate=0.20, initial_resource=0.7,
        max_population=20000, min_survival_energy=4.0, name="exp276a_test",
        enable_thermosense=False, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.30, food_optimal_period=150.0,
        food_concentration=6.0, food_band_width=0.06, enable_food_coupling=True,
        thermosense_forage_mode=True, enable_band_staleness=True, band_responsiveness=0.50,
        shuffle_creature_order=True,
    )
    base.update(kw)
    return EcologyConfig(**base)


def test_learnable_use_off_byte_identical():
    """BINDING: enable_learnable_use=False ⇒ the new params (theta_upkeep_floor, theta_cost_slope,
    and a mutated founder band_responsiveness) do NOT move the events_hash.  The OFF path is
    byte-identical regardless of the theta trait value or the theta-cost knobs."""
    cfg_a = _theta_cfg(enable_learnable_use=False)
    hash_a = Ecology(cfg_a, seed=7).run()["events_hash"]

    # Same config, OFF, but with a MUTATED founder theta AND nonsense theta-cost params —
    # none of this may matter when the gate is OFF (creature reads the CONFIG scalar, no cost).
    f_mut = D.replace(cfg_a.founder, band_responsiveness=0.123)
    cfg_b = _theta_cfg(enable_learnable_use=False, founder=f_mut,
                       theta_upkeep_floor=0.7, theta_cost_slope=0.9)
    hash_b = Ecology(cfg_b, seed=7).run()["events_hash"]

    assert hash_a == hash_b, (
        f"learnable-use params perturbed the OFF-path events_hash: {hash_a} != {hash_b}")


def test_learnable_use_on_differs():
    """L42 (mechanism live): enable_learnable_use=True with a nonzero cost produces a DIFFERENT
    events_hash from the OFF path — the mechanism actually does something (not a no-op)."""
    hash_off = Ecology(_theta_cfg(enable_learnable_use=False), seed=7).run()["events_hash"]
    hash_on = Ecology(
        _theta_cfg(enable_learnable_use=True, theta_upkeep_floor=0.05, theta_cost_slope=0.2),
        seed=7,
    ).run()["events_hash"]
    assert hash_on != hash_off, (
        "enable_learnable_use=True with nonzero cost did not change the events_hash "
        "(mechanism is a no-op)")
