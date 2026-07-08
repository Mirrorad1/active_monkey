"""Guard tests for the Exp 277 WITHIN-LIFE theta learner (gate enable_theta_learning).

A creature keeps its INNATE theta (heritable genotype.band_responsiveness, Exp 276) AND a
per-life REALIZED theta it LEARNS within its lifetime by a 1-D stochastic hill-climb starting
from the innate prior.  When the gate is ON the tracker uses the REALIZED theta; when OFF it
uses the innate theta — byte-identical to the Exp 276 enable_learnable_use path.

Covers the binding gate:
  - OFF byte-identical (BINDING): enable_theta_learning=False with differing new params
    (theta_learn_period / theta_learn_step / theta_learn_cost) ⇒ IDENTICAL events_hash.
  - ON differs (mechanism live): enable_theta_learning=True ⇒ a DIFFERENT hash.
  - NON-LAMARCKIAN (binding): in an ON run, a child's INNATE genotype.band_responsiveness is a
    MUTATION of its parent's INNATE band_responsiveness — NOT a copy of the parent's LEARNED
    realized_theta.

Mirrors tests/test_exp276_learnable_use.py (the golden-hash / byte-identical-OFF template).
"""
from __future__ import annotations

import dataclasses as D

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import founder as _founder


def _theta_cfg(**kw) -> EcologyConfig:
    """The Exp 276a band-staleness forage config with the theta channel LIVE.

    band_responsiveness is exercised through the Exp 201 tracker branch (enable_band_staleness
    + thermosense_forage_mode), so theta actually feeds the percept.  enable_learnable_use is
    ON so the innate theta channel is active; enable_theta_learning is toggled per test.
    """
    f = D.replace(_founder(), thermosense_intensity=0.30, thermosense_inefficiency=0.20,
                  temperature_tolerance=0.10, band_responsiveness=0.10)
    base = dict(
        rows=8, cols=8, horizon=300, initial_population=24, founder=f,
        mutation_rate=0.05, capacity=10.0, regen_rate=0.20, initial_resource=0.7,
        max_population=20000, min_survival_energy=4.0, name="exp277_test",
        enable_thermosense=False, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        food_optimal_base=0.5, food_optimal_amplitude=0.30, food_optimal_period=150.0,
        food_concentration=6.0, food_band_width=0.06, enable_food_coupling=True,
        thermosense_forage_mode=True, enable_band_staleness=True, band_responsiveness=0.50,
        shuffle_creature_order=True,
        enable_learnable_use=True,
    )
    base.update(kw)
    return EcologyConfig(**base)


def test_theta_learning_off_byte_identical():
    """BINDING: enable_theta_learning=False ⇒ the new hill-climb params (theta_learn_period,
    theta_learn_step, theta_learn_cost) do NOT move the events_hash.  The OFF path is
    byte-identical to the Exp 276 innate-theta path regardless of the learner knobs."""
    cfg_a = _theta_cfg(enable_theta_learning=False, theta_cost_slope=0.2, theta_upkeep_floor=0.05)
    hash_a = Ecology(cfg_a, seed=7).run()["events_hash"]

    # Same config, learner OFF, but with NONSENSE hill-climb params — none may matter OFF.
    cfg_b = _theta_cfg(enable_theta_learning=False, theta_cost_slope=0.2, theta_upkeep_floor=0.05,
                       theta_learn_period=3, theta_learn_step=0.5, theta_learn_cost=0.9)
    hash_b = Ecology(cfg_b, seed=7).run()["events_hash"]

    assert hash_a == hash_b, (
        f"theta-learning params perturbed the OFF-path events_hash: {hash_a} != {hash_b}")


def test_theta_learning_on_differs():
    """Mechanism live: enable_theta_learning=True produces a DIFFERENT events_hash from OFF —
    the within-life learner actually does something (not a no-op)."""
    hash_off = Ecology(
        _theta_cfg(enable_theta_learning=False, theta_cost_slope=0.2), seed=7).run()["events_hash"]
    hash_on = Ecology(
        _theta_cfg(enable_theta_learning=True, theta_cost_slope=0.2), seed=7).run()["events_hash"]
    assert hash_on != hash_off, (
        "enable_theta_learning=True did not change the events_hash (learner is a no-op)")


def test_theta_learning_non_lamarckian():
    """BINDING (non-Lamarckian): a child's INNATE genotype.band_responsiveness is a MUTATION of
    its PARENT's INNATE band_responsiveness, NEVER a copy of the parent's LEARNED realized_theta.

    We run an ON population where the learner demonstrably MOVES realized_theta away from the low
    innate prior, then check every parent→child pair: the child's innate theta must sit near the
    parent's INNATE theta (within a few mutation sigmas), and must NOT be systematically pulled
    toward the parent's (much higher) learned realized_theta.
    """
    cfg = _theta_cfg(enable_theta_learning=True, theta_cost_slope=0.05,
                     regen_rate=0.6, theta_learn_period=10, theta_learn_step=0.2)
    eco = Ecology(cfg, seed=3)
    while eco.t < cfg.horizon and eco.has_alive():
        eco.step()

    creatures = eco._creatures
    by_id = {c.creature_id: c for c in creatures}

    # Sanity: the learner actually moved realized_theta above the innate prior for many creatures
    # (otherwise the test is vacuous — realized≈innate can't distinguish the two inheritance rules).
    moved = [c for c in creatures
             if c.policy is not None and c.policy.realized_theta is not None
             and c.policy.realized_theta - c.genotype.band_responsiveness > 0.1]
    assert len(moved) >= 5, (
        f"learner did not move realized_theta meaningfully above innate for enough creatures "
        f"({len(moved)}) — non-Lamarckian test would be vacuous")

    # Collect parent→child pairs where the parent's realized theta is well ABOVE its innate theta,
    # so a Lamarckian bug (copying realized into the child genotype) would be detectable.
    child_innate: list[float] = []
    parent_innate: list[float] = []
    parent_realized: list[float] = []
    for child in creatures:
        pid = child.parent_id
        if pid is None or pid not in by_id:
            continue
        parent = by_id[pid]
        if parent.policy is None or parent.policy.realized_theta is None:
            continue
        gap = parent.policy.realized_theta - parent.genotype.band_responsiveness
        if gap <= 0.1:
            continue  # parent's learned value not distinguishable from its innate value
        child_innate.append(child.genotype.band_responsiveness)
        parent_innate.append(parent.genotype.band_responsiveness)
        parent_realized.append(parent.policy.realized_theta)

    assert len(child_innate) >= 5, (
        f"not enough parent→child pairs with a learned/innate gap to test ({len(child_innate)})")

    child_innate = np.array(child_innate)
    parent_innate = np.array(parent_innate)
    parent_realized = np.array(parent_realized)

    # A child's innate theta must track the parent's INNATE theta (mutation), not the parent's
    # LEARNED realized theta.  So the mean |child_innate - parent_innate| must be MUCH smaller
    # than the mean |child_innate - parent_realized| (the parents were selected for a big gap).
    err_to_innate = float(np.mean(np.abs(child_innate - parent_innate)))
    err_to_realized = float(np.mean(np.abs(child_innate - parent_realized)))
    assert err_to_innate < err_to_realized, (
        f"child innate theta tracks parent's LEARNED realized theta more than its innate theta "
        f"(err_to_innate={err_to_innate:.4f} !< err_to_realized={err_to_realized:.4f}) "
        f"— inheritance looks LAMARCKIAN")

    # And each child's innate theta must be within a few mutation sigmas of the parent's innate
    # theta (mutation_rate*(hi-lo) = 0.05*1.0 = 0.05; allow 5 sigma slack for the clamp/tails).
    sigma = cfg.mutation_rate * 1.0
    max_dev = float(np.max(np.abs(child_innate - parent_innate)))
    assert max_dev <= 6 * sigma, (
        f"a child's innate theta deviates {max_dev:.4f} from its parent's innate theta "
        f"(> 6*sigma={6*sigma:.4f}) — not a plain mutation of the innate trait")
