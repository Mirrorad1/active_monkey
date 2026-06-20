"""tests/test_exp260_preflight.py — guard for the Exp-260 one-sided-queen preflight.

The preflight's entire validity rests on its DRIFT NULLS being genuine: a regime where the
focal trait is causally INERT must show no selection, so that a positive readout elsewhere is
real selection and not a measurement artifact. These tests pin that property directly on the
substrate (fast, deterministic) so it cannot silently rot:

  - In the prey drift-null regime (escape_cost=0 AND predator attack high enough that a
    resident+eps mutant still has escape < attack => v == 1), relabeling half the prey as the
    +eps mutant must NOT change the trajectory at all — the events_hash (a counts-only digest)
    is byte-identical. (binding control is causally inert)
  - Non-vacuity: in a regime where escape IS active (predator attack == resident escape, cost
    on), the SAME relabeling DOES change the trajectory — so the inertness test above is a real
    constraint, not a tautology.
"""
from __future__ import annotations

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim


def _regime(escape_cost, pred_attack):
    return dict(
        n_patches=4, topology="ring", horizon=40,
        attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=0.0,
        escape_cost=escape_cost, escape_baseline=1.0,
        prey_escape=1.0, pred_attack=pred_attack,
        freeze_prey_trait=True, freeze_predator_trait=True,
        n_prey0_per_patch=20, n_pred0_per_patch=6,
    )


def _hash(escape_cost, pred_attack, relabel, eps=0.1, seed=500):
    cfg = PatchMosaicConfig(**_regime(escape_cost, pred_attack))
    sim = PatchMosaicSim(cfg, seed)
    if relabel:
        for patch in sim.patches:
            for k, c in enumerate(patch.prey):
                if k % 2 == 1:
                    c.trait = 1.0 + eps
    return sim.run()["events_hash"]


def test_drift_null_escape_is_causally_inert():
    """escape_cost=0 + predator attack=2.0 (>= mutant escape 1.1) => v==1 for both lineages =>
    relabeling half the prey as the +eps mutant leaves the trajectory byte-identical."""
    base = _hash(escape_cost=0.0, pred_attack=2.0, relabel=False)
    relabeled = _hash(escape_cost=0.0, pred_attack=2.0, relabel=True)
    assert base == relabeled, (
        "drift-null escape trait must be causally INERT (counts byte-identical); "
        "if this fails the Exp-260 drift control is not actually a no-effect null"
    )


def test_active_escape_does_change_dynamics():
    """Non-vacuity: with the cost on and predator attack == resident escape (so a faster mutant
    is genuinely less vulnerable), the same relabeling MUST change the trajectory."""
    base = _hash(escape_cost=0.15, pred_attack=1.0, relabel=False)
    relabeled = _hash(escape_cost=0.15, pred_attack=1.0, relabel=True)
    assert base != relabeled, (
        "an ACTIVE escape trait must change the trajectory when relabeled; "
        "if this fails, the inertness test above is vacuous"
    )
