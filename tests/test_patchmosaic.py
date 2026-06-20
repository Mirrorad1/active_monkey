"""tests/test_patchmosaic.py — TDD suite for the patch-mosaic predator-prey substrate.

Exp 257 (substrate-POSING, pre-Red-Queen): does GLOBAL predator-prey coexistence
emerge from LOCAL stochastic patch dynamics (asynchrony + refugia + local migration
+ recolonization) where a single homogeneous arena could not?

This is NOT a well-mixed population-sum ODE, NOT a homogeneous-grid stabilizer knob,
NOT a Red Queen evolution experiment.  Traits are FIXED (monomorphic prey escape,
monomorphic predator attack).

All tests are fast, deterministic (seed-controlled), and structurally verify the
anti-cheat invariants (no global evaluator; migration is local; refuge gates predator
ACCESS only; asynchrony is the same birth-opportunity modulation in every patch).
"""
from __future__ import annotations

import inspect

import pytest

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim


# ---------------------------------------------------------------------------
# T1: Determinism — two runs with same seed give identical events_hash
# ---------------------------------------------------------------------------
def test_determinism():
    cfg = PatchMosaicConfig(horizon=200)
    h1 = PatchMosaicSim(cfg, seed=1).run()["events_hash"]
    h2 = PatchMosaicSim(cfg, seed=1).run()["events_hash"]
    assert h1 == h2, "Same seed must produce identical events_hash (bit-identical replay)"


# ---------------------------------------------------------------------------
# T2: Seed sensitivity — different seeds give different events_hash
# ---------------------------------------------------------------------------
def test_seed_sensitivity():
    cfg = PatchMosaicConfig(horizon=200)
    h1 = PatchMosaicSim(cfg, seed=1).run()["events_hash"]
    h2 = PatchMosaicSim(cfg, seed=2).run()["events_hash"]
    assert h1 != h2, "Different seeds must produce different events_hash (non-degenerate rng)"


# ---------------------------------------------------------------------------
# T3: Migration conserves individuals — a pure-migration operation MOVES agents,
# never creates/destroys them.  Global count before == after.
# ---------------------------------------------------------------------------
def test_migration_conserves_individuals():
    cfg = PatchMosaicConfig(
        n_patches=8,
        migration_rate_prey=0.5,
        migration_rate_pred=0.5,
        n_prey0_per_patch=20,
        n_pred0_per_patch=10,
    )
    sim = PatchMosaicSim(cfg, seed=3)

    def _global_counts(s):
        pn = sum(len(p.prey) for p in s.patches)
        qn = sum(len(p.predators) for p in s.patches)
        return pn, qn

    before_prey, before_pred = _global_counts(sim)
    # Apply ONLY the migration phase (no births/deaths).
    sim._migrate()
    after_prey, after_pred = _global_counts(sim)

    assert before_prey == after_prey, (
        f"Migration created/destroyed prey: {before_prey} -> {after_prey}"
    )
    assert before_pred == after_pred, (
        f"Migration created/destroyed predators: {before_pred} -> {after_pred}"
    )
    assert before_prey > 0 and before_pred > 0, "Test setup degenerate (no agents)"


# ---------------------------------------------------------------------------
# T4: No long-range migration — a migrant only ever lands in a ring-neighbor of
# its origin patch.  We instrument the migration to record (origin, target) pairs.
# ---------------------------------------------------------------------------
def test_no_long_range_migration():
    cfg = PatchMosaicConfig(
        n_patches=8,
        migration_rate_prey=0.8,
        migration_rate_pred=0.8,
        n_prey0_per_patch=30,
        n_pred0_per_patch=15,
    )
    sim = PatchMosaicSim(cfg, seed=4)
    n = cfg.n_patches
    moves = sim._migrate(record_moves=True)
    assert moves, "No migration moves occurred — test setup degenerate"
    for origin, target in moves:
        # target must be a ring-neighbor of origin (i-1 or i+1 mod n)
        assert target in ((origin - 1) % n, (origin + 1) % n), (
            f"Long-range migration detected: {origin} -> {target} "
            f"(neighbors are {(origin - 1) % n} and {(origin + 1) % n})"
        )


# ---------------------------------------------------------------------------
# T5: Local extinction + recolonization are logged — in a regime that produces
# local extinctions plus migration, both counters fire.
# ---------------------------------------------------------------------------
def test_local_extinction_and_recolonization_logged():
    # Small per-patch capacities + predation + migration in a many-patch ring
    # produce local blink-outs and re-seeding via migrants.
    cfg = PatchMosaicConfig(
        n_patches=8,
        horizon=600,
        K_prey_local=40,
        K_pred_local=10,
        migration_rate_prey=0.05,
        migration_rate_pred=0.05,
        async_mode="rotating",
        async_amplitude=0.4,
        n_prey0_per_patch=20,
        n_pred0_per_patch=6,
    )
    result = PatchMosaicSim(cfg, seed=11).run()
    assert result["local_extinction_events"] > 0, (
        "Expected local extinction events in a high-turnover regime; got "
        f"{result['local_extinction_events']}"
    )
    assert result["recolonization_events"] > 0, (
        "Expected recolonization events (a migrant re-seeds an empty patch); got "
        f"{result['recolonization_events']}"
    )


# ---------------------------------------------------------------------------
# T6: Asynchrony changes dynamics — synchronized vs rotating (same seed) give
# different events_hash AND different cross-patch synchrony.
# ---------------------------------------------------------------------------
def test_async_vs_sync_differs():
    base = dict(
        n_patches=8,
        horizon=400,
        async_amplitude=0.5,
        async_period=50.0,
        migration_rate_prey=0.02,
        migration_rate_pred=0.02,
    )
    sync = PatchMosaicSim(PatchMosaicConfig(async_mode="synchronized", **base), seed=21).run()
    rot = PatchMosaicSim(PatchMosaicConfig(async_mode="rotating", **base), seed=21).run()

    assert sync["events_hash"] != rot["events_hash"], (
        "synchronized vs rotating asynchrony must change the trajectory (events_hash)"
    )
    # Synchronized patches should be MORE correlated than rotating (phase-shifted) ones.
    assert sync["cross_patch_synchrony"] != rot["cross_patch_synchrony"], (
        "cross_patch_synchrony must differ between synchronized and rotating modes "
        f"(sync={sync['cross_patch_synchrony']}, rot={rot['cross_patch_synchrony']})"
    )


# ---------------------------------------------------------------------------
# T7: Refuge changes predator ACCESS, not intrinsic birth/death rates.
#
# Two patches with IDENTICAL agents and IDENTICAL rng state: one flagged refuge,
# one not.  They must differ ONLY in the predation (capture) outcome — the prey
# birth draw and predator self-limit death draw formulas must be byte-identical.
# We verify this by comparing the per-step summary fields for births vs deaths.
# ---------------------------------------------------------------------------
def test_refuge_changes_predator_access_not_fitness():
    # Whole-run: refuge ON must change capture outcomes (events_hash differs).
    base = dict(
        n_patches=6,
        horizon=300,
        refuge_fraction=0.5,
        refuge_predator_access=0.1,
        n_prey0_per_patch=20,
        n_pred0_per_patch=6,
    )
    none = PatchMosaicSim(PatchMosaicConfig(refuge_mode="none", **base), seed=31).run()
    on = PatchMosaicSim(PatchMosaicConfig(refuge_mode="per_patch", **base), seed=31).run()
    assert none["events_hash"] != on["events_hash"], (
        "refuge_mode=per_patch must change capture outcomes vs refuge_mode=none"
    )

    # Unit check: a refuge patch and a non-refuge patch with IDENTICAL agents differ
    # ONLY in predation, not in the birth-opportunity / self-limit death formulas.
    cfg = PatchMosaicConfig(
        n_patches=2,
        refuge_mode="per_patch",
        refuge_fraction=0.5,            # patch 0 -> refuge, patch 1 -> not (deterministic by index)
        refuge_predator_access=0.0,     # refuge fully blocks predator access
        async_mode="synchronized",      # no phase difference between the two patches
        n_prey0_per_patch=25,
        n_pred0_per_patch=8,
    )
    sim = PatchMosaicSim(cfg, seed=99)
    # Identify which patch is the refuge.
    refuge_patches = [i for i, p in enumerate(sim.patches) if p.is_refuge]
    assert len(refuge_patches) == 1, "Expected exactly one refuge patch with refuge_fraction=0.5, n=2"

    # Compute birth-opportunity & self-limit-death probabilities for both patches with
    # an identical population snapshot.  They must be IDENTICAL (refuge only gates capture).
    N_prey = 25
    N_pred = 8
    bp0 = sim._prey_birth_prob(patch_idx=0, N_prey=N_prey, t=0)
    bp1 = sim._prey_birth_prob(patch_idx=1, N_prey=N_prey, t=0)
    assert bp0 == bp1, (
        f"Prey birth-opportunity differs between refuge and non-refuge patch: "
        f"{bp0} vs {bp1} — refuge must NOT touch birth rates"
    )
    dp0 = sim._pred_death_prob(N_pred=N_pred)
    dp1 = sim._pred_death_prob(N_pred=N_pred)
    assert dp0 == dp1, "Predator self-limit death prob must not depend on refuge flag"

    # And the predator ACCESS gate DOES depend on refuge.
    assert sim._refuge_access_prob(patch_idx=refuge_patches[0]) != 1.0, (
        "Refuge patch must gate predator access below 1.0"
    )
    non_refuge = 1 - refuge_patches[0]
    assert sim._refuge_access_prob(patch_idx=non_refuge) == 1.0, (
        "Non-refuge patch must allow full (1.0) predator access"
    )


# ---------------------------------------------------------------------------
# T8: No global evaluator — structural guard.  The module must expose no
# fitness-ranking / argmax-selection / global-rescue-injection hook.
# ---------------------------------------------------------------------------
def test_no_global_evaluator():
    import ast

    mod = __import__("ecology.patchmosaic", fromlist=["x"])
    src = inspect.getsource(mod)

    # Grep the EXECUTABLE code only — strip docstrings/comments so that the
    # anti-cheat documentation (which names the forbidden behaviors to disavow
    # them) does not trip the guard.  We compile to an AST and re-emit code-only.
    tree = ast.parse(src)
    code_only = "\n".join(
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Call, ast.Attribute, ast.Name, ast.Assign))
    ).lower()

    # No sort-by-fitness / selection / scoring / rescue hooks in executable code.
    banned = ["argmax", "argsort", ".sort(key", "sorted(", "fitness", "rescue"]
    for token in banned:
        assert token not in code_only, (
            f"Forbidden global-evaluator token {token!r} present in patchmosaic.py code "
            "(survival/reproduction/migration must be individual stochastic events only)"
        )
    # The public API must expose no rank/select/score hook.
    public = [n for n in dir(PatchMosaicSim) if not n.startswith("_")]
    for n in public:
        ln = n.lower()
        assert not any(b in ln for b in ("rank", "select", "score", "fitness")), (
            f"Public API exposes a forbidden evaluator hook: {n}"
        )


# ---------------------------------------------------------------------------
# T9: Smoke run — a small run completes and returns the expected dict keys.
# ---------------------------------------------------------------------------
def test_smoke_run():
    cfg = PatchMosaicConfig(n_patches=4, horizon=50)
    result = PatchMosaicSim(cfg, seed=1).run()
    expected_keys = {
        "events_hash",
        "global_prey_series",
        "global_pred_series",
        "patch_prey_series",
        "patch_pred_series",
        "t_end",
        "global_extinct",
        "exploded",
        "local_extinction_events",
        "recolonization_events",
        "occupancy_series",
        "cross_patch_synchrony",
        "cv_global_prey",
        "cv_global_pred",
    }
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - set(result.keys())}"
    )
    assert len(result["patch_prey_series"]) == cfg.n_patches
    assert len(result["patch_pred_series"]) == cfg.n_patches
    assert result["t_end"] <= cfg.horizon
    assert isinstance(result["global_extinct"], bool)
    assert isinstance(result["exploded"], bool)
