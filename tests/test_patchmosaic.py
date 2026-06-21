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


# ---------------------------------------------------------------------------
# T10: Ring topology unchanged — topology="ring" (default) gives IDENTICAL
# events_hash to a run that does not set topology at all (byte-identical default).
# The golden hash is captured at commit time; a regression means the ring's rng
# stream changed and the test must FAIL (not be updated silently).
# ---------------------------------------------------------------------------
_RING_GOLDEN_HASH = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"

def test_ring_topology_unchanged():
    cfg_default = PatchMosaicConfig(horizon=200)
    cfg_explicit = PatchMosaicConfig(horizon=200, topology="ring")

    h_default  = PatchMosaicSim(cfg_default,  seed=1).run()["events_hash"]
    h_explicit = PatchMosaicSim(cfg_explicit, seed=1).run()["events_hash"]

    assert h_default == _RING_GOLDEN_HASH, (
        f"Ring default hash changed! Expected {_RING_GOLDEN_HASH!r}, got {h_default!r}. "
        "The ring rng stream must be byte-identical to the original implementation."
    )
    assert h_explicit == _RING_GOLDEN_HASH, (
        f"topology='ring' explicit hash changed! Expected {_RING_GOLDEN_HASH!r}, "
        f"got {h_explicit!r}. Setting topology='ring' must not alter the rng stream."
    )


# ---------------------------------------------------------------------------
# T11: grid2d adjacency — 4x4 grid gives every patch exactly 4 distinct neighbors,
# symmetric (i in neighbors[j] iff j in neighbors[i]), and torus-wrapped.
# ---------------------------------------------------------------------------
def test_grid2d_adjacency():
    cfg = PatchMosaicConfig(n_patches=16, topology="grid2d", grid_cols=4)
    sim = PatchMosaicSim(cfg, seed=42)
    n = cfg.n_patches
    nb = sim.neighbors

    for i in range(n):
        assert len(nb[i]) == 4, (
            f"Patch {i} has {len(nb[i])} neighbors, expected 4 (von Neumann on torus)."
        )
        assert len(set(nb[i])) == 4, f"Patch {i} has duplicate neighbors: {nb[i]}"
        assert i not in nb[i], f"Patch {i} is its own neighbor (self-loop)."
        for j in nb[i]:
            assert i in nb[j], (
                f"Asymmetry: {j} in neighbors[{i}] but {i} not in neighbors[{j}]."
            )


# ---------------------------------------------------------------------------
# T12: smallworld adjacency — connected, symmetric, deterministic under seed,
# and differs from the pure ring when rewire > 0.
# ---------------------------------------------------------------------------
def test_smallworld_adjacency():
    cfg = PatchMosaicConfig(n_patches=12, topology="smallworld", smallworld_rewire=0.3)
    sim1 = PatchMosaicSim(cfg, seed=7)
    sim2 = PatchMosaicSim(cfg, seed=7)
    n = cfg.n_patches
    nb1, nb2 = sim1.neighbors, sim2.neighbors

    # Deterministic under same seed.
    assert nb1 == nb2, "smallworld neighbors are not deterministic under same seed."

    # Symmetric and no self-loops.
    for i in range(n):
        assert i not in nb1[i], f"Patch {i} is its own neighbor."
        for j in nb1[i]:
            assert i in nb1[j], f"Asymmetry: {j} in neighbors[{i}] but {i} not in neighbors[{j}]."

    # Connected (every node reachable from 0 via BFS).
    from collections import deque
    visited = set()
    q = deque([0])
    visited.add(0)
    while q:
        node = q.popleft()
        for nb in nb1[node]:
            if nb not in visited:
                visited.add(nb)
                q.append(nb)
    assert len(visited) == n, f"smallworld graph is disconnected: only {len(visited)}/{n} reachable."

    # Differs from pure ring (some long-range edges present when rewire > 0).
    ring_nb = [sorted([(i - 1) % n, (i + 1) % n]) for i in range(n)]
    assert nb1 != ring_nb, (
        "smallworld neighbors are identical to the ring despite rewire=0.3; "
        "expected at least one rewired edge."
    )


# ---------------------------------------------------------------------------
# T13: Migration only goes to graph-neighbors, for all three topologies.
# ---------------------------------------------------------------------------
def test_migration_neighbors_only_all_topologies():
    common = dict(
        n_patches=8,
        migration_rate_prey=0.9,
        migration_rate_pred=0.9,
        n_prey0_per_patch=20,
        n_pred0_per_patch=10,
    )
    for topo in ("ring", "grid2d", "smallworld"):
        extra = {}
        if topo == "grid2d":
            extra["grid_cols"] = 4  # 8 patches / 4 cols = 2x4 grid
        cfg = PatchMosaicConfig(topology=topo, **common, **extra)
        sim = PatchMosaicSim(cfg, seed=55)
        moves = sim._migrate(record_moves=True)
        assert moves, f"topology={topo!r}: no migration moves occurred — test setup degenerate."
        for origin, target in moves:
            assert target in sim.neighbors[origin], (
                f"topology={topo!r}: long-range move {origin} -> {target}; "
                f"valid neighbors of {origin} are {sim.neighbors[origin]}."
            )


# ---------------------------------------------------------------------------
# T14: grid2d requires n_patches divisible by grid_cols.
# ---------------------------------------------------------------------------
def test_grid2d_requires_valid_cols():
    # n_patches=10 is not divisible by grid_cols=3 (10 % 3 = 1).
    cfg = PatchMosaicConfig(n_patches=10, topology="grid2d", grid_cols=3)
    with pytest.raises(ValueError, match="not divisible"):
        PatchMosaicSim(cfg, seed=1)

    # grid_cols=0 auto-derives cols=round(sqrt(10))=3; 10 % 3 = 1 => also ValueError.
    cfg2 = PatchMosaicConfig(n_patches=10, topology="grid2d", grid_cols=0)
    with pytest.raises(ValueError, match="not divisible"):
        PatchMosaicSim(cfg2, seed=1)

    # grid_cols=0, n_patches=9 => cols=round(sqrt(9))=3; 9 % 3 = 0 => should succeed.
    cfg3 = PatchMosaicConfig(n_patches=9, topology="grid2d", grid_cols=0)
    sim3 = PatchMosaicSim(cfg3, seed=1)
    assert len(sim3.neighbors) == 9
    for i in range(9):
        assert len(sim3.neighbors[i]) == 4


# ---------------------------------------------------------------------------
# T15: Trait evolution OFF is byte-identical — explicit False and default both
# produce the same events_hash as the existing ring golden T10.
# ---------------------------------------------------------------------------
def test_trait_evolution_off_byte_identical():
    """Gate OFF (explicit and default) must be byte-identical to the baseline ring run."""
    _RING_GOLDEN_HASH = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"

    cfg_default = PatchMosaicConfig(horizon=200)
    cfg_explicit_off = PatchMosaicConfig(horizon=200, enable_trait_evolution=False)

    h_default = PatchMosaicSim(cfg_default, seed=1).run()["events_hash"]
    h_explicit_off = PatchMosaicSim(cfg_explicit_off, seed=1).run()["events_hash"]

    assert h_default == _RING_GOLDEN_HASH, (
        f"Default config hash changed from ring golden! Got {h_default!r}. "
        "enable_trait_evolution=False (default) must be byte-identical to the baseline."
    )
    assert h_explicit_off == _RING_GOLDEN_HASH, (
        f"enable_trait_evolution=False (explicit) hash changed from ring golden! "
        f"Got {h_explicit_off!r}. Must be byte-identical to the baseline."
    )
    assert h_default == h_explicit_off, (
        "Default and explicit False must produce identical events_hash."
    )


# ---------------------------------------------------------------------------
# T16: Trait evolution ON produces a different events_hash from the gate-OFF run.
# ---------------------------------------------------------------------------
def test_trait_evolution_on_differs():
    """Gate ON with mutation must diverge from gate OFF (different rng stream)."""
    cfg_off = PatchMosaicConfig(horizon=200)
    cfg_on = PatchMosaicConfig(horizon=200, enable_trait_evolution=True, mutation_rate=0.1)

    h_off = PatchMosaicSim(cfg_off, seed=1).run()["events_hash"]
    h_on = PatchMosaicSim(cfg_on, seed=1).run()["events_hash"]

    assert h_off != h_on, (
        "enable_trait_evolution=True with mutation_rate=0.1 must produce a different "
        "events_hash than the gate-OFF run (trajectories diverge due to cost + mutation)."
    )


# ---------------------------------------------------------------------------
# T17: Escape cost reduces prey birth probability for high-escape individuals.
# ---------------------------------------------------------------------------
def test_escape_cost_reduces_prey_birth():
    """A prey with escape > escape_baseline has strictly lower effective birth_p."""
    cfg = PatchMosaicConfig(
        horizon=1,
        enable_trait_evolution=True,
        escape_cost=0.15,
        escape_baseline=1.0,
    )
    sim = PatchMosaicSim(cfg, seed=1)

    # Compute the base birth probability for a typical patch state.
    N_prey = 40
    base_birth_p = sim._prey_birth_prob(patch_idx=0, N_prey=N_prey, t=0)
    assert base_birth_p > 0.0, "Test setup degenerate: base birth_p must be positive."

    # A prey at baseline incurs zero cost.
    prey_at_baseline = 1.0  # == escape_baseline
    cost_at_baseline = max(0.0, 1.0 - cfg.escape_cost * max(0.0, prey_at_baseline - cfg.escape_baseline))
    birth_p_at_baseline = base_birth_p * cost_at_baseline

    # A prey above baseline incurs a cost.
    prey_above = 2.0  # above escape_baseline=1.0
    cost_above = max(0.0, 1.0 - cfg.escape_cost * max(0.0, prey_above - cfg.escape_baseline))
    birth_p_above = base_birth_p * cost_above

    assert birth_p_above < birth_p_at_baseline, (
        f"Escape cost must reduce birth probability for trait > escape_baseline: "
        f"birth_p(trait=1.0)={birth_p_at_baseline:.6f} vs birth_p(trait=2.0)={birth_p_above:.6f}."
    )
    assert birth_p_at_baseline == pytest.approx(base_birth_p, rel=1e-10), (
        "A prey AT the escape_baseline must pay no fecundity cost."
    )


# ---------------------------------------------------------------------------
# T18: freeze_predator_trait keeps predator trait-mean constant while prey can move.
# ---------------------------------------------------------------------------
def test_freeze_predator_trait():
    """freeze_predator_trait=True -> predator trait frozen; =False -> can drift."""
    common = dict(
        horizon=300,
        n_patches=4,
        enable_trait_evolution=True,
        mutation_rate=0.5,   # high mutation so trait moves quickly
        mutation_sd=0.2,
        escape_cost=0.0,     # no cost so prey population survives for many steps
    )

    # Frozen predator: predator trait should never change from initial pred_attack.
    cfg_frozen = PatchMosaicConfig(freeze_predator_trait=True, **common)
    sim_frozen = PatchMosaicSim(cfg_frozen, seed=42)
    result_frozen = sim_frozen.run()

    # Check that final predator trait-mean is still the initial pred_attack.
    all_pred_traits_frozen = [
        q.trait for patch in sim_frozen.patches for q in patch.predators
    ]
    if all_pred_traits_frozen:
        assert all(t == cfg_frozen.pred_attack for t in all_pred_traits_frozen), (
            f"freeze_predator_trait=True: predator traits should remain at "
            f"{cfg_frozen.pred_attack}, but got {set(all_pred_traits_frozen)!r}."
        )

    # Unfrozen predator: predator trait should drift from initial value.
    cfg_free = PatchMosaicConfig(freeze_predator_trait=False, **common)
    sim_free = PatchMosaicSim(cfg_free, seed=42)
    sim_free.run()

    all_pred_traits_free = [
        q.trait for patch in sim_free.patches for q in patch.predators
    ]
    if all_pred_traits_free:
        # With mutation_rate=0.5 over 300 steps, at least some predators should
        # have a trait != pred_attack (1.0) with overwhelming probability.
        assert not all(t == cfg_free.pred_attack for t in all_pred_traits_free), (
            "freeze_predator_trait=False with high mutation_rate: predator traits "
            "should drift from the initial value but all remained unchanged."
        )


# ---------------------------------------------------------------------------
# T19: Trait evolution with gate ON is deterministic (same seed => same hash).
# ---------------------------------------------------------------------------
def test_trait_evolution_deterministic():
    """Gate ON run must be deterministic: two identical configs + seed => same events_hash."""
    cfg = PatchMosaicConfig(
        horizon=200,
        enable_trait_evolution=True,
        mutation_rate=0.1,
        mutation_sd=0.05,
    )
    h1 = PatchMosaicSim(cfg, seed=7).run()["events_hash"]
    h2 = PatchMosaicSim(cfg, seed=7).run()["events_hash"]

    assert h1 == h2, (
        "enable_trait_evolution=True run must be deterministic: "
        f"two identical runs with seed=7 produced different hashes: {h1!r} vs {h2!r}."
    )


# ---------------------------------------------------------------------------
# T20 (Exp 261): predator attack_cost is byte-identical OFF and LIVE when ON.
# The cost factor (a fecundity scaler on predator birth, mirroring escape_cost)
# must add ZERO rng draws: with attack_cost=0.0 (default) it is exactly 1.0, so
# every prior hash is preserved.  T16 pins no ON-hash literal (only h_off != h_on),
# so we pin a fresh ON golden here and prove (a) default attack_cost=0 leaves it
# byte-identical, (b) attack_cost>0 on a high-attack evolving config DIVERGES (the
# cost is a live branch, not a silent no-op), (c) attack_cost>0 with the gate OFF
# never perturbs the ring golden.
# ---------------------------------------------------------------------------
_TRAIT_ON_GOLDEN = "790a8499be51644f255f3e431ac0488612dd625047d696d1f646b7919fef7623"
_RING_GOLDEN_HASH_T20 = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"


def test_attack_cost_byte_identical_off_and_live_on():
    def run_hash(**kw):
        seed = kw.pop("seed", 1)
        return PatchMosaicSim(PatchMosaicConfig(**kw), seed).run()["events_hash"]

    # (a) trait-evolution ON with attack_cost at its 0.0 default == pinned ON golden.
    h_on_default = run_hash(horizon=200, enable_trait_evolution=True, mutation_rate=0.1)
    assert h_on_default == _TRAIT_ON_GOLDEN, (
        f"attack_cost=0.0 (default) must leave the evolving-ON hash byte-identical to the "
        f"pre-attack_cost golden. Expected {_TRAIT_ON_GOLDEN!r}, got {h_on_default!r}."
    )

    # (b) attack_cost>0 on a high-attack evolving config MUST diverge (cost branch is live).
    h_on_costed = run_hash(horizon=200, enable_trait_evolution=True, mutation_rate=0.1,
                           attack_cost=0.4, pred_attack=2.5)
    assert h_on_costed != _TRAIT_ON_GOLDEN, (
        "attack_cost=0.4 with high-attack predators must change the events_hash — "
        "if it matches the cost=0 golden, the cost factor is a silent no-op."
    )

    # (c) attack_cost>0 with the gate OFF must NOT perturb the ring golden (gated path).
    h_off_costed = run_hash(horizon=200, enable_trait_evolution=False, attack_cost=0.4, pred_attack=2.5)
    assert h_off_costed == _RING_GOLDEN_HASH_T20, (
        "A non-default attack_cost with enable_trait_evolution=False must never execute the "
        f"cost branch. Expected ring golden {_RING_GOLDEN_HASH_T20!r}, got {h_off_costed!r}."
    )


# ---------------------------------------------------------------------------
# T21: Critter carries aggr + lineage fields with correct defaults (byte-identical)
# ---------------------------------------------------------------------------
def test_critter_new_fields_default_and_byte_identical():
    from ecology.patchmosaic import Critter, PatchMosaicConfig, PatchMosaicSim
    c = Critter("prey", 1.0, 0)
    assert c.aggr == 0.0 and c.lineage == -1
    h = PatchMosaicSim(PatchMosaicConfig(horizon=200), seed=1).run()["events_hash"]
    assert h == "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"


# ---------------------------------------------------------------------------
# T22: Contest config fields exist with correct defaults and are inert
# ---------------------------------------------------------------------------
def test_contest_config_defaults_inert():
    from ecology.patchmosaic import PatchMosaicConfig
    c = PatchMosaicConfig()
    assert c.enable_contest is False and c.aggr0 == 0.0 and c.track_lineages is False
    assert c.contest_cost == 0.10 and c.contest_seize == 0.50 and c.contest_dissipation == 0.0


# ---------------------------------------------------------------------------
# T23: Founders seed aggr0 + per-patch lineage ids (byte-identical)
# ---------------------------------------------------------------------------
def test_founders_seed_aggr_and_lineage_byte_identical():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    sim = PatchMosaicSim(PatchMosaicConfig(aggr0=0.3, n_patches=4), seed=1)
    assert all(c.aggr == 0.3 for p in sim.patches for c in p.prey)
    assert all(c.lineage == p.idx for p in sim.patches for c in p.prey)
    # aggr0 alone (contest OFF) must not change the golden
    h = PatchMosaicSim(PatchMosaicConfig(horizon=200), seed=1).run()["events_hash"]
    assert h == "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"


# ---------------------------------------------------------------------------
# T24: Contest gate OFF is byte-identical; ON with aggr0=0 is also byte-identical
# ---------------------------------------------------------------------------
def test_contest_byte_identical_off_and_inert():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    GOLD = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
    # OFF (default)
    assert PatchMosaicSim(PatchMosaicConfig(horizon=200), 1).run()["events_hash"] == GOLD
    # ON but aggr0=0 and no aggr mutation => no creature contests => byte-identical
    cfg_inert = PatchMosaicConfig(horizon=200, enable_contest=True, aggr0=0.0)
    assert PatchMosaicSim(cfg_inert, 1).run()["events_hash"] == GOLD


# ---------------------------------------------------------------------------
# T25: Contest LIVE when aggr0>0 — events_hash must differ from ring golden
# ---------------------------------------------------------------------------
def test_contest_live_changes_hash():
    from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
    GOLD = "d063c91fe091c3591529036dd102e35480319632e286fd2c17e71c9d4aafcbc5"
    cfg = PatchMosaicConfig(horizon=200, enable_contest=True, aggr0=0.5)
    assert PatchMosaicSim(cfg, 1).run()["events_hash"] != GOLD  # contest draws + transfers fire
