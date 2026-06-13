"""ecology/runtime.py — fork / snapshot / restore / replay / distill as FIRST-CLASS primitives.

The paper's "lossless replication, pausing/resuming, memory-state copying, replaying digital
experience" advantages, made concrete for the ecology substrate. A running Ecology is fully
determined by (cfg, seed) PLUS its mutable state (creatures + world + the live rng cursor), so its
entire developmental state is a value you can snapshot, restore, fork, replay, and distill from.

Primitives
----------
- ``snapshot(eco) -> Snapshot``  : capture the FULL live state (creatures, world, rng cursor, t, events).
- ``restore(snap) -> Ecology``   : rebuild a running Ecology from a snapshot. Continuing a restored run
  is BIT-IDENTICAL to never having paused (the losslessness invariant, asserted in
  tests/test_ecology_runtime.py). This is pause/resume + memory-state copying.
- ``fork(eco, seed=...) -> Ecology`` : a counterfactual twin sharing all history up to the fork point;
  re-seeding (or changing the env) makes the post-fork divergence causally attributable to the change,
  not prior history — the scientific control. Same role as creature.Creature.fork() but for a whole
  population. seed=None forks a faithful (identical-future) twin.
- ``replay(snap, until, expect_hash=...) -> Ecology`` : deterministic replay with an optional bit-match
  gate against a committed events_hash (the L14 replay-with-bit-match pattern, generalised).
- ``distill(ecos, strategy=...) -> Genotype`` : distil a FOUNDER genotype from SUCCESSFUL runs (the
  surviving / most-reproductive lineages), so a new run can be seeded from distilled success — the
  "replay_or_distill(successful trajectories)" step. Returns a valid Genotype.
- ``run_to`` / ``fork_run_compare`` : the fork -> run -> log -> compare-against-unforked-baseline pipeline.

Relationship to creature/ (nira): the persistent-creature spine already has Creature.fork() (full
deepcopy + lineage) and growth.replay_nll() (a replay buffer). This module provides the SAME family of
primitives for the population/ecology substrate, where losslessness is provable bit-for-bit. A single
unified cross-substrate AgentState is the documented next step (see ecology/README or RESUME).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Any, Callable

import numpy as np

from ecology.engine import Ecology, EcologyConfig
from ecology.creature import Creature
from ecology.genotype import Genotype, clamp_traits, is_valid, INT_TRAITS


# ---------------------------------------------------------------------------
# Snapshot — a lossless, picklable value capturing a full Ecology run state.
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    """The complete live state of an Ecology run — enough to restore bit-identically.

    Picklable (deepcopies of plain objects + numpy arrays + the rng cursor dict), so a developmental
    trajectory can be paused to disk and resumed later. ``state_hash`` fingerprints the captured state.
    """
    cfg: EcologyConfig
    seed: int
    t: int
    next_id: int
    creatures: list[Creature]      # ALL ever-born (dead included), for the newborn/lineage summary
    alive_list: list[Creature]     # the live cohort (object-identical to a subset of `creatures`)
    world: Any                     # a GridWorld (deepcopied; holds the resource array + scalars)
    rng_state: dict                # eco.rng.bit_generator.state — the LIVE rng cursor
    events: list[dict[str, Any]]
    exploded: bool
    strip_log: list[dict[str, Any]]
    state_hash: str = ""


def snapshot(eco: Ecology) -> Snapshot:
    """Capture the full live state of ``eco`` as a lossless, picklable Snapshot.

    creatures + alive_list are deepcopied TOGETHER so the object-aliasing (alive_list elements ARE
    members of creatures) is preserved. The rng cursor is captured via bit_generator.state.
    """
    creatures, alive_list = copy.deepcopy((eco._creatures, eco._alive_list))
    snap = Snapshot(
        cfg=eco.cfg,
        seed=eco.seed,
        t=eco.t,
        next_id=eco.next_id,
        creatures=creatures,
        alive_list=alive_list,
        world=copy.deepcopy(eco.world),
        rng_state=copy.deepcopy(eco.rng.bit_generator.state),
        events=copy.deepcopy(eco.events),
        exploded=eco.exploded,
        strip_log=copy.deepcopy(eco.strip_log),
    )
    snap.state_hash = _snapshot_hash(snap)
    return snap


def restore(snap: Snapshot) -> Ecology:
    """Rebuild a running Ecology from a snapshot. Continuing it is BIT-IDENTICAL to never pausing.

    Constructs a fresh Ecology(cfg, seed) (cheap throwaway founder placement) and overwrites every
    piece of mutable state from the snapshot, including the live rng cursor.
    """
    eco = Ecology(snap.cfg, seed=snap.seed)
    eco.t = snap.t
    eco.next_id = snap.next_id
    eco._creatures, eco._alive_list = copy.deepcopy((snap.creatures, snap.alive_list))
    eco.world = copy.deepcopy(snap.world)
    eco.events = copy.deepcopy(snap.events)
    eco.exploded = snap.exploded
    eco.strip_log = copy.deepcopy(snap.strip_log)
    eco.rng.bit_generator.state = copy.deepcopy(snap.rng_state)
    return eco


# ---------------------------------------------------------------------------
# fork — a counterfactual twin (the scientific control primitive).
# ---------------------------------------------------------------------------
def fork(eco: Ecology, *, seed: int | None = None) -> Ecology:
    """Branch a counterfactual twin sharing ALL history up to now.

    seed=None  -> a faithful twin (identical future if stepped the same way).
    seed=<int> -> re-seed the live rng so the post-fork future DIVERGES; any difference from the
                  unforked baseline is then causally attributable to the new seed/environment, not
                  prior history. (Mutate the returned twin's cfg/world to change its environment.)
    """
    twin = restore(snapshot(eco))
    if seed is not None:
        twin.rng = np.random.default_rng(seed)
    return twin


# ---------------------------------------------------------------------------
# replay — deterministic re-run with an optional bit-match gate.
# ---------------------------------------------------------------------------
def replay(snap: Snapshot, until: int, *, expect_hash: str | None = None) -> Ecology:
    """Replay forward from a snapshot to step ``until``. If expect_hash is given, ASSERT the resulting
    events_hash matches it (a committed-row bit-match gate — an unmatched replay is a different run)."""
    eco = restore(snap)
    while eco.t < until and not eco.exploded and eco._alive_list:
        eco.step()
    if expect_hash is not None and eco.events_hash() != expect_hash:
        raise AssertionError(
            f"replay diverged from committed hash (got {eco.events_hash()[:12]}, want {expect_hash[:12]})"
        )
    return eco


# ---------------------------------------------------------------------------
# distill — a founder genotype from SUCCESSFUL trajectories (the missing piece).
# ---------------------------------------------------------------------------
def distill(ecos: list[Ecology], strategy: str = "survivor_mean") -> Genotype:
    """Distil a FOUNDER genotype from the successful lineages of completed runs.

    strategy:
      'survivor_mean'  -> the (clamped) MEAN genotype over all SURVIVING creatures across the runs
                          (the gene pool that the environment kept).
      'top_reproducer' -> the genotype of the single creature (alive or dead) with the most offspring
                          (the most reproductively successful lineage).
    Seed a new run with the returned genotype to "replay distilled success" — bootstrap from what worked.
    """
    if strategy == "survivor_mean":
        genos = [c.genotype for eco in ecos for c in eco._alive_list]
        if not genos:
            raise ValueError("distill('survivor_mean'): no survivors across the given runs")
        keys = list(asdict(genos[0]).keys())
        blended = {k: float(np.mean([getattr(g, k) for g in genos])) for k in keys}
        result = Genotype(**clamp_traits(blended))
    elif strategy == "top_reproducer":
        best: Creature | None = None
        for eco in ecos:
            for c in eco._creatures:
                if best is None or c.phenotype.offspring_count > best.phenotype.offspring_count:
                    best = c
        if best is None:
            raise ValueError("distill('top_reproducer'): no creatures across the given runs")
        result = best.genotype
    else:
        raise ValueError(f"unknown distill strategy: {strategy!r}")
    assert is_valid(result), f"distill produced an invalid genotype: {result}"
    return result


# ---------------------------------------------------------------------------
# The pipeline: fork -> run -> log -> compare-against-unforked-baseline.
# ---------------------------------------------------------------------------
def run_to(start: Ecology | tuple[EcologyConfig, int], until: int) -> Ecology:
    """Run an Ecology (or build one from (cfg, seed)) forward to step ``until``; return it."""
    eco = start if isinstance(start, Ecology) else Ecology(start[0], seed=start[1])
    while eco.t < until and not eco.exploded and eco._alive_list:
        eco.step()
    return eco


def _summarize(eco: Ecology) -> dict[str, Any]:
    alive = eco._alive_list
    return {
        "t": eco.t,
        "alive": len(alive),
        "total_ever": len(eco._creatures),
        "mean_intensity": float(np.mean([c.genotype.thermosense_intensity for c in alive])) if alive else float("nan"),
        "events_hash": eco.events_hash(),
        "exploded": eco.exploded,
    }


def fork_run_compare(
    base: Ecology,
    until: int,
    *,
    treatment_seed: int,
    treatment_env: Callable[[Ecology], None] | None = None,
    baseline_seed: int | None = None,
) -> dict[str, Any]:
    """Fork ``base`` into a TREATMENT (re-seeded, optionally re-environed) and a BASELINE twin, run both
    to ``until``, and return both summaries + their divergence — the counterfactual experiment in one call.

    treatment_env(eco) may mutate the treatment twin's world/cfg in place before it runs (e.g. drop it
    into a different environment). The baseline shares the same fork point with no intervention.
    """
    treat = fork(base, seed=treatment_seed)
    if treatment_env is not None:
        treatment_env(treat)
    baseline = fork(base, seed=baseline_seed)
    treat = run_to(treat, until)
    baseline = run_to(baseline, until)
    ts, bs = _summarize(treat), _summarize(baseline)
    return {
        "treatment": ts,
        "baseline": bs,
        "diverged": ts["events_hash"] != bs["events_hash"],
        "delta_mean_intensity": (ts["mean_intensity"] - bs["mean_intensity"]),
        "fork_t": base.t,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _snapshot_hash(snap: Snapshot) -> str:
    """A determinism fingerprint of the captured state (creatures by id + world resource + rng cursor)."""
    payload = {
        "t": snap.t,
        "next_id": snap.next_id,
        "alive_ids": sorted(c.creature_id for c in snap.alive_list),
        "resource": [round(float(x), 6) for x in np.asarray(snap.world.resource).tolist()],
        "rng": json.dumps(snap.rng_state, sort_keys=True, default=str),
        "events_len": len(snap.events),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
