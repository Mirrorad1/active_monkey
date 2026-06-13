"""ecology/batch.py — parallel batch runner for INDEPENDENT ecology simulations.

Each run is a fully independent ``Ecology(cfg, seed)``: its RNG is derived only
from ``seed`` (see Ecology.__init__) and no run ever reads another run's state.
So a batch of runs is embarrassingly parallel, and — crucially — running them
concurrently changes NOTHING about any result: each ``events_hash`` depends only
on ``(cfg, seed)``, never on execution order or concurrency.  Parallelism here is
a pure wall-clock win with bit-identical outputs (asserted in
tests/test_ecology_batch.py: parallel == sequential).

The atomic unit of parallelism is ONE full run; the per-step time loop INSIDE a
run is inherently sequential and is not parallelised.

SEQUENTIAL-DEPENDENCY ESCAPE HATCH: if a future design seeds run *k* from run
*k-1*'s end-state (true sequential dependence), those jobs must NOT share a batch
— keep them in a sequential chain and only parallelise across the independent
outer seeds.  ``run_batch(..., sequential=True)`` forces single-process execution
for debugging or such chains.

Why processes (not threads): the per-creature step loop is pure-Python and
GIL-bound, so threads do not parallelise it; ProcessPoolExecutor (spawn on macOS)
does.  ``run_one``/``RunSpec`` live in this importable module precisely so the
spawn workers can re-import them (a function defined in a ``__main__`` script is
not reliably picklable under spawn).
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Hashable

import numpy as np

from ecology.engine import Ecology, EcologyConfig


def default_workers() -> int:
    """Leave a couple of cores free; cap to avoid oversubscription on big boxes."""
    n = os.cpu_count() or 4
    return max(1, min(n - 1, 16))


@dataclass
class RunSpec:
    """One independent run + how to summarise it.  Pure-data and PICKLABLE
    (scalars + an EcologyConfig + a frozen Genotype founder), so it ships cleanly
    to spawn workers.

    key: a hashable label returned with the result (e.g. (arm, seed)).
    trait_means: genotype traits to report the NEWBORN-window mean for
      (first entry is the PRIMARY metric used by the L21 validity gate).
    max_trait: genotype trait to report the max over all-ever-born creatures for.
    """
    key: Hashable
    cfg: EcologyConfig
    seed: int
    window_start: int
    checkpoint_stride: int
    trait_means: tuple[str, ...] = ("thermosense_intensity",)
    max_trait: str = "thermosense_intensity"
    min_valid_pop: int = 10


def _newborn_mean(creatures: list, lo: int, hi: int, attr: str) -> float:
    """Mean of genotype.<attr> over NON-founder creatures with birth_t in [lo, hi]; NaN if none."""
    nb = [c for c in creatures if c.parent_id is not None and lo <= c.phenotype.birth_t <= hi]
    if not nb:
        return float("nan")
    return float(np.mean([getattr(c.genotype, attr) for c in nb]))


def run_one(spec: RunSpec) -> dict[str, Any]:
    """Execute ONE independent run; return its summary + checkpoint trajectory.

    Top-level + picklable so ProcessPoolExecutor (spawn) can ship it to a worker.
    Bit-identical to calling Ecology(spec.cfg, spec.seed) and stepping to horizon
    in the caller's process — determinism is per-seed, order-independent.
    """
    cfg = spec.cfg
    horizon = cfg.horizon
    eco = Ecology(cfg, seed=spec.seed)
    checkpoints = set(range(spec.checkpoint_stride, horizon + 1, spec.checkpoint_stride))
    trajectory: list[dict[str, Any]] = []

    while eco.t < horizon and not eco.exploded:
        eco.step()
        if eco.t in checkpoints:
            alive = eco._alive()
            nb = _newborn_mean(eco._creatures, eco.t - spec.checkpoint_stride, eco.t, spec.max_trait)
            mx = max((getattr(c.genotype, spec.max_trait) for c in alive), default=float("nan"))
            trajectory.append({
                "t": eco.t, "pop": len(alive),
                "newborn_mean_intensity": float("nan") if math.isnan(nb) else nb,
                "max_intensity": float("nan") if math.isnan(mx) else mx,
            })
        if not eco._alive():
            break

    alive = eco._alive()
    end_means = {a: _newborn_mean(eco._creatures, spec.window_start, horizon, a) for a in spec.trait_means}
    all_vals = [getattr(c.genotype, spec.max_trait) for c in eco._creatures]
    max_ever = max(all_vals) if all_vals else float("nan")
    primary = spec.trait_means[0]
    valid = (eco.t >= horizon and not eco.exploded
             and len(alive) >= spec.min_valid_pop
             and not math.isnan(end_means[primary]))

    out = {
        "key": spec.key, "seed": spec.seed, "steps_run": eco.t,
        "reached_horizon": eco.t >= horizon and not eco.exploded,
        "extinct": len(alive) == 0, "exploded": eco.exploded, "final_pop": len(alive),
        "end_means": end_means, "max_ever": max_ever, "valid": valid,
        "extinction_step": eco.t if len(alive) == 0 else None,
        "events_hash": eco.events_hash(), "trajectory": trajectory,
    }
    # Exp 202: band-strip validity summary (only present when cfg.track_band_strip populated it).
    if eco.strip_log:
        late = [s["strip"] for s in eco.strip_log if s["t"] >= horizon - 1000]
        occ = [s["occupants"] for s in eco.strip_log if s["t"] >= horizon - 1000]
        out["strip_late_mean"] = float(np.mean(late)) if late else 0.0
        out["strip_late_frac_pos"] = float(np.mean([x > 0 for x in late])) if late else 0.0
        out["occ_late_mean"] = float(np.mean(occ)) if occ else 0.0
    return out


def run_batch(specs: list[RunSpec], *, max_workers: int | None = None,
              sequential: bool = False) -> dict[Hashable, dict[str, Any]]:
    """Run all specs and return {key: result}.

    Order-independent and deterministic: bit-identical to running the specs one by
    one (each result depends only on its own (cfg, seed)).  Parallel across
    processes by default; ``sequential=True`` (or a single spec) runs in-process.
    """
    if sequential or len(specs) <= 1:
        return {s.key: run_one(s) for s in specs}
    results: dict[Hashable, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers or default_workers()) as ex:
        futs = {ex.submit(run_one, s): s.key for s in specs}
        for fut in as_completed(futs):
            r = fut.result()
            results[r["key"]] = r
    return results
