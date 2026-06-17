"""Host-robust drop-in for ``eval.affect_score.score_affect`` (NOT frozen — a wrapper).

Two independent problems make the frozen scorer impractical on a constrained host:

1. **XLA JIT exhaustion (the binding one).** The frozen agent anneals a *static* gamma
   over the session, so ``infer_policies`` recompiles every turn (~300 distinct XLA
   executables per seed).  Those compiled dylibs accumulate until the CPU backend fails
   to materialize new symbols (``JaxRuntimeError: Failed to materialize symbols`` /
   ``xla_jit_dylib_N``) — a full 8-seed score crashes mid-run here, sequential OR parallel.
   The fix (proven in experiments/exp226): call ``jax.clear_caches()`` between independent
   seeds so the executable cache is freed and the dylib count stays bounded.  It frees only
   compiled executables — the numbers are unchanged.

2. **Wall-clock.** Each seed pays the full per-turn recompilation (~160 s/seed here), so a
   sequential cache-cleared score is ~20 min.  Running a *few* seeds at a time across
   processes (each worker still clearing between its own seeds, so per-worker memory stays
   bounded) recovers a ~2x wall-clock win without reintroducing the JIT exhaustion.

Result equals ``score_affect`` **bit-for-bit** — it reuses the FROZEN ``_run_session`` per
seed and the frozen aggregation constants; only executable-cache freeing and the seed loop's
scheduling differ.  ``tests/test_affect_score_fast.py`` pins exact equality.  Default worker
count is deliberately small (memory-safe: each seed peaks ~2.6 GB during compilation).

Functional valence only — no sentience claim.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np

# Each compiling seed peaks ~2.6 GB here; keep concurrency low so N workers fit in RAM.
_DEFAULT_MAX_WORKERS = 2


def _run_one(seed: int, turns: int) -> dict:
    """Run ONE frozen per-seed session, then free the JIT executable cache.

    Lazy imports so loading this module is cheap and (under spawn) each worker imports the
    heavy JAX-pulling frozen module once.  jax.clear_caches() drops the compiled executables
    accumulated by the per-turn gamma recompilation — bounding memory and the XLA dylib count
    — without touching any value (bit-identity preserved, guarded by tests)."""
    import jax  # noqa: PLC0415
    from eval.affect_score import _direct_head_factory, _run_session  # noqa: PLC0415
    row = _run_session(_direct_head_factory, seed, turns)
    jax.clear_caches()
    return row


def _worker(args: tuple[int, int]) -> tuple[int, dict]:
    seed, turns = args
    return seed, _run_one(seed, turns)


def _limit_worker_threads() -> None:
    """Cap intra-op threading so workers don't oversubscribe cores (these models are tiny;
    the cost is dispatch/compilation, not matmul)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def score_affect_fast(seeds=None, turns: int | None = None,
                      max_workers: int | None = None):
    """Host-robust, bit-identical equivalent of ``eval.affect_score.score_affect``.

    Clears the JIT cache between seeds (so the full config COMPLETES on a constrained host)
    and optionally runs a few seeds at a time across processes for a wall-clock win.
    ``max_workers=1`` is the pure sequential cache-cleared path; ``None`` -> a memory-safe
    small default.  Returns the frozen ``AffectScoreReport``.
    """
    from eval.affect_score import (  # noqa: PLC0415
        CEIL, GENUINE_FLOOR, IMPROVEMENT_FLOOR, REALIZED_FLOOR, SEEDS_DEFAULT,
        TURNS_DEFAULT, AffectScoreReport,
    )

    seeds = tuple(SEEDS_DEFAULT if seeds is None else seeds)
    turns = TURNS_DEFAULT if turns is None else int(turns)
    if max_workers is None:
        max_workers = min(_DEFAULT_MAX_WORKERS, len(seeds), os.cpu_count() or 1)

    _limit_worker_threads()  # set in parent so spawn children inherit the caps

    if max_workers <= 1 or len(seeds) <= 1:
        # Pure sequential, cache-cleared between seeds (mirrors exp226 _score(cache_clear)).
        rows = {s: _run_one(s, turns) for s in seeds}
    else:
        # spawn (not fork): each worker gets a clean JAX/XLA backend (no fork-after-init
        # hazard) and clears its own cache between its seeds, so per-worker memory stays
        # bounded and the dylib count never exhausts.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=max_workers, initializer=_limit_worker_threads) as pool:
            rows = dict(pool.map(_worker, [(s, turns) for s in seeds]))

    firsts = [rows[s]["first"] for s in seeds]
    lasts = [rows[s]["last"] for s in seeds]
    csels = [rows[s]["csel"] for s in seeds]
    ask_rates = [rows[s]["ask_rate"] for s in seeds]
    genuine_flags = [bool(rows[s]["csel"] >= 0.5 and rows[s]["last"] > CEIL) for s in seeds]

    mean_first = float(np.mean(firsts))
    mean_last = float(np.mean(lasts))
    improvement = mean_last - mean_first
    genuine_fraction = float(np.mean(genuine_flags))
    ask_rate = float(np.mean(ask_rates))

    guardrails = {
        "realized_above_ceiling": mean_last > REALIZED_FLOOR,
        "learned_improvement": improvement >= IMPROVEMENT_FLOOR,
        "genuine_reliable": genuine_fraction >= GENUINE_FLOOR,
    }
    verdict = all(guardrails.values())

    return AffectScoreReport(
        metric=mean_last,
        mean_first=mean_first,
        mean_last=mean_last,
        improvement=improvement,
        genuine_fraction=genuine_fraction,
        ask_rate=ask_rate,
        n_seeds=len(seeds),
        guardrails=guardrails,
        verdict=verdict,
    )
