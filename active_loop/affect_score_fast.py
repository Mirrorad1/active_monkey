"""Parallel drop-in for ``eval.affect_score.score_affect`` (NOT frozen — a wrapper).

The frozen scorer runs the 8-seed ensemble SEQUENTIALLY; each seed is a 300-turn
closed-loop session whose per-turn cost is dominated by XLA dispatch overhead
(~0.7 s/turn here), so a full score is ~28 min and the autopilot — which scores
once per candidate — is impractical in-container.

This wrapper changes ONE thing: it runs the per-seed sessions across a process pool
instead of a Python ``for`` loop.  It calls the FROZEN ``eval.affect_score._run_session``
verbatim for each seed and aggregates with the FROZEN constants and the same arithmetic
as ``score_affect``, so the result is **bit-identical by construction** — the only
difference is wall-clock.  ``tests/test_affect_score_fast.py`` asserts exact equality
field-by-field against the frozen scorer.

Seeds are independent, so this is embarrassingly parallel; on an N-core host the speedup
is ~min(n_seeds, N).  Functional valence only — no sentience claim.
"""
from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np


def _worker(args: tuple[int, int]) -> tuple[int, dict]:
    """Run ONE frozen per-seed session in this worker process.

    Imported lazily so the (heavy, JAX-pulling) frozen module is loaded once per worker
    rather than in the parent before forking/spawning.  Returns (seed, row) where row is
    exactly what eval.affect_score._run_session returns.
    """
    seed, turns = args
    from eval.affect_score import _direct_head_factory, _run_session  # noqa: PLC0415
    return seed, _run_session(_direct_head_factory, seed, turns)


def _limit_worker_threads() -> None:
    """Cap intra-op threading so N workers don't oversubscribe the cores (these models are
    tiny; the cost is dispatch, not matmul, so single-threaded workers are fastest)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "XLA_FLAGS"):
        if var == "XLA_FLAGS":
            os.environ[var] = (os.environ.get(var, "") +
                               " --xla_cpu_multi_thread_eigen=false"
                               " intra_op_parallelism_threads=1").strip()
        else:
            os.environ.setdefault(var, "1")


def score_affect_fast(seeds=None, turns: int | None = None, max_workers: int | None = None):
    """Parallel, bit-identical equivalent of ``eval.affect_score.score_affect``.

    Returns the same frozen ``AffectScoreReport``.  ``seeds``/``turns`` default to the
    frozen defaults.  ``max_workers`` defaults to ``min(n_seeds, cpu_count)``.
    """
    # Imported here (not at module top) so importing THIS module is cheap; these pull JAX.
    from eval.affect_score import (  # noqa: PLC0415
        CEIL, GENUINE_FLOOR, IMPROVEMENT_FLOOR, REALIZED_FLOOR, SEEDS_DEFAULT,
        TURNS_DEFAULT, AffectScoreReport,
    )

    seeds = tuple(SEEDS_DEFAULT if seeds is None else seeds)
    turns = TURNS_DEFAULT if turns is None else int(turns)
    if max_workers is None:
        max_workers = min(len(seeds), os.cpu_count() or 1)

    _limit_worker_threads()  # set in parent so spawn children inherit the caps

    # spawn (not fork): each worker imports JAX cleanly with the thread caps in effect,
    # avoiding fork-after-XLA-init hazards.
    ctx = mp.get_context("spawn")
    if max_workers <= 1 or len(seeds) <= 1:
        rows = dict(_worker((s, turns)) for s in seeds)
    else:
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
