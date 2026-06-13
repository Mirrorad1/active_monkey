# Ecology engine performance — profile, fixes, and limits

**Status:** analysis + optimizations (branch `perf/ecology-hotpath`, 2026-06-13). Prompted by
experiment runtimes growing from minutes to ~30–120 min, with one runaway-growth case.

## 1. The runtime model

A run costs, to first order:

    T ≈ c · Σ_t population(t) · (arms × seeds)

where `c` ≈ **4–5 µs per creature-step** of pure-Python, GIL-bound work (sense → choose action →
move → eat → metabolise → maybe reproduce → maybe die). Measured: a 12×12, horizon-12000 run at a
~630 standing population is **~38 s** (≈ 7.2M creature-steps). The cost is **distributed** across
~10 small operations per creature-step — there is no single dominant hot spot to cut.

Three things made experiments slow, in order of impact:

1. **Σ population(t) grew.** Early experiments equilibrated at ~150 creatures on short horizons;
   later regimes support **1500–2000** creatures on **12000**-step horizons. Both factors multiply:
   total creature-steps rose ~50–100×, so wall-clock did too. This is the "steady growth."
2. **Runaway growth.** A no-scarcity / no-cap regime grows geometrically toward the `max_population`
   guard (20000); a single such job is enormous. (The user's "one runaway".)
3. **Memory → swap (on smaller machines).** A single 12000-step run peaked at ~**815 MB** RSS;
   `min(16, cpu-2)` parallel workers × that ≈ **9 GB**. On a 16 GB machine (with an OS + browser +
   IDE) this **swaps**, and swap thrashing turns minutes into hours. (Not a factor on the 38 GB dev
   box, but the dominant failure mode on constrained machines.)

## 2. A profiling caveat that cost us a wrong turn

`cProfile`'s per-call instrumentation **massively over-weights functions called millions of times**
(`neighbors()`, `max()`, the policy key-lambda). Its `tottime` made them look like ~30% of the run;
precomputing/eliminating them gave **~0 real speedup** (8.36 s → 8.26 s, within noise). Lesson:
**confirm a profiler hot spot with a clean wall-clock A/B before optimizing it** — for million-call
cheap functions, use `time.perf_counter` A/Bs or a sampling profiler (`py-spy`), not `cProfile`
`tottime`.

## 3. What was implemented (all byte-identical — `events_hash` unchanged; the committed-hash
regression tests `fc19d23f…`, `502e0539…` still pass)

- **Free dead creatures' belief maps on death** (`HomeostaticPolicy.release_maps()`). A dead
  creature's two `n_cells` maps (`m`, `visit_t`) are never read again; freeing them stops the heap
  growing with total-ever-born. **RSS 815 → ~590 MB (−27%)** on the 12000-step run. The policy
  object + `band_estimate` (a float) are kept for post-hoc inspection.
- **Memory-aware worker cap** in the runtime pre-flight (`runtime_budget.preflight` →
  `recommended_workers`). It projects per-job peak RSS from the projected population and caps
  workers so `workers × per-job RSS ≤ 60% of physical RAM` — **preventing the swap that is the real
  multi-hour killer.** On a 16 GB box it caps 11 → 7 workers for a heavy batch; on 32 GB+ it leaves
  all workers. Experiments call it and pass `recommended_workers` to the batch.
- **Runaway guard** (L25, prior turn): the pre-flight blocks (`require_safe=True`) a batch whose
  population projects toward the cap — catching the explosion case before it burns hours.
- **Micro-cleanups** (byte-identical, ~0 measured speedup but correct hygiene): a precomputed static
  neighbor table; `_alive()` returns the already-sorted alive-list without a per-step `sorted(...)`.

Net effect: the **runaway** and **swap** failure modes (the two that produce *hours*) are fixed; the
baseline CPU cost on a memory-unconstrained machine is ~unchanged (see §5 for why).

## 4. The experiment-design levers (the real CPU control — quantified)

Since `T ∝ Σ population(t) × arms × seeds`, the largest safe wins are at the design level:

| lever | measured effect |
|---|---|
| **Horizon** | ~linear: 3000 → 7 s, 6000 → 17 s, 12000 → 38 s (12×12). Halve the horizon where the metric has plateaued (the L23 plateau check) → ~half the time. |
| **Grid size → population** | dominant: 8×8 (pop ~234) → 3.7 s, 12×12 (pop ~609) → 10.9 s, 16×16 (pop ~1186) → 21.6 s (horizon 4000). Use a **smaller grid for pilots/diagnostics** → ~3× faster than 12×12. |
| **Seeds × arms** | linear. Reserve the full seed set (≥8) for the *primary verdict* arm; run diagnostic/confirmatory arms at 3–4 seeds. |
| **Workers** | the pre-flight now picks the memory-safe maximum automatically. |

Always run the pre-flight (`runtime_budget.preflight(..., require_safe=True)`) at the top of an
experiment's `main()` — it catches runaways, projects wall-clock + memory, and right-sizes workers.

## 5. Why the per-creature CPU cost is largely irreducible (the limits)

The cost is pure-Python per-creature work, and the engine's **bit-exact determinism contract** (the
`events_hash` fingerprints that every experiment's reproducibility + regression guards depend on)
blocks the usual speedups:

- **Vectorising across creatures** (numpy batch sense/act/eat) would need the same rng draws in the
  same order AND the **sequential `consume()` depletion race** (creature A eats, changing the cell
  before creature B) — which is inherently serial. A vectorised engine would change every hash
  (new baseline) and is a major, risky rewrite. Potential 5–20×; recommended only as a scoped
  project, not a quick win.
- **Batching the rng draws** (one buffer per step instead of per-creature calls) reorders the rng
  stream → changes every hash. Blocked by determinism.
- **GIL**: the per-creature loop is single-threaded Python; threads don't parallelise it (we already
  use processes across the *outer* seed/arm axis).
- **Micro-opts** (precompute neighbors, list-vs-numpy scalar access ~2× per access) net to single
  digits because the work is spread thin across ~10 ops/creature-step — confirmed by A/B.

The honest summary: on a memory-unconstrained machine, the per-step CPU is ~irreducible without a
determinism-breaking rewrite (vectorisation) or a different runtime (PyPy could JIT the pure-Python
loop for a possible 5–10×, at the cost of dependency/compat work). The high-leverage, safe wins are
the **runaway guard**, the **swap-avoiding worker cap**, and **right-sizing horizon/grid/seeds**.
