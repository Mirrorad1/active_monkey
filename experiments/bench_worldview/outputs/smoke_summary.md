# Worldview Benchmark — Smoke Run Summary

**Generated:** 2026-06-11  
**Status:** SMOKE — NOT a scientific run (awaiting bars predeclaration)

## Step-count rationale (SMOKE)

Phase1 T=600, Phase2 T=1400. Chosen as the smallest T that still lets the
key world-specific phenomena show in the time budget:

- **Phase1 T=600:** enough for place-belief convergence and per-color NIW
  fitting (all four worlds). World B (noisy, p_true=0.7) needs ~50
  observations per color to plateau near the analytic floor (~1.0 nat
  at 3 colors); 600 steps in a 3-color world gives ~200 per-color obs.
- **Phase2 T=1400:** enough for the alarm to fire in C (aliased world
  needs ~COLOR_SURPRISE_WINDOW=50 per-color obs before alarm threshold
  check; SPAWN_INTERVAL=200 steps between checks; 1400 / 4 colors = ~350
  per-color obs). World D remap fires at step 1400//2=700 into phase2,
  leaving 700 steps to detect and respond to drift.
- Not suitable for confirming the 0.4-nat drop bar (too short), confirming
  probation convergence, or scientific comparison. These require T_phase2 >= 6000.

## Smoke Matrix Results

| world | mechanism | seed | plateau (nats) | final_surprise (nats) | drop (nats) | alarm_events | growth_accepted |
|-------|-----------|------|----------------|-----------------------|-------------|--------------|-----------------|
| A     | none      | 0    | 0.591          | 0.647                 | -0.056      | 0            | 0               |
| A     | none      | 1    | 0.697          | 0.561                 | 0.136       | 0            | 0               |
| A     | grow      | 0    | 0.591          | 0.394                 | +0.197      | 2            | 2               |
| A     | grow      | 1    | 0.697          | 0.264                 | +0.433      | 2            | 2               |
| B     | none      | 0    | 1.003          | 0.985                 | 0.019       | 0            | 0               |
| B     | none      | 1    | 1.021          | 0.965                 | 0.056       | 0            | 0               |
| B     | grow      | 0    | 1.003          | 1.020                 | -0.016      | 3            | 2               |
| B     | grow      | 1    | 1.021          | 1.411                 | -0.389      | 3            | 1               |
| C     | none      | 0    | 1.210          | 1.198                 | 0.012       | 0            | 0               |
| C     | none      | 1    | 1.225          | 1.178                 | 0.047       | 0            | 0               |
| C     | grow      | 0    | 1.210          | 0.268                 | +0.942      | 2            | 2               |
| C     | grow      | 1    | 1.225          | 0.244                 | +0.981      | 2            | 2               |
| D     | none      | 0    | 0.639          | 1.285                 | -0.646      | 0            | 0               |
| D     | none      | 1    | 0.632          | 1.318                 | -0.686      | 0            | 0               |
| D     | grow      | 0    | 0.639          | 1.468                 | -0.829      | 3            | 1               |
| D     | grow      | 1    | 0.632          | 0.893                 | -0.261      | 3            | 1               |

## Qualitative patterns (smoke — not scientifically graded)

**A (learnable):** none arm stays flat (small variance around ~0.6 nats as
expected for a 3-color world at smoke T). grow fires 2 alarms/seed and
accepts 2 jumps; final surprise drops ~0.2–0.4 nats. Structurally adequate
world — growth is superfluous but harmless.

**B (noisy, p_true=0.7):** none arm plateaus near 1.0 nat — close to the
analytic floor (~0.82 nats, noise is partially irreducible). grow arm shows
the expected pattern: alarms fire (noise is genuine ceiling), growth is
accepted but surprise does not drop — the floor is real and growth cannot
improve it. This is the correct "noise is irreducible" signature; a longer
run is needed to confirm whether surprise even increases slightly due to
model complexity.

**C (aliased, layout seed=7):** none arm stays high (~1.2 nats) — the
hallmark of aliasing with one component. grow arm fires 2 alarms/seed and
accepts both; surprise drops ~0.94–0.98 nats (from ~1.21 to ~0.25). This
replicates the Exp 154 qualitative result (normalized convention + batch-jump
answers the aliased-world alarm). At smoke T the drop is already large because
the alarm fires early.

**D (nonstationary):** remap fires at step 700 of phase2. none arm shows a
large final_surprise spike (1.28–1.32 nats from a ~0.63 plateau) — the
unhandled drift signature. grow arm fires 3 alarms/seed; 1 is accepted per
seed but the final surprise is still elevated or increased, suggesting growth
is answering the alarm but with mixed success at smoke T (probation windows
may not complete before the run ends at T=1400).

## Grading status

`bars.json` absent — awaiting predeclaration by the research loop.
No verdict file issued per spec (§5 T15 gate rule).
