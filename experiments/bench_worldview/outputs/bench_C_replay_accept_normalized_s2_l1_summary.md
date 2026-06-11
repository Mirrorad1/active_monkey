# Worldview Benchmark — Summary

Generated: 2026-06-11T13:52:47Z
World: C  Mechanism: replay_accept  Convention: normalized
Phase1: 600 steps  Phase2: 1400 steps

## SMOKE RUN

**This is NOT a scientific run.** Step counts chosen as the smallest T that still
lets the key world-specific phenomena show:
- Phase1 T=600: enough for place-belief convergence (A/B/C) and
  noise-floor stabilisation (B requires ~50 obs/color at 0.7 noise to plateau).
- Phase2 T=1400: enough for the alarm to fire in C (aliased world needs
  ~50 per-color obs + 200-step check interval;
  1400 steps gives ~350 per-color obs in a 4-color world).
  World D (nonstationary) remap fires at mid-phase2.

Smoke T_phase1=600, T_phase2=1400 chosen empirically.

## Results Table

| world | mechanism | layout_seed | seed | plateau | final_surprise | drop | alarm_events | growth_accepted |
|-------|-----------|-------------|------|---------|----------------|------|--------------|-----------------|
| C | replay_accept | 7 | 0 | 1.2104 | 0.3509 | 0.8595 | 3 | 2 |
| C | replay_accept | 7 | 1 | 1.2247 | 0.0049 | 1.2199 | 4 | 3 |

## Aggregate

- Mean plateau: 1.2176 nats
- Mean final surprise: 0.1779 nats
- Mean drop: 1.0397 nats
- Total alarm events: 7
- Total growth accepted: 5

## Grading

bars.json absent — awaiting predeclaration. No verdict issued.
