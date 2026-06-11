# Worldview Benchmark — Summary

Generated: 2026-06-11T06:03:01Z
World: C  Mechanism: grow  Convention: normalized
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
| C | grow | 7 | 0 | 1.2104 | 0.2679 | 0.9425 | 2 | 2 |
| C | grow | 7 | 1 | 1.2247 | 0.2436 | 0.9812 | 2 | 2 |

## Aggregate

- Mean plateau: 1.2176 nats
- Mean final surprise: 0.2557 nats
- Mean drop: 0.9618 nats
- Total alarm events: 4
- Total growth accepted: 4

## Grading

bars.json absent — awaiting predeclaration. No verdict issued.
