# Worldview Benchmark — Summary

Generated: 2026-06-11T06:02:46Z
World: A  Mechanism: none  Convention: normalized
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
| A | none | 42 | 0 | 0.5910 | 0.6469 | -0.0560 | 0 | 0 |
| A | none | 42 | 1 | 0.6973 | 0.5612 | 0.1361 | 0 | 0 |

## Aggregate

- Mean plateau: 0.6441 nats
- Mean final surprise: 0.6040 nats
- Mean drop: 0.0401 nats
- Total alarm events: 0
- Total growth accepted: 0

## Grading

bars.json absent — awaiting predeclaration. No verdict issued.
