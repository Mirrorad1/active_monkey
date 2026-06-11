# Worldview Benchmark — Summary

Generated: 2026-06-11T06:03:02Z
World: D  Mechanism: grow  Convention: normalized
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
| D | grow | 17 | 0 | 0.6389 | 1.4675 | -0.8286 | 3 | 1 |
| D | grow | 17 | 1 | 0.6320 | 0.8932 | -0.2612 | 3 | 1 |

## Aggregate

- Mean plateau: 0.6355 nats
- Mean final surprise: 1.1804 nats
- Mean drop: -0.5449 nats
- Total alarm events: 6
- Total growth accepted: 2

## Grading

bars.json absent — awaiting predeclaration. No verdict issued.
