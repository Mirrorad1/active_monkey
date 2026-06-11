# Worldview Benchmark — Summary

Generated: 2026-06-11T13:51:35Z
World: B  Mechanism: decay  Convention: normalized
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
| B | decay | 42 | 0 | 1.0207 | 1.0172 | 0.0035 | 0 | 0 |
| B | decay | 42 | 1 | 1.0951 | 1.0576 | 0.0374 | 0 | 0 |

## Aggregate

- Mean plateau: 1.0579 nats
- Mean final surprise: 1.0374 nats
- Mean drop: 0.0205 nats
- Total alarm events: 0
- Total growth accepted: 0

## Grading

bars.json absent — awaiting predeclaration. No verdict issued.
