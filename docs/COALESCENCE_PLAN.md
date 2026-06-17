# COALESCENCE_PLAN — from experiments to a reusable artifact economy

## Why this exists

active_monkey has run 225+ honest experiments across a dozen directions — sense-organ /
costly sensing, hidden-state memory, active sensing, the affective dyad, belief-like state,
social emergence, transfer, population ecology, evolvability geometry, and the N-order
self-modeling lines. Each experiment reliably produces a *verdict*. What it has not reliably
produced is a **reusable artifact**: a citable, composable object the next experiment can
import. The result is drift — energy spread across directions, findings re-derived, negative
results remembered only as prose.

This plan moves active_monkey from **experiment-by-experiment exploration** to **cumulative
mechanism accumulation**. The unit of progress stops being "another verdict" and becomes
"another artifact in the library."

## The chain

```
experiments  →  evidence  →  bundles  →  mechanism cards  →  geometry maps
                                   ↘  checkpoints  ↘  scorers  ↘  adapters  ↘  boundary notes
```

- **Experiments produce evidence** — scripts, outputs, metrics, sometimes trajectories.
- **Bundles preserve evidence** — an `ExperimentBundle` is an immutable, provenance-stamped
  record that references (never mutates) the raw files and pins the repo commit + scorer hash.
- **Mechanism cards distill reusable principles** — a `MechanismCard` states the claim, the
  `works_when` / `fails_when` conditions, the falsifiers, and the provided-vs-learned ledger.
- **Geometry maps preserve works/fails boundaries** — many active_monkey findings are
  "worked/failed only after trying." The reusable artifact is then a `GeometryMap` or a
  `BoundaryNote`, not a positive result. A negative result is a *constraint*, and constraints
  compose.
- **Checkpoints preserve runnable agents** — an `AgentCheckpoint` (already implemented, see
  `docs/ARTIFACTS.md`) is a copyable agent: numeric state + frozen scorer + provenance.
- **Adapters compose mechanisms** — an `AdapterCard` documents how one mechanism's output
  becomes another's input (e.g. a dyad's belief-like state feeding an active-sensing probe).

## The common spine

Every direction is a segment of one spine. Artifacts make the segments composable:

```
sensing → hidden-state inference → belief-like state → action → feedback/selection → memory → communication → ecology re-entry
```

A `MechanismCard` declares where on this spine it lives and what it consumes/produces, so the
next experiment can ask: *what artifact can I import?* rather than starting from zero.

## What this layer answers

For any experiment, the artifact standard lets a future experiment recover:
1. **What world/equations were tested?** → `ExperimentSpec`
2. **What data was produced?** → `TrajectoryData` / `metrics_refs`
3. **What state did the agent learn?** → `AgentCheckpoint`
4. **What scorer judged it?** → `ScorerCard` (sha256-pinned, frozen)
5. **What mechanism was validated / falsified / constrained?** → `MechanismCard`
6. **What parameter boundary was discovered?** → `GeometryMap` / `BoundaryNote`
7. **What can the next experiment import?** → `AdapterCard` + the bundle's refs

## Honesty is the load-bearing wall

The whole point collapses if a bundle lies. So:
- Bundles **reference** original committed paths; they never copy or mutate raw data.
- The inventory and exporter report the **highest honestly achievable** backfill level and
  **refuse to over-claim** (you cannot export a `trajectory_bundle` for an experiment that
  logged no trajectories).
- A re-run is a **new reproduction run**, never relabeled as original historical data.
- Frozen scorers are sha256-pinned; a scorer change is a new version, never a silent edit.
- Conservative language only (functional valence, belief-like state, posterior over hidden
  state) — see `loop/VALIDATION.md`, which is binding.

## The path forward

The near-term composition target, grounded in the current frontier (RESUME.md):

```
affective dyad (validated, toy scale)  →  BeliefBench  →  active sensing  →  communication  →  ecology re-entry
       Exp 214–225                          benchmark        Exp 210–211 (wall)   social-emergence    Exp 194–207
   functional-valence-dyad-v0         active_loop/benchmarks   active-sensing-      (scaffold only)    costed-sensing-wall
   (mechanism + checkpoint)                                    benefit-wall (boundary)
```

The affective dyad is the one **validated** mechanism with a runnable checkpoint. Active
sensing and costed sensing are **boundaries** — reusable constraints that tell the next
experiment which regions not to re-enter. Communication remains a **scaffold**: the comm_v0
benchmark exists as an existence test, but no selection-pressure result does — it is declared
speculative, not validated.

See `docs/ARTIFACT_SPEC.md` for the schema, `docs/EXPERIMENT_BACKFILL.md` for the current
inventory and rerun plan, and `docs/MECHANISM_LIBRARY.md` for the seeded cards.
