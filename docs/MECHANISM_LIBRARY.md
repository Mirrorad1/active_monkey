# MECHANISM_LIBRARY — the seeded reusable artifacts

This indexes the first coalescence artifacts distilled from committed repo evidence. Each is
schema-valid (built by `tools/seed_coalescence_cards.py` from the real dataclasses) and carries
an **honest status**. List them live:

```bash
uv run active-monkey coalesce mechanisms list
uv run active-monkey coalesce geometry list
uv run active-monkey coalesce validate --all
```

## Mechanisms (`mechanisms/`)

| id | type | status | source exps |
|---|---|---|---|
| `functional-valence-dyad-v0` | functional-valence-learning | **validated** (toy scale) | 215–222, 225 |
| `communication-scaffold-v0` | costed-signaling | **scaffold** (NOT validated) | — (none) |

- **functional-valence-dyad-v0** — the one validated mechanism with a runnable checkpoint
  (`artifacts/active-monkey-affect-dyad-v0`). A symbolic dyad learns which response earns
  approval per inferred intent-like state, judged by a frozen constant-unfakeable scorer.
  Ships an `AdapterCard` (`adapters.json`, belief → active-sensing — a composition *hypothesis*,
  not a validated bridge) and a `ScorerCard` (`scorer_refs.json`, the sha256-pinned affect
  scorer). Functional valence only — not subjective feeling.
- **communication-scaffold-v0** — explicitly speculative. The `comm_v0` benchmark is an
  existence test (costed signaling beats shuffled/muted, ~1.9 bits MI); no selection-pressure
  or emergence result exists. `source_experiments` is honestly empty. Emergent compositional
  grammar remains the documented open problem.

## Geometry maps (`geometry_maps/`)

| id | mechanism | what it maps |
|---|---|---|
| `dyad-session-length-curve-v0` | functional-valence-dyad-v0 | session length × precision schedule → genuine discrimination (Exp 218–221) |
| `active-sensing-benefit-wall-v0` | active-sensing-probe | probe rate × cost → flat selection slope; no valley (Exp 210–212) |
| `costly-sensing-wall-v0` | costed-sensing-organ | seven ecology levers → no functional organ (Exp 199–205) |

## Boundary notes (`boundary_notes/`)

| id | failed mechanism | the constraint |
|---|---|---|
| `active-sensing-benefit-wall-v0` | costed active information-gathering | useful-when-gifted ≠ locally evolvable; benefit magnitude is the wall (Exp 210–213) |
| `costly-sensing-wall-v0` | evolution of a costed sensory organ | no costed sense becomes functional at this substrate; the fitness valley is the barrier (Exp 199–207) |
| `hidden-state-memory-boundary-v0` | passive hidden-state memory as a locally evolvable trait | the local-gradient wall generalises from senses to memory/inference (Exp 208–209) |

## Sample experiment bundles (`experiment_bundles/`)

Three honest bundles demonstrate the backfill spectrum (each references original committed
files in place — never copies or mutates raw data):

| bundle | level | evidence |
|---|---|---|
| `exp222` | checkpoint_bundle | runnable checkpoint artifact + frozen scorer card |
| `exp199` | trajectory_bundle | per-seed trajectory JSON (referenced in place) |
| `exp210` | repro_bundle | committed deterministic script; **no** raw-trajectory claim |

## Reading the statuses honestly

`validated` means: controlled by a frozen, constant-unfakeable scorer, with the
provided-vs-learned split named, at the stated (toy) scale. `constrained` / boundary means: a
negative result preserved as a reusable constraint — which regions not to re-enter, and why.
`scaffold` / `speculative` means: machinery exists but the claim is not demonstrated. The
library is only useful if these stay honest; `tests/test_coalescence_library.py` guards them.
