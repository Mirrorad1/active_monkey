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
| `recipe-symmetry-breaking-v0` | hidden-state-belief | **validated** (toy scale) | 26–35 |
| `functional-valence-dyad-v0` | functional-valence-learning | **validated** (toy scale) | 215–222, 225 |
| `meta-calibration-n3-v0` | identity-self-modeling | **validated** (toy richness) | 155–168 |
| `online-structure-growth-v0` | hidden-state-belief | **validated** (toy scale) | 152–154 |
| `identity-n4-monitor-v0` | identity-self-modeling | **constrained** (detection only) | 176–180 |
| `communication-scaffold-v0` | costed-signaling | **scaffold** (NOT validated) | — (none) |

- **recipe-symmetry-breaking-v0** — the project's flagship durable finding as a card: the
  RECIPE (embodiment + grounding + continuous registered experience + **one** innate anchor +
  taught labels) that lets a toy creature run the full chain perceive → learn → want → act →
  form-its-own-values → answer-in-words, where a disembodied symbol stream collapses. Labels
  taught, content self-formed; tabula-rasa + grammar remain open frontiers.
- **functional-valence-dyad-v0** — a validated mechanism with a runnable checkpoint
  (`artifacts/active-monkey-affect-dyad-v0`). A symbolic dyad learns which response earns
  approval per inferred intent-like state, judged by a frozen constant-unfakeable scorer. Ships
  an `AdapterCard` (`adapters.json`) and a `ScorerCard` (`scorer_refs.json`).
- **meta-calibration-n3-v0** — agency-over-metacognition at toy richness: a controller that
  detects and repairs miscalibration in the creature's own diagnoses (forecast-scoring trust
  monitor + lock-on-consistency). All N3 constants provided; the ratchet residual is named;
  behavioral consequences untested.
- **online-structure-growth-v0** — detector→grow→quiet works once evaluation uses normalized
  densities; the prior five-design "growth wall" was a capped-footprint **evaluation-convention**
  artifact, not a growth-geometry fact.
- **identity-n4-monitor-v0** — a real, specific read-only identity-displacement monitor (AUROC
  0.894). Detection only; **defense** (commitment control) is the boundary below.
- **communication-scaffold-v0** — explicitly speculative. The `comm_v0` benchmark is an
  existence test (~1.9 bits MI); no selection or emergence result exists. `source_experiments`
  honestly empty.

## Geometry maps (`geometry_maps/`)

| id | mechanism | what it maps |
|---|---|---|
| `dyad-session-length-curve-v0` | functional-valence-dyad-v0 | session length × precision schedule → genuine discrimination (Exp 218–221) |
| `active-sensing-benefit-wall-v0` | active-sensing-probe | probe rate × cost → flat selection slope; no valley (Exp 210–212) |
| `costly-sensing-wall-v0` | costed-sensing-organ | seven ecology levers → no functional organ (Exp 199–205) |

## Boundary notes (`boundary_notes/`)

| id | failed mechanism | the constraint |
|---|---|---|
| `disembodied-stream-collapse-v0` | latent-structure emergence from a disembodied symbol stream | collapse (symmetric saddle / non-identifiability) is substrate-independent; the RECIPE is the symmetry-breaker (Exp 31, 135) |
| `identity-n4-commitment-v0` | N4 commitment control as agency-over-identity | commitment is **config, not agency** at this richness; closed constructively by a stopwatch at fixed-L, flicker-taxed at variable-L (Exp 181–190) |
| `active-sensing-benefit-wall-v0` | costed active information-gathering | useful-when-gifted ≠ locally evolvable; benefit magnitude is the wall (Exp 210–213) |
| `costly-sensing-wall-v0` | evolution of a costed sensory organ | no costed sense becomes functional at this substrate; the fitness valley is the barrier (Exp 199–207) |
| `hidden-state-memory-boundary-v0` | passive hidden-state memory as a locally evolvable trait | the local-gradient wall generalises from senses to memory/inference (Exp 208–209) |

The `disembodied-stream-collapse-v0` boundary and `recipe-symmetry-breaking-v0` mechanism are
two halves of the same finding — the collapse and what breaks its symmetry. The
`identity-n4-commitment-v0` boundary pairs with the `identity-n4-monitor-v0` mechanism —
detection is real, defense is config.

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
