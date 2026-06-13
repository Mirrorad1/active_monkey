# ecology/runtime.py — fork / replay / restore as first-class primitives

Makes the "lossless replication, pausing/resuming, memory-state copying, replaying digital experience"
advantages concrete for the **population/ecology** substrate. A running `Ecology` is fully determined by
`(cfg, seed)` **plus** its mutable state (creatures + world + the live rng cursor), so its entire
developmental state is a *value* you can capture, restore, fork, replay, and distil from.

## Primitives (`ecology/runtime.py`)

| Primitive | What it does | Invariant (tested) |
|---|---|---|
| `snapshot(eco) -> Snapshot` | capture the full live state (all creatures, the world resource array, the **rng cursor**, t, events) | picklable → pause-to-disk |
| `restore(snap) -> Ecology` | rebuild a running Ecology from a snapshot | **continuing a restored run is BIT-IDENTICAL to never pausing** |
| `fork(eco, seed=…)` | a counterfactual twin sharing all history to the fork point; `seed=None` is faithful, `seed=<int>` diverges | re-seeded future ≠ baseline; pre-fork history identical |
| `replay(snap, until, expect_hash=…)` | deterministic re-run with an optional committed-hash bit-match gate | unmatched replay raises (a different run) |
| `distill(ecos, strategy=…)` | distil a **founder genotype from successful lineages** — `survivor_mean` (the gene pool the environment kept) or `top_reproducer` (the most-reproductive lineage) | returns a valid `Genotype` |
| `fork_run_compare(base, until, treatment_seed=…, treatment_env=…)` | fork → run → compare against the unforked baseline, in one call | re-seeded treatment diverges from baseline |

The losslessness mechanism: numpy's `Generator.bit_generator.state` is captured/restored, so the exact
rng stream resumes — `snapshot`+`restore`+continue reproduces the straight-through `events_hash` to the
bit. Guards: `tests/test_ecology_runtime.py`.

## `distill` closes the "replay_or_distill" gap

Previously the repo logged trajectories but had no distillation step. `distill` takes the successful runs
(surviving / most-reproductive lineages) and produces a founder genotype to seed a new run — "replay
distilled success", bootstrapping the next generation from what worked. Example:

```python
from ecology.runtime import run_to, distill, fork_run_compare
from ecology.engine import Ecology

runs = [run_to((cfg, s), 6000) for s in range(5)]      # 5 developmental trajectories
founder = distill(runs, strategy="survivor_mean")       # distil the kept gene pool
boosted = Ecology(replace(cfg, founder=founder), seed=99)  # re-seed from distilled success
```

## Honest scope (what is and isn't unified)

- **Ecology substrate: complete + bit-provable.** fork/snapshot/restore/replay/distill + the
  compare-against-baseline pipeline, all tested.
- **Creature spine (nira, `creature/`):** already has its own `Creature.fork()` (full deepcopy + lineage,
  the counterfactual control) and `growth.replay_nll()` (a replay buffer). This module provides the same
  family for the population side.
- **Not yet built:** a *single unified `AgentState`* spanning both substrates (the rich
  code+model+beliefs+preferences+memory+meta-controller composite of the creature spine AND the
  population), and distillation *into a learned generative model* (the ecology creatures use a heuristic
  policy, so `distill` operates on the heritable genotype, not on A/B matrices). Those are the documented
  next steps.
