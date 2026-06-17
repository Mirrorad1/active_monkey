# EXPERIMENT_BACKFILL — current inventory + honest rerun plan

This is the human-readable companion to the machine-generated inventory. Regenerate the
authoritative version any time:

```bash
uv run active-monkey coalesce inventory --json            # full per-experiment inventory
uv run active-monkey coalesce backfill-plan --out plan.json
```

The numbers below were produced by `active_loop/coalescence/inventory.py` against the repo at
the time of writing. Treat the CLI output as the source of truth; this doc explains it.

## Coverage

- **225 experiments** inventoried (Exp 1–225), joining `site/data/experiments-data.js`,
  `EXPERIMENTS.md` (verdict tags), and on-disk evidence under `experiments/` + `artifacts/`.
- **Confidence:** high = 88, medium = 137, low = 0, unknown = 0. Every logged experiment has
  a committed, deterministic script, so none fall to low/unknown — an honest reflection of the
  repo's discipline, not an inflation (the inventory *can* emit low/unknown; nothing here
  qualifies).

### By direction

| Direction | n | Exp range |
|---|---:|---|
| language | 16 | 1–16 |
| embodiment-valence-recipe | 24 | 17–40 |
| persistent-creature | 84 | 41–124 |
| affective-dyad | 20 | 125–132, 214–225 |
| continuous-substrate | 22 | 133–154 |
| meta-calibration-n3 | 19 | 155–173 |
| identity-n4 | 20 | 174–193 |
| population-ecology | 5 | 194–198 |
| costed-sensing | 9 | 199–207 |
| hidden-state-memory | 2 | 208–209 |
| active-sensing | 1 | 210 |
| uncertainty-gated-active-sensing | 1 | 211 |
| evolvability-geometry | 2 | 212–213 |

## What can be backfilled, and at what level

Backfill levels (see `docs/ARTIFACT_SPEC.md`) report the **highest honestly achievable**
bundle for an experiment. The exporter **refuses to over-claim** a level the evidence cannot
support.

| Level | Where it applies (representative) | Why |
|---|---|---|
| `checkpoint_bundle` (5) | Exp 222, 225 (affective-dyad) | Covered by the runnable `active-monkey-affect-dyad-v0` checkpoint artifact (`SOURCE_EXPERIMENTS = [222, 225]`). |
| `trajectory_bundle` (4) | Exp 194–206 (ecology series) | Per-seed `traj_arm*_seed*.json` / `trajectories.json` / `events.jsonl` exist on disk. |
| `repro_bundle` (3) | Exp 210, 211 + most non-ecology exps | Committed deterministic script exists; raw trajectories do **not** — so a *trajectory* claim is refused; a rerun reproduces the result. |
| `metrics_bundle` (2) | Exp 217, 221, 225b, etc. | Structured per-seed metrics JSON exists; no step-level trajectories. |
| `summary_bundle` (1) | (none currently — all logged exps have a script) | Would apply to a summary-only entry. |
| `index_only` (0) | (none currently) | Would apply to a mention with no data. |

### Backfill immediately (no rerun needed)
Any experiment at `metrics_bundle` or above can be exported now from committed evidence:

```bash
uv run active-monkey coalesce export --experiment exp222 --level checkpoint_bundle --out experiment_bundles/exp222
uv run active-monkey coalesce export --experiment exp199 --level trajectory_bundle --out experiment_bundles/exp199
uv run active-monkey coalesce export --experiment exp210 --level repro_bundle      --out experiment_bundles/exp210
```

These reference original committed files in place; they never copy or mutate raw data.

### Need a rerun to reach a higher level
- **To reach `trajectory_bundle`** for a `repro_bundle` experiment (e.g. Exp 210/211, most of
  the affective-dyad and early ranges): re-run the committed script to regenerate raw
  per-timestep logs. This is a **new reproduction run**, recorded as such — never relabeled as
  original historical data. The exact command is emitted per-experiment in
  `coalesce backfill-plan` (`rerun_commands`).
- **To reach `checkpoint_bundle`** for an affective-dyad experiment without one: export an
  `AgentCheckpoint` via `active-monkey artifact export` (the affect-dyad preset already
  supports before/after state).
- **Do not rerun expensive experiments automatically.** The plan lists the commands; running
  them is a deliberate, separately-logged act.

## Known gaps (honest)
- **Exp 226** (the M4b autonomous find-and-keep run) exists in `experiments/` but is **not yet
  in `EXPERIMENTS.md` or `experiments-data.js`**, so it is intentionally absent from the
  inventory (1–225). It should be logged to the journey before backfilling.
- **Direction is range-assigned**, then cross-checked against the JS `chapter`. A handful of
  experiments span direction boundaries (e.g. the M4a attempts at 125–132 vs the main arc at
  214–225); both are labelled `affective-dyad`.
- The inventory verifies file **existence**, not content correctness. A `trajectory_bundle`
  claim means trajectory files are present, not that they have been re-validated this session.

## Rerun-plan principles (binding)
1. Never invent raw trajectories from summaries.
2. Never infer hidden states unless logged or reconstructable from a deterministic seed/config.
3. Preserve original provenance; reference, don't copy.
4. A small, cheap, already-supported smoke rerun may be run — but it is marked a **new
   reproduction run**, not original data.
