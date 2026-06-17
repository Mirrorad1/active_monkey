# ARTIFACT_SPEC — the coalescence artifact standard (v0.1.0)

This is the canonical specification for **coalescence artifacts**: the reusable
scientific objects an active_monkey experiment must produce so that past, current, and
future experiments accumulate instead of each living and dying as a one-off narrative.

It **extends** `docs/ARTIFACTS.md` — that document specifies the *AgentCheckpoint*
sub-layer (the copyable agent artifact: `active_loop/state.py` + `active_loop/artifacts.py`,
already implemented). This document specifies the broader taxonomy that surrounds a
checkpoint: bundles, mechanism cards, geometry maps, scorers, adapters, and boundary notes.

> **Core principle.** An experiment is not complete until it produces at least one reusable
> artifact. A verdict in `EXPERIMENTS.md` is *evidence*; an artifact is the *citable,
> composable object* the next experiment can import.

## Non-negotiables (the honesty contract for artifacts)

These mirror `loop/VALIDATION.md` and the mission constraints. The validator enforces the
mechanical ones; the rest are review discipline.

- **Never invent data.** A bundle MUST NOT reference a file it does not have
  (`validate_bundle` raises if a ref is missing). A metrics-only bundle has
  `raw_data_refs == []`. Missing data is marked missing, not inferred.
- **Do not infer hidden states** unless the environment logged them or they are
  reconstructable from a deterministic seed + committed config.
- **Do not mutate old raw data or rewrite old results.** Bundles are immutable records;
  re-runs are *new reproduction runs*, never overwrites of historical data.
- **Conservative terms only.** Use: *functional valence*, *belief-like state*, *posterior
  over hidden state*, *symbolic interaction codes*, *costed signaling*, *proto-communication*,
  *adaptive information mechanism*. Never claim sentience, consciousness, AGI, true feeling,
  or natural-language understanding.
- **Frozen scorers stay frozen.** A scorer change is a new `scorer_version` + new `sha256`,
  documented; old artifacts keep pointing at the old hash. Never silently change a scorer.
- **Local-first, deterministic.** No network. No `pickle` for public artifacts. Numeric
  arrays use `safetensors`; metadata uses sorted-key (canonical) JSON.

## Implementation map

| Concern | Where |
|---|---|
| Schemas (dataclasses + serialization) | `active_loop/coalescence/schema.py` |
| Validation | `active_loop/coalescence/validate.py` |
| Inventory of existing experiments | `active_loop/coalescence/inventory.py` |
| Backfill planning | `active_loop/coalescence/backfill.py` |
| Bundle export | `active_loop/coalescence/export.py` |
| CLI | `active_loop/cli/coalesce.py` → `active-monkey coalesce …` |
| AgentCheckpoint / AgentState (reused) | `active_loop/state.py`, `active_loop/artifacts.py` |
| Frozen scorer (reused) | `eval/affect_score.py` |

`SCHEMA_VERSION = "0.1.0"`. Loading refuses a MAJOR.MINOR mismatch unless
`allow_schema_mismatch=True` (same policy as `active_loop/state.py`).

## Serialization rules

- **Canonical JSON** (`schema.write_json` / `read_json`): sorted keys, compact, trailing
  newline, UTF-8. Used wherever a byte-stable hash matters.
- **Optional YAML** (`schema.dump` / `load`): a `.yaml`/`.yml` path is written as YAML *iff*
  `pyyaml` is importable, otherwise it falls back to JSON. `pyyaml` is a dev-only optional
  dependency (matches `ecology/evolvability/config.py`) — artifacts never *require* it.
- **Trajectories**: JSONL is canonical (`schema.write_trajectory_jsonl`). Parquet/CSV are
  acceptable alternates for bulk numeric data. Old experiments without raw trajectories are
  **not** forced into this format.
- Every artifact dict carries `artifact_type` and `schema_version`.

## The artifact types

Each dataclass exposes `to_dict()`, `from_dict(d, allow_schema_mismatch=False)`, an
`ARTIFACT_TYPE` string, and is registered in `schema.ARTIFACT_REGISTRY`. **Required** fields
have no default; the validator lists every missing one. **Optional** fields default to
`None`/`[]`/`{}` and are never auto-populated (honesty).

### 1. ExperimentBundle — immutable record of one experiment
`artifact_type: "experiment_bundle"`. The on-disk `manifest.json` of a bundle directory.

- **Required:** `experiment_id`, `direction`, `question`, `hypothesis`, `status`, `verdict`,
  `repo_commit`, `created_at`, `confidence`, `backfill_level`.
- **Optional ref-lists** (empty unless the file genuinely exists): `source_files`,
  `raw_data_refs`, `metrics_refs`, `scorer_refs`, `state_refs`, `mechanism_refs`,
  `geometry_refs`, `caveats`, `reproduction_command`.

### 2. ExperimentSpec — the world/equations tested
`artifact_type: "experiment_spec"`. Required `experiment_id`; all else optional (an honest
spec may only know some): `variables`, `hidden_state`, `observations`, `actions`,
`reward_or_fitness`, `costs`, `update_rules`, `selection_rules`, `environment_dynamics`,
`agent_dynamics`, `stop_condition`, `pass_condition`, `fail_condition`.

### 3. TrajectoryData — raw timestep-level evidence
`artifact_type: "trajectory_row"`, written one-per-line as JSONL. Required `experiment_id`,
`seed`, `t`. Optional: `direction`, `episode`, `agent_id`, `environment_id`, `hidden_state`,
`observation`, `belief_state`, `action`, `message`, `reward_or_valence`, `fitness`,
`next_observation`, `next_hidden_state`, `metadata`. Helpers: `write_trajectory_jsonl`,
`read_trajectory_jsonl`, `validate_trajectory_row`.

### 4. AgentCheckpoint / AgentState — portable learned/runtime state
**Already implemented** in `active_loop/state.py`; see `docs/ARTIFACTS.md`. "Weights" here
means any numeric state determining behavior: probability tables, learned Dirichlet counts,
priors, posterior beliefs, policy parameters, or neural tensors if they ever exist. Stored as
`state.safetensors` + `*.config.json` + a validated `manifest.json`
(`state.REQUIRED_MANIFEST_KEYS`). A stable `content_hash()` excludes volatile
`created_at`/`repo_commit` so identical agents hash identically. The coalescence layer
references these via a bundle's `state_refs`; it does not re-implement them.

### 5. ScorerCard — reusable pass/fail evaluator
`artifact_type: "scorer_card"`. Required: `scorer_id`, `scorer_version`, `file_path`,
`sha256`. Optional: `metrics`, `required_controls`, `pass_conditions`, `fail_conditions`,
`limitations`. `ScorerCard.from_file(...)` computes the sha256 via `artifacts.hash_file`.
Controls to name where relevant: constant-response, shuffled-belief, stale/reset-belief,
shuffled/muted-message, sensor-ablation, decoy-feature, cost-increase, held-out-seeds/env.

### 6. MechanismCard — a distilled, reusable scientific mechanism
`artifact_type: "mechanism_card"`. Required: `mechanism_id`, `mechanism_type`, `claim`,
`status`, `source_experiments`. Optional: `works_when`, `fails_when`, `required_conditions`,
`reusable_interface`, `inputs`, `outputs`, `state_requirements`, `costs`, `metrics`,
`falsifiers`, `known_confounds`, `next_compositions`.
- `status ∈ {validated, falsified, constrained, speculative, scaffold}`.
- `mechanism_type ∈ {functional-valence-learning, hidden-state-belief, costed-sensing,
  uncertainty-gated-probing, costed-signaling, selection-stabilized-trait,
  transfer-invariant-abstraction, identity-self-modeling}`.

### 7. GeometryMap — where a mechanism works/fails in parameter space
`artifact_type: "geometry_map"`. Required: `geometry_id`, `source_experiments`. Optional:
`mechanism_id`, `swept_parameters`, `fixed_parameters`, `metrics`, `outcome_regions`,
`thresholds`, `negative_regions`, `positive_regions`, `confounds_excluded`,
`next_experiment_suggestions`. Many active_monkey findings are "worked/failed only after
trying" — so the reusable artifact is often a **boundary**, not a positive result.

### 8. AdapterCard — how one mechanism composes into another
`artifact_type: "adapter_card"`. Required: `adapter_id`, `from_mechanism`, `to_mechanism`.
Optional: `input_contract`, `output_contract`, `required_state`, `assumptions`,
`failure_modes`, `tests`.

### 9. BoundaryNote — a negative result preserved as a reusable constraint
`artifact_type: "boundary_note"`. Required: `boundary_id`, `source_experiments`,
`failed_mechanism`, `observed_failure`. Optional: `tested_conditions`, `excluded_confounds`,
`implication`, `next_safe_region_to_test`.

## Standard on-disk layout

```
experiment_bundles/
  exp222/
    manifest.json          # ExperimentBundle
    spec.json              # ExperimentSpec (optional)
    metrics.json           # measured numbers
    scorer_card.json       # ScorerCard for the frozen scorer used
    reproduction.sh        # exact rerun command (if available)
    README.md
    # state_refs may point at artifacts/active-monkey-affect-dyad-v0/
mechanisms/
  functional-valence-dyad-v0/
    mechanism_card.yaml     # MechanismCard
    scorer_refs.json
    adapters.yaml           # AdapterCard(s)
    README.md
geometry_maps/
  dyad-session-length-curve-v0.yaml
  active-sensing-benefit-wall-v0.yaml
boundary_notes/
  costly-sensing-wall-v0.yaml
artifacts/
  active-monkey-affect-dyad-v0/   # AgentCheckpoint artifact (see docs/ARTIFACTS.md)
```

## Confidence levels (inventory)
- **high** — raw data / checkpoint present, or script + structured metrics (rerun-able with raw artifacts).
- **medium** — metrics + source files present.
- **low** — only a summary/log present.
- **unknown** — a mention exists but the source is unclear.

## Backfill levels (what a bundle can honestly claim)
| Level | Name | Meaning |
|---|---|---|
| 0 | `index_only` | The experiment exists; no reliable data found. |
| 1 | `summary_bundle` | Docs/summary only. Confidence low/medium. No raw-trajectory claim. |
| 2 | `metrics_bundle` | Structured metrics exist. Store `metrics.json` + refs. No raw-trajectory claim unless raw logs exist. |
| 3 | `repro_bundle` | A committed, deterministic, runnable script/config exists. Store the command. |
| 4 | `trajectory_bundle` | Raw trajectories exist (or can be freshly re-run). Store trajectory data. |
| 5 | `checkpoint_bundle` | Agent before/after state can be saved. Store the AgentCheckpoint. |
| 6 | `mechanism_bundle` | MechanismCard + GeometryMap are supported by evidence; promote to `mechanisms/`. |

The inventory reports the **highest honestly achievable** level plus every `has_*` boolean,
so nothing is lost. A re-run that produces new data is recorded as a *new reproduction run*,
never as original historical data.

## CLI

```bash
uv run active-monkey coalesce inventory --json
uv run active-monkey coalesce backfill-plan --out /tmp/active_monkey_backfill_plan.json
uv run active-monkey coalesce export --experiment exp222 --level metrics_bundle --out experiment_bundles/exp222
uv run active-monkey coalesce validate experiment_bundles/exp222
uv run active-monkey coalesce validate --all
uv run active-monkey coalesce mechanisms list
uv run active-monkey coalesce geometry list
```

See `docs/COALESCENCE_PLAN.md` for the why and the path forward, `docs/EXPERIMENT_BACKFILL.md`
for the current inventory + rerun plan, and `docs/MECHANISM_LIBRARY.md` for the seeded cards.
