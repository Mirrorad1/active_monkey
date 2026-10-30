# Local Replay World Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare a small local recurrent predictor using three bounded replay selectors, with reproducible data and honest verification.

**Architecture:** A synthetic delayed-cue stream feeds an identical GRU and three prediction heads in every arm. Only replay admission differs. A local runner writes checkpoints and raw evaluation records; independent verification consumes the frozen criteria and raw data.

**Tech Stack:** Existing Python, JAX, Equinox, NumPy, pytest, Matplotlib; CPU reference execution; no LLM API calls in training.

**Spec:** `docs/superpowers/specs/2026-10-30-local-replay-world-model.md` (read alongside this plan).

## Global constraints

- Planning complete; implementation and training have not started. Execute on a later implementation instruction.
- Run Python from the repo root with `uv run --python .venv`.
- Never edit FROZEN paths, persistent creature state, or unrelated working changes.
- Fixed 128-unit GRU, 3 heads, 1,000 clips, 32 transitions/clip, 8-step burn-in,
  horizons {1,4}, 1 current + 7 replay clips/update, Adam 0.001, clip norm 1.0.
- Three 32,768-transition phases; smoke 512 transitions; 2 development seeds;
  8 fresh confirmation seeds. No automatic sweep or shortening of the horizon.
- Research scripts/raw outputs stay in experiments/ and accompany the append-only
  experiment entry and site update in one atomic experiment commit.
- Workers do not commit, edit global research logs, grade scientific results, or spawn children.
- Keep this project’s conductor/worker separation from loop/ROUTING.md. The preferred
  legacy worker model there is not exposed in this session; the mapping below uses
  available models for the same bounded roles, without changing project routing files.
  A Codex lead can deliver engineering and a proposed verdict; the designated Loop B
  conductor remains responsible for ratifying the scientific predeclaration/verdict.

## Usage policy and dispatch order

Use this chat as coordinator. Do not create separate user-visible chats. Default:
**three worker launches total**, with at most two implementation workers concurrently.
Parallelism reduces elapsed time; it is not itself a token saving.

| Role | Model / effort | Scope | Launch |
|---|---|---|---|
| Coordinator | Current lead; Sol Medium is a suitable future coordinator | Contracts, integration, pilot, runs, documentation | Existing chat |
| Worker A | `gpt-6-luna`, high | Simulator, tape contract, replay selectors, focused tests | After Task 0 |
| Worker B | `gpt-6.1-sol`, medium | GRU, training/checkpoints, metrics, runner, focused tests | After Task 0; parallel with A |
| Verifier C | `gpt-6-luna`, high | Independent recomputation from frozen criteria/raw output; read-only | After confirmation |

Use `fork_turns="none"` and an explicit compact packet. Mandatory repository bootstrap
still applies; do not inherit the whole conversation, old reports, or other agents'
transcripts. Supply the spec, shared contract, assigned task section, allowed paths,
verification command, and requested return format. Return <=400 words plus file paths
and test outputs. Send follow-ups to the same worker for bounded fixes. One repair
cycle is the default; persistent failures return a concrete blocker to the lead.

The lead reviews both code slices once before training. No automatic review swarm,
separate planning agent, research fan-out, or one agent per seed. If a correctness
ambiguity remains, add ONE narrowly scoped Sol High review; disclose the extra call.
Fresh verifier C must not inherit results narratives. Its mandatory RESUME bootstrap
may expose project context, so report it as independent raw verification, not strict blinding.

Use the standard speed setting where available; this plan does not change app settings.
Local numerical training uses this machine's compute; it does not call a language
model or spend API credits. Agent orchestration/debugging still consumes allowance.
Write progress to local artifacts, inspect only at stage completion/error, and do not
keep asking an agent to narrate epochs. No recurring automation is part of this plan.

Account snapshot on 2026-10-30: tool reported 15% used in a 10,080-minute window,
85% remaining at that instant, and no purchased credits. This is shared account usage,
not a project budget or an estimate of how much this plan will consume. Check again
at dispatch and completion, not every run. Exact per-model subscription deductions
are not established here; do not convert API prices into subscription percentages.
Official guidance supports narrower context and smaller outputs, and Luna for bounded
subtasks: [usage guidance](https://learn.chatgpt.com/docs/pricing),
[subagent model selection](https://learn.chatgpt.com/docs/agent-configuration/subagents).

## Review focus

1. A future target or private phase flag leaks into admission/training: reject and record it.
2. Replay starts from a stale hidden state or corrupts the streaming life: reconstruct independently.
3. Phase boundaries and partial clips yield off-by-one targets: mask immature/out-of-range targets.
4. Capacity equality omits selector metadata or raw-tape access: account for all learner storage.
5. Resuming changes RNG, update count, or optimizer state: match uninterrupted CPU execution.

## Task 0 — coordinator: freeze shared interfaces and workspace

**Files owned:** `experiments/local_replay/__init__.py`, `contracts.py`, `config.json`,
and `tests/test_local_replay_contracts.py`; the numbered script and reporting files
remain coordinator-owned later. All paths below are relative to repository root.

- [ ] Inspect existing attached worktrees and current changes; reuse an appropriate
  isolated checkout or create a managed worktree. Read applicable instructions and
  FROZEN. The main checkout currently has unrelated changes; never copy/stage them.
  Check git hooks: this repo has an auto-push-on-main post-commit hook. Work on a
  `codex/` branch and do not publish as a side effect of local commits.
- [ ] Freeze the spec and numbered experiment's hypothesis/prediction/falsifier with
  the conductor. Reserve the next free number; do not assume the main checkout has
  every experiment from other worktrees. Confirm artifact/atomic-commit requirements.
- [ ] Write a failing `test_clip_shapes_and_private_fields` asserting a learner Clip
  has `obs: uint8[33,2]`, `actions: uint8[32]`, `clip_id: int`, and no phase/rule/noise
  flag. Observation field ranges are {0..5,0..1}; actions are {0..2}. A Tape holds
  public arrays only; audit metadata is a separate object never passed to a model.
- [ ] Implement these dataclasses and validation in contracts.py. Specify action
  convention `actions[t]` connects `obs[t]` to `obs[t+1]`. Freeze enum mappings in
  config.json (WAIT=0, LEFT=1, RIGHT=2; symbol order as in the spec).
- [ ] Run `uv run --python .venv pytest tests/test_local_replay_contracts.py -q`;
  require a pass before dispatch. Provide workers this actual contract, not stubs
  whose meaning may change. Workers A and B can now proceed independently.

## Task 1 — Worker A: simulator and replay

**Files owned:** `experiments/local_replay/world.py`, `replay.py`,
`tests/test_local_replay_world.py`, `tests/test_local_replay_replay.py`.

**Interfaces:** `make_tape(seed: int, phase_steps: int) -> tuple[Tape, Audit]`;
`iter_clips(tape: Tape) -> Iterator[Clip]`;
`ReplayBuffer(mode: str, capacity: int, seed: int)` with
`admit(clip: Clip, score: float) -> None`, `sample(n: int) -> list[Clip]`,
`state_dict() -> dict`, `load_state_dict(state: dict) -> None`, and
`allocated_bytes() -> int`. Modes: `fifo`, `reservoir`, `disagreement`.

- [ ] Add failing tests: `test_delayed_cue_and_rule_inversion` checks identical cue
  and door action produce opposite outcomes across A/B; `test_noise_does_not_change_door`
  changes only distractor RNG and requires identical symbols/actions;
  `test_phase_boundary_alignment` checks exact phase lengths and target indices.
- [ ] Implement the spec's generator and streaming clip iterator. Training sampler
  uses observations/legal policy only; private audit fields never shape admission.
- [ ] Add failing replay tests: FIFO evicts oldest; reservoir inclusion agrees with
  uniform expectation over a fixed deterministic Monte Carlo cohort with declared
  tolerance; disagreement keeps the specified top scores; seeded ties reproduce;
  all modes allocate exactly equal arrays at capacity; serialization preserves
  future sampled clip IDs; nonfinite scores raise rather than enter the archive.
- [ ] Implement selectors. Scores stay frozen; no rescoring sweep. Allocate IDs,
  scores, tie keys, and counters in every arm, even when a selector ignores them.
- [ ] Run `uv run --python .venv pytest tests/test_local_replay_world.py tests/test_local_replay_replay.py -q`.
  Return paths, pass/fail output, and any contract issue; no git commit or research verdict.

## Task 2 — Worker B: neural predictor, training, evaluation

**Files owned:** `experiments/local_replay/model.py`, `training.py`, `metrics.py`,
`run.py`, `tests/test_local_replay_model.py`, `tests/test_local_replay_training.py`.

**Interfaces:** `init_model(seed: int) -> Model`;
`predict(model: Model, clip: Clip) -> Predictions` (probabilities [3,32,2,6] for
symbols and [3,32,2,2] for noise, horizon axis {1,4}, plus bool[32,2] eligibility);
`disagreement(predictions: Predictions) -> float`;
`update(state: TrainState, clips: list[Clip]) -> tuple[TrainState, dict]`;
`evaluate(model: Model, probes: list[Clip]) -> list[dict]`;
`save_checkpoint(state: TrainState, path: Path) -> None` and
`load_checkpoint(path: Path) -> TrainState`. Define Model/Predictions/TrainState
locally in the owned modules; a model must not accept Audit or an environment object.

- [ ] Test causality: modifying future observations must not alter earlier
  predictions; modifying query actions 2–4 must not alter horizon-1 probabilities.
  Tests must actually change future inputs and exercise different action queries.
- [ ] Implement GRU/heads and masks. Verify normalization, finite gradients, actual
  parameter count in the proposed 100k–300k range, bootstrap diversity, and zero
  disagreement for identical predictions. Disagreement never consumes targets.
- [ ] Test `test_mature_targets_only` and `test_replay_cannot_mutate_stream_hidden`;
  implement burn-in, current/replay scheduling, separate RNGs, and equal score calls.
  Reuse deterministic tiny Clip fixtures so this work need not wait for Worker A.
- [ ] Implement raw metrics with a hand-calculated Brier fixture. Record every
  probe ID, probabilities, target, cue-stratum audit join key, training step, and
  phase in evaluation output; learner selection cannot read that output.
- [ ] Implement train/resume/checkpoint and CLI stages `smoke`, `dev`, `confirm`.
  Test uninterrupted vs checkpoint/resume equality on CPU, including optimizer,
  selected IDs, predictions, RNGs, streaming state, and counters. No swallowed NaNs.
- [ ] Add wall-time limits with checkpoint and an explicit incomplete marker.
  Write all outputs below the user-supplied experiments/ output directory. Refuse
  accidental overwrite. Hash config, sources, data, and versions. Refuse confirm
  without frozen config and disjoint development/confirmation seed namespaces.
- [ ] Run `uv run --python .venv pytest tests/test_local_replay_model.py tests/test_local_replay_training.py -q`.
  Return <=400 words plus commands/output. Training on final seeds is outside this task.

## Task 3 — coordinator: integration and bounded local runs

**Files owned:** final integration fixes by agreement with workers, README.md under
`experiments/local_replay/`, numbered `experiments/expNN_local_replay.py`, outputs
under `experiments/outputs/local_replay/`, and `experiments/outputs/expNN.txt`.

- [ ] Review A and B for the five failure modes above; resolve interfaces once.
  Run `uv run --python .venv pytest tests/test_local_replay_*.py -q` and the relevant
  existing fast suite. Require green checks before any numerical interpretation.
- [ ] Implement the numbered wrapper with the frozen docstring and commands:
  `uv run --python .venv python -m experiments.local_replay.run --stage smoke --output experiments/outputs/local_replay/smoke`;
  corresponding `dev` and `confirm` stages use distinct output paths. Document
  `--resume CHECKPOINT` semantics and the exact seed/config manifest.
- [ ] Run smoke; inspect output schema, finite loss, real parameter changes, and
  normalized probabilities. A pass is engineering evidence, not scientific support.
- [ ] Run the two development seeds within 60 minutes at full horizon. Record
  first-phase learning and all failures. Estimate 24-run confirmation time from
  measured steady-state timings separately from compilation. Report the estimate
  if it exceeds four hours; do not silently reduce work or launch a sweep.
- [ ] Freeze final config, code hashes, namespace and metrics. Execute confirmation
  as one local job with a four-hour cap, checkpoints, and a completion manifest.
  No agent per seed; no epoch narration. Incomplete output cannot be called a result.
- [ ] Prepare verifier packet: frozen hypothesis/prediction/falsifier, raw numeric
  artifacts and their hashes, schema/metric definitions. Exclude narrative verdicts.

## Task 4 — Verifier C and coordinator: verification and delivery

**Verifier scope:** read-only frozen criteria and raw artifacts, with mandatory
repository bootstrap. May use shell calculations but must not modify outputs/logs.

- [ ] Recompute Brier from probabilities/targets, eight checkpoints per window,
  paired differences against BOTH baselines, and the joint seed pass count.
- [ ] Check all 8 seeds, data hashes, budget/update equality, complete records,
  held-out IDs, and finite distributions. Report any uncheckable field explicitly.
- [ ] Return component table, arithmetic verdict proposal, and caveats in <=400
  words; printed runner verdicts are not evidence. Escalate only numerical ambiguity.
- [ ] Coordinator/conductor ratifies interpretation; append log and site entry,
  update resume notes, and commit numbered script, helpers/tests, complete raw data,
  config, compact checkpoints and log/site together. Use allowlisted paths only.
  Run `uv run --python .venv python loop/check_iteration.py` and required checks
  before committing; then collect passive process memory per PROTOCOL.md.
- [ ] Deliver training/resume commands, trained checkpoint, learning-curve file,
  seed table and honest limitations. If NEGATIVE, stop this hypothesis; a changed
  selector or world needs a fresh explicitly named hypothesis and predeclaration.

## Copyable execution handoff

> Implement docs/superpowers/plans/2026-10-30-local-replay-world-model.md and its
> linked spec. Start with Task 0. Use detached, tightly scoped Luna High for Worker A
> and Sol Medium for Worker B; no child agents. The lead integrates and runs local
> jobs, then launches one fresh Luna High verifier with raw data and frozen criteria.
> Preserve the research-conductor boundary in loop/ROUTING.md and all honesty rules.
> Reuse suitable isolated worktrees; protect unrelated changes and FROZEN files.
> Stay within the staged compute budgets and do not optimize on confirmation data.
> Report implementation, smoke/pilot evidence, and scientific results separately.

## Plan self-review

The spec maps to Tasks 0–4; shared Clip layout is fixed before parallel dispatch;
every review-focus item has an owning test; compute limits and partial failures have
explicit outcomes. No implementation, model training, or findings are claimed here.
