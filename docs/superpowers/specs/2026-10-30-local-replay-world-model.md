# A small local world model with selective replay

Status: implementation specification and proposed research protocol; unimplemented,
unrun, 2026-10-30. Updated after the user's storage and observatory mockup requests.
The user requested a plan for later subagent dispatch. This document
does not authorize an unattended campaign or report a scientific result.

## Question and scope

Can disagreement-based memory selection improve a small recurrent model's adaptation
and recovery relative to uniform and recent replay, at the same storage and update
budget, in a world containing delayed cues and unpredictable distractors?

Build a local, randomly initialized neural predictor, not an LLM. This first study
tests replay selection under recurring dynamics, not autonomous concept invention,
active inference, novel goal discovery, or arbitrary future-task compression. Those
are separate follow-ups. No changes to the persistent creatures or Loop A.

## Smallest useful world

Use a synthetic stream of 32-transition encounters. At offset 0 the observation
reveals a binary cue, with P(cue=1)=0.1. At a delay drawn uniformly from {8,16,24},
the observation presents a door opportunity. A supplied behavior policy chooses
left or right equiprobably there and WAIT elsewhere. At the following observation,
the door is open iff the selected door equals cue XOR rule. All other observations
are blank. The next encounter begins at offset 32. Encounters are an explicit
world structure; the streaming learner's recurrent state and weights never reset.

Observation has two categorical fields: six symbols (blank, cue0, cue1, door,
opened, closed) and a binary distractor. Independently, 20% of encounters have an
IID Bernoulli(0.5) distractor at every observation; the rest have distractor zero.
The noisy-encounter flag, cue after its first presentation, rule, delay, and time
since the last change are evaluator-only metadata. Inputs expose no phase or
encounter index. Sequence boundaries are provided for batching and are disclosed.

Each phase lasts 32,768 transitions. Run rule A=0, B=1, then A=0 again (98,304 total).
Generate one immutable tape per seed; all arms receive exactly the same tape in
the same order through a streaming reader. A persisted full tape is a reproduction
artifact, never extra learner memory. Private audit metadata travels separately.
This is an action-conditioned prediction task with a supplied exploration policy;
it does not establish exploration or navigation ability.

## Model and training contract

- Existing JAX + Equinox + NumPy; CPU is the reference backend. Use a local Adam
  implementation or an already available optimizer with declared dependency;
  the lead owns any dependency decision. Do not install another ML framework.
- One GRU with 128 hidden units; input is observation one-hots (8 values) and
  previous action one-hot (3). Initial state is zero only at the start of a life.
- Three independently initialized prediction heads, each an MLP with 128 hidden
  units. Inputs are recurrent state plus four query-action one-hots (12 values).
  Each emits categorical distributions for symbol and distractor at horizons 1
  and 4. Horizon-1 prediction must not depend on later query actions: evaluate
  its output with query slots 2–4 zeroed. Anticipated total: roughly 115k parameters;
  record the actual count. Shared backbone means this is not a posterior ensemble.
- Future actions are explicitly supplied queries, not future observations. A
  four-step target is eligible only after it arrives and only when all intervening
  actions are recorded. Do not let four-step labels enter training early.
- Mask targets outside each 32-step encounter. Replay reconstructs hidden state
  from zero using the first 8 steps as burn-in, with no loss there. All relevant
  cues are inside the clip; never cache stale hidden states. Streaming hidden state
  continues separately; replay must not replace it.
- Update once per completed encounter: current clip plus 7 replay clips, uniformly
  sampled from the retained buffer with replacement, excluding the current clip.
  On the first encounter, repeat the current clip for the missing slots. Adam
  learning rate 0.001; global gradient clipping at 1.0; one update per batch.
- Average categorical cross-entropy equally across both fields and horizons,
  then across eligible times. Each head uses a deterministic bootstrap inclusion
  mask P=0.8 per clip, keyed by clip ID and head ID; if all exclude a clip, include
  head 0. The shared backbone gets gradients from the included heads.
- All arms have identical heads, bootstrap masks, initialization, update counts,
  inference scoring calls, and allocated arrays. Separate RNG streams for world,
  initialization, admission, bootstrap, and replay sampling prevent accidental
  coupling. Save a complete resumable state including optimizer, RNG, memory,
  stream cursor, and streaming hidden state.

## Three memory arms

Capacity is 1,000 completed encounters. Use equal fixed-size storage for clip
payloads, IDs, admission scores, and selector metadata in every arm. Report total
allocated bytes, model/optimizer bytes, temporary scoring peak, and wall time.

1. FIFO: replace oldest retained clip.
2. Reservoir: standard uniform reservoir sampling over completed clips.
3. Disagreement: score a completed clip before learning from it. For every eligible
   time, field, and horizon, calculate H(mean_head(p)) - mean_head(H(p)); average
   these values. This is Jensen–Shannon disagreement, not target prediction error.
   Retain the top 1,000 admission scores, breaking ties by a seeded random key.
   Scores are frozen at admission. Staleness/early-lock-in is an explicit limitation
   and possible reason for failure; no refresh mechanism is silently added.

Every arm computes the same candidate score before updating, even when its selector
ignores it. Score first, train using the old buffer, then admit the clip. No hidden
metadata, evaluation targets, future phases, or future clips enter selection.

## Measurements and provisional predeclaration

The lead must freeze the executable docstring/config before coding the numbered
experiment and before confirmatory runs. This plan proposes these exact criteria;
changes after development must be dated and disclosed, never fitted to held-out data.

Evaluate frozen model clones every 512 training transitions, including phase ends
and boundaries. At each checkpoint use 64 fresh held-out encounters balanced by
cue and door action, with delays and noise sampled as above. Probes are matched
across arms. Each probe starts a counterfactual zero hidden state and reconstructs
the visible history; probes never train, fill memory, or advance the training RNG.

Primary error: mean sum of squared differences between the six-symbol horizon-1
distribution and the one-hot true symbol, measured only at door outcomes. Its range
is [0,2]; an ideal equal open/closed guess scores 0.5. Report per-cue errors too.
For each seed, adaptation error is the average of the eight checkpoints 512 through
4,096 transitions after entry into B; recovery error uses the same window after
return to A. Other checkpoints are descriptive, not substitute primary endpoints.

Proposed confirmation: 8 fresh seed namespaces, all arms, with a joint pass requiring:

- In at least 6/8 seeds, disagreement lowers BOTH adaptation and recovery error
  by at least 0.02 absolute against EACH of FIFO and reservoir.
- In those same seeds, its final A-phase-1 error is <=0.25 and no more than 0.02
  worse than either baseline (prevent an adaptation claim from starting untrained).
- All 8 seeds complete with finite probabilities, matched data, matching update
  counts, equal allocated learner storage and all required probe events present.

A valid failure of the performance conjunction is NEGATIVE; report every seed and
component. A numerical/leakage/budget defect is an instrument failure, not evidence
for the hypothesis; preserve the failure and repair before a newly frozen run.
Do not exclude bad-performing seeds as invalid. The 6/8 rule is a toy replication
criterion, not a p-value or evidence of general superiority.

Report noise-field Brier separately, retained noisy-encounter proportion, age/cue
composition of memory, three-head disagreement, training loss, runtime, and bytes.
These are diagnostics: they do not by themselves establish noise immunity, correct
uncertainty, explicit context retrieval, or neural plasticity. Larger/fresh world
families, a stationary noise-only control, and active action selection are deferred.

## Cost and execution gates

1. Unit/contract checks and a 512-transition smoke run. No scientific verdict.
2. Two development seeds at the FULL three-phase horizon, three arms. Benchmark
   CPU compilation, training, checkpoint size, and peak memory. Initial local cap:
   60 minutes total. Save partial artifacts if the cap is reached.
3. Freeze configuration and seed derivation before 8-seed confirmation (24 runs).
   Initial confirmation compute ceiling: four wall-clock hours, one training
   process at a time. This is a budget, not a runtime promise. If projected cost
   exceeds it, stop with the estimate; shortening the horizon requires a disclosed
   redesign and new predeclaration. No automatic hyperparameter sweep.
4. One independent raw-data verifier, then lead adjudication and an atomic numbered
   experiment commit under loop/PROTOCOL.md. A negative result ends this hypothesis.

Development can reveal insufficient learning or instrumentation problems. Do not
advance to confirmation until the instrument is credible; disclose all development
changes. Never tune on confirmation seeds. Assign a unique seed namespace and the
next free experiment number at execution time, not in this planning document.

## What the user gets

A local training command; a resume command; saved model/checkpoint and configuration;
raw per-probe probabilities and targets; learning curves; per-seed comparison; an
honest result. Implementation and results live under experiments/, with focused
tests. The requested local observatory below is part of delivery. It observes the
same experiment; it does not introduce another model family or change its rules.

## Model files and complete continuation state

Use the existing safetensors dependency for model weights. Each immutable checkpoint
directory contains:

```text
weights.safetensors    # named GRU/head tensors; no executable objects
config.json            # architecture, encodings, hyperparameters, schema version
metadata.json          # code/data hashes, arm, seed, step, package versions
optimizer.npz          # optimizer array leaves
replay.npz             # retained clips, IDs, scores, tie keys, selector counters
hidden-state.npz       # streaming recurrent state
run-state.json         # stream cursor, update count, RNG/tree reconstruction schema
rng-state.npz          # any RNG array leaves needed for exact continuation
manifest.json          # required filenames, sizes, SHA-256 hashes, complete marker
```

Use explicit typed schemas and NumPy loads with allow_pickle=False. Validate schema,
tensor names/shapes/dtypes, architecture and hashes before loading. Write into a
temporary sibling directory, then atomically rename it into place after validation;
never overwrite a checkpoint. A latest.json pointer may update atomically only
after a complete checkpoint exists. Inference requires the matching model code,
weights and config; loading weights alone does not resume training.

Provide export_model(checkpoint_dir, export_dir) for a portable weights/config/
metadata/manifest subset and load_model(export_dir) for inference. Test identical
predictions after export/load, and identical subsequent CPU training after full
checkpoint/resume. Report cross-device tolerance separately from exact CPU replay.
Retain phase-boundary/final checkpoints and one rolling recovery checkpoint; do not
save a full replay buffer at every UI refresh. Committed scientific outputs follow
the repo's artifact rules. Keep development, confirmation and UI fixtures separate.

## Local observatory

Reference: docs/superpowers/mockups/recall-observatory.html. It is the approved-style
interactive HTML fragment from this conversation, with illustrative data. Reuse its
world-first layout, small isometric chamber, mint/teal palette, restrained inspector,
memory tiles and comparison curves. The scene is an illustration of a sequence task,
not a claim that the model learns navigation. Do not copy hard-coded measurements
or the mockup's fabricated memory records into the real viewer.

Build a small local browser UI with plain HTML/CSS/JavaScript and Python's standard
library HTTP server, bound to 127.0.0.1. No new frontend framework, hosted service,
API credentials, accounts, or network assets. Serve only UI assets and allowlisted
artifacts beneath the selected experiments/outputs run directory. Reject traversal
and paths escaping through symlinks. No arbitrary filesystem browser or execution
endpoint. UI controls replay recorded events; they do not pause or modify training.

Required interactions:

- Select a run and replay arm; distinguish smoke/development/confirmation status.
- Play/pause/step recorded encounters; scrub by training step and A/B/A phase.
  Show the recorded cue/action/forecast and reveal the outcome at the correct time.
- Inspect an actual retained clip and its recorded admission score/selection reason.
- Compare measured per-arm Brier curves with labeled axes and missing-data states.
- List completed checkpoints, export files and full-resume contents; display real
  sizes/hashes and a copyable documented resume command. Never imply a save occurred
  from merely opening this panel. Export itself remains a CLI operation initially.

The runner writes schema-versioned telemetry.json by atomic replacement at most
once per 512 transitions and at completion/error: run ID, stage, arm, seed, step,
phase, status, latest completed encounter, recorded per-head forecasts with their
prediction timestamps, sampled memory IDs/scores, measured curve points and completed
checkpoint manifest paths. Bound telemetry size independently of the raw archive.
Keep raw probabilities/targets for independent grading in separate scientific outputs.
An observer may poll every two seconds without advancing the simulation/model RNG.

Always label whether playback is live-tail, historical, incomplete or demo fixture.
With no output, show 'No training run loaded'; with stale/partial output preserve the
last valid sample and its timestamp. Never fall back to invented curves. Render the
mockup's demo only when explicitly selected and retain its simulated-data badge.
UI verification uses a known fixture plus an actual smoke artifact, checks keyboard
controls, 320px/1024px layouts, and confirms repeated reads leave training state unchanged.

## Prior art and provenance

- [McCallum, 1994](https://proceedings.neurips.cc/paper_files/paper/1994/file/d2ed45a52bc0edfa11c2064e9edee8bf-Paper.pdf): retaining experience for later state identification.
- [Dohare et al., 2024](https://www.nature.com/articles/s41586-024-07711-7): continual neural learning can lose plasticity; this proposal does not isolate that mechanism.
- Earlier project results on supplied history, inactive progress signals, and
  fixed forgetting motivate the study; they do not predict a selective-replay win.
  No novelty claim is made for GRUs, replay, bootstrapping, or disagreement.
