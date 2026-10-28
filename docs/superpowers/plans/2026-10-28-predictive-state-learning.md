# Predictive State Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small, reproducible comparison of predictive-state, finite-history
and controlled-HMM learners on aliased action–observation streams.

**Architecture:** Separate simulator, observation-only learner protocol, fitting,
planning and evaluation. Train on shared continuous tapes; use frozen model clones
for evaluation. The implementation must expose numerical failure instead of grading
itself as a scientific success.

**Tech Stack:** Python 3.11+, existing NumPy and pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-28-predictive-development-design.md`

## Global Constraints

- Follow every common contract and safeguard in the spec and repository FROZEN rules.
- Work only on the new package, named tests, these plans and the delivery report.
- Do not modify established creatures, historical scripts, research conclusions or dependencies.
- Tests and bounded smokes are authorized; full scientific runs require the formal
  predeclaration/logging workflow described in the spec.
- Sol is the implementation lead at the user's explicit request. Give coding workers
  disjoint file scopes; the lead owns integration and no worker owns scientific verdicts.

## Review Focus

- Empty/short tapes and unseen histories: reject malformed input; use disclosed smoothing.
- Zero/rank-deficient moments: finite diagnostics and explicit model failure, never NaNs hidden as success.
- Latent relabeling: learned inputs and predictions do not depend on simulator indices.
- Evaluation boundaries: targets and holdout observations cannot change fitted parameters.
- Continuous state and reproducibility: a resumed clone matches uninterrupted filtering.

## Task 1 — contracts, worlds and shared evaluation

**Files:** create `experiments/predictive_development/__init__.py`, `contracts.py`,
`worlds.py`, `evaluation.py`; test `tests/test_predictive_development_contracts.py`.

**Interfaces:** `Tape(observations: np.ndarray, actions: np.ndarray)` with validated
T+1/T lengths and integer alphabets; `Predictor` protocol exposing
`fit(tape: Tape) -> None`, `start(observation: int) -> None`,
`predict(action: int) -> np.ndarray`, `observe(action: int, observation: int) -> None`,
`clone() -> Predictor`, `diagnostics() -> dict`. `observe` filters state, and does not
silently refit parameters. Constructors carry alphabet sizes and RNG seed.
`collect_tape(world, actions: np.ndarray) -> Tape`; simulator internals do not cross
the learner boundary. `score_tape(model, tape) -> dict` scores then filters a clone.

- [ ] Write tests for T+1/T alignment, empty/malformed input rejection, normalized
  distributions, training-tape equality across arms and latent permutation invariance.
- [ ] Implement the Exp 31 five-cell ring observation pattern `[0,0,1,0,1]`, an
  all-identical-observation control, and seeded small finite-state world generation.
  Expose only integer observations to learner adapters.
- [ ] Implement teacher-forced held-out log loss, Brier score and per-step records.
  Training and evaluation RNGs are independent; clones preserve trained filter state
  when continuing the same life, and declared fresh evaluation branches are labeled.
- [ ] Implement shared bounded expectimax for an observable target sequence, with
  horizon at most 4 and action tie-breaking seeded. Planner receives only a predictor.
  Evaluate target sequences of length 2–3 and report random-policy and true-model ceilings.
- [ ] Test that scoring does not alter the training model, that a perfect known model
  solves a deterministic fixture, and that impossible observations produce explicit scores.
- [ ] Run `uv run --python .venv pytest -q tests/test_predictive_development_contracts.py`.

## Task 2 — strong learned baselines

**Files:** create `history.py`, `controlled_hmm.py` in the new package;
test `tests/test_predictive_development_baselines.py`.

**Interfaces:** `HistoryPredictor(n_actions, n_observations, max_history=3, alpha=0.5,
seed=0)` and `ControlledHMM(n_actions, n_observations, n_states=5,
n_starts=3, max_iter=40, seed=0)` implement Task 1's protocol.

- [ ] Write tests for unseen-history backoff, action-conditioned effects, no training
  access to latent states, and model-clone independence.
- [ ] Implement a smoothed suffix/backoff action–observation predictor with bounded
  history. Record the memory and context counts.
- [ ] Write small forward/backward tests against brute-force path likelihood; pin
  transition orientation and emission timing to the Tape convention.
- [ ] Implement scaled/log-stable controlled-HMM forward/backward EM, repeated E/M
  iterations, multiple seeded starts chosen by training likelihood, and likelihood
  convergence diagnostics. Do not copy Exp 31's one-update approximation.
- [ ] Test normalized learned matrices, nondecreasing regularized fitting objective
  where applicable, restart determinism, and continuation without refitting/resetting.
- [ ] Run `uv run --python .venv pytest -q tests/test_predictive_development_baselines.py`.

## Task 3 — predictive-state learner

**Files:** create `psr.py`, `ALGORITHMS.md` in the new package;
test `tests/test_predictive_development_psr.py`.

**Interfaces:** `SpectralPSR(n_actions, n_observations, max_history=3,
max_test=3, rank=5, ridge=1e-6, seed=0)` implements Task 1's protocol.

- [ ] Read the primary spectral/no-reset sources from the spec. Document exact
  history/test moments, action-conditioning corrections, SVD projection, observable
  operators, normalization and filtering equations in ALGORITHMS.md before coding.
  Cite the implemented variant; do not relabel an ordinary history table as a PSR.
- [ ] Enumerate the bounded action–observation test family without using the world map.
  Estimate moments from windows of the same continuous tape. Use only the declared
  behavior-policy probabilities for intervention weighting; the first harness uses
  uniform random actions. Reject unsupported policy inputs instead of assuming random.
- [ ] Test analytically known moments/operators on a tiny controlled process, action
  interventions versus observational conditioning, and predictions after filtering.
- [ ] Implement the documented observable-operator learner. Record singular values,
  effective rank, conditioning, normalizer failures, clipping/projection frequency and
  magnitude. If the nominal rank exceeds empirical rank, record the reduced rank.
- [ ] Test rank deficiency, zero-mass events, probability correction accounting,
  deterministic fitting, hidden-label permutation invariance and clone/resume equality.
- [ ] Run `uv run --python .venv pytest -q tests/test_predictive_development_psr.py`.

## Task 4 — comparison CLI and reviewable output

**Files:** create `run_representation.py`, `configs/representation-smoke.json`,
`configs/representation-development.json`; test `tests/test_predictive_development_runners.py`.

**Interface:** `python -m experiments.predictive_development.run_representation
--config PATH --output PATH [--workers N]` writes JSON containing config, seed lineage,
tape hashes, per-arm raw metrics, parameter/memory/runtime diagnostics and status.

- [ ] Build a deterministic small smoke config and a development config with the
  spec's interaction budgets. Keep confirmation seeds out of the default command.
- [ ] Add runtime preflight and reject an over-budget projected batch. Set development
  default projected wall-time ceiling to 600 seconds; smoke ceiling to 60 seconds.
- [ ] Test repeated execution equivalence excluding measured timings, serial/parallel
  metric equivalence, malformed config rejection and explicit numerical failure records.
- [ ] Run the smoke command once into a clearly labeled technical smoke output under
  the new package; do not interpret the numbers as confirmation of a hypothesis.
- [ ] Review all implementations against the cited algorithms and spec. Run
  `uv run --python .venv pytest -q tests/test_predictive_development_*.py` after integration.
- [ ] Commit only owned implementation/test/technical-report files. A future numbered
  research run must atomically include its wrapper, raw output, log and site record.

## Delegation and finish

The Sol lead first fixes the Task 1 contract, then can delegate baseline and PSR
implementations to separate workers. Workers must not edit each other's files or
git history. Use an independent algorithm/code reviewer after integration; send fixes
back to the owner and rerun affected tests. Record evidence and unresolved limitations
in `docs/research/predictive-development-implementation.md`.

Then implement the independent practice plan. Do not combine the two experimental
variables yet. Deliver runnable commands, file links, tests and all unresolved concerns.
