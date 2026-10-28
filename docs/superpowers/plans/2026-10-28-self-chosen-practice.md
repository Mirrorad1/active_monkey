# Self-Chosen Practice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a bounded playroom instrument comparing practice schedulers with the
same learner, goals and interaction budget.

**Architecture:** A fully observed small world feeds a shared transition-count learner
and planner. Scheduler choice is the experimental variable. The learned repertoire
and continuous learner state persist across practice, held-out probes and a rule change.

**Tech Stack:** Python 3.11+, existing NumPy and pytest; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-28-predictive-development-design.md`

## Global Constraints

- Follow the spec's common contracts, repository FROZEN rules and provided-versus-learned distinctions.
- Implement independently of PSR performance using a supplied observable representation.
- Do not claim that curriculum selection invents goals or is equivalent to active inference.
- All schedulers use the same learner, planner, candidate goals, memory and practice budget.
- Only instrument tests and bounded smokes in this initial delivery; formal science uses PROTOCOL.

## Review Focus

- Unvisited goals receive bounded exploration opportunities instead of permanent starvation.
- Irreducible noise must not earn progress solely through random score fluctuations.
- Rule changes update the same learner; unchanged skills remain measurable.
- Evaluation forks cannot leak task outcomes into the practice scheduler.
- Goal/action enumeration is bounded and cannot contain oracle solution sequences.

## Task 1 — small playroom and learner

**Files:** create `playroom.py`, `practice_learner.py` under
`experiments/predictive_development/`; test `tests/test_predictive_development_playroom.py`.

**Interfaces:** `Playroom(seed: int)` with `observe() -> tuple[int, ...]`,
`step(action: int) -> tuple[int, ...]`, `clone() -> Playroom`,
`change_rule(rule_id: str) -> None`; `PracticeLearner(seed: int)` with
`observe(before: tuple, action: int, after: tuple) -> None`,
`choose_action(observation: tuple, goal: tuple, horizon: int) -> int`,
`clone() -> PracticeLearner`, `state_dict() -> dict`.

- [ ] Define a small reversible world: movement among three stations, an interaction
  primitive, a switch controlling access to a second manipulable object, and a noisy
  observation channel independent of actions. Randomize placements and action effects
  by seed. Supply observation fields and primitive action identities explicitly.
- [ ] Test that the second effect requires the first, that all required states are
  reachable by ordinary actions, and that noise is uncontrollable in the generator.
- [ ] Implement bounded transition-count learning with disclosed forgetting/pseudocounts
  and finite-horizon planning; every scheduler shares these exact settings. Separate
  input observations from evaluator-only causal metadata and solutions.
- [ ] Give all arms the same bounded candidate grammar/projections (at most 32 goals,
  planning horizon at most 6). Learn the repertoire of achieved outcomes and useful
  action sequences from experience. Candidate goals must not differ across schedulers
  merely because their histories differ. Mark the grammar as provided, not invented.
- [ ] Test snapshot/clone independence and identical learning for identical trajectories.
- [ ] Run `uv run --python .venv pytest -q tests/test_predictive_development_playroom.py`.

## Task 2 — scheduler comparison

**Files:** create `schedulers.py`; test `tests/test_predictive_development_schedulers.py`.

**Interfaces:** scheduler classes `ProgressScheduler`, `UniformScheduler`,
`CountScheduler`, `InformationGainScheduler`, each with
`select(goals: list[tuple], learner: PracticeLearner) -> tuple`,
`record(goal: tuple, success: float) -> None`, `state_dict() -> dict`.

- [ ] Specify learning progress as the difference between successive non-overlapping
  competence windows (8 attempts each), smoothed across windows; use positive progress
  with a disclosed exploration floor of 0.1. Do not silently substitute raw prediction
  error or absolute surprise for progress. Export all window scores for auditing.
- [ ] Write controlled scheduler tests using constant-zero, constant-one, improving
  and seeded stationary-noise success streams. Check bounded attention, finite scores,
  exploration support and sensitivity to sustained improvement rather than noise level.
- [ ] Implement uniform-goal and inverse-visit-count schedulers on the identical goal
  set, plus model-based expected information gain for the shared count learner.
  Document the exact posterior/entropy approximation and distinguish it from a full
  Bayes-adaptive planner. No privileged access to world transitions.
- [ ] Test deterministic seed handling, empty repertoire behavior and that scheduling
  alone cannot mutate world/learner parameters.
- [ ] Run `uv run --python .venv pytest -q tests/test_predictive_development_schedulers.py`.

## Task 3 — continuous practice, transfer probes and change

**Files:** create `run_practice.py`, `configs/practice-smoke.json`,
`configs/practice-development.json`; extend `tests/test_predictive_development_runners.py`
only after the representation worker releases ownership of that file.

**Interface:** `python -m experiments.predictive_development.run_practice
--config PATH --output PATH [--workers N]` writes raw curriculum traces, competence
probes, interaction counts, checkpoints/hashes, configuration and timing diagnostics.

- [ ] Implement one persistent training life per seed/arm. Practice attempts consume
  ordinary actions; no free reset to a goal's starting state or oracle start placements.
- [ ] At fixed checkpoints, evaluate clones on common held-out composite goals without
  changing the live learner. Use the same probe starts for all arms and charge probe
  interactions separately; do not feed probe outcomes into training or scheduling.
- [ ] Change one documented world interaction in the live life, continue learning,
  and re-probe both affected and unaffected competencies. Log hashes at every boundary.
- [ ] Test no silent reset, no probe leakage, matched training interaction budgets,
  fair candidate goals, and serial/parallel metric equivalence.
- [ ] Add the same 60-second smoke / 600-second projected development runtime guard
  as the representation runner. Do not run a full development/confirmation campaign.
- [ ] Produce one bounded technical smoke and a readable trace showing chosen goals,
  outcomes and measured progress. Label it illustrative, not evidence of superiority.
- [ ] Review the scheduler equations, noise handling, goal priors and intervention
  fairness with a separate reviewer. Run focused tests after fixes and one appropriate
  broader regression check after both plans integrate.

## Task 4 — handback and research readiness

**Files:** `docs/research/predictive-development-implementation.md` (Sol lead owns).

- [ ] Include commands for both smoke runners and the complete focused test suite.
- [ ] Describe all supplied maps/representations/actions/goals/test families and all
  learned quantities. Distinguish fit-from-prefix, online updates, filtering and planning.
- [ ] Record numerical failures, incomplete features, algorithm substitutions and their
  consequences. A toy test passing cannot close a scientific question.
- [ ] Give the next formal predeclaration checklist and proposed fresh seed blocks
  from the spec, leaving unsupported performance claims explicitly unmade.
- [ ] Commit only owned files; preserve unrelated work. No automatic push or merge.

## Sol coordination

Use one scoped worker for the playroom/learner, one for schedulers after their protocol
is fixed, and independent review. Keep at most two implementation workers active at
once so a reviewer slot is available. Serialize shared runner/report edits. The lead
owns integration, test execution, commits and reporting. Do not delegate official
scientific grading to coding workers or rewrite historical claims.
