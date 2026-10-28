# Predictive state and self-chosen practice

Status: research roadmap and authorized implementation specification, 2026-10-28.
The human requested overall plans and implementation by Sol with subagents. This
authorizes building the instruments below; it does not establish any research result.

## Aim

Build two independently testable instruments for a continuously learning creature:

1. Learn useful state from action-conditioned predictions without receiving the
   world's sensory map, movement graph, or latent labels.
2. Choose practice according to improvement in competence, and test whether that
   practice helps on tasks introduced later.

The eventual demonstration is a creature that distinguishes similar-looking things
by their consequences, develops useful skills through practice, and selectively
relearns after one interaction changes. The initial implementation is a small CPU
research harness, not a neural scaling project or a claim of open-ended intelligence.

## Why this is a new question here

Exp 31's recovered script accumulates beliefs using initially random A/B and performs
one parameter update at the end. Its collapse is real; it does not compare all ways
to learn useful predictive state. Exact latent-index recovery also differs from
behavioral sufficiency. Do not edit the historical entry or declare its conclusion
refuted merely because another learner has been implemented.

Exp 138 supplies the true likelihood; Exp 272 supplies cell identities; the abstraction
suite uses supplied latent labels. Exp 272–273 supplies the exploration regime, and
Exp 277 studies a scalar learning response. These are relevant precursors, not tests
of the two complete capabilities proposed here.

## Roadmap and decision gates

### A — representation, first

Use a five-cell ring with aliased observations, then a small randomized family of
controlled finite-state worlds. Every learned arm gets the same continuous training
tape. Compare an observable-operator predictive-state learner, a smoothed finite-history
predictor, and a controlled HMM trained with forward/backward EM and multiple starts.
The true-model reference is an explicitly privileged ceiling.

Score held-out conditional prediction and subsequent control using observable goals.
Never score literal latent-label agreement as the primary endpoint. Demonstrating
unanchored behavioral competence and demonstrating a PSR-specific advantage are
different hypotheses; report them separately. A strong HMM tying or beating PSR is
useful evidence, not a failed instrument.

### B — practice selection, independent of A's scientific outcome

Use a small fully observed playroom with an explicitly supplied sensory representation.
All arms share one transition learner, planner, goal repertoire, memory limit, and
interaction budget. Change only the practice scheduler: learning progress, uniform
goals, visit-count novelty, or model information gain. Include an uncontrollable
noise channel. Later evaluate tasks requiring combinations of effects; change one
interaction rule and measure both adaptation and retained competence.

Initially supply primitive actions and a bounded goal grammar. Learning a curriculum
within that grammar is the claim under test. Do not claim autonomous invention of
the goal representation. Learning progress is an intrinsic control mechanism; do not
rename it expected free energy or assert equivalence to active inference.

### C — integration, conditional and deferred

Only combine A and B after separate measurements establish that each works and its
costs are understood. Use a 2×2 representation × scheduler comparison to separate
representation benefits from curriculum benefits. Representation learning must use
experience collected by its own policy, not an oracle's hidden states. A neural
recurrent predictor is a later comparator, not a prerequisite.

### D — optional alternative, deferred

Dual control: keep perception anchored, alter motor contingencies, and test whether
ordinary actions become task-sensitive calibration experiments. Compare certainty-
equivalent planning, posterior sampling, uncertainty triggers and fixed calibration.
Do not implement D in this initial handoff.

## Common contracts and safeguards

- Use Python and existing NumPy; no new dependencies or package metadata changes.
- Keep new implementation under `experiments/predictive_development/` and focused
  tests under `tests/test_predictive_development_*.py`.
- World latent state is private to the simulator/evaluator. Learned arms receive
  only observations, actions and the declared goal, never an environment object.
- `Tape` contains `observations` of length T+1 and `actions` of length T: action[t]
  connects observation[t] to observation[t+1]. Validate this convention everywhere.
- Seeds control world generation, data generation, fitting and evaluation separately.
  Serialize configuration, seed lineage, data hashes, numerical diagnostics and versions.
- No episode reset during a training life. Fitting from a growing tape is allowed
  and must be distinguished from incremental sufficient-statistic updates.
- Evaluation may use snapshot clones, explicitly labeled as counterfactual controls.
  It must not silently reset or mutate the trained life. No changes to mirro/nira.
- Score predictions before seeing their targets; fit/model-select only on training
  and development data. Final evaluation cannot select a rank, seed or parameter.
- Invalid probabilities must produce diagnostics. If PSR outputs are projected onto
  a simplex, report correction frequency and magnitude, and score the corrected model
  explicitly. Never hide a numerical failure behind unconditional uniform fallbacks.
- Match interaction budgets and model-selection opportunities. Report parameter,
  memory and compute differences; equal interactions do not imply equal compute.
- Bound horizons, candidate sequences, test dictionaries and worker counts. Time a
  smoke run before a larger run. Independent seeds may parallelize, with serial/parallel
  equality checked and memory-aware worker limits.
- Preserve FROZEN paths, all existing edits and historical experiment entries.
- Run Python using `uv run --python .venv ...` from the repository root; an isolated
  worktree can use the established checkout venv explicitly if needed.

## Initial delivery versus scientific campaign

The authorized first delivery is executable A and B harnesses, substantive tests,
an illustrative trace command, reproducible CLI configuration, and a technical
implementation report. Tests/smokes establish instrument behavior only. Do not append
an EXPERIMENTS entry merely because tests pass, and do not run a new full research
campaign while the instrument is still changing.

After instrument review, the next research iteration must reserve the next experiment
number, predeclare one hypothesis in its `experiments/expNN_<slug>.py` wrapper, set
property-level thresholds before confirmation, run fresh seeds, obtain an independent
blinded numerical verdict and commit script/output/log/site together under PROTOCOL.
This roadmap is not that completed predeclaration.

Recommended research sequence:

1. A feasibility: development seeds 101–104, fixed 1,200/4,800 interaction budgets.
   Disclose every tried setting and choose a bounded model-selection protocol.
2. A confirmation: reserve seeds 21001–21016 without inspecting them. Predeclare
   both a behavioral sufficiency bar and a practical prediction/control effect size
   for any PSR-specific claim. Base bars on goal difficulty and ceiling, not observed
   favorable treatment values. If no stable instrument exists, stop at instrument failure.
3. B feasibility: separate development seeds 301–304. Choose equal practice budgets
   with observable learning curves; disclose any ceiling/floor effects.
4. B confirmation: reserve seeds 31001–31016. Predeclare held-out composite-task
   advantage, recovery cost after a change, and tolerated loss on unchanged tasks.
   Uniform-goal ties and attraction to noise are explicit negative possibilities.
5. C only if the separate results support its premise. A negative result does not
   authorize unbounded tuning or a rewritten success criterion.

These seed blocks are proposed reservations; check the repository before formal use.
Pilot outcomes must be logged honestly through the normal experiment workflow if run.

## Success of the software delivery

Both CLIs can run bounded deterministic smoke fixtures and emit raw per-arm data.
The tests catch state leakage, action/observation misalignment, evaluation leakage,
noise-seeking mistakes, silent resets and reproducibility failures. A separate reviewer
checks that the claimed algorithms match their implementations. The report names all
provided priors and any incomplete algorithms. No empirical advantage is promised.

## Primary antecedents

- Littman, Sutton & Singh (2001), Predictive Representations of State:
  https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf
- Wolfe, James & Singh (2005), learning PSRs without reset:
  https://icml.cc/Conferences/2005/proceedings/papers/124_PredictiveState_WolfeEtAl.pdf
- Boots, Siddiqi & Gordon, spectral learning and planning:
  https://arxiv.org/abs/0912.2385
- Balle & Maillard (2017), single-trajectory spectral learning:
  https://proceedings.mlr.press/v70/balle17a.html
- Schmidhuber (1991), Curious Model-Building Control Systems:
  https://people.idsia.ch/~juergen/curioussingapore/curioussingapore.html
- Oudeyer, Kaplan & Hafner (2007), Intrinsic Motivation Systems:
  https://www.pyoudeyer.com/ims.pdf
- Forestier et al. (2017), Intrinsically Motivated Goal Exploration Processes:
  https://arxiv.org/abs/1708.02190

This is application and instrument development based on established methods, not an
algorithmic novelty claim. All research interpretations remain subject to VALIDATION.
