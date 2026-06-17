# Experiment Index

[EXPERIMENTS.md](../EXPERIMENTS.md) is the append-only lab notebook. This file is a
reader map: it groups the 209 logged experiments into chapters, says what each chapter
changed, and points to evidence without rewriting the historical record.

## Start Here

**For engineers:** read [docs/ARCHITECTURE.md](ARCHITECTURE.md), then
[docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md), then inspect representative tests such as
[tests/test_experiments_parser.py](../tests/test_experiments_parser.py) and
[tests/test_exp206_niche.py](../tests/test_exp206_niche.py).

**For ML / active-inference readers:** read [docs/RELATED_WORK.md](RELATED_WORK.md),
[docs/research/problem2-phase-picture.md](research/problem2-phase-picture.md), and
[docs/research/continuous-creature-migration.md](research/continuous-creature-migration.md).

**For artificial-life readers:** read [docs/research/sense-axis-organ-evolution.md](research/sense-axis-organ-evolution.md),
[sense-evolution.html](../sense-evolution.html), and [ecology/README.md](../ecology/README.md).

**For nontechnical readers:** read [README.md](../README.md), then
[sense-evolution.html](../sense-evolution.html), then skim the public tracker
[journey.html](../journey.html).

## Chapter 1: Early Active-Inference Language Toys (Exp 1-16)

The first chapter built and broke tiny character models. It found that context depth,
not raw state count, was the early lever for ordered text; it also found that a fully
unsupervised topic factor failed to differentiate in the disembodied symbol stream.

**Key experiments:** Exp 1, Exp 4-8, Exp 9-13, Exp 16.

**What changed after this chapter:** the project pivoted from bare symbols toward
grounding, embodiment, and explicit hidden state.

**What failed:** more hidden states alone did not fix order; unsupervised topic emergence
collapsed; mean-field factorization severed the needed cross-factor message.

**Why it matters:** these negatives define the first honest wall and prevent later docs
from claiming language-from-scratch.

**Evidence:** [EXPERIMENTS.md](../EXPERIMENTS.md); recovered scripts and outputs under
[experiments/recovered/](../experiments/recovered/).

## Chapter 2: Embodiment, Valence, Memory, And Hierarchy (Exp 17-40)

The embodied toy chain showed place-like hidden state, learned facts, action, value-like
dispositions, queryable taught labels, and a capstone demo with two differently raised
creatures.

**Key experiments:** Exp 17-21, Exp 23-35, Exp 36-40.

**What changed after this chapter:** the "recipe" became the durable framing:
embodiment + grounding + continuous registered experience + one innate anchor + taught
labels.

**What failed:** learning both the sensory map and motor model from pure noise collapsed
(Exp 31); emergent grammar remained out of scope.

**Why it matters:** this is the toy-scale moonshot chain, with the strongest caveats.

**Evidence:** [active_loop/cli/converse_demo.py](../active_loop/cli/converse_demo.py),
[experiments/recovered/exp35_converse_demo.py](../experiments/recovered/exp35_converse_demo.py),
and [experiments/recovered/outputs/](../experiments/recovered/outputs/).

## Chapter 3: Persistent Creature And Individual History (Exp 41-132)

This long chapter introduced a persistent creature spine and used counterfactual twins,
idle epochs, and registered audits to measure how accumulated history changes later
behavior.

**Key experiments:** Exp 45-49, Exp 54-60, Exp 63-74, Exp 75-91, Exp 92-124, Exp 125-132.

**What changed after this chapter:** individual history became a first-class experimental
variable rather than a resettable episode detail.

**What failed:** several candidate interpretations died by their own falsifiers; the M4a
"talk to it" increment halted twice because learning could not affect the intended pathway.

**Why it matters:** it separates agent-side state and history from observer-side labels.

**Evidence:** [creature/README.md](../creature/README.md), [creature/](../creature/), and
the relevant entries in [EXPERIMENTS.md](../EXPERIMENTS.md).

## Chapter 4: Continuous Substrate And Structure Growth (Exp 133-154)

This chapter moved from tabular assumptions toward a continuous creature substrate and
tested how structure should grow when the current worldview is too small.

**Key experiments:** Exp 133-140, Exp 141-151, Exp 152-154.

**What changed after this chapter:** live probation and normalized-density evaluation
became the honest acceptance surface for growth in this substrate.

**What failed:** replay/BMR scoring disagreed with live outcomes; greedy growth hit a
fitness valley; some "growth wall" readings were later re-bounded to the evaluation
convention rather than online growth itself.

**Why it matters:** it produced the live-probation standard used in later infrastructure
thinking.

**Evidence:** [docs/research/problem2-phase-picture.md](research/problem2-phase-picture.md),
[docs/research/continuous-creature-migration.md](research/continuous-creature-migration.md),
[experiments/exp145_m3c_live_probation.py](../experiments/exp145_m3c_live_probation.py),
[experiments/exp154_growth_confirmation.py](../experiments/exp154_growth_confirmation.py),
[experiments/outputs/exp145_rows.json](../experiments/outputs/exp145_rows.json), and
[experiments/outputs/exp154_rows.json](../experiments/outputs/exp154_rows.json).

## Chapter 5: Meta-Calibration And Identity Control (Exp 155-193)

The N3/N4 chapters tested monitoring, parameter authority, identity commitment, and
post-release retention under toy attack schedules.

**Key experiments:** Exp 155-168, Exp 169-173, Exp 174-183, Exp 184-190, Exp 191-193.

**What changed after this chapter:** several apparent "agency" stories were narrowed to
config, clocks, surfaces, and named laws. A reopened crack closed constructively under
normal tolerance but left explicit edge cases.

**What failed:** identity defense did not become a broad agency-over-identity result;
some controller interpretations collapsed into timing/configuration.

**Why it matters:** it is a discipline chapter: detection, defense, revision, and retention
must be separated before claiming a control layer.

**Evidence:** [docs/research/n3-meta-calibration-chapter.md](research/n3-meta-calibration-chapter.md),
[docs/research/n4-identity-commitment-chapter.md](research/n4-identity-commitment-chapter.md),
and [docs/research/n4-crack-chapter.md](research/n4-crack-chapter.md).

## Chapter 6: Ecology And Costly Sensing (Exp 194-198)

The ecology substrate introduced finite regenerating resources, inherited traits,
homeostatic death, reproduction, senescence, and thermosense as a costed expressed
capability.

**Key experiments:** Exp 194-198.

**What changed after this chapter:** the project reframed from "can a sensor appear" to
"under what conditions does a costly sensor become worth maintaining?"

**What failed:** early complexity-selection readings had horizon and survivor-bias
confounds; thermosense rose only to a low attractor, not a functional organ.

**Why it matters:** it made environment-as-selector concrete enough to test evolvability
instead of gifted usefulness.

**Evidence:** [ecology/README.md](../ecology/README.md),
[experiments/exp194_n5_homeostatic_population.py](../experiments/exp194_n5_homeostatic_population.py),
[experiments/exp197_n5_maintenance_cost.py](../experiments/exp197_n5_maintenance_cost.py),
[experiments/exp198_n5_thermosense_attractor.py](../experiments/exp198_n5_thermosense_attractor.py),
and [experiments/outputs/exp194_n5_homeostatic_population/](../experiments/outputs/exp194_n5_homeostatic_population/).

## Chapter 7: Thermosense Walls And Evolvability Preflight (Exp 199-207)

The sense-evolution sub-arc tested seven structurally distinct ways to make a costed
thermosense organ evolve. It closed negative: in this toy substrate, gifted usefulness
and even a functional bulk optimum were not enough when the local gradient at the resident
was non-positive.

**Key experiments:** Exp 199-207.

**What changed after this chapter:** Evolvability Preflight became a method: measure the
local gradient or load-bearing premise before full evolution batches.

**What failed:** avoidance, foraging, increasing returns, interference competition,
residue false positives, rotating private niches, and sensor-controller co-adaptation.

**Why it matters:** it sharpens the central public claim: meaningful costly traits need
a locally positive path, not just an installed benefit.

**Evidence:** [docs/research/sense-axis-organ-evolution.md](research/sense-axis-organ-evolution.md),
[sense-evolution.html](../sense-evolution.html), [ecology/sense_axis.py](../ecology/sense_axis.py),
[experiments/outputs/exp203.txt](../experiments/outputs/exp203.txt),
[experiments/outputs/exp206_design_audit.json](../experiments/outputs/exp206_design_audit.json),
and [experiments/outputs/exp207.txt](../experiments/outputs/exp207.txt).

## Chapter 8: Phase 3 Hidden-State Memory (Exp 208-209)

Phase 3 asked whether passive hidden-state memory or belief persistence could succeed
where scalar senses did not. It closed negative under Evolvability Preflight: passive
cue integration was useful when gifted big, but small local steps did not robustly beat
the resident under drift-controlled common-garden tests.

**Key experiments:** Exp 208-209.

**What changed after this chapter:** the local-gradient wall generalized from costed
senses to costed information-processing capacity. The next structural test is active
sensing: costly information-gathering actions, not more passive memory knobs.

**What failed:** `memory_horizon` 1->2 and `belief_persistence` 0.50->0.55 both failed
the local-gradient gate. The continuous trait ruled out the main granularity artifact;
the perfect-percept controls showed the residual advantage was drift/noise, not a
robust denoising edge.

**Why it matters:** it prevents the public roadmap from treating memory or passive hidden
state as the obvious next fix. The cleaner bridge toward active inference is whether an
agent can pay to reduce uncertainty before acting.

**Evidence:** [docs/research/local-gradient-wall.md](research/local-gradient-wall.md),
[loop/directions/hidden-state-memory.md](../loop/directions/hidden-state-memory.md),
[experiments/outputs/preflight_memory_rung1/](../experiments/outputs/preflight_memory_rung1/),
and [experiments/outputs/preflight_belief_persistence_rung1b/](../experiments/outputs/preflight_belief_persistence_rung1b/).

## Chapter 9: Meta Tooling And Public Legibility

This is not a separate science chapter. It covers the repo systems that keep the record
auditable: experiment parsing, site data, route cards, validation rules, and docs checks.

**Key files:** [loop/PROTOCOL.md](../loop/PROTOCOL.md),
[loop/VALIDATION.md](../loop/VALIDATION.md),
[loop/META.md](../loop/META.md),
[active_loop/experiments_parser.py](../active_loop/experiments_parser.py),
[active_loop/site_data.py](../active_loop/site_data.py),
[tools/check_docs.py](../tools/check_docs.py), and [tests/](../tests/).

**What changed after this chapter:** public claims, experiment navigation, reproducibility,
and architecture now have separate front doors.

**What failed:** before this PR, a cold reader had to infer the current claim surface from
the whole notebook and site.

**Why it matters:** public legibility should improve traceability, not replace it.
