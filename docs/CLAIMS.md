# Claims Ledger

This ledger is the public claim boundary for active_monkey. It does not replace
[EXPERIMENTS.md](../EXPERIMENTS.md); it summarizes what the current evidence supports,
what failed, and what would change the conclusion.

Status meanings:

- **supported**: current toy-scale evidence supports the claim in this setup.
- **negative**: the tested hypothesis failed or a wall was found.
- **provisional**: evidence is useful but scope, instrumentation, or replication remains limited.
- **open**: motivation or next direction, not current evidence.

## C1 - Context Depth, Not Raw Capacity, Was The Early Language Lever

**Plain-English claim:** In the early character toys, memory of recent context mattered
more than simply increasing hidden-state capacity.

**Status:** supported.

**Evidence:** Exp 4-8 in [EXPERIMENTS.md](../EXPERIMENTS.md); recovered scripts and outputs
under [experiments/recovered/](../experiments/recovered/).

**What was measured:** held-out surprise, generated character order, and question-to-answer
recall under first-order, bigram, n-gram, and pair-state active-inference variants.

**What this does NOT show:** natural language understanding, emergent grammar, or a scalable
language model.

**Main caveat:** the tasks are tiny and highly controlled; later long-range binding still failed
without a slower held factor.

**Falsifier / what would change the conclusion:** a capacity-only first-order model matching
the context-depth controls on the same tasks would weaken this claim.

**Next need:** no immediate experiment; this is historical scaffolding for the memory and
hierarchy chapters.

## C2 - Unsupervised Disembodied Topic Structure Collapsed In This Setup

**Plain-English claim:** Fully unsupervised topic-like latent structure did not crystallize
from the disembodied symbol stream used here.

**Status:** negative.

**Evidence:** Exp 9-13 and Exp 16 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[experiments/recovered/exp12_unsupervised_topic_fails.py](../experiments/recovered/exp12_unsupervised_topic_fails.py);
[experiments/recovered/exp16_warmstart_meanfield.py](../experiments/recovered/exp16_warmstart_meanfield.py).

**What was measured:** topic belief, generated answers under different primes, and whether
a slow topic factor differentiated enough to bind subject and predicate.

**What this does NOT show:** that unsupervised structure learning is impossible in general.
It shows a collapse mode for this substrate and approximation.

**Main caveat:** the negative result is tied to the toy symbolic setup and mean-field
factorization.

**Falsifier / what would change the conclusion:** an otherwise comparable unsupervised
symbol-only variant that reliably differentiates topic factors and binds held topics at
test time.

**Next need:** any revival should state the new symmetry-breaking channel up front.

## C3 - Embodiment, Grounding, Continuity, One Anchor, And Taught Labels Enabled The Toy Chain

**Plain-English claim:** The strongest early toy chain required embodied experience,
grounding, continuous registered belief, one structural anchor, and taught labels for
word-to-concept mapping.

**Status:** supported at toy scale.

**Evidence:** Exp 17-35 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[active_loop/cli/converse_demo.py](../active_loop/cli/converse_demo.py); recovered scripts under
[experiments/recovered/](../experiments/recovered/).

**What was measured:** model-learning error, place localization, navigation success,
value/opinion formation from history, action driven by those values, and queryable toy
answers using taught labels.

**What this does NOT show:** sentience, consciousness, emergent grammar, or fully
tabula-rasa structure learning.

**Main caveat:** one innate anchor is explicitly allowed; labels are taught, not invented.

**Falsifier / what would change the conclusion:** a comparable no-anchor, no-label, continuous
embodied run that reaches the same queryable concept/opinion behavior would revise the recipe.

**Next need:** keep long-term conversation ambitions separated from current evidence.

## C4 - Persistent Creature State Made Individual History A Real Experimental Variable

**Plain-English claim:** Once the creature state persisted across sessions, accumulated
history changed later behavior and adaptation in measurable ways.

**Status:** supported, with many toy-specific caveats.

**Evidence:** Exp 45-49, Exp 75-91, and idle-mode audits through Exp 124 in
[EXPERIMENTS.md](../EXPERIMENTS.md); [creature/README.md](../creature/README.md).

**What was measured:** committed creature checkpoints, later preference revision,
forecast errors, value-mass inertia, forgetting windows, and registered depth-law calls.

**What this does NOT show:** subjective experience or biological personality.

**Main caveat:** the "first-person" language means agent-side state variables and history,
not a claim about phenomenology.

**Falsifier / what would change the conclusion:** checkpoint/replay audits showing that
history-dependent effects disappear under controlled counterfactual twins.

**Next need:** a unified AgentState abstraction across the creature and ecology substrates.

## C5 - Live Probation Was The Right Surface For Structure Growth

**Plain-English claim:** Replay-only structure scoring was not enough; live, prequential
probation better captured whether a structure change helped the active creature.

**Status:** supported for the continuous-creature growth chapter.

**Evidence:** Exp 143-154 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[experiments/exp145_m3c_live_probation.py](../experiments/exp145_m3c_live_probation.py);
[experiments/exp154_growth_confirmation.py](../experiments/exp154_growth_confirmation.py);
[experiments/outputs/exp145.txt](../experiments/outputs/exp145.txt);
[experiments/outputs/exp154.txt](../experiments/outputs/exp154.txt);
[docs/research/continuous-creature-migration.md](research/continuous-creature-migration.md).

**What was measured:** replay-vs-live disagreement, kept/reverted structure changes,
windowed surprise, normalized-density evaluation, and fresh-seed confirmation.

**What this does NOT show:** a general solution to structure learning or a new active-inference
algorithm.

**Main caveat:** the Exp 154 positive depended on changing the evaluation convention to
normalized densities; that convention is part of the claim.

**Falsifier / what would change the conclusion:** live probation accepting changes that
later systematically harm prediction under the same gates, or replay-only scoring matching
live outcomes across the documented disagreement cases.

**Next need:** apply the same preflight/probation discipline to future trait and belief-state
growth.

## C6 - The Ecology Substrate Lets The Environment Select, But It Is Still Toy-Scale

**Plain-English claim:** The ecology chapter created a reproducing, multi-generation substrate
where finite resources and homeostatic pressure select traits without an external evaluator
directly awarding the target capability.

**Status:** provisional / supported as infrastructure.

**Evidence:** Exp 194-198 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[ecology/README.md](../ecology/README.md);
[experiments/exp194_n5_homeostatic_population.py](../experiments/exp194_n5_homeostatic_population.py);
[experiments/outputs/exp194_n5_homeostatic_population/](../experiments/outputs/exp194_n5_homeostatic_population/).

**What was measured:** population survival, births, deaths, lineage, resource use, inherited
trait values, senescence effects, maintenance cost, and thermosense intensity.

**What this does NOT show:** biological realism, open-ended evolution, or robust ecological
complexity.

**Main caveat:** several early ecological measures were confounded or horizon-limited and
were corrected in later entries.

**Falsifier / what would change the conclusion:** guard tests or audits showing that target
traits receive direct reward or that inheritance/resource pressure is not actually driving
the measured outcomes.

**Next need:** keep anti-cheat guards close to every new ecological affordance.

## C7 - Gifted Usefulness Was Not Enough For Thermosense Evolvability

**Plain-English claim:** In the thermosense arc, a sensor could be useful when gifted or
globally fitter in bulk and still fail to evolve because small local steps did not pay.

**Status:** negative / supported wall in this setup.

**Evidence:** Exp 197-207 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[docs/research/sense-axis-organ-evolution.md](research/sense-axis-organ-evolution.md);
[sense-evolution.html](../sense-evolution.html);
[experiments/exp199_n5_valley_sweep.py](../experiments/exp199_n5_valley_sweep.py);
[experiments/exp200_n5_foraging_escape.py](../experiments/exp200_n5_foraging_escape.py);
[experiments/exp203_n5_sense_gradient_audit.py](../experiments/exp203_n5_sense_gradient_audit.py);
[experiments/exp206_n5_rotating_niche.py](../experiments/exp206_n5_rotating_niche.py);
[experiments/exp207_corner_grid.py](../experiments/exp207_corner_grid.py).

**What was measured:** newborn thermosense intensity, population survival, monomorphic
fitness optima, pairwise selection coefficients, local gradients, controller/sensor
corner-grid effects, and anti-cheat controls.

**What this does NOT show:** that senses cannot evolve in general. It shows that seven
structurally distinct levers failed on this toy substrate.

**Main caveat:** thermosense is a simplified noisy percept and the policy/controller is
provided or constrained by the current engine.

**Falsifier / what would change the conclusion:** a regime in this substrate where a small
heritable increase in sensor precision at the resident reliably produces a positive local
selection gradient and full evolution follows it to functional values.

**Next need:** use the preflight discipline on Phase 4 active-sensing probes before any
full evolution batch.

## C8 - Passive Hidden-State Memory Did Not Clear The Local-Gradient Bar

**Plain-English claim:** In the Phase 3 hidden-state-memory tests, passive cue integration
and belief persistence were useful when gifted in large jumps, but small local steps did
not robustly beat the resident under fair common-garden preflight.

**Status:** negative / supported wall in this setup.

**Evidence:** Exp 208-209 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[docs/research/local-gradient-wall.md](research/local-gradient-wall.md);
[loop/directions/hidden-state-memory.md](../loop/directions/hidden-state-memory.md);
[experiments/outputs/preflight_memory_rung1/](../experiments/outputs/preflight_memory_rung1/);
[experiments/outputs/preflight_belief_persistence_rung1b/](../experiments/outputs/preflight_belief_persistence_rung1b/).

**What was measured:** local pairwise gradient, invasion from rarity, perfect-percept drift
controls, large-step gifted benefit, and guard checks for the hidden-mode mechanism.

**What this does NOT show:** that memory is solved, useless, or impossible in general. It
shows that passive hidden-state integration did not produce a robust local adaptive
gradient in this toy ecology.

**Main caveat:** the hidden mode, cues, belief estimator, and payoff coupling are provided
substrate. The result is about whether the costed heritable capacity pays locally.

**Falsifier / what would change the conclusion:** a matched hidden-state regime where a
small passive-memory or belief-persistence step beats the resident under drift controls
and passes invasion-from-rarity without reward leakage.

**Next need:** move from passive memory knobs to Phase 4 active sensing: costly probing or
sampling actions whose only benefit is better future action selection.

## C9 - Evolvability Preflight Is Now A Method, Not A Result Claim

**Plain-English claim:** The project should often measure the local gradient or premise
of a proposed evolution before launching full batches.

**Status:** supported as methodology; not itself a biological claim.

**Evidence:** Exp 203 and Exp 207 in [EXPERIMENTS.md](../EXPERIMENTS.md);
[ecology/sense_axis.py](../ecology/sense_axis.py);
[experiments/exp207_corner_grid.py](../experiments/exp207_corner_grid.py).

**What was measured:** local pairwise selection gradient and the sensor-controller
cross-partial premise before a full co-adaptation batch.

**What this does NOT show:** that preflight can replace evolution experiments in all cases.

**Main caveat:** preflight can reject a premise cheaply, but a positive preflight still
needs full-run confirmation.

**Falsifier / what would change the conclusion:** repeated cases where preflight predicts
the wrong sign or misses a full-run effect under matched gates.

**Next need:** apply the method to a costly active-sensing axis, where the binding gate is
still the local pairwise gradient before full evolution.

## C10 - The Lab's Main Public Contribution Is Auditability And Negative Results

**Plain-English claim:** active_monkey's durable value is not a novel base algorithm; it is
an integrated, falsifier-disciplined toy lab with visible null results and traceable claims.

**Status:** supported as repo/process framing.

**Evidence:** [docs/RELATED_WORK.md](RELATED_WORK.md), [loop/VALIDATION.md](../loop/VALIDATION.md),
[loop/PROTOCOL.md](../loop/PROTOCOL.md), [tools/check_docs.py](../tools/check_docs.py), and
the append-only [EXPERIMENTS.md](../EXPERIMENTS.md).

**What was measured:** consistency of experiment parsing, site data, verifier logs,
claim caveats, raw-output presence, and current docs references.

**What this does NOT show:** that every claim is final or externally replicated.

**Main caveat:** this is still a one-person lab with LLM-assisted execution and toy worlds.

**Falsifier / what would change the conclusion:** claims that cannot be traced to scripts,
outputs, caveats, or falsifiers; broken docs links; or null results hidden from the public
record.

**Next need:** keep docs consistency checks and claim-ledger updates part of any public-facing
research PR.
