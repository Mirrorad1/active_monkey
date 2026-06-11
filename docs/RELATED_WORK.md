# Related Work and Prior Art

This page maps the active_monkey lab's components to the literature that preceded them.
The purpose is honest placement: where does established work end and where does this
lab's specific contribution begin? The short answer is that nearly every individual
mechanism here has a prior-art lineage; what is ours is the integrated testbed, its
falsifier-first discipline, and its catalog of honest negative results at toy scale.

---

## Summary Table

| Entry | Overlap with this lab | What this lab adds |
|---|---|---|
| Active inference / expected free energy (Friston et al.) | Total framework overlap — the agent architecture, free-energy objective, and variational updates are entirely from this literature | The testbed, benchmark discipline, and negative-results catalog |
| Bayesian model reduction (Friston & Penny) | The `active_loop/structure.py` `bmr_delta_f` function is a direct implementation | Integration into a live probation pipeline; honest negative results on its use as a replay scorer |
| Structure learning in active inference (Smith, Schwartenbeck, Parr, Friston ~2020) | Direct thematic overlap — same problem of expanding a discrete generative model | Measured 57–74% replay-vs-live acceptance disagreement (Exp 144–146) as a pointed datum; Exp 153–154 re-bounded the wall to the unnormalized-footprint evaluation convention |
| Prequential evaluation (Dawid 1984) | Live probation IS prequential acceptance applied to structure growth | Application to discrete mixture components in an active-inference loop |
| Perceptual aliasing / state splitting (McCallum USM/U-Tree) | Closest prior art to the aliased-world growth problem — same structural motive | Online honest acceptance discipline and the probation-revert mechanism |
| HDP-HMM / infinite HMM (Beal et al.; Fox et al. sticky HDP-HMM) | Nonparametric answer to unknown state counts — the general problem this lab addresses at toy scale | The background-floor design's kinship to DP residual mass; online honest acceptance via live probation rather than posterior inference |
| Split-merge mixtures (Ueda et al. SMEM; Jain & Neal) | The greedy-growth valley found in Exp 144–146 is the classical motivation for split-merge moves | Rediscovery of this valley under online honest acceptance in an active-inference substrate; negative results logged honestly |
| Concept drift detection (DDM, ADWIN) | Cousins of the ceiling detector for nonstationary worlds | The ceiling detector is adapted to the active-inference surprise scale, not a classification error rate |
| Forgetting factors in adaptive filtering | Exp 137's decay-counts law is the online analog of exponential forgetting | Direct application to discrete Dirichlet counts in a persistent creature |
| World models / model-based RL | Conceptual kinship — the creature maintains and updates an internal generative model | Deliberately toy scale; the contribution is auditability and the negative-results catalog, not a new MBRL algorithm |

---

## Entry-by-Entry Notes

### Active inference / expected free energy (Friston et al.)

The entire agent architecture — the generative model factored into observation likelihoods
and transition priors, the variational free-energy objective, the belief-update equations,
and the expected free energy used for action selection — is taken directly from the active
inference framework due to Friston and collaborators (Friston et al. 2017, *Neural
Computation*; Friston et al. 2015, *PLOS Computational Biology*). The lab makes no claim
to the theory. The practical implementation layer uses the pymdp library (Heins et al.
2022). The contribution of this lab is the testbed built on top of that framework: the
benchmark problem families, the persistent-creature setup, the predeclared-falsifier
discipline, and — most importantly — the honest catalog of what did not work at toy scale.

### Bayesian model reduction (Friston & Penny)

Bayesian model reduction (Friston & Penny 2011, *PLOS Computational Biology*) is a
technique for comparing a full model against a reduced model using only the sufficient
statistics of the posterior, without refitting. In this lab it is implemented directly as
`bmr_delta_f` in `active_loop/structure.py` and used as a replay-based scoring function
to evaluate proposed structure changes against the creature's accumulated experience
buffer. The negative finding — that replay-scored acceptance disagreed with live outcomes
57–74% of the time (Exp 144–146) — is a pointed datum for the use of BMR as a
structure-acceptance criterion in online agents. The disagreement was later attributed in
part to the unnormalized-footprint evaluation convention rather than to BMR itself
(Exp 152–154), but the measured disagreement is a real and documented effect.

### Structure learning in active inference (Smith, Schwartenbeck, Parr, Friston ~2020)

Smith, Schwartenbeck, Parr, and Friston (~2020) studied structure learning within the
active inference framework, asking how an agent can revise the topology of its generative
model from experience. This lab addresses the same question in a discrete mixture
emission model on a toy gridworld. The most pointed datum this lab contributes to that
literature is the measured 57–74% replay-vs-live acceptance disagreement across Exp
144–146: structure changes that score as improvements on the replay buffer are rejected
by live probation at this rate — meaning the replay heuristic is a systematically
misleading acceptance criterion for the online, embodied setting. Separately, Exp 153–154
re-bounded the Exp 144–146 growth wall: it binds to the unnormalized-footprint evaluation
convention, not to online structure growth per se. Under the normalized-density convention
(conjugate update untouched, only evaluation switched), the creature grew structure and
silenced its own alarm in 24/24 runs (Exp 154).

### Prequential evaluation (Dawid 1984)

Dawid (1984, *Journal of the American Statistical Association*) introduced prequential
(predictive-sequential) evaluation: a model is scored by the sequential log-loss it
accrues on a stream of data, without a held-out test set. The live probation protocol
in this lab — install a candidate structure change provisionally, evaluate it by the
creature's live windowed surprise over the next 400 steps, keep it only if surprise
drops at least 0.1 nats — is prequential acceptance applied to structure growth.
The probation test is validated as honest: 70–86% of kept spawns showed sustained
benefit (Exp 145), meaning live probation correctly distinguishes helpful from harmful
structure changes, unlike the replay-based alternative.

### Perceptual aliasing and state splitting (McCallum's Utile Suffix Memory / U-Tree)

McCallum's Utile Suffix Memory (McCallum 1993) and U-Tree algorithm (McCallum 1996)
addressed the perceptual aliasing problem in reinforcement learning: when an agent
cannot distinguish distinct states from its current observation, it must split its
state representation to resolve the ambiguity. This is the closest prior art to the
aliased-world growth problem that forms the M3-series chapter of this lab (Exp 143–154).
The lab's contribution relative to this line of work is the online honest acceptance
discipline — every proposed split or addition is evaluated by the agent's own lived
surprise via live probation, not by an asymptotic criterion — and the explicit probation-
and-revert mechanism that allows provisional exploration without committing to a change
that proves harmful.

### HDP-HMM / infinite HMM (Beal et al.; Fox et al. sticky HDP-HMM)

The infinite hidden Markov model (Beal et al. 2002) and the sticky HDP-HMM (Fox et al.
2011, *Annals of Applied Statistics*) provide a principled nonparametric answer to the
unknown-state-count problem: place a Dirichlet process prior over the model order and
let the posterior determine how many states are needed. This lab attacks the same problem
with a different prior commitment (finite, discrete, explicitly grown) and a different
acceptance discipline (online live probation rather than posterior inference). The
background-floor design explored in Exp 153 — keeping a permanent broad component as a
low-weight residual while sharp components are added on top — is structurally akin to
the DP's residual mass: probability is never fully committed to any single tight cluster,
preserving coverage while allowing specialization. The background-floor arm was found to
be harmless but insufficient under the unnormalized-footprint convention (Exp 153 Arm A),
consistent with the dilution mechanism: the broad component diluted the sharp pieces'
footprint, preventing them from out-competing the vague cover.

### Split-merge mixtures (Ueda et al. SMEM; Jain & Neal)

Ueda et al.'s Split and Merge EM (SMEM; Ueda et al. 2000, *Neural Networks*) and the
split-merge samplers of Jain & Neal (2004, *Journal of Computational and Graphical
Statistics*) were motivated precisely by the observation that greedy single-component
addition falls into fitness valleys: intermediate models with incomplete coverage are
worse than the original, so no local EM step from the starting point leads to the global
optimum. The greedy-growth valley found in Exp 144–146 is the same phenomenon,
rediscovered under online honest acceptance in an active-inference substrate. All three
of ADD (Exp 144–145), SPLIT (Exp 146), and BACKGROUND-FLOOR (Exp 153 Arm A) failed the
live acceptance test for the same underlying reason: partial coverage loses to complete
vague coverage under log-loss. The batch-jump design (Exp 152–154) — fitting all K
components simultaneously from the replay buffer and probating the complete cover as a
single move — resolves the valley by construction, analogously to a split-merge step
that reaches the global configuration without passing through the valley.

### Concept drift detection (DDM, ADWIN)

Gama et al.'s Drift Detection Method (DDM; Gama et al. 2004) and Bifet & Gavalda's
ADWIN (2007) detect distribution shifts in classification error streams and trigger
model updates. The ceiling detector in this lab — a rolling-window conjunction of mean
and slope on the creature's per-color surprise — is a cousin of these methods, adapted
to the active-inference surprise scale rather than a classification error rate. The
conceptual role is the same: trigger a structural response (growth) when the current
model is no longer adequate. Unlike DDM/ADWIN, the ceiling detector operates on the
agent's predictive surprise, not on an external error signal, and it gates a
probabilistic structure-change rather than a hyperparameter update.

### Forgetting factors in adaptive filtering

Exponential forgetting (a discount factor λ applied to older sufficient statistics,
standard in recursive least squares and Kalman filtering) is the continuous analog of
the decay-counts law derived in Exp 137: with per-observation geometric decay α applied
to Dirichlet counts, old observations lose influence at rate α^t. This law was
rediscovered empirically in the creature's persistent belief store as the natural
mechanism for adapting to nonstationary worlds without full history erasure. The
application to Dirichlet pseudocounts in a discrete active-inference agent is
straightforward, but the exact decay rate and its interaction with the probation window
are specific to this creature's timescale.

### World models and model-based RL (conceptual kinship)

World-model agents (Ha & Schmidhuber 2018, *NeurIPS*; Hafner et al. Dreamer 2019) and
model-based reinforcement learning more broadly share the high-level architecture of this
lab: an agent maintains an internal generative model of its environment, updates it from
experience, and uses it to plan. The kinship is conceptual only. This lab operates at
deliberately auditable toy scale (4×4 gridworlds, ≤16 observation classes, lives of a
few thousand steps), uses a principled variational framework rather than learned neural
world models, and makes no claim that any finding transfers to larger scale. The
contribution is not a new MBRL algorithm; it is the falsifier-disciplined, audited
research process and its catalog of honest negative results.

---

## Closing

No single mechanism in this lab is novel: the active inference framework, Bayesian model
reduction, prequential evaluation, HDP-based model-order selection, split-merge moves,
and concept drift detection are all established contributions. The contribution of
active_monkey is the integrated, falsifier-disciplined, audited toy lab and its
negative-results catalog: a record of what does and does not work when these ideas are
combined in a persistent embodied agent at toy scale, with every verdict predeclared,
every negative result logged honestly, and every disputed verdict checked by a blinded
verifier.
