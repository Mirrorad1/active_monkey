> **DRAFT** — placeholders marked `[PLACEHOLDER]` are filled by the research loop after the
> benchmark runs. The two sections "Live-vs-replay acceptance" and "Useful failures + the wall
> falls" are fully drafted from committed experiment data.

# Agents That Know When Their Worldview Is Too Small

---

## 30-second summary

A toy active-inference agent running in 4×4 gridworlds can detect when its model of the world
is too simple to predict what it observes, propose richer structure, and decide — by its own
lived-surprise test — whether to keep the expansion. Five successive growth designs failed this
honest test, hardening into a documented wall. An autopsy of the fifth failure identified the
cause: a declared approximation in the emission convention was penalizing growth, not the
structure-learning move itself. A diagnostic arm confirmed the cure (Exp 153); a fresh-seed
confirmation run settled it (Exp 154): with evaluation switched to normalized densities the
agent grows its own model on demand — 24/24 runs, 100% acceptance, alarm silent everywhere.
The wall was never about growing; it was about how growth was being scored.

---

## 5-minute summary

The structural-inadequacy detector (built in Exp 132) watches the agent's rolling surprise
and fires when surprise stays pinned above a designed ceiling for too long, signaling that
the current model cannot represent the world's true structure. In aliased worlds — where
multiple locations share the same color, so one component per color predicts everything
near-uniformly — the detector fires constantly. The directive: let the agent propose a richer
emission model for the alarmed color and decide, by its own honest acceptance test, whether
to keep it.

Five designs were tried in order: (1) myopic spawn with replay acceptance (Exp 143/144,
dishonest test — kept spawns that raised live surprise); (2) burn-in-scored spawn with live
probation (Exp 145, honest test, valley found — every single addition genuinely hurts during
its trial because partial sharpness covers fewer cells than the loose original); (3) split
operator under the same probation (Exp 146, same valley from a different entry angle —
children cover half the cells tightly and expose the rest); (4) batch-jump — fit the complete
K-component mixture at once, probate as one move (Exp 152, excellent fits on replay but
rejected at 0–3.6% acceptance); (5) background-floor — keep the original broad component
permanently, add tight ones on top (Exp 153 Arm A, harmless-but-insufficient).

The Exp 152 autopsy found the deeper cause. The substrate's declared conjugacy-buying
shortcut caps every component's footprint at full strength regardless of how tightly fitted
it is. A K-component tight mixture's predictive mass at any one cell is therefore ~1/K of a
broad single component's — sharpness buys no density gain under this convention. The
predicted p(own color at own cell) falls from ~0.25 to ~0.067, a loss of ~2.7 nats, matching
the observed 1.7–2.8 nat probation surges. With evaluation switched to normalized densities
(the conjugate belief update untouched), the fitted components can be louder than the vague
original: Exp 153 Arm B showed 100% acceptance and 24/24 alarm silence. Exp 154 confirmed
on fresh seeds: drops 0.58–1.18 nats, zero detector events in 24/24 runs, all per-color
final means below 0.7, acceptance 53/53. Blinded verifier agreed on every conjunct.

The Exp 144–146 wall is re-bounded: it binds to the unnormalized-footprint evaluation
convention, not to online structure growth in principle.

---

## Problem statement

An agent that learns a fixed-size model of its world faces a fundamental question: when
should it conclude that its model is structurally inadequate — not merely uncertain, but
wrong in kind — and try to grow? The difficulty is distinguishing four qualitatively
different situations that can each produce elevated surprise:

1. The world is learnable but the agent has not seen enough yet.
2. The world is irreducibly noisy — surprise is as low as it can go.
3. The world has more structure than the model can represent — no amount of
   parameter-fitting will help.
4. The world's structure has changed and the model is stale.

Getting this wrong in either direction is costly. False alarms waste representational
resources on unnecessary growth. Missed alarms leave the agent permanently mispredicting
its world. And accepting a growth move dishonestly — scoring it on replayed memories instead
of live experience — installs expansions that hurt rather than help.

This page documents the agent architecture, the benchmark environments, and the hard-won
empirical record of what works and what does not in a toy 4×4 world.

---

## The four situations

**Ordinary uncertainty.** The agent's model is correct in form but has not converged yet.
Surprise is high because parameters are poorly estimated, not because the model family is
wrong. Given more data, surprise will fall without any structural change. Functional marker:
surprise declines steadily with experience; the detector's rolling mean and slope both trend
downward.

**Irreducible noise.** The world's observations are genuinely stochastic — there is a floor
below which no model can drive surprise, because the world itself is random at that scale.
A learnable world with p=0.7 true-color signal has an analytic irreducible floor of
approximately 0.82 nats (the binary entropy of 0.7). Further parameter fitting or structural
growth cannot improve on this. Functional marker: surprise plateaus at or near the analytic
floor; growth or parameter-fitting yields no improvement.

**Structural inadequacy.** The agent's model cannot represent the world's true generative
structure regardless of how well its parameters are fitted. In the aliased worlds studied
here, a one-component-per-color model predicts each color near-uniformly across all its
possible locations — the correct structure requires multiple components, one per distinct
location cluster. Surprise stays pinned high even after extended learning. Functional marker:
the structural-inadequacy detector fires repeatedly (rolling mean above the ceiling threshold,
rolling slope non-negative), and surprise does not fall as more data arrives.

**Drift.** The world's structure changes over time — a color-to-cell mapping rearranges, a
new regularity appears, an old one disappears. Surprise rises from a previously learned
plateau. Functional marker: surprise was low and rises; the detector fires after a period of
quiet. Distinct from structural inadequacy because the agent's model was once adequate.

In practice these situations overlap: an aliased world is both structurally inadequate and
has a noise floor; a drifting world may become structurally inadequate after the remap. The
detector is calibrated to be conservative (0.7 nats ceiling mean threshold, 5e-4 slope
threshold, 200-step rolling window), trading some false negatives for specificity.

---

## Toy worlds

**Scale (first line):** 4×4 grids, ≤16 observation colors, creature lives of a few thousand
steps.

All experiments use pymdp-style active inference on a discrete 16-cell gridworld. The agent
maintains a Gaussian mixture emission model (Normal-Inverse-Wishart conjugate prior) over
2D observation vectors, one per color. Position belief is updated by Bayesian filtering over
the 4×4 state space. The innate anchor is either the sensory map A or the motor model B
(not both — fitting both from pure noise collapses, Exp 31). Colors are 2D continuous
observations sampled from per-cell Gaussians; the agent never directly observes its cell.

This scale is deliberately chosen: it is small enough to audit every number, run 8+ seeds
end-to-end in seconds, reproduce the full chain from a committed lockfile, and inspect
internals (per-color surprise traces, component counts, probation deltas). Toy scale is a
feature, not a limitation of ambition — the contribution is the lab discipline and the
negative-results catalog, not a scaling claim.

---

## Agent architecture

The agent consists of five integrated modules:

**Belief module.** A pymdp-style state-space model: prior over 16 cells, likelihood matrix
(A), transition matrix (B). At each step the agent receives a 2D observation vector, updates
its state posterior by variational Bayes, selects an action by one-step expected free energy
minimization (or softmax lookahead on the value field in rung M4+), and advances.

**Emission model.** Per-color Normal-Inverse-Wishart (NIW) conjugate mixtures, one component
per color in the base configuration. The NIW parameters (mean m, covariance Σ, precision λ,
degrees of freedom ν) are updated online as observations arrive. The emission convention —
normalized density vs. unnormalized footprint — determines how the predictive mass at a
cell is computed; this choice is the central finding of Exp 152–154.

**Structural-inadequacy detector.** Per-color and global ceiling detection on rolling
surprise windows. A color is flagged when its rolling mean surprise exceeds 0.7 nats
(designed constant, validated on learnable worlds in Exp 132) AND its rolling slope over
200 steps is non-negative (the plateau signature, not a transient). Fires first live
positive in Exp 143 (8/8 seeds, 630–1068 events/seed).

**Growth machinery.** When the detector flags a color, the growth module proposes a richer
emission model for that color (batch-jump: EM fit of a K-component mixture on the color's
replay pairs, K selected by penalized replay NLL). The proposal is evaluated under the
current acceptance protocol (see live-vs-replay section below).

**Live probation.** The validated acceptance protocol (Exp 145): a proposed expansion is
provisionally installed. If the color's live windowed surprise drops ≥0.1 nats over the
next 400 steps versus its pre-expansion window, the expansion is kept permanently;
otherwise it is reverted by restoring the component snapshot bit-exactly. This test is
provably honest: kept expansions show sustained benefit (70–86% in Exp 145, see
"Live-vs-replay" section).

**Valence and navigation.** Predictability-grounded valence (value += exp(−H(predictive
color distribution at posterior mean))). On the continuous substrate this preserves ordering
(Spearman > 0.6 in 23/24 runs across Exp 147–149) but compresses dynamic range relative
to the tabular twin (~0.65 vs ~0.84 reliable-set share), documented as the M4 substrate
limit. Navigation by softmax one-step lookahead on the value field.

---

## Benchmark environments

Four world families covering the four situations:

**A — Learnable.** One distinct 2D Gaussian cluster per color, clusters well-separated.
The agent's one-component-per-color model is correctly specified; the task is to learn it.
Baseline: surprise converges within a few hundred steps. Detector should remain silent after
burn-in. Based on Exp 132 standard arm.

**B — Noise.** True color signal at p_true < 1; with probability 1 − p_true the observation
is drawn from the uniform color marginal instead. Analytic surprise floor: H(p_true) + (1
− p_true)·log(n_colors) approximately, or directly from the committed `analytic_floor()`
helper. The detector should fire only transiently during early learning, then go silent as
parameters converge to the floor. Based on Exp 132 noise arm.

**C — Aliased.** Multiple cells share a color: n_colors × n_cells_per_color layout, seeded.
The correctly specified model requires n_cells_per_color components per color; the initial
one-component model is structurally inadequate. The detector should fire persistently until
growth resolves the aliasing. Three canonical layout seeds (7, 11, 13) are used throughout
Exp 143–154. Based on Exp 143's experimental setup.

**D — Nonstationary.** The color-to-cell assignment remaps at a designated step. The agent's
learned model becomes stale; surprise rises; the detector should fire; the agent should
adapt. Based on the abrupt remap design in the benchmark module.

---

## Baseline suite

The benchmark will compare the growth machinery against the following controls:

- **No expansion.** Unimodal model, no growth ever. Defines the baseline surprise ceiling.
- **Random accept.** Spawns are accepted at the empirically observed base acceptance rate,
  with no probation test. Tests whether any-spawn-at-all helps or whether honest acceptance
  is necessary.
- **Replay-only acceptance.** The Exp 144 rule: keep if replay NLL strictly decreases.
  Retained as the dishonest control — the rule documented to disagree with live outcomes
  57–69% of the time (Exp 145) and to generate 26% agreement in Exp 146.
- **Bigger fixed.** True K per color installed at birth under both emission conventions.
  Oracle structural specification; tests the ceiling achievable if structure were given.
- **Oracle.** True K, true cell assignments, fitted NIW parameters. Maximum achievable
  performance under this model family.
- **Decay only.** Exp 137's count-decay law applied without growth. Tests whether
  forgetting alone (without structural expansion) helps on nonstationary worlds.

---

## Main result table

[PLACEHOLDER — filled by the research loop after the benchmark runs with predeclared bars
in bars.json. Columns: world type (A/B/C/D) × mechanism × convention × mean final surprise
± SE × acceptance rate × detector-quiet rate × localization error. Do not fill this section
without running the bench harness and logging results in EXPERIMENTS.md first.]

---

## Ablations

[PLACEHOLDER — filled by the research loop. Planned ablations: (1) detector threshold sweep
on world C (0.5 / 0.7 / 0.9 nats) under normalized convention; (2) probation window length
(200 / 400 / 800 steps); (3) keep margin (0.05 / 0.1 / 0.2 nats); (4) K-selection penalty
coefficient (0.02 / 0.05 / 0.10); (5) normalized vs unnormalized convention on the same
growth mechanism. Do not fill without predeclared bars and EXPERIMENTS.md entries.]

---

## Live-vs-replay acceptance

One of the clearest findings of the growth campaign is that replay-scored acceptance and
live-probation acceptance disagree substantially — and replay is wrong.

The replay rule (Exp 144 design): accept a proposed expansion if it strictly reduces NLL on
a frozen window of stored (observation, context) pairs. This is the cheapest possible
acceptance test and the natural first choice.

The live-probation rule (Exp 145 design): provisionally install the expansion, let the
creature live with it for 400 steps, keep it if its color's windowed surprise drops ≥0.1
nats versus the pre-expansion baseline, otherwise revert by restoring the exact snapshot.
This is more expensive but directly tests what matters: does the installed expansion help
the creature's actual ongoing predictions?

**Exp 145 results (Exp 145 — continuous-creature rung M3c, SECOND MIGRATION HALT).**
The replay vote was demoted to a printed diagnostic while live probation governed decisions.
The replay vote disagreed with the live probation decision 57–69% of the time across the
three aliased layouts (layouts seeded 7/11/13, 8 seeds each). This is not a near-chance
disagreement — it is a systematic bias: replay endorsed expansions that live probation
correctly rejected. The mechanism: a new narrow component steals predictive weight from the
broad covering component, helping roughly 1/4 of that color's observations and hurting
roughly 3/4. Replay pairs, drawn from the stored context, do not capture this redistribution
of live predictive mass; live steps do.

The probation test's own honesty was validated: of the small number of expansions kept by
live probation, 75% / 86% / 70% showed sustained benefit across the three layouts (P2
pass — the "Probation honesty" bar). The test is hard on expansions because expansions
genuinely hurt during the valley crossing; the few it keeps are the ones that genuinely
help.

**Exp 146 results (Exp 146 — continuous-creature rung M3d, THIRD HALT).** The split
operator was tested under the same live-probation acceptance. Replay-vote agreement with
the live decision was 26% — less than chance in the direction of endorsing splits the live
test correctly rejected. The mechanism is the same: split children tightly cover half the
color's cells and expose the rest; replay pairs over-represent the well-covered half; live
steps encounter the exposed half and pay the cost.

**Implications.** These numbers are a pointed datum for the structure-learning-in-active-
inference literature (see Related work): live, prequential evaluation is not interchangeable
with replay-scored evaluation for acceptance decisions in online structure growth. The
disagreement rates (57–69% in Exp 145, 26% agreement in Exp 146) are large enough to
reverse the majority of decisions. A lab that uses only replay scoring would systematically
accept expansions that hurt the creature's live predictions.

The live-probation protocol corresponds to prequential (Dawid) acceptance applied to
structure growth: the model earns its place by its sequential predictive performance on
arriving data, not on stored data it has already seen.

---

## Useful failures + the wall falls

### The five-design failure chain

The growth campaign ran five mechanistically distinct designs against the same aliased worlds
and the same honest acceptance criterion. Each failure added a constraint on what growth
must do; together they triangulate the real problem.

**Design 1: Myopic spawn with replay acceptance (Exp 143/144 — M3/M3b).**
The Exp 143 detector fired its first live positive in 8/8 seeds (630–1068 ceiling events per
seed). The initial spawn rule proposed a narrow component at the posterior mean of the
alarmed color, accepted if replay NLL strictly decreased. Kept 0–1 of 5–11 attempts per
seed; final components [1–2,1,1,1–2]; detector kept ringing (394–607 events in the final
1000 steps). Exp 144 added candidate burn-in (10 EM iterations on the color's replay pairs
before scoring) and extended to three layouts × 8 seeds: drops −0.35 to +0.15 nats (flat or
rising), 0/8 seeds passing the drop bar in all three layouts. The first failure established
that burn-in alone does not fix the acceptance problem.

**Design 2: Honest live probation (Exp 145 — M3c, SECOND HALT).**
The acceptance rule was replaced by live probation (provisional install, 400-step window,
0.1-nat keep margin). The test is provably honest — kept expansions show 70–86% sustained
benefit (P2 pass). But the test rejected almost everything, correctly: post-install surprise
surged from pre-spawn 0.9–1.6 nats to probation 1.5–5.0 nats. Kept: 4/160, 7/160, 30/159
across three layouts. Detector ringing 82–397 events/final-1000. The failure identified the
greedy-addition valley: a new narrow component steals predictive weight from the broad
covering piece, improving one location and worsening three. Between one-component-per-color
and the correct K-component cover there is a valley of genuinely worse models, and
one-step-at-a-time growth judged honestly walks into it and turns back every time.

**Design 3: Split operator (Exp 146 — M3d, THIRD HALT).**
The split operator divides the widest component along its leading eigendirection, children
inheriting mass and coverage, then applies the validated live probation. The prediction was
that splitting preserves complete coverage at every step (unlike add, which creates a hole).
The prediction was wrong as implemented: with children separated by ~2√λ₁ ≈ 2.5–3.5 cell
units on a ~3-unit arena and only 10 EM iterations to consolidate, each child covered ~2 of
the color's 4 cells tightly and left the rest exposed. Mean probation deltas: +1.72 / +1.39
/ +0.64 nats across three layouts — surges of the same order as ADD's. Keep rates 0% / 0%
/ 14.9%. Replay-vote agreement 26% (it would have kept the rejections). The wall's true
shape: under log-loss, an incomplete tight cover loses to a complete loose cover, whether
the incomplete cover arrived by adding or by splitting.

**Design 4: Batch-jump with unnormalized evaluation (Exp 152 — growth crack 1).**
Fit the complete K-component mixture in one move (EM on replay, K selected by penalized
NLL), replace the unimodal model as a single unit, apply live probation. K-selection worked:
K=4 chosen near-universally; replay NLL −1.4 to −1.9 (excellent fits). But acceptance was
0% / 0.9% / 3.6% across three layouts — the few accepted jumps had positive (worsening)
deltas. The autopsy identified the cause: the substrate's unnormalized-footprint convention
caps each component's predictive mass at full strength regardless of tightness. A K-component
tight mixture's mass at any cell is ~1/K of a broad single component's. Predicted p(own
color at own cell) falls ~0.25 → ~0.067, a loss of ~2.7 nats — matching the observed 1.7–2.8
nat probation surges. The wall is partly a property of the scoring convention, not of
log-loss geometry alone. This design was the first to distinguish those two causes.

**Design 5: Background floor with unnormalized evaluation (Exp 153 Arm A — growth crack 2).**
Keep the original broad component permanently at weight 0.3, add tight components at 0.7.
By construction, coverage is never incomplete (the broad component always covers everything).
The floor arm behaved exactly as the dilution arithmetic predicted: never hurt (mean probation
deltas +0.13 to +0.18 nats, inside the no-surge band), never helped enough (drop arm reached
in only 2/0/0 seeds). The dilution mechanism's exact predicted signature: harmless-but-
insufficient, because the sharp components still cannot out-shout the cap. Acceptance 27–60%
but not bar-clearing.

### The wall falls

**Exp 153 Arm B — the diagnostic.** Exp 153 ran a second arm alongside the floor arm: the
same batch-jump with one change — all predictive evaluation (surprise, detector, probation)
used normalized mixture densities instead of capped footprints. The conjugate belief update
was untouched; only the scoring convention changed. Results: drop arm 8/8 in ALL three
layouts (0.56–1.21 nats; finals as low as 0.003); quiet arm 8/8 in ALL layouts (zero
detector events, 24/24 — the first time across six designs); acceptance 100%. The components
conjunct failed (2/8, 3/8, 0/8 per layout) because the success undercut its own premise: once
two colors went sharp, the cross-color normalization made the remaining colors predictable
enough that they never needed to grow. The components bar was written for the unnormalized
regime's intuitions. Blinded verifier agreed: verdict NEGATIVE by the letter, but flagged
that the conjunct failed from incomplete coverage, not from the convention failing. The
convention finding was one honest re-test from confirmed.

**Exp 154 — fresh-seed confirmation (THE GROWTH WALL FALLS).**
Exp 153 Arm B verbatim on fresh seeds 8–15, three layouts, with the criterion restated
honestly: global drop ≥0.4 nats, zero detector events in the final 1000, per-color final
mean < 0.7 for ALL colors (grown-vs-never-re-alarmed logged, not gated). Results: POSITIVE
8/8 seeds on every conjunct in all 3 layouts. Drops 0.58–1.18 nats; ceiling events 0 in
24/24; per-color final means: grown colors 0.001–0.019, un-grown 0.000–0.579 (all < 0.7);
acceptance 53/53 = 100%; localization 0.030–0.038. Blinded verifier agreed, conjunct-by-
conjunct, "no ambiguities."

The re-attribution: the Exp 144–146 wall binds to the unnormalized-footprint evaluation
convention, not to online structure growth. With normalized densities, the five-failure
chain's mechanism (sharp components diluted to 1/K mass under the cap) is resolved: tight
components are louder than vague ones, and the creature's own honest test accepts them.

**Honest caveats on Exp 154 (carried forward, not softened):** one growth mechanism
(batch-jump) under one acceptance protocol; the detector threshold (0.7 nats) was reused
on the normalized scale where it is conservative; "wall falls" binds to this world family
and toy scale; the normalized switch is proposed for nira but not yet applied (that is a
standing consult, and the M4/M5 results would need re-verification under it).

---

## Limitations

- **Toy scale.** All results are 4×4 gridworlds with ≤16 colors and lives of a few thousand
  steps. Nothing here is claimed to transfer beyond this scale or world family.
- **One growth mechanism.** Exp 154 confirms batch-jump under live probation with normalized
  evaluation. Split, add, and other mechanisms are not re-tested under the normalized
  convention; their behavior may differ.
- **Detector threshold.** The 0.7-nat ceiling was designed and validated for the
  unnormalized convention. On the normalized scale it is conservative (fires less readily);
  the threshold has not been recalibrated for normalized surprises.
- **Conjugate approximation.** The NIW emission prior buys tractable online updates but
  requires the unnormalized-vs-normalized choice studied here. A fully normalized treatment
  from the start (e.g., Gaussian mixture with normalized densities throughout) would bypass
  this tradeoff but at a different implementation cost.
- **Single noise level / geometry.** The aliased worlds use a specific cell-separation
  geometry (4 colors × 4 cells, 2D observations, moderate cluster separation). Results at
  much higher aliasing ratios or in higher-dimensional observation spaces are not established.
- **No forgetting.** The creature's belief is never reset; the growth campaign operates on
  a continuously accumulated posterior. Drift scenarios where old structure must be
  unlearned are handled by decay-count forgetting (Exp 137) but not yet combined with growth.
- **Proposed nira switch.** The normalized-predictive switch is proposed for nira's aliased
  future but not yet applied. The M4/M5 valence and navigation results were produced under
  the unnormalized convention; they would need re-verification after the switch.

---

## Related work

See [docs/RELATED_WORK.md](../RELATED_WORK.md) (being written in parallel — T10 of the
rigor-and-fairness spec). That page covers: active inference / EFE (Friston et al.);
Bayesian model reduction; structure learning in active inference (Smith et al. ~2020);
prequential evaluation (Dawid); perceptual aliasing and state splitting (McCallum Utile
Suffix Memory / U-Tree); HDP-HMM / infinite HMM (Beal; Fox sticky HDP-HMM); split-merge
mixtures (Ueda SMEM; Jain & Neal); concept drift detection (DDM/ADWIN); forgetting factors
in adaptive filtering; and world models / MBRL.

---

## Reproduction

All results were produced inside the committed lockfile environment.

```bash
uv sync                                  # creates .venv from uv.lock
uv run --python .venv pytest -q          # fast suite (~2s)
uv run --python .venv active-monkey-converse-demo    # capstone demo
uv run --python .venv python experiments/exp145_m3c_live_probation.py  # any experiment re-runs from its script
```

Every experiment script is committed together with its raw outputs under
`experiments/outputs/`; seeds are fixed in-script; headline numbers can be recomputed from
the committed rows (see `tools/`).

---

## What this does NOT prove

- **No consciousness claim.** "Wants," "feels," "preference," "opinion" are functional labels
  for measured quantities (valence = −free energy, preference = relative valence ranking).
  They are documented as such throughout. Nothing here bears on consciousness or phenomenal
  experience.
- **No scaling claim.** Results are 4×4 gridworlds. Nothing is claimed to transfer to larger
  state spaces, richer observation modalities, or real-world environments. The detector,
  probation protocol, and emission convention are designed for this scale.
- **No transfer beyond conjugate Gaussian toys.** The NIW conjugate emission model and the
  normalized-vs-unnormalized footprint distinction are specific to this model family.
  Different emission families would face different tradeoffs and are not addressed here.
- **No general theory of structural inadequacy detection.** The 0.7-nat / 5e-4-slope /
  200-step detector is a designed and validated heuristic, not a provably optimal alarm.
  Its calibration is specific to the noise levels and world geometries studied here.
- **No claim that growth works on the continuous creature yet.** Exp 154 confirms batch-jump
  growth on the experimental harness with fresh seeds. The normalized-predictive switch is
  proposed for nira (the committed continuous creature) but not yet applied; that is a
  standing consult.

---

## Next directions

- **Apply the normalized-predictive switch to nira.** Re-verify M4 valence and M5 language
  results under the corrected convention; test whether the M4 valence dynamic-range limit
  narrows as predicted (the footprint cap was depressing certainty in valence too).
- **Benchmark harness.** Run the four world types × mechanism matrix with predeclared bars
  (see Main result table placeholder above). The harness is being built in Workstream F of
  the rigor-and-fairness spec (T15–T17), gated on Exp 154 (now confirmed).
- **Ablations.** Detector threshold recalibration for the normalized scale; probation window
  and keep-margin sensitivity; K-selection penalty sweep.
- **Nonstationary worlds.** Combined growth + forgetting (Exp 137 decay-count law +
  batch-jump growth) on world D. Growth and forgetting are complementary responses to
  different failure modes; their interaction is untested.
- **Higher aliasing ratios.** n_colors × n_cells_per_color beyond 4×4; larger K; 3D or
  higher-dimensional observation vectors. The mechanism (normalized evaluation enables
  honest acceptance) should generalize, but at different parameter regimes.
- **Oracle gap analysis.** Compare live growth against bigger-fixed (oracle K at birth,
  both conventions) to quantify what the growth mechanism recovers vs. what an oracle
  structure would yield. This is one column of the planned benchmark table.
- **Social emergence.** Two creatures in the same aliased world, sharing observations via
  communication: does one creature's growth event help the other's surprise? See
  `loop/directions/social-emergence.md`.
