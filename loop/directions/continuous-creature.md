# direction: continuous-creature (substrate migration — human directive "A", 2026-06-10)

**Question.** Does the RECIPE chain (perceive → learn → want → plan/act → form values →
answer in words) survive porting from the tabular substrate to the continuous one
(`active_loop/continuous.py`), and where exactly does the port bind? The phase picture
(Exp 133–138, `docs/research/problem2-phase-picture.md`) licenses the attempt; this card
runs it under the ratified build guardrails.

**Binding guardrails (adopted from the M4a pattern, recorded in IDEAS.md).**
- Every increment predeclares properties + falsifiers; ANY failed predeclared test
  HALTS the thread for explicit human input.
- mirro's and vela's tabular spines are UNTOUCHED. The continuous creature is a NEW
  species line (clade model): new name, own `creature/state/<name>/` path, continuity
  guard extended to it at birth (rung M6).
- Old-substrate behavior stays bit-invariant: all code is ADDITIVE
  (`active_loop/creature_continuous.py`); the existing suite + continuity test enforce.
- M4a increment 1c stays halted behind its own fence. FROZEN paths untouched.
- All Exp-133–140 build rules bind: log-space categorical filtering, count-decay
  forgetting (keep_mean), three-way verdicts, re-run re-quote, moment-matched NIW.

**The substantive core (faced, not dodged).** The creature's world is ALIASED — several
cells share a color; that aliasing is what makes localisation non-trivial. One NIW
Gaussian per color CANNOT represent an aliased color map (the unimodal limitation,
Exp 134 / Gap 1). The predicted resolution connects the two human directives: the
structure-learning toolkit (Exp 132: ceiling detector + spawn/split/merge + BMR,
"awaiting a world worth growing for") supplies mixture components per color, spawned
when irreducible surprise fires. The aliased continuous world is the world it was
built for. Rung M3 tests exactly this, with failure predeclared as a loggable boundary
in both directions.

**Migration ladder (one rung per iteration; predeclarations at experiment time per
PROTOCOL; the tabular creature on IDENTICAL action/observation streams is the twin
throughout).**

- **M1 — perceive, anchor given.** Continuous place belief (GaussianBelief, s ∈ ℝ²,
  grid embedded in the plane) with KNOWN per-color emission Gaussians on a NON-aliased
  world; motor anchor = known Δ(a) with the wall-clamp handled by a DECLARED
  approximation (moment-matched predict through clamped dynamics — clamping breaks
  linear-Gaussianity; the cost is measured, not hidden). Metrics: localization error +
  held-out obs NLL vs the tabular twin. FAIL = continuous localization worse beyond
  the predeclared band, or the posterior escapes the arena (clamp approximation
  unsound).
- **M2 — learn the map.** NIW-learned emissions under the motor anchor (the Exp 135
  pattern inside the creature's world), non-aliased; place-field formation (Exp 21
  analog) + map error vs the Dirichlet twin. FAIL = no localization-grade map within
  the dose the twin needs (×predeclared factor).
- **M3 — the aliasing wall (the centerpiece).** Unimodal NIW on an ALIASED map:
  PREDICTED to hit an irreducible-surprise ceiling (the Exp 132 detector must fire —
  its first live positive). Then: spawn mixture components per the structure-learning
  Phase-3 rule, selection by strict F decrease on replay. FAIL-A = the detector stays
  silent while the map is provably inadequate (instrument gap). FAIL-B = spawned
  mixtures don't recover localization (the toolkit doesn't compose with the continuous
  substrate — a deep negative, logged as such). Either failure HALTS per the guardrail.
- **M3b — spawn-for-prediction (inserted after Exp 143's finding).** The wall is in
  PREDICTION space, not localization: the detector fires correctly (first live
  positive, 8/8) but the strict-decrease spawn rule is myopic (fresh components
  immediately reverted). M3b: candidates get a burn-in (EM adaptation on the replay
  window) BEFORE scoring; success = predictive-surprise drop + detector quiet (NOT
  localization, already fine); varied aliased layouts; E[Sigma] instrumentation to
  verify the self-regulation mechanism Exp 143 hypothesized. FAIL = burn-in-scored
  spawning still cannot reduce predictive surprise (HALT — the growing toolkit cannot
  feed the alarm it answers).
  **STATUS: HALTED at Exp 144** — the HALT arm fired (3/3 layouts). Diagnosis +
  recommended M3c (per-color alarms, round-robin scheduling, live-probation
  acceptance) in the IDEAS.md consult. The thread waits for the human's word.
- **M4 — want + act.** Grounded valence (Exp 26 mechanism: predictability-weighted
  value accumulation, continuous predictive entropy at the posterior mean) and
  value-seeking action (Exp 30 analog on the continuous value field). FAIL = valence
  doesn't track the predictability structure the twin's does, or nav doesn't reach
  preferred regions above chance.
- **M5 — words + the chain end-to-end.** teach_word / answer_* on the continuous
  creature (taught labels, content self-formed — the RECIPE's last links);
  converse-parity check against the tabular capstone. FAIL = the few-shot word↔concept
  mapping needs more than the taught-label budget the twin needs (×predeclared factor).
- **M6 — birth of the species.** A continuous creature line birthed, named, lived past
  its first committed epochs under `creature/state/<name>/` with biography, save/load
  determinism, and the continuity guard extended. This rung is engineering +
  verification, not hypothesis; its predeclarations are round-trip exactness and
  guard coverage.

**Stop condition.** M1–M6 all green (the chain demonstrated on the continuous
substrate, the new line committed) → closing synthesis + consult. Any HALT → explicit
human word, per the guardrail. Insight flattening across ~3 rungs → say so and consult.
