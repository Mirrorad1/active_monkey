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
  **STATUS: HALTED at Exp 144, RESUMED by human word ("go with m3c", 2026-06-10).**
- **M3c — live-probation growth (the human's chosen resumption).** Per-color alarms +
  round-robin scheduling (growth shared across alarmed colors, not worst-first) +
  live-probation acceptance (provisional install after burn-in; keep iff the color's
  live observed surprise drops >= 0.1 nats over 400 probation steps vs its pre-spawn
  window; revert restores the color's snapshot). FAIL = the surprise arm fails again
  (>= 3/8 seeds in >= 2/3 layouts) — HALT with both designs logged.
  **STATUS: HALTED at Exp 145 (second halt).** The probation test is honest (70-86%
  sustained benefit among keeps) — the MOVE is wrong: a fitness valley separates 1
  component from K (single additions help 1/4 and hurt 3/4 of a color's
  observations; the trap weakens once >=2 components exist). Consult recommends
  M3d = the SPLIT operator under the validated live-probation test.
- **M3d — split (HALTED at Exp 146, third halt).** Split surges like add (+0.64 to
  +1.72 mean deltas; P2 anti-valley signature FAILED). The wall's true shape: under
  log-loss, incomplete tight covers lose to complete loose covers — local growth
  moves judged by honest short-horizon tests cannot reach a K-modal emission.
  PARKED per the predeclared flattening rule, with two named untried cracks
  (batch-jump complete-cover probation; background-floor no-hole mixtures).
  The migration continues at M4 on the human's word.
  **PARK RATIFIED ("continue with m4", 2026-06-10) — M4 is the active rung.**
  **RE-OPENED by human word (2026-06-11): both cracks authorized in sequence —
  M3e batch-jump (Exp 152), then M3f background-floor (Exp 153) if M3e's surprise
  arm fails. Both failing re-confirms the wall with the cracks spent.**
  **WALL RE-BOUNDED at Exp 154 (confirmation POSITIVE, fresh seeds, verifier
  agreed): the Exp 144–146 growth wall binds to the UNNORMALIZED-FOOTPRINT
  EVALUATION convention, not to online structure growth. Under normalized
  predictive evaluation (conjugate update untouched), detector→grow→quiet runs
  end-to-end: 100% jump acceptance, zero alarm events, every color quiet, 24/24.
  The normalized-predictive switch for the creature is a standing consult.**
- **M4 — want + act.** Grounded valence (Exp 26 mechanism: predictability-weighted
  value accumulation, continuous predictive entropy at the posterior mean) and
  value-seeking action (Exp 30 analog on the continuous value field). FAIL = valence
  doesn't track the predictability structure the twin's does, or nav doesn't reach
  preferred regions above chance.
  **M4 SUBSTRATE LIMIT (documented at Exp 147-149):** predictability-grounded
  valence on this substrate preserves ORDERING (Spearman > 0.6 in 23/24 runs) but
  compresses DYNAMIC RANGE (~0.65 vs ~0.84 reliable share) — Gaussian emission
  footprints retain neighbor-overlap entropy that categorical columns do not. Both
  mechanism hypotheses (uncalibrated bar; port infidelity) were tested and refuted;
  the limit is real. Accepted on the human's word; ordering is the load-bearing
  property for the opinion chain.
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

**STATUS: COMPLETE 2026-06-10 at Exp 151.** The chain is demonstrated end-to-end and
nira is committed; the two walls (growth valley, valence range) are documented with
named cracks/limits. Closing synthesis: docs/research/continuous-creature-migration.md.
Closure consult in loop/IDEAS.md; this card re-opens only on an explicit choice there.

**STATUS.** state: flagship-candidate · latest: Exp 154 · depends-on: continuous-substrate · reusable: continuous.py, growth.py, worlds.py, NIW inference patterns · why: ladder complete at Exp 151; growth wall fell at Exp 154 (re-bounded to the footprint convention); the worldview-too-small arc is the flagship candidate · next-falsifiable: full-scale worldview bench under predeclared bars.json (experiments/bench_worldview/), and the standing consult on applying the normalized-predictive switch to nira
