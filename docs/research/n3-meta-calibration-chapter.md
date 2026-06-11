# The N3 meta-calibration chapter — synthesis (Exp 155–166)

**Status:** chapter closed at a documented wall (rung 3), 2026-06-11. Rungs 1–2 of the
N3 ladder PASSED; rung 3 walled for the promise-checking controller class. The consult
with next-step options lives in `loop/IDEAS.md`; the card is
`loop/directions/meta-calibration-n3.md`; the ladder's design doc is
`docs/specs/n-order-self-modeling.md`.

**Honesty header.** Functional control competencies only — higher N is not "more
conscious" (VALIDATION.md). Every instrument form in this chapter is PROVIDED design;
what the creature self-forms is the instruments' contents from its own lived stream.
All experiments ran on forks of mirro; the spine was never touched.

---

## 1. What the chapter asked

The meta-calibration-n3 card's question: is there a useful third-order self-model — a
controller that regulates whether the agent's own metacognitive diagnoses (N2) should
be trusted, and rewrites N2's policy parameters (θ_N2)? The governing anti-regress law:
a layer is real iff a constructible perturbation degrades the lower layer and the
higher layer detects + corrects it via a control surface the lower layer lacks.

## 2. The prereq build (Exp 155–159): N2 had to be built before N3 could be tested

- **Exp 155 (NEGATIVE):** the body's natural channels fail the N2 prereq — confidence
  ⊥ accuracy (pooled type-2 AUROC ≈ 0.496), and the plateau detector slept through a
  hidden-context world with 50% errors (0/8 alarms). The lag-1 residual-structure
  statistic separates noise from structure perfectly (0.9858) — the classifier's seed.
- **Exp 156 (NEGATIVE, mechanisms pinned):** no mislocalization laundering — the
  point-mass dead-reckoning belief gives observations zero authority over position, so
  mismatch stands in plain view as surprise (4.2 nats); the detector's silence is its
  own slope gate (window means 4× threshold, zero events). Law: a plateau detector is
  structurally blind to mismatch with period < window.
- **Exp 157 (POSITIVE):** the per-place expected-uncertainty channel (EWMA of own
  correctness, indexed by believed cell) reaches type-2 AUROC 0.80 pooled vs 0.56 for
  the natural channel — the confidence half of N2 exists.
- **Exp 158 (POSITIVE):** the slope-gate-free OK/NOISE/STRUCTURAL classifier separates
  perfectly across randomized geometries (ρ 0.983 structure vs −0.009 noise). The
  ungated probe refuted the designer: place-noise reads NOISE with negative ρ (parity
  alternation) — the statistic measures TEMPORAL compressibility; place reliability
  belongs to the Exp 157 channel. The two N2 pieces factor the predicament space.
- **Exp 159 (POSITIVE, CONSOLIDATION):** the prereq re-confirmation under per-fork
  randomization of every regime parameter passes everything (channel 0.8091 pooled
  under random placement — geometry-free; classifier majorities 8/8 per regime).
  N2 exists on this body; the gate to rung 1 opened per the human's standing word.

## 3. Rung 1 — the discriminating perturbation (Exp 160–162)

- **Exp 160 (NEGATIVE, candidate 1):** fast reliability swaps deceive the adaptive
  channel exactly once (first-swap AUROC ≈ 0.22) — alternation re-validates the
  learned map and repeated fast swaps blur it to uninformative (0.5408 pooled).
- **Exp 161 (NEGATIVE, candidate 2):** lag-matched swaps fail too; the tempo law
  completes — pooled calibration rises monotonically with period (0.5408 / 0.5747 /
  0.6203 at T=250/500/750). **Deception-tempo law:** an online-adaptive channel cannot
  be persistently deceived at any tempo — fast change makes it useless, never wrong.
  The attackable surface is N2's FIXED parameters (θ_N2's dials).
- **Exp 162 (POSITIVE — RUNG 1 PASSES):** the window blind spot. Hidden-context
  alternation slower than the classifier's fixed 200-step window makes every glance
  context-pure: honest phases read OK, lying phases read irreducible NOISE (the
  repair-suppressing label); STRUCTURAL drops to 0.23 with majority OK — while the
  SAME stream at W′=800 reads 100% STRUCTURAL. The failure is sustained (fixed dials
  cannot self-heal) and pinned at a θ_N2 parameter an N3 could rewrite.

## 4. Rung 2 — N3 detects N2's miscalibration (Exp 163)

- **Exp 163 (POSITIVE — RUNG 2 PASSES):** the forecast-scoring trust monitor. N2's
  labels make implicit promises (OK: errors stay rare; NOISE: this level persists);
  scoring them against the next 100 steps of the creature's own record yields trust
  1.0000 in every valid cell and 0.6897 in the broken regime (gap 0.3103). Design
  boundary measured: the Brier-style alternative FAILS — it tracks world difficulty,
  not diagnostic brokenness. Score the diagnoses' forecasts, not the channel's
  statistics.

## 5. Rung 3 — the load-bearing test WALLS (Exp 164–166)

Three attempts, each killed by its predeclared machinery, each leaving a law:

- **Exp 164 (NO VERDICT, PC4):** the burst regime breaks only wide windows — no gap
  for the W=200 baseline to recover; and the 0.7 trust trigger is transition-density-
  sensitive (silent at H=1000). Salvage, measured: **dial-disjointness** — no single
  window constant wins the (slow, burst) regime pair (best constant 0.64 combined vs
  1.0 adaptive). Rung 3's prize is winnable by an adaptive layer and nothing constant.
- **Exp 165 (NEGATIVE):** the controller fires once then stalls at **false peace** —
  the partially-better dial routes transition windows into STRUCTURAL, the unscored
  class, silencing distrust while the dial is still wrong. Plus **escalation latency**:
  evidence resets on each dial change, so the search advances one step per world
  transition. No-harm was clean (zero thrash in valid regimes).
- **Exp 166 (NO VERDICT by PC1; the wall):** closing the false peace opens **false
  war** — scoring STRUCTURAL's promise ("errors continue") punishes the CORRECT dial
  during the world's honest quiet phases; the dial cycles 200→400→800→1600→200
  forever (8/8) and escalates the wrong way in the burst world. Also: drift bounds are
  horizon-relative (the 0.05 bound was calibrated for 4000-step sessions).

**The wall's shape (the chapter's deepest finding).** OK and NOISE are STATIONARY
claims — checkable by a fixed-horizon promise-checker. STRUCTURAL is a
REGIME-CONDITIONAL claim: its truth lives in statistics conditioned on a hidden rhythm.
A horizon-bound forecast-checker therefore cannot hold a dial on structural diagnoses:
at a wrong dial the evidence hides in the unscored class; at the right dial the honest
quiet phases score as violations. Validating "the world has a hidden rhythm" requires
modeling the rhythm — **N3 needs regime structure of its own, an N1 inside N3**. The
anti-regress law bites from the other side: each layer's controller needs the substrate
the bottom layer has.

## 6. The five controller laws (Exp 164–166)

1. **Dial-disjointness (164):** distinct failure regimes can pull a fixed dial in
   opposite directions; within {200,400,800,1600}, no constant wins both slow-H1000
   and 50-step bursts.
2. **Escalation latency (165):** dial-search under evidence reset converges at ~one
   step per world transition; the horizon must afford dial-distance × transition-period.
3. **False peace (165):** a monitor with an unscored diagnosis class is silenced by
   partial improvement that routes evidence into that class — coverage must be total.
4. **False war (166):** scoring a regime-conditional claim with a stationary forecast
   punishes correct diagnoses during honest quiet phases — total coverage with the
   wrong forecast type destabilizes the controller.
5. **Horizon-relative bounds (166):** instrument preconditions (drift) scale with
   session length and must be declared per horizon.

Also standing from the chapter: the deception-tempo law (160–161), the slope-gate
blindness law (156), the temporal-vs-spatial factoring of the N2 instruments (157–158),
and the forecast-vs-Brier monitor design boundary (163).

## 7. What this says beyond the repo (bridge claims, bounded)

- Fixed-window/fixed-threshold anomaly detectors have constructible, SUSTAINED blind
  spots at perturbation periods beyond their window — and adaptive-estimator components
  self-heal where fixed components cannot. The attack surface of a monitoring stack is
  its constants. (Functional-form; toy scale.)
- Meta-monitors that score their subordinate's predictions can detect miscalibration
  cheaply (rung 2) — but converting detection into stable parameter authority requires
  the meta-layer to model the environment's regime structure, not merely audit
  promises. This is a concrete, mechanistic instance of why "metacognition about
  metacognition" cannot be a thin wrapper. (Functional-form; one body, one world
  family.)

## 8. Open edges

- The named untried crack: **lock-on-label-consistency** — the dial freezes when the
  label distribution collapses to one class (a regime statistic, the minimal "N1
  inside N3"). Requires a human word (consult in IDEAS.md).
- Rung 4 (override concentration / no silent collapse to N2) was never reached.
- Graded sensitivity of the trust monitor (the Exp 163 ceiling), partial-degradation
  regimes, and θ_N3's own regime-sensitivity (the regress, observed at Exp 164) are
  all unmeasured.
- The Exp 162/163 geometry determinism means fraction-valued results in the broken
  regimes are single-realization; behavioral variation enters only through the
  creature's walk.

## 9. Reproduce

Scripts and committed outputs: `experiments/exp155_n2_prereq.py` through
`experiments/exp166_n3_loadbearing3.py` with outputs under `experiments/outputs/`
(exp155–exp166, rows + verdict JSON). Every gated verdict was blind-verified
(PROTOCOL 4.5); entries in `EXPERIMENTS.md` quote committed outputs only.
