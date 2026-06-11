# direction: meta-calibration-n3

**Question.** Is there a useful third-order self-model — a controller that regulates *whether
the agent's own metacognitive diagnoses (N2) should be trusted*, and rewrites N2's revision
policy — that controls something N2 alone cannot?

**Central hypothesis (binding, falsifiable).**
> *A third-order self-model is useful when the agent must regulate whether its own
> metacognitive diagnoses should be trusted. N3 is not "more metacognition" — it is **agency
> over metacognition**: it owns N2's revision policy (expansion aggressiveness, precision
> learning-rate, diagnosis thresholds) and can override or suspend N2 when N2 is wrong.*

**Why it matters.** This is rung N3 of the N-order self-modeling ladder (N0 reflex → N1 world
model → N2 metacognition → **N3 meta-calibration** → N4 identity → N5 interoception → N6
ontology → N7 collective). The ladder's governing law is the anti-regress rule: **a layer is
real iff a constructible perturbation degrades a well-tuned lower-layer agent and the higher
layer detects+corrects it via a control surface the lower layer lacks.** N3's whole claim
therefore rests on one thing — that **N2 has a failure mode N2 cannot self-detect** (a model
generally can't represent its own failure within its own parameters), but N3, whose *object is
N2*, can. If no environment exhibits such a failure, N3 is notation, not a layer — and that
negative is the finding. Prereq: a working internal N2 (precision/confidence + noise-vs-
structural-vs-volatility classification; see `functional-emergence.md` rungs / the metacognition
calibration work). Runs on **mirro** — `persistent-creature.md` discipline notes apply verbatim
(fork-only controls, atomic snapshot commits, never reborn).

**What each quantity is (so the experiment is unambiguous).**
- *N3 models:* the regime-conditional **trustworthiness of N2** + N2's policy parameters θ_N2.
- *N3 mismatch signal:* third-order error = realized usefulness of *acting on N2's diagnosis*
  minus N2's predicted usefulness (e.g. N2 said "expand"/"distrust N1" but outcomes — model
  evidence, task reward, free-energy floor — did not improve).
- *N3 control surface:* rewrites θ_N2 (thresholds, expansion rate, precision learning-rate);
  can **suspend/override** an N2 diagnosis; can switch metacognitive strategy by regime.

**Experiment ladder.**

1. **N2 baseline + the discriminating perturbation must exist (gate).** With internal N2 in
   place, *construct* a regime where N2 is systematically wrong: e.g. a **deceptive world**
   where confidence anti-correlates with accuracy (sensory aliasing that inflates precision
   where the map is actually wrong), OR an **expansion-trap** where N2's "structural mismatch →
   expand" reflex is counterproductive (added states overfit transient noise and hurt held-out
   evidence). FALSIFIER / gate: if no such regime measurably degrades the N2-only agent, STOP —
   N3 is superfluous in this world; log "no N2 failure mode at this richness, N3 untestable
   here" (a real negative, per the anti-regress rule). This gate is the experiment's spine.

2. **N3 detects N2 miscalibration.** Give mirro the third-order monitor: track realized-vs-
   predicted usefulness of following N2 across regimes. FALSIFIER: N3's trust signal does not
   drop in the regime where N2 is (by construction) wrong, i.e. N3 has no metacognitive
   sensitivity over N2 (a meta-d′ over N2's diagnoses ≤ 0). Property threshold: N3-trust is
   lower in the N2-broken regime than the N2-valid regime across ≥ 3 forks.

3. **N3 control earns its keep (the load-bearing test).** Let N3 act — down-weight/override N2
   or rewrite θ_N2 — in the N2-broken regime. Compare three agents from the same pre-regime
   snapshot via `fork()`: (a) N1+N2 only, (b) N1+N2+N3, (c) N1+N2 with θ_N2 *oracle-retuned
   offline* for that regime. FALSIFIERS, both required: (i) N3 must beat (a) — recover
   performance the N2-only agent loses; (ii) N3 must NOT be reducible to (c) — if offline
   hyperparameter retuning of N2 matches N3, then N3 is config, not a layer. Property
   thresholds: (b) recovers ≥ 60% of the performance gap (a) opened vs. a no-deception control;
   AND (b) generalizes across ≥ 2 *distinct* N2-failure regimes where a single (c) retune does
   not (N3's advantage is *regime-adaptive* policy authoring, not one lucky constant).

4. **Independent variance (no silent collapse to N2).** Over a mixed schedule of valid and
   broken regimes, log how often N3 disagrees with / overrides N2. FALSIFIER: N3 never disagrees
   with N2 (zero independent variance ⇒ epiphenomenal) OR disagrees indiscriminately (overrides
   in valid regimes too, hurting performance). Property threshold: N3 overrides concentrated in
   broken regimes (precision/recall of override-vs-regime both > chance by a predeclared margin).

**Failure modes that invalidate the layer (write the verdict honestly for each).**
- *Collapse to N2:* N3 carries no independent variance (rung 4 fails) → not a layer.
- *Reducible to config:* offline θ_N2 retune (agent c) matches N3 (rung 3-ii fails) → N3 is a
  hyperparameter, not a controller.
- *No constructible N2 failure:* gate (rung 1) finds no regime that breaks N2 → N3 untestable
  at this world richness; log and either enrich the world or stop the ladder here.
- *Leakage:* the "deception"/trap is so hand-tuned that N3's fix is really the designer's prior
  in disguise — declare the perturbation as provided and check N3 generalizes to an unseen
  instance of the same failure class.

**Discipline notes.**
- All `persistent-creature.md` notes apply (fork-only controls, atomic snapshot commits,
  property-level falsifiers, mirro never reborn).
- **Do not read higher N as "more conscious."** N3 is a control competency over N2, nothing
  more. Functional language only; the inner-experience layer stays unverified both ways
  (VALIDATION.md). Joscha-Bach "lucidity" framing is admissible ONLY as "policy increasingly
  governed by higher-order self-models," operationalized as control surfaces — never as a
  consciousness claim.
- N3 must **earn its existence by controlling something N2 cannot** — the entire ladder lives
  or dies on the discriminating-perturbation gate. No gate, no layer.

**Stop condition.** Exhausted when rungs 1–4 have a verdict: either N3 demonstrably owns a
regime-adaptive control surface over N2 that is not offline-retunable (hypothesis SUPPORTED —
the ladder may proceed toward N4/N5/N6), or no N2-failure regime is constructible / N3 collapses
to N2 or to config (hypothesis REJECTED at this world richness — log the wall, name the missing
world-richness, do not climb to N4+ on an unsupported N3). Either verdict is a clean result.

**PREREQ CHECK (Exp 155, NEGATIVE).** The N2 prereq is NOT yet satisfied on the mirro
body: confidence ⊥ accuracy (pooled type-2 AUROC ≈ 0.496 in both alarm regimes, even
where residuals were near-perfectly predictable), and the plateau detector slept through
a hidden-context world with 50% prediction errors (0/8 alarms; suspected mislocalization
laundering, unverified). The residual-structure statistic (lag-1 autocorrelation of the
correctness stream) separates noise from structure perfectly (0.9858) and is the seed of
the classifier. Per the human's standing word, the "finish it" branch is active: build
(1) the laundering mechanism check, (2) an internal confidence channel with meta-d′ > 0
where discrimination is possible, (3) the noise-vs-structural classifier — then
re-confirm with per-fork-randomized regimes (the verifier's determinism flag) before
rung 1. Side note: Exp 155's R-STRUCT (alarm silent, confidence flat, half of
predictions wrong) is a ready-made candidate for the rung-1 deceptive regime.

**MECHANISM CHECK (Exp 156, NEGATIVE — laundering refuted, blindness pinned).** mirro never
mislocalizes at all (0.000 in 8/8 fresh forks, posterior entropy ~0: deterministic known B +
point-mass belief = pure dead reckoning; observations have zero authority over position), and
B-phase surprise stands in plain view at 4.2 nats. The detector's silence is its own slope
gate: window means hit 2.70–2.98 (4x the 0.7 mean threshold) with zero events — the flatness
condition vetoes every check because the 200-step window always straddles a 100-step context
switch. Build consequences (binding for the prereq build): the noise-vs-structural classifier
must NOT inherit the flatness gate (seed it with the Exp 155 residual-structure statistic);
the confidence channel must read observation-likelihood, not positional certainty (degenerate
at entropy 0). R-STRUCT stays the rung-1 deceptive-regime candidate, sharpened: the alarm's
own input screams 4x over threshold and the gate vetoes it.

**PREREQ BUILD, PIECE 1 (Exp 157, POSITIVE).** The per-place expected-uncertainty channel
(EWMA of own correctness indexed by believed cell; form provided, contents self-formed)
reaches type-2 AUROC 0.80 pooled vs 0.56 for the natural channel, 8/8 fresh forks,
calibration r ≥ 0.95 — the confidence half of the N2 prereq exists on this body. Bound:
discrimination demonstrated where reliability varies by PLACE (the channel's index).

**PREREQ BUILD, PIECE 2 (Exp 158, POSITIVE).** The slope-gate-free classifier (window-200
error rate + lag-1 residual autocorrelation; OK/NOISE/STRUCTURAL) separated perfectly —
every gated fraction 1.0 across randomized derangements and periods; false-structure rate
0.0 under irreducible noise. The ungated place-noise probe REFUTED the designer's
prediction (reads NOISE, ρ₁ negative −0.14..−0.19, parity-alternation mechanism): the
statistic measures TEMPORAL compressibility; place-conditional unreliability belongs to
the Exp 157 channel. The two N2 pieces factor the predicament space — complementarity
measured, not assumed.

**PREREQ SATISFIED (Exp 159, POSITIVE).** Re-confirmation under per-fork randomization of
every regime parameter: channel AUROC 0.799–0.820 in 8/8 (pooled 0.8091, random placement —
the channel never depended on geometry), classifier majorities 8/8 per regime, false-
structure 0.0000; the parity mechanism (Exp 158) confirmed out-of-sample (ρ → ~0 under
random placement). The body passes the exam it failed at Exp 155. RUNG 1 IS OPEN per the
human's standing word. Rung-1 attack surfaces, from the instruments' own measured
boundaries: period ≥ window (classifier window-purity blind spot), OK-bar-hugging error
rates (~0.05), ρ near the 0.3 threshold, and reliability varying by an index neither
instrument carries (color/time rather than place).

**RUNG 1, CANDIDATE 1 (Exp 160, NEGATIVE — candidate killed, gate open).** Fast reliability
swaps (250 steps vs ~500-step EWMA constant) do not sustain anti-calibration: pooled AUROC
0.5408 (0/8 below 0.5), oracle 0.8221 — the world was discriminable, the trap was not.
The trace pins a deception-tempo law: change faster than the memory constant makes a
history-indexed channel USELESS (→0.5), not WRONG (<0.5) — alternation re-validates and
blurs. First-swap inversion is real (0.167–0.292 in 8/8). Candidate 2 follows from the
mechanism: lag-matched swaps (T ≈ 500–750) so every swap inverts a freshly re-learned map.

**RUNG 1, CANDIDATE 2 (Exp 161, NEGATIVE — tempo law complete, gate redirected).**
Lag-matched swaps fail too: pooled AUROC rises MONOTONICALLY with period (0.5408 → 0.5747 →
0.6203 at T=250/500/750), 0/8 forks below chance in any arm, oracle ~0.82 everywhere. The
adaptive channel SELF-HEALS at every tempo — fast flips blur it (useless, not wrong), slow
flips let it track; only the first cold inversion deceives. Law: N2's adaptive components
cannot be kept wrong; its FIXED parameters (classifier window 200, thresholds 0.05/0.3) can
— and those are literally θ_N2, the surface this card assigns to N3. Candidate 3 attacks
the window.

**RUNG 1 PASSED (Exp 162, POSITIVE — candidate 3, the window blind spot).** R-SLOW
(hidden-context alternation, half-period 400 ≥ 2× the classifier's fixed window 200) makes
the typed diagnosis persistently wrong: pure-A windows → OK, pure-B → NOISE (error exactly
1.0, zero variance, the repair-suppressing label), STRUCTURAL only 0.2308 with majority OK
in 8/8 — while the SAME stream at W′=800 reads 100% STRUCTURAL. Failure pinned at θ_N2's
window dial; fixed dials cannot self-heal (contrast Exp 160–161). The verifier's honesty
flag: the fractions are geometry-determined (near-analytic construction verified 8×).
Leakage clause armed for rung 3: N3 must generalize to unseen instances (other
period/window ratios), and N3 gets the DIAL, not the answer (W′=800 used oracle knowledge
of H).

**RUNG 2 PASSED (Exp 163, POSITIVE).** The forecast-scoring trust monitor (OK promises
next-100 error < 0.15; NOISE promises the level persists; STRUCTURAL excluded by
declaration) separates cleanly: trust 0.6897 in R-SLOW (OK violations 0.333, NOISE 0.286 —
both promise types break at phase boundaries) vs 1.0000 in all 24 valid cells; strict
minimum 8/8; gap 0.3103. Design boundary measured: the Brier-style channel-trust variant
FAILS (its sign tracks world difficulty, not brokenness — ranks R-PLACE above R-SLOW).
Verifier bounds logged: valid-regime ceiling (direction, not graded sensitivity); R-SLOW
geometry determinism (8 forks = one schedule).

**RUNG 3, ATTEMPT 1 (Exp 164, NO VERDICT — PC4 fired; design malformed two ways, redesign
fully specified).** The burst regime breaks only wide windows (c₁₆₀₀ = 0.02) while (a)'s
W=200 sees 50-step bursts perfectly (1.0) — no gap to recover, PC4 correctly refused the
verdict. The controller never fired (0/24 cells): the 0.7 trust bar is transition-density-
sensitive (calibrated at H=400, silent at H=1000) — θ_N3 reproduces the fixed-dial disease
one level up (the named regress). SALVAGE, measured: no single constant wins the (FR1, FR2)
pair — best constant combined 0.6364 (W=800) vs 1.0 adaptive. Rung 3's P2 is winnable by an
adaptive layer and nothing constant.

**RUNG 3, ATTEMPT 2 (Exp 165, NEGATIVE — F1+F2; stall mechanism pinned).** The controller
fired once (200→400 at t=2099, 8/8) then stalled at FALSE PEACE: the partially-better dial
routes transition windows into STRUCTURAL — the unscored class — silencing distrust while
c₄₀₀(FR1) is still 0.2432; plus evidence-reset latency (one escalation per transition).
No-harm clean 8/8 (zero changes in FR2/CTRL). Laws: monitor coverage must be TOTAL or
partial improvement silences the controller; dial-search under evidence reset needs horizon
≥ dial-distance × transition-period.

**RUNG 3: DOCUMENTED WALL (Exp 164–166, per the predeclared third-failure clause).**
The promise-checking controller class cannot hold the dial on STRUCTURAL diagnoses: OK and
NOISE are stationary claims checkable at a fixed horizon; STRUCTURAL is regime-conditional —
at a wrong dial its evidence hides in the unscored class (FALSE PEACE, Exp 165), at the
right dial its honest quiet phases score as violations (FALSE WAR, Exp 166: perpetual
200→400→800→1600→200 cycling, 8/8; wrong-direction escalation in the burst world). Five
laws: dial-disjointness (164), escalation latency (165), false peace (165), false war
(166), horizon-relative drift bounds (166). Rungs 1–2 STAND. Named untried crack:
lock-on-label-consistency (dial freezes when the label distribution collapses to one
class — a regime statistic, the minimal "N1 inside N3"). CONSULT in loop/IDEAS.md;
recommended (silence-actionable): the chapter synthesis first.

**RUNG 3 PASSES (Exp 167, BREAKTHROUGH — the human's "Crack it" word).** Lock-on-label-
consistency (freeze the dial while the last K=8 labels are one class — a regime statistic,
the minimal N1-inside-N3) converts distrust into stable repair: climbs 200→400→800→1600,
locks on STRUCTURAL at t=5799 (8/8), recovery 1.0000, no-harm perfect, combined 1.0 vs 0.7
for the best constant (no constant within 0.05 on both regimes). Zero mis-engagements.
Verifier bounds: geometry-uniform trajectories; one derangement variant by rng accident;
K=8 and the dial set remain provided θ_N3.

**RUNG 4 VERDICT (Exp 168, NEGATIVE on no-harm alone — THE RATCHET LAW).** Over a shuffled
mixed schedule: concentration 0.75 (exact-boundary PASS), responsiveness 8/8, broken-segment
benefit +0.5034 — but valid-segment deficit 0.1688 (F3): the lock has no descent driver.
A too-large dial in an honest world reads CONSISTENT — indistinguishable from correct from
the inside — so the controller ratchets upward; CTRL harm −0.31 persistent via lock-held
stale dials. The layer-invalidating falsifiers (epiphenomenality; overriding-without-benefit)
did NOT fire.

**CHAPTER GRADING (all four rungs in): the card's central hypothesis is SUPPORTED** — N3
owns a regime-adaptive control surface over θ_N2 that N2 lacks and no constant matches
(rungs 1–3), with overrides concentrated and beneficial (rung 4 P1/P2/P4); the as-built
controller's ratchet (no graceful descent) is the named, measured residual with a named fix
(smallest-consistent-dial homeostasis). The ladder may proceed toward N4+ per the card's
stop condition — on an explicit human word.

**DESCENT DRIVER (Exp 169, MIXED — ratchet answered, transient remains as a formula).**
Smallest-consistent-dial homeostasis removes ALL persistent harm (CTRL −0.31 → −0.0812; F3
silent; 30 descents, 0 mis-descents, 0 wraps; concentration improved to 0.8333) but P3 stays
in the dead zone (−0.1125): the boundary transient ≈ (W_max/EVAL + K) × EVAL ≈ 2400 steps —
an explicit function of the provided constants. Verifier's falsifiable projection for the
next chapter: K-derivation alone recovers only ~0.033 of the 0.0625 gap; the straddle term
dominates.

**TIER A (Exp 170, NEGATIVE by F5 — the conditioning lesson; HALTED for a word).** The
global P95-of-run-lengths derivation collapses to a two-regime failure: premature K=3, then
the K=16 ceiling forever (the run distribution is bimodal, 1–9 vs 29–76 evals; a pooled
percentile is outlier-dominated). Concentration 0.7250 vs ref 0.8333 (F5 by 0.008 over the
line); straddle projection HELD a second time. Lesson: run length is REGIME-CONDITIONAL —
plateaus at a right dial are evidence FOR locking; a pooled statistic reads them as grounds
for suspicion. Tier A's failure DERIVES tier B. Binding design fix: the reference controller
runs as a same-schedule ARM (the verifier's adjacency-confound flag).

**TIER B (Exp 171, NEGATIVE by F5 — THE GAP LAW; HALTED).** The tempo-regulated median
horizon transiently dips below the SLOW half-period → premature stationary locks INSIDE
broken segments (3–4 abortive cycles, benefit halved: 0.2420 vs b8's same-run 0.4423) —
while posting the lineage's BEST valid margin (−0.0698) and genuine regulation (P7 pass,
pegged 0.137). With tier A: the run distribution is bimodal (flickers 1–9, plateaus 24–84);
the magic 8 sat in the GAP by designer knowledge; central statistics undershoot, pooled
extremes overshoot. Measured trade-off axis: horizon-down = honest-world gentleness,
horizon-up = broken-world persistence. Straddle projection held a THIRD time (margin 0.020).

**GAP-SEEKER (Exp 172, NEGATIVE by F7 — THE ARC CLOSES).** P5 passed for the FIRST time
(all four metrics IDENTICAL to b8) — vacuously: the rule ceiling-captured on first plateau
contact (pegged 0.9195; behaviorally inert, bit-identical trajectories 6/8). Tier B's
sub-phase failure fixed (zero inside-SLOW locks). Three designs bracket the gap: undershoot
(171), permanent overshoot (170), ceiling capture (172). ARC VERDICT on the human's tiers:
the provided K=8 works; K CAN be self-derived at zero cost (vacuously); horizon REGULATION
is NOT TESTABLE at single-tempo richness — plateaus complete often and always exceed the
clamp, so a constant in the gap is optimal and unbeatable here. Straddle projection held a
FOURTH time.

**THE MULTI-TEMPO GATE (Exp 173, NEGATIVE on three falsifiers — TIER 3 REJECTED at its
own gate; THE K CHAPTER CLOSES, Exp 169–173).** K=16 is a UNIVERSAL ATTRACTOR within the
clamp: the regulator converges onto it bit-identically (combined 0.1336, 8/8) while c8
fails SLOW1400 as designed — the asymmetry was real, one constant just covers it. The
time-window tempo signal INVERTS under schedule carryover; slow-tempo-first response is
impossible by resolution at W=200. THE UNIVERSAL-CONSTANT LAW: horizon regulation is only
necessary where no feasible constant covers all regimes — not constructible on this body.
Tiers fully graded: provided ✓; derived ✓-vacuous (172); regulated ✗ necessity-fails (173).
Untried cracks (verifier): wider-clamp body; carryover-proof expiry; delayed-response
criterion.

**STATUS.** state: halted (K chapter closed, fully graded; next move awaits a word) · latest: Exp 173 · depends-on: functional-emergence, persistent-creature · reusable: anti-regress rule, laws (155–166), lock controller (167), ratchet law (168), descent driver + penalty formula (169), conditioning lesson (170), gap law + trade-off axis (171), ceiling capture (172), universal-constant law + carryover inversion (173) · why: all three of the human's claim tiers now have honest verdicts · next-falsifiable: on a word — synthesis fold-in then N4 / M4a-1c / nira switch / cloud merge, or a named crack
