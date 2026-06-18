# RESUME — bootstrap a fresh session and continue this work

**What this file is:** the single source of truth to pick this project back up. Drop it (or its
contents) into a new Claude session and it will know the premise, where we are, and exactly how to
continue. Read top to bottom once; everything else is a pointer from here.

---

## 1. The premise (what we're actually trying to do)

Build an **active-inference agent you can talk to that formed its own opinions from lived
experience — never pretrained on having opinions or on what to think.** The honest, FEP-native
framing locked in across this project:

- **Free energy is the reward.** Low surprise = "understanding"; the agent minimizes free energy.
- **Hidden states = meaning.** Valence = −free energy (a thing "feels good" when its consequences
  are predictable). These are functional, not claims of sentience.
- The moonshot is reached **at toy scale** and walls off at a documented research frontier (below).

## 2. The one durable finding

Unsupervised emergence of latent structure from a **disembodied symbol stream collapses**
(symmetric saddle / posterior collapse / non-identifiability / mean-field severs cross-factor
inference). What breaks the symmetry is the **RECIPE**:

> **embodiment + grounding + continuous *registered* experience (belief never reset) + ONE innate
> anchor** (give either the sensory map `A` *or* the motor model `B`; learning BOTH from pure noise
> collapses — Exp 31) + **taught labels** for the few-shot word←→concept mapping.

On that recipe the full chain works: a creature **perceives** (place fields self-organize) →
**learns** facts → **wants** (grounded valence) → **plans + acts** (value-iteration nav) → **forms
its own values** (same architecture + different history ⇒ different opinion, Exp 26) → **acts on
them** → **answers in words** what it thinks (content self-formed, labels taught, Exp 28/34/35).

**Honest ceilings (NOT toy-crackable, genuine research frontiers):** emergent compositional
**grammar / language-from-scratch**, and **fully tabula-rasa** structure (no innate anchor). These
are written up in `open_problem.html`.

As of Exp 41 the recipe's "continuous registered experience" is raised to the program level: a
single persistent creature (`creature/`) accumulates across experiments instead of restarting at
step 0.

## 3. Where we are (state as of Exp 40)

- **M1–M3b built and green.** Inner controller loop, outer autopilot loop, character language
  model, PR-style autopilot. 65 fast tests pass (5 slow integration tests deselected by default).
- **Exp 1–40 done** — full honest log in `EXPERIMENTS.md`. Exp 36–40 were consolidation
  (place-scale to 6×6, value/converse to 6 concepts, integrated stack, noise-robustness,
  opinion-revisability) — all POSITIVE, diminishing insight. **We are in the consolidation phase:
  each new experiment confirms more than it discovers.** Be honest about that with the user.
- **Capstone:** `active-monkey-converse-demo` (`active_loop/cli/converse_demo.py`) — two
  creatures raised differently answer the same questions
  differently. Verified runnable (see §5).

### 3b. Where we are NOW (folded 2026-06-13, state as of Exp 202)

- **Continuous-substrate chapter (Exp 133–140, closed-positive):** the tabular substrate was
  not load-bearing for the collapse finding but IS brittle under out-of-model input; phase
  picture + amortized comparison in `docs/research/problem2-phase-picture.md`.
- **Continuous-creature migration (Exp 141–151, complete):** nira born — the first committed
  continuous-substrate spine (M4 valence-range limit documented; M5 words/converse parity).
- **THE GROWTH WALL FELL (Exp 152–154, BREAKTHROUGH):** detector→grow→quiet end-to-end, 24/24,
  once evaluation switched to normalized densities — the five-design wall belonged to the
  capped-footprint evaluation convention, not to online structure growth.
- **The N2/N3 meta-calibration chapter (Exp 155–168):** the N2 prereq was built (per-place
  expected-uncertainty channel + OK/NOISE/STRUCTURAL classifier) and re-confirmed under
  randomization; N3 rungs 1–3 PASSED (the window-blind-spot regime; the forecast-scoring
  trust monitor; the lock-on-label-consistency controller — first conversion of metacognitive
  distrust into stable, regime-adaptive parameter authority, after a documented five-law
  wall); rung 4 found the ratchet law. **The N3 hypothesis (agency over metacognition) is
  SUPPORTED at toy richness.** Full synthesis: `docs/research/n3-meta-calibration-chapter.md`.
- **The K-endogenization coda (Exp 169–173, closed):** the human's three claim tiers graded —
  the provided lock horizon works; it CAN be self-derived at zero cost (vacuously, by
  constant-convergence); self-REGULATION is rejected at its own gate (THE UNIVERSAL-CONSTANT
  LAW: regulation is only necessary where no feasible constant covers all regimes — not
  constructible on this body). Chapter doc §11.
- **The N4 identity chapter (Exp 174–183, CLOSED-NEGATIVE, graded 2026-06-11):** rung 1
  found a displacement gate (captivity rewrites identity without a layer; no whipsaw —
  overwrite); rung 2 built a real read-only identity monitor (sensitive and specific after
  the per-burst-matched control); rung 3 failed as a layer three ways — write-gating was
  the wrong surface (Exp 181), freeze-gating showed the surface is sufficient but
  timing-limited (Exp 182), and the fast-trigger attempt killed the controller hypothesis
  (Exp 183): the regulated E* concession becomes a surrender schedule, while fixed-H freeze
  arms defend 7/8 and revise within tolerance. **Chapter verdict:** N4 monitoring is real,
  but commitment control is CONFIG, not agency-over-identity — detection without defense;
  defense, where achievable, needs only a stopwatch. Closed on the human's explicit word
  (option (a) of the Exp 183 consult); the seed-229/variable-length crack is DEFERRED,
  logged as a future crack only. Full synthesis:
  `docs/research/n4-identity-commitment-chapter.md`.
- **The identity-n4-crack chapter (Exp 183 addendum + Exp 184–186, ACTIVE — reopened
  on the human's explicit word, 2026-06-12):** the deferred seed-229 crack became a
  full crack hunt. The autopsy pinned the mechanism (the ~75-step detection-head dose
  vs the margin ledger; repetition NOT the discriminant); the exploratory squeeze map
  (184) showed commitment-as-config does not extend across attack-schedule space
  (freeze-spanning law found); the classification + dissolution check (185) dissolved
  most of the map (CALM2600 gap-spanning config) but left a hard core; and the
  fresh-seed confirmation (186, BREAKTHROUGH, blind-verified) confirmed it 6/6 + 3
  more from the variance sample: **NINE attack schedules where no constant can both
  defend and revise, while the oracle defends everywhere — the crack is REAL at this
  body.** Exp 187 (the licensed controller re-test) then CLOSED the crack
  constructively at normal tolerance: INT-C2900 — a stopwatch on the
  CONTINUOUS-pressure clock, reset by every gap — passes both bars in 9/9 cells,
  with the interval law C ∈ (single-burst stretch, tolerance] traced exactly by the
  C-sweep; the sufficient-surface law completes at the concession level (the
  universal-constant law refined, not broken). Residuals (each needs a word): the
  pressure-gated E-form (untested as pre-registered — instrument-fidelity gap), the
  tight-mode core (no surface tried covers where tolerance < a single burst), and
  post-release retention. Exp 188 (the human's single-regulated-controller retest)
  then ANSWERED the reopened question: YES — one online regulated controller
  (REG-TB, no oracle, no cell-specific tuning) passes both bars in 9/9 cells — at
  EXACT defense parity with the constant INT-C2900 (the single-stretch ambiguity
  bound: at fixed-L geometry regulation ties the right clock; its tempo learning
  never bound a decision). Open edges, exactly named: variable-L geometry, the
  tight-tolerance core, post-release retention. THE CHAPTER IS CLOSED
  (closed-positive, 2026-06-12): synthesis at `docs/research/n4-crack-chapter.md`
  (fact-checked, ~46 claims, 1 correction folded). The variable-L edge then got
  its word ("a") and CLOSED as the rung-6 postscript (chapter §8, fact-checked,
  1 correction folded): Exp 189 — NEGATIVE with THE KAPPA-REACH LAW confirmed 6/6
  (REG-TB defends escalating trains no constant survives, E1/E3 8/8, but loses
  the separation on its own revision 5/8 via THE FLICKER TAX: onset pressure-
  flicker resets the continuity clock); Exp 190 — the one authorized repair
  attempt, NEGATIVE AT DESIGN TIME (total-active-time refuted by arithmetic;
  de-assert hysteresis refuted by measurement + a complete counterfactual sweep:
  revision floor h>=36 vs defense ceiling h<=20, 0/19 buildable) — THE
  TIMESCALE-OVERLAP LAW: changed-world onset intermittency (25–2,600 steps)
  spans the attack-gap scale (525–1,175), so the flicker tax is structural to
  the reset surface. The chapter's final shape: regulation ties the constant at
  fixed-L (the ambiguity bound), uniquely defends at variable-L (kappa-reach),
  and cannot revise in time there (the flicker tax, unrepairable on the named
  channels — the timescale overlap). Card CLOSED (2026-06-12); the remaining
  edges (the tight-tolerance core, post-release retention) re-open only on a
  word. Card: `loop/directions/identity-n4-crack.md`.
- **The n4-crack-edges thread (Exp 191-193, ACTIVE-PAUSED, 2026-06-12):** opened on
  the human's word ("Chapter's remaining edges"). Rung 1 (post-release retention)
  produced three honest instrument failures that outline a wall: Exp 191 (NEGATIVE
  - deferral EXISTS: one fully-measured spanning defense surrendered everything at
  release while 10/11 measured seats retained; the post-release RE-FREEZE CYCLE
  discovered), Exp 192 (NEGATIVE - instrument-bounded again at 12,000-step tails:
  THE NON-SETTLING TAIL; H-mech refuted: the surrendering seed had the LARGER
  stored margin - divergence tracks post-release obs composition), Exp 193 (the
  return-home intervention, NEGATIVE - A NULL INTERVENTION: 186/192 sessions
  byte-identical to no-treatment, 0/192 defense changes - THE POSITION NULL: at
  this 25-cell body position is not a causal lever; "locality" re-named STREAM
  COMPOSITION, seed-fated). Baseline displacement durability and oracle retention
  held at full strength throughout. The corrected causal design (the QUIET-CHAMBER
  stream-level intervention) is the posted consult's recommended option; rung 2
  (the tight-tolerance core) waits behind it. PAUSED on the human's word; card:
  `loop/directions/n4-crack-edges.md`.
- **The population-ecology direction (Exp 194–202, ACTIVE — a NEW parallel line on the
  human's word, 2026-06-12):** the human's "N5 population ecology" — a fresh `ecology/`
  substrate (genotype/phenotype/regenerating GridWorld/pluggable HomeostaticPolicy),
  DISTINCT from the locked N0–N7 ladder's N5 (= interoception; this is ladder-N7 territory,
  kept separately numbered). **Exp 194 (MIXED)** — the homeostatic substrate: energy /
  finite regenerating resources / reproduction with inherited mutated traits / homeostatic
  death, environment-as-selector (a verified design invariant); scarcity crashes the
  population 95–100% (the predeclared P5 starvation-fraction was ill-posed → L18).
  **Exp 195 (MIXED)** — senescence / aging as a complexity-scaled SECOND death cause (behind
  `enable_senescence`; OFF == Exp 194 byte-identical): age kills distinct from starvation,
  lifespan shrinks with complexity (variable, non-linear; ρ −0.67..−0.86); MIXED from a
  confounded regime cause-mix + a 600-step selection null, not the mechanism (→ L19).
  **Exp 196 (POSITIVE)** — over a long horizon (5000 steps, cap raised to carrying capacity)
  aging DEPRESSES the standing population's complexity below a matched senescence-OFF control
  (gap 0.09–0.28, 5/5 fresh seeds, emerging only after ~t=2000) — Exp 195's selection null
  was a HORIZON limit; mechanism under-claimed (heritable selection vs survivor bias — a
  newborn-complexity tracker is the named follow-up). **Exp 197 (POSITIVE-MIXED, on branch
  `exp197-maintenance-cost` → PR)** — the human's mechanism REVISION: complexity = EXPRESSED-
  MACHINERY cost (a costed, evolvable THERMOSENSE organ under temperature), NOT a scalar
  "complexity bad" penalty. Temperature ON (treatment) vs OFF (control), with thermal tolerance
  ALSO costed + a drifting comfort zone (no free escapes). Thermosense is RETAINED under
  temperature (living active-fraction ~0.52 vs ~0.06; tolerance stays low → adapts by SENSING)
  but the newborn tracker shows it's survivor-bias-DOMINATED with only a WEAK heritable shift →
  POSITIVE-MIXED, blind-verified 5/5 fresh seeds (PR #32, MERGED to main). Lessons L20 (remove cheap
  escapes before testing whether a costed capability pays its way). **Exp 198 (MIXED, core-positive)**
  — the survivor-bias-FREE follow-up on main: start the organ at intensity 0 (a trait nobody has can't
  be survivor-biased), 2×2 temperature × seeding, fresh seeds {13–17}. CLEAN de-novo heritable
  emergence — from zero, temperature heritably grows the gene-pool sensor (~0.045 vs ~0.029, 5/5),
  CONFIRMING Exp 197's heritable signal is real; it converges to the same low ATTRACTOR from above
  (seeded 0.20) and below (0), and never runs away to a functional organ (a FITNESS VALLEY caps it).
  MIXED only on a predeclaration validity gap (seed-17 seeded arm collapsed to pop 1 → unmeasurable
  NaN → L21), not a runaway. **Exp 199 (NEGATIVE / NEW INSIGHT — a WALL)** — the fitness-valley sweep:
  is the primitive-sensor ceiling FUNDAMENTAL? YES. Under a deck deliberately stacked FOR a functional
  organ (cheap efficient organ, near-perfect info, strong steering, harsh stress), a primitive sensor
  never climbs (all ~0.05, noise sweep FLAT) and a seeded-functional organ (0.50) decays to primitive
  or drives extinction (2/5); high-intensity mutants arise (max 0.4–1.0) but are always culled — a
  genuine selective valley. Mechanism = benefit-SATURATION (a little sensing suffices to reach safety;
  once safe, more organ is pure cost). Scoped to THIS avoidance regime (only noise swept). The pointed
  escape (named): a FUNCTIONAL organ needs a NON-saturating benefit — sensing for FORAGING, not avoidance.
  **Exp 200 (NEGATIVE / NEW INSIGHT — a WALL)** — that foraging escape, TESTED and FAILED: food regen
  concentrated in a DRIFTING thermal band, thermosense steers toward it (FORAGE mode), no temperature
  stress so foraging is the ONLY benefit. The benefit is real and LARGE in isolation — a forced strong
  forager reproduces ~4× more than a no-organ creature — yet INVISIBLE to evolution: across fresh seeds
  {23–27} the gene-pool newborn intensity stays primitive (WIDE ~0.082, NARROW ~0.088), only WEAKLY above
  a useless-thermosense CONTROL (~0.063), never any seed >0.15, nowhere near functional (>0.30). High-
  intensity individuals (max 0.56–0.66) keep arising in every arm including control but never fix. So the
  primitive-sensor ceiling is GENERAL across BOTH avoidance (199) and foraging (200): NO GOLDILOCKS
  GRADIENT — a crude sensor grabs the easy part of any findable benefit, and the marginal payoff of
  precision is too small / too noisy to out-breed its cost. A forced/behavioral benefit test does NOT
  predict evolvability (→ L22). On branch `exp200-foraging-sense` → PR. Card:
  `loop/directions/population-ecology.md`.
  **Exp 201 (NEGATIVE / NEW INSIGHT — a WALL, unanimously blind-verified 3/3 — the human's pick "a": the
  INCREASING-RETURNS escape, TESTED and FAILED).** The one untested escape from the no-Goldilocks-gradient
  ceiling: a regime where precision's marginal value GROWS. A 9-agent design+adversarial-audit workflow
  found three engine confounds defeat most such designs (resource-memory free-ride, creature-id-ordered
  competition, free-trait substitution); the survivor is a MOVING band each creature must privately TRACK.
  exp200 read the band centre for FREE (its fatal flaw); exp201 (`enable_band_staleness`, gated, exp194-200
  byte-identical / hash-verified, full exp200 hash 502e0539 reproduces) replaces it with a per-creature EMA
  tracker whose responsiveness + reading-noise key to thermosense_intensity. RESULT (fresh seeds {33-37},
  6 arms): FAST gene-pool newborn intensity 0.128 (4/5 <0.15, 0/5 >0.30 functional), MEDIUM 0.116, SLOW null
  0.108, SATURATING free-read 0.083, USELESS 0.053; P1+P2 hold, 0 extinct, internal degeneracy clean →
  NEGATIVE. A REAL but SUB-THRESHOLD speed-gated gradient exists (FAST>MEDIUM>SLOW; FAST tracked > SATURATING
  free-read — the FIRST non-flat null in the arc) but never pays: a frozen-precision returns probe shows the
  benefit of precision is CONCAVE (marginal +0.043→+0.018) and dominated by the linear cost, and the moving
  band LOWERS everyone's intake (FAST < SLOW at every p) — so 'increasing-returns geometry' did NOT produce
  increasing RETURNS. NO GOLDILOCKS GRADIENT, a third time. CLAMPED-LR ≈ FAST → the weak climb is genuine
  thermosense, not memory substitution (LR-confound flag triggered but MOOT, no positive). The primitive-
  sensor ceiling is now GENERAL across avoidance (199) + foraging (200) + increasing-returns (201). L23
  (pilot at the full verdict horizon; a transient peak over-states — caught by the v2 full-horizon pilot).
  On branch `exp201-increasing-returns` → PR.
  **Exp 202 (NEGATIVE / NEW INSIGHT — a WALL, unanimously blind-verified AGREE).** The interference-competition
  escape FAILED. COMPETE (depleting band + shuffle neutralising id-order confound) decayed to mean 0.0285 BELOW
  the 0.10 founder at healthy populations (1033-1080), strip_frac=1.00 certifying the test was genuinely
  interference-competitive (unlike exp201's strip≈0). CLAMPED_LR ≈ COMPETE (0.0284) confirms genuine
  thermosense suppression, not resource-memory substitution. NO_SHUFFLE mean 0.130 is higher but only at
  drift-prone collapsed populations (214-461, corr(pop,intensity)=-0.82 — predeclared drift artifact, not
  selection). The primitive-sensor ceiling is now MAXIMALLY GENERAL across FOUR distinct regimes: avoidance
  (199), foraging (200), increasing-returns (201), and real interference competition (202); competition SUPPRESSES
  the organ below founder. On branch exp202-... → PR. AWAITING HUMAN STEER on next direction.
- **The SENSE-EVOLUTION sub-arc (Exp 203–205, the human's 2026-06-13 reframe "what conditions make a
  costed sense EVOLVABLE, not just useful when gifted?"; reusable instrument ecology/sense_axis.py +
  the enable_residue engine + the L25 runtime pre-flight):** all MIXED/NEW INSIGHT, converging on a
  clean wall. **Exp 203** (selection-gradient audit) MEASURED the local gradient instead of running
  another evolution and found the FIRST positive local gradient in the arc (band-staleness FORAGE,
  0.10→0.15 wins 7/8) — but purely competitive + weak, one seed short of POSITIVE. **Exp 204**
  (residue / false-positive bridge — precision AVOIDS a costly mistake instead of grabbing more food)
  created the FIRST functional MONOMORPHIC optimum in the arc (N* h*=0.60) but it was un-earnable;
  NO_VERDICT because the false-positive cost collapsed 2/5 populations. **Exp 205** (survivable-loss
  sweep) RESOLVED that NO_VERDICT and REFUTED its own mechanism hypothesis: at survivable losses
  (0.8/1.2/1.5, pops ≥4/5 valid) the monomorphic optimum IS functional (h*=0.60) yet evolution STILL
  stays primitive (0/5 functional, local pairwise ≤0) — so the FITNESS VALLEY, not demographic
  collapse, is the SOLE binding barrier (the L22 forced-vs-evolvable gap in its purest form: a
  functional, survivable, bulk-fitter optimum still un-evolvable because small steps don't pay). Six
  experiments (199–205) now converge: no costed sense becomes a functional organ at this substrate.
  Post-205 CONSULT live in loop/IDEAS.md (206 barcode-niches = the last structurally-distinct escape /
  207 controller / accept + synthesize). Card: loop/directions/population-ecology.md.
- **The Evolvability Preflight + Phase 3 (Exp 208–209, CLOSED-NEGATIVE, 2026-06-14):** the sense-arc
  instruments were generalised into a reusable, trait-agnostic **Evolvability Preflight** framework
  (`ecology/evolvability/`, PRs #46/#49) that measures the binding LOCAL gradient cheaply before any full
  batch. Phase 3 then asked whether INFORMATION-PROCESSING capacity evolves where senses didn't: a gated
  hidden-state-mode substrate (`enable_hidden_mode`) + the first non-thermosense traits. **Exp 208**
  (memory_horizon 1→2) and **Exp 209** (continuous belief_persistence ρ 0.5→0.55, ruling out a
  granularity artifact) are BOTH FAIL_LOCAL_GRADIENT — the local gradient is a coin-flip vs the
  perfect-percept drift control, while large gifted steps DO pay (mechanism live). The local-gradient wall
  GENERALISES from scalar senses to memory/inference. Synthesis: `docs/research/local-gradient-wall.md`;
  card `loop/directions/hidden-state-memory.md` (closed-negative).
- **Phase 4 / Rung 3 — active sensing (Exp 210, CLOSED-NEGATIVE, 2026-06-14; human `/lab` directive):**
  the pre-active-inference bridge — does a costed information-gathering ACTION (pay to probe extra cues
  before acting) pay locally where passive capacity didn't? New gated mechanism (`enable_active_sensing` +
  heritable `information_sampling_rate` + probe; byte-identical OFF, golden-hash-guarded; `tests/test_active_sensing.py`).
  **FAIL_LOCAL_GRADIENT (blind-verified).** Two disclosed validity fixes the pilot forced: (1) DRIFT is a
  population-size problem — at cap-50 the common garden is drift-dominated, so raised carrying capacity to
  250 (pops ~950) and read the drift-robust selection slope; (2) calibrate the cost to the empirical benefit
  ceiling (~0.034 energy/step) — fair `probe_cost` 0.01. Result (FRESH seeds 50–65): the small probing
  mutant (rate 0→0.10) is neutral (7/16, slope ≈ 0) and LOSES to a pure-cost perfect-percept control;
  invasion 0/16; mechanism LIVE (gifted probing cuts wrong-cell occupancy 0.40→0.35); memory re-tested at
  the same cap-250 regime is also flat ⇒ the wall is NOT a drift artifact. **Theory B: staleness was not the
  killer** — the local-gradient wall now spans scalar senses (199–207), passive memory/inference (208–209),
  and ACTIVE information-seeking (210). The one named untried lever: an UNCERTAINTY-GATED probe (true active
  inference), re-opens only on a human word. New lessons L29 (drift = pop-size, not cost) + L30 (calibrate a
  costed action's cost to its benefit ceiling).
- **Phase 4 / Rung 4 — uncertainty-gated active sensing (Exp 211, CLOSED-NEGATIVE, 2026-06-15; human
  `/lab`):** the named lever — probe only when the creature's own action margin (|single-cue belief−0.5|)
  is ambiguous. FAIL_LOCAL_GRADIENT (blind-verified): gating WORKS when imposed (beats budget-matched
  random, enriched, pivotal) but does NOT beat fixed-rate (single cue often confidently WRONG, invisible
  to the margin gate), benefit ceiling ~0, flat even at zero cost. Wasted-budget hypothesis REFUTED.
  New lesson L32 (works-when-imposed ≠ has a benefit ceiling). Active sensing CLOSED across fixed-rate + gated.
- **The evolvability-geometry direction (Exp 212–213, ACTIVE — CONVERGING-NEGATIVE, human's word
  2026-06-15):** reframes the wall — is the blocker the capability or the search GEOMETRY?
  `loop/directions/evolvability-geometry.md`. **Exp 212 (NO_HIGHER_REGION)** — a landscape assay of
  active-sensing N*(rate) is monotone but +1.5% at full probing, NO valley ⇒ small-benefit wall, not
  geometry. **Exp 213 (GEOMETRY_INDEPENDENT_WALL)** — an affordance audit (SMOOTH graded vs DISCRETE
  high-stakes eat-vs-skip at matched cost) found no affordance locally evolvable and the discrete geometry
  no steeper ⇒ payoff geometry does not change evolvability. Both rungs: the blocker is benefit MAGNITUDE,
  not payoff shape; mutation-geometry/standing-variation (Rung 2/3) have no valley to cross. DECISION POINT.
- **M4a "talk to it" thread (Exp 125/127/128/214/215/216/217 — IGNITED, now RELIABLE on the generous
  scaffold; POSITIVE / NEW INSIGHT):** the most direct path to the moonshot goal — a toy affective dyad
  that infers intent + has functional valence + learns to earn approval (docs/specs/m4-affective-dyad.md;
  `active_loop/affect_spec.py` + `affect_agent.py`). Increments 1/1b (Exp 125/127) halted on F3 (never
  learns); Exp 128 pinned a TIMING flaw; **Exp 214 (1c)** fixed the timing (REAL but not sufficient, 0/8);
  **Exp 215 (1d)** built a DIRECT response→valence head (`DirectHeadAgent`) and RULED OUT aliasing (K=U
  also 0/8) — narrowing the wall to exploration. **Exp 216 (1e, ignition-envelope diagnostic, POSITIVE)**
  decomposed the failure: policy-only 1.0 (exploitation works), update-only 0.83 at K=U+lr+data (learning
  works), closed-loop 1/4 ignite (seed20 POS 0.10→0.34) — the FIRST ignition; the Exp 215 wall was a
  CONJUNCTION whose binding remainder is the exploration cold-start. **Exp 217 (1f, cold-start break,
  POSITIVE, blind-verified)** turned that fragile 1/4 into a RELIABLE 7/8: an HONEST optimistic POS prior
  (+2.0 uniform across correct AND wrong cells — no leakage) makes 7/8 seeds learn (mean-last 0.44 ≫ 0.20
  chance); ε-greedy 5/8; control 2/8 byte-reproduces Exp 216 (regression PASS ⇒ gains real). Reliable ONLY
  on the generous scaffold. **Exp 218 (1g, ratchet, PARTIAL / NEW INSIGHT, blind-verified)** turned the
  difficulty back up one knob at a time (optimism held on): optimism does NOT rescue the realistic corner
  (γ1/K4/100t = 0/8, and it DEGRADES); the anchor reproduces Exp 217 byte-for-byte; the real blockers are
  SHORT SESSION (100t 1/8) + LOW PRECISION (γ1 1/8), NOT aliasing (γ4 8/8 even improves). KEY METRIC CATCH
  (self-healed): the ignition threshold (last≥0.30) sits BELOW the 0.333 constant-response ceiling, so a
  ~1/3 POS-rate proves nothing (a constant "always reply 0" policy already scores 1/3) — K4's 6/8 is a
  constant-ceiling artifact (the "aliasing rescued" read is WITHDRAWN), and the 216/217 reliable counts
  overstate genuine discrimination. Guard added: `affect_spec.constant_response_ceiling` + a pinning test.
  **Exp 219 (1h, discrimination readout, PARTIAL / NEW INSIGHT, blind-verified, Codex-coded)** added the
  honest probe `DirectHeadAgent.correct_select` (read-only, constant-UNFAKEABLE: a constant policy caps at
  2/6) and attacked the blockers at K=4. Result: the dyad GENUINELY discriminates (best K=4 g4_300 = 5/8,
  up to 5/6 codes right) but NOT reliably — no cell reaches ≥6/8, even the K=6 controls (4–5/8). The two
  blockers SEPARATE cleanly: SHORT SESSION blocks LEARNING (near-chance csel at 100t), LOW PRECISION blocks
  EXPLOITATION (γ1 has the highest csel 0.54 — it learns the mapping — but won't act on it). Realistic K=4 ≈
  K=6 ⇒ capacity is NOT the blocker. The honest metric TEMPERS 217/218: the anchor's lenient 7/8 is only
  4/8 GENUINE (learning real, reliability overstated). N=8 ⇒ no cell ranking (g4_600 3/8 < g4_300 5/8 is
  noise). **Exp 220 (1i, precision schedule, POSITIVE / NEW INSIGHT, blind-verified, Sonnet-coded)** added
  a per-turn γ schedule (`DirectHeadAgent.gamma_schedule`, eqx.tree_at on a throwaway agent — no leak into
  learning; byte-identical when off) and tested anneal-vs-fixed at K=4, N=16: gradually annealing γ 1→8 over
  the full session (sched_full) reaches **13/16 GENUINE** — the FIRST reliable genuine discrimination in M4a
  — fixing the learn-but-don't-exploit decoupling; it beats fixed γ4 (7/16) and fixed γ8 is WORST (3/16,
  over-commits early). NOT a breakthrough (blind-verified): every cell is the LONG 300t session — the short
  100t realistic session (Exp 219's open learning blocker) is untested, so length does much of the work and
  the schedule was never separated from it; the 13/16 is bar-fragile (csel≥0.67 → 7/16) and N=16 is modest.
  **Exp 221 (1j, NEGATIVE / NEW INSIGHT, blind-verified)** SEPARATED the schedule from session length: the
  precision schedule does NOT rescue the short session (short sched 0/2/3 of 16, bar 12) — it beats fixed
  only at 300t (+6) and is slightly WORSE at short lengths; mean_csel rises with length for BOTH conditions,
  so **short sessions block LEARNING** (the schedule only optimizes EXPLOITATION) and **the long 300t session
  is load-bearing**. Regression-anchored EXACT to Exp 220 (sched_300=13/16, fixed_300=7/16). **Exp 222 (1e,
  POSITIVE / milestone close, blind-verified) CLOSED the M4a chapter** at toy scale: built
  `active-monkey-converse` (the honest "talk to it" REPL) + `eval/affect_score.py` (FROZEN
  learns-to-positive scorer) + `_json` + tests.
  MILESTONE_VALIDATED: the DirectHeadAgent learner passes (verdict True: realized POS 0.42>1/3, genuine 6/8,
  improvement 0.24, reproduces Exp 220) AND a constant-reply control FAILS (verdict False, genuine 0) — a
  constant-UNFAKEABLE metric. Synthesis + provided-vs-learned ledger: `docs/research/m4a-affective-dyad-chapter.md`.
  **The M4b autopilot arc (Exp 223–225):** built the PR-style autopilot over MUTABLE `affect_spec.py` scored
  by the FROZEN `eval/affect_score.py` (higher=better; `active_loop/affect_pr_loop.py` + `run_affect_loop.py`).
  **Exp 223** (first real run) + **Exp 224** (clean re-run) were an instrument shakedown — repurposing the lang
  autopilot surfaced 4 bugs (180s timeout, lang mission/context, shared world_model, an untracked-dir git wipe;
  all fixed + guarded, L35); Exp 224 ran clean (rc=0) but found NO improvement at N=2 — CRITIC-GATED (both
  proposals rejected before scoring). **Exp 225 (POSITIVE / NEW INSIGHT, blind-verified) RESOLVED it:** the
  AffectClaudeCritic is SOUND (it rejects baking the code→intent map into A0 = gaming, and approves a legit C1
  NEU-aversion preference), so Exp 224's NEGATIVE was the critic doing its job; AND that approved, non-gaming
  change GENUINELY improves the metric — **C1 neu=-0.5 → mean_last 0.4225→0.5188, genuine 6/8→8/8** — so the
  autopilot HAS a real honest improving move (Exp 224's 0/2 was sampling, not a ceiling).
  NEXT (each needs a human word): (a) a full multi-iteration M4b run to show AUTONOMOUS find-and-keep of the C1
  improvement (+ the journal-accumulation fix); (b) ACCEPT C1=-0.5 into `affect_spec.py` (raises the milestone
  baseline to ~0.52 / genuine 8/8 — edits the spine the FROZEN scorer reads); (c) stop / pivot to the
  short-session LEARNING lever (lr/optimism/replay, Exp 221). Guards: tests/test_affect_agent.py +
  tests/test_affect_score.py + tests/test_affect_pr_loop.py; card: loop/directions/affective-dyad.md.
- Other standing options in loop/IDEAS.md (each needs its own word): nira's normalized-predictive switch
  (standing consult from Exp 154); the cloud-branch merge (renumber-on-merge plan).
- Suite ~250 fast tests green; every Exp 152+ verdict blind-verified (PROTOCOL 4.5).
- Efficiency / context discipline: see loop/EFFICIENCY.md (hot vs cold context, lean workflow results, slim output commits, STATUS-line cap).
- Fork/replay/restore as first-class primitives: `ecology/runtime.py` (+ `ecology/RUNTIME.md`) — lossless snapshot/restore (bit-identical pause/resume), counterfactual `fork`, `replay` with a bit-match gate, and `distill` (a founder genotype from successful runs — closes the replay_or_distill gap). The `creature/` spine has its own `Creature.fork()` + `growth.replay_nll()`. A unified cross-substrate AgentState is the documented next step.
- **Coalescence artifact standard (2026-06-17, PRs #63–66, merged):** experiments now accumulate
  into reusable artifacts. `active_loop/coalescence/` (schema/validate/inventory/backfill/export) +
  the `active-monkey coalesce …` CLI + `docs/ARTIFACT_SPEC.md`/`COALESCENCE_PLAN.md`/`MECHANISM_LIBRARY.md`.
  The library holds ExperimentBundles, MechanismCards, GeometryMaps, BoundaryNotes (validate --all green).
  When an experiment/chapter closes, distill it into a card. Memory: [[coalescence-artifact-standard]].
- **The self-other-modeling chapter (Exp 228–234, CLOSED-POSITIVE, 2026-06-18):** descends from the
  closed social-emergence direction's named gap (creatures are solipsistic). **Exp 229** = the FIRST
  functional self-other modeling — a creature infers another's latent goal from its trajectory and
  predicts it better than learning the other's transition dynamics (the solipsism gap crossed at toy
  scale), but LIGHT (policy provided). **Exp 230–231:** the benefit is LOCALISED to transitions
  (cold-start, goal-changes) and grows monotonically with the other's non-stationarity (direction of
  the law confirmed; magnitude knife-edge). **Exp 232–234 (three straight sanity-gate fires):** on this
  gridworld the SIMPLE baselines are robustly near-optimal — goal-directed BFS is too LEGIBLE to make
  inference hard via targets (difficulty = behavioral separability not goal-space size), and reactive
  stigmergy already coordinates so naive ToM-action HURTS. Distilled to
  `mechanisms/functional-goal-inference-v0` + `boundary_notes/self-other-substrate-legibility-wall-v0`;
  card `loop/directions/self-other-modeling.md` (closed-positive). Re-opens only on a substrate where
  the simple baselines FAIL (a less-legible/partially-observed other; or a coordination task stigmergy
  cannot solve). Introduced LESSONS L37 (stale direction-card STATUS) + L38 (predeclare a manipulation
  check; separability != cardinality).
- **The environmental-complexity direction (Exp 235, ACTIVE — the human's word 2026-06-18: "make
  the environment progressively more complex … see if they develop better ways to move"):** the last
  structurally-distinct escape from the local-gradient wall — movement was NEVER an evolvable axis
  (199–213). A NEW substrate (`enable_terrain`, byte-identical OFF, golden-hash guarded) adds a static
  2.5D sealed-plateau heightmap + a costed heritable `climb_ability` LOCOMOTION axis (Preflight
  `trait_axis.py`). **Exp 235 (NULL/INVALID / NEW INSIGHT, blind-verified):** the gate-open deflation
  control showed `climb_ability` is behaviorally INERT — the comfort-gated LOCAL greedy forager
  (`creature.py choose_action`, steps only to the best ADJACENT cell, no path-planning) never
  navigates to the distant sealed plateau (gate-open plateau intake ~1%, every geometry-sweep cell
  noise-level, worse under scarcity). The bottleneck is the POLICY, not the gate; the predeclared L38
  manipulation check fired its abort condition before any gradient batch. NOT a local-gradient wall
  result (no valid gradient measured). **NEXT = Exp 236 (the human's pick):** add a navigation-capable
  forage policy (path-planning toward distant high-value cells) behind a flag (byte-identical OFF), so
  the trait is expressible, then re-run the manipulation check and the Gate C local-gradient batch.
  **Exp 236 (NULL/INVALID / NEW INSIGHT, blind-verified):** the `enable_navigation` policy does NOT
  rescue it. METHODOLOGY CATCH (L40): the first build GAMED the expressibility metric (63.7% plateau
  access) via a higher-index target tie-break (plateau cells have high indices) — caught by the
  validator re-running with the neutral codebase-convention tie-break (→ 0.5%). Honest food-driven
  navigation reaches the plateau only under scarcity (~14%, gate-open), and the closed stochastic gate
  yields a ~0 climb marginal (retreat-on-failed-roll; persistence starves the pop). So this gridworld
  cannot jointly satisfy REACH + CROSS + SURVIVE for a locomotion trait — a substrate boundary across
  BOTH the local-greedy and navigation policies. **NEXT = Exp 237 (the human's pick):** a
  PERCEPTION-driven substrate — give the creature a resource-gradient sense (reuse the
  thermosense/band-tracking machinery) so it perceives + navigates toward the rich plateau, movement as
  the primary survival challenge; FAIL = even with distant-food perception no geometry makes
  climb_ability expressible (then close the gridworld for locomotion; next = a continuous-space mover).
  **Exp 237 (MIXED / NEW INSIGHT, blind-verified):** food-gradient PERCEPTION (`enable_food_sense`, a
  distance-decayed scent of the live resource field) SOLVES reach+survive — gate-open plateau intake
  1%->62%, anti-gaming-clean (flat-resource world -> 33% uniform null, food-driven not an index
  artifact), population persists. So the substrate VALIDLY POSES the locomotion-evolvability question
  for the first time. The answer: climb_ability does NOT cleanly evolve — the gen-0 MONOMORPHIC benefit
  curve is FLAT (64.3-64.7% across climb 0->1; saturation) and `invasion_from_rarity` says
  DOES_NOT_INVADE (rare mutant extinct 7/8). The Preflight's pairwise Gate C "PASS" (mutant wins 7/8
  at 50/50) is POSITIVE FREQUENCY-DEPENDENCE / a priority effect (get-there-first on the shared
  plateau), NOT directional selection — a caught FALSE-POSITIVE. So the **local-gradient wall EXTENDS
  to locomotion via benefit saturation**, now general across senses (199-201), memory (208-209), active
  sensing (210-211), and movement (237). SELF-HEAL (L41): fixed the Preflight aggregate that
  over-claimed PASS despite invasion-from-rarity failing; invasion-from-rarity is the binding criterion.
  **NEXT:** distil the arc into coalescence cards (a MechanismCard perception-enabled-locomotion +
  extend the local-gradient-wall BoundaryNote to locomotion), then CLOSE the direction (the gridworld's
  locomotion question is answered) OR — on a human word — a continuous-space mover with richer movement
  physics. Card: `loop/directions/environmental-complexity.md`.
  **Exp 238 (MIXED / NEW INSIGHT, self-corrected):** moved to the
  `continuous-locomotion` direction. New `ContinuousWorld` substrate (gated
  `enable_continuous_locomotion`, byte-identical OFF; full suite 817 passed). The discrete-grid
  saturation does NOT reappear in continuous space (bump Spearman 0.943, non-flat, curve turns over at
  speed 4 as cost bites) — the L39 expressibility prerequisite is met. BUT the rise is dominated by
  DISTANCE ARITHMETIC: the flat-null (uniform field, Spearman 1.000) and the navigation-disabled
  diagonal billiard (nav-off bump Spearman 1.000) both rise at the same rate. The genuine
  navigation/spatial bonus is real but secondary (+~0.07 slope). An intermediate commit (06b9e6d)
  over-claimed "L39 PASS" on a degenerate wall-clamping nav-off probe (zero-length sweeps made intake
  speed-invariant by construction — a spurious null). 894087d fixed the probe (diagonal billiard) and
  self-corrected to MIXED. **RUNG 2 IS GATED**: do not run invasion-from-rarity until the confound is
  isolated (Rung 1b: cost-neutralise coverage so only efficient navigation pays). Card:
  `loop/directions/continuous-locomotion.md`.

## 4. The two loops (IMPORTANT — don't confuse them)

This repo contains **two different "loops."** "Continue the moonshot" means loop B.

| | **A. Code-mutating autopilot** | **B. Claude-driven experiment loop** |
|---|---|---|
| What it is | `active-monkey-loop` / `active-monkey-pr-loop` machinery (`active_loop/cli/`) | Claude (you) writing & running experiment scripts |
| What it optimizes | the free-energy / bits-char **metric**, by editing `model_spec.py` / `lang_model_spec.py` | the **moonshot question**, by designing Exp 41, 42, … |
| Governed by | `MISSION.md` + `policy.md` (FROZEN trust boundary) | `loop/` modules (PROTOCOL + VALIDATION) via the `/loop` prompt in §6 |
| Output | kept/reverted diffs, `world_model/` grows | new entries appended to `EXPERIMENTS.md` |
| Human role | guardrail-only | guardrail-only; gently reminded it's a natural stop point |

Both keep everything in git. The moonshot exploration (Exp 1–40) is **loop B** — Claude proposing
experiments, not the autopilot. To continue the moonshot, re-issue the loop B prompt below.

## 5. Re-run what exists (smoke test the world before extending it)

```bash
cd /Users/mirro/Projects/active-loop
uv run --python .venv active-monkey-converse-demo    # capstone: two creatures, self-formed opinions
uv run --python .venv active-monkey-life --steps 200 # continue mirro's life (parallel track; born in Exp 45)
uv run --python .venv pytest -q                      # fast suite (~58 tests, well under a minute)
uv run --python .venv pytest -q -m 'slow or not slow' # full suite (~70 tests, ~4 min)
uv run --python .venv active-monkey-talk             # char-level babbler REPL (honest ceiling)
# autopilot (loop A), bounded so it doesn't run forever:
uv run --python .venv active-monkey-loop --iterations 1
```

Always use `uv run --python .venv` — the shell auto-activates conda base and shadows the venv.
Run experiment scripts from the repo root (or `PYTHONPATH=.`) so imports resolve.

## 6. Continue the moonshot — compose the prompt, or paste the verbatim fallback

**Preferred (modular):** generate the `/loop` prompt from the pluggable modules in `loop/`
(premise + direction + persona + optional one-off idea):

**Fastest (in-session command):** type `/lab <direction>` in a Claude session — it
runs `loop/compose.py`, shows the assembled prompt + a one-line summary, and begins
iterating only after you reply "go" (the human-consent gate). Steer mid-flight with
`/steer "<idea>"`; answer an open consult with `/steer --resume <word>`. The terminal
`compose.py` recipe below still works as a fallback.

```bash
uv run --python .venv python loop/compose.py --list                      # see modules
uv run --python .venv python loop/compose.py <direction>   # pick one explicitly (no default); bare invocation lists the choices
uv run --python .venv python loop/compose.py --direction transfer --persona default
uv run --python .venv python loop/compose.py --direction red-team --persona skeptic \
    --idea "anything you want this run to prioritize"
```

Paste its output into a fresh session. Steering happens by swapping cards
(`loop/directions/`, `loop/personas/`) or dropping bullets into `loop/IDEAS.md` — the loop
reads that inbox at the start of every iteration. `loop/PROTOCOL.md` is the per-iteration
procedure; `loop/VALIDATION.md` is the binding honesty contract (predeclared falsifiers,
negatives logged as negatives, provided-vs-self-formed named, no seed-shopping).

**Fallback (verbatim, pre-modular — still works):**

```
/loop Keep running the moonshot active-inference experiments until I stop you. GOAL: an agent I can
eventually talk to and ask what it thinks, with its opinions self-formed from experience, never
pretrained. Read RESUME.md and EXPERIMENTS.md first. STATE: realistic moonshot reached at toy scale;
Exp 1-40 done; we are in CONSOLIDATION (diminishing insight) — be honest about that. NEXT Exp 41+:
prefer experiments that probe the RECIPE's edges or push toward a ceiling rather than re-confirming
settled results, e.g. transfer (a creature reuses its recipe in a new world), multi-step relational
"thoughts", or a minimal sequence substrate toward short Q->A (the M4 affective-dyad spec in docs/specs/m4-affective-dyad.md is the designed-but-unbuilt next rung). Run experiments back-to-back on a short ~5-minute cadence; if
mid-task, continue across wakes. RECIPE: embodiment + grounding + continuous-registered-experience +
ONE innate anchor + taught labels; keep belief CONTINUOUS (never reset per episode); reuse the
verified pymdp patterns from Exp 21/26/30/34/35. CEILINGS (not toy-crackable, don't keep banging on
them): emergent grammar, fully tabula-rasa structure (open_problem.html). DISCIPLINE: one
hypothesis-driven experiment per iteration; append a brief honest entry to EXPERIMENTS.md each time
(mark consolidation vs. new insight); scripts in repo or PYTHONPATH=.; keep responses lightweight;
gently remind me it's a natural stopping point when insight flattens.
```

## 7. Map of everything

| File | What it is |
|---|---|
| `RESUME.md` | this bootstrap (you are here) |
| `CLAUDE.md` | auto-loaded session bootstrap — points every fresh session at this file |
| `loop/` | modular prompt OS for loop B: PREMISE / PROTOCOL / VALIDATION (honesty contract) / METHODOLOGY (advisory design-and-evaluation heuristics), `LESSONS.md` distilled rules card, `check_iteration.py` mechanical rubric, direction & persona cards, `IDEAS.md` human inbox, `compose.py` prompt builder + META.md (meta-improvement) |
| `loop/META.md` | the meta-improvement loop — when you find a noteworthy fixable issue or reusable insight, fix it AND add a durable guard (test/rule/skill) so it can't recur (self-healing) |
| `EXPERIMENTS.md` | append-only honest log, Exp 1–40 — the central artifact |
| `RESEARCH.md` | parallel math / frontier analysis |
| `open_problem.html` | the actual open problem written up (restyled "active_monkey" page — intentional, don't revert) |
| `active_loop/cli/` | console command entrypoints (`active-monkey-*`) |
| `active_loop/cli/converse_demo.py` | the capstone "talk to it" demo |
| `MISSION.md` / `policy.md` | governing docs for loop A (the code-mutating autopilot) |
| `active_loop/cli/talk.py` | char-model REPL babbler |
| `active_loop/` | the code: controller, worker, specs, lang model, critic, loop machinery |
| `world_model/` | persistent belief store (loop A); grows, never resets |
| `eval/` | FROZEN scorers (never edit) |
| `creature/` | persistent creature state — one creature's weights/belief/values/vocab + append-only biography, committed each experiment; `fork()` = counterfactual twin controls |

**The designed-but-unbuilt next rung:** `docs/specs/m4-affective-dyad.md`
— "talk to it and watch it learn to feel positive" (functional valence grounded in free energy,
intent inferred from utterances). Spec-only; this is the most direct path toward conversation.

Persistent memory for this project lives at
`~/.claude/projects/-Users-mirro-Projects-active-loop/memory/`.
