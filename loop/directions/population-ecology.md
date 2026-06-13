# direction: population-ecology

**Question.** Can a population of simple active-inference creatures sustain an *ecology*
— individuals maintain homeostasis, diverge through lived experience, reproduce under
viability constraints they each inherit, and branch into lineages — with the
**environment** (homeostatic death + finite, locally-regenerating resources) as the
selection pressure, and an external evaluator that only **records, summarizes, and
diagnoses**, never selects?

**Why it matters.** Every prior experiment (Exp 1–193) studied ONE creature (or a fork
control / a spine). This opens the population axis: treat the toy creature as an
*ancestor* and ask whether developmental diversity (same architecture, different lived
traffic → different learned state) plus inherited, mutated traits plus environmental
viability constraints produce non-trivial, reproducible population dynamics. It is the
substrate every later rung (temperature pressure, costly organs, niche/species
archives, mate choice, neural transition models) builds on.

**Naming honesty (binding).** The human opened this line as **"N5 — a population /
ecology / reproduction layer."** That label is THIS direction's name; it is
**DISTINCT** from the locked N0–N7 self-modeling ladder (`memory:
n-order-self-modeling-ladder`, design-locked 2026-06-10), whose **N5 = interoception /
motivation** (the M4 spec) and whose **N7 = collective / multi-agent**. Conceptually
this population work is closest to the ladder's **N7**; it is kept as its own
separately-numbered parallel direction (exp194+) per the human's explicit framing
(2026-06-12 reconciliation: "New parallel direction"), and does NOT redefine or
supersede the locked ladder's N5/N7 meanings. When this card says "N5" it means
*population ecology*, the human's name for this line — not the ladder slot.

**Hard design constraint (the central thesis, binding on every experiment here).**
Selection emerges from the environment, never from an external ranking.
- ALLOWED: a creature dies because `energy <= 0` (or, behind a flag, a temperature /
  damage / chronic-stress viability bound); a creature reproduces because its age
  `>=` its OWN inherited `maturity_age` and energy `>=` its OWN inherited
  `reproduction_energy_threshold` and it can pay the inherited transfer + overhead;
  a creature explores because local resources are depleted and its policy makes
  exploration useful.
- FORBIDDEN: a creature dies because the evaluator says `map_accuracy < 0.8`; a
  creature reproduces because the evaluator ranks it top-K; any survival/reproduction
  decision that reads a global fitness score. The "no hidden evaluator" property is a
  **verified design invariant** (a structural test + code review), reported as such —
  it is not dressed up as an empirical discovery.

**Experiment ladder** (each one iteration of PROTOCOL.md; each names its FAILURE):
- **Exp 194 — homeostatic population (this card's pre-registration below).** The
  ecology substrate: energy, finite/regenerating resources, homeostatic death,
  inherited-trait reproduction with bounded mutation, lineages. FAILURE: non-determinism
  under fixed seed; trivial dynamics (explodes unbounded, or extinct in all settings);
  homeostatic constraint inert (nothing starves); unconstrained reproduction; unsafe
  mutation; or scarcity exerts no measurable pressure.
- **Exp 195 — senescence / aging as a SECOND death cause** (behind an `enable_senescence`
  flag; off preserves Exp 194 exactly). Self-maintenance efficacy degrades with age and the
  degradation is COMPLEXITY-SCALED (more-complex creatures age out earlier/faster) — so age
  kills even a well-fed creature, at a complexity-dependent, non-fixed, non-linear age. Run
  senescence-OFF (the Exp 194 model = control) vs senescence-ON. FAILURE: aging never kills,
  or kills everyone before reproduction, or age-at-death does NOT scale with complexity (the
  core requirement), or the two-cause death mix stays degenerate. (Also fixes Exp 194's L18
  ill-posed cause-fraction: two causes make the cause-of-death fraction well-posed.)
- **Exp 196 — does senescence's complexity-frailty trade-off SELECT against complexity over a
  LONGER horizon?** (Exp 195 P6 was a null at 600 steps.) Senescence-OFF (control, flat
  complexity) vs ON, balanced regime with the safety cap raised so it equilibrates at its
  resource carrying capacity (no truncation — the L19 fix), 5000 steps, FRESH seeds. FAILURE:
  the senescence-on population's mean complexity does NOT diverge measurably below the control
  even at long horizon (the P6 null is then a real absence, not a horizon artifact), or the
  divergence does not GROW with time, or it is in the wrong direction.
- **Exp 197 — complexity = EXPRESSED-MACHINERY cost: a costed, evolvable THERMOSENSE organ under
  temperature** (the human's mechanism revision — complexity emerges from expressed capabilities,
  cost attaches to active machinery with nonzero floors, NOT a scalar "complexity bad" penalty;
  temperature pulled forward as the pressure that makes thermosense useful). Thermosense gives
  noisy thermal info the policy uses to flee stress, at upkeep = floor + inefficiency·intensity;
  thermal tolerance is ALSO costed and the comfort zone DRIFTS (no free escapes); death stays
  energy-mediated ("starvation"). Temperature ON (treatment) vs OFF (control); NEWBORN vs standing
  thermosense separates heritable selection from survivor bias. FAILURE: thermosense is NOT retained
  under temperature even among the living (doesn't pay even when useful); or unstable/confounded.
  ("Heritable selection FOR efficient capability" only if NEWBORN thermosense shifts vs control.)
- **Exp 198 — the thermosense intensity ATTRACTOR + clean de-novo emergence** (resolves Exp 197's
  survivor-bias confound). 2×2 temperature × seeding (founder intensity 0 vs 0.20): does thermosense
  EMERGE from zero under temperature (a pure heritable signal — no survivor bias possible when the
  trait starts at 0), and do seeded-high and de-novo-from-0 CONVERGE to the same low equilibrium
  (an attractor)? FAILURE: no clean temp-ON vs temp-OFF gap from zero (Exp 197's heritable signal
  was survivor bias / noise); or no convergence; or runaway to a functional organ.
- **Exp 199 — is the primitive-sensor ceiling FUNDAMENTAL? (the fitness-valley sweep).** Exp 197/198
  found thermosense pinned at a low primitive equilibrium (~0.045). Can a FUNCTIONAL organ evolve under
  a shallower valley (lower sensing noise, cheaper organ, stronger steering, harsher stress)? Sweep the
  valley depth (founder primitive 0.10) + a functional-RETENTION arm (founder 0.50). FAILURE (= the
  finding, NEGATIVE): the equilibrium stays primitive (<0.15) at every depth AND a seeded-functional
  organ decays / drives extinction — i.e. the functional state is selected against from both directions
  (benefit saturates: a little sensing suffices to reach safety, cost keeps rising). POSITIVE only if a
  functional organ (mean >0.30) evolves or is retained at the shallowest valley.
- **Exp 200 — niche / species archive** from trait + behavior clustering (the evaluator
  RECORDS clusters, never selects them). FAILURE: no separable clusters under pressure.
- **Exp 201 — self-selection / mate choice** on observed traits, only after viability
  ecology works. FAILURE: mate choice changes no lineage outcome vs random pairing.
- **Exp 202 — neural Bθ / deep active-inference transition models inside lineages.**

**Stop condition.** Exhausted when either (a) the ladder reaches a documented wall
(e.g. toy gridworld stops supplying discriminating viability failures — the same
saturation governor as the ladder memo), or (b) the central thesis is falsified (the
environment cannot be made to select without an external ranking). On stop, write the
synthesis in EXPERIMENTS.md and flip STATUS to closed-positive / closed-negative.

---

**EXP 194 — PRE-REGISTRATION (homeostatic population; committed BEFORE any data).**

- **Hypothesis (one sentence).** A population of simple homeostatic
  active-inference-flavored creatures, governed only by energy/resource constraints and
  inherited (mutated) traits, sustains a reproducing multi-generation ecology in which
  the *environment* exerts selection (homeostatic death + finite resources, no external
  ranking), and the qualitative regime shifts predictably with resource abundance —
  reproducibly under fixed seeds.

- **World.** A small gridworld (12×12) of resource cells with finite per-cell capacity
  and local regeneration; movement actions; per-move energy cost and per-step baseline
  metabolic cost (both inherited traits); crowding/depletion is emergent (many creatures
  deplete a cell faster than it regenerates). Three scenario configs share the SAME
  founder genotype and the SAME mechanics; only the resource parameters differ:
  **balanced** (persists, no explosion), **scarce** (pressure / die-off),
  **overabundant** (growth toward a runaway guard). The runaway guard is a SAFETY
  assert that flags explosion (falsifier F2) — it never culls (never selects).

- **Predictions if TRUE** (property-level, explicit thresholds; ≥3 fixed seeds per
  scenario, report ALL):
  - **P1 (determinism).** For each scenario, two runs with the same seed produce a
    byte-identical event-stream hash. [Binding infra requirement.]
  - **P2 (bounded persistence, balanced).** Balanced final population > 0 at the horizon
    AND ≤ the runaway cap, in 3/3 seeds (persists without exploding).
  - **P3 (multi-generation lineage).** Balanced: births > 0 and max generation ≥ 3, in
    3/3 seeds.
  - **P4 (homeostatic death is the only selector).** ≥1 death in balanced, and 100% of
    all deaths (every scenario) carry a homeostatic `cause_of_death` (energy≤0 etc.);
    0 deaths from any evaluator/ranking cause (structural; reported).
  - **P5 (scarcity bites — environment as pressure, with effect size).** Scarce vs
    balanced, in ≥3/3 seeds: scarce mean final population lower by ≥25% AND scarce
    starvation-death fraction higher by ≥0.15. (Effect-size + count per L6.)
  - **P6 (traits not frozen).** In ≥1 scenario, ≥1 genotype trait's last-generation
    population mean differs from the founder value by ≥1 mutation-σ (inherited variation
    propagates; this is drift/variation, NOT a claim of adaptation).
  - **P7 (regimes separate).** NOT all scenarios extinct AND NOT all scenarios exploding:
    at least one regime persists bounded while the regimes are measurably different.

- **Falsifiers (any ⇒ NEGATIVE or MIXED as marked).**
  - **F1.** Non-determinism: same seed → differing event hash in any scenario ⇒ NEGATIVE.
  - **F2.** Trivial dynamics: balanced explodes past the cap in all seeds, OR population
    extinct in ALL scenarios/seeds ⇒ NEGATIVE.
  - **F3.** Homeostatic constraint inert: zero homeostatic deaths anywhere ⇒ MIXED
    (substrate runs but does not model the claimed pressure).
  - **F4.** Reproduction unconstrained: any creature reproduces before its OWN
    maturity_age or below its OWN reproduction_energy_threshold ⇒ NEGATIVE.
  - **F5.** Unsafe mutation: >5% of births produce out-of-bounds genotypes ⇒ NEGATIVE.
  - **F6.** Hidden evaluator: any survival/reproduction path reads a global
    ranking/fitness score ⇒ NEGATIVE (design failure).
  - **F7.** Scarcity inert: scarce indistinguishable from balanced on P5's metrics at
    the predeclared sizes in a majority of seeds ⇒ MIXED/NEGATIVE on the central claim.

- **Verdict rule (conjunct-by-conjunct).** POSITIVE iff P1 ∧ P2 ∧ P3 ∧ P4 ∧ P5 and none
  of F1/F2/F4/F5/F6 fire. If P1 fails ⇒ NEGATIVE. If the substrate persists/reproduces/
  dies-homeostatically but scarcity shows no effect (F7) ⇒ MIXED (substrate established,
  environmental-selection claim unproven). F2/F4/F5/F6 ⇒ NEGATIVE. P6/P7 are supporting,
  not gating.

- **Honesty stakes (written before data).** The balanced/scarce/overabundant *resource*
  parameters are CHOSEN to place each scenario in its regime — that is legitimate
  environment design, but it must be DISCLOSED: the finding is "the by-construction
  environmental-selection design yields a non-trivial, reproducible ecology that responds
  to resource pressure," NOT "ecologies emerge inevitably." The policy is a simplified,
  PROVIDED homeostatic heuristic (resource-seeking value map), not the full pymdp active
  inference stack — named as provided. Trait change (P6) is drift/variation, not
  demonstrated adaptation; specialization/niches are deferred to Exp 198.

---

**EXP 195 — PRE-REGISTRATION (senescence / aging death; committed BEFORE any data).**

- **Hypothesis (one sentence).** Adding complexity-scaled senescence — age-accelerated
  degradation of self-maintenance, with onset earlier and rate higher for more-complex
  creatures (the SAME capability-blend complexity that already prices reproduction) — makes
  age a genuine death cause DISTINCT from starvation, killing even a well-fed creature at a
  complexity-dependent, non-fixed, non-linear age; and against a senescence-OFF control this
  yields a non-degenerate two-cause death mix that varies by resource regime and shortens
  lifespan with complexity — reproducibly under fixed seeds.

- **Model (PROVIDED, behind `enable_senescence`; OFF reproduces Exp 194 byte-for-byte).**
  `complexity ∈ [0,1]` = the existing blend of (energy_capacity, sensor_precision,
  memory_length). Each step past a complexity-shortened onset age, a `damage` variable
  accrues by `deg = base · (1 + k·complexity) · (age − onset)^p` (p > 1, super-linear),
  offset by self-maintenance `∝ energy/energy_capacity` (a well-fed creature resists, but
  `deg` eventually outgrows any maintenance — self-maintenance becomes impossible); death by
  cause `senescence` when `damage ≥ 1`. `onset = onset0 · (1 − k'·complexity)` so complex
  creatures begin aging out earlier. The senescence DIRECTION (complexity → frailty) is
  IMPOSED, not discovered; the constants are chosen only to place senescence in an OBSERVABLE
  regime (both causes occur, population still reproduces) — disclosed mechanism design.

- **Design.** Two arms on the SAME seeds/scenarios: **CONTROL** (senescence OFF = the Exp 194
  model — the baseline Exp 194 lacked) vs **TREATMENT** (senescence ON). 3 scenarios × 3 seeds
  × 2 arms + determinism reruns.

- **Predictions if TRUE** (≥3 seeds; report ALL):
  - **P1 determinism.** Same seed → identical event hash, both arms, every scenario.
  - **P2 senescence real & distinct.** Treatment: senescence deaths > 0 AND starvation deaths
    > 0 in ≥1 regime; the cause-of-death fraction is strictly in (0,1) — non-degenerate (the
    L18 fix). Control: 0 senescence deaths (by construction).
  - **P3 complexity → shorter senescence lifespan (THE CORE).** In treatment, age-at-
    senescence-death is negatively associated with complexity: top-third-complexity cohort
    mean senescence-age < bottom-third by ≥ a predeclared margin (effect size), OR Spearman
    ρ ≤ −0.2, in ≥3/3 seeds.
  - **P4 not-fixed / variable.** Senescence-death ages have real spread (coefficient of
    variation ≥ a predeclared threshold) — not a single fixed lifespan.
  - **P5 cause-mix varies by regime (WELL-POSED; the Exp 194 replacement).** Senescence-death
    fraction (senescence / total deaths) differs between the abundant and scarce regimes by
    ≥ a predeclared effect size (abundant → higher senescence fraction; scarce → lower),
    ≥3/3 seeds.
  - **P6 controlled selection signal (supporting).** Treatment mean population complexity
    diverges from the CONTROL (same seeds) by ≥ a predeclared effect size by horizon end in
    ≥1 regime (direction: senescence selects against complexity → lower in treatment).
  - **P7 substrate intact (regression).** Treatment still persists/reproduces/multi-gen in
    balanced; senescence-OFF reproduces Exp 194 (balanced 170/155/155, births 628/622/509).

- **Falsifiers.**
  - **F1.** Non-determinism (any arm/scenario) ⇒ NEGATIVE.
  - **F2.** Senescence inert OR lethal-degenerate: 0 senescence deaths anywhere, OR senescence
    kills before any reproduction in all regimes ⇒ NEGATIVE.
  - **F3 (THE CORE).** Complexity-independent: P3 fails (no complexity↔senescence-age relation
    at the effect size) ⇒ "scales with complexity" FALSE ⇒ NEGATIVE.
  - **F4.** Fixed lifespan: senescence ages have ~zero spread ⇒ "not fixed" FALSE ⇒ NEGATIVE.
  - **F5.** Cause-mix degenerate: senescence fraction ≈ 0 or ≈ 1 everywhere / no regime
    difference ⇒ MIXED (the well-posed-metric goal unmet).
  - **F6.** Regression break: senescence-OFF no longer reproduces Exp 194 numbers, OR adding
    senescence collapses the balanced ecology ⇒ NEGATIVE.
  - Reproduction gates / mutation safety / no-evaluator invariant preserved (regression).

- **Verdict rule (conjunct-by-conjunct).** POSITIVE iff P1 ∧ P2 ∧ P3 ∧ P5 ∧ P7 and none of
  F1/F2/F3/F4/F6 fire. P3 fail ⇒ NEGATIVE (complexity-scaling is the human's core
  requirement). If senescence is real & complexity-scaled (P2,P3) but the regime cause-mix
  (P5) or the selection signal (P6) is null ⇒ MIXED. P4/P6 supporting.

- **Honesty stakes (written before data).** The senescence direction (complexity → frailty)
  is IMPOSED by the model, disclosed — what is TESTED is the emergent, regime- and complexity-
  dependent consequences + the controlled selection signal, none of which the imposed
  direction directly forces. Senescence constants are tuned only to an observable regime
  (disclosed, like Exp 194's resource tuning; mechanics/metrics/falsifiers fixed before
  tuning). The policy is still PROVIDED; complexity is a derived blend, not pymdp.

---

**EXP 196 — PRE-REGISTRATION (does senescence select against complexity at a longer horizon;
committed BEFORE any data).**

- **Hypothesis (one sentence).** Exp 195 found NO complexity-selection from senescence at 600
  steps (P6 null); the hypothesis is that this was a HORIZON limit, not an absence — over many
  more generations, senescence's complexity-frailty trade-off progressively selects the
  population toward LOWER complexity, so against a senescence-OFF control (whose mean complexity
  stays ~flat) the senescence-ON population's mean complexity diverges measurably and
  progressively below it.

- **Reconnaissance disclosed (per L7).** A throwaway 2-seed probe (seeds 0,1; not committed) at
  horizon 5000 with the cap raised showed control complexity flat (~0.54 / ~0.43) and treatment
  complexity drifting down to ~0.34 / ~0.29 (a 0.13–0.20 gap by t=5000, vs <0.03 at t=600). The
  claim is FIXED from the probe and TESTED on FRESH seeds {3,4,5,6,7} never used in the probe.

- **Setup.** Balanced regime with the safety cap RAISED (max_population ~5000) so the population
  equilibrates at its RESOURCE carrying capacity (~530) instead of halting at the safety guard —
  a disclosed design change that removes the artificial truncation (the L19 fix) and lets both
  arms run full-length. horizon 5000; both ARMS (senescence OFF = control, ON = treatment) on
  the SAME fresh seeds {3,4,5,6,7}; sample population mean complexity every 500 steps (a
  trajectory per arm/seed); senescence rng-free (deterministic).

- **Predictions if TRUE** (5 seeds, report ALL):
  - **P1 determinism.** Same seed → identical event hash, both arms.
  - **P2 validity (INPUT gate, L5).** Both arms reach t=5000 WITHOUT explosion or extinction, in
    ≥4/5 seeds (the comparison is valid; tests the run completed, not the outcome).
  - **P3 (CORE) selection emerges.** At the horizon, treatment mean complexity < control mean
    complexity by ≥ 0.05, in ≥4/5 seeds (effect size + count, L6; probe showed 0.13–0.20).
  - **P4 progressive / emergent.** The gap (control − treatment complexity) at t=5000 exceeds the
    gap at t=600 (Exp 195's horizon) by ≥ 0.05, in ≥4/5 seeds — the signal GROWS with time; AND
    control mean complexity stays ~flat (|control(5000) − control(600)| < 0.05).
  - **P5 direction.** Treatment < control at the horizon in ALL valid seeds.

- **Falsifiers.**
  - **F1.** Non-determinism → NEGATIVE.
  - **F2 (CORE).** No emergence: treatment−control gap < 0.05 at horizon in a majority of valid
    seeds → senescence does NOT select against complexity even at long horizon → NEGATIVE (Exp
    195's P6 null is then a real absence, not a horizon artifact).
  - **F3.** Not progressive: the horizon gap does not exceed the t=600 gap by ≥0.05 in a majority
    → the divergence is not a horizon effect → MIXED.
  - **F4.** Wrong direction: treatment complexity > control in any valid seed → NEGATIVE.
  - **F5.** Invalid: arms fail to persist full-length (explosion/extinction) in a majority of
    seeds → NO_VERDICT (validity fails — re-tune the cap/regime).

- **Verdict rule.** POSITIVE iff P1 ∧ P2 ∧ P3 ∧ P4 ∧ P5 and none of F1/F4/F5 fire. F2 (P3 fail)
  → NEGATIVE. F3 (P4 fail, not progressive) → MIXED.

- **Honesty stakes (written before data).** The raised cap is a DISCLOSED design change that
  removes Exp 194/195's artificial safety-cap truncation so the balanced regime reaches its
  resource equilibrium (the L19 fix), NOT tuning-to-a-metric. The selection DIRECTION (frailty
  costs complexity → selection against it) is EXPECTED from the imposed cost, but the MAGNITUDE,
  TIMESCALE, and whether it overcomes complexity's foraging BENEFITS is the emergent, falsifiable
  question — Exp 195's 600-step null made it live. The control arm isolates senescence's effect
  from drift. Probe seeds (0,1) disclosed; tested on fresh seeds {3,4,5,6,7}.

---

**EXP 197 — PRE-REGISTRATION (REVISED 2026-06-13 — SUPERSEDES the scalar version below; complexity
as EXPRESSED-MACHINERY cost via a costed, evolvable THERMOSENSE organ under temperature; committed
BEFORE the fresh-seed run).**

- **Mechanism-revision note.** The human corrected the scalar `complexity_cost_scale` framing
  (committed e6c259e, now RETIRED as the mechanism): cost must attach to EXPRESSED MACHINERY, not a
  global complexity label. Realized complexity is a READOUT (base body + active capabilities,
  intensity-weighted); each active capability pays `upkeep = floor + inefficiency·intensity`
  (nonzero floor — efficiency can't make it free); death stays energy-mediated. Temperature is
  pulled forward (it was the next ladder rung) because thermosense is meaningless without it.

- **Hypothesis (one sentence).** A costed, expressed, evolvable THERMOSENSE organ (gives noisy
  temperature-gradient info the policy uses to flee thermal stress, at a floored per-tick upkeep)
  is RETAINED under temperature — where its information value offsets its metabolic burden — and
  selected AWAY without temperature (pure cost); and because thermal tolerance is ALSO costed and
  the safe zone DRIFTS (no free refuge), the population must adapt through machinery, so
  thermosense's fate tests whether EFFICIENT capability-bearing complexity is selected FOR when it
  pays its way. NEWBORN/gene-pool thermosense (vs the standing population) separates heritable
  selection from survivor bias.

- **Mechanism (PROVIDED).** Genotype: thermosense_intensity (0 = absent), thermosense_inefficiency
  (efficiency multiplier, floor 0.2). Intensity > 0.05 ⇒ organ ACTIVE: noisy thermal reading (noise
  ↓ with intensity) → policy steers toward low-stress cells; upkeep = floor 0.02 + inefficiency·
  intensity. Temperature = a static left→right gradient [0,1] whose COMFORT CENTER drifts (sin,
  amplitude 0.4, period 1500) so there is no fixed refuge; outside its temperature_tolerance band a
  creature pays stress energy (scale 1.0) → "starvation". Thermal tolerance is COSTED
  (tolerance_cost_scale 1.5 · tolerance/tick) — robustness is machinery too, so immunity isn't free.
  CONTROL = temperature OFF (no stress, no tolerance cost, no drift → thermosense is pure cost).
  TREATMENT = temperature ON. BOTH arms enable_thermosense (it can evolve in both). Founder SEEDS a
  primitive organ (intensity 0.20) + narrow tolerance (0.10): the test is RETENTION/loss of a seeded
  primitive capability, NOT de-novo emergence (infeasible at this horizon — disclosed). Regression-
  safe: thermosense traits skip rng when enable_thermosense off ⇒ exp194-196 byte-identical.

- **Disclosed pilots (per L7).** Pilots on seeds {100,101} (deleted, not committed) FIXED the regime:
  temperature_stress_scale 1.0, tolerance_cost_scale 1.5, comfort amplitude 0.4 / period 1500, upkeep
  floor 0.02, thermal_avoidance_weight 2.5, noise_base 0.2; founder intensity 0.20, tolerance 0.10. At
  t=5000 both pilot seeds: thermosense LIVING active-fraction ~0.52–0.55 (temp-ON) vs ~0.06–0.10
  (temp-OFF); NEWBORN intensity ~0.045–0.051 (ON) vs ~0.034–0.036 (OFF); tolerance stays LOW under
  temperature (~0.02). A first pilot exposed the cheap escapes (free tolerance + static refuge) that
  the costed-tolerance + drifting-comfort design removed. ALL params FIXED here; the final verdict
  runs on FRESH seeds {8,9,10,11,12} never used in any pilot.

- **Setup.** Balanced regime, max_population 5000, horizon 5000, fresh seeds {8,9,10,11,12}, temp-ON
  (treatment) vs temp-OFF (control). Instrumented at checkpoints + end: LIVING thermosense
  active-fraction + mean intensity + mean inefficiency; NEWBORN (per birth_t window) thermosense
  active-fraction + intensity; mean temperature_tolerance (living + newborn); age-stratified living
  thermosense; deaths by cause; population; food. (end = t=5000; newborn = creatures born in
  [4000,5000].)

- **Predictions if TRUE** (5 fresh seeds, report ALL):
  - **P1 determinism.** Same seed → identical event hash, both arms.
  - **P2 validity.** Both arms persist to t=5000 (no extinction/explosion), ≥4/5 seeds.
  - **P3 STANDING retention (L).** treatment living active-fraction − control ≥ 0.15, ≥4/5 seeds
    (thermosense retained among the living under temperature). [pilot ~0.45]
  - **P4 HERITABLE retention (H).** treatment NEWBORN mean intensity − control ≥ 0.008 (treatment >
    control), ≥4/5 seeds (gene pool retains thermosense ⇒ heritable, not pure survivor bias).
    [pilot ~0.012]
  - **P5 stability.** treatment > control holds in 5/5 seeds for BOTH living active-fraction and
    newborn intensity.
  - Supporting: under temperature, mean temperature_tolerance stays LOW (treatment < control) — the
    population adapts via SENSING, not by buying robustness.

- **Verdict taxonomy (predeclared).** SB (strong survivor bias) := (living active-frac gap ≥ 0.30 in
  ≥3/5 seeds) AND (newborn intensity gap < 0.02 in ≥3/5 seeds).
  - **F1** non-determinism → NEGATIVE. **F2** arms fail to persist (majority) → INCONCLUSIVE.
  - **NEGATIVE** if NOT P3 (thermosense not retained under temperature even among the living ⇒
    doesn't pay even when useful).
  - **INCONCLUSIVE** if NOT P5 (sign flips) or a density confound dominates.
  - **POSITIVE-HERITABLE** if P3 AND P4 AND NOT SB (gene pool clearly shifts; not survivor-dominated).
  - **POSITIVE-MIXED** if P3 AND P4 AND SB (heritable retention AND strong survivor bias — the
    pilot-expected outcome).
  - **POSITIVE-SURVIVOR-BIAS** if P3 AND NOT P4 (living retains thermosense, the gene pool does not).
  - Repo token on the `- Verdict:` line: POSITIVE for any POSITIVE-* label, NEGATIVE, or MIXED for
    INCONCLUSIVE — plus the fine-grained label.

- **Interpretation discipline (binding — the human's language).** Baseline claim: "a costed expressed
  capability (thermosense) is RETAINED among the living under temperature where its info value offsets
  its cost." Upgrade to "heritable selection FOR efficient capability-bearing complexity" ONLY if
  NEWBORN thermosense also shifts (P4). The point: under temperature, complexity (the organ) is NOT
  uniformly selected against — the capability that pays its way is kept; useless capability
  (thermosense under temp-OFF) is shed; and tolerance staying low shows the adaptation is sensing, not
  bought robustness.

- **Honesty stakes (written before data).** All mechanism + world params tuned on the disclosed pilots
  and FIXED before the fresh-seed run; thresholds predeclared here, not tuned after results. Founder
  SEEDS a primitive organ (intensity 0.20) ⇒ this is RETENTION/loss, NOT de-novo emergence (disclosed).
  Death is energy-mediated (no thermosense_death / complexity_death). Thermal tolerance is costed and
  comfort drifts — disclosed design choices that removed the cheap escapes a first pilot exposed.
  Treatment's much lower equilibrium population (~70 vs ~500) is a CONSEQUENCE of the temperature
  pressure, reported as a density diagnostic. Policy + founder PROVIDED; complexity is a readout from
  expressed machinery.

  _(SUPERSEDED scalar version, retained for the record: complexity_cost_scale · complexity(genotype)
  flat per-tick penalty, matched control cost_scale 0, fresh seeds {8,9,10,11,12} — replaced 2026-06-13
  by the expressed-machinery thermosense mechanism above on the human's correction.)_

---

**EXP 198 — PRE-REGISTRATION (the thermosense intensity ATTRACTOR + clean de-novo emergence;
committed BEFORE the fresh-seed run).**

- **Motivation.** Exp 197 found thermosense RETAINED under temperature but mostly by SURVIVOR BIAS,
  with a WEAK heritable shift — the seeded founder (intensity 0.20) confounds living-enrichment
  (survivor bias) with gene-pool shift (heritable). Starting the organ at intensity 0 REMOVES the
  confound: a trait nobody has cannot be survivor-biased, so any rise in the gene pool (newborns) is
  PURE heritable selection. This runs the clean test + characterizes the equilibrium.

- **Hypothesis (one sentence).** Under temperature the thermosense organ EMERGES de novo from
  intensity 0 to a low gene-pool equilibrium (clean heritable selection, no survivor-bias confound),
  reaching the SAME convergent attractor a seeded-high founder (0.20) RELAXES down to — a low,
  near-activation-threshold equilibrium that never runs away to a functional organ (the fitness valley
  + neutral-sub-threshold drift cap it); without temperature the equilibrium is lower.

- **Design (2×2 temperature × seeding; fresh seeds {13,14,15,16,17}, horizon 12000, the Exp 197
  thermosense regime — floor 0.02, active_threshold 0.05, noise 0.2, stress 1.0, tol_cost 1.5, comfort
  amp 0.4/period 1500, thermal_weight 2.5; founder tolerance 0.10):**
  - **A** temperature ON, founder intensity 0 (de-novo). **B** ON, founder 0.20 (seeded).
  - **C** temperature OFF, founder 0. **D** OFF, founder 0.20.
  Metric = NEWBORN (gene-pool) mean thermosense intensity, end = mean over the last 2000 steps; +
  the trajectory.

- **Disclosed pilots (per L7).** 2-seed pilots {100,101} (deleted) at horizon 15000: end newborn
  intensity A 0.040–0.044, B 0.048–0.050, C 0.027–0.030, D 0.033–0.035 ⇒ A>C (temp raises the from-0
  equilibrium ~0.013), A≈B and C≈D (convergence within ~0.01), all < 0.05 (no runaway). Params +
  thresholds FIXED here; final on FRESH seeds {13–17} never used in any pilot.

- **Predictions if TRUE** (5 fresh seeds, report ALL):
  - **P1 determinism.** Same seed → identical event hash, all arms.
  - **P2 validity.** All 4 arms persist to the horizon (no extinction/explosion), ≥4/5 seeds.
  - **P3 (CORE) clean de-novo heritable temperature effect.** A end − C end ≥ 0.008 in ≥4/5 seeds,
    AND A rose from 0 (A end > 0.02). Both A,C start at 0 ⇒ the gap is PURE heritable (no survivor
    bias). [pilot ~0.013]
  - **P4 (attractor / convergence).** |A − B| ≤ 0.02 AND |C − D| ≤ 0.02 in ≥4/5 seeds (seeded-high and
    de-novo-from-0 converge to the same equilibrium, temp-ON and temp-OFF respectively). [pilot ~0.004–0.007]
  - **P5 (primitive ceiling / no runaway).** All four arms' end newborn intensity < 0.10 in 5/5 seeds —
    no functional organ evolves. [pilot max ~0.050]
  - Supporting: A RISES from 0 (≥0.01) while B FALLS from 0.20 (< 0.10) — convergence from opposite directions.

- **Falsifiers.** F1 non-determinism → NEGATIVE. **F2 (CORE)** A−C gap < 0.008 in a majority → no
  clean heritable temperature boost from zero ⇒ Exp 197's heritable signal was survivor bias / noise →
  NEGATIVE. F3 |A−B|>0.02 or |C−D|>0.02 in a majority → seeding sets the equilibrium, not an attractor →
  MIXED. F4 any arm end newborn intensity > 0.15 → a functional organ evolved → reinterpret (POSITIVE
  surprise). F5 arms fail to persist (majority) → INCONCLUSIVE.

- **Verdict rule.** POSITIVE iff P1 ∧ P2 ∧ P3 ∧ P4 ∧ P5. P3 fail → NEGATIVE (the Exp 197 heritable
  signal does not survive the survivor-bias-free test). P4 fail → MIXED. P5 fail (runaway) → reinterpret.
  Repo token: POSITIVE / NEGATIVE / MIXED.

- **Honesty stakes.** All params/thresholds tuned on disclosed pilots {100,101} and FIXED before the
  fresh-seed run. The temperature heritable effect is SMALL (~0.013) but CLEAN (no survivor-bias
  confound — both A and C start at 0). The equilibrium is LOW/primitive (near the activation threshold),
  NOT a functional organ — honest about the fitness-valley ceiling. temp-ON arms persist at low
  population (~70–90) vs temp-OFF (~500), a density consequence. Policy + founder + costs PROVIDED.

---

**EXP 199 — PRE-REGISTRATION (is the primitive-sensor ceiling FUNDAMENTAL? the fitness-valley sweep;
committed BEFORE the fresh-seed run).**

- **Motivation.** Exp 197/198 pinned thermosense at a low primitive equilibrium (~0.045, near the
  activation threshold), never a functional organ. Is that ceiling FUNDAMENTAL, or just the deep-valley
  regime? This sweeps the valley depth AND tests whether a functional organ, if SEEDED, is even stable.

- **Hypothesis.** The primitive-sensor ceiling is FUNDAMENTAL: even under aggressively shallow-valley
  regimes (low sensing noise, cheap efficient organ, strong steering, harsh stress), thermosense does
  NOT evolve into a functional organ — the equilibrium rises only weakly with shallower valley but stays
  primitive (<0.15), AND a seeded-functional organ (0.50) is NOT retained (it decays toward primitive or
  drives extinction) — because the benefit SATURATES (a little sensing suffices to reach safety) while
  the cost keeps rising with intensity, so the functional state is selected against from both directions.

- **Arms** (all temperature ON, founder tolerance 0.10, CHEAP efficient organ: inefficiency 0.20,
  upkeep_floor 0.0, strong steering thermal_avoidance_weight 8.0, harsh stress temperature_stress_scale
  3.0, comfort amp 0.4/period 1500, active_threshold 0.05; horizon 12000; fresh seeds {18,19,20,21,22}):
  - **NOISE SWEEP** (founder intensity 0.10 — "can a primitive organ CLIMB?"): V1 noise 0.20, V2 0.10,
    V3 0.05, V4 0.02 (shallowest).
  - **RETENTION** (founder intensity 0.50 — "is a functional organ STABLE?"): F noise 0.02 (best case).
  Metric = NEWBORN (gene-pool) mean thermosense intensity, end = mean over creatures born in [10000,12000].

- **Disclosed pilots (per L7).** Pilots {100,101} (deleted): the noise sweep moved the gene-pool mean
  only 0.05→0.067 (never functional; max individual ~0.31); a seeded 0.50 organ DECAYED (0.41→0.10) or
  went EXTINCT (0.80 worse). Params/thresholds FIXED; final on FRESH seeds {18–22}.

- **Predictions if TRUE** (5 fresh seeds; L21-compliant validity = arm reached horizon with a
  MEASURABLE newborn cohort, pop ≥ 10; extinct/collapsed arms are INVALID for the intensity metric and
  reported as extinctions):
  - **P1 determinism.** Same seed → identical event hash (V4 + F on seed 18).
  - **P2 validity.** The noise-sweep arms (V1–V4, primitive founder) persist measurably in ≥4/5 seeds.
  - **P3 (CORE) no climbing.** Across V1–V4, every arm's end gene-pool mean intensity < 0.15 (primitive)
    in ≥4/5 seeds — a primitive organ does not climb to functional at ANY swept depth.
  - **P4 (CORE) no retention.** Arm F (founder 0.50) end intensity < 0.20 in its measurable runs (decays
    from 0.50) OR F goes extinct in a majority — the functional state is not stable even in the best case.
  - **P5 monotone-but-capped (supporting).** The sweep equilibrium rises weakly as noise drops (V4 ≥ V1)
    but stays < 0.15. Supporting: arm F has HIGHER extinction than the V arms.

- **Falsifiers (these would make it POSITIVE — the valley IS crossable):** FP1 any V arm end mean
  intensity > 0.30 in ≥4/5 seeds → a primitive organ CLIMBED → POSITIVE. FP2 arm F retains end intensity
  > 0.30 in ≥4/5 measurable seeds → functional state stable → POSITIVE. F1 non-determinism → NEGATIVE (infra).

- **Verdict rule.** NEGATIVE (= the finding: ceiling FUNDAMENTAL) iff P3 ∧ P4 and neither FP1 nor FP2
  fires. POSITIVE iff FP1 or FP2 (a functional organ evolves or is retained). MIXED if ambiguous
  (partial climbing / non-monotone / unstable). Repo token POSITIVE/NEGATIVE/MIXED. NOTE: NEGATIVE here
  is the SCIENTIFICALLY INTERESTING outcome (a structural ceiling), not a failure.

- **Interpretation.** If NEGATIVE (expected): costed sensing organs in an AVOIDANCE-based ecology
  self-limit to a primitive sensor — the functional state is unreachable AND unstable — because the
  benefit saturates once safety is reached while the cost grows with intensity. Structural evolutionary
  constraint; points the next direction (a functional organ may need a NON-saturating benefit, e.g.
  sensing for FORAGING rather than avoidance).

- **Honesty stakes.** The base regime is deliberately FAVORABLE to a functional organ (cheap efficient
  organ, strong steering, harsh stress, low noise) — so a NEGATIVE is the STRONG conclusion (even the
  best case fails). The functional-seed arm's extinctions are handled per L21 (invalid for the metric,
  reported as extinctions = evidence). Pilots disclosed; thresholds FIXED pre-run; founders + costs
  PROVIDED. The result is conditional on this avoidance-based sensing model.

**STATUS.** state: active · latest: Exp 198 (MIXED / NEW INSIGHT, core-POSITIVE, blind-verified, on main — CLEAN de-novo heritable emergence: from intensity 0 (survivor-bias impossible), temperature heritably grows the gene-pool sensor (~0.045 vs ~0.029, 5/5), confirming Exp 197's heritable signal is real; a CONVERGENT ATTRACTOR (de-novo-from-0 ≈ seeded-0.20) capped LOW by a FITNESS VALLEY; MIXED only on a predeclaration validity gap, seed-17 seeded arm collapsed → unmeasurable NaN → L21) · depends-on: ecology/ substrate (Exp 194–198) · reusable: ecology engine (+ temperature / drifting comfort / costed tolerance / costed-expressed-evolvable thermosense organ) · why: opens the population axis distinct from the locked ladder (human's "N5 population ecology" = ladder-N7 territory, kept parallel) · next-falsifiable: Exp 199 (pre-registered) — is the primitive-sensor ceiling FUNDAMENTAL? a fitness-valley NOISE SWEEP (founder 0.10, V1–V4 noise 0.20→0.02) + a functional-RETENTION arm (founder 0.50), all under the BEST-CASE shallow valley, fresh seeds {18–22}. NEGATIVE (= the finding: ceiling fundamental) iff no V arm climbs >0.30 (P3) AND the seeded-functional organ is not retained / goes extinct (P4); POSITIVE iff a functional organ (mean >0.30) evolves or is retained.
