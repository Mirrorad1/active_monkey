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
- **Exp 200 — the FORAGING escape: can a non-saturating benefit cross the valley?** (Exp 199's
  pointed escape.) Food concentrated in a DRIFTING thermal band; thermosense used to FORAGE (find
  food — a need that never ends) instead of avoid. Does a FUNCTIONAL organ (gene-pool mean >0.30)
  then evolve? FAILURE (the strong general finding, NEGATIVE): the gene-pool mean stays primitive
  (<0.15) even under foraging (wide AND narrow/precision-demanding bands) — so the ceiling is general
  across avoidance AND foraging, not specific to avoidance-saturation.
- **Exp 201 — niche / species archive** from trait + behavior clustering (the evaluator RECORDS
  clusters, never selects them). FAILURE: no separable clusters under pressure.
- **Exp 202 — self-selection / mate choice** on observed traits, only after viability ecology works.
  FAILURE: mate choice changes no lineage outcome vs random pairing.
- **Exp 203 — neural Bθ / deep active-inference transition models inside lineages.**

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

---

**EXP 200 — PRE-REGISTRATION (the FORAGING escape: can a non-saturating benefit cross the valley?
committed BEFORE the fresh-seed run).**

- **Motivation.** Exp 199 found the primitive-sensor ceiling fundamental for AVOIDANCE sensing (a
  little sensing reaches safety → benefit saturates). The named escape: a NON-saturating benefit —
  sensing for FORAGING (you always need food). This tests it.

- **Mechanism (new engine, gated; exp194-199 byte-identical, hash-verified).** Food regeneration
  concentrated in a thermal band that DRIFTS; thermosense in FORAGE mode steers toward the
  food-optimal temperature (finding food) instead of away from stress; NO temperature stress (the only
  benefit is foraging). The engine behavioral test confirms a strong forager (intensity 0.8)
  reproduces ~4x more than a no-organ creature — so foraging-sense is genuinely useful.

- **Hypothesis (the escape, from Exp 199's mechanism).** A non-saturating foraging benefit lets a
  FUNCTIONAL thermosense organ EVOLVE (gene-pool mean intensity climbs > 0.30), crossing the valley.

- **Arms** (temp ON for food coupling, stress 0, cheap efficient organ inefficiency 0.20 / floor 0,
  founder intensity 0.10, enable_thermosense, forage mode, food_concentration 8, forage weight 4,
  noise 0.5; horizon 12000; fresh seeds {23,24,25,26,27}):
  - **WIDE forage**: food_band_width 0.15, amplitude 0.3, period 1500.
  - **NARROW (precision) forage**: food_band_width 0.05, amplitude 0.4, period 600.
  - **CONTROL**: enable_food_coupling=False (uniform food → thermosense useless) — the primitive baseline.
  Metric = NEWBORN (gene-pool) mean thermosense intensity, end = born in [10000,12000].

- **Disclosed pilots (per L7).** Pilots {100,101} (deleted) — wide + narrow + precision foraging
  regimes; the gene-pool mean stayed 0.04–0.09 (never functional), max individual 0.3–0.6; NARROWER
  bands gave LOWER mean. So the pilots SUGGEST the escape FAILS; the experiment formally tests it on
  FRESH seeds {23–27}.

- **Predictions if TRUE** (5 fresh seeds; L21 validity = measurable cohort):
  - **P1 determinism.** **P2 validity** (arms persist measurably ≥4/5).
  - **P3 (CORE escape test).** SOME foraging arm (WIDE or NARROW) end gene-pool mean intensity > 0.30
    in ≥4/5 seeds → a functional organ evolved under foraging → the escape WORKS → POSITIVE.

- **Falsifier → NEGATIVE (escape fails, ceiling general).** All foraging arms stay primitive (end
  mean < 0.15) in a majority → the foraging escape FAILS; the primitive ceiling holds across avoidance
  AND foraging (a stronger, more general finding than Exp 199). F1 non-determinism → NEGATIVE (infra).

- **Diagnostics (reported, not gating).** The engine behavioral test (a strong forager out-reproduces
  a no-organ one ~4x) confirms foraging IS useful — so a NEGATIVE is NOT because foraging is useless,
  but because the MARGINAL fitness gradient on intensity is too weak (crude sensing captures the easy
  benefit; a narrow band is too stochastic to reliably select on — no Goldilocks gradient). Report the
  max individual intensity reached (a high-intensity individual exists but does not dominate).

- **Verdict rule.** POSITIVE iff P3 (a functional organ evolves under foraging). NEGATIVE iff all
  foraging arms stay primitive (the escape fails, ceiling general). MIXED if partial (one regime
  crosses but not robustly, or unstable). Repo token POSITIVE/NEGATIVE/MIXED.

- **Honesty stakes.** The base regime is deliberately FAVORABLE to a functional organ (cheap efficient
  organ, strong foraging benefit confirmed by the behavioral test). Predicting POSITIVE (the human's
  escape hypothesis) and getting NEGATIVE (pilot-suggested) is an HONEST negative, not a forced result.
  Pilots {100,101} disclosed (they showed no crossing); the experiment tests on FRESH seeds {23–27}.
  Engine changes gated; exp194–199 byte-identical (hash-verified). Founders + costs PROVIDED.

**EXP 201 — PRE-REGISTRATION (the INCREASING-RETURNS escape: does a convex / speed-gated benefit
cross the valley? committed BEFORE the fresh-seed run).**

- **Motivation.** Exp 199 (avoidance) + Exp 200 (foraging) found the primitive-sensor ceiling GENERAL:
  a crude sensor grabs the easy part of any *saturating* benefit, so precision's marginal payoff is too
  small to out-breed its cost (NO GOLDILOCKS GRADIENT). The ONE untested escape (the human's pick "a"):
  a regime with INCREASING returns to precision — where the marginal value of precision GROWS. A
  9-agent design+adversarial-audit workflow found that three engine features defeat most such designs
  (the lifetime resource-memory map lets crude sensors free-ride to known cells; competition is resolved
  by creature-id order not precision; free traits like learning_rate substitute for the costed organ).
  The one mechanism surviving all three: a MOVING target precision must TRACK — a moving band cannot be
  memorised as a static attractor or won by birth-order.

- **Mechanism (new engine, gated; exp194-200 byte-identical, hash-verified — exp200 WIDE seed23 hash
  502e0539… reproduces exactly).** exp200's fatal flaw was ONE line: the forage policy read the band
  centre `world.current_food_optimal` for FREE, so precision bought nothing. Exp 201 (`enable_band_
  staleness`) replaces that free read with a per-creature EMA tracker `band_estimate`: alpha =
  clamp(intensity·responsiveness, 0, 1) and reading-noise = noise_base·(1−intensity), so a crude tracker
  chronically LAGS a fast-drifting band into already-depleted cells while a precise tracker locks on.
  **L19 GUARD (binding):** intensity keys ONLY the tracker's alpha and reading-noise; food intake is
  NEVER written as reward=f(intensity) — it falls out of where the creature steps and the unchanged
  consume() depletion. Liveness verified: a precise tracker (0.80) holds mean error 0.070 to the centre
  vs a crude tracker's (0.10) 0.120.

- **Hypothesis (the increasing-returns escape).** Under the fast-band tracking regime a FUNCTIONAL
  thermosense organ evolves de-novo: gene-pool NEWBORN mean intensity crosses 0.30 in the FAST arm.

- **Arms** (temp ON for food coupling, stress 0, cheap efficient organ inefficiency 0.20 / floor 0,
  founder intensity 0.10, enable_thermosense, forage mode, food_concentration 8, forage weight 4,
  noise 0.5, band_width 0.12, regen 0.20, responsiveness 1.0; horizon 12000; fresh seeds {33,34,35,36,37}):
  - **FAST** (primary treatment): food_optimal_period 60 — the de-novo CLIMB test (>0.30 ⇒ POSITIVE).
  - **MEDIUM** (diagnostic): period 120 — predeclared to land BETWEEN SLOW and FAST (graded speed-gating).
  - **SLOW** (NULL control, binding L19 falsifier): period 2400 — band barely drifts, a crude tracker keeps
    up ⇒ must stay <0.15. If SLOW also climbs, the relation was IMPOSED, not earned — discard the positive.
  - **CLAMPED_LR** (confound-killer): = FAST but learning_rate FROZEN at the founder value ⇒ the climb
    cannot be resource-memory substitution. A POSITIVE requires CLAMPED_LR to also cross 0.30.
  - **SATURATING** (within-experiment exp200 baseline): = FAST geometry but `enable_band_staleness=False`
    (the free read) ⇒ predeclared primitive (~0.08); FAST-vs-SATURATING isolates exactly tracked-vs-free.
  - **USELESS** (primitive anchor): enable_food_coupling=False (organ pure cost) ⇒ predeclared ~0.06.
  Metric = NEWBORN (gene-pool) mean thermosense intensity, end = born in [10000,12000]; L21 validity
  (pop ≥ 10 + a newborn cohort).

- **Disclosed pilots (per L7).** Pilot seeds {110,111}. A REAL speed-gated climb exists (fast band ~0.18
  at t=4000 vs slow null ~0.10) — UNLIKE the flat avoidance/foraging gradients of exp199/200 — but at full
  horizon it is a TRANSIENT peaking ~t=2000 then DECAYING to a low equilibrium: the chosen FAST regime
  settled 0.13 / 0.20 (the best, most reliable of a swept set; scarcity/narrow/low-responsiveness variants
  were equal-or-lower and higher-variance, transient peaks up to 0.32 but all decaying ≤0.21). `strip≈0`:
  the band is NOT depleted within a step, so interference competition does NOT bite — the signal is pure
  convex tracking-threshold geometry, NOT frequency dependence (disclosed scoping). The regime was fixed
  BEFORE the fresh-seed run; pilots SUGGEST MIXED/NEGATIVE (no sustained crossing of 0.30).

- **Returns-curve probe (diagnostic, separate seeds {120,121}, run + committed BEFORE the verdict).**
  Gross per-capita intake(p) at FROZEN intensity p (enable_thermosense=False pins p, organ active, NO
  upkeep) vs the known linear cost 0.2·p — convexity of the benefit resolved separately in [0.10,0.30] and
  [0.30,0.50] (must persist above 0.25 to reach functional), FAST vs SLOW. Diagnostic of the landscape;
  the VERDICT is the evolving gene-pool metric (L22).

- **Predictions if TRUE** (5 fresh seeds): **P1 determinism**; **P2 validity** (FAST valid ≥4/5);
  **P3 (CORE)** FAST end gene-pool mean > 0.30 in ≥4/5 valid seeds AND speed-gating (FAST_mean > SLOW_mean
  AND SLOW_mean < 0.15) AND CLAMPED_LR also > 0.30 in ≥4/5 ⇒ POSITIVE.

- **Falsifiers.** PRIMARY → NEGATIVE (the stronger, more general wall): FAST end mean < 0.15 in a majority
  of valid seeds ⇒ the ceiling survives INCREASING-RETURNS-to-precision. MIXED: FAST in [0.15, 0.30]
  (a real speed-gated climb above primitive but not functional — the predeclared 'convexity flattens above
  the threshold' outcome). INTERNAL DEGENERACY (⇒ any positive DISCARDED): SLOW_mean ≥ 0.30 OR USELESS_mean
  ≥ 0.15. CONFOUND (⇒ POSITIVE uninterpretable): FAST > 0.30 but CLAMPED_LR not, OR newborn learning_rate
  rises in FAST vs USELESS. F1 non-determinism → NEGATIVE; regression OFF-path hash mismatch → NEGATIVE (infra).

- **Verdict rule.** NEGATIVE if not P1; else DEGENERATE-NEGATIVE if the internal falsifier fires; else
  POSITIVE iff P3 and no LR confound; else NEGATIVE iff FAST majority-primitive; else MIXED. Token
  POSITIVE/NEGATIVE/MIXED.

- **Honesty stakes.** The regime is chosen to give the escape its best fair shot (the most reliable climber
  of a disclosed sweep, natural responsiveness 1.0), fixed before the verdict. Predicting the escape FAILS
  to SUSTAIN functional (pilots show a transient climb decaying to ~0.13–0.20) — the human's increasing-
  returns hypothesis tested on FRESH seeds {33–37}. The band tracker is a PROVIDED heuristic (not learned);
  the interference-competition / frequency-dependence component turned out weak (strip≈0) — disclosed, so
  the test is of the convex tracking-threshold, not Red-Queen dynamics. Engine gated; exp194–200 byte-
  identical (hash-verified, durable guards added). Founders + costs PROVIDED.

**EXP 202 — PRE-REGISTRATION (the INTERFERENCE-COMPETITION escape; committed BEFORE the verdict run; full
binding predeclaration in experiments/exp202_n5_interference_competition.py docstring).**
- **Motivation.** Three walls (199 avoidance, 200 foraging, 201 increasing-returns) leave ONE sensing escape:
  REAL positive frequency dependence — precision pays MORE as the trait spreads — via interference
  competition at a GENUINELY DEPLETING band. exp201's competition was inert (strip≈0); exp202 makes the band
  deplete AND neutralises the engine's ascending-creature_id "eat-first" confound.
- **Mechanism (gated; exp194-201 byte-identical, hash-verified).** `shuffle_creature_order` randomises the
  per-step processing order (rng.shuffle, ON-branch only) so a contested cell is won by navigation skill, not
  birth order — `consume()` UNCHANGED. `track_band_strip` is a gated validity auditor (no rng, not in
  events_hash). An independent **Codex diagnosis AND a Claude design workflow BOTH converged on shuffle** and
  BOTH rejected the alternatives: precision-weighted split (= a smuggled cross-creature genotype evaluator,
  invariant violation), precision-ordering (= forbidden global ranking), mixed-founders (re-correlates id with
  precision mid-run). Durable guards: tests/test_exp202_competition.py.
- **Arms** (founder intensity 0.10, fresh seeds {38-42}, horizon 12000, conc 14 / band 0.08 / regen 0.08):
  COMPETE (depleting + shuffle, primary) · NO_SHUFFLE (id-order ON, isolates the fair queue) · CLAMPED_LR
  (learning_rate frozen, confound-killer) · ABUNDANT (regen 0.8, no scarcity = no-competition baseline) ·
  USELESS (no coupling, organ pure cost). Metric = NEWBORN gene-pool mean intensity in [10000,12000].
- **Disclosed pilots {130,131,140-147}.** STRIP-GATE PASSES (strip_frac 100%, ~170 contested in-band
  occupants — genuine depletion, UNLIKE exp201). But the de-novo climb DECAYS to ~0.03 (organ selected
  AGAINST); a lone v_narrow seed hit 0.736 at a COLLAPSED pop (101), and an 8-seed characterisation gave 0/8
  functional with corr(pop,intensity) = -0.92 (high intensity ONLY at collapse = DRIFT). Regime FIXED pre-run.
- **Predictions.** P1 determinism; P2 validity (COMPETE valid ≥4/5 AND strip_frac>0.5 = genuine competition);
  P3 (CORE → POSITIVE): COMPETE mean >0.30 in ≥4/5 valid seeds AT HEALTHY pop (≥300; functional at pop<300 is
  DRIFT-flagged) AND CLAMPED_LR also >0.30.
- **Falsifiers.** PRIMARY → NEGATIVE (MAXIMALLY general wall): COMPETE <0.15 majority. MIXED: [0.15,0.30].
  DRIFT (positive DISCARDED): functional only at collapsed pops. INERT (pause-and-consult, NOT negative): strip
  ≤0.5 (band didn't deplete). F1 non-determinism / OFF-hash mismatch → NEGATIVE (infra).
- **Honesty stakes.** Pilots SUGGEST NEGATIVE; predicting the human's escape, tested on FRESH seeds. A NEGATIVE
  here is the strongest publishable form of the wall (the ceiling survives even real Red-Queen competition with
  the id-order confound removed). Engine gated; scheduler + founders + costs PROVIDED.

---

# THE SENSE-EVOLUTION SUB-ARC (Exp 203–206) — the human's steer, 2026-06-13

**The reframe.** Exp 199–202 closed four walls (avoidance / foraging / increasing-returns /
interference-competition), all NEGATIVE: a costed thermosense organ never becomes functional
and, under competition, is actively suppressed. The question is no longer *"make thermosense
win."* It is the reusable meta-question:

> **What environmental, informational, and evolutionary conditions are REQUIRED for a costed
> sense-like trait to become increasingly selected over generations — to become organ-like
> rather than merely useful when gifted?** — phrased so the answer transfers from thermosense
> to sight, hearing, localization, and eventually communication.

**The central distinction (the load-bearing one).** A *forced/installed* benefit probe can show
`B(0.60) − B(0.00) ≫ 0` (a gifted strong sensor reproduces far more) while evolution still fails,
because the *local* marginal benefit at the current trait value is below the marginal cost:
`B'(0.08) − C'(0.08) ≤ 0`. Exp 200 already logged this as L22 (a 4× forced benefit, invisible to
evolution). The whole sub-arc instruments the **local slope**, not the endpoint advantage.

**Hard anti-cheat constraints (binding on every experiment here; extend the no-hidden-evaluator
invariant).** Do NOT directly reward high h. No rule `if intensity > θ: bonus`. Food/fitness is
NEVER written as `f(intensity)` (the L19 guard from exp201, now generalized). Any advantage must
emerge through the creature's interaction with the world: sensing hidden structure, choosing
actions, avoiding mistakes, exploiting niches. Costs stay ON for every verdict; cost-off is a
DIAGNOSTIC only (the B(h) overlay), never the verdict. The niche archive (Exp 205) must not
preserve high h directly — only ecological roles. These are asserted in code and re-checked by
the blinded verifier.

**Definitions (the human's terms + their programmer translation).**
- **Selection gradient** `g(h) = dE[w|h]/dh` — local slope of reproductive fitness w.r.t. the
  trait. *Translation:* the optimizer (mutation+selection) only accepts a small +Δh if that Δh
  improves realized deployment fitness at the CURRENT population state.
- **Affordance** — an action opportunity opened by the sensor↔environment↔policy coupling.
  *Translation:* the sensor exposes a NEW usable API, not a higher-resolution version of an
  already-solved call.
- **Concavity** — diminishing returns: each extra unit of h gives less benefit than the last
  (β₂ < 0 in the fit). The four walls were all concave + cost-dominated.
- **Red-Queen dynamic** — my trait grows MORE valuable as rivals improve: `∂²B/∂h_i ∂(mean h_rivals) > 0`.
  *Translation:* an arms race, not a stampede toward one shared resource (exp202 got the stampede).
- **Sympatric divergence** — one population self-partitions into specialized subtypes without
  physical separation. *Translation:* one shared service cluster splits into specialized
  configs because different workloads reward different settings.
- **Mutual information** `I(h; niche)` — how much knowing h reduces uncertainty about which
  niche/lineage an agent occupies. *Translation:* does the trait actually PREDICT the role?

**General fitness form** (the instrument all four bridges share):
`w_i = B(a_i, z, h_i, θ_i, ρ) − C_h(h_i) − C_θ(θ_i)`, with `h`=sensor precision/intensity,
`θ`=controller/policy ability to USE the sensor, `z`=hidden world state, `s_i ~ P(s|z,h_i)` the
observation, `a_i = π(s_i, θ_i)` the action, `ρ`=ecological state (density/depletion/residue/
niche occupancy). Gradual organ growth requires `dB/dh > dC/dh + drift/noise/transmission erosion`
near the resident — the bar every bridge must clear.

**Failure modes being addressed (from 199–202; tagged in the script docstrings).** Avoidance
saturation (199); foraging installed-benefit vs evolvable-benefit mismatch (200, L22); moving-band
concavity (201, L23); competition herding not Red-Queen (202, L24); id-order / birth-order confound
(neutralized by `shuffle_creature_order`, exp202); population-collapse / drift confound (healthy-pop
floor + corr(pop,trait), L24); cheap-trait / memory substitution (`freeze_learning_rate`, L19);
sensor not coupled to a policy that can exploit precision (the Exp 206 hypothesis).

**Reusable abstraction (Phase 1, `ecology/sense_axis.py`).** A `SenseAxis` treats a sense as a
generic instance of (h, C_h, z, P(s|z,h), θ, a=π(s,θ), ρ, w) plus diagnostics for B(h), C(h),
the local slope, and the realized selection gradient. Thermosense is the first instance; the
docstring maps sight / hearing / communication onto the same fields (Phase 6). Meta-principle:
*a sense is evolvable when it exposes private actionable information whose marginal value stays
positive across small heritable improvements.*

**Gated engine features this sub-arc adds (all default-OFF ⇒ exp194–202 byte-identical, hash-guarded).**
- `freeze_thermosense` — pins `thermosense_intensity` + `thermosense_inefficiency` to the PARENT
  value across mutation while STILL charging upkeep (cost ON). Mirrors `freeze_learning_rate`
  exactly (draw-then-discard the perturbation so the rng stream is unchanged). Lets a clamp value
  breed true with cost on — the unit of a gradient audit.
- `founder_mix` — seed the initial population from an explicit `((genotype, count), …)` list
  instead of one founder (None ⇒ existing single-founder seeding, byte-identical). Lets a single
  shared world host a polymorphic common-garden (resident + clamp grid).

---

**EXP 203 — PRE-REGISTRATION (the SELECTION-GRADIENT AUDIT; committed BEFORE the verdict run).**

- **Motivation.** Before building a new ecology to MAKE the gradient positive, MEASURE whether the
  four existing ecologies even contain a positive local gradient. This unifies the four walls into
  one diagnostic: does installed benefit B(h) > 0 coexist with realized gradient g(h_resident) ≤ 0?

- **Hypothesis (one sentence).** In all four existing ecologies (avoidance, foraging, competition,
  null), the installed-benefit curve B(h) is strongly increasing in the foraging/competition arms
  (a gifted strong sensor out-reproduces a no-sensor one) while the LOCAL realized selection
  gradient near the founder/resident value (h≈0.10) is ≤ 0 — i.e., the gift is real but un-earnable,
  because B(h) is concave and cost-dominated (β₂ < 0, dB/dh − dC/dh ≤ 0 at the resident).

- **Method (two complementary measures; cost ON for the verdict, shuffle ON, h breeds true via
  `freeze_thermosense`; the clamp grid is h ∈ {0.00, 0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60}).**
  1. **COMMON-GARDEN selection coefficient (PRIMARY — the realized gradient).** One shared world
     seeded EQUALLY across the clamp grid (`founder_mix`, all inefficiency 0.20, all breeding true).
     Cost ON, shuffle ON. Measure each clamp's realized per-capita reproduction over a window (after a
     burn-in, while multiple clamps remain present) → the realized fitness curve w(h). The relative
     reproduction across clamps IS the selection differential; its local slope at h≈0.10 is g. Report
     the per-clamp frequency trajectory too (which clamps grow/shrink). A `CLAMPED_LR` variant
     (learning_rate frozen) rules out memory substitution.
  2. **MONOMORPHIC benefit/cost curves (DIAGNOSTIC — the B(h) overlay).** Per clamp h, a monomorphic
     population: gross per-capita intake B(h) with cost OFF (`enable_thermosense=False` pins h, organ
     active, no upkeep — the exp201 returns-probe pattern) AND realized reproduction with cost ON. Makes
     the `B(0.60) − B(0.00) ≫ 0` vs `g(0.10) ≤ 0` distinction explicit.

- **Ecologies (arms).** FORAGE (exp200/201 drifting band — clearest gift), COMPETE (exp202 depleting
  band + shuffle), AVOID (exp199 avoidance/stress), NULL (no coupling — organ pure cost; g ≤ 0 by
  construction, the sanity anchor). Fresh seeds {50–57} (8, to satisfy the ≥7/8 slope criterion);
  monomorphic diagnostic on {60–62}; horizon 6000; balanced regime.

- **Regression.** Fit `w = α_seed + β₁·h + β₂·h² + β₃·(h·density) + ε` on the common-garden per-clamp
  reproduction. Report β₁, β₂ (concavity), β₃, optimum h*, local slope at h≈{0.03, 0.10, 0.20},
  per-seed sign of the slope at the resident, the fraction of seeds with a positive resident slope,
  and bootstrapped uncertainty where cheap.

- **Metrics (per clamp/seed).** clamped_h, lifetime_food (resource_eaten), reproduction/offspring
  count, age_at_death, movement_distance (steps_moved) + movement_cost, sensor_cost_paid (upkeep·life),
  fresh_food_visits / depleted_visits (resource above/below a threshold at the visited cell), local
  density, population_size, extinction flag, per-clamp realized reproduction (the parent-weighted
  contribution). FALSE/TRUE positives are undefined without residue → they arrive with Exp 204
  (disclosed).

- **Predeclared verdict (three-way, the human's pass/fail).**
  - **POSITIVE_GRADIENT** (a candidate evolvable ecology exists) iff, in SOME ecology: estimated
    optimum h* > 0.30 AND the local slope at the resident (h≈0.10) is > 0 in ≥7/8 seeds AND no
    population collapse AND no id-order confound AND no cheap-trait/memory substitution (CLAMPED_LR
    agrees) AND no direct-h-reward (asserted). [This would be the FIRST positive in the arc — surprising.]
  - **NEGATIVE_GRADIENT** iff the installed high-h benefit is real (FORAGE B(0.60) − B(0.00) > 0,
    confirming the gift) BUT the local resident slope is ≤ 0 in a majority of seeds in EVERY ecology
    — the gift exists and is un-earnable. [The expected, foundational result that explains all four walls.]
  - **NO_VERDICT** iff populations collapse, the common garden fixes before the measurement window, or
    the instrument otherwise fails to deliver a measurable per-clamp reproduction curve (L21).
  - Repo token: POSITIVE for POSITIVE_GRADIENT; NEGATIVE for NEGATIVE_GRADIENT; MIXED for NO_VERDICT /
    ambiguous (e.g. a positive slope only in collapsed/drift populations, or non-monotone across seeds).

- **Falsifiers / confounds.** A positive slope appearing ONLY in collapsed populations (corr(pop,
  reproduction) strongly negative) ⇒ DRIFT, discard (L24). A positive slope that vanishes when
  CLAMPED_LR ⇒ memory substitution, discard (L19). Non-determinism (same seed → different hash with
  flags off) ⇒ NEGATIVE (infra). The regression β₁ alone is not the verdict — the per-seed local-slope
  sign at the resident is (L1/L2: every conjunct checked).

- **Honesty stakes.** Predicting NEGATIVE_GRADIENT (the four walls predict it). This is a DIAGNOSTIC,
  not a new escape — its value is reframing the arc from "run another evolution → NEGATIVE" to "here is
  the measured gradient that explains all four walls, and here is the exact bar a bridge must clear."
  Founders + policy + costs PROVIDED. Cost-off is used ONLY for the B(h) overlay. Engine features gated;
  exp194/exp200/exp202 hashes reproduce (guards in tests).

---

**EXP 204 — FULL PRE-REGISTRATION (FRESHNESS/RESIDUE FALSE-POSITIVE BRIDGE; committed BEFORE any
data, on the human's explicit word "a continue" 2026-06-13; expands the staged sketch).**

- **Motivation.** Exp 203 found the FIRST positive local gradient in the arc (0.10→0.15 wins 7/8 in
  the band-staleness FORAGE regime) but it was PURELY competitive + weak (s~0.01, B(h) actually
  concave/negative, h*=0.0) — one seed short of POSITIVE. Exp 202's competition was a herding
  STAMPEDE, not Red-Queen. The bridge changes the PAYOFF STRUCTURE: instead of precision grabbing
  marginally MORE food (a saturating/concave benefit), precision AVOIDS A COSTLY MISTAKE — eating
  misleading residue. Avoiding a loss is steeper and compounds; and competition RAISES residue
  (rivals deplete → traces accumulate), so the marginal value of precision should RISE with rival
  density (the Red-Queen-via-residue derivative ∂²B/∂h_i ∂ρ > 0).

- **Mechanism (PROVIDED, new gated engine feature `enable_residue`; default-OFF ⇒ exp194–203
  byte-identical, hash-guarded).** The world carries a `residue` field. When a creature eats `eaten`
  units of fresh food at a cell, that cell gains `residue += eaten · residue_yield` (eaten food
  leaves a misleading trace); residue DECAYS each step (`residue *= 1 − residue_decay`). At the eat
  step (ON-branch only, ONE rng draw — OFF path keeps the exact current unconditional `consume`):
  let `R = resource[pos]`, `D = residue[pos]`, true fresh fraction `f = R/(R+D+ε)`; the creature
  reads a NOISY freshness percept `f̂ = f + N(0, σ)` with `σ = residue_confusion · (1 − h)`
  (`h = thermosense_intensity` = the evolvable sensor precision; `residue_confusion` = the
  signature-gap DIFFICULTY, higher = more aliasing). If `f̂ ≥ eat_threshold` the creature EATS:
  `eaten = consume(pos, deficit)` (the UNCHANGED engine intake path) and, if the cell was ACTUALLY
  residue-dominated (`f < fp_threshold`), it paid a FALSE POSITIVE — energy `-= residue_loss` (the
  cost of ingesting a misleading trace, a real ecological cost of the ACTION). If `f̂ < eat_threshold`
  it SKIPS (no gain, no loss; a missed-fresh cell is a false negative). So higher h → fewer FP losses
  AND fewer missed-fresh, via a better PERCEPT — never via a reward on h.

- **Anti-cheat (binding; extends `assert_no_direct_h_reward`).** Food intake is the unchanged
  `consume()` — NEVER written as `f(h)`. `h` keys ONLY the perceptual noise `σ`. `residue_loss` is
  charged by the ACTION of eating a residue-dominated cell, identical regardless of `h`. No rule
  `if h>θ: bonus`. Cost ON for every verdict (cost-off only for the B(h) overlay). The blinded
  verifier re-checks the no-direct-h-reward property from the script + raw output.

- **Two measurement modes (the exp203 instrument + an evolution confirmation).**
  - **MODE A — GRADIENT AUDIT (decisive, cheaper, mirrors Exp 203).** The clamp-grid common garden
    (`founder_mix`, `freeze_thermosense`, cost ON, shuffle ON) in the RESIDUE_COMPETE ecology:
    pairwise selection coefficient s(0.10 vs 0.15) [invader-won fraction + AUC], competitive optimum
    h* (carrying-capacity N*/R*), and the B(h) overlay (cost-off intake) — vs the NO_RESIDUE control,
    across a difficulty sweep.
  - **MODE B — EVOLUTION (confirmation).** h evolves freely from the primitive founder (h=0.10,
    `freeze_thermosense=False`, `mutate_thermosense=True`) in RESIDUE_COMPETE: does the gene-pool
    NEWBORN mean h climb >0.30 in a stable lineage, MEDIATED by a falling FP-rate?

- **Arms** (founder h=0.10, inefficiency 0.20; cost ON, shuffle ON unless noted; depleting band as
  Exp 202: regen 0.08, conc 14, band 0.08): **RESIDUE_COMPETE** (primary) · **NO_RESIDUE** (control:
  `enable_residue=False`, else identical — the no-FP baseline, predeclared primitive as Exp 202) ·
  **RESIDUE_NO_COMPETE** (control: residue ON but ABUNDANT/low-density, regen 0.8 → little eating →
  little residue → flat — competition is necessary) · **NO_SHUFFLE** (artifact: residue+compete,
  id-order ON) · **CLAMPED_LR** (confound: residue+compete, `freeze_learning_rate` — memory can't
  substitute) · **COST_OFF** (diagnostic B(h) overlay). Difficulty sweep `residue_confusion ∈
  {easy, medium, hard}` (exact values FIXED on a disclosed pilot). Seeds: audit {50–57} (8, for the
  ≥7/8 slope bar); evolution {70–74} (5 fresh). Horizons: audit 3500 (Exp-203-matched), evolution at
  the plateau horizon FIXED on a disclosed full-horizon pilot (L23). Runtime pre-flight
  (`ecology.runtime_budget.preflight`, L25) at the top of `main()` — the batch refuses to launch on a
  flagged (EXPLOSION/SUPERLINEAR/OVER_BUDGET) config.

- **Predeclared verdict (three-way, conjunct-by-conjunct; L1/L2 — every conjunct checked).**
  - **POSITIVE** (the bridge works — would be the FIRST functional sensor in the arc ⇒ candidate
    BREAKTHROUGH) iff in RESIDUE_COMPETE: **P1** determinism (same seed → identical events_hash);
    **P2** healthy pop (no collapse; per-arm pop ≥ a predeclared healthy floor, drift-flagged per
    L24); **P3 (CORE gradient)** pairwise s(0.10→0.15) > 0 AND invader_won ≥ 7/8 AND competitive
    optimum h* > 0.30; **P4 (CORE evolution)** evolution gene-pool newborn mean h > 0.30 in ≥ 4/5
    seeds; **P5 (MEDIATION)** FP-rate decreases with h AND the reproduction advantage is MEDIATED by
    fewer FPs (higher-h creatures show both lower FP-rate and higher realized reproduction); **P6
    (confound-clean)** CLAMPED_LR agrees (also > 0.30 / positive slope) AND NO_SHUFFLE does not flip
    the sign AND no direct-h-reward (asserted); **P7 (necessity contrast)** NO_RESIDUE stays
    primitive (h* ≤ 0.15 AND evolution < 0.15) AND RESIDUE_NO_COMPETE stays primitive.
  - **NEGATIVE** (the ceiling holds even for false-positive AVOIDANCE — the strongest, most general
    wall) iff RESIDUE_COMPETE h stays primitive (evolution < 0.15 in a majority AND/OR h* ≤ 0.15)
    DESPITE a real installed FP-avoidance benefit in the B-overlay (high-h demonstrably avoids the
    loss when gifted) — the gift exists and is un-earnable, a FIFTH wall.
  - **MIXED** iff partial: a real positive gradient ABOVE primitive but not functional ([0.15,0.30],
    the Exp 203 outcome), or positive only in a sub-arm / only Mode A not Mode B, or positive but
    weakly mediated.
  - Repo token: POSITIVE / NEGATIVE / MIXED.

- **Falsifiers / confounds (any ⇒ the named disposition).** **F1** non-determinism (same seed →
  different events_hash with flags off, OR OFF-path regression hash mismatch) → NEGATIVE (infra).
  **F2** populations collapse / clamps under-represented (L21) → NO_VERDICT → MIXED. **F3** NO_RESIDUE
  ALSO climbs > 0.30 → the climb is NOT residue-mediated → POSITIVE DISCARDED (confound). **F4** a
  positive only at collapsed pops (corr(pop,h) strongly negative, L24) → DRIFT, DISCARD. **F5** a
  positive that vanishes under CLAMPED_LR → memory substitution (L19), DISCARD. **F6** a positive
  that flips under NO_SHUFFLE → id-order artifact, DISCARD. **F7** h rises but FP-rate does NOT fall
  / the advantage is not FP-mediated (P5 fails) → the climb is via a different channel than the
  predeclared mechanism → reinterpret (NOT a clean POSITIVE for this bridge).

- **Honesty stakes (written before data).** The four walls (199–202) predict NEGATIVE, and Exp 203
  predicts at most MIXED — so a POSITIVE here is a genuine surprise, not a forced result. What is NEW
  is the PAYOFF STRUCTURE (avoiding a costly mistake, steeper than marginal foraging), which is why
  the experiment can honestly succeed. Residue params (`residue_yield`, `residue_decay`, `residue_loss`,
  `eat_threshold`, `fp_threshold`, `residue_confusion` grid) are tuned ONLY on disclosed pilots and
  FIXED before the verdict run; thresholds predeclared here. Founders + policy + costs PROVIDED; the
  sensor is a noisy-percept heuristic, not learned pymdp. Engine feature gated; exp194–203 hashes
  reproduce (regression guard in tests). A NEGATIVE is the strongest publishable form of the wall (the
  ceiling survives even a loss-avoidance payoff with competition-coupled residue).

**EXP 205 — FULL PRE-REGISTRATION (SURVIVABLE-LOSS RESIDUE SWEEP — resolving Exp 204's NO_VERDICT;
committed BEFORE any data, on the human's word "continue experimenting" 2026-06-13).**

- **Motivation.** Exp 204 found the residue/false-positive bridge creates the arc's FIRST functional
  MONOMORPHIC optimum (N* h*=0.60) at residue_loss=1.5 — but evolution did NOT climb AND 2/5
  populations COLLAPSED, so the evolution arm was under-licensed (NO_VERDICT/MIXED, F2). The open
  question (Exp 204's honest caveat, the core claim of its NEGATIVE lean): is there NO regime giving
  BOTH a functional optimum AND a survivable, climbing population — i.e., is the collapse-vs-functional-
  optimum tradeoff FUNDAMENTAL? This SWEEPS residue_loss across the survivable range to resolve it.
  Reuses the committed `enable_residue` engine (NO new feature); only residue_loss varies.

- **Hypothesis (one sentence).** The tradeoff is fundamental: at every SURVIVABLE residue_loss (where
  >=4/5 RESIDUE_COMPETE populations persist) the sensor stays primitive under evolution (newborn mean
  h < 0.15) and the local resident gradient is <=0, while the only losses with a FUNCTIONAL monomorphic
  optimum (h*>0.30) are the ones that COLLAPSE the population — so NO single loss gives both.

- **Design.** residue_confusion=0.6, residue_yield=1.0, residue_decay=0.05 (the Exp 204 med regime);
  SWEEP residue_loss in {0.5, 0.8, 1.0, 1.2, 1.5} (1.5 = the Exp 204 collapse anchor). Per loss:
  (1) monomorphic competitive optimum h* (carrying-capacity N*(h) over the clamp grid), RESIDUE_COMPETE
  — is the optimum functional?; (2) survival = RESIDUE_COMPETE evolution final pops (>=4/5 valid =
  survivable, L21); (3) evolution gene-pool NEWBORN mean h + functional fraction, RESIDUE_COMPETE vs
  NO_RESIDUE, fresh seeds {80-84}, horizon 8000 (Exp 204-matched); (4) the local pairwise s(0.10 vs
  0.15), RESIDUE_COMPETE, + a CLAMPED_LR check at the best-climbing loss. Runtime pre-flight (L25,
  require_safe=True) at the top of main().

- **Predeclared verdict (conjunct-by-conjunct).**
  - **POSITIVE** (the bridge works at a survivable loss — would be the FIRST functional EVOLVED sensor
    in the arc) iff SOME loss with >=4/5 valid RESIDUE_COMPETE pops gives: evolution newborn mean h >
    0.30 in >=4/5 seeds AND pairwise s(0.10->0.15) won >=7/8 AND CLAMPED_LR agrees AND NO_RESIDUE at
    that loss stays primitive.
  - **NEGATIVE** (the tradeoff is FUNDAMENTAL — confirms + sharpens Exp 204 into a clean fifth wall)
    iff: at every survivable loss (>=4/5 valid) evolution stays primitive (mean h < 0.15) AND every
    loss whose monomorphic optimum is functional (h*>0.30) collapses the population (<4/5 valid) — no
    loss gives both functional-optimum AND survivable-climb.
  - **MIXED** iff a real but sub-functional push at survivable losses (evolution above NO_RESIDUE but
    < 0.30 — the Exp 203/204 weak-gradient shape), without crossing functional.
  - **F1** non-determinism (same seed -> different events_hash) -> NEGATIVE (infra). **F2** all swept
    losses collapse (<4/5 valid everywhere) -> NO_VERDICT/MIXED (no survivable regime to test).
  - Repo token: POSITIVE / NEGATIVE / MIXED.

- **Honesty stakes.** This RESOLVES the Exp 204 NO_VERDICT (a known loose end I created by choosing the
  collapse-prone loss=1.5 for the fairest-shot monomorphic optimum) into a clean verdict; predicting
  NEGATIVE (the tradeoff is fundamental). Reuses the committed, guarded enable_residue engine — no new
  mechanism, no new degrees of freedom beyond the disclosed loss sweep. Fresh seeds {80-84} never used
  in Exp 204. Founders + policy + costs PROVIDED. A NEGATIVE makes the residue bridge a clean fifth
  wall: precision can be made genuinely valuable at the population level, but the very cost that makes
  it valuable forecloses the demographic room to evolve it.

**EXP 206 — PRE-REGISTRATION SKETCH (THERMAL-BARCODE NICHE / SYMPATRIC DIVERGENCE; staged; was Exp 205
before the survivable-loss sweep took the 205 slot).**
- **Motivation.** A single global band makes all chase one resource. A stronger sensor may need to
  unlock a DIFFERENT, less-crowded niche rather than compete harder for the same one.
- **World.** Resource classes with overlapping thermal signatures: niche A (easy, low reward, crowded),
  B (medium precision, medium/high reward), C (high precision, rare/high reward), + optional decoys.
  `w_i = Σ_j p_j(h_i,θ_i)·R_j/(1+N_j) − C_h − C_θ` (frequency-dependent: crowding N_j discounts a
  niche). A niche/lineage archive RECORDS occupied roles (never rewards h). Controls: single-niche,
  barcode-shuffled (signatures don't map to value), cost-off, archive-off, the clamp-grid audit.
- **Predeclared pass (NICHE_BRANCH_POSITIVE / LINEAGE_ONLY_POSITIVE allowed — global mean may stay
  moderate).** A stable high-sensor lineage with h>0.30; niche occupancy predicted by h (`I(h;niche)>0`);
  sensor knockout hurts that lineage; parent-weighted h elevated within the niche. A lineage-only
  positive counts ONLY if stable, heritable, ecological, and knockout-sensitive. **Fail:** no separable
  niches, or branch erased by global averaging with archive on, or only via confound.

**EXP 207 — PRE-REGISTRATION SKETCH (SENSOR–CONTROLLER CO-ADAPTATION BUNDLE; OPTIONAL, staged; was
Exp 206 before the survivable-loss sweep took the 205 slot).**
- **Motivation.** A stronger sensor may fail because the policy θ cannot exploit it: if θ≈0,
  `dw/dh ≈ −C_h'(h)` (selected against even though useful with the right θ).
  `w(h,θ) = R·σ(k·h·θ − d) − C_h(h) − C_θ(θ)`. Test θ-mutation alone / h alone / both independent /
  both bundled (correlated mutation raises the chance mutually-dependent partial traits co-occur — it
  must NOT guarantee success). **Pass:** h AND θ jointly rise; the h·θ interaction predicts
  reproduction; knockout of either damages the lineage; across seeds without collapse. **Fail:** h alone
  or θ alone explains it, or no joint rise.

---

**STATUS.** state: active · latest: Exp 204 (MIXED / NEW INSIGHT — the RESIDUE/FALSE-POSITIVE bridge: making precision avoid a costly mistake creates the FIRST functional MONOMORPHIC optimum in the arc [N* rises to h*=0.60 vs NO_RESIDUE 0.06], but it is still un-earnable — the LOCAL resident gradient is ≤0 [pairwise 0.10→0.15 wins 2/8, a fitness valley], evolution decays to 0.07 [0/5 functional, above NO_RESIDUE's 0.03], the false-positive cost COLLAPSES 2/5 populations [evolution arm under-licensed, F2], and mediation is cost-dominated [precise = fewer FPs but fewer babies]; blind-verified AGREE) · sub-arc SENSE-EVOLUTION (203–206); enable_residue gated + guarded · next-falsifiable: post-204 CONSULT — 205 niches / 206 controller / survivable-loss re-run / accept the five-regime answer

