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
- **Exp 197 — temperature as a hidden viability pressure** (isolated behind a config flag).
  FAILURE: temperature changes nothing measurable about survival/lineages.
- **Exp 198 — costly thermosense organ.** A sensor that costs energy becomes useful ONLY
  when temperature matters. FAILURE: the costly sensor is favored (or disfavored) identically
  whether or not temperature is a pressure.
- **Exp 199 — niche / species archive** from trait + behavior clustering (the evaluator
  RECORDS clusters, never selects them). FAILURE: no separable clusters under pressure.
- **Exp 200 — self-selection / mate choice** on observed traits, only after viability
  ecology works. FAILURE: mate choice changes no lineage outcome vs random pairing.
- **Exp 201 — neural Bθ / deep active-inference transition models inside lineages.**

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

**STATUS.** state: active · latest: Exp 196 (POSITIVE / NEW INSIGHT, POSITIVE-SINGLE — over a LONG horizon (5000 steps) aging DOES depress the standing population's mean complexity below a matched senescence-OFF control: gap 0.09–0.28 in 5/5 fresh seeds, progressive, emerging only after ~t=2000 — Exp 195's P6 selection null was a HORIZON limit, not an absence; blind-verified. Mechanism under-claimed: standing-population metric conflates heritable selection with survivor bias + density — newborn-complexity tracker is the named follow-up) · depends-on: ecology/ substrate (Exp 194–196) · reusable: ecology engine (genotype/phenotype/world/policy + senescence flag, pluggable for pymdp nav) · why: opens the population axis distinct from the locked ladder (human's "N5 population ecology" = ladder-N7 territory, kept parallel) · next-falsifiable: disentangle selection vs survivor bias by tracking NEWBORN / lineage + age-stratified complexity over the long horizon (does the gene pool, not just the standing snapshot, shift?); OR Exp 197 — temperature as a hidden viability pressure. FAILURE (disentangle) = newborn complexity does NOT fall with the standing population ⇒ the long-horizon effect was survivor bias, not heritable selection.
