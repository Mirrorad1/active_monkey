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
- **Exp 195 — temperature as a hidden viability pressure** (isolated behind a config
  flag). FAILURE: temperature changes nothing measurable about survival/lineages.
- **Exp 196 — costly thermosense organ.** A sensor that costs energy becomes useful
  ONLY when temperature matters. FAILURE: the costly sensor is favored (or disfavored)
  identically whether or not temperature is a pressure.
- **Exp 197 — niche / species archive** from trait + behavior clustering (the evaluator
  RECORDS clusters, never selects them). FAILURE: no separable clusters under pressure.
- **Exp 198 — self-selection / mate choice** on observed traits, only after viability
  ecology works. FAILURE: mate choice changes no lineage outcome vs random pairing.
- **Exp 199 — neural Bθ / deep active-inference transition models inside lineages.**

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
  demonstrated adaptation; specialization/niches are deferred to Exp 197.

**STATUS.** state: active · latest: Exp 194 (MIXED / NEW INSIGHT — ecology substrate established, blind-verified; predeclared P5 starvation-death-fraction sub-metric ill-posed, post-hoc substitution refused) · depends-on: ecology/ substrate (new) · reusable: ecology engine (genotype/phenotype/world/policy, pluggable for pymdp nav) · why: opens the population axis distinct from the locked ladder (human's "N5 population ecology" = ladder-N7 territory, kept parallel) · next-falsifiable: Exp 195 — temperature as a hidden viability pressure behind a config flag, with a WELL-POSED pressure metric (a denominator that varies / a second death cause), a baseline (random-policy or zero-mutation cohort), and a counterfactual control; FAILURE = temperature changes nothing measurable about survival or lineages.
