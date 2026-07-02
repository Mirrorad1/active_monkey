# direction: learning-guided-evolution

**Question.** The local-gradient wall (Exp 199–247, general across senses, memory, active sensing,
locomotion, hearing) is an **evolution-only** result: a large functional optimum exists (gifted/forced
benefit is big — Exp 200's forced forager breeds ~4×; Exp 205's monomorphic optimum h*=0.60 is
survivable and bulk-fitter) but the *genetic* gradient toward it is flat/valleyed, so small mutational
steps don't pay and the trait never evolves. **Does adding WITHIN-LIFETIME LEARNING of the trait's use —
so a partially-correct genotype can behaviorally reach the optimum during its life — create a selective
gradient the genetic landscape lacked, and drive genetic change toward the functional trait (the Baldwin
effect / genetic assimilation)? Or does plasticity instead MASK the gene from selection and stall it (the
shielding / hiding effect)?** This is the one cell the program has never filled: every ecology experiment
(194–268) has populations + selection + zero lifetime learning (agents are genetically-parameterized fixed
policies); every learning experiment (`creature/`, the M4a affect dyad, the Exp 270–274 identity learner)
has lifetime learning + no reproduction/selection. The two research halves have never shared a substrate.

**Why it matters.** The wall is the program's dominant negative finding, and lifetime learning is the
classical, literature-backed escape from *exactly* this landscape — Hinton & Nowlan (1987), "How learning
can guide evolution," is a needle-in-a-haystack fitness function (a good genotype exists but the genetic
gradient is flat until you reach it) that genetic search cannot find UNLESS lifetime learning lets
near-correct genotypes find the optimum behaviorally, manufacturing a smooth gradient toward learnable
genotypes. Exp 205 is the same shape (functional, survivable, bulk-fitter optimum, still un-evolvable
because small steps don't pay). If learning crosses the wall here, the wall was never about the trait —
it was about **evolution searching without a within-life guide**, and the program gains a general
valley-crossing mechanism. If it STALLS or SHIELDS, that is an equally sharp, named result (plasticity's
cost / masking is real at this substrate), and it tightens the wall rather than loosening it. It also
touches the PREMISE directly: the RECIPE grants "ONE innate anchor + taught labels" as *provided*; a
heritable **learning bias** is an innate anchor that is itself *under selection* — so this asks whether
the recipe's anchor can be **evolved** rather than gifted, a disciplined probe at the edge of the closed
"fully tabula-rasa" ceiling (open_problem.html) without claiming to cross it.

**Not this (binding framing).**
- **NOT Lamarckian.** A parent's *learned* state is NEVER written back to its genotype. Offspring inherit
  only the genetic learning-bias/prior; the learned weights are discarded at reproduction. Gene
  frequencies change ONLY through individual reproduction/death events. (A separate, clearly-labelled
  social/cultural-inheritance rung may test inheritance of *learned* information as a DISTINCT mechanism —
  never smuggled into the Baldwin rungs.)
- **NOT providing the trait.** The heritable knob is a learning *bias / rate / prior* (how easily this
  genotype learns to use the trait), NOT the functional trait value. Trivial-provision guard: if the
  "evolved" functional trait only appears when the optimum is gifted rather than learned-then-selected,
  that is NEGATIVE.
- **NOT an external fitness evaluator.** Selection stays individual-event-based (reproduce/die); no global
  ranker picks, protects, or teleports the better learners.
- Reuse the existing substrate; every new path `enable_*`-gated, byte-identical OFF, golden-hash guarded;
  one new mechanism per iteration; deterministic under fixed seed.

**The provided-vs-earned line (the crux).** The genome MAY provide (a) the *capacity* to learn and (b) a
heritable *bias/prior* that makes the trait easier or faster to learn — mirroring the recipe's one innate
anchor. It MAY NOT provide the trait's optimal value. The Baldwin claim is specifically: **selection on
learnability drives genetic change a direct trait-mutation could not.** So the load-bearing control is a
learning-OFF arm that byte-reproduces the prior wall NEGATIVE, plus a genetic-vs-behavioral split readout
(assimilated *genotype* vs merely-learned *phenotype*) so a shielding stall (good behavior, unmoved genes)
is never misread as a Baldwin climb.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE; next number = Exp 275)

1. **Exp 275 — lifetime-learning substrate + expressibility gate (plumbing, consolidation).** Port a
   within-lifetime learner into an `ecology/` agent behind `enable_lifetime_learning` (reuse the Exp 272–273
   model-based learner and/or a minimal tabular learner over the `ecology/sense_axis.py` foraging trait).
   The learner adjusts the *use* of a genetically-scaffolded trait during life, starting from a heritable
   prior. Verify: (a) OFF == prior ecology byte-identical (golden-hash); (b) a creature gifted the *best
   learning bias* reaches the documented forced benefit in-population (learning actually works here); (c) the
   heritable knob is a learning bias — a creature with a bad bias learns worse — and does NOT shortcut to
   providing the trait value (L38/L39 manipulation check + trivial-provision guard). **FAIL** = learner inert
   in-population even when gifted the best bias (substrate can't express lifetime learning), OR the knob
   games the expressibility metric by providing the trait ⇒ fix the substrate before any evolution batch;
   NOT a Baldwin result.

2. **Exp 276 — the binding test: does learning cross the wall?** Take a documented local-gradient-wall
   regime with a large-but-unevolvable optimum — the purest is **Exp 205's survivable-loss thermosense-forage**
   (`ecology/sense_axis.py`, monomorphic optimum functional & bulk-fitter, evolution stays primitive 0/5) —
   and run the SAME evolution with lifetime learning ON vs OFF (OFF byte-reproduces the prior NEGATIVE). Use
   the `ecology/evolvability/` Preflight to measure the binding LOCAL gradient in both arms. **Predeclare BOTH
   failure branches (this is the crux):**
   - **PASS (Baldwin):** the heritable learning-bias AND the assimilated genetic trait climb toward functional
     over generations, beating the learning-OFF control by a declared margin at ≥ the seed threshold, with the
     Preflight gradient turning from flat (OFF) to climbable (ON).
   - **FAIL-STALL:** learning-ON stays primitive like OFF ⇒ learning does not guide evolution here (a clean
     NEGATIVE — log it; the wall is deeper than "no within-life guide").
   - **FAIL-SHIELDING:** the *behavioral* phenotype reaches the optimum (learners forage well) but the
     *genetic* trait does NOT assimilate / regresses vs OFF ⇒ the hiding effect (plasticity masks the gene
     from selection). A DISTINCT named NEGATIVE, never reported as a positive. (Watch the mean-of-opposites
     trap: check per-seed dispersion + bimodality before calling either arm "converged.")

3. **Exp 277 — is the assimilation earned? (cost of learning).** ONLY if Exp 276 shows a Baldwin climb.
   Add a cost to learning (time-to-learn, per-step energy, mistakes during the naive period — reuse Exp 204's
   false-positive-cost machinery). Predeclared: does genetic assimilation still occur when learning is costed —
   i.e. do genes that *reduce the need to learn* (canalize the trait) become favored, the signature of the
   full Baldwin→assimilation cycle — or does the learning cost re-erect the wall (L20/L22: costs kill
   sub-threshold benefits)? **FAIL** = the climb exists only at zero learning cost ⇒ the escape is cost-fragile,
   the same shape as the local-gradient wall it claimed to cross.

4. **Exp 278 — the doubly-empty cell: learning × frequency-dependence.** ONLY if 276–277 are positive. The
   two known valley-crossers combined — put the learners under a frequency-dependent / co-evolving regime
   (reuse the patch-mosaic Red Queen, Exp 259–262, or the emergent-contest substrate, Exp 263–267) so payoff
   is interactive AND the trait is learned. Predeclared: does the combination produce what neither alone does
   (faster assimilation; a learned strategy that co-evolution then canalizes; an arms-race in *learnability*)
   measured against the two main-effect controls (learning-only, frequency-dependence-only)? **CEILING
   (binding):** no claim of culture / social learning as a second inheritance channel beyond the toy — the
   honest claim is a measured interaction effect vs its two main effects. **FAIL** = the interaction is
   statistically indistinguishable from the additive main effects (they don't synergize here).

**Transfer (why this is a real experiment, not a foregone escape).** From the adjacent literature the
prediction is genuinely two-sided: Hinton & Nowlan (1987) and West-Eberhard's plasticity-first evolution
predict learning *accelerates* genetic assimilation over a flat landscape (PASS); Mayley (1996) on the cost
of the Baldwin effect and the shielding/hiding effect predict plasticity can *mask genotypes from selection
and stall or prevent* assimilation, especially when learning is cheap and near-perfect (which Exp 272–273
showed the learner can be) — that is FAIL-SHIELDING, and it is why Exp 276 must split genotype from
phenotype and Exp 277 must cost the learning. Within the project the transfer is cheap: the learner already
exists (Exp 272–273 model-based learner; the pymdp active-inference patterns from Exp 21/26/30/34/35), the
population loop already exists (`ecology/`, heritable genotype→phenotype, snapshot/restore), and the
wall-regime + its flat-gradient instrument already exist (`ecology/sense_axis.py`, `ecology/evolvability/`).
The port is: drop the learner into an ecology agent, make the heritable knob a learning bias, re-run Exp 205
with the Preflight watching the gradient. (Compute note: N learners in a population is a real cost — run the
`compute-batch-runtime-preflight` before the full batch.)

**Stop condition.** Exhausted when Exp 275 (expressibility) + Exp 276 (the binding PASS/STALL/SHIELDING
verdict) return a clear result: if PASS, run Exp 277 (cost) then Exp 278 (interaction), then distil a
MechanismCard (learning-guided-evolution) + extend the local-gradient-wall BoundaryNote with the
learning-escape result; if FAIL-STALL or FAIL-SHIELDING, write the named NEGATIVE + its genotype/phenotype
evidence to EXPERIMENTS.md and close with a BoundaryNote ("lifetime learning does not / cannot guide genetic
evolution across the local-gradient wall at this substrate because <stall vs shielding vs cost>"). Either way
the verdict + the learning-OFF control evidence + the per-seed dispersion go to EXPERIMENTS.md.

**STATUS.** state: proposed (not yet run; drafted 2026-07-02 from a cross-arc lacuna analysis) · latest: none · depends-on: ecology/ population loop + creature/ model-based learner (Exp 272–273) + a documented local-gradient-wall regime (Exp 205 thermosense-forage) + ecology/evolvability/ Preflight · reusable: TBD (target: a gated enable_lifetime_learning learner reusable across trait axes) · why: the local-gradient wall is an EVOLUTION-ONLY wall (199–247); lifetime learning (the Baldwin effect) is the untested classical escape from exactly that flat-gradient landscape, and the program's learning half and evolution half have never shared a substrate · next-falsifiable: Exp 275 expressibility — a gifted-bias learner reaches the forced benefit in-population, byte-identical OFF, knob is a learning bias not the trait value.
