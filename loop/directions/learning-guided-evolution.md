# direction: learning-guided-evolution

**Question.** The local-gradient wall (Exp 199–247) is an evolution result: a large functional optimum
exists (gifted/forced benefit is big — Exp 200's forced forager breeds ~4×; Exp 205's monomorphic optimum
h*=0.60 is survivable and bulk-fitter) but the *genetic* gradient toward it is flat/valleyed, so small
mutational steps don't pay and the trait never evolves. The program has already coupled within-life
learning with evolution — and found it acts as a **SUBSTITUTE**: the ecology creature carries an EMA-learned
resource map (`ecology/creature.py:184`, heritable `learning_rate`) and the Exp 201 band tracker, and the
`CLAMPED_LR` control (`experiments/exp201_n5_increasing_returns.py:162`) showed freezing that learning
doesn't change the null for the *sensor* gene — learning a **different** channel (map memory) to skip
evolving the sensor, not learning that guides the sensor's evolution. **The genuinely untested cell is the
Baldwin/assimilation cycle on a SINGLE axis: if the ability to USE the sensor (`theta`, the reserved
controller trait in `ecology/sense_axis.py:15`) is BOTH learnable within life AND heritable, does selection
canalize it — does the innate `theta` climb over generations (genetic ASSIMILATION) where a direct
`theta`-mutation could not — or does plasticity mask the gene and stall it (the shielding/substitution
result the arc already leans toward)?**

**Why it matters.** The wall is the program's dominant negative finding, and lifetime learning is the
classical, literature-backed escape from exactly this landscape — Hinton & Nowlan (1987), "How learning can
guide evolution": a good genotype exists but the genetic gradient is flat until you reach it, and genetic
search finds it ONLY when lifetime learning lets near-correct genotypes reach the optimum behaviorally,
manufacturing a smooth gradient toward learnable genotypes, which selection then assimilates. Exp 205 is the
same shape. **But the escape is real only if the LEARNED axis and the EVOLVED axis are the SAME** — learn
`theta`, assimilate innate `theta`. Learning a *different* axis (map memory) to avoid needing the sensor is
SUBSTITUTION, which the arc already saw (Exp 201/202) and which is NOT Baldwin. If same-axis assimilation
occurs, the wall was "evolution searching without a within-life guide," and the program gains a general
valley-crossing mechanism. If it STALLS or SHIELDS, that is an equally sharp named result (plasticity's cost
/ masking is real here) and it tightens the wall. It also probes the PREMISE: the RECIPE grants "ONE innate
anchor" as *provided*; a heritable, learnable `theta` is an anchor that is itself *under selection* — asking
whether the recipe's anchor can be **evolved** rather than gifted, a disciplined probe at the edge of the
closed "fully tabula-rasa" ceiling (open_problem.html) without claiming to cross it.

**Not this (binding framing).**
- **NOT SUBSTITUTION (the arc's caught confound — the crux).** The learned axis and the assimilated axis
  must be the SAME trait (`theta`). Learning the resource map `m` / the band tracker to skip evolving the
  sensor is the already-documented resource-memory substitution (`CLAMPED_LR`, `freeze_learning_rate`, the
  band-staleness free-read control) — a NEGATIVE for Baldwin, never a positive. Every Baldwin rung inherits
  those existing controls; it does not invent fresh ones.
- **NOT Lamarckian.** A parent's *learned* `theta` is NEVER written back to its genotype. Offspring inherit
  only the genetic `theta` prior; learned state is discarded at reproduction. Gene frequencies change ONLY
  through individual reproduction/death. (A separately-labelled social/cultural-inheritance rung may test
  inheritance of *learned* info as a DISTINCT mechanism — never smuggled into the Baldwin rungs.)
- **NOT providing the trait.** The heritable knob is `theta`'s prior / learnability, not its optimal value.
  Trivial-provision guard: if assimilation only appears when the optimum is gifted rather than
  learned-then-selected, that is NEGATIVE.
- **NOT an external fitness evaluator.** Selection stays individual-event-based; no global ranker picks,
  protects, or teleports the better learners.
- Every new path `enable_*`-gated, byte-identical OFF, golden-hash guarded; one new mechanism per iteration;
  deterministic under fixed seed.

**The provided-vs-earned line (the crux).** The genome MAY provide the *capacity* to learn and a heritable
*prior* on `theta` (how well it starts / how fast it learns to use the sensor) — mirroring the recipe's one
innate anchor. It MAY NOT provide `theta`'s optimal value. The Baldwin claim is: **selection on learnability
drives genetic change in the SAME axis a direct mutation could not.** Load-bearing readouts: (1) a
learning-OFF arm that byte-reproduces the prior wall NEGATIVE; (2) a genotype-vs-phenotype split (assimilated
innate `theta` vs merely-learned `theta`) so a shielding stall — good behavior, unmoved genes — is never
misread as a climb; (3) the inherited substitution controls (CLAMPED_LR / freeze_lr / band free-read) so a
map-memory free-ride is never misread as `theta` assimilation.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE; next number = Exp 275 — but
re-confirm at launch; a live cron auto-syncs this shared checkout)

1. **Exp 275 — a learnable AND heritable `theta`, separable from the caught confounds (plumbing/build).**
   `theta` is reserved-but-unimplemented (`ecology/sense_axis.py:15`); build it behind `enable_learnable_use`:
   a heritable `theta` prior + a within-life update of `theta` from the sensor channel (how to act on
   thermosense cues), starting from the genetic prior. Verify: (a) OFF == prior ecology byte-identical
   (golden-hash); (b) a creature gifted the *best* `theta` prior reaches the documented forced forage benefit
   in-population (learning works here); (c) learnable-`theta` is provably SEPARABLE from the existing map
   memory `m` and the band-staleness tracker — the CLAMPED_LR / freeze_lr / band free-read controls must move
   the `theta` readout but not vice-versa (L38/L39 manipulation check + the substitution controls). **FAIL** =
   learner inert even when gifted the best prior (substrate can't express learn-to-use); OR `theta` is
   inseparable from map-memory substitution (the confound the whole direction must avoid) ⇒ fix the substrate
   before any evolution batch; NOT a Baldwin result.

2. **Exp 276 — the binding test: same-axis assimilation vs substitution vs shielding.** In the Exp 205
   survivable-loss thermosense-forage regime (`ecology/sense_axis.py`; optimum functional & bulk-fitter,
   evolution stays primitive 0/5), run evolution of innate `theta` with learn-to-use ON vs OFF (OFF
   byte-reproduces the prior NEGATIVE), Preflight (`ecology/evolvability/`) measuring the binding LOCAL
   gradient in both arms, WITH the inherited CLAMPED_LR/band controls in every arm. **Predeclare all
   branches (the crux):**
   - **PASS (Baldwin):** innate `theta` climbs toward functional over generations, beating learning-OFF by a
     declared margin at ≥ the seed threshold, Preflight gradient flat (OFF) → climbable (ON), AND the climb
     survives the CLAMPED_LR/band controls (not map-memory substitution) AND the genotype (not just phenotype)
     moves (not shielding).
   - **FAIL-STALL:** learning-ON stays primitive like OFF ⇒ learning does not guide `theta` evolution here.
   - **FAIL-SHIELDING:** behavioral `theta` reaches the optimum but innate `theta` does NOT assimilate /
     regresses vs OFF ⇒ the hiding effect. A DISTINCT named NEGATIVE.
   - **FAIL-SUBSTITUTION:** the apparent climb collapses under CLAMPED_LR / band free-read ⇒ it was map-memory
     substitution (the arc's known confound), NOT `theta` assimilation. (Watch the mean-of-opposites trap:
     check per-seed dispersion + bimodality before calling any arm "converged.")

3. **Exp 277 — is the assimilation earned? (cost of learning).** ONLY if Exp 276 PASSES. Add a learning cost
   (time-to-competence, per-step energy, mistakes during the naive period — reuse Exp 204's false-positive
   machinery). Predeclared: does assimilation still occur when learning is costed — do genes that reduce the
   need to learn (canalize `theta`) get favored, the full Baldwin→assimilation signature — or does the cost
   re-erect the wall (L20/L22: costs kill sub-threshold benefits)? **FAIL** = the climb exists only at zero
   learning cost ⇒ cost-fragile, the same shape as the wall it claimed to cross.

4. **Exp 278 — the doubly-empty cell: learning × frequency-dependence.** ONLY if 276–277 PASS. Combine the
   two known valley-crossers — put learnable-`theta` learners under a frequency-dependent / co-evolving
   regime (patch-mosaic Red Queen Exp 259–262, or the emergent-contest substrate Exp 263–267). Predeclared:
   does the combination produce what neither alone does (faster assimilation; a learned strategy co-evolution
   canalizes; an arms-race in learnability) vs the two main-effect controls (learning-only,
   frequency-dependence-only)? **CEILING (binding):** no claim of culture / social learning as a second
   inheritance channel beyond the toy — the honest claim is a measured interaction vs its main effects.
   **FAIL** = the interaction is indistinguishable from the additive main effects.

**Transfer (why it's a real experiment, and what the port actually costs).** The prediction is genuinely
two-sided: Hinton & Nowlan (1987) / West-Eberhard plasticity-first predict learning ACCELERATES assimilation
over a flat landscape (PASS); Mayley (1996, cost of the Baldwin effect) and the shielding/hiding effect
predict plasticity can MASK genotypes and stall assimilation, especially when learning is cheap and
near-perfect (Exp 272–273 showed the learner can be) — FAIL-SHIELDING. The arc's own Exp 201/202 substitution
finding is a third, substrate-native failure mode the standard Baldwin literature doesn't foreground.
**Port cost is real, not cheap:** the Exp 272–273 model-based learner is NOT a reusable module — it lives in
84 KB experiment scripts (`experiments/exp272_learnable_actuator.py`) on the identity *gridworld* substrate
(refuge planning, no reproduction), so it must be extracted and bridged into the `ecology/` population loop,
and a per-agent learner across a large population (~950 agents × steps × seeds × arms) is a compute-class
jump over the fixed-policy ecology. Run `compute-batch-runtime-preflight` before the full batch. What IS
cheap and ready: the wall regime + its flat-gradient instrument (`ecology/sense_axis.py`,
`ecology/evolvability/`) and the substitution controls (CLAMPED_LR / freeze_lr / band free-read) all exist on
this branch.

**Stop condition.** Exhausted when Exp 275 (expressibility + separability) + Exp 276 (the
PASS/STALL/SHIELDING/SUBSTITUTION verdict) return a clear result: if PASS, run Exp 277 (cost) then Exp 278
(interaction), then distil a MechanismCard (learning-guided-evolution) + extend the local-gradient-wall
BoundaryNote with the learning-escape result; if any FAIL branch, write the named NEGATIVE + its
genotype/phenotype + substitution-control evidence to EXPERIMENTS.md and close with a BoundaryNote. Either way
the verdict + the learning-OFF control + the per-seed dispersion go to EXPERIMENTS.md.

**STATUS.** state: proposed (not yet run; drafted 2026-07-02 from a cross-arc lacuna analysis, then revised after a repo audit found the ecology creature ALREADY has within-life EMA learning + the CLAMPED_LR substitution control, so the axis was retargeted from the hardware trait h to the reserved controller trait theta) · latest: none · depends-on: ecology/ population loop + a documented local-gradient-wall regime (Exp 205 thermosense-forage) + ecology/evolvability/ Preflight + the CLAMPED_LR/freeze_lr/band substitution controls + a NEW learn-to-use theta channel (reserved, unbuilt) + an EXTRACTED learner from Exp 272–273 (not a module) · reusable: TBD (target: a gated enable_learnable_use theta channel reusable across trait axes) · why: the wall is evolution-only, and the ecology arc coupled learning as SUBSTITUTION (201/202); the untested cell is same-axis Baldwin ASSIMILATION of a learnable+heritable theta · next-falsifiable: Exp 275 — a gifted-best-theta learner reaches the forced forage benefit in-population, byte-identical OFF, and learnable-theta is separable from the map-memory/band confounds.
