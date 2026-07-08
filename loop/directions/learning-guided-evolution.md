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

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE; re-confirm the number at
launch — a live cron auto-syncs this shared checkout.)

**PROGRESS: Exp 275 DONE — POSABLE (POSITIVE / NEW INSIGHT, blind-verified AGREE, 2026-07-02).** Per
L28/L45 the original "build theta" rung was SPLIT: Exp 275 became a CHEAP posability pre-flight (no engine
surgery) and the theta BUILD moved to Exp 276. Result: the live-sensor channel IS separably load-bearing
for band-tracking (in-band occupancy 0.48 GOOD vs 0.23 BLIND, ~2.0x; map can't substitute) AND theta (proxied
by `band_responsiveness`, the faithful non-scale-invariant use-quality knob — `thermal_avoidance_weight`
was an argmax-scale-invariant trap, L44) has a large bad→good range that GATES survival (worst theta 8/8
extinct, good theta 0/8). So the theta-Baldwin test is posable; rung 2 (the build) is licensed. Instrument
lessons: occupancy over intake when the substrate is demographically unstable; faithful vs scale-invariant
proxy (L50).

1. ~~**Exp 275 — posability pre-flight (was: build theta).**~~ DONE above (POSABLE). The BUILD it originally
   described is folded into Exp 276.

**RE-ANALYSIS NOTE (2026-07-06, adversarial re-read of Exp 275 + an L28 pre-flight).** Exp 275's occupancy
and intake metrics are DENSITY-CONFOUNDED (better tracking → higher survival → crowding → per-capita intake
INVERTS: cost-on, the worst theta 0.05 posts the HIGHEST intake because 7/8 of its populations go extinct and
survivors gorge; Exp 202 live), and its S3 survival gradient is MONOMORPHIC, not invasion (L41). BUT the
pessimistic reading (density competes the advantage away) was REFUTED by a cheap cost-on equilibrium pre-flight:
R\* (mean standing resource; lower = better competitor, Tilman) drops MONOTONICALLY with theta
(2.07→1.87→1.70→1.52→1.46 over theta 0.15→0.80) while N\* rises (82→90), STABLE (0/3 extinct) — a clean
equilibrium competitive-dominance gradient WITH COST ON. Also confirmed: the sensor's occupancy separability
(S1) survives cost-on (2.00x, ~unchanged from 2.07x). So the fitness-posability *appeared* to lean positive —
but the 2026-07-07 CORRECTION below REVERSES how to read that (kept both notes per VALIDATION: no retconning).

**CORRECTION NOTE (2026-07-07, premise audit — the decisive reframe; two verified facts invert the test).**
(i) `band_responsiveness` (theta) is UNCOSTED and a CONFIG scalar, not heritable (verified: `creature.py:589`
scales the tracker EMA alpha only — `alpha=clamp(intensity*resp,0,1)`; no energy/upkeep anywhere; default 1.0).
So every pre-flight above measured a FREE beneficial trait, and a free trait is not a wall-trait. (ii) The
Baldwin effect requires the LEARNED trait's DIRECT genetic gradient to be FLAT/VALLEYED — a WALL — that
learning then rescues (Hinton & Nowlan). Therefore a smooth/monotone gifted gradient on theta is **NOT good
news**: it is a YELLOW FLAG that innate theta may be DIRECTLY EVOLVABLE, which would VOID the direction
(nothing for learning to guide). The R\* pre-flight is doubly discounted — R\* is grid-mean AVAILABILITY, the
repo's own wrong-KIND-of-measure (Exp 244/247: the correct arbiter is density-dependence of per-capita intake
at FIXED density, e.g. Exp 240's fixed-N probe) — AND a clean gradient argues AGAINST a wall, not for the
direction. NET: theta must be made HERITABLE **and COSTED** (L30-calibrated, mirroring `memory_cost_slope`),
and the binding test's polarity is INVERTED relative to the pre-correction 276a (superseded below).

2a. **Exp 276a — build costed+heritable theta, then ESTABLISH THE WALL (the learn-OFF arm; CORRECTED polarity).**
   Promote `band_responsiveness` to a mutable GENOTYPE field AND add an L30-calibrated per-step COST
   (`floor + slope*theta`, mirroring `memory_cost_slope`; calibrate the slope to theta's empirical benefit
   ceiling so neither verdict is foreordained, L30), gated byte-identical OFF + golden-hash. Then run
   invasion-from-rarity on innate costed theta **WITHOUT learning**, via the Evolvability Preflight binding gate
   — drift-robust selection slope + invasion-from-rarity + a **fixed-density per-capita-intake** benefit curve
   (NOT R\*/availability, the Exp 244/247 wrong-KIND trap), per L29/L41. **CORRECTED POLARITY (this is the
   whole point of the 07-07 audit):**
   - **Innate theta INVADES from rarity / has a usable realized gradient ⇒ NO WALL ⇒ theta is directly evolvable
     ⇒ the Baldwin premise is VOID** (learning cannot "guide" evolution across a wall that isn't there) ⇒ close
     the direction NEGATIVE, honestly: the reframe from h to theta dissolved the phenomenon; the wall stays on
     h (sensor precision), which is unlearnable-by-construction. (A too-clean R\* gradient was the tell.)
   - **Innate costed theta is WALLED — gifted-good yet does NOT invade from rarity (the h pattern, L22/L41)
     ⇒ a wall exists for learning to potentially cross ⇒ proceed to 276b.**
   This arm IS 276b's learn-OFF control — keep them ONE experiment (OFF establishes the wall, ON tests the
   rescue); do not green-light on a naive "it invades → build."

2b. **Exp 276b — the binding Baldwin test: same-axis assimilation vs substitution vs shielding.** ONLY if 276a
   found the WALL (innate costed theta gifted-good but does NOT invade). Build the learnable `theta` behind
   `enable_learnable_use` (a within-life-learned tracking-use `theta`
   from the heritable prior of 276a; byte-identical OFF + golden-hash), keeping the theta gradient inside the
   STABLE band. Then, in the Exp 205
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

**STATUS.** state: active (rung 1 POSABLE) · latest: Exp 275 (POSITIVE / NEW INSIGHT, blind-verified, 2026-07-02) — posability pre-flight: the sensor channel is separably load-bearing for band-tracking (~2.0x sensor vs sensorless; map can't substitute) AND theta (band_responsiveness) has a bad→good range gating survival → a same-axis theta-Baldwin test IS posable · depends-on: ecology/ + Exp 205 forage regime + evolvability Preflight + CLAMPED_LR/band controls + an unbuilt theta channel (Exp 276) · reusable: Exp 275 occupancy instrument + faithful-proxy discipline (L50) · next-falsifiable: Exp 276a — build COSTED+heritable theta, invasion-from-rarity WITHOUT learning; polarity (07-07 fix): invades ⇒ no wall ⇒ VOID; walled ⇒ proceed to 276b learn-ON.
