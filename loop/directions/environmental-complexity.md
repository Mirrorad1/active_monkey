# direction: environmental-complexity

**Question.** The closed `population-ecology` / `evolvability-geometry` arc established a robust
**local-gradient wall**: across ~8 regimes (sensing, foraging, increasing-returns, competition,
passive memory/inference, active sensing, payoff geometry — Exp 199–213) NO costed trait became a
functional organ, because a *crude* version grabs the easy part of the benefit (saturation) faster
than cost falls, so the next small mutational step never pays. Every one of those regimes varied
SENSING / MEMORY / TOLERANCE / ORGAN-UPKEEP / REPRODUCTION. **Movement was never an evolvable axis**
(creatures move on a flat 5-action grid and teleport to a random cell on reproduction). This
direction asks the last structurally-distinct escape question: **if the environment is made
progressively more complex — terrain with elevation and dropoffs, where the richest food is
physically sealed behind a barrier and reaching it costs energy — does a costed, heritable
LOCOMOTION trait become locally evolvable (creatures develop better ways to move), or does the
benefit-magnitude wall generalise from senses/memory to movement too?**

**Why it matters.** It is the cleanest test of whether the wall is a *substrate-poverty* fact (the
2D flat grid was too legible — a crude mover already does nearly as well) or a *fundamental* fact of
this evolutionary regime. A POSITIVE would be the FIRST escape of the local-gradient wall and the
first emergent movement specialization in the project; a NEGATIVE extends the wall to locomotion and
sharpens it (a discretely-unlocking, non-saturating, above-resident-fit affordance still un-evolvable).
It also opens the path the human named — multi-need survival (thermosense + terrain coupled) and,
eventually, genuine 3D/multi-layer movement — but only once a lower rung shows the wall is escapable.

**The provided-vs-earned line (binding, the crux).** Richer terrain is trivially "useful when
gifted": of course a forced strong climber that can reach sealed food out-reproduces one that
cannot (L22: forced/gifted benefit does NOT predict evolvability; L32: works-when-imposed ≠ has an
evolvable benefit ceiling). The honest test is the **local selection gradient at the resident** via
the Evolvability Preflight (Gate C): resident `climb_ability`=0.05 → mutant 0.10, ≥7/8 invader-wins
across fresh seeds to PASS. The trait must be COSTED (floored upkeep, monotone in intensity) and the
richest regen must be genuinely SEALED behind the barrier (a crude mover gets none of it). The
load-bearing controls: a **flat-map** control (terrain ON but zero relief ⇒ the trait gates nothing,
pure cost ⇒ gradient must be ≤0) and a **gate-open** control (relief present but everyone treated as
a perfect climber ⇒ if intake is unchanged, the food was never withheld and the substrate is too
legible). The escape is a GATE, never a graded slope TAX — a tax is exactly the saturating regime
already refuted (a crude mover just pays a bit more and still eats).

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE)

1. **Sealed-plateau terrain + costed `climb_ability` (single need: energy).** A STATIC 2.5D elevation
   field: a reachable low basin around the founder spawn, ringed by one ridge, behind which a high
   plateau carries concentrated regen (conserved-total mask, mirroring the thermal-band food
   coupling). A heritable, costed `climb_ability ∈ [0,1]` stochastically gates upslope crossings via
   a **calibrated sigmoid ramp** (each ε buys a small increase in crossing odds — an honest local
   gradient, not a binary jump). Founder basin tuned **thin-survival** (pressured but ≥50% seeds
   viable). **Predeclared (run BEFORE the gradient batch — L38 manipulation check):** flood-fill over
   the expected-crossing graph must show (a) ≥30% of steady-state regen sealed behind the ridge at
   the resident, AND (b) the marginal reachable-food unlocked across the 0.05→0.10 step is LARGE
   relative to the upkeep cost slope (non-saturating at the resident, unlike Exp 199–201). Then the
   binding verdict: Gate C local gradient ≥7/8. **FAIL** = FAIL_LOCAL_GRADIENT (<7/8 at the resident
   step even when the manipulation check passes AND a gifted strong climber is fitter) ⇒ the wall
   extends to locomotion. **NULL/INVALID (abort+fix, not a wall result):** manipulation check fails
   (too-legible substrate); Gate-G byte-identity fails (a steering/cost channel leaks); extinct
   fraction ≥0.5 (ridge too harsh — retune basin regen, cf. Exp 205); the flat-map OR gate-open
   control shows a positive gradient (forbidden side-channel).

2. **Staircase of plateaus + depletion.** Replace the single ridge with rising contour bands so each
   ε of `climb_ability` unlocks the NEXT band; deplete each plateau so sustained access (not a
   one-time crude flood) is required; engineer the marginal-reachable-food curve flat-or-rising across
   [0.05, 0.6]. **FAIL** = gradient still ≤0 once a lineage breaches band 1 and the rest re-saturate ⇒
   even a non-saturating staircase with depletable rewards can't make the marginal step pay (the wall
   is geometry-independent for locomotion too).

3. **Second interacting need — terrain-coupled thermosense.** The plateau is colder/hotter, so
   reaching the sealed food needs `climb_ability` AND `temperature_tolerance` jointly. Reuse the
   existing thermosense organ + the Gate-H cross-partial (2×2 corner grid). **FAIL** = Gate-H
   cross-partial ≤0 (antagonistic / no interaction) and Gate C still <7/8 ⇒ multi-need coupling does
   not create a joint gradient (mirrors the Exp 207 co-adaptation negative).

4. **Drifting terrain.** The plateau location drifts (mirroring the food-optimal drift), so the
   reachable frontier moves and creatures must RE-CROSS — a recurring, pivotal demand for climb rather
   than a one-time unlock. **FAIL** = drift just adds noise and `climb_ability` decays to founder
   (atrophy) ⇒ recurrence does not raise the per-step marginal benefit above upkeep cost.

5. **Genuine 3D / multi-layer movement (gated on a prior positive).** Only if Rungs 1–4 surface ANY
   positive local gradient: move from the 2.5D heightmap to a true layered/8-neighbour action space
   and/or a third need, to test whether richer movement strategy (climb + burrow/flight) co-evolves.
   Full 3D is costly on the current discrete 5-action grid, so this rung is NOT attempted unless a
   lower rung shows the wall is genuinely escapable. **FAIL** = added dimensionality multiplies the
   search space without raising any resident gradient ⇒ "better movement" does not evolve even with
   the physical room to express it.

**Discipline notes.** New substrate behind `enable_terrain` (default False), byte-identical OFF,
golden-hash guarded (mirror `tests/test_active_sensing.py` + the per-feature `test_ecology_*` pattern);
add a paired cost-divergence test proving the climb cost is causally live ON. The new trait is
declared via the documented add-a-trait recipe (Genotype field appended LAST with default 0.05; a
`mutate_locomotion` skip-guard so the Exp 194–213 mutation stream stays byte-identical) and registered
as a `LOCOMOTION_AXIS` in `ecology/evolvability/trait_axis.py` with EXHAUSTIVE `disconnect_overrides`
(every channel `climb_ability` feeds). The binding metric is always the Gate C LOCAL gradient, never a
gifted-N* demo. Conservative language only: functional, costed, heritable, local gradient, evolvable —
no claims about intelligence or agency beyond the measured behavior. Honest prior: the EXPECTED outcome
is FAIL (the wall extends to locomotion); that is a real, loggable result, and this is the
best-justified escape attempt yet because the sealed-reachability geometry is the one structural
condition (non-saturating, large-per-small-step benefit) the arc named as required to escape.

**Stop condition.** Exhausted when Rung 1 has a verdict (PASS = first escape, must survive Gate G +
both deflation controls + fresh-seed replication before any claim; FAIL = wall extends to locomotion),
and — if Rung 1 is NEGATIVE — Rungs 2–4 have each tested a named structural escape (staircase /
multi-need / drift) to a verdict, OR a PASS opens Rung 5. Then write in EXPERIMENTS.md either the
surviving locomotion-evolvability result with its control + replication record, or "the local-gradient
wall extends to locomotion across <named rungs>; the next step would require <named substrate>" — and
distil to a coalescence MechanismCard or BoundaryNote.

**STATUS.** state: active · latest: Exp 237 (MIXED — food-gradient PERCEPTION solves reach+survive (plateau intake 1%->62%); but climb_ability does NOT evolve: flat monomorphic benefit + invasion_from_rarity DOES_NOT_INVADE; the pairwise PASS is frequency-dependence, a caught false-positive; blind-verified) · depends-on: population-ecology (local-gradient wall), evolvability-geometry · reusable: enable_terrain/climb_ability/enable_navigation/enable_food_sense (byte-identical OFF), the L40/L41-clean Preflight · why: the local-gradient wall EXTENDS to locomotion via benefit saturation (now general across senses, memory, active-sensing, movement) · next-falsifiable: distil to coalescence cards, then CLOSE (gridworld answered) or — on a human word — a continuous-space mover.
