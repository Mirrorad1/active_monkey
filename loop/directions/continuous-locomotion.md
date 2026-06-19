# direction: continuous-locomotion

**Question.** The discrete gridworld CLOSED the locomotion-evolvability question (Exp 235–237,
`environmental-complexity`): the local-gradient wall extends to locomotion — once perception solves
reach+survive, a costed climbing trait's benefit SATURATES (a crude mover eventually crosses), it does
NOT invade from rarity, and the only positive signal is frequency-dependent priority. Every wall result
(senses 199–201, memory 208–209, active sensing 210–213, locomotion 235–237) lives on a DISCRETE grid
where movement is a one-cell step, crossing is a near-binary gate, and benefits saturate over a lifetime.
**Does a CONTINUOUS-SPACE substrate — continuous position, momentum/inertia, a graded terrain-traversal
cost field, real Euclidean distance — make movement skill NON-saturating and the PRIMARY survival
challenge, so a costed heritable locomotion trait becomes locally evolvable (invades from rarity); or
does the local-gradient wall hold even there, making it substrate-general (discrete AND continuous)?**

**Why it matters.** The wall arc named exactly one structural escape condition: a benefit that is
LARGE-per-small-step and NON-saturating — where each increment of the trait keeps paying. On a discrete
grid that never materialised (crude reflexes grab the easy part). Continuous movement physics is the
natural place for it to exist: speed/efficiency/path-quality can yield a graded, individually-accruing
advantage that does not wash out over a lifetime (a faster mover covers more ground every step, not just
"eventually crosses"). A POSITIVE (a movement trait that invades from rarity with a non-flat monomorphic
benefit) would be the FIRST escape of the local-gradient wall in the whole project. A NEGATIVE would make
the wall substrate-general — a much stronger statement than "discrete grids are too legible."

**The provided-vs-earned line (binding, the crux).** The movement PHYSICS (continuous kinematics, the
traversal-cost field) and any PERCEPTION (a sensed resource gradient, reusing the Exp 237 food-sense
pattern) are PROVIDED priors, declared like taught labels. The heritable trait under selection is a
single costed LOCOMOTION parameter (e.g. locomotor efficiency, max speed, or stride/step-size) — costed
monotonically so a bigger trait is more expensive to build/carry. The binding metric is **invasion-from-
rarity** (L41 — NOT a 50/50 pairwise contest, which can be frequency-dependent priority) on a **non-flat
monomorphic benefit curve** (L39 — verify the trait is behaviorally EXPRESSED and the per-capita benefit
is graded BEFORE claiming evolvability). Anti-gaming (L40): re-run with neutral conventions / a flat-field
null to prove any effect is physics-driven, not an artifact. Any shared engine path stays behind an
`enable_*` flag, byte-identical OFF, golden-hash guarded.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE)

1. **Build the continuous-space substrate + ONE heritable movement trait (plumbing + expressibility).**
   Continuous position/velocity, a graded traversal-cost field, real-distance foraging; a heritable
   costed locomotion trait. Verify (gen-0, no evolution) the trait is EXPRESSED — the monomorphic
   per-capita intake/coverage curve is MONOTONE and NON-FLAT in the trait (unlike the saturated gridworld
   curve), and a flat-field null shows the effect is physics-driven. **FAIL** = the monomorphic benefit
   curve is flat/saturating (the wall's saturation reappears in continuous space ⇒ no escape, log it) or
   the trait is inert. No evolvability claim yet — this is the L39 expressibility gate.
2. **Local evolvability via invasion-from-rarity (the binding test).** A rare single-step mutant of the
   locomotion trait must INCREASE from ~5% in a resident background (Evolvability Preflight, the
   FREQUENCY_DEPENDENT-aware aggregate), with the monomorphic benefit non-flat. **FAIL** =
   DOES_NOT_INVADE, or pairwise-PASS-but-no-rarity-invasion (frequency-dependence, not evolvability) ⇒
   the wall holds in continuous space too — a substrate-general negative, distilled to a BoundaryNote.
3. **Graded terrain + the traversal-cost field as the selective surface.** Make terrain traversal cost
   continuous (slopes cost more, proportional to the movement trait), so path-quality/efficiency pays
   continuously. **FAIL** = still no invasion-from-rarity ⇒ even a continuous traversal-cost gradient
   does not make locomotion evolvable.
4. **Multi-need / morphology (only if 1–3 surface a positive).** Add a second need or a second movement
   trait and test whether a movement STRATEGY (not just a scalar) co-evolves. **FAIL** = added
   dimensionality multiplies the search space without raising any resident gradient.

**Discipline notes.** Reuse where honest: the continuous-substrate spine (nira / `creature/` continuous
state), `ecology/runtime` snapshot/fork/replay, the Evolvability Preflight (now FREQUENCY_DEPENDENT-aware,
L41), the Exp 237 food-sense perception pattern, and the byte-identity/golden-hash gating discipline.
One new mechanism per iteration. Binding metric is ALWAYS invasion-from-rarity on a non-flat monomorphic
curve, never a pairwise win. Conservative language only: functional, costed, heritable, local gradient,
evolvable — no claims of intelligence or agency beyond measured behavior.

**Stop condition.** Exhausted when Rung 1 establishes expressibility (or finds saturation again) and
Rung 2 has an invasion-from-rarity verdict — PASS (the first escape; must survive Gate-G guards + a
fresh-seed replication + an anti-gaming null before any claim) or FAIL (the wall is substrate-general).
If NEGATIVE, Rungs 3–4 test the named continuous escapes to a verdict. Then write in EXPERIMENTS.md
either the surviving evolvable-locomotion result with its replication record, or "the local-gradient wall
is substrate-general (discrete AND continuous); locomotion does not evolve at this scale because <named
mechanism>" — and distil to a coalescence MechanismCard or extend the wall BoundaryNote to continuous space.

**STATUS.** state: closed-negative (WEAKLY-POSABLE; bump-geometry lever NEGATIVE; refill-rate-vs-revisit-rate mechanism robust across 4 levers) · closed: Exp 243 (confirmed-with-mechanism), Exp 244 (correction), Exp 245 (geometry lever NEGATIVE) — the toy continuous substrate CANNOT host a non-degenerate locomotion-evolvability equilibrium: runaway when intake ignores depletion (Exp 238-241, foundational intake bug), oscillatory when it does not (Exp 242); Exp 243 added crowding mortality (Mechanism A) that damps oscillations — originally logged as NO-GO because the whole-field availability gate (availability 0.98-0.99 >> 0.85 ceiling) flagged it degenerate. Exp 244 CORRECTS the Exp-243 gate: the whole-field availability mean is the WRONG KIND of non-degeneracy arbiter (measures spatial dilution, not population competition); the correct arbiter is density-dependence of per-capita intake. Measured directly: 4/4 runs show corr(N, per-capita intake) NEGATIVE (-0.395 / -0.326 / -0.483 / -0.550) — genuine resource competition EXISTS. REVISED VERDICT: WEAKLY-POSABLE. BUT competition is weak (~7-9% per-capita intake drop across observed N range); an invasion-from-rarity test would likely show directional/runaway invasion (the ~+43% speed benefit unbounded by weak competition) rather than a clean ESS. Exp 245: sharpened bump geometry (sigma 1.5->0.5, total food constant, 5 sigmas x 2 seeds) did NOT strengthen competition — %intake-drop stays 3-9% at every sigma, whole-field availability ~0.978-0.982 unchanged. MECHANISM (the deeper finding): with floored regen (rate 0.5, capacity 2.0) a grazed cell refills in ~4 steps; the population (~50) is sparse over 576 sub-cells so a creature grazes a cell, moves on, and the cell REFILLS BEFORE another creature revisits it — standing-crop depletion never accumulates. The viability<->depletion tension is really a REFILL-RATE-vs-REVISIT-RATE tension: strong competition needs slow refill, slow refill starves the population. Weak competition is STRUCTURAL, robust across FOUR levers: founder calibration, A-strength (hmax), regen rate (Exp-243), bump geometry (Exp-245). Discrete local-gradient wall STILL STANDS (Exp 199-237). Instrument lesson: whole-field availability gate in cert_run DEPRECATED as too crude; density-dependence of per-capita intake is the metric going forward. Last structurally-distinct untried lever: MOVING/dynamic depletable patch (forced crowding) — being pursued as Exp 246 (caveats: overlaps discrete Exp-200 drifting-band NEGATIVE; a patch small enough to force crowding risks trivializing within-patch movement).
