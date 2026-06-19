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

**STATUS.** state: closed-negative · closed: Exp 242 — the toy continuous substrate CANNOT host a clean locomotion-evolvability verdict: runaway when intake ignores depletion (Exp 238-241, foundational intake bug), oscillatory when it doesn't (Exp 242); 0 stable fixed points in a 45-cell sweep; candidate escape RETIRED; discrete local-gradient wall STANDS (Exp 199-237). Reopens only on a human word AND a fundamentally better continuous substrate with stable equilibria across a range of speeds.
