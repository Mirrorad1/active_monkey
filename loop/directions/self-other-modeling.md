# direction: self-other-modeling

**Question.** Does a clade-mate that maintains a belief over the OTHER's *latent goal*
(inferred from the other's behavior, never provided) predict and coordinate measurably better
than one that only tracks the other's *position* — i.e., is there a functional theory-of-mind
benefit at this toy scale, or does goal-agnostic tracking suffice?

**Why it matters.** Every result so far treats the other as an environmental feature or a
scalar cue. The closed `social-emergence` ladder (Exp 63–74) proved this exhaustively: the
"other-here" modality is `pA2 = P(other-here | my_cell)` (a statistic of MY cell, not a model
of THEIR mind); the cue channel passes `argmax(emitter.value_counts)` as a label with no model
of *why*; coordination deflated to stigmergic lock-in *because* no creature represents the
other as an agent with hidden states. So the substrate's own closure named the gap: creatures
are **solipsistic**. This direction asks whether adding the smallest honest *model of the
other's latent state* — a posterior over the other's GOAL, the analogue of the creature's own
`value_counts` but for the other — produces behavior (prediction, coordination) that
goal-agnostic position-tracking cannot. It is the rung-1 precursor to the N-order ladder's N7
(collective/multi-agent integration, `docs/specs/n-order-self-modeling.md`), kept far below the
N7 grammar/ontology ceiling: the claim is *latent-goal inference*, never shared language.

**The provided-vs-earned line (binding, the crux).** Modeling the other is trivially provided
if you hand the creature the other's goal. The honest test requires: the other is **goal-
directed** (navigates toward its own favorite color/source via the Exp 68–72 comfort-gated
policy), the modeler observes only the other's **position** (a declared provided modality) plus
the **shared innate B** movement model, and the other's **goal is INFERRED** from its observed
trajectory — never read off. The load-bearing control is a **position-tracking baseline**
(a belief over the other's *location* with no goal latent) so that any benefit is attributable
specifically to inferring the latent goal, not to mere object permanence.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE)

1. **Goal-directed-other substrate + position-tracking baseline (consolidation, plumbing).**
   Reuse the Exp 64 shared-world harness + the Exp 70 comfort-gated BFS policy so the OTHER is
   a goal-seeker (heads to its favorite-color source). Wire a provided `obs_other_pos` modality
   and a running `q_other` position belief updated by the shared innate B (a goal-AGNOSTIC
   tracker). Verify: both spines load, trunk byte-identical, the tracker's one-step position
   prediction beats a static occupancy marginal. **FAIL** = the tracker is not better than the
   marginal (substrate mis-specified) or promoting the harness mutates a committed spine. No
   ToM claim — this builds the baseline the ToM rung must beat.

2. **Latent-goal inference (the ToM test).** The modeler maintains a posterior `q_other_goal`
   over which color/source the other seeks, updated from the other's observed trajectory via a
   provided goal-directed movement model `P(other_move | other_pos, goal)`. Predeclared:
   goal-inferring prediction of the other's next-K positions beats BOTH (a) the rung-1
   position-tracker (no goal latent) AND (b) a goal-agnostic occupancy marginal, by a declared
   margin on held-out steps, ≥8 seeds. **FAIL** = goal-inference does not beat position-tracking
   (then inferring the latent goal adds nothing — the other is predictable without a goal model;
   a real NEGATIVE about ToM at this scale, logged as such). **TRIVIAL-PROVISION GUARD:** if the
   effect only appears when the goal is *provided* rather than inferred, that is NEGATIVE, not
   positive.

3. **Does ToM pay behaviorally?** On the shared-resource task (Exp 70 ecology), does a modeler
   that ACTS on `q_other_goal` (e.g., yields/avoids a source it infers the other is heading to)
   achieve better joint comfort / cleaner avoidance than the position-tracking baseline and than
   a solipsist? Predeclared coordination metric + threshold; any deviation enters the
   `functional-emergence.md` rung-5 cascade (≥3 fork reproductions + deflationary sweep) before
   the word "coordination" is used. **FAIL** = behavior statistically indistinguishable from the
   baseline (goal-inference is real but behaviorally inert at this scale).

4. **Mutual modeling (N7-adjacent — declare the ceiling).** Both clade-mates model each other's
   goal. Does mutual ToM produce coordination neither one-sided model achieves, or does it
   destabilize (each chasing a model of the other's model)? Predeclared stability/coordination
   metric. **CEILING (binding):** this is N7 territory — no claim of shared/negotiated ontology
   or emergent grammar (`open_problem.html`); the honest claim is mutual goal-prediction gain
   vs. its absence, the N7 invalidation frame ("communication carries no mutual-predictability
   gain; collapses to independent agents").

**Discipline notes.** Fork-only on the clade (mirro/vela load as the two creatures; never mutate
a committed spine — `tests/test_creature_continuity.py`). Every modality and movement model for
the other is a PROVIDED prior, declared in the entry like a taught label; one new mechanism per
iteration or attribution is lost. Functional language only ("functionally infers the other's
goal"); the inner-experience layer stays unverified both ways (VALIDATION.md). Higher N is NOT
"more conscious" — functional competencies only.

**Stop condition.** Exhausted when: the position-tracking baseline exists and is validated
(rung 1), the latent-goal-inference rung has a verdict either way (rung 2), the behavioral-payoff
rung has a verdict with every deviation cascaded (rung 3), and the mutual-modeling rung reaches
its verdict or hits the declared N7 ceiling (rung 4). Then write in EXPERIMENTS.md either the
surviving ToM novelty-candidate(s) with their cascade record, or "no disciplined self-other
modeling benefit at this substrate; the next step would require <named substrate>" — and open
that substrate as a new direction or stop.

**STATUS.** state: closed-positive (Exp 228–234) · latest: Exp 234 · depends-on: social-emergence (closed), persistent-creature · reusable: q_other goal beliefs, the goal-directed-other harness, the learned-transition baseline control, the L38 manipulation-check · why: FIRST functional self-other modeling — solipsism gap crossed (229), but LIGHT and localised to transitions, growing with non-stationarity (229–231); difficulty = separability not size (232); simple baselines near-optimal here (legible BFS 233; stigmergy coordinates 234) · distilled to functional-goal-inference-v0 + self-other-substrate-legibility-wall-v0 · next-falsifiable: re-opens only where the SIMPLE baselines FAIL (less-legible/partial-obs other, or coordination stigmergy cannot solve).
