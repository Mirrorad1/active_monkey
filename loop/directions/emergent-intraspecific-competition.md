# direction: emergent-intraspecific-competition

**Question.** The predator-prey chapter (Exp 255–262) had competition only BETWEEN species and only
IMPLICIT within a species (carrying-capacity K silently punishes crowding). Can ACTIVE intraspecific
competition — and, up the ladder, cooperation, emergent predator/prey ROLES, and grouping — **EMERGE
under selection** purely from energy-preservation pressure, when we add a permissive, costed,
**unrewarded** affordance (the OPTION to take another creature's energy/resources) and let evolution
discover whether to use it? **R1 (the binding first test): when prey can OPTIONALLY contest each other
for the shared food (a heritable, costed, unrewarded `aggr` propensity), does an aggressive strategy
get genuinely SELECTED — invade from rarity AND show a positive LOCAL selection gradient — under food
SCARCITY but NOT under abundance, with distinct aggressive vs peaceful-forager LINEAGES observable?**

**Why it matters.** It separates two things this program must never conflate: *we provided an
affordance* vs *a behavior was selected*. The repo's hard wall (population-ecology closed-NEGATIVE; the
local-gradient wall; Exp-260) is that COSTED traits/behaviors often CANNOT invade even when a population
that has them would be fitter — the individual gradient at the resident is ≤0. Aggression is costed. So
"will they figure out to fight?" is genuinely open. If aggression emerges (scarcity-gated, selection not
drift), the substrate can host OPEN-ENDED social-strategy evolution and we proceed up the ladder
(cooperation, roles, grouping); if it hits the wall, costed social behavior does not self-organize here
— log it honestly and decide whether the richer chosen-action substrate is needed. Either way it builds
the **possibility-engine incrementally**: one permissive affordance per rung, each gated on
selection-vs-drift, with lineage tracking so divergence is a measured claim, not a story.

**Engine philosophy (the human's steer — binding framing).** Build the engine to ALLOW possibilities,
not to test one pre-chosen behavior. (1) PERMISSIVE: affordances are available, costed, and NEVER
rewarded — energy-preservation (short- AND long-term) is the ONLY driver; the evaluator only records,
summarizes, diagnoses, NEVER selects/ranks/protects/teleports (the standing anti-cheat law). (2)
DISCIPLINED MEASUREMENT (not a permissive substrate excuse): every "emerged" claim is checked for
SELECTION vs DRIFT (positive local gradient / invasion-from-rarity / persistence), with the
[[mean-of-opposites-guard]] applied per-seed and per-lineage — providing an affordance is NOT evidence a
behavior emerged. (3) LINEAGE TRACKING is first-class instrumentation: every creature carries a lineage
id (parent→child), and we record each lineage's strategy-trait distribution over time so divergent
strains (aggressive-contest vs peaceful-forager, later compete vs cooperate) are visible AND checkable —
it observes, it never steers. (4) GROW INCREMENTALLY: one new affordance per rung; do NOT build a giant
open-ended engine upfront (unfalsifiable, and would hit the wall everywhere at once).

**Not this (binding).** NOT hard-coded aggression — no reward/score for fighting; if `aggr` is not
SELECTED it is drift, report it as such. NOT pareidolia — a moving/­diverging lineage mean is not
selection; check the local gradient (trait-mean ≠ gradient; [[verify-verdict-critical-numbers-yourself]]).
NOT an external evaluator that selects. NOT the closed homogeneous-grid arena. Agents stay DISCRETE
individuals; contest/birth/death/migration are individual stochastic events; deterministic under fixed
seed. The new affordance is `enable_*`-gated, byte-identical OFF, golden-hash guarded (the
[[continuous-locomotion-rung2-suspect]] / patchmosaic discipline). Reuse `ecology/patchmosaic.py`,
`ecology/evolvability/metrics.py`, and the existing co-evolution machinery; one mechanism per iteration.

**Experiment ladder.** (each a PROTOCOL iteration; each names its FAILURE; each adds ONE permissive
affordance + lineage tracking + a selection-vs-drift gate)

1. **R1 — does costed intraspecific contest EMERGE? (the binding beachhead).** Patch-mosaic, ONE focal
   species (prey-vs-prey for the shared food; predator present as ecological context, its trait frozen).
   Add a heritable `aggr` ∈ [trait_min, trait_max] propensity + a **contest affordance**: within a patch,
   a creature with `aggr>0` may pay an energy/fecundity cost `~ c·aggr` to attempt to SEIZE a share of
   another co-occurring creature's energy/food intake; win-prob rises in own `aggr` relative to the
   target's; on a win the aggressor gains roughly what the target loses minus dissipation (zero-sum +
   loss). Gated `enable_contest` (default OFF, byte-identical, golden-guarded), no reward for contesting.
   Add lineage tracking. Sweep FOOD SCARCITY (rich → scarce). **VERDICT = AGGRESSION_EMERGES** iff `aggr`
   (a) invades from rarity AND (b) has a positive LOCAL selection gradient at the resident under SCARCITY
   (7/8-strict, breed-true common garden, reusing the Exp-260 instrument) AND (c) does NOT under
   abundance (scarcity-gated, not an artifact) AND (d) the drift-null (contest made causally inert) is
   neutral. **FAIL = WALL**: no positive local gradient under scarcity ⇒ costed intraspecific aggression
   does not self-organize on this substrate (the local-gradient wall holds again) — log it. **NO_VERDICT**:
   drift-null fires, or populations collapse, or per-seed bimodality un-gated.
2. **R2 — scarcity-dependence map + integrate into the full predator-prey system.** Map P(aggressive
   lineage dominant) vs food scarcity (basin-style, à la Exp 262); then turn the predator back ON
   (predator trait co-evolving) and ask whether prey-prey contest changes the predator-prey outcome
   (does intraspecific competition buffer the Exp-259 prey collapse?). **FAIL** = aggression that emerged
   in isolation is washed out / inert once the predator co-evolves.
3. **R3 — emergent ROLES (what is a predator/prey).** Add the cannibalism / intraguild-predation bridge:
   a forager that evolves to kill+consume conspecifics BECOMES a predator. Single starting type; test
   whether distinct predator-like vs prey-like lineages DIVERGE and persist by selection. **FAIL** = roles
   do not differentiate (one strategy fixes, or it is drift).
4. **R4 — emergent GROUPING / cooperation.** Add a cooperate/share affordance (or a coordinated-contest
   bonus) and test whether grouping (packs that hunt together / herds) emerges and whether coordinated
   beats solo. NOTE: genuine grouping may need finer spatial structure than patches — flag a substrate
   extension here. **FAIL** = no cooperative lineage is selected (defection dominates; the cooperation
   wall). Connects to [[social-emergence]].

**Discipline notes.** Reuse `ecology/patchmosaic.py` (gated mechanisms, byte-identical OFF + golden
T-tests), `ecology/evolvability/metrics.py` (7/8-strict local-gradient gate), and the session's guards:
[[mean-of-opposites-guard]] (per-seed/per-lineage dispersion, never a pooled mean), byte-identical-OFF
events_hash goldens, controller re-runs the decisive measurement itself
([[verify-verdict-critical-numbers-yourself]]). Lineage tracking is observation-only. One affordance per
rung. Conservative language; negatives (the wall) are results.

**Stop condition.** Exhausted (for now) when R1 returns a clear verdict: AGGRESSION_EMERGES
(scarcity-gated, selection not drift, drift-null neutral) ⇒ proceed up the ladder; or WALL (no local
gradient under scarcity) ⇒ costed intraspecific aggression does not self-organize on the patch substrate
— log a BoundaryNote, and decide on a human word whether to try the richer chosen-action substrate
(engine.py) where roles/tools/senses could co-emerge (higher risk — that is where the sense-evolution
arc hit the wall). Either way write the verdict + the scarcity-arm + drift-null evidence to EXPERIMENTS.md.

**STATUS.** state: active · R1 DONE (Exp 263, POSITIVE-SINGLE): costed prey-vs-prey contest aggression EMERGES under reproduction scarcity — clears the local-gradient WALL that closed population-ecology (FIRST costed-behavior emergence; wall is trait-specific). Selected (invades 9/10 scarce vs 0/10 abundant; gradient 8/8 scarce, NEG abundant at resident aggr=0.3), not hard-coded (no reward, real cost, drift-null neutral 4/8). substrate: gated enable_contest + heritable aggr + lineage tracking in ecology/patchmosaic.py (byte-identical OFF, 27 tests) · next: R2 scarcity-map at realistic resident aggr + integrate contest into the co-evolving predator-prey system (buffer Exp-259 prey collapse?); then R3 cannibalism/roles, R4 grouping/cooperation
