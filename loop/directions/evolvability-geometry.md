# direction: evolvability-geometry

**Opened 2026-06-15** on the human's explicit word, after the active-sensing / hidden-state-memory
line closed-negative (Exp 199–211). This direction opens only because that line is closed: it does
NOT reopen it.

**Question.** When a costed capability is useful when imposed but fails the local-gradient test, is
the blocker the capability itself, or the evolutionary search geometry around it?

**Programmer translation.** The feature works in an integration test, but small production rollouts
never beat the incumbent. This direction asks whether the deployment path is broken: mutation step
size, standing variation, correlated trait bundles, or valley-crossing mechanics.

**Why it matters / core finding that motivates this direction.** The local-gradient wall now spans
scalar senses / organs (199–207), passive memory / belief-like state (208–209), fixed-rate active
sensing (210), and uncertainty-gated active sensing (211). The important pattern is not merely
"sensing failed." The stronger pattern is:

> A capability can be real, useful when gifted, and still not locally selectable from the resident.

This direction treats that pattern as the object of study — turning the wall into a question about
PREMISE.md's "lived, registered experience": is a useful capability unreachable because it is
useless, or because the evolutionary operators only test small local steps that do not pay?

---

## Binding closures

Do not reopen the closed-negative sense / memory / active-sensing line by adding another sensory
knob, another memory knob, or another probe policy unless the experiment is explicitly testing
evolvability geometry.

Closed-negative results stand:
- costed scalar sensing did not become a functional organ under the tested ecology;
- passive hidden-state memory / belief persistence failed local gradient;
- fixed-rate active sensing failed local gradient;
- uncertainty-gated active sensing failed local gradient.

A new experiment in this direction must change the **path through trait space**, not merely add
another capability variant.

---

## Definitions

**Local selection gradient.** The fitness slope near the resident for a small trait perturbation.
Programmer translation: a tiny config change or feature rollout should show directional improvement
before we trust a big migration.

**Fitness valley.** A region where intermediate versions are worse even though a larger, completed
version may be better. Programmer translation: partial rollout makes the system worse, but the fully
migrated design might be better.

**Mutation geometry.** The distribution and correlation structure of heritable changes. Programmer
translation: what kinds of diffs the evolutionary system is allowed to generate — tiny scalar
tweaks, rare large jumps, bundled changes, or coordinated multi-file changes.

**Standing variation.** Pre-existing diversity in the population before selection acts. Programmer
translation: instead of waiting for a production deploy to discover the feature from zero, several
variants are already live at low frequency.

**Adaptive valley crossing.** A population reaches a higher-fitness configuration even though the
first local step is not beneficial. Programmer translation: the system crosses a bad intermediate
migration state without an external evaluator hand-picking the winner.

---

## Central hypothesis

The wall may not be "information is useless." The wall may be:

> The useful version of the capability lives across a valley, and the current evolutionary operators
> only test small local steps that do not pay.

This direction asks whether any honest, non-evaluator evolutionary mechanism can cross that valley.

---

## Hard constraints

Selection must still come from the ecology, not an external ranking.

Allowed:
- larger or heavy-tailed mutations, if disclosed;
- standing variation seeded at known frequencies, if reported as an initial-condition intervention;
- correlated trait bundles, if they are heritable and not selected by evaluator ranking;
- recombination or lineage mixing, if mechanically implemented and audited;
- monomorphic fitness-surface probes used only as diagnostics;
- local-gradient preflights before full evolution.

Forbidden:
- external top-K selection by evaluator score;
- deleting negative results;
- changing cost after seeing results to force a sign;
- giving the mutant direct reward;
- making the environment read the trait and help it;
- calling seeded functional variants "de novo emergence";
- claiming full active inference or open-ended evolution from a valley-crossing scaffold.

---

## Required methodology guards

Use the existing loop discipline: predeclare falsifiers; run fresh seeds; blind-verify the verdict;
preserve raw outputs; log negative results as negative; keep the provided-vs-self-formed distinction
explicit. Specific guards from prior lessons (loop/LESSONS.md):
- **L22:** a forced/gifted benefit does not imply evolvability.
- **L28:** preflight the scientific premise before a full batch.
- **L29:** drift is handled by population size and the drift-robust selection slope, not by raising cost.
- **L30:** calibrate any costed action/capability against its empirical benefit ceiling.
- **L32:** if a targeting/gating policy works when imposed, still measure its benefit ceiling against
  the indiscriminate baseline before reading an evolvability verdict.

---

## Experiment ladder

### Rung 1 — landscape assay
**Question.** Is there actually a higher-fitness region beyond the local wall? Run a frozen /
monomorphic / pinned-trait landscape assay over a representative closed-negative trait (candidate
axes: thermosense intensity, memory_horizon / belief_persistence, information_sampling_rate, the
uncertainty-gated probe threshold, or a small 2D combination if prior data says the interaction is
plausible). The goal is not evolution; the goal is to CLASSIFY the landscape.
Possible outcomes: (1) **No higher region** ⇒ the capability is not useful enough even completed —
stop this trait. (2) **Higher region exists, no local path** ⇒ true fitness valley — continue to
Rung 2. (3) **Positive local slope exists but evolution failed** ⇒ the prior experiment had a
measurement / drift / implementation problem — audit before continuing. (4) **Higher region only
under evaluator/imposed conditions** ⇒ not an ecology result — stop or redesign the substrate honestly.
FAILURE / STOP: if there is no bulk-fitter region, do NOT test mutation geometry — there is no valley.

### Rung 2 — heavy-tailed mutation preflight
**Question.** Does allowing rare larger mutations reveal a path that small local mutations cannot?
Compare: resident local-step kernel, heavy-tailed kernel, matched-neutral heavy-tailed control,
useless-trait heavy-tailed control. (Heavy-tailed = mostly small mutations, rare large ones.)
ACCEPTANCE: heavy-tailed mutations produce more visits to the bulk-fitter region; the visits persist
or reproduce more often than matched neutral controls; the effect is not just drift; the useful trait
improves more than a useless trait with the same mutation geometry. FAILURE: rare large mutants arise
but are culled; controls behave the same; the result only appears at collapsed population sizes.

### Rung 3 — standing variation assay
**Question.** If the useful capability is already present at low frequency, can ecology amplify it?
Seed a small known fraction of functional / near-functional variants (NOT de novo emergence — a test
of maintenance/amplification once present). Sweep initial frequencies {0.1%, 1%, 5%, 10%}. Controls:
neutral seeded trait at same frequencies; useless capability at same frequencies; matched-cost variant
without information/functional benefit. ACCEPTANCE: a reproducible critical frequency above which the
functional variant grows; controls do not show the same amplification; survives drift-suppressed cap +
fresh seeds. FAILURE: functional variants decay at every frequency; amplification also occurs for
useless/neutral controls; the required frequency is so high the result is only "bulk replacement,"
not evolvability.

### Rung 4 — correlated trait bundle preflight
**Question.** Does the valley exist because the useful capability requires two or more traits to move
together? Test only if the landscape assay suggests a real interaction. Candidate bundles: sensor +
controller; sensor + lower upkeep/efficiency; memory + policy that can use memory; probe +
hidden-state coupling; sense + reproductive-timing/movement policy. Use a corner-grid before any full
batch: resident/resident, A only, B only, A+B. ACCEPTANCE: A alone does not pay; B alone does not pay;
A+B pays; discrete cross-partial positive; bundled heritable mutation beats unbundled controls.
FAILURE: one trait pays alone; A+B does not beat the best single-trait arm; cross-partial near zero;
the bundle only wins because cost accounting is wrong. (Reuse `run_controller_cross_partial` /
corner-grid tooling from the Evolvability Preflight; cf. Exp 207.)

### Rung 5 — full evolutionary batch only after a positive preflight
Full evolution is allowed ONLY after one of the above rungs shows a real path (heavy-tailed mutation
reaches and maintains the fitter region; OR standing variation is amplified above controls; OR a
correlated bundle has a positive cross-partial and passes local preflight). If no preflight passes,
do NOT run full evolution.

---

## Verdict labels (use exactly)

- **NO_HIGHER_REGION** — no bulk-fitter functional configuration exists. Stop the trait.
- **FITNESS_VALLEY_CONFIRMED** — a fitter region exists, but the local gradient is non-positive.
- **VALLEY_CROSSING_SUPPORTED** — a non-evaluator mechanism reaches/amplifies the fitter region and
  beats drift/neutral controls.
- **VALLEY_CROSSING_FAILED** — a fitter region exists but heavy-tail / standing-variation / bundle
  mechanisms fail.
- **ARTIFACT_OR_NO_VERDICT** — controls fail, drift dominates, populations collapse, byte-identity
  guards fail, or the evaluator accidentally selects.

---

## Stop condition

Stop this direction when: (1) at least one representative wall trait has a completed landscape assay;
(2) heavy-tailed mutation has a verdict; (3) standing variation has a verdict; (4) correlated bundle
has a verdict iff the corner-grid supports testing it; (5) no honest valley-crossing path remains.
Then write the synthesis: *"The ecology contains useful imposed capabilities, but the evolutionary
operators tested here do (or do not) provide a non-evaluator path across the local-gradient wall."*

---

## Next falsifiable step

**Exp 212 should be a LANDSCAPE ASSAY, not another evolution run.** Pick one representative
closed-negative capability and map whether a higher-fitness region actually exists beyond the
resident. Preferred candidate: the best-instrumented active-sensing / hidden-mode substrate from
Exp 210–211 (it already has active-sensing OFF byte-identity guards, the probe-policy abstraction,
hidden-state controls, cost-calibration discipline, the drift-suppressed cap-250 regime, fresh-seed
common-garden tooling, and benefit-ceiling instrumentation). Exp 212 question: *does any larger
information-sampling configuration produce a bulk-fitter region, or is the active-sensing landscape
flat/negative everywhere once cost and controls are honest?* Only if a higher region exists should
Exp 213 test heavy-tailed mutation or standing variation.

**STATUS.** state: closed-negative (2026-06-15, human's word "b") — a benefit-MAGNITUDE wall, not search geometry or payoff shape · latest: Exp 213 GEOMETRY_INDEPENDENT_WALL + Exp 212 NO_HIGHER_REGION (both blind-verified): landscape assay found no valley; affordance audit found smooth and discrete-high-stakes payoffs equally non-evolvable ⇒ Rung 2/3 not run (nothing to cross to) · depends-on: Exp 204/210–211 substrates + Evolvability Preflight · reusable: yes · why: landscape-assay + affordance-audit instruments classify whether a trait/geometry has a reachable fitter region · next-falsifiable: re-opens only on a human word with a benefit-MAGNITUDE lever (bigger per-step payoff / higher pivotal density / richer world) · synthesis: docs/research/local-gradient-wall.md §3.
