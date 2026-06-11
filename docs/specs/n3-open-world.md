# Design: N3 bounded-map / open-world repair controller

**Date:** 2026-06-11
**Status:** Design (direction-level; not yet run). Companion card:
`loop/directions/n3-bounded-map-open-world.md`.
**Repo:** extends the persistent-creature substrate; reuses `active_loop/worlds.py`,
`active_loop/growth.py`, `active_loop/verdict.py`, and the `fork()` control discipline.

---

## 0. How this slots into what already exists (read first)

This is **not a new, competing definition of N3.** The repo already has:

- the **N-order ladder** (`docs/specs/n-order-self-modeling.md`): N0 reflex → N1 world
  model (green) → **N2 metacognition** (in progress: noise-vs-structural-vs-volatility
  classification + precision + expansion trigger) → **N3 meta-calibration** (card live:
  *agency over metacognition* — owns and can override N2's revision policy);
- the **growth machinery** (`active_loop/growth.py`): per-color/global surprise-ceiling
  detector (`check_ceiling`, `check_color_alarm`), `LiveProbation` (snapshot → provisional
  install → keep/revert on a live ≥0.1-nat drop), round-robin alarm scheduling, batch-jump
  EM with penalized K-selection;
- the **procedural world family** (`active_loop/worlds.py`): `learnable`, `noisy(p_true)`,
  `aliased(...)`, `nonstationary(base, remap_at_step, ...)`;
- the **Exp 154 result**: under the *normalized-density* predictive convention, on-demand
  growth works on fresh seeds (surprise drops 0.58–1.18 nats, zero spurious detector events,
  alarms quiet) — i.e. the creature can already *grow a component when one color is aliased*.

What this document adds is the **open-world benchmark instantiation** of the N2→N3 spine:

| Ladder term (existing)                       | This document's name        | Mode        |
| -------------------------------------------- | --------------------------- | ----------- |
| N2 failure-mode classifier (extended)        | **diagnostic layer**        | shadow      |
| N3 control surface over N2's repair policy    | **repair controller**       | control     |

The user's "N3 as a bounded diagnostic meta-model that chooses repair actions" decomposes
cleanly: **diagnosis is N2-order work; choosing/authoring the repair is the N3 control
surface.** Shadow mode tests the diagnosis (a meta-d′-style sensitivity claim over N2's own
state); control mode is exactly the existing card's load-bearing test — *does acting on the
diagnosis beat diagnosis-only, and is it irreducible to an offline retune?*

The open-world piece (horizon, entropic spreading) is genuinely **new relative to the existing
card**, which only contemplates static-richness "deceptive world / expansion-trap" regimes. The
new content is: (a) a procedural world *that can grow past the model over a long life*, and
(b) two repair actions the current machinery lacks — **forget/decay** and **quarantine** — plus
a **frontier** belief the current map has no representation for.

**Binding honesty (inherits `loop/VALIDATION.md` + the ladder's anti-regress law):**

> A layer Nₖ is real iff a *constructible perturbation* degrades a well-tuned Nₖ₋₁ agent, and
> Nₖ detects+corrects it via a control surface Nₖ₋₁ lacks. No discriminating test ⇒ notation.

Concretely for this document: **the repair controller earns its existence only if choosing
among {continue, explore, grow, forget, quarantine} beats the best single fixed repair policy
AND is not reproducible by offline hyperparameter retuning of the existing growth thresholds.**
If "always run the Exp-154 growth machinery" wins everywhere, there is no N3 here — that is a
real negative and the headline finding, logged as such.

No cosmology. "Expanding universe" is an *intuition pump only* — a bounded observer inside an
open-ended environment. No consciousness/AGI/recursive-self-improvement claims. Functional
language only.

---

## 1. The core idea in plain English

The creature has a **bounded map** (a fixed-capacity continuous mixture model over place →
color, the M3 substrate). The **world can become larger or more complex than the map**: new
regions open, old regions drift, signals smear, or structure appears beyond where the creature
can currently reach.

A dumb agent has one hammer: when surprise stays high, grow the map. That fails in most of the
ways a world can surprise you. The research bet is that a bounded agent should instead **read
its own learning state and pick the matching repair**:

- **Stable & known** → do nothing; don't grow, don't forget.
- **Ordinary ignorance** → keep learning with the current model (more data fixes it).
- **Exploration deficit** → go visit states it has not sampled.
- **Structural inadequacy** (aliasing) → grow / split a component (the Exp-154 move).
- **Nonstationarity / stale evidence** → forget/decay old evidence; do *not* grow.
- **Irreducible noise** → quarantine the region; do *not* keep accepting growth there.
- **Entropic spreading** → compress/accept uncertainty rather than chase diffusing structure.
- **Unreachable frontier** → mark "unknown but not actionable"; hold uncertainty, don't grow.
- **Evaluation artifact** → distrust the metric; halt/consult rather than act.

The single research question:

> Can a bounded active-inference agent distinguish *why* persistent surprise is high — ordinary
> ignorance vs. new structure vs. irreducible noise vs. stale evidence vs. aliasing vs.
> nonstationarity vs. unreachable frontier — and choose the correct repair?

The trap to avoid (named so we can fail honestly): an impressive demo that only works because
each world was hand-tuned so the "right" answer was obvious. Every claim below is gated on
**fresh-seed, multi-layout** generalization and on an **offline-retune control** that would
embarrass the layer if it won.

---

## 2. The N3 layer, precisely

Two sub-layers, tested in sequence (shadow before control).

### 2A. Diagnostic layer (N2-order; runs in **shadow mode** first)

**Observations** (all already computable from the M3 loop; no new creature internals required):

| signal                  | source                                                            |
| ----------------------- | ----------------------------------------------------------------- |
| global live surprise    | rolling mean of per-step −log p(obs) (`SURPRISE_WINDOW=200`)      |
| surprise slope          | `np.polyfit` slope over the window (already in `check_ceiling`)    |
| ceiling-event rate      | fraction of steps `check_ceiling` is True                          |
| per-color residual mean | per-color deque mean (`COLOR_SURPRISE_WINDOW=50`)                  |
| localization uncertainty| trace/det of place posterior `Sigma_p_diag`                       |
| evidence mass           | per-color `counts` / replay-buffer fill                           |
| component count         | `len(components[k])` per color                                     |
| growth attempts/outcome | `LiveProbation` kept/reverted ledger                              |
| exploration coverage    | fraction of reachable cells visited in the last window           |
| novelty rate            | rate of first-visits / out-of-support observations               |
| frontier-hit rate       | rate of "boundary / no-transition-available" encounters (new)    |
| post-forget recovery    | surprise-recovery time after a decay op (new)                    |
| old-region stability    | surprise variance in regions not visited recently                 |

**Latent categories** (the diagnosis; a 10-way classification per region/time-window):
`stable_known`, `ordinary_ignorance`, `exploration_deficit`, `structural_inadequacy`,
`aliased_context`, `nonstationarity`, `stale_evidence`, `irreducible_noise`,
`entropic_spreading`, `unreachable_frontier`, plus a reserved `evaluation_artifact`.

**Output in shadow mode:** for each decision window, the predicted category and the predicted
*correct repair action*. It does **not** touch the creature. Logged against ground truth (the
world generator knows the true regime).

**Success (shadow):** failure-mode classification macro-F1 and correct-repair top-1 accuracy,
each **predeclared** with a threshold and reported per world type, across **≥3 fresh seeds and
≥2 layouts**. Suggested gate: macro-F1 ≥ 0.70 and *no single confusion pair > 0.20* of mass
(esp. noise↔structural and nonstationarity↔structural — the two that, if confused, sink the
whole idea).

**Falsifiers (shadow):** any of —
(i) cannot separate `irreducible_noise` from `structural_inadequacy` (the classic over-growth
failure); (ii) confuses `nonstationarity` with `structural_inadequacy` (would "grow a bigger
map" when it should forget); (iii) predicts `grow` in `stable_known`/`noisy`/`frontier` worlds
(grows everywhere). Any one firing ⇒ shadow FAILS; log which, with the confusion matrix.

### 2B. Repair controller (N3-order; runs in **control mode** only after shadow passes)

**Models:** regime-conditional usefulness of each repair action — i.e. which repair lowers
*live, held-out* surprise given the current diagnosis.

**Mismatch signal:** third-order error = realized usefulness of the chosen repair − predicted
usefulness (did acting on the diagnosis actually help on fresh experience).

**Control surface (the repair-action set; §4):** `{continue, explore, grow, split,
add_context, forget, compress/merge, quarantine, mark_frontier, halt/consult}`. The first
release tests the **5-action core**: `{continue, explore, grow, forget, quarantine}` (the
others are stretch; `grow` already exists, `forget`/`quarantine` are the new operators built in
N3b).

**Success (control):** lower live surprise than (a) no-controller / always-Exp-154-growth, and
(b) random-repair, on the same fork; fewer false expansions in noise/static; better old-region
stability after world change; correct action in each world type — all predeclared thresholds.

**Falsifiers (control):** (i) no better than always-grow (no independent value); (ii) matched by
an *offline-retuned* fixed policy (config, not a layer); (iii) over-forgets stable worlds or
fails to grow in genuinely structured-expanding worlds. Any ⇒ control FAILS for that rung.

**Mode discipline:** **shadow first, always.** Control is unlocked per world type only after
shadow passes its gate on that type. This is the project's standing pattern (the surprise
ledger / exocortical-N2 was shadow; the live-probation growth was promoted only after replay
validated it).

---

## 3. The procedural world family

All deterministic given (seed, layout). Extend `active_loop/worlds.py` (which already returns
plain serialisable dicts). Each world declares: generation procedure · controllable params ·
observations · actions · what is hidden · ground-truth structure · correct repair · failure
signature. The existing three (`learnable`, `noisy`, `aliased`, `nonstationary`) cover A/C/D/E;
B/F/G are new.

| id | world                | exists?                  | ground truth                         | correct repair        | failure to catch        |
| -- | -------------------- | ------------------------ | ------------------------------------ | --------------------- | ----------------------- |
| A  | static bounded       | `learnable()`            | map already adequate                 | continue / nothing    | spurious grow/forget    |
| B  | expanding learnable  | **new** `expanding()`    | new *reachable* structured regions   | explore → grow        | never explores; no grow |
| C  | expanding aliased    | `aliased()` + reveal     | one cell hides ≥2 causes             | grow / split          | "more data" only        |
| D  | expanding noisy      | `noisy(p_true)`          | irreducible entropy floor            | quarantine            | repeated false grow     |
| E  | nonstationary        | `nonstationary(...)`     | old region remapped at step T        | forget/adapt          | grows a bigger map      |
| F  | entropic spreading   | **new** `spreading()`    | clusters smear; covariances inflate  | compress / accept     | endless grow chasing it |
| G  | horizon-limited      | **new** `horizon()`     | structure beyond reachable frontier  | mark unreachable      | treats unseen as absent |

**B — `expanding(open_schedule, ...)`:** start with a sub-grid reachable; unlock new cells at
scheduled steps, each carrying *learnable* deterministic color structure. Hidden: the schedule
and the new cells' colors. Correct: coverage rises (explore), then a clean surprise drop after
learning (no growth needed if it's just more cells of existing colors; growth needed only if a
new color appears). Failure: never visits unlocked cells, or grows on cells it simply hasn't
sampled.

**F — `spreading(smear_rate, ...)`:** start from `learnable`/`aliased`; over time inflate the
*effective* emission covariance (jitter each cell's color with growing probability, or in the
continuous substrate widen the generating Gaussians). Hidden: the smear rate. Ground truth: the
*irreducible* entropy is rising — same shape as D but **time-varying**, so the agent must tell
"world getting noisier" from "I need more components." Correct: compress/accept (raise tolerance,
maybe merge), *not* grow. Failure: monotonically growing component count.

**G — `horizon(reach_mask, ...)`:** the full cmap is larger than the agent's reachable set; a
`reach_mask` marks which cells the agent's actions can actually enter. The agent gets *hints*
of beyond-frontier structure (e.g. boundary observations, or a low-rate leak of an
unreachable-cell color) but cannot act to resolve them. Hidden: the true beyond-frontier map.
Ground truth: structure exists but is non-actionable until/unless `reach_mask` later opens.
Correct: hold calibrated frontier uncertainty, `mark_frontier`, do **not** grow to "explain"
the leak. Failure: spends growth budget modeling cells it can never reach, or asserts the
frontier is empty.

Every world ships with `analytic_floor`-style ground truth where applicable (D and F have a
computable irreducible entropy; the quarantine/compress decisions are graded against it).

---

## 4. The repair-action set

For each: when to use · when **not** · expected metric signature · falsifier · independent test.

| action          | use when                              | do NOT use when                     | metric signature                                   | independent test                                          |
| --------------- | ------------------------------------- | ----------------------------------- | -------------------------------------------------- | --------------------------------------------------------- |
| **continue**    | ordinary ignorance; slope still −     | plateaued high                      | surprise still descending, slope < 0               | on `learnable`, beats grow (no spurious components)       |
| **explore**     | exploration deficit; low coverage     | already covered                     | coverage ↑ then surprise ↓ on new cells            | on `expanding`, raises coverage before any grow           |
| **grow/split**  | structural inadequacy / aliasing      | noise, nonstationarity, frontier    | `LiveProbation` keeps (≥0.1-nat live drop)         | the Exp-154 move; on `aliased` keeps, on `noisy` reverts  |
| **add_context** | aliasing unsolved by spatial split    | spatial split already worked        | held-out drop only after a context dim added       | stretch; needs a context variable the substrate lacks     |
| **forget/decay**| nonstationarity / stale evidence      | stable world                        | recovery-time ↓ after decay; old map *replaced*    | on `nonstationary`, beats grow on post-remap recovery     |
| **compress/merge** | entropic spreading; unused structure | active distinct structure        | component count ↓, held-out NLL flat or better     | on `spreading`, count stops climbing, NLL not worse       |
| **quarantine**  | irreducible noise                     | learnable-but-unlearned             | growth attempts on region → 0; floor respected     | on `noisy`, suppresses repeated false grow vs. always-grow|
| **mark_frontier** | unreachable frontier                | reachable-but-unexplored          | frontier uncertainty stays high & *calibrated*     | on `horizon`, no growth spent on unreachable cells        |
| **halt/consult**| evaluation artifact; conflicting signals | clear diagnosis               | controller abstains; logs the conflict             | inject a metric-artifact world; controller must abstain   |

The **new operators to build** (N3b): `forget` (per-region evidence decay — down-weight
`counts`/replay or inflate NIW `kappa`/`nu` toward the prior for a region) and `quarantine`
(freeze the spawn budget for a region: set `spawn_budget[k]=0` and stop feeding its alarm).
Both are small, local additions to the existing growth state; neither edits a FROZEN path.

---

## 5. Benchmark metrics

**Prediction / surprise:** global live surprise; per-region & per-color surprise; surprise
slope; ceiling-event rate.
**Structural growth:** #expansions; accepted/rejected; **false expansions** (noisy/static —
the over-growth failure); **missed expansions** (aliased/learnable — the under-growth failure);
growth efficiency = surprise-drop per added component.
**Exploration:** coverage; frontier visits; repeated visits to stale/noisy regions (waste);
info-gain if available.
**Map quality:** localization error; map accuracy; calibration; posterior entropy; **old-region
stability** (variance in un-visited regions); **frontier-uncertainty calibration** (is "unknown"
held where it should be).
**Forgetting/adaptation:** recovery time after world change; stale-evidence decay; over-forgetting
in stable worlds; stability/plasticity trade.
**N3 / metacognition:** failure-mode classification F1; correct-repair rate;
false-positive / false-negative intervention rate; shadow-mode prediction accuracy; live
improvement once control is enabled; **regret vs. oracle repair policy**.

The headline numbers are the **false-expansion rate** (must be ≈0 in noisy/static/frontier) and
**missed-expansion rate** (must be ≈0 in aliased/learnable). Everything else is supporting.

---

## 6. Baselines

Every claim is run against these on the *same fork-from-snapshot*:

| baseline                          | controls for                          | embarrasses N3 if…                          |
| --------------------------------- | ------------------------------------- | ------------------------------------------- |
| no N3 / no diagnostic             | the layer doing anything              | matches N3 live surprise                     |
| no growth (fixed model)           | growth being the source of gains      | matches N3 (then growth is the story, not N3)|
| **always-grow** (Exp-154 machinery)| the diagnostic's selectivity         | matches N3 on aggregate live surprise        |
| random repair                     | structure of the policy               | matches N3 (policy is noise)                 |
| more-data-only                    | exploration/structure value           | solves aliasing alone (then C is mis-built)  |
| more-explore-only                 | growth value                          | solves everything                            |
| bigger fixed model                | budget vs. selectivity                | matches N3 at equal compute                  |
| forget-only                       | single-hammer adaptation              | matches N3 across world mix                  |
| **oracle repair policy**          | regret ceiling                        | (defines best achievable — N3 measured vs it)|
| **offline-retuned fixed policy**  | "is N3 just config?"                  | **matches N3 → N3 is a hyperparameter, REJECT**|
| hand-coded env-type classifier    | learned vs. rules                     | beats the learned diagnostic                 |

The two **load-bearing** comparators are **always-grow** (does the diagnostic's selectivity buy
anything?) and **offline-retuned fixed policy** (is the controller a real layer or a constant?).
A result where always-grow ties N3 on every world is the clean negative.

---

## 7. The experiment ladder (N3a–N3e)

Each rung names: script · setup · prediction · metrics · falsifier · artifacts · docs. All rungs
follow `fork()`-control discipline, commit script + `experiments/outputs/expNN.txt` +
`expNN_verdict.json` in one commit with the EXPERIMENTS.md entry, and run ≥3 fresh seeds.

**N3a — Shadow diagnostic.** *Script:* `expNN_n3a_shadow_diagnostics.py`. *Setup:* run the M3
loop across A–G; the diagnostic predicts category + repair each window without acting; score vs.
the generator's ground-truth regime. *Predict:* macro-F1 ≥ 0.70, noise↔structural and
nonstat↔structural confusion each < 0.20. *Falsifier:* §2A (i)/(ii)/(iii). *Artifacts:*
confusion matrix per world, per-seed F1 table. *Docs:* this spec + EXPERIMENTS.md + card STATUS.
**This is the gate; nothing controls anything until it passes.**

**N3b — Controlled intervention (5-action core).** *Script:* `expNN_n3b_controlled_repair.py`.
*Setup:* unlock control on the world types that passed N3a; build `forget` + `quarantine`
operators; controller picks among `{continue, explore, grow, forget, quarantine}`. *Predict:*
lower live surprise than always-grow AND random-repair on the same fork; false-expansion ≈0 in
D/A; correct recovery in E. *Falsifier:* §2B (i)/(ii)/(iii) — especially the **offline-retune
control matching N3**. *Artifacts:* per-world surprise traces, repair ledger, the 4-way
fork comparison (none / always-grow / N3 / offline-retune). *Docs:* EXPERIMENTS.md + card.

**N3c — Open-horizon benchmark.** *Script:* `expNN_n3c_open_horizon.py`. *Setup:* a long life on
`expanding` + `horizon` with periodic unlocks; bounded map. *Predict:* useful growth in reachable
structured regions; *no* endless growth in noisy/unreachable regions; stable old maps; frontier
uncertainty calibrated. *Falsifier:* component count grows without bound, OR frontier asserted
empty, OR old regions destabilize. *Artifacts:* lifetime component-count curve, frontier
calibration plot. *Docs:* EXPERIMENTS.md + `docs/research/bounded-map-open-world.md`.

**N3d — Budgeted cognition.** *Script:* `expNN_n3d_budget.py`. *Setup:* impose a model-size
budget; controller must choose grow vs. compress/merge vs. forget. *Predict:* better
surprise-per-unit-budget than always-grow; graceful degradation; compression does not destroy
important old knowledge (held-out NLL on old regions stays within ε). *Falsifier:* compression
craters old-region accuracy, OR budget makes no difference vs. unbounded. *Docs:* EXPERIMENTS.md.

**N3e — Transfer to 2D continuous world.** *Script:* `expNN_n3e_continuous_transfer.py`. *Setup:*
move from grid to a simple continuous 2D world with local/raycast-style sensors (the
`creature_continuous` substrate). *Predict:* the *same* failure-mode distinctions (esp.
noise vs. structural vs. nonstationary) survive richer geometry. *Falsifier:* the diagnostic
that worked on grids collapses to chance on continuous geometry ⇒ the result was grid-specific.
*Docs:* `docs/research/agents-that-know-when-worldview-is-too-small.md`.

Gate discipline (ladder law): **do not climb a rung on an unsupported lower rung.** If N3a's
gate fails for a world type, that world type is out until the diagnostic is fixed. If the toy
world stops supplying *discriminating* failures (e.g. always-grow ties N3 everywhere), that is
the terminus — log it, name the missing world-richness, stop. Saturation is a finding.

---

## 8. Concrete implementation plan

Additive only; no FROZEN paths (`eval/`, `model_spec.py` autopilot trust boundary) touched.

```
active_loop/open_world.py      # new: expanding(), spreading(), horizon() generators
                               #   + reach_mask / open_schedule plumbing; mirrors worlds.py
                               #   (plain serialisable dicts, deterministic given seed/layout)
active_loop/frontier.py        # new: frontier-belief state (reachable set, hint accumulator,
                               #   calibrated unknown-but-not-actionable estimate)
active_loop/n3_diagnostics.py  # new: signal extractors (table §2A) + the 10-way classifier;
                               #   pure functions over the M3 loop's observable signals
active_loop/repair_actions.py  # new: forget()/quarantine()/compress() operators over the
                               #   existing growth state; grow() delegates to growth.py
active_loop/meta_metrics.py    # new: F1/regret/false-expansion/frontier-calibration scorers
experiments/expNN_n3a_shadow_diagnostics.py
experiments/expNN_n3b_controlled_repair.py
experiments/outputs/expNN.txt + expNN_verdict.json   # via verdict.write_verdict
docs/specs/n3-open-world.md                          # this file
docs/research/bounded-map-open-world.md              # narrative writeup (after N3c)
docs/research/agents-that-know-when-worldview-is-too-small.md  # flagship (after N3e)
tests/test_open_world.py, tests/test_repair_actions.py, tests/test_n3_diagnostics.py
```

Build order: **worlds B/F/G + tests → diagnostic signals + classifier → N3a shadow run →
forget/quarantine operators + tests → N3b control run.** Stop and write the verdict after each.

Engineering invariants (inherit the lab's standards): procedural seeds; deterministic replay;
small fast runs (target < ~1 min like the existing suite); fresh-seed validation distinct from
design seeds; multiple layouts; one-command reproduction; `verdict.json` per experiment;
**headline-number verification** via a tiny audit assert in the script (false-expansion rate,
macro-F1) so the claimed number is recomputed from raw output, not transcribed.

Reuse, don't reinvent: `check_ceiling`/`check_color_alarm` for the surprise signals;
`LiveProbation` for the grow keep/revert (forget/quarantine reuse the same snapshot→resolve
shape); `worlds.nonstationary`/`noisy`/`aliased` for D/E/C; `verdict.write_verdict` for output;
`fork()` for every control.

---

## 9. The first minimal experiment to run

**N3a-min — shadow diagnostic on the four worlds that already exist.** Do *not* build B/F/G yet.
Run the M3 loop on `learnable` (A), `aliased` (C), `noisy` (D), `nonstationary` (E) — all already
in `worlds.py` — and have a first-cut diagnostic emit, per window, a label in
`{stable_known, structural_inadequacy, irreducible_noise, nonstationarity}` (the four those
worlds actually instantiate). Score macro-F1 and the **two confusion pairs that matter**
(noise↔structural, nonstat↔structural) over **≥3 fresh seeds × the 3 aliased layout seeds
(7/11/13)**.

- *Predeclared pass:* macro-F1 ≥ 0.70 **and** both critical confusions < 0.20.
- *Predeclared falsifier:* either confusion ≥ 0.20, or F1 ≤ 0.40 (≈ chance for 4 classes).
- *Why this first:* it needs **zero new world code**, reuses the existing detectors as features,
  and directly tests the load-bearing claim — *can it tell noise from structure, and change from
  inadequacy?* If it cannot on the worlds we already trust, the whole direction halts here, and
  that is the cheapest possible honest negative.
- *Artifacts:* `experiments/expNN_n3a_min_shadow.py`, `outputs/expNN.txt`, `expNN_verdict.json`,
  a 4×4 confusion matrix in the output, EXPERIMENTS.md entry tagged shadow / NEW-INSIGHT-or-
  NEGATIVE, card STATUS bumped.

---

## 10. Safest public-facing claim if N3a succeeds

> *In a controlled toy benchmark, a bounded active-inference agent's diagnostic layer
> distinguished why its prediction error was high — separating irreducible noise from
> structural inadequacy and from world-change — above a predeclared accuracy bar, on held-out
> seeds and layouts, in shadow mode (no control yet).*

What it does **not** license: any control claim (that's N3b), any consciousness/awareness claim,
any open-world/lifelong claim (that's N3c), any scale claim. "Shadow mode" and "toy" stay in the
sentence. This is the project's standard altitude — an instrument that classifies the *cause* of
surprise, nothing more.

---

## 11. The strongest falsifier that should halt or redirect

> **Always-grow ties (or beats) the full N3 controller on live held-out surprise across the
> world family, AND/OR an offline-retuned fixed repair policy matches N3.**

If that holds, then in this world richness there is **no repair-selection competency that beats
the single existing hammer** — the diagnostic carries no decision-relevant information the
growth machinery doesn't already act on, and N3 is config, not a layer. Per the anti-regress
law this is the terminus: log the wall, name the missing world-richness (likely: worlds where
the *wrong* repair is actively harmful enough that selectivity pays — and if our toy can't
supply that, say so), and **do not climb to N3c+ on an unsupported N3b.** A negative here is the
most valuable outcome the direction can produce, because it kills an impressive-but-meaningless
toy before it is built.

Secondary halt (shadow stage): the noise↔structural or nonstationarity↔structural confusion
cannot be driven below the predeclared bar — the diagnostic cannot tell apart the two
distinctions the entire direction rests on. Redirect to enriching the worlds or sharpening the
signals before any control work.

---

## Pointers

- `loop/directions/n3-bounded-map-open-world.md` — the companion direction card (STATUS-tracked).
- `docs/specs/n-order-self-modeling.md` — the ladder this slots into (N2/N3 rungs).
- `loop/directions/meta-calibration-n3.md` — the abstract N3 control-surface card this benchmark
  instantiates and extends (adds open-world/frontier + forget/quarantine operators).
- `active_loop/worlds.py` — A/C/D/E generators reused as-is; B/F/G extend it via `open_world.py`.
- `active_loop/growth.py` — `check_ceiling`, `check_color_alarm`, `LiveProbation`, K-selection.
- `active_loop/verdict.py` — per-experiment `verdict.json`.
- `loop/VALIDATION.md` — binding honesty contract (predeclared falsifiers, fresh seeds, fork
  controls, negatives-as-findings); `loop/PROTOCOL.md` — per-iteration discipline.
