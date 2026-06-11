# Design: Mirro N-order self-modeling ladder (N0–N7)

**Date:** 2026-06-10
**Status:** Design (direction-level; N3 card live, N1 green, N2 in progress)
**Repo:** `/Users/mirro/Projects/active-loop` (extends the persistent-creature substrate)

## Origin

User goal: explore whether **increasingly higher-order self-modeling layers** can be
implemented as **falsifiable functional capabilities**, not vague consciousness claims —
grounded in active inference / predictive processing, Anil Seth–style top-down self-modeling
and interoception, and Joscha Bach–style "more meta" levels of lucidity, *without overclaiming
consciousness*. Each layer must earn its existence by **controlling something the layer below
cannot control.**

## Honesty up front (binding)

- **Higher N is NOT "more conscious."** N is order-of-self-reference, not a sentience scale.
  Every layer is a measurable control competency; the inner-experience question stays
  explicitly unverified in both directions (see `loop/VALIDATION.md`).
- **Bach "lucidity" is admissible only operationally** — "policy increasingly governed by
  higher-order self-models" — measured as control surfaces, never as a consciousness claim.
- **Functional language only** ("functionally prefers", "functional valence", "confidence").

## The governing law (falsifiability + anti-regress in one)

> A layer **Nₖ is real iff** there exists a *constructible perturbation* that degrades a
> well-tuned Nₖ₋₁ agent's performance, which an Nₖ agent **detects and corrects via a control
> surface Nₖ₋₁ does not possess.** No discriminating test ⇒ Nₖ is notation, not a layer.

Consequence (useful finite hierarchy vs. pointless infinite regress): the ladder **terminates
where the environment stops supplying discriminating failures.** Bach-style "more meta" is
meaningful only while each order adds a *control surface*; a level that can only re-describe the
level below without controlling anything new is the regress to avoid. A toy gridworld likely
saturates around N3–N4 — and that saturation is itself a finding: higher N needs a richer world
even to be *testable* (same world-richness governor as Exp 46).

## The ladder

Each layer: **[models] · [mismatch signal] · [control surface it owns] · [falsifiable test] ·
[what invalidates it].**

### N0 — Reflexive / homeostatic regulation
- **Models:** essential variable `x` vs fixed setpoint `x*` (energy, integrity). No world model.
- **Mismatch:** regulation error `x − x*` (reactive, post-violation).
- **Control:** fixed reflex actions pushing `x → x*`.
- **Test:** perturb `x`; reflex restores within bounds without learning; no-N0 twin drifts to death.
- **Invalidated if:** restoration secretly needs world-state knowledge (N1 in disguise / leakage).

### N1 — First-order world model  *(green)*
- **Models:** hidden world states `s_t` via `p(o|s;A)`, `p(s|s',u;B)`.
- **Mismatch:** sensory prediction error = variational free energy `F`.
- **Control:** action by expected free energy `G` (epistemic + pragmatic) — anticipatory nav/foraging.
- **Test:** place-field self-organization (✓ Exp 20, 0.00 bits); goal-directed nav beats random.
- **Invalidated if:** posterior collapse / symmetric saddle (the `open_problem.html` failure).

### N2 — Metacognitive model over N1  *(in progress)*
- **Models:** reliability of N1 — precision `π` of `q(s)`, N1's free-energy floor, and a
  classification of N1 failure: **noise vs. structural mismatch vs. volatility**.
- **Mismatch:** second-order error — predicted N1 reliability vs. realized N1 performance
  (predicted confidence vs. actual accuracy = calibration error).
- **Control:** precision-weighting (gain on N1 errors), epistemic foraging when confidence low,
  and *triggering N1 structural expansion*.
- **Test:** metacognitive calibration **meta-d′ > 0** (confidence tracks accuracy); anti-overfit
  falsifier — must NOT trigger expansion under pure noise (Yu–Dayan expected uncertainty);
  N2-less twin can't grow to a richer world or over-expands.
- **Invalidated if:** confidence ⊥ accuracy; expansion on noise; or `π` is just a relabel of `F`
  with no independent control effect.
- **Status:** the **surprise ledger (exp54) is an exocortical N2** — the diagnosis lives outside
  the creature. Internal-N2 rungs are in `loop/directions/functional-emergence.md`.

### N3 — Meta-calibration / self-authoring controller over N2  *(card live)*
- **Models:** regime-conditional **trustworthiness of N2** + N2's policy parameters θ_N2.
- **Mismatch:** third-order error — realized vs. predicted usefulness of *acting on N2's diagnosis*.
- **Control:** rewrites θ_N2 (thresholds, expansion rate, precision learning-rate); can
  **override/suspend** an N2 diagnosis; switches metacognitive strategy by regime.
- **Test:** construct a regime where N2 is systematically wrong (confidence anti-correlated with
  accuracy, or an expansion-trap); N3 detects and recovers where N2-only underperforms; and
  N3's fix is NOT reproducible by offline retuning of θ_N2 (else it is config, not a layer).
- **Invalidated if:** N3 never disagrees with N2 (zero independent variance); or reducible to an
  offline hyperparameter retune.
- **Central hypothesis:** *N3 is not "more metacognition" — it is **agency over metacognition**.*
  Full design in `loop/directions/meta-calibration-n3.md`.

### N4 — Identity / policy-continuity model
- **Models:** the agent's dispositions over long timescales — trait/value vector + drift rate.
- **Mismatch:** identity prediction error — predicted future policy vs. actual policy drift.
- **Control:** commitment / policy regularization — the *inertia* of value revision.
- **Test:** consistency under transient pressure, revision under sustained evidence (the Exp
  48/49 inertia law, now self-modeled); N4 twin shows higher temporal self-similarity.
- **Invalidated if:** freezes policy (rigidity) or has no effect on drift (epiphenomenal).
- **Status:** **exp55 already probed it** — "identity dominated by recent history" (ages
  anti-correlate −0.71) is a low-policy-continuity reading ⇒ mirro currently has no N4.

### N5 — Interoceptive / motivational self-model  *(= the M4 spec)*
- **Models:** internal economy — drives, energy/compute budget, uncertainty tolerance — *and
  their predicted trajectories* (interoceptive inference). Seth's beast-machine layer.
- **Mismatch:** interoceptive/allostatic prediction error — predicted vs. actual internal state.
- **Control:** motivational arbitration + **allostasis** (act before setpoint violation);
  exploration-vs-conservation trade by budget.
- **Test:** anticipatory regulation lead-time > 0 (forages before depletion); under induced
  scarcity, shifts uncertainty tolerance; N5-less twin depletes more.
- **Invalidated if:** collapses to N0 (reactive only); drives don't arbitrate; interoceptive
  prediction adds no control over the reactive baseline.
- **Note:** N5 is the **predictive/self-modeling version of N0** — same axis (the body),
  different order; shares the "model-predicts-itself" substrate with N2. See
  `docs/specs/m4-affective-dyad.md`.

### N6 — Ontology / implementation self-model  *(deep target)*
- **Models:** the agent's own representational machinery as an object — state-space, observation
  model `A`, preference-formation, memory compression, expansion criteria.
- **Mismatch:** structural model-evidence comparison at the meta level — "my state-space *carves*
  the world wrong" (not "more states" but "a different factorization").
- **Control:** ontological revision — split/merge states, re-factorize, redesign `A`, change
  memory compression. Structure learning *as a self-directed act*.
- **Test:** in a world requiring a different *carving* (relevant latent is a conjunction the
  current `A` can't express), N6 restructures where an N1+N2 agent (states-within-fixed-
  factorization) provably cannot.
- **Invalidated if:** reduces to N2's "add a state" (same hypothesis space); the new ontology is
  designer-supplied (leakage); or hits the **awareness wall** (can't represent the needed
  ontology even as a candidate — the honest expected ceiling). This is `open_problem.html`
  tab-4A (tabula-rasa structure); see `docs/specs/structure-learning.md`.

### N7 — Collective / multi-agent integration  *(speculative; partly behind a ceiling)*
- **Models:** other agents as generative models; self as one node; shared/negotiated ontology.
- **Mismatch:** theory-of-mind error; ontology misalignment across agents.
- **Control:** communication, coordination, norm/ontology alignment — shared signaling.
- **Test:** multiple mirros develop coordinated signaling that improves *joint* free energy
  beyond isolated agents; emergence of shared signal↔referent mappings (`converse_demo` lineage).
- **Invalidated if:** "communication" carries no mutual-predictability gain; coordination is
  hardwired; collapses to independent agents.
- **Honest flag:** N7 contains the **language-from-scratch / grammar ceiling**
  (`open_problem.html` tab-4B) — reachable as coordination, not as emergent grammar.

## Build order

Spine: **N1 ✓ → N2 (now) → N3 (next).** Then:
- **N5 in parallel** — shares the "model-predicts-itself" substrate; M4 spec already exists.
- **N4 passively accumulates** via the persistence substrate (exp55 already probes it).
- **N6 deep reach** after N2/N3 are solid (needs N2 inadequacy-detection + a structure operator).
- **N7 last**, only as far as the grammar ceiling allows.

Do NOT climb to N4+ on an unsupported N3. Each layer's discriminating-perturbation gate must
pass before the next is built — and if the toy world stops supplying discriminating failures,
log that as the terminus (enrich the world or stop), per the governing law.

## Pointers

- `loop/directions/meta-calibration-n3.md` — the live N3 direction card (next build target).
- `loop/directions/functional-emergence.md` — internal-N2 / surprise-ledger / personality rungs.
- `docs/specs/m4-affective-dyad.md` — N5 (interoception/motivation).
- `docs/specs/structure-learning.md` — N6 (ontology revision) frontier.
- `open_problem.html` — the ceilings N6 (tab 4A) and N7 (tab 4B) confront.
- Memory: `n-order-self-modeling-ladder.md`, `research-stance-functional-emergence.md`.
