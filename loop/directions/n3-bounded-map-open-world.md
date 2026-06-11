# direction: n3-bounded-map-open-world

**Question.** Can a bounded active-inference agent distinguish *why* its prediction error is
high — ordinary ignorance vs. genuine new structure vs. irreducible noise vs. stale evidence vs.
aliasing vs. nonstationarity vs. unreachable frontier — and choose the matching repair
(continue / explore / grow / forget / quarantine / mark-frontier) instead of reaching for the
one hammer (grow) every time?

**Central hypothesis (binding, falsifiable).**
> *A bounded agent in an open-ended world has a repair-selection competency that beats the
> single best fixed repair. A **diagnostic layer** reads the agent's own learning state and
> classifies the cause of surprise; a **repair controller** acts on that diagnosis. The
> controller is a real layer iff selecting among repairs beats always-grow AND is not
> reproducible by offline retuning of the growth thresholds.*

**Relation to the ladder (not a competing N3).** This is the **open-world benchmark
instantiation** of the N2→N3 spine (`docs/specs/n-order-self-modeling.md`): diagnosis is
N2-order work (extends N2's noise-vs-structural-vs-volatility classifier to a 10-way failure-mode
set); choosing/authoring the repair is the N3 control surface already defined in
`meta-calibration-n3.md`. It *extends* that card with two things the static-richness version
lacks: an open-ended world that can grow past the model over a long life (horizon / entropic
spreading), and two repair operators the current machinery has no representation for —
**forget/decay** and **quarantine**. Full design: `docs/specs/n3-open-world.md`.

**Why it matters.** Exp 154 showed the creature can already *grow a component on demand* when a
color is aliased. But growth is the wrong move for most of the ways a world can surprise you
(noise → quarantine; world-change → forget; unreachable frontier → hold uncertainty). The whole
claim rests on the anti-regress law: a repair controller earns its existence only if the *wrong*
repair is harmful enough that selectivity pays. If always-grow ties it everywhere, that is the
finding — there is no N3 competency at this world richness; log the wall and stop.

**What each quantity is.**
- *Diagnostic observations:* live surprise + slope + ceiling-event rate (`check_ceiling`),
  per-color residuals (`check_color_alarm`), localization uncertainty (`Sigma_p_diag`), evidence
  mass (`counts`), component count, growth keep/revert ledger (`LiveProbation`), coverage,
  novelty rate, frontier-hit rate. (All already computable from the M3 loop.)
- *Diagnostic latent:* 10-way regime label {stable_known, ordinary_ignorance,
  exploration_deficit, structural_inadequacy, aliased_context, nonstationarity, stale_evidence,
  irreducible_noise, entropic_spreading, unreachable_frontier}.
- *Controller control surface:* the repair-action set; first release tests the 5-action core
  {continue, explore, grow, forget, quarantine}.

**Experiment ladder (N3a–N3e; full detail in the spec §7).**
1. **N3a — shadow diagnostic (the gate).** Predict regime + repair without acting; score macro-F1
   and the two load-bearing confusions (noise↔structural, nonstat↔structural) across A–G on fresh
   seeds. FALSIFIER: confusion ≥ 0.20 on either pair, or grows-everywhere. No control until this
   passes. **First minimal run (N3a-min):** the four worlds that already exist in `worlds.py`
   (learnable/aliased/noisy/nonstationary), zero new world code — pass iff macro-F1 ≥ 0.70 and
   both critical confusions < 0.20.
2. **N3b — controlled intervention.** Build forget + quarantine; controller picks the 5-action
   core. FALSIFIERS (both load-bearing): must beat always-grow, must NOT be matched by an
   offline-retuned fixed policy.
3. **N3c — open-horizon.** Long life on expanding + horizon worlds; bounded map; calibrated
   frontier uncertainty; no endless growth.
4. **N3d — budgeted cognition.** Model-size budget; grow vs. compress vs. forget.
5. **N3e — continuous transfer.** Same distinctions survive 2D continuous geometry.

**Failure modes that invalidate the layer (write each verdict honestly).**
- *No selectivity value:* always-grow ties N3 on live surprise across the world family → the
  diagnostic carries no decision-relevant information; not a layer (the strongest falsifier).
- *Reducible to config:* an offline-retuned fixed repair policy matches N3 → hyperparameter, not
  a controller.
- *Cannot tell noise from structure / change from inadequacy:* the two confusions the whole
  direction rests on stay above the bar → shadow fails; enrich worlds or sharpen signals.
- *Leakage:* worlds hand-tuned so the right answer is obvious → declare the generator as provided
  and require fresh-seed, multi-layout generalization (the standing gate).

**Discipline notes.** All `persistent-creature.md` / `VALIDATION.md` notes apply: predeclared
property-level falsifiers, ≥3 fresh seeds reported in full, `fork()`-only controls, `verdict.json`
per experiment, script+output committed with the EXPERIMENTS.md entry, headline-number
re-verification in-script. Additive code only; no FROZEN paths. **Shadow before control, always.**
No consciousness/AGI/recursive-self-improvement claims — N3 here is a repair-selection control
competency, nothing more. "Expanding universe" is an intuition pump, not a cosmology claim.

**Stop condition.** Exhausted when N3a–N3b have a verdict: either the controller demonstrably
beats always-grow and is not offline-retunable (SUPPORTED — proceed toward N3c+), or always-grow
ties it / offline-retune matches it / the load-bearing confusions can't be driven below the bar
(REJECTED at this world richness — log the wall, name the missing world-richness, do not climb).
Either verdict is clean.

**STATUS.** state: exploratory · latest: Exp 155 · depends-on: continuous-creature, meta-calibration-n3, persistent-creature · reusable: open-world procedural benchmark (expanding/spreading/horizon), failure-mode diagnostic over learning state (held-out-predictive split test = shadow analog of live probation), repair-action set with offline-retune falsifier, forget/quarantine operators · why: instantiates the N2→N3 spine as an auditable open-world benchmark; tests whether repair-selection beats the single grow hammer · next-falsifiable: N3b controlled repair — build forget+quarantine operators, let the controller pick among {continue,explore,grow,forget,quarantine}; must BEAT always-grow on live held-out surprise AND must NOT be matched by an offline-retuned fixed policy (else N3 is config, not a layer)
