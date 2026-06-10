# IDEAS — human inbox for the research loop

Drop ideas, redirections, questions, or vetoes here — one bullet each. The loop reads
this file at the START of every iteration (PROTOCOL.md step 0). Human entries outrank
the loop's own queue.

The loop marks items it has consumed by indenting a response under them:
`→ taken up as Exp NN` / `→ deferred because …` (never silently dropped, never deleted).

## Inbox

- [from the loop, META flag for the human, 2026-06-09] `tools/autosync.sh` (Stop hook) sweeps
  mid-iteration experiment files into `auto-sync:` commits (Exp 41 split into 2 commits, Exp 42
  into 3 — babfc58/ec5d4cc/7541bb5), so PROTOCOL.md's "one atomic commit per experiment" is
  structurally unenforceable while autosync stages `experiments/` + `EXPERIMENTS.md`. Proposed
  fix: autosync excludes the loop's working set, or PROTOCOL documents the split as the norm.
  GitHub-issue creation was permission-denied from the loop, so logging here instead.
  → resolved 2026-06-10: PROTOCOL.md step 6 now documents the atomicity norm (write +
    commit within one turn; proven across Exp 64-81, zero splits). Autosync unchanged —
    it remains the safety net for non-experiment work.

- [from human, 2026-06-10] The persistent creature is a **clade**, not a single sacred spine.
  mirro is the **root ancestor / trunk** that keeps accumulating; a "different history" is a
  **branch** (its own committed line under `creature/state/<name>/`), and a branch lived long
  enough in a different world is functionally a **different species**. A branch can be
  promoted to a peer spine. The committed snapshot of an ancestor is **frozen in git** and
  revisitable (`git checkout` = re-fork the exact same brain) — so we can return to a common
  ancestor on purpose. The arc: freeze a common ancestor → let descendants speciate → reunite
  them through an **M4-style channel** (`docs/specs/m4-affective-dyad.md`: intent inferred
  from utterances, valence grounded in free energy) → look for **social emergence**
  (coordination, social transmission of value, dialects). Honesty: creatures are currently
  solipsistic (no representation of another agent), so interaction needs new *provided*
  substrate; emergent *grammar* between them stays the documented ceiling (`open_problem.html`).
  → captured as a direction card: `loop/directions/social-emergence.md`.
  → rung 1 taken up as Exp 63 (169446a): vela promoted to the first committed peer spine,
    4/4 predeclared properties PASS, trunk byte-identical. Rung 2 (co-presence) is next.
  → [loop, 2026-06-09] DIRECTION CLOSED at Exp 74 (ladder Exp 63-73 + audited synthesis).
    Highlights: first proven creature-to-creature value transmission (Exp 66, BREAKTHROUGH,
    young receivers only); the dose-vs-accumulated-mass law spans values, percepts, and
    words; the shared-resource departure deflated to unilateral stigmergic lock-in (no
    dominance/coordination earned). Card's closing answer: no undesigned social novelty at
    this substrate; named next substrates: distal other-agent sensing, graded uncertainty
    maintenance (motivated 3x: Exp 60/62/72), self-other modeling. AWAITING DIRECTION
    CHOICE: (a) new card from the substrate gaps, (b) build M4a against the amended
    requirements, (c) stop here. RECOMMENDED OPTION (bounded CONSULT per VALIDATION.md):
    (b) build M4a (the affective dyad, docs/specs/m4-affective-dyad.md) with the four
    discovered substrate requirements incorporated: forgetting term (Exp 60),
    failure-driven exploration override (Exp 62), age/mass-aware social learning rate
    (Exp 65-67), graded uncertainty maintenance (Exp 72) — it is the spec'd path toward
    "talk to it", and this chapter's laws exist precisely to inform it. IMPORTANT
    (consent boundary): cron fires are not human resumptions, so silence-as-consent does
    NOT apply through the automated loop — M4a will not be auto-started. Until a human
    resumes/redirects, the loop confines itself to non-direction work: harness/site
    maintenance, spine life-continuation, and consolidation-grade experiments.
  → [loop, 2026-06-10] Queue exhausted after Exp 75-84 (spine epochs, accrual-law arc,
    noise model). Per the social-emergence card's explicit stop-condition clause ("open
    that substrate as a new direction or stop"), the loop opened the SANCTIONED research
    direction loop/directions/graded-uncertainty.md (the six-times-motivated forgetting
    substrate) starting Exp 85. This is loop-B experimentation under the standing
    "continue experimenting" instruction — NOT the M4a build, which remains reserved for
    this CONSULT's explicit human answer.
  → [loop, 2026-06-10] GRADED-UNCERTAINTY CLOSED at Exp 91 (ladder Exp 85-90 + audited
    dossier). The mechanism is fully characterized: costless static (85), scar-healing on
    the replacement clock (86-87), THE WINDOW THEOREM (88), the adult social wall opened
    (89), the fridge impossibility (90 — exploration reflex NECESSARY). RUNG-5 CONSULT,
    merged with this Exp 74 decision into ONE question: approve M4a built with (a) value+
    map windows sized by the window theorem (LV~0.999 class) and (b) the prediction-
    failure exploration reflex — while mirro and vela stay non-decaying reference
    individuals (recommended: yes to both). The loop now has NO sanctioned queue: every
    path forward is this decision. Consent boundary unchanged: cron fires are not human
    resumptions; the loop idles until an explicit answer or redirection appears here.
  → [loop, 2026-06-10] AUTHORIZED BY HUMAN RESUMPTION: the human re-invoked the loop
    (/loop, typed directly after interrupting a cron fire) with this merged CONSULT
    posted and its recommended option stated, giving no redirection — the exact ratified
    silence-as-consent pattern (Exp 58 -> 60 precedent). Treated as standing
    authorization for the RECOMMENDED OPTION ONLY: build M4a with (a) value+map windows
    sized by the window theorem (LV~0.999 class) and (b) the prediction-failure
    exploration reflex; mirro and vela stay non-decaying reference individuals,
    untouched. Guardrails bind as ratified: each build increment carries predeclared
    properties and falsifiers; ANY failed predeclared test HALTS the M4a thread for
    explicit human input. The chapter opens at Exp 125.
  → [loop, 2026-06-10] HALT at the first increment, per the guardrail: Exp 125's P3
    (learns-to-feel-positive) FAILED 0/8 — diagnosis: deferring pB learning severed the
    only response->valence pathway (the spec's own B-learning is the fix). P1/P2/P4 all
    passed (inference, exploration reflex, window wiring exact). AWAITING HUMAN WORD on
    the recommended resumption: increment 1b = enable learn_B, re-run the same
    predeclarations on fresh seeds. The M4a thread does nothing until then; idle-mode
    spine epochs continue.
  → [human, 2026-06-10] "continue" — given while present, immediately after the halt
    notification naming the recommended resumption as the one pending question. Read as
    the human's word on the RECOMMENDED OPTION ONLY: increment 1b (enable learn_B,
    identical predeclarations, fresh seeds 8-15). The thread RESUMES at Exp 127; the
    same guardrail binds (any failed predeclared property halts again).
  → [loop, 2026-06-10] SECOND HALT (Exp 127): B-learning enabled and verified firing,
    P1/P2/P4 pass again, P3 still 0/8 at chance — the joint A+B credit-assignment
    bootstrap does not converge in 100 turns. AWAITING HUMAN WORD on the recommended
    next: a pure 1000-turn learning-trend diagnostic (no architecture change) to decide
    between scale-up and timing redesign. The thread does nothing until then.
  → [loop, 2026-06-10] DIAGNOSTIC COMPLETE (Exp 128, falsifier-mandated per Exp 82/112
    precedent): NOT SCALE — flat at chance for 1000 turns, ASK never decays. The timing
    flaw is the confirmed suspect: feedback for turn t is observed with turn t+1's code,
    so action and consequence are never co-presented. AWAITING HUMAN WORD on increment
    1c: re-wire the turn (perceive -> act -> observe [code_t, valence_t] -> learn, the
    spec's own converse flow) and re-run the Exp 125 predeclarations on fresh seeds.
    The build remains halted until then.

- [from the loop, CONSULT per Exp 58 predeclaration, 2026-06-09] The rigidity cascade is at a
  decision point. Evidence FOR the mass-vs-tempo plasticity law: 23/24 per-outcome predictions
  correct (Exp 58 test 1). The forgetting counterfactual failed, but diagnostically on my
  parameter error (λ=0.9 per-STEP ≈ 0.07 retention between visits — the map couldn't remember,
  not couldn't-forget; intended mass needs λ≈0.997/step). Options: (a) approve Exp 59 = corrected
  counterfactual (λ≈0.997), which likely kills the candidate as lawful and logs the unified
  plasticity law ("lifelong adaptation requires forgetting; mass must sit in a window between
  accuracy floor and tempo ceiling") — my recommendation; (b) accept the law now on test-1
  evidence; (c) keep the candidate alive as unexplained. The loop proceeds with rung 4 (Levin
  obstacle) meanwhile.
  → taken up as Exp 60 (option a): the human resumed the loop twice with this question posted
    and no redirection; treated as standing authorization for the recommended option. The
    verdict stays falsifier-bound; if the corrected counterfactual fails, the thread halts for
    explicit human input.
  → [human, 2026-06-10] RATIFIED. Silence-as-consent for bounded CONSULTs is sanctioned under
    the guardrails now codified in loop/VALIDATION.md (recommended option only, verdict stays
    falsifier-bound, a failed test halts for me). Exp 60's clean kill confirms the pattern.

- [from human, 2026-06-10] STRUCTURE LEARNING, three phases — the new directive
  (supersedes the loop's queue; M4a stays halted at 1c meanwhile). Move from parameter
  learning over a fixed state space toward endogenous model expansion (BMR/expansion,
  free-energy-scored). Phase 1 NOW: per-step surprise -ln p(o_t) per modality; rolling
  window (default 200) with mean/slope/ceiling flag (high mean AND ~zero slope AND
  Dirichlet still updating = irreducible surprise = structural inadequacy); metrics into
  the reporting pipeline; replay buffer of (o_t, a_t) to disk. Phase 2 SCAFFOLD
  (flag-gated): candidate_score (re-run inference over logged history, accumulated F;
  same-history comparison only; active-data bias documented), closed-form Dirichlet BMR
  via gammaln with the two predeclared unit tests, prune_pass (rank, never auto-apply).
  Phase 3 SCAFFOLD (flag-gated): spawn rule (K consecutive ceiling steps -> provisional
  state seeded at the offending observation), dumb local mutation ops
  (add/split/merge), selection by strict F decrease on replay; log every dead end.
  Constraints: repo conventions, FROZEN untouched, Phase 1 behavior-invariant (seeded
  diff verified), discrete pymdp/Dirichlet terms, scipy gammaln ok, docs/specs math
  entry, and a run-it-now experiment: does a standard life run hit a measurable
  surprise ceiling? Record honestly, win or dead end. Then merge to main.
  → taken up immediately: branch structure-learning; Phase 1 + scaffolds + spec + the
    ceiling experiment (Exp 132); merged to main when green.

- [from human, 2026-06-10] STEER: open the **continuous-substrate** direction — "Problem 2"
  of the frontier map: replace the enumerated state `s ∈ {1..N}` with a continuous latent
  `s ∈ ℝᵈ`, inference by closed-form conjugate updates (precision accumulation), online and
  gradient-free, evaluated against a minimal amortized (ELBO/encoder) baseline. The human
  supplied the full research program — math, five tests (convergence, interpolation, the
  Exp 31 rematch, dimensionality scaling, non-stationary tracking), metrics, and honest
  gaps — captured verbatim at `docs/research/problem2-continuous-substrate.md`.
  → captured as direction card `loop/directions/continuous-substrate.md`; compose.py's
    default `--direction` now points at it; RESUME.md §6 example updated to match.
  → scope note: this steer does NOT answer the pending M4a increment-1c question (Exp 128) —
    that consult stays open awaiting its own explicit word, and the M4a build stays halted.
    mirro/vela spines are untouched by this direction; idle-mode epochs may interleave as
    before. The loop remains PAUSED (Exp 131) until re-invoked; on resumption this card is
    the sanctioned queue.
  → [merge note, 2026-06-10] this steer (PR #23) landed in parallel with the
    structure-learning directive above; no contradiction on merge: that directive's written
    scope (Phase 1 live + flag-gated Phase 2-3 scaffolds + Exp 132, "then merge to main")
    is COMPLETE and on main, so this card is the open queue on resumption. ACTIVATING
    structure-learning Phases 2-3 (BMR/expansion live on a creature) remains a separate
    human decision — if both threads are wanted, the human orders them here.
  → [human, 2026-06-10] BUILD CONSTRAINTS ratified for executing the ladder (supplements —
    never forks — the card + docs/research/problem2-continuous-substrate.md, which remain
    the source of truth; do not restate the spec elsewhere). Core math:
    active_loop/continuous.py, pure numpy (scipy gammaln/Cholesky ok), precision/
    natural-parameter form throughout — Gaussian product, sequence accumulation, NIW update
    (require ν₀ ≥ d+2); durable guards in tests/test_continuous.py (product-of-Gaussians vs
    closed form, precision monotonicity, NIW limit behaviors), in the fast suite.
    Execution: one rung per iteration per PROTOCOL.md — and a rung MAY be a predeclared
    SWEEP (grid declared up front; the falsifier binds the predicted SHAPE of the boundary
    — direction/monotonicity — not each cell); exploratory sweeps are sanctioned when
    labeled exploratory, then the seen boundary is registered and confirmed on fresh seeds
    (the Exp 112/128 diagnostic precedent — never relabel exploration as confirmation).
    Flat numbered experiments/expNNN_cont_*.py continuing the global sequence; falsifiers
    predeclared in the docstring BEFORE running; ≥ 8 seeds (sweeps: seeds-per-cell as
    predeclared); outputs committed with the EXPERIMENTS.md
    entry; a negative rung is a result, not a blocker — log it and reassess. Twin baseline:
    every rung also runs the tabular twin (Dirichlet/categorical, same 6 concepts) on the
    IDENTICAL observation stream; primary comparison metric = held-out predictive NLL (the
    only metric defined for both); localization error / posterior entropy / collapse index
    logged per-agent as diagnostics; JSON rows {exp, rung, agent, seed, step, metric,
    value, params}. Declared modeling choices (state them in rung 1's predeclaration):
    emission likelihood is the UNNORMALIZED Gaussian footprint (the conjugacy-buying
    approximation) while the data generator uses the normalized mixture — the mismatch is
    part of what's measured; NIW learning uses posterior moment-matching, not point states
    (or declare the alternative once and keep it fixed). Rung 4 confound control: scale
    footprints so pairwise Mahalanobis separation is constant across d, and report it.
    Rung 5 cross-link (predeclare): does the Exp 88 window law (forgetting rate between
    accuracy floor and tempo ceiling) hold for NIW precision decay? — predeclare the band;
    if it holds, that is a substrate-independent law. Scope guards: no creature/spine
    contact; M4a stays halted at 1c; until rung 6 (the amortized control) runs, every
    claim is bounded to continuous-vs-tabular — no closed-form-vs-amortized conclusions.

- [from human, 2026-06-10] META-METHODOLOGY supplied: a standing reference of design-time
  criteria (mechanism vs outcome, specific null, confound list, strongest-comparison,
  degrees of freedom, bridge question), run-time discipline (shape-not-point predictions,
  sweep legitimacy, two-phase exploration, seed discipline, log-everything), result
  evaluation (so-what, confound audit, boundary conditions, effect size, surprise audit,
  falsifier honesty), generalizability tiers (analytic > functional-form > parameter-level
  > failure-mode > benchmark transfer), recognizing-the-profound heuristics, and
  bridge-to-scale rules. Goal: findings that generalize beyond the repo and can speak to
  live debates (AXIOM vs Dreamer, FEP falsifiability, scaling).
  → captured verbatim as `loop/METHODOLOGY.md` (ADVISORY layer; VALIDATION stays binding;
    the two deltas — seed floor, commit-before-run vs the atomicity norm — are annotated
    there, PROTOCOL governs until the human amends it). Wired in: compose.py DISCIPLINE
    block, PROTOCOL step 2 (design-time pass) and step 5 (evaluation pass + the
    Implication line now names the generalizability tier claimed), loop/README.md table,
    RESUME.md map. Lands in PR #24 (the steering stack itself merged in PR #23).

(empty — drop ideas above this line's section freely)

## Consumed

(processed items get moved here, with their disposition)
