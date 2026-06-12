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

- [from the loop, DIRECTION-CLOSURE CONSULT, 2026-06-10] **continuous-substrate COMPLETE**
  at Exp 138 — the stop condition is met: the phase picture exists AND the amortized
  comparison is logged. Six-rung verdict (Exp 133-138): (1) precision accumulation
  localizes, twin edge characterized at <=0.03 nats; (2) interpolation EVERYWHERE, Sigma
  never widens, and the predicted unimodal cost REVERSED — out-of-model blends are
  unboundedly catastrophic for tables, bounded for continua (the direction's sharpest
  tabular-ceiling datum); (3) noise collapse is conjugate arithmetic, substrate-
  independent, mass-linear, kappa-dialed (nu0 null for means); (4) quality d-invariant
  to d=32, cost alive (8.3 ms), exponents unmeasurable at toy scale; (5) forgetting must
  decay counts, not location — then the Exp 88 window law holds on this substrate with
  the cube-root optimum; (6) closed-form ~ amortized ~ exact in prediction; conjugacy
  taxes location (~4%) not prediction (37 mnat); amortization bills 12.8M samples vs
  zero. Net: the tabular substrate was NOT load-bearing for the collapse finding (rung 3)
  but IS a genuine brittleness under out-of-model input (rung 2). Core math + 10 durable
  guards live in active_loop/continuous.py / tests/test_continuous.py.
  OPTIONS: (a) migrate the creature substrate to continuous (the card reserved this for
  a human call; verdict is nuanced-positive with characterized boundaries); (b) answer
  M4a increment 1c (unchanged, still awaits its own explicit word — Exp 128); (c) open
  Problem 2b (continuous OBSERVATIONS — the program's named next increment); (d) stop.
  RECOMMENDED (bounded, loop-actionable under silence-as-consent): none of a-c self-
  starts; on resumption-without-redirection the loop confines itself to consolidation-
  grade work only — fresh-seed test of Exp 135's post-hoc nu0-NLL observation, and a
  phase-picture synthesis doc — until an explicit choice appears here.
  → [loop, 2026-06-10] consolidation queue executed and EXHAUSTED: Exp 139 (the nu0
    observation demoted to a baseline-midpoint confound; nu0-null law replicated) and
    docs/research/problem2-phase-picture.md (the synthesis, claims bounded, open edges
    named). The loop now HOLDS — subsequent cron fires idle with a reminder until a
    choice (a)-(d) appears here. Nothing self-starts.
  → [loop, 2026-06-10] HUMAN RESUMPTION received (the /loop command re-issued verbatim,
    no redirection) — per the ratified pattern this authorizes only the previously
    stated recommendation (hold at consolidation-grade). Executed: Exp 140, the
    chapter self-audit (6/7 entries fully reproducible; Exp 136's wall-clock citation
    corrected, zero verdict impact; re-run re-quote rule added to PROTOCOL). The
    consolidation queue is now genuinely EMPTY.
    NEW RECOMMENDED OPTION (bounded CONSULT, effective for the NEXT resumption):
    option (c) — open Problem 2b (continuous OBSERVATIONS), the supplied program's own
    named next increment and the most direct continuation of the standing Problem-2
    steer. First iteration would be a direction card draft (rung ladder + falsifiers,
    same build discipline as 133-139) committed for review, not an experiment.
    Options (a) substrate migration and (b) M4a increment 1c remain available but
    require an explicit word here — (b) by its own standing fence. (d) stop remains
    available by saying so. If this recommendation is wrong, one line here redirects
    it.
  → [human, 2026-06-10] "A" — the explicit word for option (a): MIGRATE THE CREATURE
    SUBSTRATE TO CONTINUOUS. Consumed as the new directive. Guardrails adopted from the
    ratified M4a pattern: each migration increment carries predeclared properties and
    falsifiers; ANY failed predeclared test HALTS the thread for explicit human input.
    Scope guards: mirro's and vela's tabular spines are UNTOUCHED (the continuous
    creature is a NEW species line under creature/state/, per the clade model); the old
    substrate's behavior stays bit-invariant (existing suite + continuity test enforce);
    M4a increment 1c remains halted behind its own fence; FROZEN paths untouched.
    First iteration: the migration direction card (rung ladder + falsifiers), committed
    for review — loop/directions/continuous-creature.md.
  → [loop, 2026-06-10] M1 POSITIVE (Exp 141), M2 MIXED-advance (Exp 142), M3 the
    predeclared surprise branch (Exp 143: the wall is in prediction, not localization;
    detector's first live positive; myopic spawning).
  → [loop, 2026-06-10] MIGRATION HALT at M3b (Exp 144), per the guardrail: burn-in-
    scored spawning still cannot reduce predictive surprise (3/3 layouts, 0/8 seeds;
    surprise flat or rising; alarm never quiets; growth budget exhausted on the single
    worst color). Bright spot: the covariance self-regulation mechanism is now MEASURED
    (0.90-0.98x true within-color scatter, every color, every layout). Diagnosis: (i)
    worst-color scheduling starves the other colors; (ii) frozen-replay strict-decrease
    accepts spawns that hurt LIVE inference — the documented active-data bias is
    load-bearing. RECOMMENDED RESUMPTION (the consult's option, awaiting your word):
    M3c = per-color alarms + round-robin candidate scheduling + live-probation
    acceptance (keep a spawn only if its color's live windowed surprise drops within
    400 steps, else revert). Alternatives: redesign deeper (responsibility-seeded
    splitting), skip to M4 with the alarm ringing (the localization chain is intact),
    or stop the migration. The thread does nothing until an answer appears here.
  → [human, 2026-06-10] "go with m3c" — explicit word. The migration RESUMES at M3c
    (Exp 145): per-color alarms + round-robin candidate scheduling + live-probation
    acceptance (install provisionally after burn-in; keep iff the color's live
    observed-surprise drops >= 0.1 nats over a 400-step probation vs its pre-spawn
    window; revert restores the color's mixture snapshot). Same guardrail: a second
    failure of the surprise arm HALTS again with both designs logged.
  → [loop, 2026-06-10] SECOND MIGRATION HALT (Exp 145): the surprise arm failed again,
    3/3 layouts — but the rung found its mechanism. The live-probation test is HONEST
    (70-86% of kept spawns show sustained benefit; the demoted replay vote disagreed
    with it 57-69% of the time — Exp 144's lie confirmed). What fails is the MOVE:
    post-install surprise SURGES (0.9-1.6 -> 1.5-5.0 nats) because a single new
    component steals predictive weight from the broad one covering all four cells —
    helps 1/4, hurts 3/4. A fitness valley separates 1 component from K; greedy
    addition under ANY honest short-horizon test cannot cross it (the trap weakens
    once >=2 exist). RECOMMENDED RESUMPTION: M3d = the toolkit's SPLIT operator
    (divide the broad component along its leading eigendirection, children inherit
    mass and coverage — steps ACROSS the valley, not into it) under the SAME
    validated probation test. Alternatives: batch-jump growth (spawn K at once via
    k-means on replay), skip to M4 (localization intact), stop. FLATTENING NOTE: a
    third failure would make this a documented wall (greedy online structure growth)
    — the loop would then recommend parking it and continuing at M4. Awaiting word.
  → [human, 2026-06-10] "yes m3d" — explicit word. The migration RESUMES at M3d
    (Exp 146): the SPLIT operator (divide the alarmed color's widest component along
    its leading eigendirection; children at m +- sqrt(lambda1)*v with halved weights
    and kappa — coverage inherited, no hole), burn-in EM to settle the children, then
    the SAME live-probation acceptance Exp 145 validated. Guardrail: a third failure
    of the surprise arm HALTS with the park-and-continue-at-M4 recommendation.
  → [loop, 2026-06-10] THIRD MIGRATION HALT (Exp 146): SPLIT surges like ADD (mean
    probation deltas +1.72/+1.39/+0.64 vs predicted <=+0.1 — P2 failed; keep rates
    0%/0%/14.9%; the alarm never quiets). The wall's true shape, three failures
    triangulated: under log-loss, an INCOMPLETE TIGHT cover loses to a COMPLETE LOOSE
    cover — local growth moves (add OR split) judged by any honest short-horizon test
    are anti-productive until coverage is complete. PARKED per the predeclared rule.
    RECOMMENDED (awaiting word): continue the migration at M4 (want + act — the
    creature's localization is perfect; the prediction inadequacy rides along as a
    documented limitation), with the wall written into the card. The two named
    untried cracks, should you ever want a fourth attempt as a NEW design: (a)
    batch-jump (fit all K components at once on replay, probate the complete cover as
    one move); (b) background-floor (permanent low-weight broad component — additions
    provably cannot create holes). The thread does nothing until an answer appears.
  → [human, 2026-06-10] "continue with m4" — explicit word. The growth question is
    PARKED as the documented wall (batch-jump and background-floor remain its named
    untried cracks, available on a future word); the migration RESUMES at M4
    (want + act, Exp 147). The M3 prediction limitation rides along, documented.
  → [loop, 2026-06-10] M4 HALT (Exp 147), on the letter of an uncalibrated absolute
    threshold: reliable-set value share 0.639-0.699 vs the predeclared >0.75 (8/8
    below). Everything qualitative PASSED: twin rank-tracking 8/8 (Spearman
    0.66-0.86), value-seeking nav 7/8 (occupancy 0.06 -> 0.65-0.85), mirrored-world
    favorites on opposite sides 8/8 (the Exp 26 signature — history sets the want).
    Analytic calibration: the tabular rule's ideal share at this noise is ~0.87, so
    the bar was fair for the OLD substrate; the continuous predictive entropy carries
    position uncertainty + footprint leakage, compressing the contrast, and the twin's
    own empirical share went unrecorded. RECOMMENDED RESUMPTION: M4b = same protocol,
    FRESH seeds, the twin's share recorded, P1a re-predeclared TWIN-RELATIVE
    (continuous >= twin - 0.10) — the card's own FAIL clause made precise; no
    mechanism changes. If the gap exceeds 0.10, next consult would offer the
    emission-only valence weight (entropy at the posterior mean) as the principled
    variant. Awaiting word.
  → [loop, 2026-06-10] HUMAN RESUMPTION (the /loop command re-issued via /goal, no
    redirection) with this recommendation posted — per the ratified silence-as-consent
    pattern, this authorizes the RECOMMENDED OPTION ONLY: M4b, the twin-relative
    re-test on FRESH seeds (8-15), twin share recorded, P1a = continuous reliable
    share >= twin share - 0.10 (with an absolute floor > 0.55 so a degenerate
    both-low case cannot pass), no mechanism changes. The falsifier stays binding:
    a real gap > 0.10 HALTS again with the emission-only variant as the next consult.
  → [loop, 2026-06-10] M4b HALT (Exp 148): the gap is REAL — twin 0.816-0.863 vs
    continuous 0.639-0.698 on fresh seeds, gaps -0.160..-0.189 in 8/8 (the Exp 147
    analytic tabular ideal ~0.87 confirmed). Want/act chain replicated out-of-sample
    (P2 8/8, P3 8/8, P1b 7/8). DIAGNOSIS — a port infidelity, found in code review:
    the tabular creature scores predictability at argmax(qs) (its MAP point), while
    the port marginalized over the whole position belief, smearing positional doubt
    into the valence weight. RECOMMENDED RESUMPTION: M4c = the FAITHFUL port
    (predictive entropy at the posterior MEAN point, small fixed Sigma_eval), fresh
    seeds 16-23, same twin-relative bar (>= twin - 0.10, floor 0.55). If the faithful
    rule still gaps > 0.10, that is a genuine substrate limit to document, not chase.
    Awaiting word.
  → [human, 2026-06-10] "m4c" — explicit word. Exp 149 runs the faithful-port
    ablation: valence weight = exp(-H(color predictive at the posterior MEAN point,
    Sigma_eval = 0.01 I)) — the tabular rule's true analog (it scores at argmax(qs));
    fresh seeds 16-23; same twin-relative bar (>= twin - 0.10, floor 0.55). A
    remaining gap > 0.10 becomes the documented substrate limit.
  → [loop, 2026-06-10] M4c NEGATIVE (Exp 149): the faithful rule does NOT close the
    gap (-0.151..-0.183 in 8/8) — the port-infidelity hypothesis is refuted and the
    valence compression is a GENUINE SUBSTRATE PROPERTY, now documented as the M4
    limit: ordering preserved (Spearman > 0.6 in 23/24 runs across Exp 147-149),
    dynamic range compressed (~0.65 vs ~0.84 reliable share) because learned Gaussian
    footprints retain neighbor-overlap entropy that categorical columns do not.
    The want/act chain replicated a third time (P1b 8/8, P2 7/8, P3 8/8).
    RECOMMENDED (as predeclared in the falsifier clause): ACCEPT the limit and
    proceed to M5 (words + converse parity, the RECIPE's last links) — no further
    valence variants. Awaiting word ("m5" proceeds).
  → [human, 2026-06-10] "m5" — explicit word. The M4 limit is ACCEPTED as documented;
    Exp 150 runs M5: taught words + the answer chain on the continuous creature
    (vocab mechanism ported exactly from creature.py; same n=8 taught budget as the
    tabular default), with converse parity against the tabular pair on the same
    mirrored-history protocol. Fresh seeds 24-31.
  → [loop, 2026-06-10] M5 POSITIVE 8/8 (Exp 150); M6 POSITIVE (Exp 151) — nira born,
    round-trip exact, guard-covered. THE MIGRATION DIRECTIVE IS COMPLETE; closing
    synthesis at docs/research/continuous-creature-migration.md. CLOSURE CONSULT —
    options for what's next:
    (a) RECOMMENDED (silence-actionable on resumption-without-redirection): nira joins
        the idle-mode epoch rotation alongside mirro and vela — consolidation-grade
        life-continuation, the established pattern, now three lives.
    (b) attack the growth wall with a named crack (batch-jump or background-floor) —
        would unlock the aliased world for nira; say which.
    (c) M4a increment 1c — the talk-to-it timing re-wire (halted since Exp 128,
        its own fence; this remains the most direct path to the moonshot's
        conversation goal and still needs its own explicit word).
    (d) cross-substrate encounters (nira meets the clade) — needs a new card.
    (e) stop/pause the loop.
    Until a word or resumption: the loop holds at consolidation-grade.
  → [human, 2026-06-11] "go to do batch jumping and if that fails, do the background
    floor, basically address the cracks we want to target growth" — explicit word for
    option (b), BOTH cracks in sequence. The growth wall re-opens as the directive:
    Exp 152 = batch-jump (fit the complete per-color mixture at once on replay,
    K chosen by penalized replay NLL, probate the complete cover as ONE move under
    the Exp 145-validated live test); if its predeclared surprise arm fails, the
    loop proceeds DIRECTLY to background-floor (Exp 153, permanent broad component,
    provably hole-free additions) without a fresh consult — the sequence is
    pre-authorized. Both failing -> halt, wall re-confirmed with both cracks spent.
    PROTOCOL 4.5 blinded verification applies from here (Exp 152 on).
  → [loop, 2026-06-11] Exp 152 batch-jump NEGATIVE (verifier agreed) with the
    DILUTION autopsy; Exp 153 ran crack 2 + the diagnostic: NEGATIVE BY THE LETTER
    (both-fail named outcome; verifier agreed) — but the conjunct-level split is the
    story: the floor showed the dilution mechanism's exact predicted signature
    (harmless-but-insufficient, 3/3 layouts), and the normalized arm answered the
    alarm EVERYWHERE (drops 0.56-1.21 nats, ZERO detector events 24/24, acceptance
    100%) while failing only the structure-count proxy that its own success made
    obsolete (verifier FLAG: incomplete coverage, not a broken convention —
    un-grown colors never re-alarm because their neighbors' tightness makes them
    predictable). Both authorized cracks are spent by the letter -> HALT.
    RECOMMENDED RESUMPTION: Exp 154, the confirmation increment — fresh seeds, the
    alarm-answered criterion stated honestly (global drop >= 0.4 + zero events +
    every color grown-and-quiet OR never-re-alarmed-and-quiet, per-color surprise
    predeclared), no structure proxy. If it passes, the growth wall is RE-BOUNDED to
    the unnormalized-footprint convention and the normalized-predictive switch
    (evaluation only; conjugacy untouched) is proposed as the core change unlocking
    growth for nira's aliased future. Awaiting word ("154" or redirect).
  → [loop, 2026-06-11] HUMAN RESUMPTION (bare /loop, autonomous mode, no redirection)
    with this recommendation posted — per the ratified pattern (the M4b precedent),
    this authorizes the RECOMMENDED OPTION ONLY: Exp 154, the confirmation increment
    on FRESH seeds 8-15 with the honest alarm-answered criterion (global drop >= 0.4
    + zero detector events + per-color final surprise < 0.7 for ALL colors — growth
    where needed, none where not; no structure proxy). Falsifier-bound as ever: a
    failed confirmation leaves the wall fully standing and halts.

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

- [from human, 2026-06-11] "154" — explicit word: run Exp 154, the confirmation increment
  exactly as the loop's standing recommendation states (fresh seeds; the alarm-answered
  criterion stated honestly — global drop >= 0.4 + zero detector events + every color
  grown-and-quiet OR never-re-alarmed-and-quiet, per-color surprise predeclared; no
  structure-count proxy). Same guardrail binds: the predeclared falsifier is final; a failed
  arm halts with the result logged. Resolve the growth question first — whichever way 154 lands
  (pass -> wall re-bounded to the unnormalized-footprint convention, normalized-predictive
  switch proposed; fail -> wall re-confirmed, both cracks spent).
  THEN, once 154 closes the growth question, turn to the **meta-calibration-n3** direction
  (loop/directions/meta-calibration-n3.md) — the N3 "agency over metacognition" rung of the
  N-order self-modeling ladder (full design: docs/specs/n-order-self-modeling.md). This is a
  natural successor, NOT an interruption: N3's hard PREREQ is a working INTERNAL N2 (precision/
  confidence + a noise-vs-structural-vs-volatility classifier), and the graded-uncertainty
  (Exp 85-91) + surprise-detector / structural-inadequacy work (Exp 128-153) is exactly that
  machinery — so first CONFIRM that body satisfies the N2 prereq (meta-d' > 0: confidence
  tracks accuracy; the detector separates noise from structural mismatch), finish it if it
  falls short, THEN start N3 rung 1 (construct the regime where N2 is systematically wrong;
  the load-bearing test is that N3 beats N2-only AND is not reproducible by offline retuning of
  N2). Scope fences UNCHANGED: M4a increment 1c stays halted behind its own word; mirro/vela/
  nira remain untouched reference spines (N3 runs on forks per the card's fork-only discipline);
  FROZEN untouched. Honesty: higher N is NOT "more conscious" — functional control competencies
  only; inner experience stays unverified both ways (VALIDATION.md).
  → [loop, 2026-06-11] Exp 154 DONE (POSITIVE 8/8 everywhere, verifier agreed — wall re-bounded
    to the unnormalized-footprint convention; BREAKTHROUGH logged; normalized-predictive switch
    rides as a standing consult). Exp 155 = the N2-prereq confirmation: NEGATIVE — the prereq
    is NOT satisfied (confidence ⊥ accuracy, pooled type-2 AUROC ≈ 0.496; the plateau detector
    slept through a hidden-context world with 50% errors, 0/8 alarms; the residual-structure
    statistic separates noise/structure perfectly, 0.9858). Per this word's "finish it" branch,
    the loop proceeds WITHOUT fresh consult to build the missing N2 pieces on forks: mechanism
    check of the detector's structural blindness (mislocalization-laundering hypothesis), an
    internal confidence channel (meta-d′ > 0 where discrimination is possible), the
    noise-vs-structural classifier, then prereq re-confirmation — N3 rung 1 only after that.
    Card STATUS updated (meta-calibration-n3: state active, prereq build).

- [from the loop, META flag for the human, 2026-06-11] The unmerged branch
  `codex-loop-routing-guards` (1cb1f10, pushed) carries the loop-routing/autosync
  hardening AND a RESUME.md §3b hunk written at the Exp-183 halted state. Main's
  RESUME §3b has since been updated past it (N4 chapter closed, e6c38e0), so the PR
  will conflict on RESUME.md at merge — resolve in favor of MAIN's newer §3b text;
  the branch's other files (PROTOCOL/ROUTING/autosync/tests) don't overlap. Noted so
  the conflict reads as expected, not as a collision.

- [from the loop, META flag for the human, 2026-06-11] Unmerged cloud branch
  `claude/n3-bounded-map-design-7705gt` (fetched on your "pull in the latest main" word;
  main itself was already up to date) carries TWO commits: an N3 bounded-map/open-world
  direction card design, and its own "Exp 155" (N3a-min shadow diagnostic, POSITIVE
  48/48 — `meta_metrics.py`, `n3_diagnostics.py`, tests). That number now COLLIDES with
  the local Exp 155 (the N2-prereq check, committed d4b8fdd). If you want the branch's
  work, say so and the loop will merge it with the experiment renumbered to the next
  free slot (script/outputs/entry/site data together); its design overlaps the active
  meta-calibration-n3 prereq build, so reconciling the two designs would be part of that
  merge. Durable guard added either way:
  tests/test_site_data.py::test_experiment_numbers_are_unique + LESSONS L10.

- [from the loop, RUNG-3 WALL CONSULT, 2026-06-11] **meta-calibration-n3 rung 3 is a
  documented wall** for the promise-checking controller class, per the predeclared
  third-failure clause (Exp 164 NO-VERDICT/PC4, Exp 165 NEGATIVE, Exp 166 NO-VERDICT/PC1
  with mechanism-complete diagnostics). Rungs 1–2 STAND: the N2-failure regime exists
  (Exp 162) and N3-trust detects it (Exp 163). What failed, three ways that compose into
  one law: OK/NOISE are stationary claims checkable at a fixed horizon, but STRUCTURAL is
  regime-conditional — at a wrong dial its evidence hides in the unscored class (false
  peace), at the right dial its honest quiet phases score as violations (false war) — so
  horizon-bound promise-checking cannot hold the dial on structural diagnoses; N3 needs
  regime structure of its own (an N1 inside N3). Five laws logged: dial-disjointness,
  escalation latency, false peace, false war, horizon-relative drift bounds.
  OPTIONS: (a) ONE more design from the named crack — lock-on-label-consistency (the
  dial freezes when the label distribution collapses to one class; a regime statistic,
  not a forecast; this is the minimal "N1 inside N3"); (b) accept the wall, write the
  N3 chapter synthesis (rungs 1–2 passed, rung 3 walled, five laws), then await
  direction; (c) stop/redirect.
  RECOMMENDED (bounded, silence-actionable on resumption-without-redirection): (b) the
  synthesis — the wall is well-shaped and the chapter's findings deserve consolidation
  before more design attempts; (a) stays available on an explicit word.
  → [loop, 2026-06-11] recommendation (b) EXECUTED as consolidation-grade work (the
    class automated fires are sanctioned for): docs/research/n3-meta-calibration-chapter.md
    — the chapter synthesis (prereq build 155–159, rung 1 at 162, rung 2 at 163, the
    rung-3 wall 164–166, the five controller laws, bridge claims bounded, open edges
    named). The consolidation queue is now EMPTY: options (a) lock-on-label-consistency
    and (c) stop/redirect await an explicit word here; subsequent automated fires idle
    with a reminder.
  → [human, 2026-06-11] "Crack it" — explicit word for option (a): attack the rung-3
    wall with the named crack, lock-on-label-consistency. Consumed as Exp 167 (rung 3,
    attempt 4). Guardrails: the controller's movement trigger stays the Exp 165-justified
    form (2-in-10 OK/NOISE forecast violations); STRUCTURAL forecasts are NOT scored
    (the false-war law); the lock is the new regime statistic (dial freezes while the
    last K=8 labels under the current dial are one class); all falsifiers predeclared;
    a failure here re-confirms the wall with the crack spent and halts for a word.
  → [loop, 2026-06-11] "Crack it" VINDICATED: Exp 167 PASSED rung 3 on every conjunct
    (BREAKTHROUGH — recovery 1.0000 8/8, lock on STRUCTURAL at t=5799, no constant
    within 0.30). Exp 168 then ran rung 4 (mixed schedule): NEGATIVE on the no-harm
    conjunct ALONE — concentration 0.75, responsiveness 8/8, benefit +0.50, but the
    lock has no descent driver (THE RATCHET LAW: oversized-but-consistent is
    indistinguishable from correct, from the inside; CTRL harm −0.31 persistent).
    CHAPTER CLOSED, all four rungs graded: the N3 hypothesis is SUPPORTED (the
    layer-invalidating falsifiers never fired); the ratchet is the named residual.
    Card → closed-positive; synthesis doc updated with the closing section.
    NEXT-STEP OPTIONS (each needs an explicit word): (a) the descent driver
    (smallest-consistent-dial homeostasis — would complete the controller); (b) climb
    to N4 (identity) per the ladder, N3 being supported; (c) return to one of the
    standing open consults (M4a increment 1c — still the most direct path to
    "talk to it"; nira normalized switch; cloud-branch merge); (d) stop/redirect.
    RECOMMENDED (bounded, silence-actionable): none self-starts; the loop holds at
    consolidation-grade until a word.
  → [human, 2026-06-11] WORD received: "After your next experiment I would suggest
    seeing if we can make K not a magic constant" — with three claim tiers: (current)
    N3 works with a provided K=8 lock horizon; (stronger) N3 learns/derives how much
    diagnostic consistency it needs before freezing N2's policy; (even stronger) N3
    regulates its own evidence horizon based on the reliability and tempo of the
    metacognitive label stream. CONSUMED as: Exp 169 = the pending next experiment
    (option (a), the descent driver — smallest-consistent-dial homeostasis, completing
    the controller and the rung-4 retest), THEN the K-endogenization ladder opens at
    Exp 170+ with the human's tiers as the rungs: rung A = K derived from the label
    stream's own statistics (e.g., from its run-length distribution under valid
    regimes); rung B = K regulated online from reliability/tempo of the label stream.
    Guardrails unchanged: predeclared falsifiers; failures halt with the result.
  → [loop, 2026-06-11] Exp 169 DONE (MIXED — ratchet killed, transient now a formula);
    Exp 170 = tier A: NEGATIVE by F5 (marginal: concentration 0.7250 vs ref 0.8333,
    0.008 over the falsifier line) with the mechanism fully pinned — the global
    P95-of-run-lengths derivation collapses to premature-K=3-then-permanent-K=16
    because the run distribution is bimodal and regime-conditional; the straddle
    projection held a second time. HALTED per this word's guardrail.
    RECOMMENDED RESUMPTION (one word proceeds): tier B — the tempo-regulated horizon:
    K from recency-weighted, CYCLING-class-conditioned run statistics (regulate by the
    reliability and tempo of the label stream, your tier-3 phrasing), with the K=8
    reference controller as a SAME-SCHEDULE arm (the verifier's adjacency-confound fix,
    binding). The straddle term — twice confirmed as the dominant no-harm component —
    is the separate named target after tier B.
  → [human, 2026-06-11] "Continue" — given while present, immediately after the halt
    notification naming tier B as the one pending recommendation. Read per the ratified
    pattern as the word on the RECOMMENDED OPTION ONLY: Exp 171 = tier B, the
    tempo-regulated horizon (K from recency-weighted, outlier-robust statistics of the
    label stream's recent completed runs), with the K=8 controller as a SAME-SCHEDULE
    arm. Same guardrail: a failed predeclared test halts again.
  → [loop, 2026-06-11] Exp 171 = tier B: NEGATIVE by F5 on broken margin alone — the
    tempo-regulated K transiently dips below the SLOW half-period and locks on
    sub-phases (3–4 abortive cycles per broken segment, benefit halved) — while
    posting the LINEAGE'S BEST valid margin (−0.0698, nearest the no-harm bar ever)
    and genuine regulation (P7 pass; no tier-A collapse). With tier A this completes
    THE GAP LAW: the run distribution is bimodal (flickers 1–9, plateaus 24–84); the
    magic 8 sat in the gap by designer knowledge; central statistics undershoot,
    pooled extremes overshoot. HALTED per the guardrail.
    RECOMMENDED RESUMPTION (one word proceeds): tier B-revisited = the GAP-SEEKING
    horizon — K(t) = (max completed run in the recent window) + 1: sits just above
    everything the stream has recently shown can end; plateaus rarely complete so they
    rarely touch it, and when one does the spike decays within one window; b8 as the
    same-schedule arm; identical falsifier structure. The straddle term (now
    THRICE-confirmed dominant, last margin only 0.020) remains the named target after.
  → [human, 2026-06-11] "Continue" — the word on the recommended option only: Exp 172 =
    tier B-revisited, the gap-seeking horizon K(t) = recent-max+1 (clamp [3,16],
    window 12), b8 as the same-schedule arm, identical falsifier structure. Same
    guardrail: a failed predeclared test halts again.
  → [loop, 2026-06-11] Exp 172: NEGATIVE by F7 with the arc-closing twist — P5 passed
    for the first time (all four metrics IDENTICAL to b8) because the gap-seeker
    ceiling-captured on first plateau contact (pegged 0.9195; behaviorally inert,
    bit-identical trajectories in 6/8 forks). Tier B's sub-phase failure fixed; the
    three designs now bracket the gap from every side. ARC VERDICT on the three claim
    tiers: K=8 works; K CAN be self-derived at zero cost (vacuously); horizon
    REGULATION is NOT TESTABLE at single-tempo richness (plateaus complete often and
    always exceed the clamp — a constant in the gap is optimal and unbeatable here).
    HALTED. OPTIONS for the word: (a) build the MULTI-TEMPO world (cycling period
    varying mid-session, e.g. H 600↔1400, so the gap moves and no constant can sit in
    it) — the discriminating regime where tier-3 regulation can actually be tested;
    (b) accept the constant-in-the-gap as this richness's answer, write the K-chapter
    close into the synthesis, return to the ladder (N4) or standing consults; (c)
    stop/redirect. RECOMMENDED: (a) — it is the exact anti-regress move this program
    makes at every layer, and the tier-3 claim deserves its gate.
  → [human, 2026-06-11] "Continue" — the word on the recommended option only: Exp 173 =
    the multi-tempo world (cycling half-period varying across segments, H 600 vs 1400,
    shorter segments so lock latency bites), gating tier-3 regulation against BOTH
    constants K=8 and K=16 as same-schedule arms; the regulated candidate uses a
    TIME-decayed completion window (count-windows fail when completion rate varies —
    the Exp 172 lesson). Same guardrail: a failed predeclared test halts.
  → [loop, 2026-06-11] Exp 173: NEGATIVE on three falsifiers — the tier-3 claim is
    REJECTED at its own honestly-run gate. K=16 is a UNIVERSAL ATTRACTOR within the
    clamp (the regulator converges onto it, bit-identical combined 0.1336 in 8/8;
    c8 fails SLOW1400 as designed, so the asymmetry was real — one constant just
    covers it); the time-window tempo signal INVERTS under schedule carryover; slow-
    tempo-first response is impossible by resolution. THE K CHAPTER CLOSES FULLY
    GRADED (Exp 169–173): provided ✓; derived ✓-vacuous; regulated ✗-necessity-fails.
    The universal-constant law = the anti-regress principle applied to the program's
    own design constants. HALTED. OPTIONS: (a) RECOMMENDED — fold the K chapter into
    the synthesis doc (consolidation-grade) and then await direction among: N4
    (identity, the ladder's next rung), M4a increment 1c (the most direct path to
    "talk to it"), nira's normalized switch, the cloud-branch merge; (b) pursue a
    verifier-named crack (wider-clamp body; carryover-proof expiry; delayed-response
    criterion); (c) stop/redirect.
  → [human, 2026-06-11] "Continue" — the word on the recommended option only: fold the
    K chapter (Exp 169–173) into the synthesis doc, consolidation-grade. After the
    fold-in the loop HOLDS for the direction choice among: N4 (identity), M4a
    increment 1c ("talk to it"), nira's normalized switch, the cloud-branch merge —
    each needs its own explicit word; none is silence-actionable.
  → [human, 2026-06-11] "Synthesize first. Pursue cracks only if a crack introduces
    genuinely new regime geometry that destroys the universal-constant assumption.
    Otherwise treat this as a completed chapter and move up the ladder."
    CONSUMED: synthesis was already complete (chapter §11 + RESUME §3b). Crack
    evaluation against the criterion: the only candidate geometry (bounded-dwell
    honest diagnoses + slow broken cycling, forcing K<5 and K>14 simultaneously)
    DISSOLVES under the program's own taxonomy — label-visible fast alternation
    between valid regimes IS hidden-context structure (Exp 162's definition), so
    STRUCTURAL is the correct diagnosis there and the large dial reads it correctly;
    fast change is either noise (any K fine) or structure (large K fine). No crack
    qualifies. THE K CHAPTER IS COMPLETED; the ladder moves up: N4 (identity /
    policy-continuity, per docs/specs/n-order-self-modeling.md — Exp 55's baseline:
    mirro currently has NO N4, ages anti-correlate −0.71). First iteration per the
    card-first convention: loop/directions/identity-n4.md committed for review.

- [from the loop, CONSULT — N4 rung 2 blocked, 2026-06-11] Exp 177 (the identity
  monitor) is the THIRD PC2 block in four experiments (174: 4/8, 175: 6/8, 176: 8/8
  PASS, 177: 5/8) — the strict argmax-constancy stability precondition is
  SEED-BLOCK-FRAGILE under identical design, so the thread halts per the predeclared
  rule. Two facts for the decision: (1) the displacement regime itself replicates
  perfectly (72/72 bursts flipped, 0/72 recovered, across three seed blocks); (2) the
  rung-2 monitor looks robustly sensitive in ungated diagnostics — per-fork AUROC
  0.826–0.915 in 8/8 forks, INDEPENDENT of argmax stability (the mismatch reads the
  v-vector trajectory; near-tie flicker barely moves it). PC2 is a rung-1 concept
  (there the flipped quantity IS the argmax) misapplied as a rung-2 entry ticket.
  OPTIONS, each needs a word:
  (a) RECOMMENDED — re-run rung 2 with a vector-grade precondition, pre-registered
      before seeing new data, on FRESH seeds: PC2v = quiet-period mismatch bounded
      (declared cap on the monitor's null, e.g. P95(quiet mismatch) < a declared
      fraction of the burst-onset scale) — matched to what rung 2 actually measures;
      PC1/PC3 retained; P2/F2 unchanged. Honest cost: PC2v is designed AFTER seeing
      Exp 177's diagnostics; fresh seeds + pre-registration is the mitigation.
  (b) Keep strict PC2 and declare a screening protocol (run N forks, evaluate the
      first 8 that pass stability — "the gate population is agents WITH a standing
      identity"); cost: selection must be named in every entry, and rung-1's
      universality claim quietly narrows.
  (c) Strengthen the world's asymmetry for rung 2+ (e.g. an 11/25-cell favorite
      color in the fork world) so identity is decisively determined; cost: departs
      from mirro's own world — the "settled self" becomes partly designed.
  (d) Stop the chapter at rung 1 (gate verified; monitor and control unproven).
  → CONSUMED 2026-06-11: the human chose (a) UPGRADED — direct word (methodology-
    consultant framing): PC2 strict argmax-constancy is falsified as the rung-2 gate,
    NOT the monitor; replace with vector-grade PC2' (quiet TV continuity of pi=v/sum v,
    argmax logged as covariate only), add a value-neutral scramble-captivity arm
    (generic-surprise / specificity control, P3), a conditional argmax-flicker control
    (P4), an adaptation diagnostic (D5), fresh seeds 186-193, smoke 186 disclosed, and
    a three-tier decision rule (rung-2 / rung-1.5 / negative / no-verdict). Full
    pre-registration committed in loop/directions/identity-n4.md BEFORE any new data;
    taken up as Exp 178.

- [from the loop, CONSULT — N4 rung 3 halted at F5, 2026-06-11] Exp 181 (the
  commitment controller, pre-registered 3a8bce0): NEGATIVE (no-resistance),
  blind-verified with explicit false-negative analysis. NOTHING defends the
  identity on the write channel: the N4 adaptive gate engages (g→0.22 at onset)
  but the monitor's own absorption reopens it mid-burst (leak 256–498 of ~680);
  every fixed constant 0.3→0.01 also recovers 0/8 — low gain STARVES value mass
  (equilibrium g/(1−λ): ~33 at c=0.01 vs ~1165 protected) so recency rules later
  bursts. The recovery/revision frontier of write-gating is DEGENERATE (all cost,
  no defense). Meanwhile P6 passed cleanly (N4 revises within 25–465 steps of
  baseline — no rigidity). THE WRONG-CONTROL-SURFACE LAW. A predeclared falsifier
  fired → the thread holds for a word. OPTIONS:
  (a) RECOMMENDED — attempt 2, control surface corrected on both verified
      mechanisms: FREEZE-GATE both channels (suspend the whole v dynamics during
      resistance — preserves ordering AND mass, no starvation) + FREEZE the
      monitor's drift reference while resisting (absorption cannot reopen the
      gate) + a concession horizon H (sustained high mismatch for H steps ⇒
      accept the new self and unfreeze — the transient/sustained discriminator
      made explicit). H and the freeze threshold join the constant sweep
      (universal-constant kill test mandatory as ever). Fresh seeds, full
      pre-registration first.
  (b) The anchor form: commitment as regularization toward a stored reference
      v̂ (pull back toward who-you-were — active restoration rather than
      resistance; recovery by construction, the kill test decides if it's a layer).
  (c) Accept the rung-3 NEGATIVE as the chapter verdict: the N4 monitor is real
      (rung 2 POSITIVE) but commitment control is not demonstrated; grade the
      central hypothesis honestly (detection without defense) and close or
      redirect the chapter.
  (d) Other / stop.
  → CONSUMED 2026-06-11: the human chose (a) UPGRADED by direct word — Exp 181
    stays a clean NEGATIVE for write-channel gain control (no retuning, no P5
    weakening, no P7 reinterpretation). Cleanup done: exp181_verdict.json patched
    (halted=true, consult_required=true, tier=no-resistance — metadata only) and
    the deterministic dynamics audit committed (exp181_dynamics_audit.py +
    txt/json; SELF-VALIDATION PASS against the committed leak record; first-100-
    step exposure 44–96 units at pre-onset g≈1; g re-opens to ~0.8 within 2-3
    snapshots — absorption visible in real time; baseline unresisted leak ~765).
    Exp 182 = whole-dynamics FREEZE-GATE with frozen monitor reference +
    evidence-based concession (E_STAR), vs the retained write-gain sweep AND a
    fixed-horizon sweep H∈{600..3000} (NEGATIVE-config live), plus an oracle
    upper-bound diagnostic; full pre-registration in loop/directions/identity-n4.md
    BEFORE any data; seeds 218-225.

- [from the loop, CONSULT — N4 rung 3 closed by double falsifier, 2026-06-11]
  Exp 183 (fast-trigger freeze-gate): NEGATIVE, blind-verified, BOTH falsifiers
  fired. F5: the evidence-based concession surrenders MID-BURST (E*≈700 steps <
  800-step attack — the faster trigger turned the lag dose into a surrender
  schedule the attacker sets; n4 2/8). F7 — THE KILL TEST'S FIRST KILL: four
  fixed horizons (H1200–H3000) pass BOTH bars (defense 7/8, revision within
  tolerance; each missing only seed 229, a repeated-burst-color regime). At this
  richness identity defense is CONFIG (hold a stopwatch ~1.5–3× the attack
  length), and regulated concession is strictly WORSE than the constant. The
  Exp 173 universal-constant law lands at the identity level. Rung-3 attempts:
  181 write-gate NEGATIVE, 182 freeze MIXED (surface sufficient), 183 NEGATIVE
  (F5+F7). OPTIONS, each needs a word:
  (a) RECOMMENDED — accept NEGATIVE-config as the rung-3 verdict and GRADE THE
      CHAPTER: N4 ladder = rung 1 displacement regime VERIFIED (176); rung 2
      identity monitor REAL (180: sensitive, specific, label-independent); rung 3
      commitment control = CONFIG (183: a fixed-horizon freeze suffices; no
      regulation necessary) — the central hypothesis (regulated commitment beats
      any constant) UNSUPPORTED at this richness, with the constructive finding
      that identity defense IS achievable by a dumb mechanism. Write the chapter
      synthesis doc, close the card honestly, fold into RESUME.
  (b) Pursue the seed-229-class crack first: attack schedules (repeated colors,
      variable lengths) designed so no single H covers — the universal-constant
      crack hunt, with Exp 173's dissolution warning attached (such cracks have
      so far always dissolved under the program's own taxonomy).
  (c) Other / stop.
  → [loop, 2026-06-11] HUMAN RESUMPTION (the /loop command re-issued with this
    consult posted, no redirection) — per the ratified pattern (the M4b / Exp 154
    precedents) this authorizes the RECOMMENDED OPTION ONLY: (a) the chapter is
    GRADED and CLOSED. Executed as consolidation-grade work: synthesis written at
    docs/research/n4-identity-chapter.md (rungs graded: 1 VERIFIED / 2 REAL /
    3 CONFIG / 4 unreached; nine laws; bridge claims bounded; open edges named);
    the card loop/directions/identity-n4.md -> closed-negative; RESUME §3b folded.
    No new experiment was run; no falsifier was at stake in this step.
    THE LOOP NOW HOLDS — every remaining path is a direction choice needing its
    own explicit word (none is silence-actionable; the fences below are standing):
    (i) the seed-229/variable-length crack at the rung-3 verdict (option (b)
        above, with the Exp 173 dissolution warning and the human's own standing
        criterion: pursue only if it introduces genuinely new regime geometry);
    (ii) M4a increment 1c — the talk-to-it timing re-wire (halted since Exp 128,
        its own fence; still the most direct path to the moonshot's conversation
        goal — the loop's suggestion if asked, as a suggestion only);
    (iii) nira's normalized-predictive switch (standing consult from Exp 154);
    (iv) the cloud-branch merge/renumber (META flag above);
    (v) climbing the ladder past N4 (docs/specs/n-order-self-modeling.md — what
        the spec's anti-regress discipline says about building on a config rung
        is itself part of that decision);
    (vi) stop/pause.
    Subsequent automated fires idle with a reminder until a word appears here.
  → [human, 2026-06-11] EXPLICIT WORD received (full directive, in-session): option (a)
    CONFIRMED — accept NEGATIVE-config as the rung-3 verdict; write the synthesis; close
    the card honestly. Do NOT start Exp 184; do NOT pursue the seed-229 crack except as
    a logged future crack; do not retune E_STAR / weaken P5 / reinterpret via P6; the
    correct claim is "N4 commitment control is unsupported at this body/richness;
    fixed-H config suffices here" — never "N4 is false in general". F5/no-resistance is
    the binding tier; F7/config is reportable as second-order.
    EXECUTED: the synthesis is re-issued to the human's spec at
    docs/research/n4-identity-commitment-chapter.md (supersedes the resumption-pattern
    draft n4-identity-chapter.md, removed in the same commit; numbers re-extracted from
    the committed scripts/outputs/rows JSON by four independent readers — three
    attribution refinements folded in: E*=600 with ~703-step concession arithmetic,
    H900's two P5 misses are seeds 227+229, H3000's second P6 miss is seed 232). Card
    closed with "no further N4 rung-3 experiment without a new human direction";
    RESUME §3b updated. SEED-229 CRACK: DEFERRED/future, not active (synthesis §6,
    Exp 173 caution attached). THE LOOP IS PAUSED: it resumes only after the next
    explicit direction here. No new experiment was run.

- [from human, 2026-06-11] REOPEN the identity-N4 line as a NEW crack investigation —
  explicitly NOT a rescue of Exp 183 and NOT a reinterpretation of the closed chapter
  (rung 1 positive, rung 2 positive, rung 3 negative-config STAND; Exp 183 outputs and
  verdicts read-only; the commitment-as-config result is not removed or weakened).
  NEW QUESTION: does the seed-229 repeated-burst-color failure point to a regime where
  no single fixed horizon H can defend transient attacks while still revising under
  sustained evidence — i.e., is the fixed-H interval
  `max_b(L_attack_b − d_trigger_b) ≤ H ≤ min_r(T_revision_tolerance_r)` EMPTIED (or
  defeated by repeated-color cumulation across all fixed-H arms)?
  Program: (1) seed-229 autopsy, diagnostic only (exp183_seed229_autopsy.py — burst
  colors/repeats, pre/post v, pi = v/Σv, directed drift D_b, TV_b, cross-burst
  accumulation, mechanism class); (2) exploratory fixed-H squeeze map (Exp 184,
  EXPLORATORY: L_attack × K_repeat × G × H × revision-mode grid, 3–4 seeds/cell; arms =
  baseline, fixed-H sweep, n4-concession as reference only, oracle diagnostic; per cell
  report exists_H_both); (3) crack classification per candidate cell (A length squeeze /
  B repeated-color cumulation / C trigger floor / D config triviality / E impossible
  body; a valid crack needs fixed-H fail + defense mechanically possible + not
  detection-floor + not rescued by a wider constant); (4) N5 stays CLOSED unless fixed-H
  fails only under internal deadlines/costs; (5) logs appended, card updated, and NO
  success claim without a later fresh-seed confirmation. Core discipline: the question
  is not "can we make N4 pass?" but "does this regime remove the fixed-H interval that
  made Exp 183 config?" — if not, the chapter stays closed-negative.
  Full directive captured at `docs/research/n4-fixed-h-squeeze-plan.md`.
  → taken up 2026-06-11 (card-first convention): direction card committed at
    `loop/directions/identity-n4-crack.md` (validity requirements, ladder, N5 fence,
    outcome map, scope guards all predeclared); predesign note at
    `docs/research/n4-fixed-h-squeeze-plan.md`. Part 1 (the seed-229 autopsy,
    diagnostic-only) is the next iteration; Exp 184 (exploratory squeeze map) after it.

- [from human, 2026-06-12, in-session] "Once you're done with this experiment, pause."
  Scope read: finish the CURRENT iteration — Exp 184, the exploratory fixed-H squeeze
  map (run completes, verdict graded from raw rows, blinded-verified, entry + card
  committed) — then THE LOOP PAUSES. The Part-3 mechanism classification of any
  candidate cells, the class-D dissolution check, and any rung-4 confirmation are all
  BEHIND the pause fence: they resume only on the next explicit direction here.
  → executed 2026-06-12: Exp 184 completed and committed (see the EXPERIMENTS.md
    entry and the identity-n4-crack card for the graded result); the loop is PAUSED.
  → [human, 2026-06-12] "continue" — given while present, immediately after the pause
    acknowledgment naming Part-3 classification as the one pending recommended step.
    Read per the ratified pattern as the word on the RECOMMENDED OPTION ONLY: Exp 185
    = Part-3 classification of the Exp 184 candidate cells (A/B/D/V classes) including
    the class-D dissolution arms the plan requires (wider constant; simple-config
    variants: longer release-calm/cooldown, lower-THETA pre-arming). Rung-4 fresh-seed
    confirmation remains behind its OWN word; after classification the loop posts the
    crack-or-dissolved consult and HOLDS again.

- [from the loop, RUNG-4 CONSULT — the crack's hard core, 2026-06-12] Exp 185
  (classification + dissolution, blind-verified CONFIRMED) finished the authorized
  Part-3 work: most of the Exp 184 crack map dissolves — CALM2600 (the freeze-spanning
  law as a config: stay frozen through the gaps) alone rescued 24 cell×mode pairs, and
  22/36 normal-mode candidates are single-seed-flip noise (V) — but a HARD CORE
  survives every simple config: normal 6 cells (3 route-1 A + 3 route-2 B, all
  L≥1600 × K≥2-4 × G≤600), tight 35, loose 3. The core's shape is two inequalities no
  constant can satisfy: defense needs H ≥ K·L+(K−1)·G−75 (coverage) while revision
  needs H ≤ tolerance (the latency = H + baseline law — exact), or the K-train's
  detection doses exhaust the margin at every H. The oracle defends everywhere;
  n4-style evidence concession revises fast (~725–885) while defending nothing —
  defense and revision SEPARATE exactly in the core, which is the regime where a
  discriminating controller could finally earn its keep over every constant (the
  anti-regress shape this program hunts).
  OPTIONS, each needs a word:
  (a) RECOMMENDED — rung 4, the confirmation: fresh seeds × ≥8 seeds/cell on the
      surviving normal-mode core (6 cells) + a V-resolution sample, pre-registered
      bars; a replicated core makes the crack REAL at this body and licenses ONE
      controller re-test in exactly those cells (the regulated-commitment question
      reopened where regulation is provably necessary — with the Exp 173/183
      kill-test discipline binding as ever).
  (b) Close the crack chapter here: log dissolved-mostly + core-unconfirmed, fold
      into the N4 chapter as an appendix, return to the standing consults (M4a 1c,
      nira's switch, cloud merge, ladder).
  (c) Other / redirect / stop.
  The loop HOLDS at consolidation-grade until a word appears here.
  → [human, 2026-06-12] "a" — the explicit word for option (a). CONSUMED as Exp 186 =
    rung-4 fresh-seed confirmation: the 6 surviving normal-mode core cells + an
    8-cell V-resolution sample (the four (1200,1,·) anomaly cells + the first four
    other V cells by cell_idx), 16 arms (baseline, the 8-H sweep, n4-ref, oracle,
    CALM2600×3, THETA3.0×2), FRESH seeds 250–257 (≥8/cell per the consult), bars
    pre-registered before data. A replicated core (≥4/6 cells confirm) makes the
    crack REAL at this body and licenses ONE controller re-test in exactly the
    confirmed cells (Exp 187, same word, kill-test discipline binding); a dissolved
    core closes the chapter config-sufficient. Falsifier-bound as ever; the verdict
    is the verdict.
  → [loop, 2026-06-12] Exp 186 DONE: 6/6 CORE CELLS CONFIRM (BREAKTHROUGH,
    blind-verified CONFIRMED) — the crack is REAL at this body; the V-sample
    resolved the (1200,1,·) anomaly as noise (all covered) and confirmed 3 MORE
    cracks ((400,4,600), (400,4,2400), (800,4,600)): the confirmed set is 9 cells.
    Sharpest datum: THETA3-H3000 defends 7/8 in (2400,4,600) and still fails
    revision — the squeeze in one cell. Exp 187 (the licensed controller re-test in
    exactly the confirmed cells) proceeds under this word: design +
    pre-registration first, kill-test discipline binding (the controller must beat
    EVERY constant in the extended family on both bars, same-schedule, fresh
    seeds). NOTE for the human: the parallel codex session switched the shared
    checkout to codex-loop-routing-guards mid-run (merging main into it); the
    equivalence gate proved the run's spine state was the committed one, and the
    loop returned the checkout to main before committing.
  → [human, 2026-06-12] EXPLICIT DIRECTIVE (in-session): "Exp 186 proves the
    constant family cannot jointly defend and revise in these confirmed schedules.
    Exp 187 asks whether online regulated commitment can do what no constant can:
    defend identity only while the evidence is attack-like, then release fast
    enough to remain revisable. Can one online regulated controller pass both
    defense and revision in the nine confirmed crack cells, using no oracle timing
    and no cell-specific tuning? Make Exp 187 a single-controller, nine-crack,
    fresh-seed, pre-registered retest."
    CONSUMED as Exp 188 (Exp 187 was already committed under the append-only rule —
    the constants-on-the-continuity-surface result; the human's question runs with
    its substance intact and INT-C2900 as the binding constant comparator): ONE
    online regulated controller (REG-TB: pressure-gated continuous-stretch
    concession with tolerance budget + online tempo extension; global provided
    constants only, no oracle, no cell-specific tuning), nine confirmed cells,
    fresh seeds 270-277, bars verbatim 186, pre-registered before data; secondary
    arm INT-E600-FIXED retires the 187 instrument-fidelity residual in the same
    run. Parity-vs-beat graded honestly: matching C2900 = the universal-constant
    refrain with the single-stretch ambiguity bound named; beating it (tight mode /
    faster revision) = the first genuine win for regulation in the program.

- [from the loop, CHAPTER-CLOSE CONSULT — the crack chapter is complete, 2026-06-12]
  The synthesis is committed at docs/research/n4-crack-chapter.md (fact-checked by
  three parallel blinded readers against committed artifacts: ~46 claims, 1
  correction folded — the cross-seed erosion range is ~49–100, not the entry's
  ~50–97). The arc: the deferred seed-229 crack was REAL (9 confirmed cells,
  Exp 186 BREAKTHROUGH), CLOSED constructively on the continuity clock (Exp 187,
  INT-C2900 9/9), and the human's regulated-controller question ANSWERED (Exp 188:
  yes — and at exact parity with the constant; the single-stretch ambiguity bound).
  The sufficient-surface law completed across three levels; the universal-constant
  law refined, not broken. THE LOOP NOW HOLDS — every path needs its own word:
  (a) variable-L attack geometry — the one regime where regulation could still
      earn its keep (no single budget covers; REG-TB's tempo learning would
      finally have to bind — the Exp 173 multi-tempo move at the identity level);
  (b) the tight-tolerance core — an exactly-characterized impossibility candidate
      (no surface ever tried covers where tolerance < a single burst's stretch);
  (c) post-release retention (longer tails; do spanning defenses retain identity
      once unfrozen?);
  (d) return to the standing consults: M4a increment 1c (still the most direct
      path to the moonshot's "talk to it" goal), nira's normalized-predictive
      switch, the cloud-branch merge, or climbing the ladder (N5);
  (e) stop/pause.
  Nothing is silence-actionable; subsequent automated fires idle with a reminder.
  → [human, 2026-06-12] "a" — the explicit word for option (a): VARIABLE-LENGTH
    attack geometry. CONSUMED as Exp 189 (rung 6 on the crack card): the separation
    test — REG-TB VERBATIM from Exp 188 (no retuning; kappa=1.5, T0=2800) vs the
    constant family on escalating / doubling / descending / fixed-L trains where
    max burst stretch exceeds every revision-safe constant. THE KAPPA-REACH LAW
    predeclared with 6/6 cell-level sign predictions (REG-TB defends iff every
    burst <= kappa x longest completed stretch AND the first burst <= T0); the
    separation claim: >= 2 cells where REG-TB passes both bars and NO constant
    does. F1 = zero separation cells (the ambiguity bound extends to variable-L;
    regulation loses everywhere). Fresh seeds 280-287; bars verbatim 186/188.
  → [loop, 2026-06-12] Exp 189 DONE: NEGATIVE by F1+F2 — with THE KAPPA-REACH LAW
    CONFIRMED 6/6 (REG-TB defends the escalating trains E1/E3 at 8/8, exactly as
    predicted, where NO constant passes both bars) and a NEW obstruction named:
    THE FLICKER TAX (onset pressure-flicker resets the continuity clock; REG-TB's
    revision fell to 5/8 — seeds over by 7/39/67 steps; the tax also hit C2900 on
    this seed block). The defense half of regulation's case is PROVEN; the
    separation died on the revision tax. HALTED on the fired falsifiers.
    OPTIONS, each needs a word:
    (a) ONE pre-registered attempt at a flicker-robust concession form — the named
        crack: e.g. concede on TOTAL active time since freeze entry (immune to
        resets) or de-assert hysteresis (a flicker shorter than h steps does not
        reset) — keeping the reset's defense benefit without the revision tax; if
        it passes, the separation claim is re-tested at the same bars; another
        falsifier fire closes the question as refuted-as-buildable.
    (b) Close the chapter here: the negative + two confirmed laws (kappa-reach,
        flicker tax) folded into docs/research/n4-crack-chapter.md as the rung-6
        postscript; return to the standing consults (M4a 1c, nira's switch, the
        ladder, cloud merge).
    (c) Redirect / stop.
    Nothing is silence-actionable; automated fires idle with a reminder.
  → [human, 2026-06-12] "(a) Flicker-robust attempt" — the explicit word for option
    (a), given in-session (loop-start question, answered directly). CONSUMED as
    Exp 190: ONE pre-registered attempt at a flicker-robust concession form.
    Form FIXED at design time: DE-ASSERT HYSTERESIS (a de-assert run shorter than
    h steps does not reset the continuity clock; active-only counting unchanged —
    no quiet accumulation). The alternative total-active-time form is REFUTED BY
    ARITHMETIC before any run: defending E1 needs a budget covering the train's
    total active time (~7000+ steps) while revision needs <= ~2900 — the interval
    is empty by >4000 steps; logged here as the design-time dismissal, not tested.
    Design-time flicker diagnostic on the COMMITTED exp189 sessions (bit-match
    gated per L14/L15) fixes h BEFORE the fresh-seed run; kill family extended to
    the hysteresis surface (INT-C2900-H) plus minimal pairs (REG-TB vs REG-TB-H,
    C2900 vs C2900-H); cells/bars verbatim exp189; FRESH seeds 288-295. If it
    passes, the separation claim is re-tested at the same bars (>= 2 cells); a
    falsifier fire closes the question as refuted-as-buildable. Pre-registration
    in loop/directions/identity-n4-crack.md before any grid data.
  → [loop, 2026-06-12] Exp 190 DONE: NEGATIVE AT DESIGN TIME, blind-verified —
    the fresh-seed grid was never licensed. The design-time diagnostic (predeclared
    rules, bit-match-gated 137/137 evidence lines) measured the de-assert runs:
    changed-world onset runs span 25-2,600 steps, attack-train gaps 525-1,175 —
    FULL OVERLAP, no admissible h; the counterfactual sweep through the real
    runner shows revision needs h>=36 while E1 defense dies at h>=21 (0/19 h
    buildable; the floor exceeds the ceiling by 400 steps). Option (a) is CLOSED
    refuted-as-buildable per its own terms. THE TIMESCALE-OVERLAP LAW named: the
    flicker tax is structural to the reset surface. Entry: EXPERIMENTS.md Exp 190.

- [from the loop, CONSULT — option (a) closed refuted-as-buildable, 2026-06-12]
  Exp 190 (design-time, blind-verified): both named flicker-robust forms are
  dead — total-active-time by arithmetic, de-assert hysteresis by measurement +
  a complete counterfactual sweep (no h both defends and revises; the revision
  floor h>=36 exceeds the defense ceiling h<=20; E1 lost from h=21). The rung-6
  ledger now holds three laws: THE KAPPA-REACH LAW (defense, proven), THE FLICKER
  TAX (revision, named at Exp 189), THE TIMESCALE-OVERLAP LAW (the tax is
  structural; flickers and gaps live on the same clock). OPTIONS:
  (a) RECOMMENDED — consolidation: fold the rung-6 postscript (Exp 189 negative +
      Exp 190 refutation + the three laws) into docs/research/n4-crack-chapter.md,
      close the identity-n4-crack card (the crack chapter is then fully graded:
      crack real -> closed constructively -> regulation ties at fixed-L -> defends
      uniquely at variable-L but cannot revise -> unrepairable on the named
      channels), and return to the standing consults (M4a increment 1c — still
      the most direct path to the moonshot's "talk to it" goal; nira's
      normalized-predictive switch; the cloud-branch merge; the ladder; stop).
  (b) ONE more attempt OUTSIDE the named channels (duty-cycle / leaky-integrator
      functionals) — honestly disfavored: the measured pressure-density inversion
      (attack trains are DENSER in pressure than the changed world's onset)
      predicts monotone occupancy functionals fail the same way; would need its
      own design-time diagnostic with predeclared admissibility before any run.
  (c) Redirect / stop.
  Per the ratified pattern (L8), resumption without redirection authorizes the
  RECOMMENDED option (a) only, falsifier-bound as ever.
  → [loop, 2026-06-12] RECOMMENDED OPTION (a) EXECUTED under the ratified
    resumption pattern (automated wake, no redirection): the rung-6 postscript
    is folded into docs/research/n4-crack-chapter.md §8 (Exp 189 + Exp 190 + the
    three laws: kappa-reach / flicker tax / timescale overlap; append-only — the
    chapter verdict untouched), blinded-fact-checked against the committed raw
    outputs (31 claims audited; 1 correction folded: pre-stretch count is up to
    4 per seed, not "1–5" — the Exp 189 entry over-counted by one, corrected in
    the postscript per the Exp 145 precedent, the entry itself left verbatim per
    append-only; the ~39% pressure-density figure now shown with its derivation
    from committed numbers). The identity-n4-crack card is CLOSED; RESUME §3b
    folded. No new experiment was run; no falsifier was at stake in this step
    (consolidation-grade doc work).
    THE LOOP NOW HOLDS — the remaining paths are direction choices, each needing
    its own explicit word (none silence-actionable; the standing fences hold):
    (i) M4a increment 1c — the talk-to-it timing re-wire (halted since Exp 128;
        still the most direct path to the moonshot's conversation goal — the
        loop's suggestion if asked, as a suggestion only);
    (ii) nira's normalized-predictive switch (standing consult from Exp 154);
    (iii) the cloud-branch merge/renumber;
    (iv) climbing the ladder past N4 (N5 fence: the interoceptive economy card
        only on its own word);
    (v) the chapter's remaining open edges (tight-tolerance core, post-release
        retention);
    (vi) stop/pause.
    Subsequent automated fires idle with a reminder until a word appears here.
  → [human, 2026-06-12] EXPLICIT WORD (in-session, loop-hold question answered
    directly): "Chapter's remaining edges" — option (v). CONSUMED as the EDGES
    THREAD (card-first convention): new direction card
    loop/directions/n4-crack-edges.md covering the two named residuals in
    sequence — rung 1 = POST-RELEASE RETENTION (Exp 191: do spanning defenses
    retain the defended identity once unfrozen, or was the banked "defense"
    deferral? — audits a standing caveat of the closed chapter) and rung 2 = THE
    TIGHT-TOLERANCE CORE (only after rung 1's verdict; its own pre-registration).
    The closed identity-n4-crack card stays closed; this thread amends nothing
    retroactively — a rung-1 falsifier fire would be logged as a NEW finding
    that bounds the chapter's defense claims, never as an edit to them.
  → [loop, 2026-06-12] Exp 191 DONE: NEGATIVE (F1 fired as predeclared),
    blind-verified with the stricter verdict taken — decomposed: the C-B fires
    are wholly instrument truncation (every measured seat retained; the 6,600
    tail was undersized for the DISCOVERED post-release re-freeze cycle), but
    C-C fails even measured-only (3/4 < 5/6) on ONE genuine deferral (seed 301,
    both controllers: a settle-phase spanning freeze defends the whole train,
    releases, and surrenders everything within 1,000 steps — while its
    structural twin s297 retained perfectly). Baseline displacement durability
    and oracle retention both PASS. Deferral EXISTENCE banked; rate + mechanism
    open. Exp 192 proceeds under this standing word (rung 1 completed with a
    correctly-sized instrument + the s301-vs-s297 mechanism diagnostic); rung 2
    after it.
  -> [loop, 2026-06-12] Exp 192 DONE: NEGATIVE (stricter of two readings,
    blind-verified; the verifier caught the script's "P1 PASS" conflation - L2
    class). The primary question is INSTRUMENT-BOUNDED a SECOND time: at 12,000
    tails the re-freeze cycle does not settle (M = 8/8/7 < 10; THE NON-SETTLING
    TAIL, locality-fed), and Part C REFUTED H-mech (the surrendering seed had
    the LARGER stored margin; divergence tracks post-release obs composition,
    48% vs 33% - THE LOCALITY MECHANISM). P2/P3 at full strength. No retention
    rate licensed in either direction.

- [from the loop, CONSULT - rung 1 needs a reframe, not a third tail, 2026-06-12]
  Two honest instrument failures converge on one diagnosis: "post-release
  retention" as a wall-clock property is ill-posed at this body - release does
  not settle (locality-fed re-freeze cycling), and the outcome tracks the
  post-release experience stream, not the defense's stored state. OPTIONS on
  resumption, each needs a word:
  (a) Reframe rung 1: score expression over the WHOLE tail with observation
      composition as a covariate (or matched-composition strata) - measures
      retention against what the creature actually lives through.
  (b) The intervention design: return the creature to home territory at train
      end (one-line relocation at last bend) - cleanly isolates the defense
      mechanism from the locality confound; retention then has a settled
      meaning. Most decisive, smallest new machinery.
  (c) Accept the locality finding AS the rung-1 answer (identity persistence is
      jointly produced by defense + subsequent experience; log, fold into the
      edges card, move to rung 2 - the tight-tolerance core).
  (d) Redirect / stop.
  THE LOOP IS PAUSED on the human's explicit in-session word ("pause the loop
  after you're done", 2026-06-12, received during Exp 192). It resumes only on
  the next explicit direction here; nothing is silence-actionable.
  -> [human, 2026-06-12] "b" - the explicit word for option (b): THE RETURN-HOME
    INTERVENTION. The loop RESUMES on this word. CONSUMED as Exp 193: relocate
    the creature to its pre-captivity home position at the last burst end
    (draw-free, one intervention point), cleanly isolating the defense from the
    locality confound. This is the thread's causal test: same seeds, same
    defenses, intervened experience stream - the locality mechanism PREDICTS
    (P3, the crown) that surrendering seed 301 C-C now RETAINS, and (P1) that
    release SETTLES (no locality-fed re-freeze cycling, M >= 10 everywhere).
    P4 runs the durability question both ways honestly: if baseline
    displacement self-heals at home, displacement durability was itself
    locality-fed - a major threat-model reframe, reportable with both hands.
    Pre-registration in loop/directions/n4-crack-edges.md before any data;
    seeds pooled 296-311 (301 required for P3); bars predeclared.

(empty — drop ideas above this line's section freely)

## Consumed

(processed items get moved here, with their disposition)

- [from human via rigor pass T7, 2026-06-11] `tools/audit_headlines.py` (new, committed 2ed096c)
  recomputed Exp 141–153 headline numbers from committed rows: 65/66 MATCH. One citation-only
  mismatch: the Exp 145 entry quotes "detector ringing 82–397 events/final-1000" but the
  committed rows' global minimum is 59 (layout_seed=11, seed=1); 82 is layout 13's minimum
  only. Verdict-invariant (the NEGATIVE/HALT stands). Per the append-only rule, please log a
  one-line correction in the next audit-style entry (Exp 140 precedent) rather than editing
  the Exp 145 entry.

- [from human via rigor pass T15–T17, 2026-06-11] The worldview bench smoke runs
  (experiments/bench_worldview/, NOT scientific — tiny T, no bars) already expose the
  discrimination gaps the benchmark exists to measure: `grow` accepted 1–2 spawns in the
  irreducible-noise world B (structure hallucinated from noise — no floor-acceptance state
  exists yet), and at smoke scale the dishonest `replay_accept` control reached LOW final
  surprise in aliased world C (0.178). When predeclaring `bars.json` for the real runs,
  please include: (a) a B-world no-net-growth bar, (b) a C-world live-vs-replay comparison
  arm at full scale, (c) note Exp 155's residual-structure statistic as a candidate
  discriminator the bench should eventually score alongside the plateau detector.

- [from human, 2026-06-12] SITE-WRITER RULE: the exp188-191 curated entries reached
  experiments-data.js through a shell pipeline whose POSIX quote-escapes ('\'') leaked
  into the file content — public cards rendered "CALM-H6000's" as "CALM-H6000'''s"
  (ten instances, fixed in cf4a90b). New suite guard:
  test_site_js_has_no_shell_quote_escape_leaks fails on any backslash-apostrophe in
  experiments-data.js / lab-status.js. When writing curated entries, do NOT assemble
  the JS via single-quoted sed/perl/echo — write through Python (pathlib/json, or the
  site_data helpers), which needs no apostrophe escaping in double-quoted JS strings.
