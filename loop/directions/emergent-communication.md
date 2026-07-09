# direction: emergent-communication

**Question.** Does a **self-formed signal↔meaning mapping** EMERGE among creatures — and PAY — in a task
where (a) coordination is genuinely REQUIRED (a stigmergic / solipsistic baseline PROVABLY fails) and (b)
the signal, once received, has an ACTIONABLE downstream advantage? These two conditions are exactly the
walls that killed the program's prior social attempts: **stigmergy short-circuited coordination** (Exp 234
self-other-modeling — simple non-communicative agents were already near-optimal, so nothing selected for
communication) and **information carried no advantage** (Exp 268 acoustic-ecology — the eavesdropping channel
carried real bits about predator density, MI 0.643, yet gifted hearing gave NO capture-hazard benefit
because the metapopulation left nowhere to flee). So the real experiment is NOT "can we build a channel" —
it is "can we design a task where communication is NECESSARY and REWARDED, and then does a shared meaning
self-organize on a channel that starts MEANINGLESS?"

**Why it matters.** The moonshot's deep goal is an agent that forms its own content from experience and can
express it. Value TRANSMISSION already works (Exp 66 — creature-to-creature value transfer, young receivers,
the dose-vs-mass law), but that is a receiver absorbing a sender's state, not two agents converging a SHARED
CODE. The closed social-emergence ladder (Exp 63–74) named the gap: creatures are solipsistic and coordination
deflated to unilateral stigmergic lock-in. This direction asks whether a genuine communicative CONVENTION
(an arbitrary signal acquiring shared meaning through selection/learning, beating no-signal AND shuffled-signal
controls) can emerge — the smallest honest step toward agents that coordinate by a code they built, not one we
handed them.

**CEILING (binding — declare it up front).** The claim is EMERGENT SIGNALING over a FINITE signal set (a
signal↔state/action convention), NOT emergent compositional GRAMMAR / language-from-scratch — that stays the
documented ceiling (`open_problem.html`, PREMISE). No claim of syntax, composition, or negotiated ontology;
the honest frame is "an arbitrary token acquired shared, actionable meaning" vs "it did not."

**Not this (binding framing).**
- **NOT provided meaning (the cardinal sin).** The signal CHANNEL may be provided (a costed EMIT action with
  K meaningless tokens + a RECEIVE percept), but the MAPPING signal→meaning must EMERGE from selection or
  within-life learning. If the harness assigns which token means what, it is not communication — it is a
  taught label, and the entry must say so. "The creatures communicated" when the harness wired the code is
  the cardinal sin here (VALIDATION).
- **NOT stigmergy-solvable (the Exp 234 wall).** The task MUST be one where a non-signalling baseline
  (stigmergic trail-following / solo search / the Exp 70 comfort-gated policy) PROVABLY fails — e.g. the
  resource MOVES each round so there is no persistent trail, or the needed information is PRIVATE to a
  non-actor. If a simple baseline already coordinates, the direction CAN'T-POSE (redesign), not "communication
  emerged".
- **NOT advantage-absent (the Exp 268 wall).** Acting on the signal must have a real, reachable payoff (a
  gifted-oracle-meaning agent must clearly beat the no-signal baseline). Information present without an
  actionable response is a NEGATIVE, logged as such.
- **NOT an external evaluator.** No global fitness ranker selects "good communicators"; signal cost, meaning,
  and reward flow only through individual action + own-state reproduction. All new paths `enable_*`-gated,
  byte-identical OFF, golden-hash guarded, deterministic under seed.

**The provided-vs-earned line (the crux).** Provided: a signalling AFFORDANCE (an emit action over K tokens,
costed so it is never free) + a receive channel (others perceive recent tokens) + the sender's PRIVATE
information or the coordination task itself. Earned / must emerge: (i) that senders emit CONTINGENTLY on
their private state (signal carries information — measured by MI, but MI alone is NOT success, cf. Exp 268),
AND (ii) that receivers ACT on tokens in a way that pays, AND (iii) that this is a SHARED CONVENTION — a
permuted/shuffled-token control (relabel the tokens for receivers only) must DESTROY the benefit; if any
token works as well, there is no convention, just a generic alarm. Honest incentive check: name WHY an honest
signal is selected for (kin structure / shared food pool / the sender also benefits) — a costed signal with
no sender payoff will not evolve, and that is a real result, not a bug.

**Experiment ladder.** (each one PROTOCOL iteration; each names its FAILURE; next number = re-confirm at launch)

1. **Rung 1 — POSABILITY pre-flight (L28/L45; the decisive gate, where 234 and 268 both died).** In a
   candidate task (recommended: a HIDDEN-RESOURCE referential game — food appears at one of N cells each
   round; a SCOUT perceives the cell privately but cannot exploit it in time; FORAGERS cannot perceive it but
   can; the resource MOVES each round so no stigmergic trail persists), verify BOTH preconditions with GIFTED
   oracles, no emergence yet: (a) a stigmergic/solo baseline PROVABLY fails (forager intake at chance/search
   level); (b) a GIFTED-ORACLE signaller (scout emits the true cell, receivers act on the true mapping)
   CLEARLY beats the baseline (real actionable advantage). **FAIL** = a simple baseline already coordinates
   (stigmergy wall, 234) OR the oracle gives no advantage (advantage-absent wall, 268) ⇒ CAN'T-POSE, redesign
   the task; do NOT run emergence on faith. Also predeclare the honest incentive (why an honest scout is
   favoured) and confirm a costed signaller is viable.
2. **Rung 2 — does a CONVENTION emerge from a meaningless channel?** Start tokens meaningless (random
   emit/act policies or priors); let selection and/or within-life learning shape emit(private_state) and
   act(token). Predeclared PASS: forager coordination beats the no-signal baseline by a declared margin at
   ≥ the seed threshold, AND signal MI(token; private_state) rises from ~0, AND a SHUFFLED-TOKEN control
   (receiver-side relabel) DESTROYS the benefit (proving a shared convention, not a generic cue). **FAIL** =
   no better than no-signal (communication does not emerge) OR benefit survives token-shuffle (no real
   convention — a degenerate "any signal helps" cue). Watch the mean-of-opposites trap on the per-seed
   convergence; watch drift (L24/L29).
3. **Rung 3 — is it REAL communication, not a confound? (only if rung 2 passes).** Controls: emit-blind
   sender (can't see private state → MI must collapse), receive-blind forager (benefit must collapse),
   cost sweep (does the convention survive a fair signal cost, L30), and a symbol-count / channel-capacity
   probe (does N-cell coordination need ≥ log2(N) tokens — a graded prediction, not a point). **FAIL** = the
   benefit is explained by a non-communicative confound (shared environment cue, positional artifact L40,
   priority effect L41).
4. **Rung 4 — mutual / multi-role (declare the ceiling).** Bidirectional or many-signaller coordination:
   does a stable shared code persist, or does it drift / fail to converge across the population? **CEILING
   (binding):** finite-signal convention only — NO claim of compositional grammar or negotiated ontology
   (`open_problem.html`). The honest claim is a shared, actionable, arbitrary-token convention vs its absence.

**Discipline notes.** Reuse where honest: the Exp 66 value-transmission machinery, the Exp 70 comfort-gated
policy as the stigmergic baseline, the patch-mosaic / ecology substrate ONLY if rung 1 shows it can pose the
task (Exp 268 already showed the metapopulation deflates advantage — prefer a purpose-built moving-resource
referential substrate if the mosaic CAN'T-POSE). One new mechanism per iteration. Every signal, channel, and
percept is a PROVIDED prior declared in the entry like a taught label; the MEANING is the only thing that may
be called emergent, and only after the shuffled-token control passes. Functional language only. Higher
coordination is NOT "more mind" — functional competencies only; inner experience stays unverified (VALIDATION).

**Stop condition.** Exhausted when rung 1 returns a posability verdict (CAN'T-POSE if stigmergy solves it or
the oracle gives no advantage — a real boundary, logged; else proceed), rung 2 returns an emergence verdict
either way (a convention emerged + beat both controls, or it did not), and — if positive — rung 3 rules out
confounds and rung 4 reaches its verdict or the declared grammar ceiling. Then write in EXPERIMENTS.md either
the emergent-convention finding with its shuffled-token + confound evidence, or "no disciplined emergent
communication at this substrate because <stigmergy-solves-it / advantage-absent / no-convention>", and open
the next substrate or stop.

**STATUS.** state: exploratory (drafted 2026-07-08, not yet run) · latest: none · depends-on: a task where a stigmergic baseline PROVABLY fails + a gifted-oracle signaller PROVABLY pays (rung 1 posability, the 234/268 double-wall) · reusable: TBD (target: a gated emit/receive channel + a moving-resource referential substrate) · why: the two prior social attempts died at stigmergy-solves-it (234) and advantage-absent (268); this direction makes those the binding rung-1 gate, then asks whether an arbitrary token acquires SHARED actionable meaning (below the grammar ceiling) · next-falsifiable: rung 1 — a stigmergic/solo baseline fails AND a gifted-oracle signaller beats it, in a moving-resource referential game (else CAN'T-POSE).
