# direction: affective-dyad

**The M4a "talk to it" thread.** The most direct rung toward the moonshot goal. Source of truth:
`docs/specs/m4-affective-dyad.md`. Code: `active_loop/affect_spec.py` (MUTABLE) + `affect_agent.py`;
guard `tests/test_affect_agent.py`. Functional valence ONLY — no sentience claim, ever.

**Question.** Can a toy active-inference agent you TALK TO learn — from scratch, in a short
session — to (1) infer your latent intent from an utterance, (2) act by expected free energy, and
(3) earn your positive feedback (a grounded, functional valence)? I.e. can you *watch it learn to
feel positive*?

**Why it matters.** This is the spec'd path that most directly serves the user's wish ("talk to it
and see if it can learn and feel positive"). It is distinct from the persistent-creature line
(`functional-emergence`) and from the char language model (M3): a turn-based affective dyad that
clusters your utterances into latent intents, has preferences over a valence channel, and learns
(Dirichlet on A/B) which responses earn your `+`. Either outcome is a finding: a working dyad is the
toy "talk to it" milestone; a documented learnability wall says *what* the AIF credit path needs.

**The honesty contract (binding, in every artifact).** Functional valence = `-F` + grounded-extrinsic
approval, NOT subjective feeling. K/R/U intents/responses/codes and the scripted partner are PROVIDED;
the agent LEARNS A (intent↔utterance/valence) + B (response→intent) from weak priors. Do not claim
comprehension, sentience, or that the moonshot (open language) is reached.

**Experiment ladder (increments; each is one falsifier-bound iteration; F3 HALTS for the human — the
ratified consult guardrail, NO self-fix).** The binding learning test is the Exp 125 predeclaration:
P1 inference proper, P2 ASK-reflex alive, **P3 = realized POS-feedback rate rises ≥ 0.15 H1→H2 in
≥ 6/8 fresh seeds (the core; F3 = P3 fails)**, P4 window arithmetic exact. The gifted-EFE liveness
control (`tests/test_affect_agent.py`) must pass first — it proves the EFE/policy is wired (so any
"does not learn" is science, not a bug).
- **Increment 1 — affective core (Exp 125, HALTED F3).** perceive→intent / EFE response / windowed
  Dirichlet. P1/P2/P4 pass; P3 0/8 — the scope cut deferred pB so action had no learnable path to valence.
- **Increment 1b — enable B-learning (Exp 127, HALTED F3).** pB learned; still P3 0/8 — joint A+B
  bootstrap does not converge; suspects scale + timing.
- **Increment 1c — the timing re-wire (Exp 214, HALTED F3).** perceive utterance alone → act → observe
  [code,valence] co-presented → learn the within-turn intent shift. Timing flaw REAL but NOT sufficient:
  P3 0/8, response distribution stays uniform; the gifted-EFE control proves the EFE is SOUND (q_pi 0.71).
  THE WALL: the response→valence credit is INDIRECT (mediated by an intent-transition B whose within-turn
  signal is ~0 until A/B are already learned) — the AIF echo of the program's useful-when-gifted ≠ learnable.
- **Increment 1d — DIRECT response→valence head (NEXT; human's steer 2026-06-15).** Give the generative
  model a direct `P(valence | intent, response)` emission (or an explicit value head) so the learner gets
  a non-vanishing gradient from action straight to feedback, bypassing the near-zero intent-transition
  signal. Re-run the Exp 125 predeclarations + the gifted-EFE control on fresh seeds. FALSIFIER: P3 still
  fails ⇒ even a direct credit path does not learn at this scale ⇒ HALT with the next suspect (capacity /
  lr / session length). Honesty: a direct head is a PROVIDED structural prior — declare it; it does not
  make success inevitable (the agent must still learn WHICH response per intent).
- **Increment 1e+ — converse REPL + FROZEN scorer (only after 1d learns).** `converse.py` honest REPL;
  `eval/affect_score.py` the frozen learns-to-positive metric; then **M4b** = the PR-style autopilot over
  `affect_spec.py` against that metric.

**Stop condition.** Close POSITIVE when an increment demonstrably LEARNS (P3 ≥ 6/8, instrument sound) —
the toy "talk to it and watch it learn to feel positive" milestone, written up honestly with the
provided-vs-learned ledger. Close NEGATIVE when the increment ladder exhausts the credit-path redesigns
(direct head, capacity, session length) without P3 ever passing — then the finding is the affective
learnability wall (the AIF face of useful-when-gifted ≠ learnable), documented for the human.

**STATUS.** state: active (M4a; reliable GENUINE discrimination at K=4 is LOAD-BEARING on the long 300t session). latest: Exp 221 (1j) schedule-vs-length, NEGATIVE / NEW INSIGHT: the precision schedule does NOT rescue the short session (short sched 0/2/3 of 16, bar 12); it beats fixed only at 300t; mean_csel rises with length for BOTH conditions, so short sessions block LEARNING and the schedule only optimizes EXPLOITATION. Regression-anchored exact to Exp 220 (sched_300=13/16, fixed_300=7/16). next: CONSULT - (a) attack LEARNING at short sessions (lr/optimism/replay, not precision); (b) accept the long-session milestone + build increment 1e (converse REPL + frozen scorer); (c) stop/redirect.
