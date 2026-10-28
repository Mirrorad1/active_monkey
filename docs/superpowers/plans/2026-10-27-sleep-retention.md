# Sleep retention and revision continuation

> **For agentic workers:** Use superpowers:executing-plans inline, with a bounded helper coder and fresh reviewers required by loop/ROUTING.md.

**Goal:** Test C1 retention/revision on fresh seeds without inflating repeated evidence or hiding an old/new tradeoff.
**Architecture:** Reuse DirectHeadAgent and the approved shared sleep lifecycle. All treatments receive one donor's identical observed encounters; compare idle, joint20, restarted chronology and a stronger sequential responsibility-refresh refitter. No persistent spine changes.
**Tech stack:** Existing NumPy/JAX/pymdp, AgentState, pytest; no new dependencies.
**Spec:** docs/superpowers/specs/2026-10-27-recursive-sleep-design.md (approved framework); loop/directions/recursive-sleep.md C1. User's continue authorizes this bounded next rung; numerical design decisions below are implementation choices, not previous human choices.

## Global constraints

Never edit FROZEN or unrelated dirt. EXPERIMENTS.md append-only; script/raw/log/site in one atomic experiment commit. Preserve source-matched reviewed briefs. All experience once-weighted from the pre-history anchor; no hidden answers in helper. Correct mappings belong only to harness/evaluator. No ecology/inheritance/lab-quality claims. Work on current codex branch; current checkout previously chosen because managed-worktree writes were sandbox blocked.

## C1 predeclaration for Exp 280 (before scientific execution)

Fresh seeds 9400..9415, smoke9399 excluded. Original K4/U6/R5/V3, optimism2, lr4, decay.999; first 100 interactions in original world c%4, then common joint20 consolidation (S0). This is the acquired-state candidate; no filtering on its quality. Install S0 A/pA in the same donor, preserve its belief, PRNG, age and continue 100 more interactions with mapping changed only for codes3,4,5 to (c%4+1)%4. Codes0,1,2 retain old meanings. Donor stage2 gamma=4/alpha=1 fixed, actual feedback only for chosen actions. All treatments see identical recorded200 events. S0/prefix source model and source donor are never mutated by candidate refits or evaluation.

After stage2 compare donor online idle, joint20 from original anchor on full200 with donor as start, chronological20 from original anchor, and incremental20 (same anchor/start/data). Each replay treatment200*20 fitting +200*21 monitoring =8200 record-posterior evaluations. Incremental20 initializes all cached responsibilities in one E-step/M-step then performs19 sequential sweeps, replacing each event's old weighted contribution with its new one. It is stronger than restarting the same chronology pass; it remains one algorithm baseline, not proven optimal. No FLOP equivalence claim.

Evaluate S0, donor and all candidates read-only using gamma=alpha=4. Save raw per-code policy and action probabilities, exact expected POS and 120-code common-uniform sampled behavior. Score unchanged codes0..2 separately from changed codes3..5; old mapping evaluated pre/post, new mapping on changed codes pre/post. Primary ALL conjunctions:
1. S0 old full-six csel mean >=.5 (scientific acquisition prerequisite, never an instrument-validity gate; failure is NEGATIVE, all seeds retained).
2. Joint unchanged-code csel minus S0 unchanged mean >=-.05 AND no decrease in >=12/16 seeds.
3. Joint changed-code NEW csel minus S0 changed-code NEW mean >=1/3 AND strictly positive in >=12/16.
4. Joint current-world full-six csel minus EACH idle and incremental20 mean >=1/6 AND strictly positive in >=12/16 separately.
Any scientific failure => NEGATIVE (subcriteria still reported), instrument failure => INVALID. No posthoc thresholds/seed selection, no inference from only a positive subset. No claim of compositional transfer or keeping obsolete answers to changed codes.

Instrument gates: source hashes, unique ordered events0..199, once-weighted count mass, joint objective monotonic tolerance1e-8, all candidate/refit/evaluation source state hashes preserved, checkpoint save/load hashes, phase/mapping change fidelity. Acquisition is an outcome and can fail normally. Runtime: source-frozen excluded pilot, actual peak RSS; two workers only if2*pilotRSS<.6RAM; forecast16*pilot_seconds (conservative serial ceiling) <=1800sec; otherwise halt before confirmation without changing science criteria. Fresh paths/no overwrites. No scientific outcomes inspected until predeclaration is fixed.

## Review focus

Old evidence must not be inherited in the anchor and re-added through the full history. Incremental updates must subtract cached contributions. World-change labels cannot enter helper inference. Failed initial acquisition must be NEGATIVE, not invalid/excluded. Warm replay must leave donor belief/RNG/age intact.

### Task1 — stronger ordinary sequential replay baseline
- [x] Add `mode='incremental'` in active_loop/sleep_replay.py and meaningful tests in tests/test_sleep_replay.py. Coder owns only these files.
- [x] RED->GREEN hand-counted mass, one-pass equivalence to joint, repeated passes conserve mass, deterministic sources unchanged, exact work accounting; existing tests remain green.
- [x] Review and commit helper separately before science.

### Task2 — C1 instrument and controlled run
- [x] Implement experiments/exp280_sleep_retention.py; tests/test_sleep_retention.py pins scientific conjunctions, no exclusion on acquisition, phase mapping partition, source-preserving install/continuation, preflight source matching, no output overwrite.
- [x] Review design/instrument independently, then source-frozen excluded smoke followed by one confirmatory run if resource-safe.
- [x] Independent blinded verifier recomputes each conjunct from docstring and raw outputs only.

### Task3 — evidence retention at lab level
- [x] Append honest Exp280 log, raw/checkpoints, bundle, site; rubric and affected tests; commit atomic science record. Update direction: NEGATIVE halts C1 advancement; positive still does not establish ecology.
- [x] Collect passive meta episode; build and independently review bounded Exp280 lab brief; demonstrate compose attachment. Keep old brief source hashes valid.
- [x] Evaluate claudeception; add guard for any noteworthy implementation issue. Record measured verification, limits, commit IDs and next permitted work.

## Builder check disposition

Independent design check: buildable, no blocker. Stage1 explicitly reuses gamma(1,8,100),
alpha1; stage2 disables the schedule and sets agent gamma4. Its code stream is explicitly
separate, seed+400000. Incremental responsibilities use the full current model before
count replacement. Scope is read-only retained/revised discrimination after continued
wake, not a causal stage3 online-life advantage. These choices are in the runner docstring.

## Completion evidence

Helper/preregistration commit2c8e39b; atomic science commitfd43f52. Scientific verdict
NEGATIVE independently blind-verified; C1 advancement halted. All176 checkpoint hashes
and64 fitted mass calculations audited; raw JSON hash
9ab1d33809052d9aee6afb9e3689738483463add6bdc9a5bddbb73758dc458f4 unchanged.
Source-matched pilot46.681554sec; full525.884678sec, peak9389.096960MB. All122 affected
checks pass;65 lab/compose/meta/site checks rechecked after passive episode and brief.
Iteration rubric passes; nine rounded/derived/preregistration warnings checked against raw.
Exp280-boundary-v1 independently source-reviewed,974bytes; actual compose CLI includes it.
Old278/279briefs remain source-valid. No ecology/inheritance/L1 or later online-life test run.
Claudeception evaluated: incremental math is already captured by helper/docstring/hand-derived
regressions; acquisition-as-outcome is already bound by VALIDATION. No duplicate global
skill or untested scientific generalization extracted.
