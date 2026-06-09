# direction: persistent-creature

**Question.** What can only be discovered by a creature that *never* restarts — one whose
accumulated history is the independent variable?

**Why it matters.** The RECIPE's third ingredient is "continuous registered experience
(belief never reset)." Every experiment in Exp 1–40 honored this within a single script
run but violated it across the program: each new experiment started the creature at step 0.
The recipe therefore held inside a session and was ignored at the session boundary. This
direction raises it to the program level. One named creature — **mirro** — persists across
experiments, accumulating weights, beliefs, self-formed values, and vocabulary committed
to `creature/state/mirro/` (arrays.npz + manifest.json + append-only BIOGRAPHY.jsonl).
Each experiment is not a fresh trial but an **episode in a life**. The API: `birth /
save / load / fork / live / teach_word` in `active_loop/creature.py`.

The thesis is also a honest constraint: persistence is **engineering + the RECIPE taken
seriously**, not itself emergence. What could be NEW INSIGHT is what only a continuous
life can show — interference effects, inertia trajectories, fork-vs-original divergence,
transfer when priors are real priors, not re-initialized ones. Those effects are
genuinely riskable; they may not appear or may appear as walls.

**Parallel track.** This direction runs ALONGSIDE the fresh-model experiment loop, not
instead of it — other direction cards remain the default for mechanism validation with
clean step-0 controls. The long-running life (`run_life.py`, optionally on a schedule)
accumulates mirro's biography BETWEEN experiments. Experiments that need the lived
creature `fork()` its latest committed snapshot, so the life feeds controlled experiments
without being consumed by them. The daily/session runner is separate from the experiment
runner (`run_loop.py`); neither controls the other.

**Experiment ladder.**

1. **Exp 41 — BIRTH.** Birth mirro in the 3×3 aliased world; live ~900 steps; commit the
   snapshot. Property-falsifier: map accuracy >= 8/9 cells correctly represented in the
   committed belief. FAIL = map accuracy < 8/9 or creature fails to converge within
   1 800 steps.

2. **Continuity across sessions.** Save mirro mid-life; reload in a fresh Python process;
   keep living for another 300 steps. Falsifier: map accuracy or belief calibration
   *degrades* measurably after the resume (it must not). FAIL = degradation outside noise
   (< 7/9 accuracy post-resume or entropy increase > 10%). Single-seed is admissible here
   because the falsifier is a magnitude bound, not a stochastic outcome.

3. **Accumulating values.** Mirro accumulates comfort experience that shapes its
   self-formed favorite. Run a `fork()` twin on the alternative comfort history. Falsifier:
   mirro and its fork-twin do NOT diverge in favorite opinion after equivalent experience
   (the Exp 26 result, now expressed as one life + one counterfactual control). Property
   threshold: >= 2 of 3 fork-pairs diverge. FAIL = no divergence in any pair.

4. **Revision with inertia in one life.** Mirro's world changes; its favorite must revise.
   Record the conviction trajectory in BIOGRAPHY.jsonl. Falsifier: revision happens
   instantly (no inertia) OR does not happen at all. Property threshold: inertia
   measurable as >= 3 steps with intermediate conviction before the new favorite locks.
   This is the Exp 40 claim, now as a continuous-life observation with a biography record.

5. **Growth — transfer in one life.** Move mirro to a LARGER world (e.g. 5×5). Does the
   prior map structure help (faster convergence than a newborn) or interfere (systematic
   errors)? This is genuinely riskable: both outcomes are findings; "no difference" is a
   finding. Property threshold: faster = convergence in < 70 % of the naive-creature
   baseline steps; interfere = systematic error rate > 20 % in previously-learned cells.
   FAIL = inconclusive (effect size < 5 % in either direction across 3 seeds).

6. **Vocabulary that accumulates.** Words taught to mirro in Exp 41 must still answer
   correctly in Exp 44+, loaded from the committed snapshot. Falsifier: answers degrade
   (accuracy < taught-session accuracy) after any save/load cycle. Property threshold:
   >= 90 % of taught labels survive intact across 3 load cycles.

**Discipline notes.**
- Every episode commits the `creature/state/mirro/` snapshot in the same atomic commit as
  the script, output, and EXPERIMENTS.md entry (PROTOCOL.md step 6).
- Controls are always via `fork()`, never via reset. Resetting mirro to run "the
  alternative condition" destroys the life that is the independent variable.
- All falsifiers are property-level with explicit thresholds (VALIDATION.md §Statistical
  reproducibility & the persistent creature). Exact-number post-hoc standards are not
  admissible for stochastic or lived experiments.
- **Mirro is never reborn.** If an experiment requires a creature with genuinely different
  innate structure, birth a new creature with its own name and record why mirro was not
  the right subject. Convenience resets are forbidden.

**Stop condition.** All six episodes answered (any verdict), or a wall is reached that
requires a substrate mirro does not have (log the wall, name the missing substrate, and
open a new direction). If episodes 1–3 all fail, audit whether the persistence engineering
is working before continuing — a broken save/load is not a research finding.
