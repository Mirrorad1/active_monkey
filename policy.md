# policy.md — rules of engagement

## Goal
Lower the free-energy metric from `eval/score_json` (mean chosen-action EFE; lower is better).
A change is KEPT only if the metric improves AND all guardrails pass; otherwise it is REVERTED.

## Mutable surface
You may edit ONLY `active_loop/model_spec.py`. Keep it valid Python and a valid pymdp model
(normalized A/B columns, correct shapes) or the evaluator errors and the change is reverted.

## Trust boundary (never touch)
Everything in the `FROZEN` manifest: the evaluator (`eval/`), task (`active_loop/task_env.py`),
oracle (`active_loop/oracle.py`), the loop machinery, and this file. Edits touching FROZEN are
auto-reverted.

## How to propose
One small hypothesis-driven change per iteration. State the hypothesis. Make the smallest edit
that tests it. Read the world model first; don't repeat settled experiments.

## Don't game the metric
Always-ASK, never-ASK, and collapsed preferences are blocked by guardrails (ask-rate band +
success floor). Real progress keeps task success up and the ask-rate in band.

## Discipline
NEVER STOP on your own. Trust the evaluator. Keep everything in git. On a guardrail breach
(repeated broken/frozen-touching proposals), write NEEDS_HUMAN.md and stop.
