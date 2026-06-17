# ROUTING - Claude conductor / Codex worker discipline for loop B

This file is the cost-control layer for the Claude `/loop` research loop. It does
not weaken `loop/PROTOCOL.md` or `loop/VALIDATION.md`; it says who should do which
kind of work. Current experiment: keep Claude as the high-level research conductor,
but route bounded subagent work through the Codex plugin on a fast/high-reasoning
profile.

## Main model: highest-thinking Claude conductor

Use the highest-thinking model for:

- choosing the experiment and reading `loop/IDEAS.md`
- writing or approving the predeclaration, prediction, and falsifier
- deciding whether the design can honestly fail
- reviewing any code returned by a coding subagent
- applying `loop/VALIDATION.md` conjunct by conjunct to raw output
- writing `EXPERIMENTS.md`, `site/data/experiments-data.js`, `RESUME.md`, and final commits
- choosing whether a result is POSITIVE / NEGATIVE / MIXED and CONSOLIDATION / NEW INSIGHT

The main model may ask for help, but it owns the scientific claim.

## Coder subagent: Codex high-fast

Use a Codex plugin worker when implementation is bounded and the spec is tight.
Preferred profile: `codex high-fast`. If the plugin needs explicit fields instead
of that profile name, use a Codex worker with `model = "gpt-5.4-mini"` and
`model_reasoning_effort = "high"`.

Give it:

- exact files it may edit
- exact behavior to implement
- exact verification command
- a reminder that other agents may be editing other files

Default write scope: one `experiments/expNN_<slug>.py` script, or one small helper
module plus its focused tests. The coder does not edit `EXPERIMENTS.md`,
`site/data/experiments-data.js`, `creature/state/`, `RESUME.md`, or git history.

## Verifier subagent: Codex high-fast

Use a separate Codex plugin worker for `loop/PROTOCOL.md` step 4.5. Use the same
preferred profile (`codex high-fast`; explicit fallback `gpt-5.4-mini` with high
reasoning). It receives only:

- the script's predeclared hypothesis / prediction / falsifier
- the committed raw output at `experiments/outputs/expNN.txt`

It must ignore verdict claims printed by the script and recompute the verdict from
raw numbers. It returns POSITIVE / NEGATIVE / MIXED plus the conjunct-by-conjunct
mapping.

## Clerk subagent: Codex explorer

Use a Codex explorer for low-risk mechanical help. Prefer `gpt-5.4-mini` with
low or medium reasoning for cheap read-only work:

- read-only grep or file inventory
- summarizing raw tables without assigning verdicts
- checking whether generated files look stale
- drafting non-binding notes for main-model review

The clerk does not write experiment code, does not write the research log, and
does not grade results.

## Fan-out rules

- Split by disjoint write sets; two agents should not own the same file.
- Prefer worker branches or worktrees for code-writing fan-out.
- Do not let autosync or subagents stage arbitrary scratch files.
- The main model integrates results, runs the focused checks, then runs the broader
  suite appropriate to the change.
