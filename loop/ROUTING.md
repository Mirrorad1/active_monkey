# ROUTING - model and subagent budget discipline for loop B

This file is the cost-control layer for the Claude `/loop` research loop. It does
not weaken `loop/PROTOCOL.md` or `loop/VALIDATION.md`; it says who should do which
kind of work.

## Main model: highest-thinking conductor

Use the highest-thinking model for:

- choosing the experiment and reading `loop/IDEAS.md`
- writing or approving the predeclaration, prediction, and falsifier
- deciding whether the design can honestly fail
- reviewing any code returned by a coding subagent
- applying `loop/VALIDATION.md` conjunct by conjunct to raw output
- writing `EXPERIMENTS.md`, `experiments-data.js`, `RESUME.md`, and final commits
- choosing whether a result is POSITIVE / NEGATIVE / MIXED and CONSOLIDATION / NEW INSIGHT

The main model may ask for help, but it owns the scientific claim.

## Coder subagent: Sonnet

Use a Sonnet coding subagent when implementation is bounded and the spec is tight.
Give it:

- exact files it may edit
- exact behavior to implement
- exact verification command
- a reminder that other agents may be editing other files

Default write scope: one `experiments/expNN_<slug>.py` script, or one small helper
module plus its focused tests. The coder does not edit `EXPERIMENTS.md`,
`experiments-data.js`, `creature/state/`, `RESUME.md`, or git history.

## Verifier subagent: Sonnet

Use a Sonnet verifier for `loop/PROTOCOL.md` step 4.5. It receives only:

- the script's predeclared hypothesis / prediction / falsifier
- the committed raw output at `experiments/outputs/expNN.txt`

It must ignore verdict claims printed by the script and recompute the verdict from
raw numbers. It returns POSITIVE / NEGATIVE / MIXED plus the conjunct-by-conjunct
mapping.

## Clerk subagent: Haiku

Use Haiku only for low-risk mechanical help:

- read-only grep or file inventory
- summarizing raw tables without assigning verdicts
- checking whether generated files look stale
- drafting non-binding notes for main-model review

Haiku does not write experiment code, does not write the research log, and does not
grade results.

## Fan-out rules

- Split by disjoint write sets; two agents should not own the same file.
- Prefer worker branches or worktrees for code-writing fan-out.
- Do not let autosync or subagents stage arbitrary scratch files.
- The main model integrates results, runs the focused checks, then runs the broader
  suite appropriate to the change.
