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

(empty — drop ideas above this line's section freely)

## Consumed

(processed items get moved here, with their disposition)
