# active-loop — session bootstrap

**Every fresh session: read `RESUME.md` first.** It holds the premise, the durable finding
(the RECIPE), where we are, and how to continue. Do not start work before reading it.

This repo contains TWO loops — don't confuse them (RESUME.md §4):
- **Loop A** (code-mutating autopilot): `run_loop.py` machinery, governed by `MISSION.md` +
  `policy.md` + the `FROZEN` manifest. Never edit FROZEN paths.
- **Loop B** (Claude-driven research): you designing/running experiments. Governed by
  `loop/PROTOCOL.md` (iteration discipline) and `loop/VALIDATION.md` (honesty rules —
  non-negotiable). The modular prompt system lives in `loop/` — directions, personas, and the
  human's `loop/IDEAS.md` inbox.

Standing rules for ALL work here:
- Honest research only. Negative results are results; log them. Never reframe a failure as a
  success. Mark consolidation vs. new insight explicitly. `loop/VALIDATION.md` is binding.
- `EXPERIMENTS.md` is append-only. Never rewrite past entries.
- Run Python via `uv run --python .venv ...` from the repo root (shell auto-activates conda
  base and shadows the venv). Experiment scripts need repo root or `PYTHONPATH=.`.
- Keep everything in git; small, single-purpose commits.
- Experiment scripts + raw outputs are committed under `experiments/` in the same commit
  as their EXPERIMENTS.md entry; never write experiment scripts to /tmp.
- Self-healing: when you find a noteworthy non-research issue or a reusable insight, follow `loop/META.md` — fix it AND add a durable guard (a fast test, a loop-module rule, or a skill via `/claudeception`) so it can't recur. A fix without a guard is incomplete.
