# EFFICIENCY — keeping the loop fast and context-lean (consult when a session feels slow)

The loop accretes history; without discipline each iteration reads more and runs longer. These rules
keep wall-clock and context cost bounded WITHOUT degrading experiment integrity.

## Context: hot vs cold (never load the cold archive whole)
- HOT (load each iteration): `RESUME.md` §3b, `loop/LESSONS.md`, the relevant direction card's STATUS
  line, and the LATEST 1-2 `EXPERIMENTS.md` entries.
- COLD (reach only on demand, via TARGETED reads — never whole):
  - `EXPERIMENTS.md` (660K+): read ONE entry with `awk '/^## Exp N /{f=1} /^## Exp M /{f=0} f' EXPERIMENTS.md`
    or Read with offset/limit; NEVER read the whole file (the latest entry is ~1.7% of it).
  - `site/data/experiments-data.js` (480K+): re-Read only the tail entry + the `AM_TALLY` line before editing (L12).
  - `experiments/outputs/**.json` (raw data, ~26M): NEVER read into context — reference the path. They are
    committed for reproducibility, not for reading.

## Workflows: return lean
- A design/verify workflow's `return` value is pasted into the main context. Return ONLY the concise
  synthesis (+ a file pointer); fetch per-agent detail from the journal/output file on demand
  (e.g. `python -c "import json; ..."`), NOT inline. A 150K-char workflow result is a context leak.
- Scale the design workflow to complexity: the full 3-design + 3-audit panel for genuinely hard or novel
  mechanisms (id-order-class confounds); skip it or use one agent for incremental experiments.

## Compute: parallelize independent runs
- Independent (arm, seed) runs go through `ecology/batch.py` (`run_batch`) — embarrassingly parallel,
  bit-identical to sequential. The per-step time loop stays sequential (it cannot parallelize).
- Sequentially-dependent seed chains (seed k seeded from run k-1) use `sequential=True`; only the
  independent outer seeds parallelize.
- Right-size populations: per-step cost is O(alive) (pure Python), so a run's wall-clock scales with
  population x steps. Keep populations in the low hundreds via concentration/regen/band geometry; a
  no-scarcity regime (high regen) has NOTHING to bound its population and runs toward the runaway cap —
  never use it as a baseline. self._creatures keeps every creature ever born; _alive() scans only
  self._alive_list (O(alive)) — do not re-introduce full-list scans on the hot path.

## Output commits: slim (exp202+)
- Commit `verdict.json` (per-arm-seed summary + hashes + end metrics) + `<exp>.txt` (the human summary)
  + ONE consolidated `trajectories.json` + the script. Do NOT commit 30 separate per-seed traj files
  (reproducible from seed; ~32 files -> 3). Keeps all the science, ~140K -> ~40K, far fewer files.

## STATUS lines: a CAPTION, not a chapter
- A direction card's STATUS line is HOT context (it feeds DIRECTIONS.md). Keep it <= 800 chars: latest
  result (1-2 sentences) + depends-on + reusable + why + next-falsifiable. Put the full narrative in the
  card BODY (cold, read on demand) and `EXPERIMENTS.md`. Guard: `tests/test_status_line_length.py`.

## Git weight (maintenance, not per-iteration)
- The slim-output policy bounds FUTURE growth. The legacy ~26M (esp. exp194's 8.8M move-event bloat)
  stays in history; a one-time prune / git-LFS migrate REWRITES history and must be a deliberate
  maintenance op (pause the cron loop, merge open PRs, coordinate the force-push) — never unilateral.
