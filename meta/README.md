# Meta Monkey Phase 0

## What this is

Meta Monkey Phase 0 is a passive process-memory layer for completed research-loop
iterations. It records structured JSON episodes under `meta/episodes/` so later work
can reason about the process record instead of scraping prose every time.

## What this is not

This is not a meta-controller, not an autonomous agent, and not a new active-inference
experiment. It does not run experiments, choose experiments, grade scientific claims,
call LLM APIs, call OpenRouter, mutate creature state, edit `EXPERIMENTS.md`, or edit
`experiments-data.js`.

## Why it exists

The loop already has strong process rules in `loop/PROTOCOL.md`, `loop/ROUTING.md`,
`loop/META.md`, `loop/LESSONS.md`, and `loop/check_iteration.py`. Phase 0 records how
completed iterations line up with those rules: expected process outcome versus actual
process outcome. That is the first substrate for future prediction-error memory.

## Commands

```bash
uv run --python .venv python -m meta_monkey.collect_iteration --exp 190 --dry-run
uv run --python .venv python -m meta_monkey.collect_iteration --exp 190 --write
uv run --python .venv python -m meta_monkey.collect_iteration --latest --dry-run
uv run --python .venv python -m meta_monkey.report
uv run --python .venv python -m meta_monkey.preflight
```

`--dry-run` prints deterministic JSON and writes nothing. `--write` writes only
`meta/episodes/expNN.json`.

## JSON episode shape

Each episode has `schema_version = 1` and records:

- `exp`, `collected_at_utc`, and optional `commit_sha`
- `artifacts`: script/output paths, file existence, and curated site trace references
- `entry`: required `EXPERIMENTS.md` fields, claimed verdict, insight tag, verifier status
- `checks`: `loop.check_iteration` hard failures, warnings, and pass status
- `process`: deterministic likely risks, process failure flag, and notes
- `future_policy_hint`: a non-binding process hint for later review

The JSON serializer uses sorted keys, two-space indentation, and a trailing newline.

## Relationship to loop docs

- `loop/PROTOCOL.md` remains the binding iteration procedure.
- `loop/ROUTING.md` remains the binding conductor / worker / verifier discipline.
- `loop/META.md` remains the rule for fixing recurring process failures.
- `loop/LESSONS.md` remains the compressed incident-derived rules card.
- `loop/check_iteration.py` remains the mechanical checker; Meta Monkey records its output.

## Future directions

- Prediction-error memory: compare expected process outcomes with observed artifacts.
- Tool affordance learning: learn which tools catch which classes of process failure.
- Offline consolidation: summarize repeated risks across recent episodes.
- Active-inference cognitive router later: use passive memory as evidence, not as control.
