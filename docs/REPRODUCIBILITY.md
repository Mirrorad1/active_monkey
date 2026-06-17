# Reproducibility

This repo aims for an auditable evidence chain, not perfect one-command reproduction of
every historical run. Many experiments are deterministic under fixed seeds; some batches
are stochastic, slow, or tied to historical scripts and outputs that should be treated as
committed evidence.

## Environment Setup

From the repo root:

```bash
uv sync
uv run --python .venv python --version
```

The project convention is to run Python through `uv run --python .venv ...` from the repo
root. In temporary worktrees, a local `.venv` may not exist; use the established project
environment rather than changing package metadata only to satisfy a worktree.

## Default Tests

Fast default suite:

```bash
uv run --python .venv pytest -q
```

Full suite including slow tests:

```bash
uv run --python .venv pytest -q -m 'slow or not slow'
```

Docs consistency:

```bash
uv run --python .venv python tools/check_docs.py
```

## Representative Runs

Small demo:

```bash
uv run --python .venv active-monkey-converse-demo
```

Structure-growth confirmation:

```bash
uv run --python .venv python experiments/exp154_growth_confirmation.py
```

Sense-gradient audit:

```bash
uv run --python .venv python experiments/exp203_n5_sense_gradient_audit.py
```

Design-stage preflight:

```bash
uv run --python .venv python experiments/exp207_corner_grid.py
```

These are representative, not a guarantee that every historical batch is cheap to rerun.

## Where Raw Outputs Live

Historical output files live under:

- [experiments/outputs/](../experiments/outputs/)
- [experiments/recovered/outputs/](../experiments/recovered/outputs/)
- some nested per-experiment directories, such as
  [experiments/outputs/exp194_n5_homeostatic_population/](../experiments/outputs/exp194_n5_homeostatic_population/)

Do not delete or rewrite these outputs to make the repo cleaner. They are part of the
audit trail.

## Seeds

Most serious experiments fix seed blocks in the script and name the seed set in the
corresponding [EXPERIMENTS.md](../EXPERIMENTS.md) entry. Fresh-seed confirmations are
logged as new evidence, not silently substituted into older entries.

Seed-sensitive scripts should be edited cautiously. A small code cleanup can change the
trajectory and therefore the evidential meaning of a historical run.

## Deterministic vs Stochastic

Deterministic or hash-pinned checks include parser tests, many unit guards, site-data
consistency checks, and some regression hashes around engine branches.

Stochastic or batch-sensitive evidence includes ecology population runs, evolution arms,
and long-horizon persistent-creature behavior. Those runs should be interpreted through
the seed blocks, caveats, and predeclared pass/fail bars in the notebook.

## Recomputing Headline Numbers

Use committed rows and scripts where available:

```bash
uv run --python .venv python tools/audit_headlines.py
```

When a headline depends on nested JSON, JSONL, or CSV outputs, prefer recomputing from
the raw rows rather than copying values from prose. If a script emits a new output file,
commit the script, output, and notebook entry together.

## Known Limitations

- This is a one-person, LLM-assisted lab, not an independently replicated benchmark.
- Some early experiments were recovered and rerun; their recovered outputs are kept
  visible rather than hidden.
- Not every historical script is equally cheap to rerun.
- Some verdicts are instrument-bounded or mixed by design.
- The public docs summarize the evidence; [EXPERIMENTS.md](../EXPERIMENTS.md) remains
  the primary record.

## Slow Tests And Expensive Runs

Default `pytest -q` deselects slow tests through project pytest configuration. Run the
full suite before a release or PR that touches shared engine behavior. For ecology or
long-horizon changes, run the narrow experiment-specific tests first, then the full fast
suite, then any slow or expensive confirmation explicitly required by the claim.
