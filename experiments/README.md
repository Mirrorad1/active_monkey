# experiments/ — convention for Loop B experiment scripts

## Convention (from Exp 41 onward)

### One script per experiment

Each experiment lives in a single file:

```
experiments/expNN_<slug>.py
```

where `NN` is the two-digit experiment number and `<slug>` is a short, lowercase
hyphen-separated description (e.g. `exp41_valence-baseline.py`).

### Predeclared hypothesis in the docstring

The script's module-level docstring MUST contain the predeclared fields from
`loop/PROTOCOL.md` step 2, written **before** running:

```python
"""
Hypothesis: <one sentence>
Prediction: what specific result you expect if the hypothesis is TRUE.
Falsifier:  what result would count as the hypothesis being FALSE.
"""
```

No result may be interpreted against a prediction written after seeing the output.
See `loop/VALIDATION.md` for the full honesty rules.

### Raw output committed alongside

The raw terminal output of the recorded run is committed at:

```
experiments/outputs/expNN.txt
```

This file must be an unedited capture of `stdout`/`stderr` from the run that the
EXPERIMENTS.md entry describes. It is the evidence base for the logged claim.

### How to run

From the repo root:

```bash
uv run --python .venv python experiments/expNN_<slug>.py
# or equivalently:
PYTHONPATH=. uv run --python .venv python experiments/expNN_<slug>.py
```

Scripts must import from the repo (e.g. `from active_loop import ...`) using the
repo-root `PYTHONPATH`, not from an installed package.

### Scripts must never live in /tmp

Scripts written to `/tmp` or any path outside the repo are gone after the session.
The `expNN:` commit MUST contain the script + output + EXPERIMENTS.md entry together
— all three in one atomic commit. A log entry without its script and output is an
unverified claim (see `loop/VALIDATION.md` reproducibility rule).

### Curated entry in experiments-data.js

Each new experiment (Exp 41+) also adds a curated entry to `experiments-data.js` in the
same atomic commit as the script and output files. The entry's `trace` block points to the
files above:

```js
trace: {
  script: "experiments/expNN_<slug>.py",
  output: "experiments/outputs/expNN.txt",
}
```

The fast test suite (`uv run --python .venv pytest -q`) enforces that both paths exist on
disk. The journey page (`journey.html`) renders the script and output in-page from those
paths, so readers can inspect the evidence without leaving the site. The `rerun` and
`verified` fields are omitted until an independent re-run is performed; the journey page
renders a note "original run committed; not yet independently re-run" in that case.

### Historical note

Scripts for **Exp 1–40** predate this convention and were not preserved — they were
typically written to `/tmp` or discarded after the session. The quantitative claims
in those entries are log-only and cannot currently be replicated from this repo. See
the correction entry at the bottom of `EXPERIMENTS.md` for the full honest record.
`converse_demo.py` (Exp 35 capstone) is the only committed runnable artifact from
that period and still runs. The recovered scripts for Exp 1–40 live in
`experiments/recovered/` — that directory is a sealed historical archive; nothing new
goes there.
