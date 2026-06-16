# Architecture

active_monkey is a research repo, not a product codebase. The architecture is organized
around traceability: core substrate code, experiment scripts, raw outputs, public docs,
and tests that guard the evidence chain.

## Core Simulation And Agent Code

- [active_loop/](../active_loop/) contains the active-inference-inspired controller,
  world models, experiment parser, site-data generation, verdict helpers, and persistent
  creature machinery.
- [ecology/](../ecology/) contains the artificial-life ecology substrate: finite resources,
  homeostatic population dynamics, trait inheritance, thermosense mechanics, batch helpers,
  runtime snapshot/restore, and the reusable sense-axis diagnostics.
- [creature/](../creature/) contains committed persistent creature state and biography.

## Experiment Scripts

- [experiments/](../experiments/) contains committed experiment scripts.
- [experiments/recovered/](../experiments/recovered/) contains recovered early scripts used
  to preserve the historical trail.
- Experiment scripts should be run from the repo root so imports and output paths resolve.

## Raw Outputs

- [experiments/outputs/](../experiments/outputs/) contains raw text, JSON, JSONL, CSV, and
  nested run artifacts.
- [experiments/recovered/outputs/](../experiments/recovered/outputs/) contains early recovered
  output files and reruns.

Raw outputs are evidence artifacts. Do not rewrite them for style, size, or consistency.

## Public Site And Tracker

- [site/pages/](../site/pages/) contains the source HTML pages.
- Root HTML files such as [index.html](../index.html), [journey.html](../journey.html), and
  [math.html](../math.html) are generated deploy entrypoints for GitHub Pages.
- [site/data/experiments-data.js](../site/data/experiments-data.js) and
  [site/data/lab-status.js](../site/data/lab-status.js) support the public experiment tracker.
- [site/data/math-data.js](../site/data/math-data.js) supports the generated math reference.
- [site/styles/](../site/styles/), [site/runtime/](../site/runtime/),
  [site/components/](../site/components/), and [site/assets/](../site/assets/) contain shared
  client code, CSS, React components, and image assets.
- [tools/site/build_static.py](../tools/site/build_static.py) copies `site/pages/*.html` to the
  root deploy entrypoints.

## Docs

- [README.md](../README.md) is the public front door.
- [docs/CLAIMS.md](CLAIMS.md) is the current claim ledger.
- [docs/EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) maps the append-only notebook into chapters.
- [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) explains test and rerun expectations.
- [docs/RELATED_WORK.md](RELATED_WORK.md) places the project against prior art.
- [docs/research/](research/) contains deeper synthesis notes and chapter write-ups.
- [loop/](../loop/) contains the Loop B research protocol, validation rules, direction cards,
  and meta-improvement rules.

## Tools And Meta Infrastructure

- [tools/ref.py](../tools/ref.py) resolves stable project references such as experiment IDs.
- [tools/audit_headlines.py](../tools/audit_headlines.py) audits headline numbers from outputs.
- [tools/check_docs.py](../tools/check_docs.py) checks local Markdown links and claim experiment
  citations.
- [meta/](../meta/) contains meta-monkey episode records and supporting notes.

## Tests And Evals

- [tests/](../tests/) contains unit, regression, site, parser, ecology, and experiment guard tests.
- [eval/](../eval/) contains frozen scoring/evaluation code. Treat frozen evals as trust-boundary
  files.
- Default test command: `uv run --python .venv pytest -q`.

## Generated Or Derived Artifacts

Some public artifacts are generated or patched from source-of-truth files. Examples include
site data derived from [EXPERIMENTS.md](../EXPERIMENTS.md), caveat injection into the public
timeline, math data derived from [math/](../math/), and root deploy pages generated from
[site/pages/](../site/pages/). If changing a generator, run the associated tests and inspect
the generated diff.

## What Not To Touch Casually

- Raw outputs under [experiments/outputs/](../experiments/outputs/).
- The historical notebook [EXPERIMENTS.md](../EXPERIMENTS.md), except append-only experiment
  entries or tiny navigation/link edits.
- Committed creature state in [creature/](../creature/).
- Frozen eval paths under [eval/](../eval/).
- Seed-sensitive experiment scripts when the goal is only documentation.
- Public site data generated from the experiment log unless the generator and tests are part
  of the change.

If a cleanup would weaken traceability, do not do it in a public-legibility PR.
