# Lab sleep cycles

This is advisory process memory, not scientific grading or hosted-model weight training.
Packets select up to eight committed experiments. Recorded findings/boundaries must be exact
excerpts from their experiment entry; every source includes an existing script and raw output.
The compiler always retains the recorded verdict and honest caveat. Generated hypotheses and
rehearsals require `origin: generated` and never become observed evidence.

```sh
uv run --python .venv python -m meta_monkey.sleep build --packet packet.json --out meta/sleep/run-id
uv run --python .venv python -m meta_monkey.sleep review --cycle meta/sleep/run-id --review review.json
uv run --python .venv python -m meta_monkey.sleep inspect --cycle meta/sleep/run-id
uv run --python .venv python loop/compose.py recursive-sleep --sleep-cycle meta/sleep/run-id
```

Packet shape:

```json
{"sources":[{"experiment":279,"script":"experiments/exp279_sleep_replay.py","output":"experiments/outputs/exp279.txt"}],"items":[{"kind":"boundary","source":279,"excerpt":"Exact text from that entry"},{"kind":"rehearsal","origin":"generated","source":279,"text":"A falsifiable next question"}]}
```

The example is a schema illustration; substitute real committed evidence. Review JSON contains
`content_hash` from the receipt, `reviewer`, boolean `accepted`, and fidelity `notes`. A fresh
reviewer checks source attribution, omitted negatives, caveats, and generated labels. The
file records that review; it does not authenticate reviewer identity. A stale, missing or
rejected review prevents prompt attachment. Source, packet or brief changes require a new
cycle and review. Cycles and review records cannot overwrite an existing cycle.

The complete brief has a conservative 1,500 UTF-8-byte ceiling; this is not an exact model
token count. Oversized packets fail without dropping evidence. Narrow the retrieval window
explicitly. The compiler extracts/carries knowledge; an independent prospective experiment
is needed before claiming it improves the lab's research.

## Implemented first slice (2026-10-27)

Shared lifecycle: experience -> record -> consolidate -> rehearse -> test/select -> retain.
Lab adapter: this directory plus `meta_monkey.sleep`. Creature adapter:
`active_loop.sleep_replay` (regularized refitting of unique observed encounters). Both emit
receipts through `active_loop.sleep`; correct-world answers never cross into creature replay.
Memory curation, model/procedure testing, and ecological reproduction are separate selection
mechanisms. Only the first two are exercised here; birth/death selection remains future work.

Two retained lab cycles demonstrate negative and positive evidence handling:
`exp278-baselines-v1` preserves the optimizer-repair failure; `exp279-replay-v1` preserves
the bounded C0 replay result. The complete experiment and reusable bundle are under
`experiments/outputs/exp279*` and `experiment_bundles/exp279/`. Historical executed source
is pinned by that bundle's repo_commit, with a later reporting-only correction disclosed
in `experiments/outputs/exp279_metadata.md`.

To prepare the next research session with the reviewed result:

```sh
uv run --python .venv python loop/compose.py recursive-sleep \
  --sleep-cycle meta/sleep/exp279-replay-v1
```

C0 result: joint refitting mean discrimination .500 vs idle .167 and restarted chronology
.083; both predeclared mean/count comparisons pass. Only 9/16 reach half-correct.
C1 retention/interference/revision needs a new preregistration. C2/C3 ecological costs
and inherited capacity, and L1 prospective research-quality benefit, are untested.

Operator safeguards learned in this implementation are executable tests, rather than
additional scientific claims: complete multiline caveats, source-to-experiment attribution,
actual compose CLI import context, response-conditioned model shapes, once-weighted evidence,
monitoring work accounting, preflight completeness, raw-output collisions and pilot-vs-batch
runtime reporting. See `tests/test_lab_sleep.py`, `test_sleep_replay.py` and
`test_sleep_experiment.py`. A claudeception review found these already covered by repo-local
guards/documentation; no duplicate global skill or untested research rule was extracted.

## C1 continuation: retained negative boundary (Exp280)

The next wake used the reviewed Exp279 rehearsal to design separate retention/revision
metrics and a stronger ordinary sequential refitter. Exp280 is NEGATIVE: acquisition
.4375<.5, retention meets its declared criterion, revision gains only1/12 with5/16 positive,
and current-world gains over idle/incremental are only1/48 (positive4/16,3/16). All16
seeds remain in the analysis. No later online-life benefit or ecology is established.

The lab retains this result in `exp280-boundary-v1`; the earlier positive brief remains
source-valid. C1 advancement is halted pending a named changed lever and fresh
predeclaration. This is evidence retention, not demonstrated causal improvement of research.

```sh
uv run --python .venv python loop/compose.py recursive-sleep \
  --sleep-cycle meta/sleep/exp280-boundary-v1
```

Reproduction: `experiment_bundles/exp280/manifest.json` pins the executed commit and raw
source hashes. Scientific source remains unchanged after the source-matched pilot.
The incremental refitter replaces cached weighted responsibilities rather than adding
them repeatedly; hand-derived tests pin counts and actual fitting/monitoring work.

Claudeception evaluation for C1: cached responsibility replacement is verified by the
helper's hand-derived tests and recorded in its docstring; acquisition-as-outcome already
belongs to VALIDATION. Those durable guards cover the reusable knowledge, so no duplicate
global skill was created and no negative toy result was promoted into a universal rule.
