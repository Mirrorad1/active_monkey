# Recursive Sleep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an advisory lab sleep brief and run one once-weighted creature replay screen.

**Architecture:** Share a canonical sleep receipt between a source-excerpt lab adapter and a
NumPy offline-inference adapter for the existing affective dyad. Keep lab advice, observed
data, synthetic proposals, and biological inheritance distinct. No hosted-model training.

**Tech Stack:** Python, standard library, existing NumPy/JAX/pymdp and safetensors; pytest.

**Spec:** `docs/superpowers/specs/2026-10-27-recursive-sleep-design.md` (approved 2026-10-27).

## Global Constraints

- Do not edit FROZEN paths or unrelated working changes.
- Raw experiment records and biography remain append-only.
- `meta_monkey` remains advisory; reviewed briefs do not choose or grade experiments.
- Every real encounter contributes evidence once; decay follows lived time.
- Cap lab retrieval at eight source bundles and the active brief at 1,500 tokens.
- No dependencies added. Use `uv run --python .venv` from the repository root.
- Experiments require preregistration, raw artifacts, independent blinded verification, and
  atomic script/output/log/site commit. Later ecology/retention/prospective-lab rungs are out
  of this implementation slice.

## Review Focus

1. Changed sources after review must invalidate the attached brief (Task 1).
2. Missing negatives or a tiny budget must never silently turn a negative into a positive (Task 1).
3. Generated proposals must never be accepted as observed evidence (Task 1).
4. Repeated replay must preserve the total evidence weight and leave live belief/RNG untouched (Task 2).
5. Smoke/partial runs must not receive the confirmatory verdict or overwrite prior raw results (Task 3).

## Execution decisions

Execute natively in the current chat under the user's approval; one fresh final reviewer.
The managed worktree was write-blocked by the sandbox, so use the existing checkout on
`codex/recursive-sleep-framework`; stage explicit paths only. Baseline: 46 focused tests pass.

Lab summaries begin as human/assistant-curated packets whose recorded findings are exact
excerpts, with SHA-256 source refs. Candidate hypotheses/rehearsals are separately labeled.
No LLM API calls or automatic free-text truth adjudication. Independent review is recorded
against the packet/receipt hashes before compose may attach the brief.

Use a conservative 1,500-UTF8-byte ceiling for the complete brief (not a claimed tokenizer
measurement), which is stricter than the design's 1,500-token cap for byte-tokenized models.
Fail on a packet that cannot fit; never silently drop evidence to satisfy the budget.

For C0, compare joint batch refitting against chronological refitting on the same 100 observed
triples. Both use 20 passes, 2,000 record-posterior evaluations, the same prior and decay, and
rebuild once-weighted counts. Joint refitting is regularized latent-mixture EM; chronological
refitting restarts the same sequential pass from the prior each time. This tests global
re-estimation vs ordinary chronological reprocessing, not Google's MoE knowledge seeding.

### Task 1: Lab receipts, source fidelity and reviewed prompt attachment

**Files:** Create `active_loop/sleep.py`, `meta_monkey/sleep.py`,
`tests/test_lab_sleep.py`, `meta/sleep/README.md`; modify `loop/compose.py` and
`tests/test_compose.py`. Author `meta/sleep/initial/packet.json` after the code is green.

**Interfaces:** `canonical_hash(value) -> str`, `make_receipt(scope, before, after,
sources, budget, metrics, decision) -> dict`; `build_cycle(root: Path, packet: dict,
max_bytes: int = 1500) -> tuple[str, dict]`; `write_cycle(root, packet, output) -> None`;
`review_cycle(root, output, review) -> None`; `load_reviewed_brief(root, directory) -> str`.
CLI: `python -m meta_monkey.sleep build --packet PATH --out meta/sleep/RUN`,
`review --cycle PATH --review PATH`, `inspect --cycle PATH`.
Compose adds optional `--sleep-cycle PATH`; no option means identical old prompt.

- [ ] Write meaningful tests: negative excerpt preserved; unsupported/missing/escaping/symlink
  source rejected; conflicting findings both retained; source hashes checked; proposal cannot
  be classified as observed; eight-bundle/byte cap enforced; no source writes; reviewed brief
  load fails after source/brief/packet mutation or negative review; CLI refuses overwrites.
- [ ] Run `uv run --python .venv python -m pytest tests/test_lab_sleep.py -q`; expect missing module.
- [ ] Implement canonical receipts and source-excerpt assembly. Bundle IDs must resolve to
  existing committed records; all claimed output refs resolve to files. Review binds the
  content hash, not a mutable filename. Output restricted to `meta/sleep/`.
- [ ] Add attachment tests and CLI integration; run focused suite including existing compose
  and meta-memory tests. Default compose/preflight behavior remains unchanged.
- [ ] Commit implementation and tests separately from any scientific result.

### Task 2: Once-weighted offline refitting instrument

**Files:** Create `active_loop/sleep_replay.py`, `tests/test_sleep_replay.py`.

**Interfaces:** immutable `Encounter(event_id, t, code, response, outcome)`;
`ReplayModel(a0, a1, p0, p1, prior)` with copied arrays; `refit(anchor, start,
events, *, passes=20, mode='joint', decay=.999, learning_rate=4.) -> (model, receipt)`.
`objective(model, anchor, events, decay, learning_rate) -> float`.

Model: `P(code,outcome|response)=sum_z prior[z]*A0[code,z,response]*A1[outcome,z,response]`.
Joint E-step: posterior proportional to that product. M-step: counts = decayed immutable
anchor counts + sum of unique responsibilities weighted by `learning_rate*decay**(T-1-t)`.
Regularized log objective includes decayed anchor counts times log probabilities. Recompute
all counts on every pass; retain no additive replay evidence. Chronological control starts
from the same anchor each pass, sequentially re-estimating and adding each event once.

- [ ] Test hand-computed K=1 counts and physical-time decay, duplicate IDs/time rejection,
  invalid shapes/indices/nonfinite values, conserved evidence mass across pass counts,
  joint objective monotonicity, source arrays unchanged, deterministic independent repeats,
  empty/zero-pass sleep no-op, and chronological repeated-pass equivalence.
- [ ] Run tests RED, implement, run GREEN. No changes to `DirectHeadAgent` or frozen scorers.
- [ ] Check export via existing `AgentState`/safetensors and shared receipt; commit helper/tests.

### Task 3: One preregistered dyad experiment and lab-cycle example

**Files:** Create `experiments/exp279_sleep_replay.py` (verify ID at launch),
`experiments/outputs/exp279*`, `tests/test_sleep_experiment.py`,
`loop/directions/recursive-sleep.md`; append `EXPERIMENTS.md`, `loop/IDEAS.md`, `RESUME.md`,
and `site/data/experiments-data.js`. Add an evidence-backed bundle and lab packet.

**Interfaces:** `run_seed(seed, turns=100, passes=20) -> dict`,
`assess(rows, *, confirmatory: bool) -> dict`; command `--smoke` (excluded seed 9299),
default confirmatory seeds 9300..9315, `--workers 2`. All arms reported.

- [ ] Write preregistration in runner docstring BEFORE scientific execution. Primary C0 pass:
  joint-refit minus chronological-refit `correct_select` mean >= 1/6 and positive on >=12/16
  seeds; joint minus idle mean >=1/6 and positive on >=12/16. Fixed evaluation gamma=4, no
  learning during evaluation. All instrument invariants must pass. Otherwise NEGATIVE;
  instrument failures INVALID. No claim of reliable short-session rescue from a small gain.
- [ ] Compare idle; chronological20; joint20; broken-association joint20; constant-response;
  longer-wake300 extra-data reference. Log held-out expected and sampled POS on 120 balanced
  codes, csel, raw observations, all input/output model tensors, live before/after belief/RNG
  hashes, evidence mass, objective curves, pass/work/runtime counts. Correct mappings live
  only in harness/evaluator. Sleep doesn't consume new world observations.
- [ ] Test verdict conjunctions/counts/missing rows, smoke non-promotion and state preservation.
  RED -> implement -> GREEN. Parallelize independent seeds with bounded processes and JAX
  thread/RSS budget; smoke-time before full batch, no hyperparameter tuning on smoke results.
- [ ] Independently review scientific preregistration/implementation before the full run.
- [ ] Run full batch once. Dispatch a blinded verifier with docstring and raw output only;
  recompute all scientific conjunctions. If negative, stop the experimental ladder and log it.
- [ ] Append honest log/site entry plus snapshot refs and seed-level evidence. Run rubric,
  focused tests and full fast suite; distinguish existing baseline failures. Commit atomic
  experiment record and collect passive meta episode. No persistent spine reset/advance:
  this is a seeded dyad cohort, not a biography experiment.
- [ ] Build a lab brief from committed evidence, independently review its fidelity, record
  review, and demonstrate optional prompt attachment. Report instrument validation separately
  from any untested claim of better research practice.
- [ ] Fresh whole-branch review; fix important findings with regression tests; final verification.

## Self-review

The initial slice covers L0 and C0; C1–C3 and L1 deliberately remain gated follow-ups.
Tasks 1/2 share only receipt construction; Task 3 consumes both. Generated material never
crosses into observed evidence. No gate depends on a desired scientific effect.
