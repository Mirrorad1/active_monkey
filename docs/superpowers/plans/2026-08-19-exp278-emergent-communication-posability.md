# Exp 278 Emergent-Communication Posability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether a moving-resource scout/forager task genuinely requires current private information and gives a costed gifted-oracle signal a large actionable advantage before attempting emergent communication.

**Architecture:** Add a small substrate-independent benchmark that generates either a never-repeat moving resource path or a static positive-control path and evaluates five policies on the same path: random solo search, literal trail following, the strongest history-only policy, a shuffled-current-signal control, and a gifted current-state oracle. A numbered experiment runs eight independent fresh seeds in parallel, prints every seed and each predeclared conjunct, and stops at a `POSABLE`/`CANT_POSE` gate; no signalling learner or emergent mapping is built in this rung.

**Tech Stack:** Python 3.12, NumPy, dataclasses, `concurrent.futures.ProcessPoolExecutor`, pytest.

**Spec:** `loop/directions/emergent-communication.md`

## Global Constraints

- This is Loop B Exp 278 and must follow `loop/PROTOCOL.md`, `loop/VALIDATION.md`, `loop/LESSONS.md`, and `loop/ROUTING.md`.
- The scout's current resource observation and the oracle token-to-site mapping are PROVIDED diagnostics; this experiment makes no emergence claim.
- The moving resource must never remain at its previous site; the strongest history-only receiver may exploit that transition law by excluding the previous site.
- Signal cost is strictly positive (`0.05`) and subtracted from oracle reward.
- Use eight fresh seeds `401..408`, `8` sites, and `2,000` rounds per seed.
- Independent seed runs execute in parallel; a serial-vs-parallel equality test guards determinism.
- The verdict is `POSABLE` only when all four predeclared gates pass: moving baselines remain at the analytic history-only ceiling, the oracle has a large advantage, the costed oracle remains viable, and trail following succeeds in the static positive control.
- Preserve unrelated working-tree changes and never edit FROZEN paths.

---

### Task 1: Moving-resource posability harness and numbered runner

**Files:**
- Create: `active_loop/benchmarks/emergent_comm_posability.py`
- Create: `tests/test_emergent_comm_posability.py`
- Create: `experiments/exp278_emergent_comm_posability.py`

**Interfaces:**
- Produces: `PosabilityConfig(n_sites: int = 8, rounds: int = 2000, signal_cost: float = 0.05)`.
- Produces: `run_seed(seed: int, config: PosabilityConfig, *, static: bool = False) -> SeedResult`.
- Produces: `run_batch(seeds: tuple[int, ...], config: PosabilityConfig, *, parallel: bool = True, max_workers: int | None = None) -> tuple[SeedResult, ...]`, ordered by seed.
- Produces: `grade_posability(rows: tuple[SeedResult, ...], config: PosabilityConfig) -> GateResult`.
- `SeedResult` exposes `seed`, `chance`, `history_ceiling`, `solo_success`, `trail_success`, `history_optimal_success`, `shuffled_signal_success`, `oracle_success`, `oracle_net_reward`, and `static_trail_success`.
- `GateResult` exposes per-conjunct booleans and `verdict` in `{"POSABLE", "CANT_POSE"}`.

- [ ] **Step 1: Write failing behavioral tests**

Add tests with hand-derived expectations:

```python
def test_moving_resource_never_repeats_and_is_deterministic():
    cfg = PosabilityConfig(n_sites=4, rounds=40, signal_cost=0.05)
    a = resource_path(17, cfg, static=False)
    b = resource_path(17, cfg, static=False)
    assert a == b
    assert len(a) == 40
    assert all(left != right for left, right in zip(a, a[1:]))

def test_oracle_is_perfect_but_pays_the_declared_cost():
    cfg = PosabilityConfig(n_sites=8, rounds=200, signal_cost=0.05)
    row = run_seed(401, cfg)
    assert row.oracle_success == 1.0
    assert row.oracle_net_reward == 0.95

def test_static_positive_control_proves_trail_policy_is_live():
    cfg = PosabilityConfig(n_sites=8, rounds=200, signal_cost=0.05)
    row = run_seed(401, cfg)
    assert row.static_trail_success >= 199 / 200

def test_predeclared_gate_accepts_clear_posability_rows():
    cfg = PosabilityConfig(n_sites=8, rounds=2000, signal_cost=0.05)
    rows = tuple(run_seed(seed, cfg) for seed in range(401, 409))
    gate = grade_posability(rows, cfg)
    assert gate.baselines_fail
    assert gate.oracle_advantage
    assert gate.costed_oracle_viable
    assert gate.static_control_live
    assert gate.verdict == "POSABLE"

def test_parallel_batch_is_identical_to_serial():
    cfg = PosabilityConfig(n_sites=8, rounds=200, signal_cost=0.05)
    seeds = (401, 402, 403)
    assert run_batch(seeds, cfg, parallel=False) == run_batch(seeds, cfg, parallel=True, max_workers=2)
```

Each test names the production break it catches: repeated resources would reopen stigmergy; an uncosted oracle would violate the incentive test; a dead trail policy would fake the baseline failure; weakened conjunct logic would overclaim posability; shared RNG/order dependence would invalidate parallel execution.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_emergent_comm_posability.py -q
```

Expected: collection fails with `ModuleNotFoundError: active_loop.benchmarks.emergent_comm_posability`.

- [ ] **Step 3: Implement the minimal harness**

Implementation rules:

```python
HISTORY_TOLERANCE = 0.03
ORACLE_ADVANTAGE_FLOOR = 0.70
ORACLE_NET_FLOOR = 0.80
STATIC_TRAIL_FLOOR = 0.95
```

Generate the moving path by choosing the first site uniformly and every later site uniformly from the other `n_sites - 1` sites. Evaluate every arm against that same path. `trail` chooses the previous resource site; `history_optimal` samples uniformly from every site except the previous resource; `solo` samples uniformly from all sites; `shuffled_signal` delivers an independently permuted current token; `oracle` delivers and decodes the gifted identity mapping. Use RNG streams derived solely from `(seed, arm)` so arm evaluation order cannot alter results. The static trail control repeats one resource site and exposes the previous resource only after each completed round.

`grade_posability` must require all seeds to satisfy:

```python
max(row.solo_success, row.trail_success,
    row.history_optimal_success, row.shuffled_signal_success)
    <= row.history_ceiling + HISTORY_TOLERANCE

row.oracle_net_reward - max_baseline >= ORACLE_ADVANTAGE_FLOOR
row.oracle_net_reward >= ORACLE_NET_FLOOR
row.static_trail_success >= STATIC_TRAIL_FLOOR
```

and must return `CANT_POSE` if any conjunct fails.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the exact command from Step 2. Expected: all focused tests pass.

- [ ] **Step 5: Add the Exp 278 runner with its binding predeclaration**

The module docstring must name: `HYPOTHESIS`, property-level `PREDICTION`, `FALSIFIER`, the provided-vs-earned ledger, and the non-emergence caveat. Before the full batch, time a 200-round smoke run, project `8 * 2,000 * 5 = 80,000` policy-round evaluations, and abort if projected wall time exceeds 30 seconds. Print config, runtime preflight, one parseable row per seed, aggregate means, each gate conjunct, the script claim, and a JSON block. Do not print an emergent-communication claim.

- [ ] **Step 6: Run focused tests again**

Run:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_emergent_comm_posability.py tests/test_comm_v0.py -q
```

Expected: all tests pass and the older Comm v0 benchmark remains unchanged.

### Task 2: Execute, independently grade, and publish the research result

**Files:**
- Create: `experiments/outputs/exp278.txt`
- Modify append-only: `EXPERIMENTS.md`
- Modify: `site/data/experiments-data.js`
- Modify: `loop/directions/emergent-communication.md`
- Modify: `RESUME.md`

**Interfaces:**
- Consumes: the Task 1 runner's module docstring and raw output.
- Produces: one atomic Exp 278 commit containing source, tests, raw output, log entry, site entry, direction status, and resume state.

- [ ] **Step 1: Run Exp 278 and save the exact raw output**

Run from the isolated repo root with the established interpreter:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python experiments/exp278_emergent_comm_posability.py
```

Capture exactly that stdout in `experiments/outputs/exp278.txt`; if the script is patched and rerun, replace the output with the final run before quoting numbers.

- [ ] **Step 2: Blindly verify the result**

Dispatch a fresh verifier with only the predeclared docstring and `experiments/outputs/exp278.txt`. It must ignore the printed claim, recompute all four conjuncts, and return `POSITIVE`, `NEGATIVE`, or `MIXED` with the conjunct mapping.

- [ ] **Step 3: Append the honest experiment entry and site card**

The controller writes `Plain / Setup / Result / Implication / Honest caveat / Verdict / Verifier / Next`. If all four gates pass, the result is `POSITIVE-SINGLE / NEW INSIGHT` for task posability only; it is not a communication breakthrough. If any gate fails, grade the strictest corresponding `NEGATIVE` or `MIXED` result. Add the matching curated `n:278` site entry with `plain` and `trace` paths.

- [ ] **Step 4: Update durable state without rewriting history**

Update the emergent-communication direction status and `RESUME.md` to record the Exp 278 gate and the licensed next rung only if the gate passed. Do not edit prior `EXPERIMENTS.md` entries.

- [ ] **Step 5: Run mechanical and broad verification**

Run:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python loop/check_iteration.py 278
/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest -q
```

If the full suite's only failure is the known multiprocessing semaphore sandbox denial, rerun the same pytest command outside the sandbox. Confirm all warnings from `check_iteration.py` are derived-not-stale before committing.

- [ ] **Step 6: Commit the atomic experiment and collect passive process memory**

Stage only the files named by this plan and commit:

```bash
git commit -m "exp278: test emergent-communication task posability"
```

Then run:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python -m meta_monkey.collect_iteration --latest --write
```

Commit the resulting passive episode separately if it changed a tracked file. Never amend the atomic scientific commit with unrelated workspace changes.
