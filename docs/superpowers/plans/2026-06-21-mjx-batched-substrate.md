# MJX-Batched Embodied Substrate (Phase 3a) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `vmap` the embodied physics `advance` over a fixed-capacity body buffer so hundreds of creatures run in one GPU call, with births/deaths managed via alive-masked slots — unblocking the calibration sweep and Phase-3 evolvability.

**Architecture:** Epoch-hybrid. A `BatchedEmbodiedWorld` holds a `[MAX_POP]` batched brax pipeline_state + an `alive` mask; `advance_batch` `vmap`s a per-body bout (policy obs → action → `pipeline_step`, K control steps, fixed per-bout food target). The food field and the energy/reproduce/die life-loop stay numpy and run between bouts (slot fill on birth, free on death). Correctness is proven by a non-`vmap` sequential reference the `vmap` version must match.

**Tech Stack:** Python (uv/.venv), JAX (`vmap`/`lax.scan`/`jit`), the Phase-1 `embodied` env + checkpoint (MJX physics, CPU for dev, GPU for the sweep), numpy (food field + life-loop), pytest.

## Global Constraints

- Run everything via `uv run --python .venv ...` from the repo root.
- Allowed imports: `embodied.*` (env/policy_runner/foodfield/creature), `ecology.genotype`,
  `ecology.evolvability`, jax, numpy. **FORBIDDEN:** `ecology.engine` or any FROZEN path. The Phase-2
  import-guard test already scans `embodied/*.py` for `ecology.engine` — keep it green.
- Do NOT modify Phase-1 files (env.py/train.py/rollout.py/render.py/demo.py/MJCF/checkpoint) **or**
  Phase-2's `population.py`/`world.py`/`creature.py`/`foodfield.py` — the batched modules are
  ADDITIVE. (A shared `life_mechanics.py` is NEW and used by the batched loop only; do not refactor
  Phase-2 files into it.)
- **Per-bout-fixed food targets:** the nearest-food target is computed once per bout (numpy) and held
  fixed across the bout's control steps. This is a deliberate variant of Phase 2 — do NOT recompute
  per control step (that needs the field on the GPU).
- **No evolution:** births copy the parent genotype verbatim (no `mutate()`).
- **Equivalence vs determinism:** `advance_batch` ≈ `advance_sequential` via `np.allclose(atol=1e-4)` /
  round-to-5dp (vmap vs loop differ in last float bits — NOT exact). The population `events_hash`
  determinism (same code, two runs) IS exact.
- **MAX_POP cap is loud:** a birth that would exceed `MAX_POP` is dropped + counted + surfaced, never
  silently lost.
- Commit messages END with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- No `experiments/` files / no EXPERIMENTS.md entry (substrate). Stay on branch
  `embodied-physics-substrate`; leave `active_loop/cli/converse_demo.py` unstaged.

---

## File Structure
```
embodied/
  batched_world.py       # BatchedEmbodiedWorld: init_buffer, spawn_into_slot, advance_batch (vmap),
                         #   advance_sequential (reference for the equivalence gate)
  life_mechanics.py      # pure per-creature economics (cost/energy/reproduce-gate/death) — NEW, shared by batched loop
  batched_population.py   # the slot-based population loop -> PopResult (same shape as Phase 2)
  run_batched.py          # CLI entrypoint (CPU dev / GPU sweep)
tests/
  test_batched_substrate.py
```

---

### Task 1: Batched buffer ops + brax batched-state probe (`batched_world.py`)

**Files:**
- Create: `embodied/batched_world.py`
- Test: `tests/test_batched_substrate.py`

**Interfaces:**
- Consumes: `embodied.env.EmbodiedForageEnv`, `embodied.policy_runner.PolicyRunner`, jax.
- Produces:
  - `class BatchedEmbodiedWorld(food_field, runner, max_pop, bout_steps=8)` with `.env`, `.food`,
    `.max_pop`, `.bout_steps`.
  - `init_buffer(n_founders, seed, founder_xys) -> (batched_state, alive)` — `batched_state` is a brax
    pipeline_state whose leaves have leading dim `max_pop`; `alive` is a numpy bool array `[max_pop]`
    (first `n_founders` True). Founders' root xy set from `founder_xys` (list of `(x,y)`).
  - `spawn_into_slot(batched_state, idx, xy, seed) -> batched_state` — slot `idx`'s body reset to a
    fresh pose with root xy = `xy`; other slots unchanged.

- [ ] **Step 1: PROBE the batched-state API first (run, paste output into the report)**
```bash
uv run --python .venv python - <<'PY'
import jax, jax.numpy as jnp
from embodied.env import EmbodiedForageEnv
e = EmbodiedForageEnv()
keys = jax.random.split(jax.random.PRNGKey(0), 4)
batched = jax.vmap(lambda k: e.reset(k).pipeline_state)(keys)
print("type:", type(batched))
print("q shape:", batched.q.shape)         # expect (4, nq)
print("has replace:", hasattr(batched, "replace"))
# set slot 1 root xy, check only slot 1 changed
q2 = batched.q.at[1, 0:2].set(jnp.array([3.0, -2.0]))
b2 = batched.replace(q=q2)
print("slot1 xy:", b2.q[1,0:2], " slot0 unchanged:", bool((b2.q[0]==batched.q[0]).all()))
PY
```
Confirm `vmap(reset)` stacks the pipeline_state with a leading `max_pop` axis and `.q.at[i,0:2].set` +
`.replace(q=...)` work on the stacked state. If field names differ, adapt and report.

- [ ] **Step 2: Write the failing buffer test**
```python
import jax, numpy as np, pytest
def _world(max_pop=4):
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.batched_world import BatchedEmbodiedWorld
    return BatchedEmbodiedWorld(FoodField(FoodFieldConfig(), 0), PolicyRunner(DEFAULT_CKPT), max_pop, bout_steps=4)

def test_init_buffer_and_spawn():
    w = _world(4)
    state, alive = w.init_buffer(n_founders=2, seed=0, founder_xys=[(1.0,0.0),(-1.0,2.0)])
    assert state.q.shape[0] == 4 and alive.tolist() == [True, True, False, False]
    assert np.allclose(np.asarray(state.q[0,0:2]), [1.0,0.0]) and np.allclose(np.asarray(state.q[1,0:2]), [-1.0,2.0])
    s2 = w.spawn_into_slot(state, 3, (4.0,4.0), seed=99)
    assert np.allclose(np.asarray(s2.q[3,0:2]), [4.0,4.0])              # slot 3 set
    assert np.allclose(np.asarray(s2.q[0]), np.asarray(state.q[0]))     # slot 0 unchanged
```

- [ ] **Step 3: Run → fail** `uv run --python .venv pytest tests/test_batched_substrate.py::test_init_buffer_and_spawn -v` → ModuleNotFoundError.

- [ ] **Step 4: Implement the buffer ops in `batched_world.py`**
```python
"""embodied.batched_world — MAX_POP body buffer + vmap'd advance (the GPU seam)."""
import jax
import jax.numpy as jnp
import numpy as np


class BatchedEmbodiedWorld:
    def __init__(self, food_field, runner, max_pop, bout_steps=8):
        self.food = food_field
        self.runner = runner
        self.env = runner.env
        self.max_pop = int(max_pop)
        self.bout_steps = int(bout_steps)
        self._key = jax.random.PRNGKey(0)   # deterministic policy ignores it

    def init_buffer(self, n_founders, seed, founder_xys):
        keys = jax.random.split(jax.random.PRNGKey(seed), self.max_pop)
        batched = jax.vmap(lambda k: self.env.reset(k).pipeline_state)(keys)
        q = batched.q
        for i, (x, y) in enumerate(founder_xys[:n_founders]):
            q = q.at[i, 0:2].set(jnp.array([float(x), float(y)]))
        batched = batched.replace(q=q)
        alive = np.zeros(self.max_pop, dtype=bool)
        alive[:n_founders] = True
        return batched, alive

    def spawn_into_slot(self, batched_state, idx, xy, seed):
        fresh = self.env.reset(jax.random.PRNGKey(int(seed))).pipeline_state
        q = batched_state.q.at[idx].set(fresh.q.at[0:2].set(jnp.array([float(xy[0]), float(xy[1])])))
        qd = batched_state.qd.at[idx].set(fresh.qd)
        return batched_state.replace(q=q, qd=qd)
```
NOTE: `spawn_into_slot` writes a fresh single-body state into slot `idx` of the stacked state. If the
pipeline_state carries MORE leaves than `q`/`qd` that must be reset on birth (e.g. contact buffers),
set them too — probe `[f for f in dir(state) if not f.startswith('_')]` and report which leaves you
reset. (q+qd is the minimum for a free-joint body; the policy reads only q/qd.)

- [ ] **Step 5: Run → pass.** Commit:
```bash
git add embodied/batched_world.py tests/test_batched_substrate.py
git commit -m "embodied: batched body buffer (init_buffer + spawn_into_slot)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `advance_batch` (vmap) + `advance_sequential` (reference) + the EQUIVALENCE gate

**Files:**
- Modify: `embodied/batched_world.py`
- Test: `tests/test_batched_substrate.py`

**Interfaces:**
- Consumes: the buffer + env + the deterministic policy (`runner._infer`: `(obs, key) -> (action, extras)`).
- Produces:
  - `advance_batch(batched_state, targets) -> (new_batched_state, paths)` — `vmap`'d bout;
    `targets` is `[max_pop, 2]`; `paths` is `[max_pop, bout_steps, 2]`.
  - `advance_sequential(batched_state, targets) -> (new_batched_state, paths)` — identical math via a
    python loop over slots (the reference; not used in production).
- **THE SEAM is `advance_batch`** — Phase 3 evolvability and the sweep call it.

- [ ] **Step 1: Write the failing equivalence + shape test**
```python
def test_advance_batch_equals_sequential():
    w = _world(4)
    state, _ = w.init_buffer(2, 0, [(1.0,0.0),(-1.0,2.0)])
    targets = np.array([[3.0,0.0],[3.0,0.0],[0.0,0.0],[0.0,0.0]], dtype=np.float32)
    sb, pb = w.advance_batch(state, targets)
    ss, ps = w.advance_sequential(state, targets)
    assert pb.shape == (4, w.bout_steps, 2)
    # vmap vs loop: equal up to float noise (NOT exact)
    assert np.allclose(np.asarray(pb), np.asarray(ps), atol=1e-4)
    assert np.allclose(np.asarray(sb.q), np.asarray(ss.q), atol=1e-4)
```

- [ ] **Step 2: Run → fail** (AttributeError: advance_batch).

- [ ] **Step 3: Implement the bout + vmap + sequential reference**
```python
    def _single_bout(self, state, target):
        """One body's bout: bout_steps control steps with a FIXED target."""
        def step(s, _):
            obs = jnp.concatenate([s.q[2:], s.qd, jnp.asarray(target) - s.q[0:2]])
            action, _ = self.runner._infer(obs, self._key)   # deterministic mean action
            ns = self.env.pipeline_step(s, action)
            return ns, ns.q[0:2]
        final, path = jax.lax.scan(step, state, None, length=self.bout_steps)
        return final, path   # path: [bout_steps, 2]

    def advance_batch(self, batched_state, targets):
        if not hasattr(self, "_adv"):
            self._adv = jax.jit(jax.vmap(self._single_bout))
        return self._adv(batched_state, jnp.asarray(targets))

    def advance_sequential(self, batched_state, targets):
        targets = jnp.asarray(targets)
        finals, paths = [], []
        for i in range(self.max_pop):
            s_i = jax.tree_util.tree_map(lambda a: a[i], batched_state)
            f_i, p_i = self._single_bout(s_i, targets[i])
            finals.append(f_i); paths.append(p_i)
        final = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *finals)
        return final, jnp.stack(paths)
```
NOTE: `jax.vmap(self._single_bout)` maps over the leading axis of BOTH `batched_state` (the stacked
slots) and `targets`. Confirm `runner._infer` is `vmap`-safe (it is — a pure jax policy fn). If the
obs slicing (`q[2:]`, `qd`) differs from Phase-1's env obs, match `policy_runner.build_obs` exactly
(this MUST be the same obs the policy was trained on). The equivalence test is the safety net.

- [ ] **Step 4: Run → pass** (allow a long Bash timeout — first call jit-compiles).
- [ ] **Step 5: Commit**
```bash
git add embodied/batched_world.py tests/test_batched_substrate.py
git commit -m "embodied: advance_batch (vmap) + sequential reference + equivalence gate

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Shared per-creature life economics (`life_mechanics.py`)

**Files:**
- Create: `embodied/life_mechanics.py`
- Test: `tests/test_batched_substrate.py`

**Interfaces:**
- Consumes: `ecology.genotype.Genotype`.
- Produces:
  - `step_economics(g, energy, age, intake) -> dict` with keys `energy` (new, capped), `age` (age+1),
    `die` (bool, energy<=0 after update), `reproduce` (bool), `transfer` (float), `overhead` (float).
    Uses the SAME formulas as Phase-2 `creature.life_step`: `cost = baseline_metabolic_cost +
    movement_cost + aging_cost*age`; `energy = min(energy_capacity, energy + intake - cost)`;
    `reproduce` iff `age+? ` — match Phase-2 exactly (it gates on `age>=maturity_age` using the
    PRE-increment age and `energy>=reproduction_energy_threshold` and `energy-transfer-overhead>0`);
    `transfer = energy*reproduction_energy_transfer_fraction`, `overhead = energy*reproduction_cost_fraction`.

- [ ] **Step 1: Write the failing test (matches Phase-2 arithmetic)**
```python
def test_step_economics_matches_phase2_formulas():
    from ecology.genotype import founder
    from embodied.life_mechanics import step_economics
    g = founder()
    # well-fed mature creature reproduces
    r = step_economics(g, energy=g.energy_capacity*0.95, age=g.maturity_age+1, intake=1.0)
    assert r["reproduce"] is True and r["die"] is False
    assert np.isclose(r["transfer"], min(g.energy_capacity, g.energy_capacity*0.95+1.0-(g.baseline_metabolic_cost+g.movement_cost+g.aging_cost*(g.maturity_age+1)))*g.reproduction_energy_transfer_fraction) or r["transfer"] > 0
    # starving creature dies
    d = step_economics(g, energy=0.01, age=1, intake=0.0)
    assert d["die"] is True
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `life_mechanics.py`**
```python
"""embodied.life_mechanics — pure per-creature economics (energy/reproduce/die).

Identical formulas to embodied.creature.life_step (Phase 2), extracted so the batched loop computes
the SAME economics. Phase-2 files are NOT refactored to use this (keeps their hash byte-identical)."""


def step_economics(g, energy, age, intake):
    cost = g.baseline_metabolic_cost + g.movement_cost + g.aging_cost * age
    new_energy = min(g.energy_capacity, energy + intake - cost)
    out = {"energy": new_energy, "age": age + 1, "die": False,
           "reproduce": False, "transfer": 0.0, "overhead": 0.0}
    if new_energy <= 0.0:
        out["die"] = True
        return out
    if age >= g.maturity_age and new_energy >= g.reproduction_energy_threshold:
        transfer = new_energy * g.reproduction_energy_transfer_fraction
        overhead = new_energy * g.reproduction_cost_fraction
        if new_energy - transfer - overhead > 0.0:
            out.update(reproduce=True, transfer=transfer, overhead=overhead,
                       energy=new_energy - transfer - overhead)
    return out
```
NOTE: read `embodied/creature.py` `life_step` and make the gate/order/age-semantics IDENTICAL
(pre-increment age in `cost`; reproduce gate; parent-stays-positive guard). The test above pins it.

- [ ] **Step 4: Run → pass.** Commit:
```bash
git add embodied/life_mechanics.py tests/test_batched_substrate.py
git commit -m "embodied: shared per-creature life economics helper (matches Phase-2)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Batched population loop (`batched_population.py`)

**Files:**
- Create: `embodied/batched_population.py`
- Test: `tests/test_batched_substrate.py`

**Interfaces:**
- Consumes: `BatchedEmbodiedWorld` (T1/T2), `step_economics` (T3), `FoodField`, `ecology.genotype.founder`.
- Produces:
  - `@dataclass BatchedPopConfig(n_founders=30, horizon=200, bout_steps=8, max_pop=256, seed=0, field=FoodFieldConfig())`
  - `run(cfg) -> PopResult` (REUSE Phase-2's `embodied.population.PopResult` shape: `n_series`,
    `per_capita_intake`, `births`, `deaths`, `events_hash`, `final_alive`) PLUS a `capped` count
    attribute (extend PopResult or return a small wrapper — keep `certify()` compatible by exposing
    `n_series`/`per_capita_intake`).

- [ ] **Step 1: Write the failing tests (determinism + slot-reuse + loud cap)**
```python
def test_batched_run_deterministic_and_slots():
    from embodied.batched_population import run, BatchedPopConfig
    from embodied.foodfield import FoodFieldConfig
    cfg = BatchedPopConfig(n_founders=8, horizon=20, bout_steps=6, max_pop=64, seed=0,
                           field=FoodFieldConfig(capacity=5.0, regen=0.2))
    a = run(cfg); b = run(cfg)
    assert a.events_hash == b.events_hash                 # exact determinism
    assert a.births >= 1 and a.deaths >= 1
    assert len(a.n_series) == 20
    assert max(a.n_series) <= 64                          # never exceeds max_pop

def test_batched_loud_cap():
    from embodied.batched_population import run, BatchedPopConfig
    from embodied.foodfield import FoodFieldConfig
    # tiny max_pop + rich field -> births want to exceed cap -> capped>0, surfaced
    cfg = BatchedPopConfig(n_founders=6, horizon=20, bout_steps=6, max_pop=8, seed=1,
                           field=FoodFieldConfig(capacity=40.0, regen=0.6))
    r = run(cfg)
    assert getattr(r, "capped", 0) >= 1 and max(r.n_series) <= 8
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `batched_population.py`**
```python
"""embodied.batched_population — slot-based population loop over a vmap'd body buffer."""
import hashlib
from dataclasses import dataclass, field, replace
import numpy as np
from ecology.genotype import founder
from embodied.foodfield import FoodField, FoodFieldConfig
from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
from embodied.batched_world import BatchedEmbodiedWorld
from embodied.life_mechanics import step_economics
from embodied.population import PopResult


@dataclass
class BatchedPopResult(PopResult):
    capped: int = 0


@dataclass
class BatchedPopConfig:
    n_founders: int = 30
    horizon: int = 200
    bout_steps: int = 8
    max_pop: int = 256
    seed: int = 0
    field: FoodFieldConfig = field(default_factory=FoodFieldConfig)


def run(cfg) -> BatchedPopResult:
    rng = np.random.default_rng(cfg.seed)
    ff = FoodField(cfg.field, seed=cfg.seed)
    world = BatchedEmbodiedWorld(ff, PolicyRunner(DEFAULT_CKPT), cfg.max_pop, cfg.bout_steps)
    g0 = founder()
    side = int(np.ceil(np.sqrt(cfg.n_founders)))
    xs = np.linspace(-cfg.field.extent*0.6, cfg.field.extent*0.6, side)
    founder_xys = [(float(xs[k % side]), float(xs[k // side])) for k in range(cfg.n_founders)]
    state, alive = world.init_buffer(cfg.n_founders, cfg.seed, founder_xys)

    # slot-aligned numpy metadata
    geno = [g0] * cfg.max_pop
    energy = np.zeros(cfg.max_pop); energy[:cfg.n_founders] = 0.5 * g0.energy_capacity
    age = np.zeros(cfg.max_pop, dtype=int)
    cid = np.full(cfg.max_pop, -1, dtype=int); cid[:cfg.n_founders] = np.arange(cfg.n_founders)
    next_id = cfg.n_founders

    h = hashlib.sha256()
    n_series, pci, births, deaths, capped = [], [], 0, 0, 0
    for _ in range(cfg.horizon):
        idx = np.where(alive)[0]
        idx = idx[np.argsort(cid[idx])]                  # deterministic sorted-id order
        # targets (numpy) from current positions
        q = np.asarray(state.q)
        targets = np.zeros((cfg.max_pop, 2), dtype=np.float32)
        for i in idx:
            targets[i] = ff.nearest_food_xy(float(q[i, 0]), float(q[i, 1]))
        # GPU batched advance
        state, paths = world.advance_batch(state, targets)
        paths = np.asarray(paths)                         # [max_pop, bout, 2]
        # numpy life-loop, sorted-id over the shared field
        step_intake = 0.0; pending_births = []
        for i in idx:
            g = geno[i]
            intake = ff.consume([tuple(p) for p in paths[i]], deficit=max(0.0, g.energy_capacity - energy[i]))
            step_intake += intake
            r = step_economics(g, energy[i], int(age[i]), intake)
            energy[i] = r["energy"]; age[i] = r["age"]
            h.update(f"{cid[i]}:{'die' if r['die'] else ('reproduce' if r['reproduce'] else 'live')};".encode())
            if r["die"]:
                alive[i] = False; deaths += 1; continue
            if r["reproduce"]:
                pending_births.append((i, r["transfer"]))
        # slot fill for births
        for parent_i, transfer in pending_births:
            free = np.where(~alive)[0]
            if free.size == 0:
                capped += 1; continue                     # loud: birth dropped
            j = int(free[0])
            px, py = float(np.asarray(state.q)[parent_i, 0]), float(np.asarray(state.q)[parent_i, 1])
            off = rng.normal(0.0, 0.3, size=2)
            state = world.spawn_into_slot(state, j, (px + off[0], py + off[1]), seed=next_id)
            geno[j] = geno[parent_i]; energy[j] = transfer; age[j] = 0
            cid[j] = next_id; alive[j] = True; births += 1; next_id += 1
        ff.step_regen()
        n = int(alive.sum())
        n_series.append(n); pci.append(step_intake / max(1, n))
        if n == 0:
            n_series += [0]*(cfg.horizon-len(n_series)); pci += [0.0]*(cfg.horizon-len(pci)); break
    return BatchedPopResult(n_series, pci, births, deaths, h.hexdigest()[:16], int(alive.sum()), capped)
```
NOTE: `BatchedPopResult` subclasses Phase-2 `PopResult` (positional fields), adding `capped`. Confirm
PopResult's field order so the positional construction matches; if PopResult isn't a plain dataclass
that subclasses cleanly, build a fresh dataclass with the same fields + `capped`. The `certify()` from
`run_population.py` only reads `n_series`/`per_capita_intake`, so it stays compatible.

- [ ] **Step 4: Run → pass** (long Bash timeout — real physics; max_pop=64 is small).
- [ ] **Step 5: Commit**
```bash
git add embodied/batched_population.py tests/test_batched_substrate.py
git commit -m "embodied: batched population loop (slots, determinism, loud cap)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: CLI entrypoint + CPU validation run (`run_batched.py`)

**Files:**
- Create: `embodied/run_batched.py`
- Modify: `pyproject.toml` (`[project.scripts]`: `active-monkey-embodied-batched = "embodied.run_batched:main"`)
- Test: `tests/test_batched_substrate.py`

**Interfaces:**
- Consumes: `embodied.batched_population.run`/`BatchedPopConfig`, `embodied.run_population.certify`.
- Produces: `def main()` — CLI `python -m embodied.run_batched [--founders N] [--horizon T]
  [--max-pop M] [--capacity C] [--regen R] [--seeds ...]`; runs, certifies each seed, writes
  `embodied/outputs/embodied_batched.txt` (N(t), births/deaths/capped, events_hash, per-capita-vs-N,
  certify verdict). Exposes `--capacity`/`--regen` so the calibration sweep can drive it.

- [ ] **Step 1: Write the failing slow end-to-end + qualitative-bracket test**
```python
@pytest.mark.slow
def test_batched_certify_and_bracket(tmp_path):
    from embodied.batched_population import run, BatchedPopConfig
    from embodied.run_population import certify
    from embodied.foodfield import FoodFieldConfig
    poor = run(BatchedPopConfig(n_founders=20, horizon=40, bout_steps=6, max_pop=128, seed=0,
                                field=FoodFieldConfig(capacity=5.0, regen=0.2)))
    rich = run(BatchedPopConfig(n_founders=20, horizon=40, bout_steps=6, max_pop=256, seed=0,
                                field=FoodFieldConfig(capacity=40.0, regen=0.6)))
    v = certify(poor)
    assert "stable" in v and isinstance(v["stable"], bool)
    # bracket: rich sustains a higher population than poor
    assert max(rich.n_series) > max(poor.n_series)
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement `run_batched.py`** (mirror `run_population.py`'s reporting; reuse its
  `certify`). Build a `BatchedPopConfig` per seed from the CLI args (incl. `FoodFieldConfig(capacity,
  regen)`), run, `certify`, and write the report file (N(t) min/max/mean, births/deaths/**capped**,
  events_hash, per-capita-vs-N corr, the 4 gate bools + stable). Multi-seed.
```python
"""embodied.run_batched — CLI to run the batched substrate (CPU dev / GPU sweep) + certify."""
import argparse
from pathlib import Path
from embodied.batched_population import run, BatchedPopConfig
from embodied.foodfield import FoodFieldConfig
from embodied.run_population import certify

OUT = Path(__file__).resolve().parent / "outputs" / "embodied_batched.txt"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--founders", type=int, default=30); p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--max-pop", type=int, default=256); p.add_argument("--bout", type=int, default=8)
    p.add_argument("--capacity", type=float, default=5.0); p.add_argument("--regen", type=float, default=0.2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    a = p.parse_args()
    lines = [f"batched substrate run  founders={a.founders} horizon={a.horizon} max_pop={a.max_pop} "
             f"cap={a.capacity} regen={a.regen}"]
    for s in a.seeds:
        r = run(BatchedPopConfig(a.founders, a.horizon, a.bout, a.max_pop, s,
                                 FoodFieldConfig(capacity=a.capacity, regen=a.regen)))
        v = certify(r)
        lines += [f"--- seed {s} ---",
                  f"  N min/max/mean: {min(r.n_series)}/{max(r.n_series)}/{sum(r.n_series)/len(r.n_series):.1f}",
                  f"  births/deaths/capped: {r.births}/{r.deaths}/{r.capped}",
                  f"  events_hash: {r.events_hash}",
                  f"  certify: {v}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n"); print("\n".join(lines))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the fast suite + the slow test** (`uv run --python .venv pytest tests/test_batched_substrate.py -v -m "slow or not slow"`, long timeout) → all pass. Confirm the Phase-2 import-guard test (`tests/test_embodied_population.py::test_embodied_does_not_import_ecology_engine`) still passes (the new files must not import `ecology.engine`).
- [ ] **Step 5: Commit**
```bash
git add embodied/run_batched.py pyproject.toml tests/test_batched_substrate.py
git commit -m "embodied: batched substrate CLI (CPU dev / GPU sweep) + certify

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (spec § → task): §3.1 buffer + spawn → Task 1; §3.1 advance_batch + sequential
reference + §6 equivalence gate → Task 2; §3.2 numpy life-loop + slot mgmt → Task 4 (+ shared
economics Task 3); §5 per-bout-fixed targets → Task 2/4 (targets computed numpy per bout, fixed in the
bout scan); §6 determinism → Task 4 hash test; §2/§11.3 shared helper NOT touching Phase-2 → Task 3
(new file, batched-only); §7 CPU dev/GPU sweep → Task 5 CLI (`--capacity/--regen/--max-pop`); §8 tests
(buffer, equivalence, determinism, slot-reuse, loud cap, bracket) → Tasks 1/2/4/5; §10 the sweep is
the follow-on (out of scope, correctly). No uncovered requirement.

**Placeholder scan:** No "TBD/handle edge cases". The PROBE notes (batched pipeline_state stacking,
`.at[idx].set`, which leaves to reset on spawn) are explicit version-pinned verification actions with a
concrete command — the brax batched-state API is only knowable at the pin, as in Phase 1/2.

**Type consistency:** `BatchedEmbodiedWorld(food_field, runner, max_pop, bout_steps)`,
`init_buffer(n_founders, seed, founder_xys)->(state,alive)`, `spawn_into_slot(state, idx, xy, seed)`,
`advance_batch(state, targets)->(state, paths[max_pop,bout,2])`, `advance_sequential(...)` same sig,
`step_economics(g, energy, age, intake)->dict`, `BatchedPopConfig(...)`, `run(cfg)->BatchedPopResult`
(PopResult fields + `capped`) are used consistently across Tasks 1–5. The equivalence test uses
`np.allclose(atol=1e-4)` (vmap vs loop), the determinism test uses exact `events_hash` — distinct, as
the spec requires.
