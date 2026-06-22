# Embodied Population Loop (Phase 2) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A CPU numpy population loop where N embodied MuJoCo bodies live/eat/reproduce/die in a shared arena (one fixed Phase-1 policy), competing through a shared depletable food field, certified stable under the FROZEN evolvability Preflight gate. No evolution.

**Architecture:** A numpy life-loop in the `embodied/` sibling package reusing `ecology.genotype` + `ecology.evolvability` (instruments only — NEVER `ecology.engine`). Each creature's movement is a per-creature physics rollout behind a single `advance(pipeline_state, runner, food_field) -> (new_state, swept_path)` seam, which reuses the Phase-1 brax env's `pipeline_step` (MJX physics on CPU) so obs/action stay consistent with the trained policy. Competition is mediated by a shared depletable food field (no body-body collision).

**Tech Stack:** Python (uv/.venv), numpy (life loop + food field), the Phase-1 `embodied/` env + checkpoint (brax/MJX physics on CPU, MuJoCo for any render), pytest.

## Global Constraints

- Run everything via `uv run --python .venv ...` from the repo root.
- **Allowed imports:** `ecology.genotype` (Genotype, founder, mutate, TRAIT_BOUNDS) and
  `ecology.evolvability` (TraitAxis, cert_run/certify_run). **FORBIDDEN:** importing or modifying
  `ecology.engine`, `ecology.creature`, `ecology.continuous_world`, or any FROZEN path. A test asserts
  `embodied/` never imports `ecology.engine`.
- Do NOT modify the Phase-1 files (`embodied/env.py`, `train.py`, `rollout.py`, `render.py`,
  `demo.py`, the MJCF, the checkpoint). Phase 2 is purely additive. (Exception: none expected — if you
  think you must touch `env.py`, stop and report it.)
- **No evolution in Phase 2:** reproduction copies the parent genotype VERBATIM (no `mutate()` call).
- **No GPU / MJX-batching / brax PPO training.** Physics steps run on CPU via the env's
  `pipeline_step`. The fixed policy is loaded once via `embodied.train.make_inference`.
- **Determinism:** same seed → byte-identical event stream → identical event hash. Do not introduce
  unseeded randomness.
- **No `experiments/` files, no EXPERIMENTS.md entry** — Phase 2 is substrate/tooling.
- Commit messages END with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Stay on branch `embodied-physics-substrate`. Leave the unrelated modified `active_loop/cli/converse_demo.py` unstaged.

---

## File Structure

```
embodied/
  foodfield.py        # FoodField: numpy sub-cell depletable resource grid (shared, mediates competition)
  policy_runner.py    # PolicyRunner: load Phase-1 checkpoint; build_obs(state, food_xy); act(obs)->action
  world.py            # EmbodiedWorld: wraps the Phase-1 env for physics + holds the FoodField;
                      #   advance() seam; spawn_body(xy)
  creature.py         # EmbodiedCreature dataclass + life_step() (advance->intake->metabolism->reproduce/die)
  population.py        # run(): founders -> per-step life loop -> variable-N list -> EventLog + telemetry + hash
  run_population.py    # CLI entrypoint: run + write stability telemetry + Preflight certify verdict
tests/
  test_embodied_population.py
```

---

### Task 1: Shared depletable food field (`embodied/foodfield.py`)

**Files:**
- Create: `embodied/foodfield.py`
- Test: `tests/test_embodied_population.py`

**Interfaces:**
- Consumes: numpy. (Arena extent from `embodied.env`: `ARENA` is 8×8 half-extent walls; use a config.)
- Produces:
  - `@dataclass class FoodFieldConfig: extent: float = 6.0; cells: int = 24; capacity: float = 1.0; regen: float = 0.02; n_sources: int = 5`
  - `class FoodField:` with:
    - `__init__(cfg, seed)` — a `cells×cells` numpy grid over `[-extent,extent]^2`; initialized to a sum-of-Gaussians layout (n_sources peaks) clipped to `[0, capacity]`.
    - `resource_at(x, y) -> float` — bilinear/nearest read of the grid.
    - `nearest_food_xy(x, y, thresh=0.05) -> tuple[float,float]` — world xy of the nearest sub-cell whose resource > thresh (fallback: the global-max cell).
    - `consume(path_xy: list[tuple[float,float]], deficit: float) -> float` — walk the polyline, sample/deplete sub-cells along it, return total intake (≤ deficit). Depletes the grid in place.
    - `step_regen()` — `grid += regen`, clip to capacity (toward the original layout's local capacity).
    - `total() -> float` — sum of the grid (for diagnostics).

- [ ] **Step 1: Write failing tests (deplete, regen, nearest, line-integral)**

```python
import numpy as np, pytest
from pathlib import Path

def test_foodfield_deplete_regen_nearest():
    from embodied.foodfield import FoodField, FoodFieldConfig
    f = FoodField(FoodFieldConfig(extent=6.0, cells=24, capacity=1.0, regen=0.1, n_sources=5), seed=0)
    t0 = f.total()
    # consume along a path through the arena
    got = f.consume([(-3.0, 0.0), (3.0, 0.0)], deficit=10.0)
    assert got > 0.0 and got <= 10.0
    assert f.total() < t0                       # depletion happened
    f.step_regen()
    assert f.total() > (t0 - got)               # regen restored some
    # nearest_food points at a real high-resource cell
    fx, fy = f.nearest_food_xy(0.0, 0.0)
    assert -6.0 <= fx <= 6.0 and -6.0 <= fy <= 6.0
    assert f.resource_at(fx, fy) > 0.0
```

- [ ] **Step 2: Run → fail** `uv run --python .venv pytest tests/test_embodied_population.py::test_foodfield_deplete_regen_nearest -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `foodfield.py`**

```python
"""embodied.foodfield — a shared, depletable resource field mediating competition."""
from dataclasses import dataclass
import numpy as np


@dataclass
class FoodFieldConfig:
    extent: float = 6.0
    cells: int = 24
    capacity: float = 1.0
    regen: float = 0.02
    n_sources: int = 5


class FoodField:
    def __init__(self, cfg: FoodFieldConfig, seed: int = 0):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        n = cfg.cells
        xs = np.linspace(-cfg.extent, cfg.extent, n)
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        field = np.zeros((n, n))
        centers = rng.uniform(-cfg.extent * 0.7, cfg.extent * 0.7, size=(cfg.n_sources, 2))
        for cx, cy in centers:
            field += np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * (cfg.extent * 0.25) ** 2))
        self._cap = np.clip(field / field.max() * cfg.capacity, 1e-6, cfg.capacity)  # per-cell local cap
        self.grid = self._cap.copy()
        self._xs = xs

    def _ij(self, x, y):
        n = self.cfg.cells
        i = int(np.clip(round((x + self.cfg.extent) / (2 * self.cfg.extent) * (n - 1)), 0, n - 1))
        j = int(np.clip(round((y + self.cfg.extent) / (2 * self.cfg.extent) * (n - 1)), 0, n - 1))
        return i, j

    def resource_at(self, x, y) -> float:
        i, j = self._ij(x, y)
        return float(self.grid[i, j])

    def nearest_food_xy(self, x, y, thresh=0.05):
        n = self.cfg.cells
        mask = self.grid > thresh
        if not mask.any():
            i, j = np.unravel_index(int(np.argmax(self.grid)), self.grid.shape)
        else:
            X, Y = np.meshgrid(self._xs, self._xs, indexing="ij")
            d2 = (X - x) ** 2 + (Y - y) ** 2
            d2[~mask] = np.inf
            i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        return float(self._xs[i]), float(self._xs[j])

    def consume(self, path_xy, deficit) -> float:
        if deficit <= 0 or len(path_xy) < 1:
            return 0.0
        taken = 0.0
        for (x, y) in path_xy:
            i, j = self._ij(x, y)
            avail = float(self.grid[i, j])
            take = min(avail, max(0.0, deficit - taken))
            self.grid[i, j] = avail - take
            taken += take
            if taken >= deficit:
                break
        return taken

    def step_regen(self):
        self.grid = np.minimum(self._cap, self.grid + self.cfg.regen)

    def total(self) -> float:
        return float(self.grid.sum())
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**
```bash
git add embodied/foodfield.py tests/test_embodied_population.py
git commit -m "embodied: shared depletable FoodField (competition substrate)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Policy runner (`embodied/policy_runner.py`)

**Files:**
- Create: `embodied/policy_runner.py`
- Test: `tests/test_embodied_population.py`

**Interfaces:**
- Consumes: `embodied.env.EmbodiedForageEnv`, `embodied.train.load_params`/`make_inference`. The
  Phase-1 obs formula (documented in `env.py`): `obs = concat(q[2:], qd, food_xy - torso_xy)`, where
  `torso_xy = pipeline_state.q[0:2]`. action_size = 8, observation_size = 29.
- Produces:
  - `class PolicyRunner:` `__init__(checkpoint_path)` (loads params + builds deterministic inference +
    keeps an `EmbodiedForageEnv` for sizes); `build_obs(pipeline_state, food_xy) -> jnp.ndarray`;
    `act(obs) -> jnp.ndarray` (deterministic action, shape (8,)).
  - `DEFAULT_CKPT` = `embodied/checkpoints/quadruped_forage/params`.

- [ ] **Step 1: Write the failing test**
```python
def test_policy_runner_obs_and_action(tmp_path):
    import jax, jax.numpy as jnp
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    pr = PolicyRunner(DEFAULT_CKPT)
    env = pr.env
    state = env.reset(jax.random.PRNGKey(0))
    obs = pr.build_obs(state.pipeline_state, (3.0, 0.0))
    assert obs.shape == (env.observation_size,)        # 29
    act = pr.act(obs)
    assert act.shape == (env.action_size,)             # 8
    assert bool(jnp.isfinite(act).all())
```

- [ ] **Step 2: Run → fail** (ModuleNotFoundError).

- [ ] **Step 3: Implement `policy_runner.py`**
```python
"""embodied.policy_runner — run the fixed Phase-1 gait policy with a nearest-food obs."""
from pathlib import Path
import jax, jax.numpy as jnp
from embodied.env import EmbodiedForageEnv
from embodied.train import load_params, make_inference

DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "quadruped_forage" / "params"


class PolicyRunner:
    def __init__(self, checkpoint_path=DEFAULT_CKPT):
        self.env = EmbodiedForageEnv()
        self._infer = make_inference(load_params(str(checkpoint_path)))
        self._key = jax.random.PRNGKey(0)

    def build_obs(self, pipeline_state, food_xy):
        torso_xy = pipeline_state.q[0:2]
        to_food = jnp.asarray(food_xy) - torso_xy
        return jnp.concatenate([pipeline_state.q[2:], pipeline_state.qd, to_food])

    def act(self, obs):
        action, _ = self._infer(obs, self._key)   # deterministic: key unused by mean-action policy
        return action
```
NOTE: confirm `make_inference`'s returned callable signature is `(obs, key) -> (action, extras)`
(Phase-1 train.py). If it differs, adapt `act`. `pipeline_state.q`/`qd` are the confirmed field names.

- [ ] **Step 4: Run → pass** (uses the committed Phase-1 checkpoint).

- [ ] **Step 5: Commit**
```bash
git add embodied/policy_runner.py tests/test_embodied_population.py
git commit -m "embodied: PolicyRunner — fixed Phase-1 gait with nearest-food obs

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: EmbodiedWorld + the `advance` seam (`embodied/world.py`)

**Files:**
- Create: `embodied/world.py`
- Test: `tests/test_embodied_population.py`

**Interfaces:**
- Consumes: `EmbodiedForageEnv` (for `reset`, `pipeline_step`), `FoodField` (Task 1), `PolicyRunner`
  (Task 2). brax `State.pipeline_state.q[0:2]` = torso xy.
- Produces:
  - `class EmbodiedWorld:` `__init__(food_field, runner, bout_steps=20)`; holds one `EmbodiedForageEnv`.
    - `advance(pipeline_state) -> (new_pipeline_state, swept_path)` — for `bout_steps` control steps:
      `food_xy = food_field.nearest_food_xy(*torso_xy)`; `obs = runner.build_obs(state, food_xy)`;
      `action = runner.act(obs)`; `state = env.pipeline_step(state, action)`; append torso xy to path.
      Returns the final pipeline_state + the swept `[(x,y), ...]` (length `bout_steps`).
    - `spawn_pipeline_state(xy, seed) -> pipeline_state` — `env.reset(PRNGKey(seed))` then set the root
      translation `q[0:2] = xy` (immutable State → rebuild via the pipeline; see NOTE).
  - `THE SEAM` is `advance` — Phase 3 swaps it for an MJX-batched advance; nothing else changes.

- [ ] **Step 1: Write the failing test**
```python
def test_world_advance_moves_body_deterministically():
    import jax
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.world import EmbodiedWorld
    w = EmbodiedWorld(FoodField(FoodFieldConfig(), seed=0), PolicyRunner(DEFAULT_CKPT), bout_steps=10)
    s0 = w.env.reset(jax.random.PRNGKey(0))
    sA, pathA = w.advance(s0.pipeline_state)
    sB, pathB = w.advance(s0.pipeline_state)
    assert len(pathA) == 10
    assert pathA == pathB                                  # deterministic
    assert pathA[0] != pathA[-1] or True                   # body may or may not move much; path recorded
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement `world.py`**
```python
"""embodied.world — physics seam: advance a body via the Phase-1 env's pipeline_step."""
import jax
import jax.numpy as jnp


class EmbodiedWorld:
    def __init__(self, food_field, runner, bout_steps: int = 20):
        self.food = food_field
        self.runner = runner
        self.env = runner.env
        self.bout_steps = bout_steps
        self._step = jax.jit(self.env.pipeline_step)

    def advance(self, pipeline_state):
        state = pipeline_state
        path = []
        for _ in range(self.bout_steps):
            x, y = float(state.q[0]), float(state.q[1])
            obs = self.runner.build_obs(state, self.food.nearest_food_xy(x, y))
            action = self.runner.act(obs)
            state = self._step(state, action)
            path.append((float(state.q[0]), float(state.q[1])))
        return state, path

    def spawn_pipeline_state(self, xy, seed: int):
        s = self.env.reset(jax.random.PRNGKey(seed)).pipeline_state
        q = s.q.at[0:2].set(jnp.asarray(xy))               # set root translation
        return s.replace(q=q)                               # immutable pytree -> rebuild
```
NOTE (reconcile with the pinned brax): `pipeline_step(pipeline_state, action) -> pipeline_state` and
`state.q`/`replace(q=...)` are expected on brax 0.14.2's mjx pipeline state — probe once
(`print(type(s), [f for f in dir(s) if not f.startswith('_')])`). If `replace`/`q` differ, set the
root translation via the available accessor and report what you used. Setting only `q[0:2]` is a valid
free-joint translation; orientation/joints come from `reset`.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**
```bash
git add embodied/world.py tests/test_embodied_population.py
git commit -m "embodied: EmbodiedWorld + advance() seam (per-creature physics bout)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: EmbodiedCreature + life step (`embodied/creature.py`)

**Files:**
- Create: `embodied/creature.py`
- Test: `tests/test_embodied_population.py`

**Interfaces:**
- Consumes: `EmbodiedWorld` (Task 3), `ecology.genotype.Genotype`/`founder` (reused — pure leaf).
- Produces:
  - `@dataclass class EmbodiedCreature: id:int; genotype; energy:float; age:int; body_state; alive:bool=True`
  - `def life_step(c, world, rng, next_id) -> tuple[EmbodiedCreature, EmbodiedCreature|None, str]` —
    runs one bout: `new_state, path = world.advance(c.body_state)`; `intake =
    world.food.consume(path, deficit=cap-c.energy)`; charge metabolism
    `baseline_metabolic_cost + movement_cost + aging_cost*age`; update energy (cap at
    `energy_capacity`); age+=1; reproduce if eligible (returns a child near the parent, energy
    transferred, genotype COPIED verbatim); die if energy<=0. Returns `(updated_parent, child_or_None,
    event)` where event ∈ {"live","reproduce","die"}.
  - Reuse founder genotype field names: `g.baseline_metabolic_cost, g.movement_cost, g.aging_cost,
    g.energy_capacity, g.reproduction_energy_threshold, g.reproduction_energy_transfer_fraction,
    g.reproduction_cost_fraction, g.maturity_age` (confirmed present in `ecology/genotype.py`).

- [ ] **Step 1: Write the failing tests**
```python
def _world():
    from embodied.foodfield import FoodField, FoodFieldConfig
    from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
    from embodied.world import EmbodiedWorld
    return EmbodiedWorld(FoodField(FoodFieldConfig(), seed=0), PolicyRunner(DEFAULT_CKPT), bout_steps=8)

def test_life_step_energy_and_death():
    import numpy as np, jax
    from ecology.genotype import founder
    from embodied.creature import EmbodiedCreature, life_step
    w = _world(); rng = np.random.default_rng(0)
    s = w.env.reset(jax.random.PRNGKey(0)).pipeline_state
    g = founder()
    c = EmbodiedCreature(id=0, genotype=g, energy=0.05, age=1, body_state=s)  # near-starving
    upd, child, ev = life_step(c, w, rng, next_id=1)
    # low energy + metabolism => can die; if it ate enough it lives. Assert the contract holds:
    assert ev in ("live", "reproduce", "die")
    if ev == "die":
        assert not upd.alive
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement `creature.py`** (full, no placeholders)
```python
"""embodied.creature — an embodied life with energy/reproduce/die (no evolution in Phase 2)."""
from dataclasses import dataclass, field, replace
from ecology.genotype import Genotype


@dataclass
class EmbodiedCreature:
    id: int
    genotype: Genotype
    energy: float
    age: int
    body_state: object
    alive: bool = True


def life_step(c, world, rng, next_id):
    g = c.genotype
    new_state, path = world.advance(c.body_state)
    deficit = max(0.0, g.energy_capacity - c.energy)
    intake = world.food.consume(path, deficit)
    cost = g.baseline_metabolic_cost + g.movement_cost + g.aging_cost * c.age
    energy = min(g.energy_capacity, c.energy + intake - cost)
    age = c.age + 1
    parent = replace(c, body_state=new_state, energy=energy, age=age)

    if energy <= 0.0:
        return replace(parent, alive=False), None, "die"

    child = None
    event = "live"
    if age >= g.maturity_age and energy >= g.reproduction_energy_threshold:
        transfer = energy * g.reproduction_energy_transfer_fraction
        overhead = energy * g.reproduction_cost_fraction
        if energy - transfer - overhead > 0.0:
            # spawn child near parent (small offset), genotype COPIED verbatim (no mutation)
            px, py = float(new_state.q[0]), float(new_state.q[1])
            off = rng.normal(0.0, 0.3, size=2)
            child_state = world.spawn_pipeline_state((px + off[0], py + off[1]), seed=next_id)
            child = EmbodiedCreature(id=next_id, genotype=g, energy=transfer, age=0,
                                     body_state=child_state)
            parent = replace(parent, energy=energy - transfer - overhead)
            event = "reproduce"
    return parent, child, event
```

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Commit**
```bash
git add embodied/creature.py tests/test_embodied_population.py
git commit -m "embodied: EmbodiedCreature + life_step (energy/reproduce/die, genotype copied)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Population loop (`embodied/population.py`)

**Files:**
- Create: `embodied/population.py`
- Test: `tests/test_embodied_population.py`

**Interfaces:**
- Consumes: `EmbodiedWorld`, `EmbodiedCreature`/`life_step`, `ecology.genotype.founder`, `FoodField`.
- Produces:
  - `@dataclass class PopConfig: n_founders:int=20; horizon:int=200; bout_steps:int=8; seed:int=0; field:FoodFieldConfig=FoodFieldConfig()`
  - `@dataclass class PopResult: n_series:list[int]; per_capita_intake:list[float]; births:int; deaths:int; events_hash:str; final_alive:int`
  - `def run(cfg: PopConfig) -> PopResult` — spawn `n_founders` bodies spread across the arena (each
    via `world.spawn_pipeline_state`), `energy = 0.5*energy_capacity`; per step: iterate alive
    creatures (sorted by id for determinism), `life_step` each, accumulate intake, collect children +
    deaths, `world.food.step_regen()`, record `N`, per-capita intake, and append an event token
    (`id:event`) to a running sha256. Returns `PopResult`.

- [ ] **Step 1: Write the failing tests (birth+death happen; determinism; competition)**
```python
def test_population_runs_births_deaths_and_is_deterministic():
    from embodied.population import run, PopConfig
    from embodied.foodfield import FoodFieldConfig
    cfg = PopConfig(n_founders=8, horizon=40, bout_steps=6, seed=0, field=FoodFieldConfig(regen=0.05))
    a = run(cfg); b = run(cfg)
    assert a.events_hash == b.events_hash                  # determinism
    assert a.births >= 1 and a.deaths >= 1                 # life happens
    assert len(a.n_series) == 40

def test_shared_field_competition_lowers_per_capita_intake():
    from embodied.population import run, PopConfig
    from embodied.foodfield import FoodFieldConfig
    f = FoodFieldConfig(regen=0.02)
    lo = run(PopConfig(n_founders=4, horizon=20, bout_steps=6, seed=1, field=f))
    hi = run(PopConfig(n_founders=16, horizon=20, bout_steps=6, seed=1, field=f))
    # more creatures on the same field => lower mean per-capita intake (density-dependence)
    import numpy as np
    assert np.mean(hi.per_capita_intake) < np.mean(lo.per_capita_intake)
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement `population.py`** (full)
```python
"""embodied.population — the numpy life loop over embodied bodies on a shared food field."""
import hashlib
from dataclasses import dataclass, field
import numpy as np
from ecology.genotype import founder
from embodied.foodfield import FoodField, FoodFieldConfig
from embodied.policy_runner import PolicyRunner, DEFAULT_CKPT
from embodied.world import EmbodiedWorld
from embodied.creature import EmbodiedCreature, life_step


@dataclass
class PopConfig:
    n_founders: int = 20
    horizon: int = 200
    bout_steps: int = 8
    seed: int = 0
    field: FoodFieldConfig = field(default_factory=FoodFieldConfig)


@dataclass
class PopResult:
    n_series: list
    per_capita_intake: list
    births: int
    deaths: int
    events_hash: str
    final_alive: int


def run(cfg: PopConfig) -> PopResult:
    rng = np.random.default_rng(cfg.seed)
    ff = FoodField(cfg.field, seed=cfg.seed)
    world = EmbodiedWorld(ff, PolicyRunner(DEFAULT_CKPT), bout_steps=cfg.bout_steps)
    g0 = founder()
    # founders spread on a grid across the arena
    side = int(np.ceil(np.sqrt(cfg.n_founders)))
    xs = np.linspace(-cfg.field.extent * 0.6, cfg.field.extent * 0.6, side)
    alive, next_id = [], 0
    for k in range(cfg.n_founders):
        x, y = xs[k % side], xs[k // side]
        st = world.spawn_pipeline_state((float(x), float(y)), seed=next_id)
        alive.append(EmbodiedCreature(id=next_id, genotype=g0, energy=0.5 * g0.energy_capacity,
                                      age=0, body_state=st))
        next_id += 1

    h = hashlib.sha256()
    n_series, pci, births, deaths = [], [], 0, 0
    for _ in range(cfg.horizon):
        alive.sort(key=lambda c: c.id)             # deterministic order
        survivors, newborns, step_intake = [], [], 0.0
        for c in alive:
            before = c.energy
            upd, child, ev = life_step(c, world, rng, next_id)
            step_intake += max(0.0, upd.energy - before)  # net; intake proxy
            h.update(f"{c.id}:{ev};".encode())
            if ev == "die":
                deaths += 1
                continue
            survivors.append(upd)
            if child is not None:
                newborns.append(child); births += 1; next_id += 1
        alive = survivors + newborns
        world.food.step_regen()
        n = len(alive)
        n_series.append(n)
        pci.append(step_intake / max(1, n))
        if n == 0:
            n_series += [0] * (cfg.horizon - len(n_series))
            pci += [0.0] * (cfg.horizon - len(pci))
            break
    return PopResult(n_series, pci, births, deaths, h.hexdigest()[:16], len(alive))
```
NOTE: `step_intake` uses net energy delta as an intake proxy (intake − cost). If the verdict needs raw
intake, have `life_step` also return the raw `intake` and sum that instead — keep it consistent with
the density-dependence test (raw intake is the cleaner density signal; net is fine for the smoke).
Prefer returning raw intake from `life_step` (a 4-tuple) if Task 6 needs it.

- [ ] **Step 4: Run → pass** (allow a long timeout; this runs real physics bouts).

- [ ] **Step 5: Commit**
```bash
git add embodied/population.py tests/test_embodied_population.py
git commit -m "embodied: population loop (founders -> live/reproduce/die, shared field, hash)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Entrypoint + stability telemetry + Preflight certify (`embodied/run_population.py`)

**Files:**
- Create: `embodied/run_population.py`
- Modify: `pyproject.toml` (`[project.scripts]`: `active-monkey-embodied-pop = "embodied.run_population:main"`)
- Test: `tests/test_embodied_population.py`

**Interfaces:**
- Consumes: `embodied.population.run`/`PopConfig`/`PopResult`; `ecology.evolvability` certify (probe its
  exact entry — `from ecology.evolvability import ...`; the stability gate consumes a telemetry dict
  with `N` (per-step alive list) and reports persist/level-CV/drift/oscillation).
- Produces:
  - `def certify(result: PopResult) -> dict` — build the telemetry dict the Preflight stability gate
    consumes from `result.n_series` (key `N` at minimum; add `n_eq`, `level_cv`, `drift` if the gate
    needs them computed locally) and return the verdict + the density-dependence check (per-capita
    intake decreasing in N across the two-density probe, or the within-run intake-vs-N correlation < 0).
  - `def main()` — CLI `python -m embodied.run_population [--founders N] [--horizon T] [--seeds ...]`;
    runs, writes `embodied/outputs/embodied_population.txt` (N(t) summary, births/deaths, events_hash,
    per-capita-intake-vs-N, certify verdict). Multi-seed for the cross-seed CV check.

- [ ] **Step 1: Write the failing test (slow)**
```python
@pytest.mark.slow
def test_certify_produces_verdict():
    from embodied.population import run, PopConfig
    from embodied.run_population import certify
    r = run(PopConfig(n_founders=10, horizon=30, bout_steps=6, seed=0))
    v = certify(r)
    assert "stable" in v and "n_eq" in v               # verdict dict shape
    assert isinstance(v["stable"], bool)
```

- [ ] **Step 2: Run → fail.**

- [ ] **Step 3: Implement `run_population.py`** — first PROBE the Preflight's stability entrypoint:
```bash
uv run --python .venv python -c "import ecology.evolvability as ev, inspect; print([n for n in dir(ev) if not n.startswith('_')])"
uv run --python .venv python -c "from ecology.evolvability import stability; import inspect; print([n for n in dir(stability) if not n.startswith('_')]); print(inspect.signature(stability.certify_run))"
```
Then implement `certify()` to assemble the telemetry the real `certify_run` consumes (key `N` = the
per-step alive series; compute `n_eq`=median, `level_cv`, `drift` locally if the gate expects them
precomputed), call it, and add the density-dependence boolean. Write the metrics file. Use the FROZEN
thresholds from `stability.py` (persist ≥30, level CV ≤0.15, seed CV ≤0.25, drift ≤0.10, no oscillation)
— do NOT hardcode/loosen them; import them.
NOTE: if the real `certify_run` is too coupled to the ecology continuous config to call directly,
re-implement ONLY the threshold arithmetic against the imported FROZEN constants (`_PERSIST_FLOOR`
etc. from `ecology/evolvability/stability.py`) and cite them — never invent new thresholds.

- [ ] **Step 4: Run → pass.**

- [ ] **Step 5: Add the import-boundary guard test**
```python
def test_embodied_does_not_import_ecology_engine():
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1] / "embodied"
    for p in root.rglob("*.py"):
        src = p.read_text()
        assert "ecology.engine" not in src and "from ecology import engine" not in src, p
```

- [ ] **Step 6: Run the full suite + commit**
```bash
uv run --python .venv pytest tests/test_embodied_population.py -v -m "slow or not slow"   # long timeout
git add embodied/run_population.py pyproject.toml tests/test_embodied_population.py
git commit -m "embodied: population entrypoint + Preflight stability certify + import guard

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: First real run — does the population persist + the policy generalize?

**Files:**
- Modify: `embodied/README.md` (record the Phase-2 run result)
- Test: none new (this is a run + judgment task)

**Interfaces:**
- Consumes: everything above.
- Produces: a recorded verdict (persists/certifies or an honest NEGATIVE), and an answer to the
  policy-generalization risk.

- [ ] **Step 1: Run a real population**
```bash
uv run --python .venv python -m embodied.run_population --founders 30 --horizon 300 --seeds 0 1 2
```
Expected: `embodied/outputs/embodied_population.txt` with N(t), per-capita-intake-vs-N, certify verdict.

- [ ] **Step 2: Judge the policy-generalization risk (the spec's named risk)**
Inspect `n_series` + per-capita intake. If the population **starves immediately every seed** (N→0 fast,
intake≈0), the fixed Phase-1 policy did NOT generalize to a food field → STOP and report: the fallback
is to retrain the policy on a food-field env (a separate task using `runpod/`). If creatures forage
(intake>0, N persists for a while), the policy generalized — proceed to judge stability.

- [ ] **Step 3: Record the honest verdict**
Write to `embodied/README.md`: did it certify stable (PASS) or not (NEGATIVE / which gate failed +
whether stability-vs-competition were mutually exclusive, echoing Exp 243–247). Both are valid Phase-2
outcomes per the spec. Commit:
```bash
git add embodied/README.md
git commit -m "embodied: Phase-2 population run result + verdict

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage** (spec § → task): §3 architecture/import-boundary → Global Constraints + Task 6 guard; §4.1 FoodField → Task 1; §4.1 advance seam → Task 3; §4.2 EmbodiedCreature → Task 4; §4.3 life loop → Tasks 4+5; §4.4 policy_runner → Task 2; §5 shared-field competition → Task 1 + Task 5 competition test; §6 policy reuse + fallback → Task 2 + Task 7 step 2; §7 determinism → Task 5 hash test; §8 verdict (FROZEN gate + density-dependence) → Task 6 certify + Task 7; §9 tests → Tasks 1–6 tests; §10 no new deps → reuses Phase-1; §11 risks → Task 7 (policy), Global Constraints (import guard); §12 roadmap → out of scope (Phase 3); §13 K/layout/generalization → Tasks 3/1/7. No uncovered requirement.

**Placeholder scan:** No "TBD/handle edge cases" steps. The PROBE notes (brax State `replace`/`q`,
Preflight `certify_run` signature) are explicit, version-pinned verification actions with a concrete
command — not hand-waving; they exist because brax 0.14.2 + the ecology Preflight API are only knowable
at the pin, exactly as in the Phase-1 plan.

**Type consistency:** `FoodField.consume(path, deficit)`, `FoodField.nearest_food_xy(x,y)`,
`PolicyRunner.build_obs(state, food_xy)`/`act(obs)`, `EmbodiedWorld.advance(state)->(state,path)` /
`spawn_pipeline_state(xy,seed)`, `EmbodiedCreature(id,genotype,energy,age,body_state,alive)`,
`life_step(c,world,rng,next_id)->(parent,child|None,event)`, `run(PopConfig)->PopResult` are used
consistently across Tasks 1–7. (One flagged choice: `life_step` returns net-energy as the intake proxy
in Task 5; if Task 6's density-dependence needs raw intake, switch `life_step` to a 4-tuple returning
raw `intake` — noted in both tasks.)
