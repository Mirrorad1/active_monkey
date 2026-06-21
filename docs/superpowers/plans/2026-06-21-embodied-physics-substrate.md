# Embodied Physics Substrate — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new `embodied/` substrate where a stock quadruped *learns to walk to food* (Brax/MJX PPO), then renders a deterministic eval rollout third-person + first-person (MuJoCo) on the Mac.

**Architecture:** A top-level `embodied/` package, sibling to the frozen numpy `ecology/` engine (never imported/modified). Brax (over MJX) defines the trainable forage env and trains a PPO gait; the rollout records a qpos/qvel trajectory from a fixed checkpoint; MuJoCo's native `Renderer` replays that trajectory and renders two cameras (a tracking third-person and a torso-mounted first-person). Training is stochastic and happens on cloud GPU; everything downstream of the committed checkpoint is deterministic and runs on Mac CPU.

**Tech Stack:** Python (uv/.venv), JAX (already installed), Brax + MuJoCo-MJX (training/physics), MuJoCo native (rendering), imageio[ffmpeg] (mp4 encode), pytest.

## Global Constraints

- Run everything via `uv run --python .venv ...` from the repo root (the shell shadows the venv otherwise).
- Do **NOT** import from, or modify, anything under `ecology/`, `eval/`, `creature/`, or any FROZEN path. `embodied/` is standalone.
- **No `exp269` number and no `EXPERIMENTS.md` entry** — Phase 1 is substrate+tooling, not a verdict-producing experiment.
- **Determinism gate (binding):** a fixed checkpoint + fixed seed must yield a byte-identical `traj_hash` on the same machine. Hash over qpos/qvel/ctrl rounded to 5 decimals (guards tiny float nondeterminism). This is the honesty analog of `events_hash`.
- **macOS rendering:** offscreen GL backend must be set; document it. Default `MUJOCO_GL=glfw` (or `egl`); the render test must catch a misconfigured backend.
- **Pin + record versions:** pin `mujoco`, `brax`, `mujoco-mjx` (brax dep), `imageio` in the lockfile and record exact versions in `embodied/README.md`. The committed checkpoint is only reproducible against pinned versions.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
- Outputs go to `embodied/outputs/` (NOT `experiments/outputs/`). Commit the checkpoint + metrics `.txt`; gitignore the `.mp4`s (regenerable from the checkpoint).

---

## File Structure

```
embodied/
  __init__.py            # package marker
  README.md              # what this is, sibling-substrate rationale, pinned versions, MUJOCO_GL note
  bodies/
    quadruped.xml        # vendored stock ant/quadruped body (copied from installed brax)
    arena.xml            # ground + walls + food site + cameras; <include>s quadruped.xml
  env.py                 # EmbodiedForageEnv (Brax PipelineEnv, mjx backend): obs/reward/reset/step
  train.py               # PPO training (brax) -> checkpoint; --smoke tiny config; CLI
  rollout.py             # deterministic eval rollout of a fixed checkpoint -> Trajectory + traj_hash
  render.py              # MuJoCo Renderer: Trajectory -> thirdperson.mp4 + firstperson.mp4
  demo.py                # load checkpoint -> rollout -> render -> write metrics; CLI
  checkpoints/
    quadruped_forage/    # committed trained params (real gait, trained on GPU)
  outputs/               # demo artifacts (.mp4 gitignored, .txt committed)
tests/
  test_embodied.py       # all Phase-1 tests (fast; train-smoke + render marked slow)
```

---

### Task 1: Toolchain — dependencies, offscreen render smoke

**Files:**
- Modify: `pyproject.toml` (add deps) and the uv lockfile
- Create: `embodied/__init__.py`
- Create: `embodied/README.md`
- Create: `embodied/.gitignore` (ignore `outputs/*.mp4`)
- Test: `tests/test_embodied.py`

**Interfaces:**
- Consumes: nothing.
- Produces: an importable `embodied` package; `mujoco`, `brax`, `imageio` importable in `.venv`; a documented `MUJOCO_GL` backend.

- [ ] **Step 1: Add dependencies and create package marker**

Add to `pyproject.toml` dependencies (exact pins resolved in step 2): `mujoco`, `brax`, `imageio[ffmpeg]`.
Create `embodied/__init__.py`:
```python
"""embodied — a physics-engine substrate (Brax/MJX + MuJoCo), sibling to ecology/.

Phase 1: a stock quadruped that LEARNS to walk to food, rendered third-person and
first-person. Does NOT import or modify the frozen ecology/ engine. See README.md.
"""
```
Create `embodied/.gitignore`:
```
outputs/*.mp4
```

- [ ] **Step 2: Install, pin, and record exact versions**

Run:
```bash
uv add mujoco brax "imageio[ffmpeg]"
uv run --python .venv python -c "import mujoco, brax, jax, imageio; print('mujoco', mujoco.__version__); print('brax', brax.__version__); print('jax', jax.__version__)"
```
Expected: prints versions with no ImportError. Copy the printed versions into `embodied/README.md` under a "Pinned versions" heading. If `brax.__version__` is absent, get it via `importlib.metadata.version("brax")`.

- [ ] **Step 3: Write the failing offscreen-render smoke test**

In `tests/test_embodied.py`:
```python
import os
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "glfw")  # offscreen backend; document in README


def test_mujoco_offscreen_render_nonblack():
    import mujoco
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 0.1" rgba="0.3 0.5 0.3 1"/>
        <body pos="0 0 0.3"><freejoint/><geom type="box" size="0.2 0.2 0.2" rgba="0.8 0.2 0.2 1"/></body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=120, width=160) as r:
        r.update_scene(data)
        frame = r.render()
    assert frame.shape == (120, 160, 3)
    assert frame.std() > 1.0  # not a constant/all-black frame
```

- [ ] **Step 4: Run it; if it fails on GL backend, switch MUJOCO_GL and document**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_mujoco_offscreen_render_nonblack -v`
Expected: PASS. If it fails with a GL/context error, try `MUJOCO_GL=egl` then `osmesa`; record the working backend in `embodied/README.md` and update the `setdefault` in the test.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock embodied/__init__.py embodied/README.md embodied/.gitignore tests/test_embodied.py
git commit -m "embodied: toolchain — mujoco+brax deps, offscreen render smoke

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Body + arena MJCF (vendored quadruped + food + cameras)

**Files:**
- Create: `embodied/bodies/quadruped.xml` (vendored)
- Create: `embodied/bodies/arena.xml`
- Test: `tests/test_embodied.py` (add cases)

**Interfaces:**
- Consumes: `mujoco` (Task 1).
- Produces: `arena.xml` loadable via `mujoco.MjModel.from_xml_path`; named camera `"firstperson"` on the torso; named camera `"track"`; a food site named `"food"`; constant `FOOD_SITE = "food"` is the body the env/render read.

- [ ] **Step 1: Vendor a known-good quadruped body**

Copy Brax's bundled ant model so we start from tested physics (do NOT hand-write a quadruped):
```bash
uv run --python .venv python -c "import brax, os, glob; base=os.path.dirname(brax.__file__); print('\n'.join(glob.glob(base+'/**/ant.xml', recursive=True)))"
```
Copy the printed `ant.xml` to `embodied/bodies/quadruped.xml`. If brax ships the ant as an env constant rather than a file, instead dump it: `uv run --python .venv python -c "from brax.envs import ant; print(ant._SYSTEM_CONFIG if hasattr(ant,'_SYSTEM_CONFIG') else '')"` and, failing that, copy MuJoCo's `ant.xml` from `python -c "import mujoco, os; print(os.path.join(os.path.dirname(mujoco.__file__)))"`-adjacent model dirs, or from `dm_control`. Record the source path in `embodied/README.md`.

- [ ] **Step 2: Write the failing MJCF-structure test**

Add to `tests/test_embodied.py`:
```python
from pathlib import Path
ARENA = Path(__file__).resolve().parents[1] / "embodied" / "bodies" / "arena.xml"


def test_arena_loads_with_cameras_and_food():
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    cams = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)}
    assert {"firstperson", "track"} <= cams
    food_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "food")
    assert food_id >= 0
    assert model.nu > 0  # has actuators (it can be driven)
```

- [ ] **Step 3: Write `arena.xml` that includes the body and adds ground, walls, food, cameras**

Create `embodied/bodies/arena.xml`. Wrap the vendored body: rename the file's root so it can be `<include>`d, OR (simpler) merge into one file. Concrete merge approach — `arena.xml`:
```xml
<mujoco model="embodied_arena">
  <include file="quadruped.xml"/>
  <statistic center="0 0 0.5" extent="6"/>
  <visual><headlight diffuse="0.6 0.6 0.6"/><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="8 8 0.1" rgba="0.25 0.35 0.25 1"/>
    <geom name="wall_n" type="box" pos="0 8 0.5" size="8 0.1 0.5" rgba="0.4 0.4 0.45 1"/>
    <geom name="wall_s" type="box" pos="0 -8 0.5" size="8 0.1 0.5" rgba="0.4 0.4 0.45 1"/>
    <geom name="wall_e" type="box" pos="8 0 0.5" size="0.1 8 0.5" rgba="0.4 0.4 0.45 1"/>
    <geom name="wall_w" type="box" pos="-8 0 0.5" size="0.1 8 0.5" rgba="0.4 0.4 0.45 1"/>
    <body name="food_body" pos="3 0 0.2" mocap="true">
      <site name="food" type="sphere" size="0.3" rgba="0.9 0.7 0.1 1"/>
    </body>
    <camera name="track" mode="trackcom" pos="0 -6 4" xyaxes="1 0 0 0 0.7 0.7"/>
  </worldbody>
</mujoco>
```
Then add a `firstperson` camera **on the torso**. Open `quadruped.xml`, find the torso body (the ant's main `<body name="torso" ...>`), and add inside it:
```xml
      <camera name="firstperson" mode="fixed" pos="0.3 0 0.1" xyaxes="0 -1 0 0 0 1"/>
```
(The `xyaxes` orients the camera looking along +x, the ant's forward. Adjust sign if the vendored ant faces -x; the render test in Task 6 will reveal orientation.)

- [ ] **Step 4: Run the structure test**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_arena_loads_with_cameras_and_food -v`
Expected: PASS. If `<include>` path errors, ensure `arena.xml` and `quadruped.xml` are in the same dir and the include uses a bare filename.

- [ ] **Step 5: Commit**

```bash
git add embodied/bodies/quadruped.xml embodied/bodies/arena.xml embodied/README.md tests/test_embodied.py
git commit -m "embodied: vendored quadruped + arena MJCF (food site, track + firstperson cameras)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: The forage environment (`env.py`)

**Files:**
- Create: `embodied/env.py`
- Test: `tests/test_embodied.py` (add cases)

**Interfaces:**
- Consumes: `arena.xml` (Task 2), brax/mjx (Task 1).
- Produces:
  - `FOOD_SITE = "food"`, `ARENA_PATH` (Path to arena.xml).
  - `class EmbodiedForageEnv(brax.envs.base.PipelineEnv)` with `reset(rng) -> State`, `step(state, action) -> State`, properties `observation_size: int`, `action_size: int`.
  - Observation = concat(qpos[2:], qvel, torso_xy_to_food[2]) (drop global x,y of root to keep it translation-invariant; keep z + orientation). Reward per `RewardWeights` below.
  - `RewardWeights = dict(progress=1.0, alive=0.5, ctrl=0.001, reach=5.0)`; `REACH_RADIUS = 0.6`.

- [ ] **Step 1: Write the failing env test**

Add to `tests/test_embodied.py`:
```python
import jax


def test_env_builds_and_steps():
    from embodied.env import EmbodiedForageEnv
    env = EmbodiedForageEnv()
    assert env.observation_size > 0 and env.action_size > 0
    state = env.reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (env.observation_size,)
    act = jax.numpy.zeros(env.action_size)
    nstate = env.step(state, act)
    assert nstate.obs.shape == (env.observation_size,)
    assert jax.numpy.isfinite(nstate.reward)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_env_builds_and_steps -v`
Expected: FAIL with `ModuleNotFoundError: embodied.env`.

- [ ] **Step 3: Implement `env.py`**

```python
"""embodied.env — a Brax/MJX forage task: a quadruped that should walk to a food site."""
from pathlib import Path
import jax
import jax.numpy as jnp
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

ARENA_PATH = Path(__file__).resolve().parent / "bodies" / "arena.xml"
FOOD_SITE = "food"
REACH_RADIUS = 0.6
RewardWeights = dict(progress=1.0, alive=0.5, ctrl=0.001, reach=5.0)


class EmbodiedForageEnv(PipelineEnv):
    def __init__(self, **kwargs):
        sys = mjcf.load(str(ARENA_PATH))
        super().__init__(sys=sys, backend="mjx", n_frames=5, **kwargs)
        self._food_site_id = mjcf.get_site_id(sys, FOOD_SITE) if hasattr(mjcf, "get_site_id") else \
            int(jnp.argmax(jnp.array([n == FOOD_SITE for n in sys.mj_model.site(...).name]))) if False else 0
        # robust food site id via the underlying mj_model:
        import mujoco
        self._food_site_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE, FOOD_SITE)
        self._torso_idx = 0  # root body

    def _food_xy(self, pipeline_state) -> jnp.ndarray:
        return pipeline_state.site_xpos[self._food_site_id][:2]

    def _torso_xy(self, pipeline_state) -> jnp.ndarray:
        return pipeline_state.x.pos[self._torso_idx][:2]

    def _obs(self, pipeline_state) -> jnp.ndarray:
        to_food = self._food_xy(pipeline_state) - self._torso_xy(pipeline_state)
        return jnp.concatenate([pipeline_state.q[2:], pipeline_state.qd, to_food])

    def reset(self, rng):
        pipeline_state = self.pipeline_init(self.sys.init_q, jnp.zeros(self.sys.qd_size()))
        obs = self._obs(pipeline_state)
        reward, done = jnp.float32(0.0), jnp.float32(0.0)
        metrics = {"dist_to_food": jnp.linalg.norm(self._food_xy(pipeline_state) - self._torso_xy(pipeline_state)),
                   "reached": jnp.float32(0.0)}
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state, action):
        prev = state.pipeline_state
        pipeline_state = self.pipeline_step(prev, action)
        prev_d = jnp.linalg.norm(self._food_xy(prev) - self._torso_xy(prev))
        d = jnp.linalg.norm(self._food_xy(pipeline_state) - self._torso_xy(pipeline_state))
        progress = prev_d - d
        upright = pipeline_state.x.pos[self._torso_idx][2]  # torso height proxy for "alive"
        alive = jnp.clip(upright, 0.0, 1.0)
        ctrl = jnp.sum(action ** 2)
        reached = (d < REACH_RADIUS).astype(jnp.float32)
        reward = (RewardWeights["progress"] * progress
                  + RewardWeights["alive"] * alive
                  - RewardWeights["ctrl"] * ctrl
                  + RewardWeights["reach"] * reached)
        done = jnp.float32(0.0)  # fixed-horizon; no early termination in Phase 1
        metrics = {"dist_to_food": d, "reached": reached}
        return state.replace(pipeline_state=pipeline_state, obs=self._obs(pipeline_state),
                             reward=reward, done=done, metrics=metrics)
```
NOTE: reconcile `pipeline_state` field names (`q`, `qd`, `x.pos`, `site_xpos`) with the pinned brax version — print `dir(state.pipeline_state)` once and fix names if they differ. Food respawn-on-reach is deferred to Task 3b only if training needs it; the fixed food is sufficient for "walk to food."

- [ ] **Step 4: Run the env test**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_env_builds_and_steps -v`
Expected: PASS. Fix field-name mismatches surfaced by the error.

- [ ] **Step 5: Commit**

```bash
git add embodied/env.py tests/test_embodied.py
git commit -m "embodied: EmbodiedForageEnv (Brax/MJX walk-to-food task)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: PPO training with a CPU smoke (`train.py`)

**Files:**
- Create: `embodied/train.py`
- Test: `tests/test_embodied.py` (add a slow test)

**Interfaces:**
- Consumes: `EmbodiedForageEnv` (Task 3); `brax.training.agents.ppo.train`; `brax.io.model`.
- Produces:
  - `def train(num_timesteps:int, seed:int, out_dir:str|Path) -> Path` — runs PPO, saves params via `brax.io.model.save_params`, returns the checkpoint path.
  - `SMOKE = dict(num_timesteps=4096, ...)` tiny config.
  - `def load_params(path)` re-export and `def make_inference(params)` returning a `(obs, rng) -> action` callable (deterministic mean action).
  - CLI: `python -m embodied.train [--smoke] [--timesteps N] [--out PATH]`.

- [ ] **Step 1: Write the failing slow smoke test**

Add to `tests/test_embodied.py`:
```python
@pytest.mark.slow
def test_train_smoke_writes_loadable_checkpoint(tmp_path):
    from embodied.train import train, load_params
    ckpt = train(num_timesteps=2048, seed=0, out_dir=tmp_path)
    params = load_params(ckpt)
    assert params is not None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_train_smoke_writes_loadable_checkpoint -v -m slow`
Expected: FAIL with `ModuleNotFoundError: embodied.train`.

- [ ] **Step 3: Implement `train.py`**

```python
"""embodied.train — short PPO to teach the quadruped to walk to food."""
import functools
from pathlib import Path
import jax
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from embodied.env import EmbodiedForageEnv

SMOKE = dict(num_timesteps=2048, num_envs=8, batch_size=8, num_minibatches=1,
             num_evals=1, episode_length=200, unroll_length=10)
FULL = dict(num_timesteps=30_000_000, num_envs=2048, batch_size=1024, num_minibatches=32,
            num_evals=5, episode_length=1000, unroll_length=20)


def train(num_timesteps: int, seed: int, out_dir, **overrides) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(SMOKE); cfg.update(overrides); cfg["num_timesteps"] = num_timesteps
    env = EmbodiedForageEnv()
    make_networks = functools.partial(ppo_networks.make_ppo_networks,
                                      policy_hidden_layer_sizes=(64, 64))
    make_inference_fn, params, _ = ppo.train(
        environment=env, seed=seed, network_factory=make_networks,
        num_timesteps=cfg["num_timesteps"], num_envs=cfg["num_envs"],
        batch_size=cfg["batch_size"], num_minibatches=cfg["num_minibatches"],
        num_evals=cfg["num_evals"], episode_length=cfg["episode_length"],
        unroll_length=cfg["unroll_length"], normalize_observations=True)
    ckpt = out_dir / "params"
    model.save_params(str(ckpt), params)
    return ckpt


def load_params(path):
    return model.load_params(str(path))


def make_inference(params):
    env = EmbodiedForageEnv()
    make_networks = functools.partial(ppo_networks.make_ppo_networks,
                                      policy_hidden_layer_sizes=(64, 64))
    nets = make_networks(env.observation_size, env.action_size)
    inference = ppo_networks.make_inference_fn(nets)(params, deterministic=True)
    return inference  # (obs, rng) -> (action, extras)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="embodied/checkpoints/quadruped_forage")
    a = p.parse_args()
    n = a.timesteps or (SMOKE["num_timesteps"] if a.smoke else FULL["num_timesteps"])
    extra = {} if a.smoke or a.timesteps else FULL
    print("checkpoint:", train(n, a.seed, a.out, **{k: v for k, v in extra.items() if k != "num_timesteps"}))
```
NOTE: reconcile `make_ppo_networks` / `make_inference_fn` signatures and the `normalize_observations` kwarg with the pinned brax version (print `help(ppo.train)` once). If `make_inference_fn` returns the deterministic fn differently, adapt `make_inference`.

- [ ] **Step 4: Run the smoke test (CPU; may take a few minutes)**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_train_smoke_writes_loadable_checkpoint -v -m slow`
Expected: PASS. The gait will be useless (tiny budget); we only assert the loop runs and the checkpoint loads.

- [ ] **Step 5: Commit**

```bash
git add embodied/train.py tests/test_embodied.py
git commit -m "embodied: PPO training (--smoke CPU loop, FULL config for GPU)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Deterministic eval rollout (`rollout.py`)

**Files:**
- Create: `embodied/rollout.py`
- Test: `tests/test_embodied.py` (add cases)

**Interfaces:**
- Consumes: `EmbodiedForageEnv` (Task 3); `make_inference`, `load_params` (Task 4).
- Produces:
  - `@dataclass(frozen=True) class Trajectory: qpos:list[np.ndarray]; qvel:list[np.ndarray]; ctrl:list[np.ndarray]; dist_to_food:list[float]; reached:list[float]; traj_hash:str`.
  - `def rollout(checkpoint, n_steps:int=400, seed:int=0) -> Trajectory` — loads params, builds deterministic inference, steps the env `n_steps` times recording qpos/qvel/ctrl + metrics, computes `traj_hash`.
  - `def _hash(arrays:list[np.ndarray]) -> str` — sha256 over `np.round(x, 5)` bytes.

- [ ] **Step 1: Write the failing determinism test**

Add to `tests/test_embodied.py`:
```python
@pytest.mark.slow
def test_rollout_is_deterministic(tmp_path):
    from embodied.train import train
    from embodied.rollout import rollout
    ckpt = train(num_timesteps=2048, seed=0, out_dir=tmp_path)
    a = rollout(ckpt, n_steps=50, seed=0)
    b = rollout(ckpt, n_steps=50, seed=0)
    assert a.traj_hash == b.traj_hash
    assert len(a.qpos) == 50
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_rollout_is_deterministic -v -m slow`
Expected: FAIL with `ModuleNotFoundError: embodied.rollout`.

- [ ] **Step 3: Implement `rollout.py`**

```python
"""embodied.rollout — deterministic eval rollout of a fixed checkpoint."""
import hashlib
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
from embodied.env import EmbodiedForageEnv
from embodied.train import make_inference, load_params


@dataclass(frozen=True)
class Trajectory:
    qpos: list
    qvel: list
    ctrl: list
    dist_to_food: list
    reached: list
    traj_hash: str


def _hash(arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.round(np.asarray(a), 5), dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def rollout(checkpoint, n_steps: int = 400, seed: int = 0) -> Trajectory:
    env = EmbodiedForageEnv()
    inference = make_inference(load_params(checkpoint))
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)
    infer = jax.jit(inference)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    qpos, qvel, ctrl, dist, reached = [], [], [], [], []
    for _ in range(n_steps):
        act, _ = infer(state.obs, rng)  # deterministic -> rng unused but accepted
        state = step(state, act)
        ps = state.pipeline_state
        qpos.append(np.asarray(ps.q)); qvel.append(np.asarray(ps.qd)); ctrl.append(np.asarray(act))
        dist.append(float(state.metrics["dist_to_food"])); reached.append(float(state.metrics["reached"]))
    return Trajectory(qpos, qvel, ctrl, dist, reached, _hash(qpos + qvel + ctrl))
```
NOTE: match `ps.q`/`ps.qd` to the pinned brax pipeline-state field names (same reconciliation as Task 3).

- [ ] **Step 4: Run the determinism test**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_rollout_is_deterministic -v -m slow`
Expected: PASS (identical `traj_hash` twice).

- [ ] **Step 5: Commit**

```bash
git add embodied/rollout.py tests/test_embodied.py
git commit -m "embodied: deterministic eval rollout + trajectory hash (events_hash analog)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Rendering — third-person + first-person mp4 (`render.py`)

**Files:**
- Create: `embodied/render.py`
- Test: `tests/test_embodied.py` (add cases)

**Interfaces:**
- Consumes: `Trajectory` (Task 5); `arena.xml` (Task 2); `mujoco`, `imageio`.
- Produces:
  - `def render(traj, out_dir, fps:int=30, height:int=480, width:int=640) -> dict[str,Path]` — replays `traj.qpos` into a `mujoco.MjData`, renders the `"track"` and `"firstperson"` cameras, writes `thirdperson.mp4` + `firstperson.mp4`, returns `{"thirdperson":Path, "firstperson":Path}`.

- [ ] **Step 1: Write the failing render test**

Add to `tests/test_embodied.py`:
```python
def _fake_traj(n=12):
    import numpy as np
    from embodied.rollout import Trajectory
    import mujoco
    from pathlib import Path
    model = mujoco.MjModel.from_xml_path(str(ARENA))
    q0 = np.zeros(model.nq)
    qpos = [q0.copy() for _ in range(n)]
    return Trajectory(qpos, [np.zeros(model.nv)] * n, [np.zeros(model.nu)] * n,
                      [1.0] * n, [0.0] * n, "deadbeefdeadbeef")


def test_render_writes_two_videos(tmp_path):
    from embodied.render import render
    out = render(_fake_traj(), tmp_path, fps=10)
    for k in ("thirdperson", "firstperson"):
        assert out[k].exists() and out[k].stat().st_size > 0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_render_writes_two_videos -v`
Expected: FAIL with `ModuleNotFoundError: embodied.render`.

- [ ] **Step 3: Implement `render.py`**

```python
"""embodied.render — replay a Trajectory through MuJoCo and write two camera videos."""
import os
from pathlib import Path
import numpy as np
import imageio
import mujoco
from embodied.env import ARENA_PATH

os.environ.setdefault("MUJOCO_GL", "glfw")  # match the backend documented in README


def _frames(model, data, qpos_seq, camera, height, width):
    out = []
    with mujoco.Renderer(model, height=height, width=width) as r:
        for q in qpos_seq:
            data.qpos[:] = np.asarray(q)
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=camera)
            out.append(r.render())
    return out


def render(traj, out_dir, fps: int = 30, height: int = 480, width: int = 640):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(ARENA_PATH))
    data = mujoco.MjData(model)
    paths = {}
    for name, cam in (("thirdperson", "track"), ("firstperson", "firstperson")):
        frames = _frames(model, data, traj.qpos, cam, height, width)
        p = out_dir / f"embodied_{name}.mp4"
        imageio.mimsave(str(p), frames, fps=fps)
        paths[name] = p
    return paths
```

- [ ] **Step 4: Run the render test**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_render_writes_two_videos -v`
Expected: PASS (both mp4s exist, non-empty).

- [ ] **Step 5: Commit**

```bash
git add embodied/render.py tests/test_embodied.py
git commit -m "embodied: MuJoCo renderer -> third-person + first-person mp4

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Demo entrypoint + metrics (`demo.py`)

**Files:**
- Create: `embodied/demo.py`
- Modify: `pyproject.toml` (add `active-monkey-embodied = "embodied.demo:main"` console script)
- Test: `tests/test_embodied.py` (add a slow end-to-end case)

**Interfaces:**
- Consumes: `rollout` (Task 5), `render` (Task 6), `train` (Task 4).
- Produces:
  - `def main() -> None` — CLI `python -m embodied.demo [--checkpoint PATH] [--steps N] [--train-smoke]`. Loads/loads-after-smoke a checkpoint, runs `rollout`, runs `render`, writes `embodied/outputs/embodied_pipeline.txt` with traj_hash + metrics (final dist_to_food, total reached, n_steps) + the checkpoint path + git SHA.

- [ ] **Step 1: Write the failing end-to-end test**

Add to `tests/test_embodied.py`:
```python
@pytest.mark.slow
def test_demo_end_to_end(tmp_path, monkeypatch):
    from embodied import demo
    from embodied.train import train
    ckpt = train(num_timesteps=2048, seed=0, out_dir=tmp_path)
    monkeypatch.setattr(demo, "OUT_DIR", tmp_path / "out")
    demo.run(checkpoint=ckpt, steps=20)
    assert (tmp_path / "out" / "embodied_pipeline.txt").exists()
    assert (tmp_path / "out" / "embodied_thirdperson.mp4").exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_demo_end_to_end -v -m slow`
Expected: FAIL with `ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 3: Implement `demo.py`**

```python
"""embodied.demo — watch the trained creature: rollout -> render -> metrics."""
import subprocess
from pathlib import Path
from embodied.rollout import rollout
from embodied.render import render

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "quadruped_forage" / "params"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def run(checkpoint=DEFAULT_CKPT, steps: int = 400) -> None:
    traj = rollout(checkpoint, n_steps=steps)
    vids = render(traj, OUT_DIR)
    lines = [
        "embodied pipeline demo (Phase 1 — substrate+tooling, no exp number)",
        f"checkpoint: {checkpoint}", f"git: {_git_sha()}",
        f"n_steps: {len(traj.qpos)}", f"traj_hash: {traj.traj_hash}",
        f"final_dist_to_food: {traj.dist_to_food[-1]:.4f}",
        f"total_reached_steps: {sum(traj.reached):.0f}",
        f"thirdperson: {vids['thirdperson']}", f"firstperson: {vids['firstperson']}",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "embodied_pipeline.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--train-smoke", action="store_true")
    a = p.parse_args()
    ckpt = a.checkpoint
    if a.train_smoke:
        from embodied.train import train
        ckpt = train(num_timesteps=2048, seed=0, out_dir=Path(a.checkpoint).parent)
    run(checkpoint=ckpt, steps=a.steps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the end-to-end test**

Run: `uv run --python .venv pytest tests/test_embodied.py::test_demo_end_to_end -v -m slow`
Expected: PASS (txt + mp4 written).

- [ ] **Step 5: Commit**

```bash
git add embodied/demo.py pyproject.toml tests/test_embodied.py
git commit -m "embodied: demo entrypoint (rollout->render->metrics) + console script

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Train the real gait on GPU + commit the checkpoint

**Files:**
- Create: `embodied/checkpoints/quadruped_forage/params` (committed binary)
- Modify: `embodied/README.md` (training command + GPU/device + final metrics)

**Interfaces:**
- Consumes: everything above.
- Produces: a committed, reproducible checkpoint that produces a *watchable* walk-to-food gait.

- [ ] **Step 1: Run full training on a CUDA GPU**

On the cloud GPU box (the "for scale" path):
```bash
uv run --python .venv python -m embodied.train --out embodied/checkpoints/quadruped_forage
```
Expected: PPO completes; eval reward trends up; a `params` file is written. If the gait is poor, raise `num_timesteps` or tune `RewardWeights` (progress↑, ctrl↓) and re-run. Phase-1 success bar = "the creature visibly moves toward the food," not an optimal gait.

- [ ] **Step 2: Render locally on the Mac and eyeball it**

Pull the checkpoint to the Mac, then:
```bash
uv run --python .venv python -m embodied.demo --steps 400
```
Expected: `embodied/outputs/embodied_thirdperson.mp4` shows the quadruped walking toward the food; `embodied_firstperson.mp4` shows the food growing in view. If the first-person camera points backward, flip the `firstperson` camera `xyaxes` in `quadruped.xml` and re-render (no retrain needed).

- [ ] **Step 3: Record metrics + commit the checkpoint**

Copy the printed `traj_hash` + `final_dist_to_food` into `embodied/README.md`. Then:
```bash
git add embodied/checkpoints/quadruped_forage/params embodied/README.md
git commit -m "embodied: committed trained walk-to-food checkpoint + recorded metrics

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Full suite green + send the videos to the human**

Run: `uv run --python .venv pytest tests/test_embodied.py -v` (fast) and `... -m slow` (slow).
Expected: all PASS. Then surface both mp4s to the human (file-send), with a one-line caption.

---

## Self-Review

**Spec coverage** (spec §→task): §3.1 sibling substrate → Global Constraints + Task 1/3; §3.2 engine split (Brax train / MuJoCo render) → Tasks 4/6; §3.3 reward seam → Task 3; §4 components → Tasks 1–7 (one per module); §5 data flow → Tasks 5–7; §6 compute (GPU train / Mac render / CPU smoke) → Tasks 4 (smoke) + 8 (GPU) + 6 (Mac render); §7 testing (model loads, rollout determinism, render frames, train smoke) → Tasks 2/5/6/4 tests; §8 deps + pinned versions → Task 1; §9 risks → addressed inline (GL backend Task 1, API churn NOTEs, determinism Task 5); §10 roadmap → out of Phase-1 scope (correctly deferred); §11 open questions → Task 8 step 1 (GPU), Task 2 (body), Task 8 step 1 (reward tuning). No uncovered requirement.

**Placeholder scan:** No "TBD/implement later/handle edge cases" steps. The reconciliation NOTEs (brax field/signature names) are explicit, version-pinned verification actions with a concrete `print(...)` to run — not hand-waving; they exist because the brax API genuinely varies by version and the pinned version is only known after Task 1.

**Type consistency:** `EmbodiedForageEnv`, `Trajectory(qpos,qvel,ctrl,dist_to_food,reached,traj_hash)`, `train(num_timesteps,seed,out_dir)`, `load_params`, `make_inference`, `rollout(checkpoint,n_steps,seed)`, `render(traj,out_dir,...) -> {thirdperson,firstperson}`, `demo.run(checkpoint,steps)` / `demo.OUT_DIR` are used consistently across Tasks 3–8.
