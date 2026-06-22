# Embodied Physics Substrate — Design Spec (Phase 1: the pipeline slice)

- **Date:** 2026-06-21
- **Status:** Approved for implementation planning (brainstorming complete)
- **Direction:** `embodied-physics` (NEW parallel line; continues continuous-locomotion / environmental-complexity)
- **Scope of THIS spec:** Phase 1 only — a vertical "pipeline slice." Phases 2–4 are sketched for
  direction but are each their own future spec.

---

## 1. Motivation (why, and why now)

Across the research arc, **locomotion was never an evolvable axis**. Every "body" so far is a
*point agent* — `Phenotype.pos` (a grid cell index) or `pos_cont` (a continuous x,y), with no mass,
joints, or orientation. "Movement" is a **scalar trait** (`climb_ability` Exp 235, `locomotor_speed`
Exp 238) that scales a kinematic step. Those scalar knobs hit the **local-gradient / benefit-saturation
wall** every time: a crude amount of the trait grabs the easy benefit and precision never pays
(Exp 199–213, 235–247). The continuous-locomotion chapter closed-negative with an explicit note that
the only remaining escape is *an agent redesign* — "a consumer" with richer movement physics than a
point mass can express.

A **real articulated body in a physics engine** is that redesign. When the movement axis is a
gait controller / morphology (high-dimensional, with plausibly **non-saturating** returns — a better
gait keeps paying), the central question reopens with a fair substrate:

> **Does locomotion escape the benefit-saturation wall when the body is real?**

Separately, the human wants to **see** the experiments and **observe them first-person** — to embody
the agent's point of view, not just read population numbers. A physics engine delivers both the new
substrate and the visualization in one move.

This spec does **not** answer the science question yet. Per the agreed plan ("pipeline first, science
TBD"), Phase 1 proves the **rig**: one embodied creature that *learns to walk to food*, produces data,
and is *watchable* third-person **and** first-person. The science questions are deferred to Phases 2–4.

---

## 2. Goals / Non-goals

### Phase 1 goals
1. A new **`embodied/` sibling substrate** that does NOT touch the frozen `ecology/` engine.
2. One **stock quadruped** body in an arena with food, in a Brax/MJX environment.
3. A **short PPO** training run that produces a **learned walk-to-food gait**, saved as a committed
   checkpoint.
4. A **deterministic evaluation rollout** of the fixed checkpoint → per-step metrics + a stable
   trajectory **hash** (the honesty/reproducibility discipline carried over from `events_hash`).
5. **Rendering**: a third-person `.mp4` and a first-person (head-mounted camera) `.mp4` of the eval
   rollout, produced **on the Mac (CPU)**.
6. A **CPU "training smoke"** that runs locally without a GPU (tiny step budget) so the loop is
   testable in CI/tests; the *real* gait is trained on cloud GPU.
7. Tests: env builds, eval-rollout hash is stable, render writes the expected number of frames,
   train-smoke completes (slow-marked).

### Phase 1 non-goals (explicitly deferred)
- **No ecology outer loop** — no energy budget, reproduction, death, population, mutation.
- **No heritable traits / evolution** — the gait is *learned by PPO*, not *evolved*.
- **No locomotion-evolvability verdict** — no EXPERIMENTS.md entry, **no `exp269` number**. Phase 1 is
  substrate + tooling (like adding `ecology/acoustic.py` was), not a verdict-producing experiment.
- **No predator-prey / multi-agent**, no acoustic/sensory channel.
- **No showcase-site integration** (rendered videos onto the site is a later nicety).

---

## 3. Architecture

### 3.1 A new sibling substrate, not an in-place mutation

The existing `ecology/` engine is **pure numpy**, with ~250 experiments pinned by **golden
event-hashes**, and a body model (point agent) fundamentally different from an articulated body.
Bolting JAX physics into its ~505-line coupled step would fight the regression suite and entangle
JAX/numpy RNG. Therefore the embodied world is a **new top-level `embodied/` package**, a sibling to
`ecology/`. It **reuses the program's ideas** — the policy/behavior seam, the
genotype→mutate→evolve outer loop (Phase 2+), the `ecology/evolvability/` Preflight, and the binding
`loop/VALIDATION.md` honesty contract — without importing or modifying the frozen engine. This *is*
the "not deviating from what we've been doing": same research program and discipline, new body.

### 3.2 Engine split

- **Brax (over MJX)** — the trainable environment and the **PPO** trainer (the learned gait). MJX is
  GPU-batched JAX physics; this is where "cloud for scale" pays.
- **MuJoCo native renderer** (`mujoco.Renderer`) — offscreen rendering on macOS/CPU for the two
  videos, including a **head-mounted camera** for first-person. The trained policy is rolled out and
  rendered with the *same MJCF model*, so the physics the agent learned in and the physics we render
  are identical.
- **Why not Brax-only or MuJoCo-only:** Brax's renderer is weaker for first-person camera control;
  MuJoCo native has no built-in PPO. Using Brax to *train* and MuJoCo to *render* the same MJCF is the
  clean division of labor. (Brax v2 physics is MJX, so there is no physics mismatch.)

### 3.3 The behavior seam = the reward

With a learned gait there is no hand-coded controller. The single knob that defines what the creature
does is the **reward function** in `env.py`:

```
reward = w_progress * (approach toward nearest food site)
       + w_alive    * (torso upright / not fallen)
       - w_ctrl     * (control effort)
       + w_reach    * (bonus on reaching a food site, then respawn the food)
```

This is intentionally the minimal "go to the food" shaping. It is the Phase-1 analog of the ecology
`choose_action` policy seam and of the foraging objective the point-mass foragers had.

---

## 4. Components

```
embodied/
  __init__.py
  bodies/
    quadruped.xml      # stock quadruped MJCF (ant / dm_control style) + a <camera> on the head/torso
    arena.xml          # ground plane, bounding walls, a food site; <include> the body
  env.py               # EmbodiedForageEnv: Brax/MJX env — obs, reward (sec 3.3), reset/step, food respawn
  train.py             # short PPO (brax.training.agents.ppo) -> policy params checkpoint
  rollout.py           # deterministic eval rollout of a FIXED checkpoint -> trajectory (qpos/qvel/ctrl),
                       #   per-step metrics (distance-to-food, food eaten, displacement), trajectory hash
  render.py            # mujoco.Renderer: render a trajectory -> thirdperson.mp4 + firstperson.mp4
  checkpoints/
    quadruped_forage/  # committed trained policy params (small; so the videos reproduce)
  outputs/             # demo artifacts -> embodied_thirdperson.mp4, embodied_firstperson.mp4,
                       #   embodied_pipeline.txt. NOT experiments/outputs/ (Phase 1 is unnumbered).
                       #   Commit policy: commit the checkpoint + metrics .txt; .mp4s are regenerable
                       #   from the checkpoint, so gitignore them (or commit only if small) and send
                       #   to the human via the file-send path rather than bloating the repo.
  demo.py              # entrypoint: load checkpoint -> rollout -> render both videos -> write metrics
embodied/README.md     # what this is + how it relates to ecology/ (the sibling-substrate rationale)
tests/test_embodied.py # see sec 7
```

### Component contracts

- **`bodies/*.xml`** — *what:* the physical model (body + arena + cameras). *Interface:* a path loaded
  by `mujoco.MjModel.from_xml_path` and by the Brax MJX env loader. *Depends on:* nothing (data).
  The head camera is declared in MJCF so first-person is "render from camera N."
- **`env.py: EmbodiedForageEnv`** — *what:* the trainable task. *Interface:* Brax `Env` API
  (`reset(rng) -> State`, `step(State, action) -> State`), `observation_size`, `action_size`.
  *Depends on:* brax, mjx, the MJCF.
- **`train.py`** — *what:* run PPO to produce a gait. *Interface:* `train(config) -> checkpoint_path`;
  CLI `python -m embodied.train [--smoke]`. `--smoke` = tiny budget for CPU/tests. *Depends on:* env,
  brax PPO.
- **`rollout.py`** — *what:* deterministic eval. *Interface:*
  `rollout(checkpoint, seed, n_steps) -> Trajectory` where `Trajectory` carries `qpos[t], qvel[t],
  ctrl[t]`, metrics, and `traj_hash` (sha over rounded qpos/qvel/ctrl). *Depends on:* env, mjx.
  **Determinism:** fixed params + fixed key ⇒ identical trajectory ⇒ identical hash.
- **`render.py`** — *what:* trajectory → videos. *Interface:* `render(trajectory, model, out_dir) ->
  {thirdperson, firstperson}`. Uses `mujoco.Renderer` and a named camera for POV. *Depends on:*
  mujoco, imageio/ffmpeg.
- **`demo.py`** — *what:* the human-facing "watch it" entrypoint. CLI `python -m embodied.demo`.

---

## 5. Data flow

```
(cloud GPU)                                  (Mac CPU)
 train.py --(PPO on MJX, batched)-->          rollout.py (fixed checkpoint, fixed seed)
   checkpoints/quadruped_forage/  ==commit==>    -> Trajectory{qpos,qvel,ctrl, metrics, traj_hash}
                                                      |
                                                      v
                                                 render.py
                                                  -> embodied/outputs/embodied_thirdperson.mp4
                                                  -> embodied/outputs/embodied_firstperson.mp4
                                                 demo.py writes -> embodied/outputs/embodied_pipeline.txt
                                                  (metrics + traj_hash + train config/seed)
```

Training (stochastic) happens on the cloud GPU and is **frozen into a committed checkpoint**.
Everything downstream of the checkpoint is **deterministic** and runs on the Mac.

---

## 6. Compute plan (matches "local Mac + cloud for scale")

- **Train on cloud GPU (CUDA):** PPO on a quadruped wants a GPU; MJX `vmap` makes this minutes, not
  hours. Output: a checkpoint, committed to the repo.
- **Watch on the Mac (CPU):** `demo.py` loads the committed checkpoint, runs the deterministic eval
  rollout, and renders both videos locally. No GPU needed to watch.
- **CPU training smoke (Mac/CI):** `train.py --smoke` runs a tiny budget so the *loop* is exercised
  without a GPU (the resulting gait is intentionally bad; it only proves the pipeline runs).
- **Assumption to confirm at planning time:** a cloud GPU target exists / is reachable (the repo's
  remote path). If not, fall back to a slower, smaller CPU training run locally and accept a rougher
  gait for Phase 1.

---

## 7. Testing & verification

`tests/test_embodied.py` (fast unless marked slow):
1. **Model loads** — `quadruped.xml` and `arena.xml` parse; the head camera exists; obs/action sizes
   are as expected.
2. **Eval rollout determinism** — same checkpoint + same seed ⇒ identical `traj_hash` across two
   runs (the `events_hash` analog). This is the binding reproducibility gate.
3. **Render writes frames** — `render()` produces both files and the expected frame count; a few
   pixels are non-constant (not an all-black frame).
4. **Train smoke (slow-marked)** — `train(--smoke)` completes and writes a loadable checkpoint.

Verification before claiming Phase 1 done (per `superpowers:verification-before-completion`): actually
run `demo.py`, confirm both `.mp4`s play and the first-person view shows the food approaching, and
paste the metrics + `traj_hash`. Evidence before assertions.

---

## 8. Dependencies

- Add `mujoco` and `brax` to the project env (`uv` / `.venv`). `brax` pulls `mujoco-mjx`, `flax`,
  `optax`; **`jax`/`equinox` are already installed** (confirmed), so JAX is not a new foreign dep.
- Add a video writer (`imageio[ffmpeg]` or `mediapy`) for `.mp4` encode on macOS.
- Pin versions in the lockfile; record the exact versions in `embodied/README.md` (MJX/Brax APIs
  move; a pinned, recorded version protects reproducibility of the committed checkpoint).

---

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| PPO won't train a walk-to-food on CPU in reasonable time | Train on cloud GPU; CPU path is a *smoke* only, not expected to produce a good gait. |
| Brax/MJX API churn breaks the checkpoint later | Pin + record versions; keep the MJCF as the source of truth; the checkpoint is params over a fixed obs/action layout. |
| Offscreen MuJoCo rendering setup on macOS (GL backend) | Use `mujoco.Renderer` (handles offscreen); document the backend (e.g. `MUJOCO_GL`); the render test catches breakage. |
| Scope creep into ecology/evolution | Phase 1 non-goals (sec 2) are explicit; evolution is Phase 2+, its own spec. |
| Determinism lost via training stochasticity | Discipline binds the *eval* rollout, not training; checkpoint committed; training seed/config recorded. |

---

## 10. Roadmap beyond Phase 1 (each its own spec)

- **Phase 2 — embodied ecology outer loop:** wrap the embodied body in an energy/reproduce/die loop
  (mirroring `ecology/`), many creatures sharing an arena. Decide the integration seam: MJX-batched
  bodies vs. per-creature MuJoCo. Still no evolved gait.
- **Phase 3 — the science: locomotion evolvability.** Make gait-controller / morphology parameters a
  **heritable trait axis** (reuse `ecology/evolvability/` Preflight: gen-0 benefit curve +
  invasion-from-rarity). Ask: does a real body escape the benefit-saturation wall? Predeclare
  falsifiers per VALIDATION; this is the first **numbered** experiment (exp269+) and the first
  EXPERIMENTS.md entry.
- **Phase 4 — embodied predator-prey + first-person sensing** at scale on cloud GPU; revisit
  coexistence / Red Queen (Exp 257–268) with real bodies and a genuine egocentric sensory POV.
- **Later nicety:** surface the rendered videos on the showcase site.

---

## 11. Open questions / assumptions to confirm during planning

1. **Cloud GPU availability** for PPO (sec 6) — confirmed reachable, or CPU-fallback for Phase 1?
2. **Body choice** — a Brax/dm_control stock quadruped ("ant"-style) is the default; confirm that's
   fine vs. a specific critter.
3. **Reward weights** (sec 3.3) — start with simple defaults; tune only enough to get a watchable
   walk-to-food (Phase 1 success is "it visibly goes to the food," not an optimal gait).
