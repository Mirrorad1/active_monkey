# embodied — physics-engine substrate

## Why embodied/ is a separate substrate, not inside ecology/

`embodied/` is a **sibling** to `ecology/`, not a layer inside it. `ecology/` is a
discrete-time, numpy-based patch-mosaic engine designed for population-level
coevolution experiments. `embodied/` is a continuous-physics substrate built on
MuJoCo/Brax/MJX for sim-and-train of physically embodied agents. The two substrates
have incompatible data models, simulation time-steps, and backend requirements
(numpy + pure Python vs. JAX + hardware-accelerated physics). Keeping them separate
avoids coupling two unrelated abstractions, prevents dependency bleed into the frozen
`ecology/` engine, and lets each substrate evolve independently.

## Offscreen rendering backend

**Working `MUJOCO_GL` backend: `glfw`**

On macOS (Apple Silicon, desktop session with display), `MUJOCO_GL=glfw` works for
offscreen rendering via `mujoco.Renderer`. The smoke test in
`tests/test_embodied.py` sets `os.environ.setdefault("MUJOCO_GL", "glfw")` and
asserts the rendered frame is non-constant (`frame.std() > 1.0`).

If `glfw` fails in a headless environment, try `egl` then `osmesa`.

## Pinned versions

Installed 2026-06-21 on macOS Apple Silicon (aarch64), Python 3.12.9:

| Package       | Version |
|---------------|---------|
| mujoco        | 3.9.0   |
| mujoco-mjx    | 3.9.0   |
| brax          | 0.14.2  |
| imageio       | 2.37.3  |
| jax           | 0.10.1  |

`mujoco-mjx` arrives as a transitive dependency of `brax`.

## Quadruped body (Task 2)

**Source:** brax 0.14.2 bundled ant model —
`.venv/lib/python3.12/site-packages/brax/envs/assets/ant.xml`

Copied to `embodied/bodies/quadruped.xml`. Modifications from the original:

| Change | Reason |
|--------|--------|
| Removed `<custom>` block (brax-only numeric params) | MuJoCo 3.x emits a warning/error on unknown custom keys; not needed for pure MuJoCo use |
| Removed `compiler angle="degree" coordinate="local"` attributes | Deprecated in MuJoCo 3.x; omitting uses modern defaults without warnings |
| Removed original `<camera name="track" .../>` from torso body | Moved to worldbody in `arena.xml` (brief requires `mode="trackcom"` at world level) |
| Added `<camera name="firstperson" .../>` inside torso body | Required by Task 2 brief; orientation deferred to Task 6 |

**Arena:** `embodied/bodies/arena.xml` uses `<include file="quadruped.xml"/>` to merge
the quadruped into a wider arena scene. The included file contributes its `<worldbody>`,
`<default>`, `<asset>`, and `<actuator>` sections. `arena.xml` adds walls, a food site
(`name="food"`, `type="sphere"`), and the `track` camera at worldbody level. No second
floor is added — the quadruped already defines one (`name="floor"`).
