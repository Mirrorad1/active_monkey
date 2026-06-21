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
