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

---

## Phase-2 Population Run — Stability Verdict

### Run Configuration

| Parameter | Value |
|-----------|-------|
| founders  | 30    |
| horizon   | 300 steps |
| seeds     | 0, 1, 2 |
| FoodFieldConfig | capacity=5.0, regen=0.2 (Phase-1 FROZEN calibration; not re-tuned) |
| bout_steps | 6 |
| Wall time | ~3.6 min total (~0.72 s/step × 300 × 3 seeds) |

No cap was applied. The pilot (founders=20, horizon=50, seed=0, ~35s) showed a
collapsing but not runaway trajectory, so the full budget-safe run proceeded at the
specified config.

### Policy Generalization

The Phase-1 fixed policy DID generalize to the food field. Per-capita intake is
non-zero throughout the alive portion of every run (seeds 1 and 2: mean PCI ≈ 1.28–1.30).
Births occurred: 1, 66, and 78 across seeds. The policy-generalization risk from the
spec is NOT the failure mode.

### Per-Seed N(t) Summary

| Seed | min N | max N | mean N | n_eq | births | deaths | final_alive | events_hash      |
|------|-------|-------|--------|------|--------|--------|-------------|------------------|
| 0    | 0     | 30    | 1.9    | 0.0  | 1      | 31     | 0 (extinct) | 40e1f5c55a57d349 |
| 1    | 3     | 33    | 10.6   | 8.5  | 66     | 89     | 7           | 4c5825292795cafe |
| 2    | 3     | 35    | 11.7   | 10.0 | 78     | 97     | 11          | 1368a66de4477eb5 |

Seed 0 went extinct at step 59 (1 birth vs. 31 deaths; b/d ≈ 0.03).
Seeds 1 and 2 did not go extinct but showed a declining trajectory from peak (≈33–35)
toward low-N quasi-plateau (≈7–11 in the last 20 steps). Birth:death ratios were
sub-replacement: 0.74 and 0.80.

### Preflight Stability Gates (FROZEN Thresholds)

| Seed | persistence (≥30) | level_cv (≤0.25) | drift (≤0.10) | oscillation (DAMPED) | stable |
|------|-------------------|------------------|----------------|----------------------|--------|
| 0    | FAIL (0)          | FAIL (3.249)     | FAIL (inf)     | FAIL (OSCILLATORY)   | False  |
| 1    | FAIL (3)          | FAIL (0.555)     | FAIL (1.608)   | FAIL (OSCILLATORY)   | False  |
| 2    | FAIL (3)          | FAIL (0.496)     | FAIL (0.904)   | FAIL (OSCILLATORY)   | False  |

Cross-seed: seeds stable = 0/3; seed_agreement = 1.176 (threshold ≤ 0.25) FAIL.

### Density-Dependence

| Seed | intake-vs-N corr | density_dependent |
|------|------------------|-------------------|
| 0    | +0.285           | False             |
| 1    | -0.320           | True              |
| 2    | -0.294           | True              |

Seeds 1 and 2 show the expected negative correlation (higher N → lower per-capita
intake), confirming that the competition mechanism is active. The signal is absent
in seed 0 because the population collapses before any density equilibrium is explored.

### Verdict: NEGATIVE

**0/3 seeds certify stable. All four Preflight gates fail on all seeds.**

The embodied substrate does NOT produce a stable population at the Phase-1 FROZEN
calibration (FoodFieldConfig capacity=5.0, regen=0.2). The Phase-1 policy generalized
(foraging works, births occur), but the metabolic/reproductive economy is insufficient
for population persistence above the stability floor (persist_floor=30).

Root cause: birth:death ratios are sub-replacement at this food-field calibration
(0.03, 0.74, 0.80). The food field supports foraging but not enough reproductive output
to balance metabolic death. This echoes the prior continuous-locomotion arc: per-creature
physics simulation is energy-expensive, and at this calibration the population cannot
close the birth:death balance.

The density-dependence signal (seeds 1–2) confirms the competition mechanism works; the
failure is in absolute birth rate, not in the feedback structure. A path to PASS would
require re-calibrating the food field or creature metabolics for b/d ≥ 1 at equilibrium
— a new task outside Phase-2 scope.
