"""embodied.rollout — deterministic eval rollout of a fixed checkpoint.

The DETERMINISM GATE: calling rollout(checkpoint, n_steps, seed) twice with the
same arguments must produce a byte-identical traj_hash.  This is the honesty
analog of the repo's events_hash guard.

Determinism is guaranteed by three properties:
  (a) make_inference returns a deterministic (mean-action) policy; rng is
      accepted but unused, so reusing PRNGKey(seed) every step is correct.
  (b) env.reset uses PRNGKey(seed) — same starting state each call.
  (c) MJX stepping is bit-exact on the same hardware (no stochastic ops).

Hash = sha256 over qpos + qvel + ctrl arrays (each np.round(x, 5) to float64
bytes).  The 5-decimal rounding guards last-bit jitter while preserving the
meaningful trajectory signal.
"""
import hashlib
from dataclasses import dataclass

import jax
import numpy as np

from embodied.env import EmbodiedForageEnv
from embodied.train import load_params, make_inference


@dataclass(frozen=True)
class Trajectory:
    """Recorded trajectory from one deterministic rollout.

    Each list has length n_steps; qpos/qvel/ctrl are np.ndarray per step.
    dist_to_food and reached are float per step.
    traj_hash is the first 16 hex chars of sha256 over all arrays.
    """

    qpos: list
    qvel: list
    ctrl: list
    dist_to_food: list
    reached: list
    traj_hash: str


def _hash(arrays) -> str:
    """sha256 over a list of arrays, rounded to 5 decimals as float64.

    Returns the first 16 hex characters (64-bit fingerprint).
    The 5-decimal rounding guards last-bit floating-point jitter while
    preserving trajectory identity.
    """
    h = hashlib.sha256()
    for a in arrays:
        h.update(
            np.ascontiguousarray(
                np.round(np.asarray(a), 5), dtype=np.float64
            ).tobytes()
        )
    return h.hexdigest()[:16]


def rollout(checkpoint, n_steps: int = 400, seed: int = 0) -> Trajectory:
    """Load a checkpoint and run a deterministic n_steps rollout.

    Args:
        checkpoint: path returned by train() (string or Path).
        n_steps: number of environment steps to record.
        seed: integer seed; controls env.reset and the rng passed to inference
              (the policy is deterministic so the rng value is unused, but we
              keep the interface consistent).

    Returns:
        Trajectory with per-step qpos, qvel, ctrl, dist_to_food, reached, and
        a traj_hash that is identical across two calls with the same arguments.
    """
    env = EmbodiedForageEnv()
    params = load_params(checkpoint)
    inference = make_inference(params)

    # JIT-compile the three hot functions once.
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)
    infer_fn = jax.jit(inference)

    # Fixed rng — same every call, same every step (policy is deterministic).
    rng = jax.random.PRNGKey(seed)

    state = reset_fn(rng)

    qpos_list: list = []
    qvel_list: list = []
    ctrl_list: list = []
    dist_list: list = []
    reached_list: list = []

    for _ in range(n_steps):
        # deterministic=True means rng is ignored inside inference; reusing the
        # same PRNGKey each step is intentional and keeps things reproducible.
        action, _extras = infer_fn(state.obs, rng)
        state = step_fn(state, action)

        ps = state.pipeline_state
        qpos_list.append(np.asarray(ps.q))
        qvel_list.append(np.asarray(ps.qd))
        ctrl_list.append(np.asarray(action))
        dist_list.append(float(state.metrics["dist_to_food"]))
        reached_list.append(float(state.metrics["reached"]))

    traj_hash = _hash(qpos_list + qvel_list + ctrl_list)

    return Trajectory(
        qpos=qpos_list,
        qvel=qvel_list,
        ctrl=ctrl_list,
        dist_to_food=dist_list,
        reached=reached_list,
        traj_hash=traj_hash,
    )
