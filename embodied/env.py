"""embodied.env — a Brax/MJX forage task: a quadruped that walks to a food site.

The ant (free-joint root) starts at the origin; food is fixed at world position
determined by the arena.xml `food` site.  Observation is translation-invariant:
drop global (x, y), keep height + orientation + joint positions + velocities +
relative xy vector to food.  Reward encourages approach, staying upright, and
reaching the food; done is always 0.0 (fixed horizon, no early termination in
Phase 1).
"""
from pathlib import Path

import jax.numpy as jnp
import mujoco
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf

from embodied import FOOD_SITE

ARENA_PATH = Path(__file__).resolve().parent / "bodies" / "arena.xml"
REACH_RADIUS = 0.6
RewardWeights = dict(progress=1.0, alive=0.5, ctrl=0.001, reach=5.0)


class EmbodiedForageEnv(PipelineEnv):
    """Walk-to-food environment built on top of brax MJX PipelineEnv.

    Observation = concat(q[2:], qd, food_xy - torso_xy)
        q[2:]  : torso z + orientation quaternion (4) + 8 joint angles = 13 dims
        qd     : 14 velocity dofs
        delta  : 2-D vector from torso to food
    Total = 13 + 14 + 2 = 29.

    Action = 8-D joint torques (one per actuator in arena.xml).
    """

    def __init__(self, **kwargs):
        sys = mjcf.load(str(ARENA_PATH))
        super().__init__(sys=sys, backend="mjx", n_frames=5, **kwargs)

        # Compute food world xy ONCE from the MuJoCo model at the default pose.
        # The food site is on a mocap body and never moves in Phase 1, so
        # reading it once is both correct and avoids brax-version-fiddly
        # site_xpos accessors on pipeline states.
        mj = self.sys.mj_model          # brax System retains the underlying MjModel
        d = mujoco.MjData(mj)
        mujoco.mj_forward(mj, d)
        sid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_SITE, FOOD_SITE)
        self._food_xy = jnp.array(d.site_xpos[sid][:2])   # constant jnp array

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self, pipeline_state) -> jnp.ndarray:
        """Build the translation-invariant observation vector."""
        torso_xy = pipeline_state.q[0:2]
        to_food = self._food_xy - torso_xy
        # q[2:] = torso-z (1) + quaternion (4) + 8 joint angles = 13 values
        # qd    = 14 velocity dofs
        return jnp.concatenate([pipeline_state.q[2:], pipeline_state.qd, to_food])

    # ------------------------------------------------------------------
    # PipelineEnv interface
    # ------------------------------------------------------------------

    def reset(self, rng) -> State:
        pipeline_state = self.pipeline_init(
            self.sys.init_q,
            jnp.zeros(self.sys.qd_size()),
        )
        obs = self._get_obs(pipeline_state)
        torso_xy = pipeline_state.q[0:2]
        dist = jnp.linalg.norm(self._food_xy - torso_xy)
        metrics = {
            "dist_to_food": dist,
            "reached": jnp.float32(0.0),
        }
        return State(pipeline_state, obs, jnp.float32(0.0), jnp.float32(0.0), metrics)

    def step(self, state: State, action) -> State:
        prev_ps = state.pipeline_state
        prev_torso_xy = prev_ps.q[0:2]
        prev_dist = jnp.linalg.norm(self._food_xy - prev_torso_xy)

        pipeline_state = self.pipeline_step(prev_ps, action)

        torso_xy = pipeline_state.q[0:2]
        torso_height = pipeline_state.q[2]          # z from free-joint root
        dist = jnp.linalg.norm(self._food_xy - torso_xy)

        progress = prev_dist - dist
        alive = jnp.clip(torso_height, 0.0, 1.0)   # upright proxy
        ctrl = jnp.sum(action ** 2)
        reached = (dist < REACH_RADIUS).astype(jnp.float32)

        reward = (
            RewardWeights["progress"] * progress
            + RewardWeights["alive"] * alive
            - RewardWeights["ctrl"] * ctrl
            + RewardWeights["reach"] * reached
        )
        done = jnp.float32(0.0)  # fixed-horizon; no early termination in Phase 1
        metrics = {"dist_to_food": dist, "reached": reached}
        obs = self._get_obs(pipeline_state)
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
        )
