"""embodied.world — physics seam: advance a body via the Phase-1 env's pipeline_step.

THE SEAM: Phase 3 swaps `advance` for an MJX-batched version; nothing else changes.
"""
import jax
import jax.numpy as jnp


class EmbodiedWorld:
    """Wraps a FoodField + PolicyRunner and exposes a single `advance` seam.

    Parameters
    ----------
    food_field : FoodField
        The shared food landscape (nearest_food_xy, consume, step_regen).
    runner : PolicyRunner
        The fixed Phase-1 gait policy (build_obs, act).
    bout_steps : int
        Number of physics control steps per call to `advance`.
    """

    def __init__(self, food_field, runner, bout_steps: int = 20):
        self.food = food_field
        self.runner = runner
        self.env = runner.env
        self.bout_steps = bout_steps
        # JIT the inner pipeline step once; reused across all calls.
        self._step = jax.jit(self.env.pipeline_step)

    def advance(self, pipeline_state):
        """Roll the fixed policy for `bout_steps` control steps.

        Parameters
        ----------
        pipeline_state : brax.mjx.base.State
            Initial physics state (brax MJX pipeline state).

        Returns
        -------
        new_pipeline_state : brax.mjx.base.State
            Physics state after the bout.
        swept_path : list of (float, float)
            Torso (x, y) recorded *after* each step, length == bout_steps.
        """
        state = pipeline_state
        path = []
        for _ in range(self.bout_steps):
            x, y = float(state.q[0]), float(state.q[1])
            food_xy = self.food.nearest_food_xy(x, y)
            obs = self.runner.build_obs(state, food_xy)
            action = self.runner.act(obs)
            state = self._step(state, action)
            path.append((float(state.q[0]), float(state.q[1])))
        return state, path

    def spawn_pipeline_state(self, xy, seed: int):
        """Reset the env and translate the root to `xy`.

        Parameters
        ----------
        xy : (float, float) or array-like
            World (x, y) coordinates for the torso.
        seed : int
            RNG seed for the reset.

        Returns
        -------
        pipeline_state : brax.mjx.base.State
            Physics state with root translated to `xy`; orientation and joints
            taken from the default reset pose.
        """
        s = self.env.reset(jax.random.PRNGKey(seed)).pipeline_state
        q = s.q.at[0:2].set(jnp.asarray(xy))   # set root translation (free-joint)
        return s.replace(q=q)                    # immutable pytree → rebuild
