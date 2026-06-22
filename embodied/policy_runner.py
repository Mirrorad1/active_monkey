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
