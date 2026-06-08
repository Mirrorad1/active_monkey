"""The active-inference controller: wraps a pymdp Agent and chooses ACT/ASK/SWITCH by EFE."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp

from pymdp.agent import Agent

from active_loop.model_spec import build_controller_arrays, DIMS


class Action(IntEnum):
    ACT = 0
    ASK = 1
    SWITCH = 2


@dataclass
class ControlOutput:
    action: Action
    raw_action: jnp.ndarray
    qs: list
    neg_efe: jnp.ndarray
    q_pi: jnp.ndarray


class Controller:
    def __init__(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)
        A, B, C, D, pA, pB = build_controller_arrays()
        self._D = D
        self.agent = Agent(
            A=A, B=B, C=C, D=D, pA=pA, pB=pB,
            num_controls=DIMS.num_controls,
            policy_len=1,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algo="fpi",
            batch_size=DIMS.batch_size,
            learn_A=True,
            learn_B=True,
        )
        self._prior = None

    def reset(self) -> None:
        self._prior = self._D

    def step(self, obs: list[int]) -> ControlOutput:
        if self._prior is None:
            self.reset()
        obs_batched = [jnp.array([o] * DIMS.batch_size) for o in obs]
        qs = self.agent.infer_states(obs_batched, self._prior)
        q_pi, neg_efe = self.agent.infer_policies(qs)
        self._key, subkey = jax.random.split(self._key)
        batched_key = jax.random.split(subkey, DIMS.batch_size)
        action = self.agent.sample_action(q_pi, rng_key=batched_key)
        action_idx = int(jnp.asarray(action).reshape(-1)[0])
        self._prior = self.agent.update_empirical_prior(action, qs)
        return ControlOutput(
            action=Action(action_idx),
            raw_action=action,
            qs=qs,
            neg_efe=neg_efe,
            q_pi=q_pi,
        )
