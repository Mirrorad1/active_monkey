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

    def learn(self, trajectory) -> None:
        """Dirichlet-update A and B from a finished episode via pymdp's infer_parameters.

        infer_parameters returns a NEW agent (equinox modules are immutable), so we
        store it back. Signature (verified):
            infer_parameters(beliefs_A, observations, actions, beliefs_B=None, lr_pA, lr_pB)
        Shapes:
            beliefs_A:    list per factor of (batch, T, num_states_f)
            observations: list per modality of (batch, T) int indices
            actions:      (batch, T, num_factors)
        Passing beliefs_B=beliefs enables B learning as well.
        """
        T = len(trajectory.obs_seq)
        num_factors = len(DIMS.num_states)
        num_modalities = len(DIMS.num_obs)
        beliefs = [
            jnp.concatenate([trajectory.qs_seq[t][f] for t in range(T)], axis=1)
            for f in range(num_factors)
        ]
        observations = [
            jnp.array([[trajectory.obs_seq[t][m] for t in range(T)]])
            for m in range(num_modalities)
        ]
        actions = jnp.concatenate([a[:, None, :] for a in trajectory.action_seq], axis=1)
        self.agent = self.agent.infer_parameters(
            beliefs_A=beliefs,
            observations=observations,
            actions=actions,
            beliefs_B=beliefs,
        )
