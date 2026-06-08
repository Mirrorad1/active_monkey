"""LangModel: a pymdp active-inference HMM over characters (learn / score / generate)."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from pymdp.agent import Agent

from active_loop.alphabet import V, encode, decode
from active_loop.lang_model_spec import build_lang_arrays, LANG_DIMS

_NOOP = jnp.array([[0]])  # (batch, num_factors)


class LangModel:
    def __init__(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)
        A, B, D, pA, pB = build_lang_arrays(seed=seed)
        self.agent = Agent(
            A=A, B=B, D=D, pA=pA, pB=pB,
            num_controls=LANG_DIMS.num_controls,
            policy_len=1, action_selection="deterministic",
            sampling_mode="full", inference_algo="fpi",
            batch_size=1, learn_A=True, learn_B=True,
        )

    def _predict(self, prior_vec: np.ndarray) -> np.ndarray:
        Amat = np.asarray(self.agent.A[0])[0]  # (V, K)
        pred = Amat @ prior_vec
        return pred / pred.sum()

    def mean_surprise(self, text: str) -> float:
        """Mean predictive surprise (nats/char) over text; no learning."""
        obs = encode(text)
        prior = [self.agent.D[0]]
        total = 0.0
        for o in obs:
            pred = self._predict(np.asarray(prior[0]).reshape(-1))
            total += -np.log(pred[o] + 1e-12)
            qs = self.agent.infer_states([jnp.array([o])], prior)
            prior = self.agent.update_empirical_prior(_NOOP, qs)
        return total / max(len(obs), 1)

    def learn_stream(self, text: str, epochs: int = 1) -> None:
        obs = encode(text)
        for _ in range(epochs):
            prior = [self.agent.D[0]]
            qs_seq, act_seq = [], []
            for o in obs:
                qs = self.agent.infer_states([jnp.array([o])], prior)
                qs_seq.append(qs)
                prior = self.agent.update_empirical_prior(_NOOP, qs)
                act_seq.append(_NOOP)
            T = len(obs)
            beliefs = [jnp.concatenate([qs_seq[t][0] for t in range(T)], axis=1)]
            observations = [jnp.array([[obs[t] for t in range(T)]])]
            actions = jnp.concatenate([a[:, None, :] for a in act_seq], axis=1)
            self.agent = self.agent.infer_parameters(
                beliefs_A=beliefs, observations=observations,
                actions=actions, beliefs_B=beliefs,
            )

    def generate(self, prefix: str, n: int) -> str:
        prior = [self.agent.D[0]]
        for o in encode(prefix):
            qs = self.agent.infer_states([jnp.array([o])], prior)
            prior = self.agent.update_empirical_prior(_NOOP, qs)
        Amat = np.asarray(self.agent.A[0])[0]           # (V, K)
        Bmat = np.asarray(self.agent.B[0])[0, :, :, 0]   # (K, K) next<-cur
        state = np.asarray(prior[0]).reshape(-1)
        out = []
        for _ in range(n):
            pred = Amat @ state; pred = pred / pred.sum()
            self._key, sub = jax.random.split(self._key)
            c = int(jax.random.choice(sub, V, p=jnp.asarray(pred)))
            out.append(c)
            self._key, sub = jax.random.split(self._key)
            s = int(jax.random.choice(sub, len(state), p=jnp.asarray(state)))
            onehot = np.zeros(len(state)); onehot[s] = 1.0
            state = Bmat @ onehot; state = state / state.sum()
        return decode(out)
