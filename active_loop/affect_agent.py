"""M4a AffectAgent: perceive -> intent posterior; EFE response selection;
windowed Dirichlet learning.  Functional valence only; no sentience claims.
"""
from __future__ import annotations

import math
import numpy as np
import jax
import jax.numpy as jnp

from pymdp.agent import Agent

from active_loop.affect_spec import LV, K


class AffectAgent:
    """Wraps a pymdp JAX Agent with the M4a dyadic generative model.

    Construction mirrors controller.py: passes A, B, C, D, pA; enables
    learn_A; uses infer_states / infer_policies / sample_action /
    update_empirical_prior / infer_parameters following the verified
    pymdp conventions.
    """

    def __init__(self, model_dict: dict, lv: float = LV, seed: int = 0):
        self.lv = lv
        self._key = jax.random.PRNGKey(seed)

        A  = model_dict["A"]
        B  = model_dict["B"]
        C  = model_dict["C"]
        D  = model_dict["D"]
        pA = model_dict["pA"]

        self.agent = Agent(
            A=A, B=B, C=C, D=D, pA=pA,
            num_controls=[len(model_dict["B"])],  # R inferred from B[0].shape[-1]
            policy_len=1,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algo="fpi",
            batch_size=1,
            learn_A=True,
            learn_B=False,   # pB not learned in increment 1 (reserved for increment 2)
        )

        # Fix num_controls: B[0] has shape (1, K, K, R); last dim = R
        R = int(jnp.array(B[0]).shape[-1])
        self.agent = Agent(
            A=A, B=B, C=C, D=D, pA=pA,
            num_controls=[R],
            policy_len=1,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algo="fpi",
            batch_size=1,
            learn_A=True,
            learn_B=False,
        )

        # empirical prior starts at D
        self._prior = D
        # snapshot initial pA[0] mass for window arithmetic (Exp 125 P4)
        self._init_pA0_sum = float(jnp.array(self.agent.pA[0]).sum())

        # keep the last qs and obs for infer_parameters
        self._last_qs: list | None = None
        self._last_obs: list[jnp.ndarray] | None = None
        self._last_action: jnp.ndarray | None = None

    # ── Perception ────────────────────────────────────────────────────────────

    def perceive(self, code: int, valence_idx: int = 1) -> np.ndarray:
        """Infer intent posterior given utterance code and last valence.

        Returns qs as a flat numpy array of shape (K,), summing to 1.
        """
        obs = [jnp.array([code]), jnp.array([valence_idx])]
        qs = self.agent.infer_states(obs, self._prior)
        self._last_qs = qs
        self._last_obs = [jnp.array([[code]]), jnp.array([[valence_idx]])]
        return np.array(qs[0]).reshape(-1)  # (K,)

    # ── Action ────────────────────────────────────────────────────────────────

    def act(self) -> int:
        """Select response by EFE (pragmatic + epistemic).

        Returns response index ∈ {0..R-1}.  ASK (index 4) is selected when
        epistemic value is high — the exploration reflex.
        """
        if self._last_qs is None:
            raise RuntimeError("call perceive() before act()")
        q_pi, _neg_efe = self.agent.infer_policies(self._last_qs)
        self._key, sk = jax.random.split(self._key)
        sk_batch = jax.random.split(sk, 1)
        action = self.agent.sample_action(q_pi, rng_key=sk_batch)
        self._last_action = action
        # update empirical prior for the next turn
        self._prior = self.agent.update_empirical_prior(action, self._last_qs)
        return int(jnp.asarray(action).reshape(-1)[0])

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe_feedback(self, code: int, valence_idx: int) -> None:
        """Windowed Dirichlet A-update: decay existing counts then add new mass.

        The window (LV=0.999, Exp 85-91) is applied by scaling pA before the
        pymdp infer_parameters call, which adds qs-weighted observation mass.
        pB is NOT updated in increment 1 (reserved for increment 2).
        """
        if self._last_qs is None or self._last_action is None:
            return

        # ── 1. Decay existing Dirichlet counts (the window) ──────────────────
        # infer_parameters returns a new immutable Agent; we mutate pA on the
        # current agent by rebuilding with decayed pA before the update.
        decayed_pA = [
            jnp.array(self.agent.pA[0]) * self.lv,
            jnp.array(self.agent.pA[1]) * self.lv,
        ]
        # Swap pA in a new agent carrying the decayed concentrations
        A  = self.agent.A
        B  = self.agent.B
        C  = self.agent.C
        D_ = self.agent.D
        R  = int(jnp.array(B[0]).shape[-1])
        tmp = Agent(
            A=A, B=B, C=C, D=D_, pA=decayed_pA,
            num_controls=[R], policy_len=1,
            action_selection="stochastic", sampling_mode="full",
            inference_algo="fpi", batch_size=1,
            learn_A=True, learn_B=False,
        )

        # ── 2. Add new observation mass via infer_parameters ─────────────────
        # Update obs slot to the actual feedback valence
        updated_obs = [jnp.array([[code]]), jnp.array([[valence_idx]])]
        updated = tmp.infer_parameters(
            beliefs_A=self._last_qs,   # list[(1, 1, K)]
            observations=updated_obs,  # list[(1, 1)] per modality
            actions=self._last_action[:, None, :],  # (1, 1, num_factors)
            lr_pA=1.0,
        )
        self.agent = updated

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def pA0_sum(self) -> float:
        """Current sum of pA[0] concentration array (for window arithmetic)."""
        return float(jnp.array(self.agent.pA[0]).sum())

    def valence_readout(self, qs: np.ndarray | None = None) -> float:
        """Provisional valence readout ∈ [0, 1].  Increment-2 refinement pending.

        0.5 * (1 - H(qs)/ln(K))   — how concentrated/confident the intent belief is
        + 0.5 * P_pos             — probability of positive valence at the MAP intent
                                    (normalized pA[1][POS, argmax(qs)]).
        """
        if qs is None:
            if self._last_qs is None:
                return 0.5
            qs = np.array(self._last_qs[0]).reshape(-1)

        qs = np.array(qs, dtype=float).reshape(-1)
        qs = qs / qs.sum()  # safety

        # entropy term
        h = -np.sum(qs * np.log(qs + 1e-12))
        h_uniform = math.log(K)
        conf = 1.0 - h / h_uniform

        # P_pos at MAP intent
        k_star = int(np.argmax(qs))
        pA1 = np.array(self.agent.pA[1]).reshape(3, K)  # (V, K)
        col = pA1[:, k_star]
        p_pos = col[2] / col.sum()  # POS=2

        return float(np.clip(0.5 * conf + 0.5 * p_pos, 0.0, 1.0))
