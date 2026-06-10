"""M4a AffectAgent: perceive -> intent posterior; EFE response selection;
windowed Dirichlet learning.  Functional valence only; no sentience claims.
pB learned as of increment 1b (Exp 127, human-authorized resumption).
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
        pB = model_dict["pB"]

        # Fix num_controls: B[0] has shape (1, K, K, R); last dim = R
        R = int(jnp.array(B[0]).shape[-1])
        self.agent = Agent(
            A=A, B=B, C=C, D=D, pA=pA, pB=pB,
            num_controls=[R],
            policy_len=1,
            action_selection="stochastic",
            sampling_mode="full",
            inference_algo="fpi",
            batch_size=1,
            learn_A=True,
            learn_B=True,   # pB learned as of increment 1b (Exp 127)
        )

        # empirical prior starts at D
        self._prior = D
        # snapshot initial pA[0] mass for window arithmetic (Exp 125 P4)
        self._init_pA0_sum = float(jnp.array(self.agent.pA[0]).sum())

        # keep the last qs, prev_qs, obs, and action for infer_parameters
        self._last_qs: list | None = None
        self._prev_qs: list | None = None   # qs from the turn BEFORE the last action
        self._last_obs: list[jnp.ndarray] | None = None
        self._last_action: jnp.ndarray | None = None

    # ── Perception ────────────────────────────────────────────────────────────

    def perceive(self, code: int, valence_idx: int = 1) -> np.ndarray:
        """Infer intent posterior given utterance code and last valence.

        Returns qs as a flat numpy array of shape (K,), summing to 1.
        """
        # Rotate: current last_qs becomes prev_qs before computing the new one
        self._prev_qs = self._last_qs
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
        """Windowed Dirichlet A- and B-update: decay existing counts then add new mass.

        The window (LV=0.999, Exp 85-91) is applied by scaling pA and pB before the
        pymdp infer_parameters call, which adds qs-weighted observation/transition mass.
        pB is learned as of increment 1b (Exp 127, human-authorized resumption).

        B-learning requires two consecutive belief frames (prev_qs at t-1 and
        last_qs at t) plus the action taken between them.  When prev_qs is not yet
        available (first turn) the B update is silently skipped for that turn only.
        """
        if self._last_qs is None or self._last_action is None:
            return

        # ── 1. Decay existing Dirichlet counts (the window) ──────────────────
        # infer_parameters returns a new immutable Agent; we rebuild with decayed
        # pA and pB before the update so the new mass is added on top of the
        # already-decayed concentrations.
        decayed_pA = [
            jnp.array(self.agent.pA[0]) * self.lv,
            jnp.array(self.agent.pA[1]) * self.lv,
        ]
        decayed_pB = [jnp.array(self.agent.pB[0]) * self.lv]

        A  = self.agent.A
        B  = self.agent.B
        C  = self.agent.C
        D_ = self.agent.D
        R  = int(jnp.array(B[0]).shape[-1])
        tmp = Agent(
            A=A, B=B, C=C, D=D_, pA=decayed_pA, pB=decayed_pB,
            num_controls=[R], policy_len=1,
            action_selection="stochastic", sampling_mode="full",
            inference_algo="fpi", batch_size=1,
            learn_A=True, learn_B=True,
        )

        # ── 2. Add new observation mass via infer_parameters ─────────────────
        # beliefs_A: current qs (1 frame) — shapes (1, 1, K) each factor
        updated_obs = [jnp.array([[code]]), jnp.array([[valence_idx]])]
        # action from act(): shape (1, num_factors); expand to (1, 1, num_factors)
        action_seq = self._last_action[:, None, :]  # (1, 1, num_factors)

        # ── 3. B-learning: two-frame belief sequence ──────────────────────────
        # beliefs_B needs shape (1, T=2, K) so the outer-product joint covers the
        # t-1 -> t transition.  We can only do this when _prev_qs is available
        # (i.e. from turn 2 onward).
        if self._prev_qs is not None:
            # each _prev_qs / _last_qs factor is (1, 1, K) from infer_states
            beliefs_B = [
                jnp.concatenate([self._prev_qs[0], self._last_qs[0]], axis=1)
            ]  # list of (1, 2, K)
        else:
            beliefs_B = None  # first turn: skip B update this step

        updated = tmp.infer_parameters(
            beliefs_A=self._last_qs,   # list[(1, 1, K)]
            observations=updated_obs,  # list[(1, 1)] per modality
            actions=action_seq,        # (1, 1, num_factors)
            beliefs_B=beliefs_B,       # list[(1, 2, K)] or None
            lr_pA=1.0,
            lr_pB=1.0,
        )
        self.agent = updated

    # ── Diagnostics ──────────────────────────────────────────────────────────

    def pA0_sum(self) -> float:
        """Current sum of pA[0] concentration array (for window arithmetic)."""
        return float(jnp.array(self.agent.pA[0]).sum())

    def pB0_sum(self) -> float:
        """Current sum of pB[0] concentration array (diagnostic for B-learning)."""
        return float(jnp.array(self.agent.pB[0]).sum())

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
