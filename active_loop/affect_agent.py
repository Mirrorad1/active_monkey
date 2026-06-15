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

from active_loop.affect_spec import LV, K, NEU


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

        # Prior over intent each turn = D.  Utterances are independent draws from the
        # partner, so there is no across-turn intent carry (Exp 128/1c): perceiving a new
        # utterance starts from D and is sharpened by the learned emission A[0].
        self._prior = D
        # snapshot initial pA[0] mass for window arithmetic (Exp 125 P4)
        self._init_pA0_sum = float(jnp.array(self.agent.pA[0]).sum())

        # Per-turn belief frames for the within-turn credit assignment (1c timing fix):
        #   _qs_perceive = intent from the utterance ALONE (pre-feedback), used by act();
        #   _last_qs     = most recent posterior (post-feedback once learned), for readout;
        #   _last_action = the response chosen this turn.
        self._qs_perceive: list | None = None
        self._last_qs: list | None = None
        self._last_action: jnp.ndarray | None = None

    # ── Perception ────────────────────────────────────────────────────────────

    def perceive(self, code: int) -> np.ndarray:
        """Infer the intent posterior from the utterance ALONE (valence NEU).

        1c timing fix (Exp 128): the previous turn's valence is NOT folded into this
        turn's intent inference — valence is feedback ABOUT the response, observed later
        in observe_feedback and bound to THIS turn's code there.  Prior = D.

        Returns qs as a flat numpy array of shape (K,), summing to 1.
        """
        obs = [jnp.array([code]), jnp.array([NEU])]
        qs = self.agent.infer_states(obs, self._prior)
        self._qs_perceive = qs
        self._last_qs = qs
        return np.array(qs[0]).reshape(-1)  # (K,)

    # ── Action ────────────────────────────────────────────────────────────────

    def act(self) -> int:
        """Select response by EFE (pragmatic + epistemic).

        Returns response index ∈ {0..R-1}.  ASK (index 4) is selected when
        epistemic value is high — the exploration reflex.
        """
        if self._qs_perceive is None:
            raise RuntimeError("call perceive() before act()")
        q_pi, _neg_efe = self.agent.infer_policies(self._qs_perceive)
        self._key, sk = jax.random.split(self._key)
        sk_batch = jax.random.split(sk, 1)
        action = self.agent.sample_action(q_pi, rng_key=sk_batch)
        self._last_action = action
        # The prior stays D (independent utterances); the response's consequence is bound
        # WITHIN this turn by observe_feedback, not carried to the next perceive.
        return int(jnp.asarray(action).reshape(-1)[0])

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe_feedback(self, code: int, valence_idx: int) -> None:
        """Co-present the action's consequence with the turn's own code, then learn.

        THE 1c TIMING FIX (Exp 128).  The partner's contingency is
        (response_t x code_t -> valence_t); the learner must see all three together.
        So we RE-INFER the intent from the FULL turn observation [code_t, valence_t]
        (same prior D as perceive), then:
          - A-update binds [code_t, valence_t] to that post-feedback intent (qs_learn);
          - B-update learns the WITHIN-TURN transition qs_perceive -> qs_learn caused by
            response_t (so a good response is learned to reach a positive-valence intent).
        This is the spec's "perceive -> act -> observe[code,valence] -> learn" flow; the
        action and its consequence are now in the SAME inference step (vs Exp 125/127/128,
        where valence_t was only seen alongside code_{t+1}, after the intent had moved on).

        The window (LV=0.999, Exp 85-91) is applied by scaling pA/pB before infer_parameters.
        """
        if self._qs_perceive is None or self._last_action is None:
            return

        # ── 0. Re-infer the post-feedback intent (the consequence, co-presented) ──
        obs_full = [jnp.array([code]), jnp.array([valence_idx])]
        qs_learn = self.agent.infer_states(obs_full, self._prior)
        self._last_qs = qs_learn

        # ── 1. Decay existing Dirichlet counts (the window) then rebuild ──────────
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

        # ── 2. A-update: bind [code_t, valence_t] to the post-feedback intent ─────
        updated_obs = [jnp.array([[code]]), jnp.array([[valence_idx]])]
        action_seq = self._last_action[:, None, :]  # (1, 1, num_factors)

        # ── 3. B-update: the WITHIN-TURN transition qs_perceive -> qs_learn ───────
        # both factors are (1, 1, K) from infer_states; concat on T -> (1, 2, K).
        beliefs_B = [
            jnp.concatenate([self._qs_perceive[0], qs_learn[0]], axis=1)
        ]

        updated = tmp.infer_parameters(
            beliefs_A=qs_learn,        # list[(1, 1, K)] — post-feedback intent
            observations=updated_obs,  # list[(1, 1)] per modality — [code, valence]
            actions=action_seq,        # (1, 1, num_factors) — response_t
            beliefs_B=beliefs_B,       # list[(1, 2, K)] — within-turn transition
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


# ---------------------------------------------------------------------------
# Increment 1d: DirectHeadAgent — the direct response->valence head (Exp 215)
# ---------------------------------------------------------------------------
class DirectHeadAgent:
    """M4a increment 1d (Exp 215): a 2-factor agent whose valence emission depends DIRECTLY on
    (intent, last_response).  Built from affect_spec.build_direct_head_model.

    Two hidden factors [intent (K), last_response (R)].  The action deterministically sets the
    last_response factor (B1); the valence head A1 = P(valence | intent, last_response) is the
    DIRECT credit path (learnable).  Only A is learned (learn_A=True, learn_B=False) — B0
    (intent identity, uncontrolled) and B1 (deterministic action-set) are fixed structural priors.
    The 1c TIMING discipline is kept: perceive the utterance alone -> act (which sets the
    last_response prior to the chosen response) -> observe [code, valence] co-presented -> learn,
    so the valence is bound to (intent, response_t) in the same inference step.

    Functional valence only — no sentience claim.
    """

    def __init__(self, model_dict: dict, lv: float = LV, seed: int = 0):
        self.lv = lv
        self._key = jax.random.PRNGKey(seed)
        R = int(jnp.array(model_dict["B"][1]).shape[-1])
        self.R = R
        self.agent = Agent(
            A=model_dict["A"], B=model_dict["B"], C=model_dict["C"], D=model_dict["D"],
            pA=model_dict["pA"], pB=model_dict["pB"],
            num_controls=[1, R],           # factor 0 (intent) uncontrolled; factor 1 (last_response) = the response
            policy_len=1, action_selection="stochastic", sampling_mode="full",
            inference_algo="fpi", batch_size=1, learn_A=True, learn_B=False,
        )
        self._D = model_dict["D"]
        self._init_pA1_sum = float(jnp.array(self.agent.pA[1]).sum())  # window arithmetic on the HEAD
        self._prior = self._D
        self._qs_perceive: list | None = None
        self._last_qs: list | None = None
        self._last_action: jnp.ndarray | None = None

    def perceive(self, code: int) -> np.ndarray:
        """Infer intent from the utterance alone (valence NEU); prior D (last_response uniform pre-act)."""
        qs = self.agent.infer_states([jnp.array([code]), jnp.array([NEU])], self._D)
        self._qs_perceive = qs
        self._last_qs = qs
        return np.array(qs[0]).reshape(-1)

    def act(self) -> int:
        """EFE response selection; then set the observe-prior so last_response = the chosen response
        (B1 deterministic; B0 leaves intent unchanged) — so the feedback is bound to THIS response."""
        if self._qs_perceive is None:
            raise RuntimeError("call perceive() before act()")
        q_pi, _ = self.agent.infer_policies(self._qs_perceive)
        self._key, sk = jax.random.split(self._key)
        action = self.agent.sample_action(q_pi, rng_key=jax.random.split(sk, 1))
        self._last_action = action
        self._prior = self.agent.update_empirical_prior(action, self._qs_perceive)
        return int(jnp.asarray(action).reshape(-1)[-1])   # control on factor 1 = the response

    def observe_feedback(self, code: int, valence_idx: int) -> None:
        """Co-present [code_t, valence_t] (1c timing): re-infer with last_response=response_t, then
        windowed-Dirichlet learn A0 (utterance|intent) + A1 (valence|intent,response) — the DIRECT head."""
        if self._qs_perceive is None or self._last_action is None:
            return
        qs_learn = self.agent.infer_states(
            [jnp.array([code]), jnp.array([valence_idx])], self._prior)
        self._last_qs = qs_learn
        # Window decay: swap pA in-place via equinox.tree_at (NOT a full Agent rebuild, which
        # re-traces the constructor and is ~10x slower per turn for the 2-factor model). The
        # resulting state (A=current expected, pA=decayed, B/C/D/pB unchanged) is identical to
        # reconstructing Agent(A=self.agent.A, ..., pA=decayed_pA), so the math is unchanged.
        import equinox as eqx  # noqa: PLC0415 (local import; cheap, avoids a module-level dep)
        decayed_pA = [jnp.array(self.agent.pA[0]) * self.lv,
                      jnp.array(self.agent.pA[1]) * self.lv]
        tmp = eqx.tree_at(lambda a: a.pA, self.agent, decayed_pA)
        updated = tmp.infer_parameters(
            beliefs_A=qs_learn,
            observations=[jnp.array([[code]]), jnp.array([[valence_idx]])],
            actions=self._last_action[:, None, :],   # required positional; B is NOT learned (learn_B=False)
            beliefs_B=None,
            lr_pA=1.0,
        )
        self.agent = updated

    def pA1_sum(self) -> float:
        return float(jnp.array(self.agent.pA[1]).sum())
