"""tests/test_affect_agent.py — M4a AffectAgent wiring guards (Exp 214 / increment 1c).

Durable guards (loop/META.md) institutionalised after Exp 214: the affect EFE/policy and the
1c timing re-wire must keep working, because a silent regression there would turn every affect
experiment into a FALSE NEGATIVE (the agent appears not to learn when really the instrument
broke).  These tests do NOT assert the agent learns (it does not, at this scale — that is the
honest Exp 214 result); they assert the MACHINERY is sound.
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import jax.numpy as jnp

from active_loop.affect_spec import build_dyad_model, build_direct_head_model, LV, NEU, POS, U, R
from active_loop.affect_agent import AffectAgent, DirectHeadAgent


def test_efe_liveness_on_gifted_model():
    """INSTRUMENT SOUNDNESS (the Exp 214 control): given a DISCRIMINATIVE model — intent 0
    emits POS strongly and response 0 (GREET) drives every intent -> intent 0 — the EFE
    policy must PREFER response 0.  If this fails, the EFE/policy is disconnected from A/B
    and any 'agent does not learn' result is a wiring bug, not science."""
    m = build_dyad_model(0)
    A1 = np.full((3, 4), 0.1); A1[POS, 0] = 5.0; A1 = A1 / A1.sum(0, keepdims=True)
    m["A"][1] = jnp.array(A1[None]); m["pA"][1] = jnp.array((A1 * 1 + 0.1)[None])
    B = np.array(m["B"][0])[0]; B[:, :, 0] = 0.0; B[0, :, 0] = 1.0
    m["B"][0] = jnp.array(B[None]); m["pB"][0] = jnp.array((B * 1 + 0.1)[None])
    ag = AffectAgent(m, lv=LV, seed=0)
    for code in range(3):
        qs = ag.agent.infer_states([jnp.array([code]), jnp.array([NEU])], ag._prior)
        q_pi, _ = ag.agent.infer_policies(qs)
        q0 = float(np.array(q_pi).reshape(-1)[0])
        assert q0 > 0.4, f"EFE does not prefer the POS-reaching response (q_pi[0]={q0:.3f}); wiring bug"


def test_perceive_uses_utterance_alone():
    """1c timing fix: perceive() takes ONLY the code (no valence carry) and returns a proper
    intent posterior.  Guards against re-introducing the stale-valence leak (Exp 128)."""
    params = list(inspect.signature(AffectAgent.perceive).parameters)
    assert params == ["self", "code"], f"perceive signature regressed: {params}"
    ag = AffectAgent(build_dyad_model(1), lv=LV, seed=1)
    qs = ag.perceive(3)
    assert qs.shape[0] == 4 and abs(qs.sum() - 1.0) < 1e-5 and not np.isnan(qs).any()


def test_window_arithmetic_and_session_runs():
    """A short session runs end-to-end and the LV-decay window arithmetic is exact (Exp 125 P4):
    pA[0].sum() == init*LV^n + sum_{k<n} LV^k.  Guards the Dirichlet window wiring."""
    ag = AffectAgent(build_dyad_model(2), lv=LV, seed=2)
    init = ag._init_pA0_sum
    n = 20
    rng = np.random.default_rng(2)
    for t in range(n):
        code = int(rng.integers(0, 6))
        ag.perceive(code)
        r = ag.act()
        val = POS if r == (code % 4) else NEU
        ag.observe_feedback(code, val)
    expected = init * (LV ** n) + sum(LV ** k for k in range(n))
    assert abs(ag.pA0_sum() - expected) < 0.5, f"window arithmetic off: {ag.pA0_sum():.3f} vs {expected:.3f}"
    assert ag.pB0_sum() > 0  # B-learning accumulated mass


# ---------------------------------------------------------------------------
# Increment 1d: DirectHeadAgent (2-factor direct response->valence head) wiring guards
# ---------------------------------------------------------------------------

def test_direct_head_efe_liveness():
    """INSTRUMENT SOUNDNESS (Exp 215 control): with a CLEANLY-gifted direct-head model —
    code0→intent0 (A0) and A1[intent0, resp2]→POS — the 2-factor EFE must PREFER resp2.
    Guards the direct head + the [1,R] control wiring against a silent disconnect."""
    import jax.numpy as jnp
    from pymdp.agent import Agent
    m = build_direct_head_model(0)
    A0 = np.array(m["A"][0])[0]; A0[:] = 0.02; A0[0, 0, :] = 0.9
    A0 = A0 / A0.sum(0, keepdims=True); m["A"][0] = jnp.array(A0[None]); m["pA"][0] = jnp.array((A0 + 0.1)[None])
    A1 = np.array(m["A"][1])[0]; A1[:, 0, 2] = [0.05, 0.05, 0.9]
    A1 = A1 / A1.sum(0, keepdims=True); m["A"][1] = jnp.array(A1[None]); m["pA"][1] = jnp.array((A1 + 0.1)[None])
    ag = Agent(A=m["A"], B=m["B"], C=m["C"], D=m["D"], pA=m["pA"], pB=m["pB"],
               num_controls=[1, R], policy_len=1, action_selection="stochastic", sampling_mode="full",
               inference_algo="fpi", batch_size=1, learn_A=True, learn_B=False)
    qs = ag.infer_states([jnp.array([0]), jnp.array([NEU])], ag.D)
    qp = np.array(ag.infer_policies(qs)[0]).reshape(-1)
    assert int(np.argmax(qp)) == 2 and qp[2] > 0.4, f"direct-head EFE wiring bug: q_pi={np.round(qp,3)}"


def test_direct_head_agent_runs_and_learns_A1():
    """A short DirectHeadAgent session runs end-to-end and the valence head A1 accumulates mass
    (learn_A on the 2-factor model); the chosen response is a valid index."""
    ag = DirectHeadAgent(build_direct_head_model(3), lv=LV, seed=3)
    rng = np.random.default_rng(3)
    for _ in range(8):
        code = int(rng.integers(0, U))
        ag.perceive(code)
        r = ag.act()
        assert 0 <= r < R
        ag.observe_feedback(code, POS if r == (code % 4) else NEU)
    assert ag.pA1_sum() > ag._init_pA1_sum - 1.0  # the head ledger is live
