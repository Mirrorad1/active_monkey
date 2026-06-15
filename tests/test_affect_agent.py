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

from active_loop.affect_spec import build_dyad_model, LV, NEU, POS
from active_loop.affect_agent import AffectAgent


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
