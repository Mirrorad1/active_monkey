"""Fast, deterministic guards for the Exp 226 find-and-keep harness logic.

These cover the anti-gaming critic and the candidate set WITHOUT running any JAX
session (importing the module pulls in pymdp/eval, but no agent is constructed here),
so they are fast and not subject to the XLA JIT memory limits that gate a full run.
"""
from __future__ import annotations

from experiments import exp226_autonomous_find_and_keep as H


def test_candidate_set_includes_honest_and_gaming():
    names = {c.name for c in H.generate_candidates()}
    kinds = {c.kind for c in H.generate_candidates()}
    assert "c1_neu_-0.5" in names        # the Exp 225 honest improving move
    assert "a0_code_to_intent" in names  # the gaming candidate the critic must reject
    assert "A0_code_to_intent_prior" in kinds


def test_critic_rejects_a0_gaming_and_approves_preference():
    cands = {c.name: c for c in H.generate_candidates()}
    ok, reason = H.critic_review(cands["a0_code_to_intent"])
    assert ok is False and "gaming" in reason.lower() or "leakage" in reason.lower()

    ok2, reason2 = H.critic_review(cands["c1_neu_-0.5"])
    assert ok2 is True and "APPROVE" in reason2

    ok3, _ = H.critic_review(cands["optimism_3.0"])
    assert ok3 is True


def test_a0_cheat_can_never_be_kept_by_construction():
    """The gaming kind is in the rejected set, so the loop can never keep it."""
    assert "A0_code_to_intent_prior" in H._REJECTED_KINDS
    assert "preference" in H._APPROVED_KINDS
    # a rejected kind is not approvable
    for c in H.generate_candidates():
        approved, _ = H.critic_review(c)
        if c.kind in H._REJECTED_KINDS:
            assert approved is False
