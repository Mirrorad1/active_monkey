"""Tests for Comm v0 (costed sender/receiver signaling). Fast + deterministic."""
from __future__ import annotations

from active_loop.benchmarks.comm_v0 import (
    run_comm_v0, CommReport, _mutual_information, _SignalingPair,
)
import numpy as np


def test_report_shape_and_verdict_domain():
    rep = run_comm_v0(seeds=(0, 1), train_turns=800, eval_turns=200)
    assert isinstance(rep, CommReport)
    assert rep.verdict in {"PASS", "FAIL", "INCONCLUSIVE"}
    assert set(rep.checks) == {
        "beats_shuffled", "beats_muted", "carries_information", "receiver_uses_message",
    }


def test_normal_beats_controls_and_carries_information():
    rep = run_comm_v0(seeds=tuple(range(4)), train_turns=3000, eval_turns=400)
    # a learned protocol should beat its shuffled and muted controls
    assert rep.reward_normal > rep.reward_shuffled
    assert rep.reward_normal > rep.reward_muted
    # and the message should carry nonzero information about the hidden state
    assert rep.mutual_information_bits > 0.0
    assert rep.receiver_policy_sensitivity >= 0.0


def test_mutual_information_bounds():
    # perfectly correlated z,m -> MI == H(m); independent -> MI ~ 0
    zs = [0, 1, 2, 3] * 25
    ms = list(zs)
    mi, hm = _mutual_information(zs, ms, 4, 4)
    assert abs(mi - hm) < 1e-6
    assert mi > 1.5  # ~2 bits for 4 equiprobable aligned symbols

    rng = np.random.default_rng(0)
    zs2 = list(rng.integers(0, 4, size=400))
    ms2 = list(rng.integers(0, 4, size=400))
    mi2, _ = _mutual_information(zs2, ms2, 4, 4)
    assert mi2 < 0.2  # near-independent -> low MI


def test_deterministic():
    a = run_comm_v0(seeds=(0, 1), train_turns=800, eval_turns=200)
    b = run_comm_v0(seeds=(0, 1), train_turns=800, eval_turns=200)
    assert a.reward_normal == b.reward_normal
    assert a.verdict == b.verdict


def test_signaling_pair_learns():
    rng = np.random.default_rng(0)
    pair = _SignalingPair(4, 4, 4, rng)
    target = np.arange(4)
    for _ in range(3000):
        z = int(rng.integers(0, 4))
        m = pair.send(z)
        a = pair.receive(m)
        pair.learn(z, m, a, (1.0 if a == target[z] else 0.0) - 0.05)
    # after training, greedy decode should be correct for most states
    correct = sum(pair.greedy_receive(pair.greedy_send(z)) == target[z] for z in range(4))
    assert correct >= 3
