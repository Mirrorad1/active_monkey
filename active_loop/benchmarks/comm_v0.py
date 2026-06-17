"""Comm v0 — costed sender/receiver signaling (proto-communication, NOT language).

A Lewis-style referential signaling game at toy scale:
  - the world has a hidden state ``z`` (Z values);
  - a SENDER observes z and emits a message ``m`` (M symbols) at a per-message cost;
  - a RECEIVER sees only m (never z) and chooses an action ``a`` (A choices);
  - reward = 1 if a matches the correct action for z, else 0, minus message cost.

There is NO reward for any particular token identity and NO hardcoded English / semantic
label — sender and receiver learn a joint mapping by simple Roth-Erev reinforcement.
We then measure whether the learned signal actually carries hidden-state information and
changes receiver behavior, against shuffled / muted / permuted / random / constant
controls.

This is "costed signaling" / a proto-communication benchmark — NOT language.

Verdict PASS requires (all, across seeds):
  - reward_normal beats shuffled AND muted controls by a predeclared delta;
  - empirical mutual information I(m; z) is nonzero;
  - the receiver's action changes with the message (receiver_policy_sensitivity > 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

# Predeclared verdict thresholds (frozen at v0)
DELTA_SHUFFLED = 0.10
DELTA_MUTED = 0.10
MI_FLOOR = 0.05            # bits
SENSITIVITY_FLOOR = 0.10  # fraction of message pairs that map to different actions


@dataclass
class CommReport:
    reward_normal: float
    reward_shuffled: float
    reward_muted: float
    reward_permuted: float
    reward_random_sender: float
    reward_constant_sender: float
    reward_delta_shuffled: float
    reward_delta_muted: float
    mutual_information_bits: float
    message_entropy_bits: float
    receiver_policy_sensitivity: float
    n_seeds: int
    train_turns: int
    eval_turns: int
    cost: float
    checks: dict = field(default_factory=dict)
    verdict: str = "INCONCLUSIVE"

    def to_dict(self) -> dict:
        return asdict(self)


class _SignalingPair:
    """Roth-Erev reinforcement sender (z->m) and receiver (m->a), seedable."""

    def __init__(self, Z: int, M: int, A: int, rng: np.random.Generator, lr: float = 0.2):
        self.Z, self.M, self.A = Z, M, A
        self.rng = rng
        self.lr = lr
        self.q_send = np.ones((Z, M))   # propensities
        self.q_recv = np.ones((M, A))

    def _softmax_choice(self, w: np.ndarray) -> int:
        p = w / w.sum()
        return int(self.rng.choice(len(w), p=p))

    def send(self, z: int) -> int:
        return self._softmax_choice(self.q_send[z])

    def receive(self, m: int) -> int:
        return self._softmax_choice(self.q_recv[m])

    def learn(self, z: int, m: int, a: int, reward: float) -> None:
        self.q_send[z, m] = max(1e-6, self.q_send[z, m] + self.lr * reward)
        self.q_recv[m, a] = max(1e-6, self.q_recv[m, a] + self.lr * reward)

    def greedy_send(self, z: int) -> int:
        return int(np.argmax(self.q_send[z]))

    def greedy_receive(self, m: int) -> int:
        return int(np.argmax(self.q_recv[m]))


def _correct_action(Z: int) -> np.ndarray:
    """The (fixed, public) z->correct-action target.  Identity for Z<=A."""
    return np.arange(Z)


def _train(pair: _SignalingPair, target: np.ndarray, turns: int, cost: float,
           rng: np.random.Generator) -> None:
    Z = pair.Z
    for _ in range(turns):
        z = int(rng.integers(0, Z))
        m = pair.send(z)
        a = pair.receive(m)
        reward = (1.0 if a == target[z] else 0.0) - cost
        pair.learn(z, m, a, reward)


def _eval_condition(pair: _SignalingPair, target: np.ndarray, turns: int, cost: float,
                    rng: np.random.Generator, mode: str) -> dict:
    """Evaluate the trained pair under a control condition; return reward + traces.

    mode: 'normal' | 'shuffled' | 'muted' | 'permuted' | 'random_sender' | 'constant_sender'
    """
    Z, M, A = pair.Z, pair.M, pair.A
    perm = rng.permutation(M)  # fixed permutation for 'permuted'
    total = 0.0
    zs: list[int] = []
    ms: list[int] = []  # message actually delivered to receiver
    for _ in range(turns):
        z = int(rng.integers(0, Z))
        if mode == "random_sender":
            m_sent = int(rng.integers(0, M))
        elif mode == "constant_sender":
            m_sent = 0
        else:
            m_sent = pair.greedy_send(z)

        if mode == "shuffled":
            m_deliver = int(rng.integers(0, M))   # message decoupled from z
        elif mode == "muted":
            m_deliver = 0                          # constant signal
        elif mode == "permuted":
            m_deliver = int(perm[m_sent])          # relabeled ids
        else:
            m_deliver = m_sent

        a = pair.greedy_receive(m_deliver)
        total += (1.0 if a == target[z] else 0.0) - (0.0 if mode == "muted" else cost)
        zs.append(z); ms.append(m_sent)
    return {"reward": total / turns, "zs": zs, "ms": ms}


def _mutual_information(zs: list[int], ms: list[int], Z: int, M: int) -> tuple[float, float]:
    """Empirical I(m; z) and H(m) in bits from paired samples."""
    joint = np.zeros((Z, M))
    for z, m in zip(zs, ms):
        joint[z, m] += 1
    n = joint.sum()
    if n == 0:
        return 0.0, 0.0
    p = joint / n
    pz = p.sum(axis=1, keepdims=True)
    pm = p.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(p * np.log2(p / (pz @ pm + 1e-12) + 1e-12))
        pm_flat = pm.flatten()
        hm = -np.nansum(pm_flat * np.log2(pm_flat + 1e-12))
    return float(max(0.0, mi)), float(max(0.0, hm))


def _receiver_sensitivity(pair: _SignalingPair) -> float:
    """Fraction of message pairs (m1,m2) that map to DIFFERENT greedy actions."""
    acts = [pair.greedy_receive(m) for m in range(pair.M)]
    pairs = 0
    diff = 0
    for i in range(pair.M):
        for j in range(i + 1, pair.M):
            pairs += 1
            diff += (acts[i] != acts[j])
    return diff / pairs if pairs else 0.0


def _run_seed(seed: int, Z: int, M: int, A: int, train_turns: int, eval_turns: int,
              cost: float) -> dict:
    rng = np.random.default_rng(seed)
    target = _correct_action(Z)
    pair = _SignalingPair(Z, M, A, rng)
    _train(pair, target, train_turns, cost, rng)

    modes = ["normal", "shuffled", "muted", "permuted", "random_sender", "constant_sender"]
    res = {mode: _eval_condition(pair, target, eval_turns, cost, rng, mode) for mode in modes}
    mi, hm = _mutual_information(res["normal"]["zs"], res["normal"]["ms"], Z, M)
    return {
        **{f"rew_{mode}": res[mode]["reward"] for mode in modes},
        "mi": mi,
        "hm": hm,
        "sensitivity": _receiver_sensitivity(pair),
    }


def run_comm_v0(
    seeds: tuple[int, ...] = tuple(range(8)),
    Z: int = 4, M: int = 4, A: int = 4,
    train_turns: int = 4000, eval_turns: int = 600, cost: float = 0.05,
) -> CommReport:
    """Run Comm v0 over a seed ensemble; honest PASS/FAIL/INCONCLUSIVE verdict."""
    rows = [_run_seed(s, Z, M, A, train_turns, eval_turns, cost) for s in seeds]

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    r_normal = mean("rew_normal")
    r_shuf = mean("rew_shuffled")
    r_mute = mean("rew_muted")
    d_shuf = r_normal - r_shuf
    d_mute = r_normal - r_mute
    mi = mean("mi")
    hm = mean("hm")
    sens = mean("sensitivity")

    checks = {
        "beats_shuffled": d_shuf > DELTA_SHUFFLED,
        "beats_muted": d_mute > DELTA_MUTED,
        "carries_information": mi > MI_FLOOR,
        "receiver_uses_message": sens > SENSITIVITY_FLOOR,
    }
    if all(checks.values()):
        verdict = "PASS"
    elif checks["carries_information"] or checks["beats_shuffled"]:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    return CommReport(
        reward_normal=r_normal,
        reward_shuffled=r_shuf,
        reward_muted=r_mute,
        reward_permuted=mean("rew_permuted"),
        reward_random_sender=mean("rew_random_sender"),
        reward_constant_sender=mean("rew_constant_sender"),
        reward_delta_shuffled=d_shuf,
        reward_delta_muted=d_mute,
        mutual_information_bits=mi,
        message_entropy_bits=hm,
        receiver_policy_sensitivity=sens,
        n_seeds=len(seeds),
        train_turns=train_turns,
        eval_turns=eval_turns,
        cost=cost,
        checks=checks,
        verdict=verdict,
    )


def main() -> None:  # pragma: no cover - manual CLI
    import json
    print(json.dumps(run_comm_v0().to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
