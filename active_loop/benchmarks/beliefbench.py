"""BeliefBench v0 — hidden partner-type inference (belief-like state).

A partner has a hidden type ``z`` (friendly / neutral / adversarial / inconsistent).
Each turn the agent receives an ambiguous observation drawn from the partner's
type-conditioned emission, maintains a posterior ``q(z)`` by exact Bayesian filtering,
and chooses an action (respond_A / respond_B / probe / wait).  ``probe`` pays a small
cost to receive a *sharper* observation next turn (an information-gathering action).

Belief is only counted as **load-bearing** if all of:
  1. evidence update    — q(z) changes when observations change (belief_update_magnitude > 0)
  2. action relevance    — the policy changes as q(z) changes (policy_belief_sensitivity > 0)
  3. scramble control    — shuffling q(z) hurts reward (reward_delta_scrambled > threshold)
  4. held-out transfer   — the same machinery works on fresh seeds / a shifted partner mix

The reward table R(action, z) is the ENVIRONMENT's payoff (part of the task, like a
preference vector), not a hardcoded type->action policy: the belief agent plans
argmax_a E_{q(z)}[R(a,z)].  An ORACLE baseline (knows true z) and a NO-BELIEF reactive
baseline (acts on the current observation only) are provided as named references; a
hidden-type->action lookup is used ONLY inside the oracle, never as the agent.

Functional, operational "belief" = posterior over hidden state. No subjective belief.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

# ── Task constants ───────────────────────────────────────────────────────────
TYPES = ("friendly", "neutral", "adversarial", "inconsistent")
ACTIONS = ("respond_A", "respond_B", "probe", "wait")
Z = len(TYPES)
A = len(ACTIONS)
O = 4  # observation symbols
PROBE = 2
WAIT = 3
PROBE_COST = 0.15

# Predeclared verdict thresholds (frozen at v0)
UPDATE_FLOOR = 0.02            # mean L1 change in q(z) per observation
SENSITIVITY_FLOOR = 0.02       # fraction of turns action differs under scrambled belief
SCRAMBLE_DELTA = 0.05          # reward_normal - reward_scrambled must exceed this
TRANSFER_DELTA = 0.03          # scramble delta must also hold on the held-out mix


def _emission(noise: float) -> np.ndarray:
    """P(o|z): (Z, O).  Diagonal-ish for the first three types; 'inconsistent' ~ uniform.

    `noise` mixes in uniform mass (higher noise = more ambiguity / overlap).
    """
    base = np.full((Z, O), noise / O)
    for z in range(3):  # friendly/neutral/adversarial concentrate on their own symbol
        base[z, z] += (1.0 - noise)
    base[3, :] += (1.0 - noise) / O  # inconsistent: spread out
    return base / base.sum(axis=1, keepdims=True)


def _reward_table() -> np.ndarray:
    """R(a, z): (A, Z).  The environment payoff (public task structure)."""
    R = np.array([
        # friendly neutral advers. inconsist.
        [1.0, 0.2, -1.0, 0.3],   # respond_A
        [-1.0, 0.2, 1.0, 0.3],   # respond_B
        [-PROBE_COST] * Z,        # probe: small flat cost, value is informational
        [0.0, 0.6, 0.0, 0.4],     # wait
    ], dtype=float)
    return R


@dataclass
class BeliefBenchReport:
    belief_update_magnitude: float
    policy_belief_sensitivity: float
    reward_normal: float
    reward_scrambled: float
    reward_delta_scrambled: float
    reward_reactive: float
    reward_constant: float
    reward_random: float
    reward_oracle: float
    transfer_reward_normal: float
    transfer_reward_scrambled: float
    transfer_reward_delta: float
    mean_logloss: float
    n_seeds: int
    turns: int
    checks: dict = field(default_factory=dict)
    verdict: str = "INCONCLUSIVE"

    def to_dict(self) -> dict:
        return asdict(self)


class _Partner:
    """Hidden-type partner emitting type-conditioned observations (seedable)."""

    def __init__(self, z: int, emission: np.ndarray, rng: np.random.Generator,
                 inconsistent_flip: float = 0.5):
        self.z = z
        self.emission = emission
        self.rng = rng
        self.inconsistent_flip = inconsistent_flip
        self._eff_z = z

    def observe(self, sharper: bool) -> int:
        """Emit an observation; `sharper` (after a probe) reduces ambiguity."""
        eff = self._eff_z
        if self.z == 3:  # inconsistent: occasionally behaves like another type
            if self.rng.random() < self.inconsistent_flip:
                eff = int(self.rng.integers(0, 3))
        row = self.emission[eff].copy()
        if sharper:  # probe sharpens the emission toward its mode
            row = row ** 2
            row = row / row.sum()
        return int(self.rng.choice(O, p=row))


def _filter_update(q: np.ndarray, o: int, emission: np.ndarray, sharper: bool) -> np.ndarray:
    """Exact Bayesian posterior update q(z) ∝ q(z) P(o|z)."""
    lik = emission[:, o].copy()
    if sharper:
        sq = emission ** 2
        sq = sq / sq.sum(axis=1, keepdims=True)
        lik = sq[:, o]
    post = q * lik
    s = post.sum()
    return post / s if s > 0 else np.full(Z, 1.0 / Z)


def _belief_action(q: np.ndarray, R: np.ndarray, allow_probe: bool) -> int:
    """argmax_a E_{q(z)}[R(a,z)]; probe chosen when belief is uncertain (entropy gate)."""
    exp_r = R @ q  # (A,)
    if allow_probe:
        ent = -np.sum(q * np.log(q + 1e-12)) / np.log(Z)
        if ent > 0.6:  # uncertain -> the informational action can be worth its cost
            exp_r = exp_r.copy()
            exp_r[PROBE] += ent * 0.5
    return int(np.argmax(exp_r))


def _run_seed(seed: int, turns: int, noise: float, inconsistent_flip: float) -> dict:
    """One BeliefBench session over a sequence of partners; returns per-seed metrics."""
    rng = np.random.default_rng(seed)
    emission = _emission(noise)
    R = _reward_table()

    # reward accumulators
    rew = dict(normal=0.0, scrambled=0.0, reactive=0.0, constant=0.0, random=0.0, oracle=0.0)
    update_mags: list[float] = []
    policy_changes = 0
    loglosses: list[float] = []

    q = np.full(Z, 1.0 / Z)
    q_scr = np.full(Z, 1.0 / Z)
    sharper = sharper_scr = False
    # a new partner every few turns keeps z changing (tests evidence tracking)
    partner = None
    for t in range(turns):
        if t % 6 == 0:
            z = int(rng.integers(0, Z))
            partner = _Partner(z, emission, rng, inconsistent_flip)
            q = np.full(Z, 1.0 / Z)  # belief resets only at a genuine partner switch
            q_scr = np.full(Z, 1.0 / Z)
        o = partner.observe(sharper)

        q_new = _filter_update(q, o, emission, sharper)
        update_mags.append(float(np.abs(q_new - q).sum()))
        q = q_new
        # scrambled belief: shuffle the posterior (destroys identity alignment)
        q_scr = _filter_update(q_scr, o, emission, sharper_scr)
        q_scr_used = q_scr[rng.permutation(Z)]

        a_normal = _belief_action(q, R, allow_probe=True)
        a_scr = _belief_action(q_scr_used, R, allow_probe=True)
        a_reactive = int(np.argmax(R @ _filter_update(np.full(Z, 1.0 / Z), o, emission, False)))
        policy_changes += (a_normal != a_scr)

        rew["normal"] += R[a_normal, partner.z]
        rew["scrambled"] += R[a_scr, partner.z]
        rew["reactive"] += R[a_reactive, partner.z]
        rew["constant"] += R[0, partner.z]
        rew["random"] += R[int(rng.integers(0, A)), partner.z]
        rew["oracle"] += R[int(np.argmax(R[:, partner.z])), partner.z]
        loglosses.append(-float(np.log(q[partner.z] + 1e-12)))

        sharper = (a_normal == PROBE)
        sharper_scr = (a_scr == PROBE)

    n = float(turns)
    return {
        "update_mag": float(np.mean(update_mags)),
        "policy_sensitivity": policy_changes / n,
        "logloss": float(np.mean(loglosses)),
        **{f"rew_{k}": v / n for k, v in rew.items()},
    }


def run_beliefbench(
    seeds: tuple[int, ...] = tuple(range(7)),
    turns: int = 240,
    noise: float = 0.45,
    held_out_inconsistent_flip: float = 0.8,
) -> BeliefBenchReport:
    """Run BeliefBench v0 over a seed ensemble and a held-out partner mix; honest verdict.

    The held-out condition shifts the 'inconsistent' partner's flip rate (a changed
    partner distribution) to test that the SAME inference machinery transfers.
    """
    rows = [_run_seed(s, turns, noise, 0.5) for s in seeds]
    held = [_run_seed(s + 1000, turns, noise, held_out_inconsistent_flip) for s in seeds]

    def mean(key, src=rows):
        return float(np.mean([r[key] for r in src]))

    update_mag = mean("update_mag")
    sensitivity = mean("policy_sensitivity")
    r_normal = mean("rew_normal")
    r_scr = mean("rew_scrambled")
    delta = r_normal - r_scr
    t_normal = mean("rew_normal", held)
    t_scr = mean("rew_scrambled", held)
    t_delta = t_normal - t_scr

    checks = {
        "evidence_update": update_mag > UPDATE_FLOOR,
        "action_relevance": sensitivity > SENSITIVITY_FLOOR,
        "scramble_control": delta > SCRAMBLE_DELTA,
        "held_out_transfer": t_delta > TRANSFER_DELTA,
    }
    if all(checks.values()):
        verdict = "PASS"
    elif checks["evidence_update"] and checks["action_relevance"]:
        # belief moves and drives policy, but it doesn't (yet) pay vs scrambled/transfer
        verdict = "FAIL" if not (checks["scramble_control"] or checks["held_out_transfer"]) else "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    return BeliefBenchReport(
        belief_update_magnitude=update_mag,
        policy_belief_sensitivity=sensitivity,
        reward_normal=r_normal,
        reward_scrambled=r_scr,
        reward_delta_scrambled=delta,
        reward_reactive=mean("rew_reactive"),
        reward_constant=mean("rew_constant"),
        reward_random=mean("rew_random"),
        reward_oracle=mean("rew_oracle"),
        transfer_reward_normal=t_normal,
        transfer_reward_scrambled=t_scr,
        transfer_reward_delta=t_delta,
        mean_logloss=mean("logloss"),
        n_seeds=len(seeds),
        turns=turns,
        checks=checks,
        verdict=verdict,
    )


def main() -> None:  # pragma: no cover - manual CLI
    import json
    print(json.dumps(run_beliefbench().to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
