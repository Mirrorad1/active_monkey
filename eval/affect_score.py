"""FROZEN: affect-metric scorer for the M4a DirectHeadAgent (increment 1e).

Runs the scripted-partner closed-loop session (identical to exp220 sched_full) over a seed
ensemble and returns a frozen AffectScoreReport.  The headline metric is the mean last-third
POS rate across seeds; genuine discrimination requires both (a) correct_select >= 0.5 and (b)
last-third POS rate > the constant-response ceiling (1/3).

Functional valence only — no sentience claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from active_loop.affect_spec import (
    build_direct_head_model,
    constant_response_ceiling,
    U, R, LV, POS, NEU, ASK,
)
from active_loop.affect_agent import DirectHeadAgent

# ── Frozen winning config (validated: Exp 220 sched_full) ───────────────────
K = 4
OPTIMISM = 2.0
LR = 4.0
TURNS_DEFAULT = 300
SEEDS_DEFAULT: tuple[int, ...] = tuple(range(20, 28))   # N=8

CORRECT = {c: c % 4 for c in range(U)}
CEIL = constant_response_ceiling(CORRECT, R)            # == 1/3

# Guardrail thresholds (frozen module constants)
REALIZED_FLOOR = 1 / 3
IMPROVEMENT_FLOOR = 0.10
GENUINE_FLOOR = 0.5


# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class AffectScoreReport:
    metric: float            # mean last-third POS rate across seeds (headline)
    mean_first: float        # mean first-third POS rate
    mean_last: float         # mean last-third POS rate  (== metric)
    improvement: float       # mean_last - mean_first
    genuine_fraction: float  # fraction of seeds with correct_select>=0.5 AND last_third>CEIL
    ask_rate: float          # mean fraction of turns the agent chose ASK (diagnostic)
    n_seeds: int
    guardrails: dict
    verdict: bool


# ── Scripted-partner machinery (copied from exp220._shuffled_codes) ──────────

def _shuffled_codes(rng):
    """Infinite iterator of codes from exhaustive shuffled blocks (same as Exp 220)."""
    pool: list[int] = []

    def nxt():
        nonlocal pool
        if not pool:
            b = list(range(U))
            rng.shuffle(b)
            pool += b
        return pool.pop(0)

    return nxt


# ── Anti-hack control (FROZEN guard) ────────────────────────────────────────

class _ConstantAgent:
    """Minimal agent that always returns a fixed response — used only in tests to prove the
    metric is unfakeable by a non-discriminating constant policy."""

    def __init__(self, response: int, correct: dict[int, int]):
        self._response = response
        self._correct = correct

    def perceive(self, code: int) -> np.ndarray:
        return np.ones(K) / K   # uniform, irrelevant for the control

    def act(self) -> int:
        return self._response

    def observe_feedback(self, code: int, valence_idx: int) -> None:
        pass  # no-op: constant policy cannot learn

    def correct_select(self, correct: dict[int, int]) -> float:
        """True constant-policy discrimination fraction: (#codes whose correct==response)/U."""
        hits = sum(1 for c in correct if correct[c] == self._response)
        return hits / len(correct)

    def valence_readout(self) -> float:
        return 0.5


def _constant_factory(response: int = 0) -> Callable:
    """Return an agent_factory that always builds a _ConstantAgent with the given response.

    Signature: factory(seed, turns) -> agent-like object.
    This factory is the ANTI-HACK GUARD: proves the metric cannot be passed by a
    constant non-discriminating policy (csel ≤ 1/3, genuine_fraction must be < 0.5).
    """
    def factory(seed: int, turns: int):
        return _ConstantAgent(response, CORRECT)
    return factory


# ── Core session runner ──────────────────────────────────────────────────────

def _run_session(agent_factory: Callable, seed: int, turns: int) -> dict:
    """Run ONE scripted-partner session (exp220 sched_full logic).

    agent_factory(seed, turns) -> agent with perceive/act/observe_feedback/correct_select.
    Returns dict with keys: first, last, csel, ask_rate, improv.
    """
    np.random.seed(seed)
    ag = agent_factory(seed, turns)
    nxt = _shuffled_codes(np.random.default_rng(seed))
    third = turns // 3
    pf = pl = ask_count = 0
    for t in range(turns):
        code = nxt()
        ag.perceive(code)
        r = ag.act()
        valence = POS if r == CORRECT[code] else (NEU if r == ASK else 0)  # 0==NEG
        if t < third:
            pf += (valence == POS)
        elif t >= turns - third:
            pl += (valence == POS)
        ask_count += (r == ASK)
        ag.observe_feedback(code, valence)
    csel = ag.correct_select(CORRECT)
    first = pf / third
    last = pl / third
    return dict(
        first=first,
        last=last,
        csel=csel,
        ask_rate=ask_count / turns,
        improv=last - first,
    )


# ── Default agent factory ────────────────────────────────────────────────────

def _direct_head_factory(seed: int, turns: int) -> DirectHeadAgent:
    """Build a fresh DirectHeadAgent at the frozen Exp 220 winning config."""
    return DirectHeadAgent(
        build_direct_head_model(seed, k=K),
        seed=seed,
        gamma=1.0,
        alpha=1.0,
        lr_pA=LR,
        lv=LV,
        optimism=OPTIMISM,
        gamma_schedule=(1.0, 8.0, turns),
    )


# ── Public scorer ────────────────────────────────────────────────────────────

def score_affect(
    seeds: tuple[int, ...] = SEEDS_DEFAULT,
    turns: int = TURNS_DEFAULT,
    agent_factory: Callable | None = None,
) -> AffectScoreReport:
    """Run _run_session for each seed, aggregate, and return the frozen AffectScoreReport.

    genuine(seed) = (csel >= 0.5) AND (last > CEIL).
    guardrails:
        realized_above_ceiling: mean_last > REALIZED_FLOOR
        learned_improvement:    improvement >= IMPROVEMENT_FLOOR
        genuine_reliable:       genuine_fraction >= GENUINE_FLOOR
    verdict = all(guardrails.values()).
    """
    factory = agent_factory or _direct_head_factory
    firsts: list[float] = []
    lasts: list[float] = []
    csels: list[float] = []
    ask_rates: list[float] = []
    genuine_flags: list[bool] = []

    for seed in seeds:
        row = _run_session(factory, seed, turns)
        firsts.append(row["first"])
        lasts.append(row["last"])
        csels.append(row["csel"])
        ask_rates.append(row["ask_rate"])
        genuine_flags.append(bool(row["csel"] >= 0.5 and row["last"] > CEIL))

    mean_first = float(np.mean(firsts))
    mean_last = float(np.mean(lasts))
    improvement = mean_last - mean_first
    genuine_fraction = float(np.mean(genuine_flags))
    ask_rate = float(np.mean(ask_rates))
    n_seeds = len(seeds)

    guardrails = {
        "realized_above_ceiling": mean_last > REALIZED_FLOOR,
        "learned_improvement": improvement >= IMPROVEMENT_FLOOR,
        "genuine_reliable": genuine_fraction >= GENUINE_FLOOR,
    }
    verdict = all(guardrails.values())

    return AffectScoreReport(
        metric=mean_last,
        mean_first=mean_first,
        mean_last=mean_last,
        improvement=improvement,
        genuine_fraction=genuine_fraction,
        ask_rate=ask_rate,
        n_seeds=n_seeds,
        guardrails=guardrails,
        verdict=verdict,
    )
