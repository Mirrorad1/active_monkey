"""FROZEN: run the suite and compute the free-energy metric + guardrail verdict."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from active_loop.controller import Controller
from active_loop.worker import MockWorker
from active_loop.task_env import TaskEnv
from active_loop.oracle import Oracle
from active_loop.hybrid import run_episode
from eval.suite import SUITE

SUCCESS_FLOOR = 0.5
ASK_RATE_LO = 0.1
ASK_RATE_HI = 0.6


@dataclass(frozen=True)
class ScoreReport:
    metric: float
    success_rate: float
    ask_rate: float
    guardrails: dict
    verdict: bool


def score_suite() -> ScoreReport:
    controller = Controller(seed=0)
    step_efe: list[float] = []
    successes: list[bool] = []
    asks: list[bool] = []

    for cfg in SUITE:
        env = TaskEnv(seed=cfg.env_seed, num_steps=cfg.num_steps)
        worker = MockWorker(seed=cfg.worker_seed)
        oracle = Oracle(seed=cfg.oracle_seed)
        traj = run_episode(controller, worker, env, oracle)
        controller.learn(traj)
        for ne in traj.neg_efe:
            step_efe.append(-float(np.max(ne)))
        successes.extend(traj.successes)
        asks.extend(a.name == "ASK" for a in traj.actions)

    metric = float(np.mean(step_efe))
    success_rate = float(np.mean(successes))
    ask_rate = float(np.mean(asks))
    guardrails = {
        "success_floor": success_rate >= SUCCESS_FLOOR,
        "ask_rate_band": ASK_RATE_LO <= ask_rate <= ASK_RATE_HI,
    }
    return ScoreReport(
        metric=metric,
        success_rate=success_rate,
        ask_rate=ask_rate,
        guardrails=guardrails,
        verdict=all(guardrails.values()),
    )
