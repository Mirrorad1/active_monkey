"""Wire controller <-> worker <-> oracle into one episode; collect a Trajectory."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import jax.numpy as jnp

from active_loop.controller import Controller, Action
from active_loop.worker import Worker
from active_loop.task_env import TaskEnv
from active_loop.oracle import Oracle
from active_loop.signals import WorkerSignal, discretize


@dataclass
class Trajectory:
    obs_seq: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    action_seq: list = field(default_factory=list)
    neg_efe: list = field(default_factory=list)
    qs_seq: list = field(default_factory=list)
    successes: list = field(default_factory=list)

    def ask_rate(self) -> float:
        if not self.actions:
            return 0.0
        return sum(a == Action.ASK for a in self.actions) / len(self.actions)

    def success_rate(self) -> float:
        if not self.successes:
            return 0.0
        return sum(self.successes) / len(self.successes)


def run_episode(controller: Controller, worker: Worker, env: TaskEnv, oracle: Oracle) -> Trajectory:
    controller.reset()
    traj = Trajectory()
    pending_hint: Optional[str] = None

    for step in range(env.num_steps):
        sig = worker.do_step(step=step, hint=pending_hint)
        pending_hint = None

        obs = discretize(sig)
        out = controller.step(obs)
        traj.actions.append(out.action)
        traj.action_seq.append(out.raw_action)
        traj.neg_efe.append(np.asarray(out.neg_efe))
        traj.qs_seq.append(out.qs)

        if out.action == Action.ASK:
            hint, feedback = oracle.answer(env, step)
            pending_hint = hint
            obs = discretize(WorkerSignal(sig.succeeded, sig.confidence, sig.error_count, feedback))
        elif out.action == Action.SWITCH:
            pending_hint = None

        traj.obs_seq.append(obs)
        traj.successes.append(env.is_success(step, sig.succeeded))

    return traj
