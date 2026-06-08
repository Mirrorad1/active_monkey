"""FROZEN: simulated human answering ASK actions.

Part of the trust boundary; the M2 outer loop must never edit it. Lets the loop
run headless during training/eval by standing in for the real human.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from active_loop.task_env import TaskEnv


class Oracle:
    def __init__(self, seed: int = 0, noise: float = 0.2):
        self.seed = seed
        self.noise = noise

    def answer(self, env: TaskEnv, step: int) -> tuple[Optional[str], str]:
        """Return (hint, feedback). hint in {correct, misleading, None}; feedback in {right, wrong}."""
        rng = np.random.default_rng(self.seed * 1000 + step)
        noisy = rng.random() < self.noise
        needs = env.needs_help(step)
        if needs:
            hint = "misleading" if noisy else "correct"
            feedback = "wrong"
        else:
            hint = None
            feedback = "wrong" if noisy else "right"
        return hint, feedback
