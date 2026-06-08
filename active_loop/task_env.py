"""FROZEN: synthetic task environment with a hidden per-step 'needs_help' flag.

This file is part of the trust boundary. The M2 outer loop must never edit it.
"""
from __future__ import annotations

import numpy as np


class TaskEnv:
    def __init__(self, seed: int = 0, num_steps: int = 8):
        self.seed = seed
        self.num_steps = num_steps
        rng = np.random.default_rng(seed)
        # ~40% of steps secretly require a correcting hint to succeed reliably
        self._needs_help = (rng.random(num_steps) < 0.4).tolist()

    def needs_help(self, step: int) -> bool:
        return bool(self._needs_help[step])

    def is_success(self, step: int, succeeded: bool) -> bool:
        """Ground-truth success for scoring (here, simply the worker's success)."""
        return bool(succeeded)
