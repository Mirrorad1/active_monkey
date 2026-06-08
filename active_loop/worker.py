"""Worker protocol + deterministic MockWorker (M1, no API spend)."""
from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from active_loop.signals import WorkerSignal


class Worker(Protocol):
    def do_step(self, step: int, hint: Optional[str]) -> WorkerSignal:
        ...


class MockWorker:
    """Deterministic stand-in for an LLM worker.

    Models an agent that is sometimes confidently wrong: ~30% of the time it
    succeeds with low confidence, occasionally it fails with high confidence.
    A 'correct' hint (from ASK) raises success probability.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def do_step(self, step: int, hint: Optional[str]) -> WorkerSignal:
        rng = np.random.default_rng(self.seed * 1000 + step)
        base_success = 0.55
        if hint == "correct":
            base_success = 0.85
        elif hint == "misleading":
            base_success = 0.25
        succeeded = bool(rng.random() < base_success)
        if succeeded:
            confidence = float(rng.uniform(0.4, 1.0))
            error_count = int(rng.integers(0, 2))
        else:
            confident_wrong = rng.random() < 0.4
            confidence = float(rng.uniform(0.6, 1.0)) if confident_wrong else float(rng.uniform(0.0, 0.4))
            error_count = int(rng.integers(1, 6))
        return WorkerSignal(
            succeeded=succeeded,
            confidence=confidence,
            error_count=error_count,
            human_feedback=None,
        )
