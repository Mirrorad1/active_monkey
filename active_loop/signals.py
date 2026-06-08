"""Worker signals and their discretization into pymdp observation indices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MODALITY_NAMES = ["step_succeeded", "confidence", "error_count", "human_feedback"]
NUM_OBS = [2, 3, 3, 3]


@dataclass(frozen=True)
class WorkerSignal:
    succeeded: bool
    confidence: float          # in [0, 1]
    error_count: int           # >= 0
    human_feedback: Optional[str] = None  # None | "right" | "wrong"


def _confidence_bin(c: float) -> int:
    if c < 0.34:
        return 0
    if c < 0.67:
        return 1
    return 2


def _error_bin(n: int) -> int:
    if n <= 0:
        return 0
    if n <= 2:
        return 1
    return 2


def _feedback_code(fb: Optional[str]) -> int:
    return {None: 0, "wrong": 1, "right": 2}[fb]


def discretize(sig: WorkerSignal) -> list[int]:
    """Map a WorkerSignal to one observation index per modality (order = MODALITY_NAMES)."""
    return [
        1 if sig.succeeded else 0,
        _confidence_bin(sig.confidence),
        _error_bin(sig.error_count),
        _feedback_code(sig.human_feedback),
    ]
