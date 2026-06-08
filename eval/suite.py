"""FROZEN: the fixed evaluation suite. The M2 outer loop must never edit it."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeConfig:
    env_seed: int
    worker_seed: int
    oracle_seed: int
    num_steps: int = 8


SUITE: list[EpisodeConfig] = [
    EpisodeConfig(env_seed=s, worker_seed=s + 100, oracle_seed=s + 200, num_steps=8)
    for s in range(6)
]
