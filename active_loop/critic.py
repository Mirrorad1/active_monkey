"""Critic: reviews a proposed diff and returns an approve/reject verdict.

MockCritic drives offline tests; ClaudeCliCritic uses the authenticated claude CLI.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Verdict:
    approved: bool
    reason: str


class Critic(Protocol):
    def review(self, diff: str, repo: Path | str) -> Verdict:
        ...


class MockCritic:
    def __init__(self, approve: bool = True):
        self.approve = approve

    def review(self, diff: str, repo: Path | str) -> Verdict:
        return Verdict(self.approve, "mock verdict")


class ClaudeCliCritic:
    """Asks claude -p to review the diff for soundness and metric-gaming."""

    def __init__(self, timeout_s: int = 180):
        self.timeout_s = timeout_s

    def review(self, diff: str, repo: Path | str) -> Verdict:
        if not diff.strip():
            return Verdict(False, "empty diff")
        prompt = (
            "You are reviewing a proposed change to a pymdp active-inference character "
            "language model's generative-model spec (active_loop/lang_model_spec.py). The "
            "objective is to LOWER held-out bits/char (free energy) WITHOUT gaming the metric "
            "(no collapsing the model, no touching the evaluator/corpus). Here is the diff:\n\n"
            f"{diff}\n\n"
            "Reply with exactly one line: 'APPROVE: <reason>' if it is a sound, honest attempt, "
            "or 'REJECT: <reason>' if it is unsound or gaming."
        )
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            cwd=str(repo), capture_output=True, text=True, timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            return Verdict(False, f"critic CLI failed: {proc.stderr[:200]}")
        out = proc.stdout.strip()
        approved = bool(re.match(r"\s*APPROVE", out, re.IGNORECASE))
        return Verdict(approved, out[:300])
