"""Proposers edit the mutable surface (active_loop/model_spec.py).

MockProposer: deterministic, valid, no LLM — drives all offline tests and CI.
ClaudeCliProposer: invokes the authenticated `claude` CLI for real autonomous runs.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Protocol

MUTABLE_FILE = "active_loop/model_spec.py"


class Proposer(Protocol):
    def propose(self, repo: Path | str) -> str:
        """Edit the mutable file in-place; return a one-line hypothesis."""
        ...


class MockProposer:
    """Deterministically nudges the first float literal in the mutable file."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def propose(self, repo: Path | str) -> str:
        path = Path(repo) / MUTABLE_FILE
        src = path.read_text()
        deltas = [0.1, -0.1, 0.25, -0.25, 0.5]
        delta = deltas[self.seed % len(deltas)]

        m = re.search(r"(\d+\.\d+)", src)
        if m is None:
            raise ValueError("no float literal found in mutable file to nudge")
        old = float(m.group(1))
        new = round(old + delta, 3)
        new_src = src[: m.start()] + repr(new) + src[m.end():]
        path.write_text(new_src)
        return f"nudge first float {old} -> {new} (seed {self.seed})"


_CODE_FENCE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


class ClaudeCliProposer:
    """Invokes `claude -p` to propose a new version of the mutable file."""

    def __init__(self, timeout_s: int = 180):
        self.timeout_s = timeout_s

    def _read(self, repo: Path, rel: str) -> str:
        p = Path(repo) / rel
        return p.read_text() if p.exists() else ""

    def propose(self, repo: Path | str) -> str:
        repo = Path(repo)
        mission = self._read(repo, "MISSION.md")
        policy = self._read(repo, "policy.md")
        index = self._read(repo, "world_model/INDEX.md")
        current = self._read(repo, MUTABLE_FILE)
        prompt = (
            f"{mission}\n\n## Rules of engagement\n{policy}\n\n"
            f"## What you know so far (world model)\n{index}\n\n"
            f"## Current {MUTABLE_FILE}\n```python\n{current}\n```\n\n"
            "Propose ONE small change to lower the controller's free energy while keeping "
            "guardrails. Output ONLY the complete new contents of the file in a single "
            "```python code fence — no prose, no explanation."
        )
        proc = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "text"],
            cwd=str(repo), capture_output=True, text=True, timeout=self.timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed: {proc.stderr[:500]}")
        m = _CODE_FENCE.search(proc.stdout)
        if m is None:
            raise RuntimeError("proposer returned no python code fence")
        new_src = m.group(1)
        compile(new_src, MUTABLE_FILE, "exec")
        (repo / MUTABLE_FILE).write_text(new_src)
        return "claude proposal applied"


LANG_MUTABLE_FILE = "active_loop/lang_model_spec.py"


class LangMockProposer:
    """Deterministically bumps K (latent state count) in the language spec.

    A valid, score-moving edit for offline loop tests — no LLM.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def propose(self, repo: Path | str) -> str:
        import re
        path = Path(repo) / LANG_MUTABLE_FILE
        src = path.read_text()
        m = re.search(r"^K\s*=\s*(\d+)", src, re.MULTILINE)
        if m is None:
            raise ValueError("no K assignment found in language spec")
        old = int(m.group(1))
        deltas = [2, 4, -2, 6, 8]
        new = max(2, old + deltas[self.seed % len(deltas)])
        new_src = src[: m.start()] + f"K = {new}" + src[m.end():]
        path.write_text(new_src)
        return f"set K {old} -> {new} (seed {self.seed})"
