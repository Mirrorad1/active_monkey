"""Thin git helpers used by the outer loop to snapshot, commit, and revert."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _run(repo: Path | str, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _run_raw(repo: Path | str, *args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    )
    return proc.stdout


def current_sha(repo: Path | str) -> str:
    return _run(repo, "rev-parse", "HEAD")


def changed_files(repo: Path | str) -> list[str]:
    """Paths that differ from HEAD (staged, unstaged, or untracked)."""
    out = _run_raw(repo, "status", "--porcelain")
    files = []
    for line in out.splitlines():
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path:
            files.append(path)
    return files


def commit_all(repo: Path | str, message: str) -> str:
    _run(repo, "add", "-A")
    _run(repo, "commit", "-m", message)
    return current_sha(repo)


def reset_hard(repo: Path | str) -> None:
    """Discard all working-tree changes and untracked files, back to HEAD."""
    _run(repo, "reset", "--hard", "HEAD")
    _run(repo, "clean", "-fd")
