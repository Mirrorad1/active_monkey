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


def commit_paths(repo: Path | str, paths: list[str], message: str) -> str:
    """Stage the given existing paths and commit; tolerate 'nothing to commit'."""
    repo_p = Path(repo)
    existing = [p for p in paths if (repo_p / p).exists()]
    if not existing:
        return current_sha(repo)
    # -f so the loop's owned artifacts are staged even if a stale .gitignore
    # (e.g. in a freshly-cloned working tree) still lists them.
    _run(repo, "add", "-f", *existing)
    if not _run(repo, "diff", "--cached", "--name-only").strip():
        return current_sha(repo)
    _run(repo, "commit", "-m", message)
    return current_sha(repo)


def reset_hard(repo: Path | str) -> None:
    """Discard tracked changes and untracked files UNDER active_loop/ back to HEAD.

    The clean is scoped to active_loop/ so the loop's root-level artifacts
    (world_model/, reports/, REPORT.md) and entry points are never wiped on a revert."""
    _run(repo, "reset", "--hard", "HEAD")
    _run(repo, "clean", "-fd", "active_loop")


def current_branch(repo: Path | str) -> str:
    return _run(repo, "rev-parse", "--abbrev-ref", "HEAD")


def create_branch(repo: Path | str, name: str) -> None:
    """Create and check out a new branch from the current HEAD."""
    _run(repo, "checkout", "-b", name)


def checkout(repo: Path | str, name: str) -> None:
    _run(repo, "checkout", name)


def merge_no_ff(repo: Path | str, branch: str, message: str) -> str:
    _run(repo, "merge", "--no-ff", "-m", message, branch)
    return current_sha(repo)


def delete_branch(repo: Path | str, name: str) -> None:
    _run(repo, "branch", "-D", name)
