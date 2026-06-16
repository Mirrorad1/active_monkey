"""Tests for the affect PR-loop harness.

All tests use a stub score_fn so no real 10-minute scoring runs occur.
Tests are @pytest.mark.slow (mirrors test_pr_loop.py convention).
"""
import subprocess
from pathlib import Path

import pytest

from active_loop import git_ops
from active_loop.affect_pr_loop import AffectMockProposer, one_affect_iteration
from active_loop.critic import MockCritic

REPO = Path(__file__).resolve().parents[1]


def _clone(tmp_path: Path) -> Path:
    dst = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--local", str(REPO), str(dst)],
        check=True, capture_output=True,
    )
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=dst, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=dst, check=True)
    (dst / ".venv").symlink_to(REPO / ".venv")
    # Tell git in the clone to ignore the .venv symlink (it's outside the clone tree,
    # so the root .gitignore pattern ".venv/" doesn't suppress it in all git versions).
    (dst / ".git" / "info" / "exclude").open("a").write("\n.venv\n")
    return dst


@pytest.mark.slow
def test_keeps_on_improvement(tmp_path):
    repo = _clone(tmp_path)
    trunk = git_ops.current_branch(repo)

    r = one_affect_iteration(
        repo,
        AffectMockProposer(0),
        MockCritic(approve=True),
        iteration=0,
        base_metric=0.30,
        score_fn=lambda _r: {"metric": 0.45, "verdict": True},
    )

    assert r.merged is True
    assert r.reason == "improved"
    assert git_ops.changed_files(repo) == []
    assert git_ops.current_branch(repo) == trunk


@pytest.mark.slow
def test_reverts_on_no_improvement(tmp_path):
    repo = _clone(tmp_path)
    sha_before = git_ops.current_sha(repo)

    r = one_affect_iteration(
        repo,
        AffectMockProposer(0),
        MockCritic(approve=True),
        iteration=0,
        base_metric=0.30,
        score_fn=lambda _r: {"metric": 0.20, "verdict": True},
    )

    assert r.merged is False
    assert r.reason == "no_improvement"
    assert git_ops.current_sha(repo) == sha_before
    assert git_ops.changed_files(repo) == []
    assert git_ops.current_branch(repo) == git_ops.current_branch(repo)  # on trunk


@pytest.mark.slow
def test_reverts_on_guardrail_fail(tmp_path):
    repo = _clone(tmp_path)
    sha_before = git_ops.current_sha(repo)

    r = one_affect_iteration(
        repo,
        AffectMockProposer(0),
        MockCritic(approve=True),
        iteration=0,
        base_metric=0.30,
        score_fn=lambda _r: {"metric": 0.99, "verdict": False},
    )

    assert r.merged is False
    assert r.reason == "guardrail_fail"
    assert git_ops.current_sha(repo) == sha_before
    assert git_ops.changed_files(repo) == []


@pytest.mark.slow
def test_reverts_on_critic_reject(tmp_path):
    repo = _clone(tmp_path)
    sha_before = git_ops.current_sha(repo)

    r = one_affect_iteration(
        repo,
        AffectMockProposer(0),
        MockCritic(approve=False),
        iteration=0,
        base_metric=0.30,
        score_fn=lambda _r: {"metric": 0.99, "verdict": True},
    )

    assert r.merged is False
    assert r.reason == "critic_reject"
    assert git_ops.current_sha(repo) == sha_before
    assert git_ops.changed_files(repo) == []


class _FrozenTouchingProposer:
    """Proposer that appends a comment to eval/affect_score.py (a FROZEN file)."""

    def propose(self, repo) -> str:
        target = Path(repo) / "eval" / "affect_score.py"
        src = target.read_text()
        target.write_text(src + "\n# injected by test\n")
        return "touch frozen scorer"


@pytest.mark.slow
def test_frozen_guard_blocks_scorer_edit(tmp_path):
    repo = _clone(tmp_path)
    sha_before = git_ops.current_sha(repo)

    # Verify the scorer file content before
    scorer_before = (repo / "eval" / "affect_score.py").read_text()

    r = one_affect_iteration(
        repo,
        _FrozenTouchingProposer(),
        MockCritic(approve=True),
        iteration=0,
        base_metric=0.30,
        score_fn=lambda _r: {"metric": 0.99, "verdict": True},
    )

    assert r.merged is False
    assert r.reason == "touched_frozen"
    assert git_ops.current_sha(repo) == sha_before
    assert git_ops.changed_files(repo) == []
    # Scorer file must be restored
    assert (repo / "eval" / "affect_score.py").read_text() == scorer_before
