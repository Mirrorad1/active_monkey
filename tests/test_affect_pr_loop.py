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


@pytest.mark.slow
def test_run_affect_pr_loop_completes_and_writes_isolated_journal(tmp_path):
    """run_affect_pr_loop must complete a full iteration and write its ISOLATED affect journal.

    Regression guard for the Exp 224 instrument bug: one_affect_iteration's branch churn
    (commit_all sweeps the untracked world_model_affect onto the proposal branch, then
    discard -> checkout(trunk) WIPES it because it is not tracked on trunk). A critic-reject
    iteration takes that discard path; without the re-mkdir fix the subsequent
    wm.append_evidence raises FileNotFoundError. With it, the loop completes and writes the
    journal under the isolated world_model_affect dir.
    """
    from active_loop.affect_pr_loop import run_affect_pr_loop
    repo = _clone(tmp_path)
    trunk = git_ops.current_branch(repo)
    # MockCritic(approve=False) -> proposal critic-rejected -> the discard-checkout path that
    # wiped world_model_affect. Stub score so no real 10-min scoring runs.
    run_affect_pr_loop(
        repo,
        AffectMockProposer(0),
        MockCritic(approve=False),
        iterations=1,
        score_fn=lambda _r: {"metric": 0.40, "verdict": True},
    )
    assert git_ops.current_branch(repo) == trunk
    journal = repo / "world_model_affect" / "evidence" / "journal.jsonl"
    assert journal.exists(), "isolated affect journal was not written"
    assert journal.read_text().strip(), "affect journal is empty"
