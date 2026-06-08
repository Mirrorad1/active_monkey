import json, subprocess
from pathlib import Path
import pytest
from active_loop import git_ops
from active_loop.proposer import LangMockProposer
from active_loop.critic import MockCritic

REPO = Path(__file__).resolve().parents[1]


def _clone(tmp_path):
    dst = tmp_path / "clone"
    subprocess.run(["git", "clone", "--local", str(REPO), str(dst)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=dst, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=dst, check=True)
    (dst / ".venv").symlink_to(REPO / ".venv")
    return dst


def test_one_pr_iteration_merges_or_discards_clean(tmp_path):
    from active_loop.pr_loop import one_pr_iteration
    repo = _clone(tmp_path)
    r = one_pr_iteration(repo, LangMockProposer(seed=0), MockCritic(approve=True), iteration=0)
    assert git_ops.current_branch(repo) == "master"
    assert git_ops.changed_files(repo) == []
    assert r.merged in (True, False)


def test_critic_rejection_discards_branch(tmp_path):
    from active_loop.pr_loop import one_pr_iteration
    repo = _clone(tmp_path)
    base = git_ops.current_sha(repo)
    r = one_pr_iteration(repo, LangMockProposer(seed=0), MockCritic(approve=False), iteration=0)
    assert r.merged is False and r.reason == "critic_reject"
    assert git_ops.current_sha(repo) == base


@pytest.mark.slow
def test_run_pr_loop_grows_world_model(tmp_path):
    from active_loop.pr_loop import run_pr_loop
    repo = _clone(tmp_path)
    run_pr_loop(repo, LangMockProposer(seed=0), MockCritic(approve=True), iterations=1)
    assert (repo / "reports" / "index.html").exists()
    assert (repo / "world_model" / "evidence" / "journal.jsonl").exists()
