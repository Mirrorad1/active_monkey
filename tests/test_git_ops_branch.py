import subprocess
from pathlib import Path
from active_loop import git_ops


def _init(tmp_path):
    subprocess.run(["git", "init", "-b", "master"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_branch_create_checkout_current(tmp_path):
    repo = _init(tmp_path)
    assert git_ops.current_branch(repo) == "master"
    git_ops.create_branch(repo, "proposal/iter-0")
    assert git_ops.current_branch(repo) == "proposal/iter-0"
    git_ops.checkout(repo, "master")
    assert git_ops.current_branch(repo) == "master"


def test_merge_no_ff_brings_branch_commit(tmp_path):
    repo = _init(tmp_path)
    git_ops.create_branch(repo, "proposal/iter-0")
    (repo / "a.txt").write_text("two\n")
    git_ops.commit_all(repo, "change on branch")
    git_ops.checkout(repo, "master")
    git_ops.merge_no_ff(repo, "proposal/iter-0", "merge proposal 0")
    assert (repo / "a.txt").read_text() == "two\n"


def test_delete_branch_discards_it(tmp_path):
    repo = _init(tmp_path)
    git_ops.create_branch(repo, "proposal/iter-0")
    (repo / "a.txt").write_text("two\n")
    git_ops.commit_all(repo, "doomed")
    git_ops.checkout(repo, "master")
    git_ops.delete_branch(repo, "proposal/iter-0")
    assert (repo / "a.txt").read_text() == "one\n"
