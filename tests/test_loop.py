import json
import shutil
import subprocess
from pathlib import Path

import pytest

from active_loop import git_ops
from active_loop.proposer import MockProposer


REPO = Path(__file__).resolve().parents[1]


def _clone_repo(tmp_path: Path) -> Path:
    dst = tmp_path / "clone"
    subprocess.run(["git", "clone", "--local", str(REPO), str(dst)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=dst, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=dst, check=True)
    (dst / ".venv").symlink_to(REPO / ".venv")
    # The .venv symlink is a test fixture, not loop output. Git's `.venv/`
    # ignore pattern only matches directories, not this symlink, so exclude it
    # explicitly to keep it out of changed_files / the scoped revert clean.
    (dst / ".git" / "info" / "exclude").write_text(".venv\n")
    return dst


def test_one_iteration_keeps_or_reverts_and_leaves_clean_tree(tmp_path):
    from active_loop.loop import one_iteration
    repo = _clone_repo(tmp_path)
    base = git_ops.current_sha(repo)
    result = one_iteration(repo, MockProposer(seed=0), iteration=0)
    assert git_ops.changed_files(repo) == []
    head = git_ops.current_sha(repo)
    if result.kept:
        assert head != base
    else:
        assert head == base


@pytest.mark.slow
def test_run_loop_writes_artifacts_and_grows_world_model(tmp_path):
    from active_loop.loop import run_loop
    repo = _clone_repo(tmp_path)
    run_loop(repo, proposer=MockProposer(seed=0), iterations=2)
    assert (repo / "reports" / "index.html").exists()
    assert (repo / "world_model" / "INDEX.md").exists()
    journal = repo / "world_model" / "evidence" / "journal.jsonl"
    assert journal.exists()
    assert len(journal.read_text().splitlines()) == 2
    tracked = subprocess.run(["git", "ls-files", "world_model"], cwd=repo,
                             capture_output=True, text=True).stdout
    assert "world_model/INDEX.md" in tracked  # world model is versioned in git
