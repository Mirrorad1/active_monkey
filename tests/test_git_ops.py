import subprocess
from pathlib import Path

from active_loop import git_ops


def _init_repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.t"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "a.txt").write_text("one\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def test_current_sha_is_stable(tmp_path):
    repo = _init_repo(tmp_path)
    sha1 = git_ops.current_sha(repo)
    assert isinstance(sha1, str) and len(sha1) >= 7
    assert git_ops.current_sha(repo) == sha1


def test_changed_files_lists_modifications(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("two\n")
    (repo / "b.txt").write_text("new\n")
    changed = set(git_ops.changed_files(repo))
    assert "a.txt" in changed and "b.txt" in changed


def test_commit_all_creates_new_sha(tmp_path):
    repo = _init_repo(tmp_path)
    before = git_ops.current_sha(repo)
    (repo / "a.txt").write_text("two\n")
    after = git_ops.commit_all(repo, "change a")
    assert after != before
    assert git_ops.changed_files(repo) == []


def test_reset_hard_discards_changes(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "a.txt").write_text("two\n")
    git_ops.reset_hard(repo)
    assert (repo / "a.txt").read_text() == "one\n"
    assert git_ops.changed_files(repo) == []


def test_reset_hard_scoped_preserves_root_untracked(tmp_path):
    repo = _init_repo(tmp_path)
    # untracked file at root survives; untracked file under active_loop/ is cleaned
    (repo / "keep_me.txt").write_text("root\n")
    (repo / "active_loop").mkdir()
    (repo / "active_loop" / "junk.txt").write_text("scratch\n")
    git_ops.reset_hard(repo)
    assert (repo / "keep_me.txt").exists()
    assert not (repo / "active_loop" / "junk.txt").exists()
