"""Guard: tools/autosync.sh must never push a non-main branch onto origin/main.

The Stop-hook autosync once ran an unconditional `git push origin HEAD:main`,
which swept feature-branch work straight onto origin/main and silently bypassed
the PR review path META.md mandates for non-experiment changes. This test runs
the real script inside a sandbox repo (fake bare origin) and fails on recurrence.
"""
import pathlib
import shutil
import subprocess

ROOT = pathlib.Path(__file__).parent.parent
SCRIPT = ROOT / "tools" / "autosync.sh"


def _git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_sandbox(tmp_path):
    """Bare origin + working clone with one commit on main, script installed."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "--bare", "--initial-branch=main", ".")

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init", "--initial-branch=main", ".")
    _git(work, "remote", "add", "origin", str(origin))

    (work / "README.md").write_text("seed\n", encoding="utf-8")
    tools = work / "tools"
    tools.mkdir()
    shutil.copy(SCRIPT, tools / "autosync.sh")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "seed")
    _git(work, "push", "-u", "origin", "main")
    return origin, work


def _run_autosync(work):
    subprocess.run(
        ["bash", "tools/autosync.sh"], cwd=work, check=True, capture_output=True
    )


def test_feature_branch_never_pushes_main(tmp_path):
    origin, work = _make_sandbox(tmp_path)
    main_before = _git(origin, "rev-parse", "main")

    _git(work, "checkout", "-b", "feature/x")
    (work / "new.txt").write_text("dirty\n", encoding="utf-8")
    _run_autosync(work)

    # origin/main untouched; the dirty tree was committed and pushed to the
    # branch's own name instead.
    assert _git(origin, "rev-parse", "main") == main_before
    branch_head = _git(origin, "rev-parse", "refs/heads/feature/x")
    assert branch_head == _git(work, "rev-parse", "HEAD")
    assert _git(work, "status", "--porcelain") == ""


def test_main_still_pushes_main(tmp_path):
    origin, work = _make_sandbox(tmp_path)

    (work / "new.txt").write_text("dirty on main\n", encoding="utf-8")
    _run_autosync(work)

    assert _git(origin, "rev-parse", "main") == _git(work, "rev-parse", "HEAD")
    assert _git(work, "status", "--porcelain") == ""
