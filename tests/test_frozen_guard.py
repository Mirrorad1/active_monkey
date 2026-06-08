from pathlib import Path

from active_loop.frozen_guard import load_frozen, is_frozen_touched


def _write_manifest(tmp_path: Path) -> Path:
    (tmp_path / "FROZEN").write_text("eval/\nactive_loop/task_env.py\nMISSION.md\n")
    return tmp_path


def test_load_frozen_reads_nonempty_lines(tmp_path):
    repo = _write_manifest(tmp_path)
    frozen = load_frozen(repo)
    assert "eval/" in frozen and "active_loop/task_env.py" in frozen


def test_touching_frozen_prefix_is_detected(tmp_path):
    repo = _write_manifest(tmp_path)
    assert is_frozen_touched(["eval/score.py"], repo) is True
    assert is_frozen_touched(["active_loop/task_env.py"], repo) is True


def test_touching_only_mutable_surface_is_allowed(tmp_path):
    repo = _write_manifest(tmp_path)
    assert is_frozen_touched(["active_loop/model_spec.py"], repo) is False
    assert is_frozen_touched([], repo) is False
