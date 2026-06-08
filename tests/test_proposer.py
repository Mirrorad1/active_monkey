from pathlib import Path

from active_loop.proposer import MockProposer


def _seed_model_spec(tmp_path: Path) -> Path:
    pkg = tmp_path / "active_loop"
    pkg.mkdir(parents=True)
    (pkg / "model_spec.py").write_text(
        "C_SUCCESS_PREF = 2.0\n\n\ndef build_controller_arrays():\n    return C_SUCCESS_PREF\n"
    )
    return tmp_path


def test_mock_proposer_edits_model_spec_and_returns_hypothesis(tmp_path):
    repo = _seed_model_spec(tmp_path)
    before = (repo / "active_loop" / "model_spec.py").read_text()
    hypothesis = MockProposer(seed=0).propose(repo)
    after = (repo / "active_loop" / "model_spec.py").read_text()
    assert after != before
    assert isinstance(hypothesis, str) and len(hypothesis) > 0


def test_mock_proposer_keeps_file_importable(tmp_path):
    repo = _seed_model_spec(tmp_path)
    MockProposer(seed=1).propose(repo)
    src = (repo / "active_loop" / "model_spec.py").read_text()
    compile(src, "model_spec.py", "exec")


def test_mock_proposer_is_deterministic(tmp_path):
    repo_a = _seed_model_spec(tmp_path / "a")
    repo_b = _seed_model_spec(tmp_path / "b")
    h1 = MockProposer(seed=5).propose(repo_a)
    h2 = MockProposer(seed=5).propose(repo_b)
    assert h1 == h2
    assert (repo_a / "active_loop" / "model_spec.py").read_text() == (repo_b / "active_loop" / "model_spec.py").read_text()
