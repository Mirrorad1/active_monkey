from pathlib import Path
from active_loop.proposer import LangMockProposer


def _seed_spec(tmp_path: Path) -> Path:
    pkg = tmp_path / "active_loop"
    pkg.mkdir(parents=True)
    (pkg / "lang_model_spec.py").write_text("K = 12\nA_CONC = 0.1\n")
    return tmp_path


def test_lang_mock_proposer_changes_K_and_returns_hypothesis(tmp_path):
    repo = _seed_spec(tmp_path)
    hyp = LangMockProposer(seed=0).propose(repo)
    src = (repo / "active_loop" / "lang_model_spec.py").read_text()
    assert "K = 12" not in src
    assert "K = " in src
    assert isinstance(hyp, str) and "K" in hyp


def test_lang_mock_proposer_keeps_file_importable(tmp_path):
    repo = _seed_spec(tmp_path)
    LangMockProposer(seed=2).propose(repo)
    compile((repo / "active_loop" / "lang_model_spec.py").read_text(), "s", "exec")
