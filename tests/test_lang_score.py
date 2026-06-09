import json, subprocess
import numpy as np
import pytest
from eval.lang_score import score_language, LangReport
from active_loop.alphabet import V


@pytest.mark.slow
def test_score_language_beats_baseline_and_reports_guardrails():
    r = score_language(epochs=6)
    assert isinstance(r, LangReport)
    assert np.isfinite(r.bits_per_char)
    assert r.baseline_bits == np.log(V) / np.log(2)
    assert r.guardrails["finite"] is True
    assert r.guardrails["beats_baseline"] is True
    assert r.bits_per_char < r.baseline_bits


@pytest.mark.slow
def test_score_json_entrypoint():
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "python", "-m", "eval.lang_score_json"],
        cwd=".", capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert "bits_per_char" in data and "verdict" in data
