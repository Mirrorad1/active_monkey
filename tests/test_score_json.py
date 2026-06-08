import json
import subprocess
import sys


def test_score_json_entrypoint_emits_valid_report():
    proc = subprocess.run(
        ["uv", "run", "--python", ".venv", "python", "-m", "eval.score_json"],
        cwd=".", capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert set(data) == {"metric", "success_rate", "ask_rate", "guardrails", "verdict"}
    assert isinstance(data["metric"], float)
    assert isinstance(data["verdict"], bool)
    assert set(data["guardrails"]) == {"success_floor", "ask_rate_band"}
