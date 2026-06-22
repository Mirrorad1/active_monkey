import json
import subprocess
import sys

from active_loop.pcc import Forecast, Trial, choose_patch, summarize_patch


def test_summarize_patch_rewards_calibrated_transfer_not_dev_luck():
    baseline = [
        Trial("a", ("multi_step",), False),
        Trial("b", ("single_step",), True),
        Trial("c", ("unit",), False),
    ]
    patched = [
        Trial("a", ("multi_step",), True),
        Trial("b", ("single_step",), True),
        Trial("c", ("unit",), False),
    ]
    forecast = Forecast(
        {
            "multi_step": {"improve": 0.8, "same": 0.15, "worse": 0.05},
            "single_step": {"improve": 0.1, "same": 0.8, "worse": 0.1},
            "unit": {"improve": 0.2, "same": 0.7, "worse": 0.1},
        }
    )

    summary = summarize_patch("p1", forecast, baseline, patched)

    assert summary.delta_accuracy == 1 / 3
    assert summary.mean_forecast_nll < 0.6
    assert summary.pcc_score > summary.delta_accuracy - 0.1


def test_choose_patch_can_penalize_wrong_forecast():
    baseline = [
        Trial("a", ("multi_step",), False),
        Trial("b", ("single_step",), False),
    ]
    patched_good = [
        Trial("a", ("multi_step",), True),
        Trial("b", ("single_step",), False),
    ]
    patched_lucky = [
        Trial("a", ("multi_step",), True),
        Trial("b", ("single_step",), True),
    ]
    calibrated = Forecast(
        {
            "multi_step": {"improve": 0.9, "same": 0.1},
            "single_step": {"same": 0.9, "improve": 0.1},
        }
    )
    overconfident_wrong = Forecast(
        {
            "multi_step": {"same": 0.9, "improve": 0.1},
            "single_step": {"same": 0.9, "improve": 0.1},
        }
    )

    summaries = [
        summarize_patch("calibrated", calibrated, baseline, patched_good, alpha=0.2),
        summarize_patch("lucky", overconfident_wrong, baseline, patched_lucky, alpha=0.2),
    ]

    assert choose_patch(summaries).patch_id == "calibrated"


def test_pcc_outer_loop_smoke_runs():
    out = subprocess.check_output(
        [sys.executable, "experiments/pcc_outer_loop.py", "--backend", "smoke", "--iterations", "1"],
        text=True,
    )
    payload = json.loads(out.splitlines()[-1])
    assert payload["backend"] == "smoke"
    assert payload["chosen"]["method"] in {"pcc", "score_only"}


def test_pcc_outer_loop_writes_output_file(tmp_path):
    output = tmp_path / "pcc.json"
    subprocess.check_call(
        [
            sys.executable,
            "experiments/pcc_outer_loop.py",
            "--backend",
            "smoke",
            "--iterations",
            "1",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text())
    assert payload["backend"] == "smoke"
    assert payload["chosen"]["patch_id"]


def test_runpod_pcc_script_is_bash_parseable_and_targets_hf_run():
    script = "runpod/setup_pcc_outer_loop.sh"
    subprocess.check_call(["bash", "-n", script])
    text = open(script, encoding="utf-8").read()
    assert "Qwen/Qwen2.5-0.5B-Instruct" in text
    assert "experiments/pcc_outer_loop.py" in text
    assert "--backend hf" in text
    assert "--output" in text
