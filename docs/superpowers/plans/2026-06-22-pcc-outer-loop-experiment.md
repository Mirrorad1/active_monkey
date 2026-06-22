# PCC Outer-Loop Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cheap frozen-model experiment harness for Prospective Causal Calibration (PCC) on open-source LLM benchmark tasks.

**Architecture:** Put deterministic PCC scoring in a small pure-Python module, then wrap it with an experiment CLI that can run either a no-dependency smoke backend or a Hugging Face backend for Qwen/GSM8K when dependencies and model downloads are available. The first scientific question is whether forecast-calibrated scaffold selection beats score-only scaffold selection before doing any weight-update RL.

**Tech Stack:** Python standard library for PCC metrics and smoke mode; optional `transformers`, `datasets`, and `torch` for `Qwen/Qwen2.5-0.5B-Instruct` + GSM8K.

---

### Task 1: PCC Metric Core

**Files:**
- Create: `tests/test_pcc.py`
- Create: `active_loop/pcc.py`

- [ ] **Step 1: Write failing tests**

```python
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
        per_tag={
            "multi_step": {"improve": 0.8, "same": 0.15, "worse": 0.05},
            "single_step": {"improve": 0.1, "same": 0.8, "worse": 0.1},
            "unit": {"improve": 0.2, "same": 0.7, "worse": 0.1},
        }
    )

    summary = summarize_patch("p1", forecast, baseline, patched)

    assert summary.delta_accuracy == 1 / 3
    assert summary.mean_forecast_nll < 0.6
    assert summary.pcc_score > summary.delta_accuracy - 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_pcc.py -q`

Expected: fail with `ModuleNotFoundError` for `active_loop.pcc`.

- [ ] **Step 3: Implement minimal code**

Create dataclasses `Trial`, `Forecast`, `PatchSummary`; implement `summarize_patch()` by comparing baseline and patched correctness per task, mapping deltas to `improve/same/worse`, and computing mean negative log likelihood from tag forecasts.

- [ ] **Step 4: Run test to verify it passes**

Run: `/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_pcc.py -q`

Expected: pass.

### Task 2: Selection Rule

**Files:**
- Modify: `tests/test_pcc.py`
- Modify: `active_loop/pcc.py`

- [ ] **Step 1: Write failing test**

```python
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
    calibrated = Forecast({"multi_step": {"improve": 0.9, "same": 0.1}, "single_step": {"same": 0.9, "improve": 0.1}})
    overconfident_wrong = Forecast({"multi_step": {"same": 0.9, "improve": 0.1}, "single_step": {"same": 0.9, "improve": 0.1}})

    summaries = [
        summarize_patch("calibrated", calibrated, baseline, patched_good, alpha=0.2),
        summarize_patch("lucky", overconfident_wrong, baseline, patched_lucky, alpha=0.2),
    ]

    assert choose_patch(summaries).patch_id == "calibrated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_pcc.py::test_choose_patch_can_penalize_wrong_forecast -q`

Expected: fail because `choose_patch` is missing or uses the wrong scoring rule.

- [ ] **Step 3: Implement selection**

Add `choose_patch(summaries)` returning the max by `pcc_score`, with deterministic tie-break by `patch_id`.

- [ ] **Step 4: Run focused tests**

Run: `/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_pcc.py -q`

Expected: pass.

### Task 3: Executable Experiment CLI

**Files:**
- Create: `experiments/pcc_outer_loop.py`
- Modify: `tests/test_pcc.py`

- [ ] **Step 1: Write CLI smoke test**

```python
import json
import subprocess
import sys


def test_pcc_outer_loop_smoke_runs():
    out = subprocess.check_output(
        [sys.executable, "experiments/pcc_outer_loop.py", "--backend", "smoke", "--iterations", "1"],
        text=True,
    )
    payload = json.loads(out.splitlines()[-1])
    assert payload["backend"] == "smoke"
    assert payload["chosen"]["method"] in {"pcc", "score_only"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_pcc.py::test_pcc_outer_loop_smoke_runs -q`

Expected: fail because the script is missing.

- [ ] **Step 3: Implement smoke and HF modes**

The smoke backend uses deterministic synthetic task outcomes to verify the loop without dependencies. The HF backend checks for `transformers`, `datasets`, and `torch`, loads `openai/gsm8k`, runs `Qwen/Qwen2.5-0.5B-Instruct`, and prints a JSON report. If optional dependencies are unavailable, it exits with a clear non-zero message.

- [ ] **Step 4: Run smoke test**

Run: `/Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_pcc.py -q`

Expected: pass.

### Task 4: Verification and Run Notes

**Files:**
- Create: `docs/research/pcc-outer-loop-experiment.md`

- [ ] **Step 1: Document commands**

Document the smoke command and real Qwen/GSM8K command:

```bash
/Users/mirro/Projects/active-loop/.venv/bin/python experiments/pcc_outer_loop.py --backend smoke --iterations 1
/Users/mirro/Projects/active-loop/.venv/bin/python experiments/pcc_outer_loop.py --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --dataset openai/gsm8k --limit 48
```

- [ ] **Step 2: Run verification**

Run focused tests and the smoke command. Attempt the HF command only after dependencies are installed or already present.

- [ ] **Step 3: Report honestly**

Report whether the smoke harness ran, whether the real Qwen/GSM8K path was runnable in the current environment, and what approval or machine setup is needed for the full run.
