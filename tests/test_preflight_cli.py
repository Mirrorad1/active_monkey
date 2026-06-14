"""
tests/test_preflight_cli.py — fast integration tests for the Evolvability Preflight
orchestration layer (runner, CLI, report).

Performance guardrail: all runs use horizon<=200, 2 seeds, <=2 gates.
A full pass should take well under 30 s total.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ecology.evolvability import (
    THERMOSENSE_AXIS,
    AggregateVerdict,
    PreflightConfig,
    run_preflight,
)
from ecology.evolvability import io

# ---------------------------------------------------------------------------
# Forage-regime base overrides (Exp 203 / thermosense_smoke)
# ---------------------------------------------------------------------------
_FORAGE_OVERRIDES = {
    "enable_thermosense": True,
    "enable_temperature": True,
    "temperature_stress_scale": 0.0,
    "thermosense_upkeep_floor": 0.0,
    "thermosense_active_threshold": 0.05,
    "thermosense_noise_base": 0.5,
    "thermal_avoidance_weight": 4.0,
    "food_optimal_base": 0.5,
    "food_optimal_amplitude": 0.3,
    "food_optimal_period": 1500.0,
    "food_concentration": 14.0,
    "food_band_width": 0.08,
    "enable_food_coupling": True,
    "thermosense_forage_mode": True,
    "regen_rate": 0.20,
    "max_population": 8000,
    "shuffle_creature_order": True,
}

_FOUNDER_OVERRIDES = {"temperature_tolerance": 0.10}

_VALID_AGGREGATE_VERDICTS = {v.value for v in AggregateVerdict}


def _make_tiny_cfg(output_dir: str) -> PreflightConfig:
    """Tiny config: 2 seeds, horizon 200, 2 gates."""
    return PreflightConfig(
        slug="thermosense_smoke",
        description="smoke test",
        base_scenario="balanced",
        base_overrides=_FORAGE_OVERRIDES,
        founder_overrides=_FOUNDER_OVERRIDES,
        trait=THERMOSENSE_AXIS,
        seeds=(38, 39),
        horizon=200,
        measurement_window=(50, 150),
        output_dir=output_dir,
        gates=("local_pairwise_gradient", "null_guards"),
        min_valid_seeds=1,
        min_population=5,
    )


# ---------------------------------------------------------------------------
# Test 1: artefact layout
# ---------------------------------------------------------------------------

def test_run_preflight_artefacts(tmp_path):
    """run_preflight creates the expected directory tree and files."""
    cfg = _make_tiny_cfg(str(tmp_path))
    result = run_preflight(cfg, run_id="t1")

    run_dir = tmp_path / "thermosense_smoke" / "t1"
    assert run_dir.is_dir(), f"run_dir not found: {run_dir}"

    # Config artefacts
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "config_hash.txt").is_file()
    assert (run_dir / "git_commit.txt").is_file()

    # Summary artefacts
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "summary.csv").is_file()
    assert (run_dir / "report.md").is_file()

    # Raw JSONL files for each requested gate
    assert (run_dir / "raw" / "local_pairwise_gradient.jsonl").is_file()
    assert (run_dir / "raw" / "null_guards.jsonl").is_file()

    # summary.json must parse and contain a valid aggregate_verdict
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["aggregate_verdict"] in _VALID_AGGREGATE_VERDICTS, (
        f"unexpected aggregate_verdict: {summary['aggregate_verdict']!r}"
    )

    # report.md must be non-empty and contain required sections
    report_text = (run_dir / "report.md").read_text()
    assert len(report_text) > 50
    assert "## Verdict" in report_text
    assert "## What This Does NOT Prove" in report_text

    # Raw JSONL rows must carry slug / run_id / config_hash
    rows = io.read_jsonl(run_dir / "raw" / "local_pairwise_gradient.jsonl")
    assert len(rows) > 0, "local_pairwise_gradient raw JSONL is empty"
    for row in rows:
        assert row.get("slug") == "thermosense_smoke"
        assert row.get("run_id") == "t1"
        assert row.get("config_hash") == cfg.config_hash()

    # null_guards raw should be empty list (gate G writes no raw rows)
    ng_rows = io.read_jsonl(run_dir / "raw" / "null_guards.jsonl")
    assert isinstance(ng_rows, list)  # could be empty — that is correct


# ---------------------------------------------------------------------------
# Test 2: no-overwrite policy
# ---------------------------------------------------------------------------

def test_no_overwrite(tmp_path):
    """Calling run_preflight twice with the same run_id raises FileExistsError."""
    cfg = _make_tiny_cfg(str(tmp_path))
    run_preflight(cfg, run_id="t1")
    with pytest.raises(FileExistsError):
        run_preflight(cfg, run_id="t1")


# ---------------------------------------------------------------------------
# Test 3: CLI smoke via subprocess (or direct main call)
# ---------------------------------------------------------------------------

_SMOKE_CONFIG = (
    Path(__file__).parent.parent
    / "experiments" / "configs" / "preflight" / "thermosense_smoke.json"
)


def test_cli_smoke(tmp_path):
    """CLI (via experiments/run_preflight.py) exits 0 even for a negative verdict."""
    smoke_script = (
        Path(__file__).parent.parent / "experiments" / "run_preflight.py"
    )

    # Prefer direct call to avoid subprocess overhead and fragility;
    # fall back to subprocess if the direct call path has issues.
    try:
        from ecology.evolvability.__main__ import main as _main
        rc = _main([
            "--config", str(_SMOKE_CONFIG),
            "--output-dir", str(tmp_path),
            "--run-id", "c1",
        ])
        assert rc == 0, f"main() returned {rc}"
    except SystemExit as e:
        assert e.code == 0, f"main() raised SystemExit({e.code})"

    # Check that the run directory and report were created
    run_dir = tmp_path / "thermosense_smoke" / "c1"
    assert run_dir.is_dir(), f"CLI run_dir not found: {run_dir}"
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "summary.json").is_file()


def test_cli_subprocess(tmp_path):
    """Full subprocess smoke: prove CLI exits 0 for a negative scientific verdict."""
    result = subprocess.run(
        [
            sys.executable, "-m", "ecology.evolvability",
            "--config", str(_SMOKE_CONFIG),
            "--output-dir", str(tmp_path),
            "--run-id", "sub1",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"CLI exited {result.returncode}\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    run_dir = tmp_path / "thermosense_smoke" / "sub1"
    assert (run_dir / "report.md").is_file()


# ---------------------------------------------------------------------------
# Test 4: thermosense smoke re-confirms closed Exp 203-207 finding
# ---------------------------------------------------------------------------

def test_aggregate_verdict_is_valid_member(tmp_path):
    """Smoke: the preflight runs end-to-end and emits a valid AggregateVerdict.

    This test does NOT assert the verdict value. At smoke scale (horizon=200,
    2 seeds) the result is scientifically meaningless: with 2 seeds the win
    threshold is 2, so two chance-wins (~25% likely under a fair coin) spuriously
    read PASS. The scientific claim — that thermosense shows no positive local
    gradient — is checked at proper scale by `test_full_scale_local_gradient_not_positive`
    below (a `slow` test). Naming honesty: this test only checks membership, so it is
    NOT named `*_not_pass` (the prior name lied about what it asserted).
    """
    cfg = _make_tiny_cfg(str(tmp_path))
    result = run_preflight(cfg, run_id="verdict_check")

    assert result.aggregate_verdict in _VALID_AGGREGATE_VERDICTS, (
        f"aggregate_verdict {result.aggregate_verdict!r} is not a valid AggregateVerdict"
    )
    print(
        f"\n[smoke finding] aggregate_verdict at horizon=200/2-seeds: "
        f"{result.aggregate_verdict} — {result.failure_reason}"
    )


@pytest.mark.slow
def test_full_scale_local_gradient_not_positive(tmp_path):
    """Proper-scale regression guard: thermosense's LOCAL gradient is NOT positive.

    Runs ONLY the binding gate (local_pairwise_gradient) at 8 seeds / horizon 800 —
    enough to discriminate, far cheaper than a full 5-gate preflight. Reproduces the
    closed Exp 203-207 conclusion: a single-step thermosense mutant does not robustly
    invade the resident (it is one or more seeds short of the 7/8 strict bar), so the
    verdict must NOT be POSITIVE_LOCAL_GRADIENT. This is the durable guard that the
    framework does not over-declare a positive gradient (acceptance: positives are not
    easier than negatives). Deselected by default (`-m 'not slow'`); ~20-40 s.

    Measured at authoring (8 seeds, horizon 800): wins 5/8 -> FLAT_OR_NOISY.
    """
    from ecology.evolvability import gates as G
    from ecology.evolvability.verdicts import GradientVerdict

    base_cfg = G.build_base_cfg("balanced", 800, _FORAGE_OVERRIDES)
    import dataclasses as D
    base_cfg = D.replace(base_cfg, founder=D.replace(base_cfg.founder, **_FOUNDER_OVERRIDES))
    out = G.run_local_pairwise_gradient(
        base_cfg, THERMOSENSE_AXIS, [38, 39, 40, 41, 42, 43, 44, 45],
        win_threshold=7, lose_threshold=3, min_valid=6,
        window=(50, 800), min_pop=10,
    )
    assert out.verdict != GradientVerdict.POSITIVE_LOCAL_GRADIENT.value, (
        f"thermosense local gradient unexpectedly POSITIVE at 8 seeds "
        f"(wins {out.aggregate['wins']}/{out.aggregate['n_valid']}) — "
        f"contradicts the closed Exp 203-207 finding; investigate before trusting the framework"
    )
