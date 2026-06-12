"""Tests for the passive Meta Monkey process-memory layer."""

from __future__ import annotations

import json
import pathlib

from meta_monkey import collect_iteration
from meta_monkey.collect_iteration import collect_episode, main as collect_main
from meta_monkey.preflight import build_checklist
from meta_monkey.report import build_report, load_episodes
from meta_monkey.schemas import (
    ArtifactStatus,
    CheckStatus,
    EntryStatus,
    MetaEpisode,
    ProcessStatus,
    SCHEMA_VERSION,
)


DOCSTRING = '''"""
Hypothesis: the synthetic metric should pass.
Prediction: metric = 1.000.
Falsifier: metric < 1.000.
"""'''


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _entry(n: int, *, verifier: str = "agree", verdict: str = "POSITIVE / NEW INSIGHT") -> str:
    return f"""\
## Exp {n} - synthetic run
- Plain: A synthetic control entry.
- Setup: Synthetic setup.
- Result: metric = 1.000.
- Implication: Synthetic implication.
- Honest caveat: synthetic only.
- Verifier: {verifier}.
- Verdict: {verdict}. Self-grade: POSITIVE-SINGLE.
- Next: none.
"""


def _make_repo(
    root: pathlib.Path,
    *,
    n: int = 200,
    output: bool = True,
    verifier: str = "agree",
) -> pathlib.Path:
    _write(root / "EXPERIMENTS.md", _entry(n, verifier=verifier))
    _write(root / "experiments" / f"exp{n}_synthetic.py", DOCSTRING)
    if output:
        _write(root / "experiments" / "outputs" / f"exp{n}.txt", "metric = 1.000\n")
    _write(
        root / "experiments-data.js",
        "window.AM_EXPERIMENTS = [\n"
        f'  {{ n: {n}, trace: {{ script: "experiments/exp{n}_synthetic.py", '
        f'output: "experiments/outputs/exp{n}.txt" }} }},\n'
        "];\n",
    )
    return root


def _episode(
    exp: int = 1,
    *,
    verdict: str = "POSITIVE",
    insight: str = "NEW INSIGHT",
    verifier_status: str = "agree",
    risks: list[str] | None = None,
    process_failure: bool = False,
    passed: bool = True,
) -> MetaEpisode:
    return MetaEpisode(
        schema_version=SCHEMA_VERSION,
        exp=exp,
        collected_at_utc="2026-01-01T00:00:00Z",
        commit_sha="abc123",
        artifacts=ArtifactStatus(
            script_path=f"experiments/exp{exp}_synthetic.py",
            output_path=f"experiments/outputs/exp{exp}.txt",
            script_exists=True,
            output_exists=True,
            site_data_references_script=True,
            site_data_references_output=True,
        ),
        entry=EntryStatus(
            entry_exists=True,
            has_plain=True,
            has_verdict=True,
            has_honest_caveat=True,
            has_verifier=True,
            claimed_verdict=verdict,
            insight_tag=insight,
            verifier_status=verifier_status,
        ),
        checks=CheckStatus(
            hard_failures=[] if passed else ["failure"],
            warnings=[],
            passed=passed,
        ),
        process=ProcessStatus(
            likely_risks=risks or [],
            process_failure=process_failure,
            notes=[],
        ),
        future_policy_hint="no immediate process repair indicated",
    )


def test_schema_roundtrip_produces_stable_json():
    episode = _episode()

    text = episode.to_json()
    assert text.endswith("\n")
    assert json.loads(text)["schema_version"] == 1
    assert text.splitlines()[1].strip().startswith('"artifacts"')

    loaded = MetaEpisode.from_json(text)
    assert loaded == episode


def test_collector_dry_run_on_synthetic_exp_does_not_write(tmp_path, monkeypatch, capsys):
    _make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(collect_iteration, "_now_utc", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(collect_iteration, "_commit_sha", lambda _root: None)

    assert collect_main(["--exp", "200", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert '"exp": 200' in output
    assert not (tmp_path / "meta" / "episodes" / "exp200.json").exists()


def test_collector_write_creates_episode_json(tmp_path, monkeypatch):
    _make_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(collect_iteration, "_now_utc", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr(collect_iteration, "_commit_sha", lambda _root: "abc123")

    assert collect_main(["--exp", "200", "--write"]) == 0

    path = tmp_path / "meta" / "episodes" / "exp200.json"
    episode = MetaEpisode.read_json(path)
    assert episode.exp == 200
    assert episode.commit_sha == "abc123"
    assert episode.artifacts.script_exists is True


def test_collector_detects_missing_output_as_process_risk(tmp_path):
    root = _make_repo(tmp_path, output=False)

    episode = collect_episode(
        root,
        200,
        collected_at_utc="2026-01-01T00:00:00Z",
        commit_sha=None,
    )

    assert episode.artifacts.output_exists is False
    assert "missing_output" in episode.process.likely_risks
    assert episode.process.process_failure is True


def test_collector_detects_verifier_agree_and_disagreed(tmp_path):
    agree_root = _make_repo(tmp_path / "agree", verifier="agree")
    disagree_root = _make_repo(tmp_path / "disagree", verifier="disagreed on verdict")

    agree = collect_episode(agree_root, 200, collected_at_utc="2026-01-01T00:00:00Z", commit_sha=None)
    disagreed = collect_episode(
        disagree_root,
        200,
        collected_at_utc="2026-01-01T00:00:00Z",
        commit_sha=None,
    )

    assert agree.entry.verifier_status == "agree"
    assert disagreed.entry.verifier_status == "disagreed"
    assert "verifier_disagreement" in disagreed.process.likely_risks


def test_report_handles_empty_episodes_directory(tmp_path):
    report = build_report(load_episodes(tmp_path))
    assert "total episodes: 0" in report
    assert "No episode records found" in report


def test_report_summarizes_synthetic_episodes(tmp_path):
    episode_dir = tmp_path / "meta" / "episodes"
    _episode(1, risks=["missing_output"], process_failure=True, passed=False).write_json(
        episode_dir / "exp1.json"
    )
    _episode(
        2,
        verdict="NEGATIVE",
        insight="CONSOLIDATION",
        verifier_status="disagreed",
        risks=["missing_output", "verifier_disagreement"],
    ).write_json(episode_dir / "exp2.json")

    report = build_report(load_episodes(tmp_path))

    assert "total episodes: 2" in report
    assert "POSITIVE: 1" in report
    assert "NEGATIVE: 1" in report
    assert "NEW INSIGHT: 1" in report
    assert "CONSOLIDATION: 1" in report
    assert "process failures: 1" in report
    assert "missing_output: 2" in report
    assert "verifier disagreements: 1" in report
    assert "check failures: 1" in report


def test_preflight_prints_advisory_and_references_protocol_routing(tmp_path):
    _write(tmp_path / "loop" / "LESSONS.md", "# Lessons\n- L1: remember the raw output.\n")
    episode_dir = tmp_path / "meta" / "episodes"
    _episode(1, risks=["missing_output"]).write_json(episode_dir / "exp1.json")
    _episode(2, risks=["missing_output"]).write_json(episode_dir / "exp2.json")

    checklist = build_checklist(tmp_path)

    assert "Advisory only: this is passive meta memory, not a controller." in checklist
    assert "loop/PROTOCOL.md" in checklist
    assert "loop/ROUTING.md" in checklist
    assert "Recent repeated risks" in checklist
    assert "missing_output: 2" in checklist
    assert "does not choose the next experiment" in checklist
