"""Collect passive process-memory records for completed experiment iterations."""

from __future__ import annotations

import argparse
import ast
import pathlib
import re
import subprocess
import sys
from datetime import datetime, timezone

from meta_monkey.schemas import (
    ArtifactStatus,
    CheckStatus,
    EntryStatus,
    MetaEpisode,
    ProcessStatus,
    SCHEMA_VERSION,
)

try:
    from loop.check_iteration import VERIFIER_FLOOR, check_entry, find_entries
except Exception:  # pragma: no cover - fallback for unusual import contexts.
    from importlib.util import module_from_spec, spec_from_file_location

    _CHECK_PATH = pathlib.Path(__file__).resolve().parents[1] / "loop" / "check_iteration.py"
    _SPEC = spec_from_file_location("loop.check_iteration", _CHECK_PATH)
    if _SPEC is None or _SPEC.loader is None:
        raise
    _MODULE = module_from_spec(_SPEC)
    _SPEC.loader.exec_module(_MODULE)
    VERIFIER_FLOOR = _MODULE.VERIFIER_FLOOR
    check_entry = _MODULE.check_entry
    find_entries = _MODULE.find_entries


def _repo_root() -> pathlib.Path:
    cwd = pathlib.Path.cwd()
    if (cwd / "EXPERIMENTS.md").exists():
        return cwd
    return pathlib.Path(__file__).resolve().parents[1]


def _read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def _experiments_text(root: pathlib.Path) -> str:
    return _read_text(root / "EXPERIMENTS.md")


def latest_experiment_number(root: pathlib.Path) -> int:
    entries = find_entries(_experiments_text(root))
    if not entries:
        raise ValueError("no experiment entries found in EXPERIMENTS.md")
    return max(entries)


def _entry_line(entry: str, label: str) -> str | None:
    match = re.search(rf"^- {re.escape(label)}:.*$", entry, re.MULTILINE)
    return match.group(0) if match else None


def _honest_caveat_line(entry: str) -> str | None:
    match = re.search(r"^- Honest caveat.*$", entry, re.MULTILINE)
    return match.group(0) if match else None


def _claimed_verdict(verdict_line: str | None) -> str | None:
    if not verdict_line:
        return None
    match = re.match(r"^- Verdict:\s*(POSITIVE|NEGATIVE|MIXED)\b", verdict_line)
    return match.group(1) if match else None


def _insight_tag(entry: str) -> str | None:
    has_consolidation = "CONSOLIDATION" in entry
    has_new = "NEW INSIGHT" in entry
    if has_consolidation and has_new:
        return "BOTH"
    if has_consolidation:
        return "CONSOLIDATION"
    if has_new:
        return "NEW INSIGHT"
    return None


def _verifier_status(exp: int, verifier_line: str | None) -> str:
    if not verifier_line:
        return "missing" if exp >= VERIFIER_FLOOR else "unknown"
    lower = verifier_line.lower()
    if "disagree" in lower or "disagreed" in lower:
        return "disagreed"
    if ("agree" in lower or "agreed" in lower) and "disagree" not in lower:
        return "agree"
    return "unknown"


def _entry_status(exp: int, experiments_text: str) -> tuple[EntryStatus, str]:
    entry = find_entries(experiments_text).get(exp, "")
    plain_line = _entry_line(entry, "Plain")
    verdict_line = _entry_line(entry, "Verdict")
    verifier_line = _entry_line(entry, "Verifier")
    honest_line = _honest_caveat_line(entry)

    return (
        EntryStatus(
            entry_exists=bool(entry),
            has_plain=plain_line is not None,
            has_verdict=verdict_line is not None,
            has_honest_caveat=honest_line is not None,
            has_verifier=verifier_line is not None,
            claimed_verdict=_claimed_verdict(verdict_line),
            insight_tag=_insight_tag(entry),
            verifier_status=_verifier_status(exp, verifier_line),
        ),
        entry,
    )


def _script_path(root: pathlib.Path, exp: int) -> str | None:
    candidates = sorted(root.glob(f"experiments/exp{exp}_*.py"))
    if not candidates:
        return None
    return candidates[0].relative_to(root).as_posix()


def _output_path(exp: int) -> str:
    return f"experiments/outputs/exp{exp}.txt"


def _site_text(root: pathlib.Path) -> str:
    path = root / "site" / "data" / "experiments-data.js"
    if not path.exists():
        return ""
    return _read_text(path)


def _has_docstring_predeclaration(root: pathlib.Path, script_path: str | None) -> bool:
    if not script_path:
        return False
    try:
        source = _read_text(root / script_path)
        docstring = ast.get_docstring(ast.parse(source)) or ""
    except (OSError, SyntaxError):
        return False
    lower = docstring.lower()
    return (
        "hypothesis" in lower
        and ("prediction" in lower or "predeclar" in lower)
        and "falsifier" in lower
    )


def _artifact_status(root: pathlib.Path, exp: int) -> ArtifactStatus:
    script_path = _script_path(root, exp)
    output_path = _output_path(exp)
    site_text = _site_text(root)

    return ArtifactStatus(
        script_path=script_path,
        output_path=output_path,
        script_exists=bool(script_path and (root / script_path).exists()),
        output_exists=(root / output_path).exists(),
        site_data_references_script=bool(script_path and script_path in site_text),
        site_data_references_output=output_path in site_text,
    )


def _check_status(root: pathlib.Path, exp: int, experiments_text: str) -> CheckStatus:
    hard_failures, warnings = check_entry(exp, experiments_text=experiments_text, root=root)
    return CheckStatus(
        hard_failures=[str(item) for item in hard_failures],
        warnings=[str(item) for item in warnings],
        passed=not hard_failures,
    )


def _has_positive_self_grade(entry: str) -> bool:
    return bool(re.search(r"\b(BREAKTHROUGH|POSITIVE-SINGLE)\b", entry))


def _likely_risks(
    *,
    entry_status: EntryStatus,
    entry_text: str,
    artifacts: ArtifactStatus,
    checks: CheckStatus,
) -> list[str]:
    risks: list[str] = []
    if not entry_status.entry_exists:
        risks.append("missing_entry")
    if not artifacts.script_exists:
        risks.append("missing_script")
    if not artifacts.output_exists:
        risks.append("missing_output")
    if entry_status.verifier_status == "missing":
        risks.append("missing_verifier")
    if entry_status.verifier_status == "disagreed":
        risks.append("verifier_disagreement")
    if checks.hard_failures:
        risks.append("check_iteration_failure")
    if checks.warnings:
        risks.append("check_iteration_warning")
    if not artifacts.site_data_references_script:
        risks.append("site_data_missing_script_reference")
    if not artifacts.site_data_references_output:
        risks.append("site_data_missing_output_reference")
    if entry_status.claimed_verdict == "POSITIVE" and not _has_positive_self_grade(entry_text):
        risks.append("positive_without_self_grade")
    return risks


def _process_notes(root: pathlib.Path, artifacts: ArtifactStatus) -> list[str]:
    notes: list[str] = []
    if artifacts.script_path and not _has_docstring_predeclaration(root, artifacts.script_path):
        notes.append("script docstring lacks hypothesis/prediction/falsifier predeclaration terms")
    return notes


def _process_status(
    root: pathlib.Path,
    entry_status: EntryStatus,
    entry_text: str,
    artifacts: ArtifactStatus,
    checks: CheckStatus,
) -> ProcessStatus:
    risks = _likely_risks(
        entry_status=entry_status,
        entry_text=entry_text,
        artifacts=artifacts,
        checks=checks,
    )
    return ProcessStatus(
        likely_risks=risks,
        process_failure=(
            bool(checks.hard_failures)
            or not entry_status.entry_exists
            or not artifacts.script_exists
            or not artifacts.output_exists
            or entry_status.verifier_status == "disagreed"
        ),
        notes=_process_notes(root, artifacts),
    )


def _future_policy_hint(process: ProcessStatus, entry_status: EntryStatus, checks: CheckStatus) -> str:
    if checks.hard_failures:
        return "repair mechanical iteration failures before interpretation"
    if entry_status.verifier_status == "disagreed":
        return "investigate verifier disagreement before logging or extending"
    if checks.warnings:
        return "audit quoted numbers and derived values before reuse"
    if (
        "site_data_missing_script_reference" in process.likely_risks
        or "site_data_missing_output_reference" in process.likely_risks
    ):
        return "regenerate/repair curated site trace before next iteration"
    return "no immediate process repair indicated"


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _commit_sha(root: pathlib.Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def collect_episode(
    root: pathlib.Path,
    exp: int,
    *,
    collected_at_utc: str | None = None,
    commit_sha: str | None = None,
) -> MetaEpisode:
    experiments_text = _experiments_text(root)
    entry_status, entry_text = _entry_status(exp, experiments_text)
    artifacts = _artifact_status(root, exp)
    checks = _check_status(root, exp, experiments_text)
    process = _process_status(root, entry_status, entry_text, artifacts, checks)

    return MetaEpisode(
        schema_version=SCHEMA_VERSION,
        exp=exp,
        collected_at_utc=collected_at_utc or _now_utc(),
        commit_sha=commit_sha if commit_sha is not None else _commit_sha(root),
        artifacts=artifacts,
        entry=entry_status,
        checks=checks,
        process=process,
        future_policy_hint=_future_policy_hint(process, entry_status, checks),
    )


def episode_path(root: pathlib.Path, exp: int) -> pathlib.Path:
    return root / "meta" / "episodes" / f"exp{exp}.json"


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def _build_parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Collect passive Meta Monkey process memory.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--exp", type=int, help="experiment number to collect")
    target.add_argument("--latest", action="store_true", help="collect the latest experiment")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print JSON without writing")
    mode.add_argument("--write", action="store_true", help="write meta/episodes/expNN.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        parser = _build_parser()
        args = parser.parse_args(argv)
        root = _repo_root()
        exp = latest_experiment_number(root) if args.latest else int(args.exp)
        episode = collect_episode(root, exp)
    except FileNotFoundError as exc:
        print(f"missing required file: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(episode.to_json(), end="")
        return 0

    path = episode_path(root, exp)
    episode.write_json(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
