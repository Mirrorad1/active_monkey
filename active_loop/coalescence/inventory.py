"""Experiment INVENTORY tool for the active_loop coalescence layer.

Enumerates every experiment with its direction/status/verdict and HONESTLY
reports what source evidence exists on disk.  NEVER hallucinate missing data.
If only a summary exists, says summary-only.

Public API
----------
build_inventory(repo_root=".") -> dict
inventory_json(repo_root=".") -> str   (deterministic canonical JSON)
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from active_loop.coalescence.schema import (
    BACKFILL_LEVEL_NAMES,
    CONFIDENCE_LEVELS,
    SCHEMA_VERSION,
    backfill_level_name,
)
from active_loop.artifacts import SOURCE_EXPERIMENTS, DEFAULT_ARTIFACT_ID, repo_commit
from active_loop.site_data import parse_experiments
from active_loop.state import canonical_json


# ---------------------------------------------------------------------------
# Direction map — ordered list of (lo, hi, direction); most specific wins.
# Single-experiment entries come first (narrowest range wins on overlap).
# ---------------------------------------------------------------------------

_DIRECTION_RANGES: list[tuple[int, int, str]] = [
    (210, 210, "active-sensing"),
    (211, 211, "uncertainty-gated-active-sensing"),
    (125, 132, "affective-dyad"),
    (214, 226, "affective-dyad"),
    (1, 16, "language"),
    (17, 40, "embodiment-valence-recipe"),
    (41, 124, "persistent-creature"),
    (133, 154, "continuous-substrate"),
    (155, 173, "meta-calibration-n3"),
    (174, 193, "identity-n4"),
    (194, 198, "population-ecology"),
    (199, 207, "costed-sensing"),
    (208, 209, "hidden-state-memory"),
    (212, 213, "evolvability-geometry"),
]


def _direction_for(n: int) -> str:
    """Return the most-specific direction for experiment number n."""
    for lo, hi, direction in _DIRECTION_RANGES:
        if lo <= n <= hi:
            return direction
    return "unknown"


# ---------------------------------------------------------------------------
# Scorer map: direction -> scorer path (relative to repo root)
# ---------------------------------------------------------------------------

_SCORER_MAP: dict[str, str] = {
    "affective-dyad": "eval/affect_score.py",
    "language": "eval/lang_score.py",
    "embodiment-valence-recipe": "eval/lang_score.py",
}

# Fallback scorer for language/embodiment if lang_score.py absent
_SCORER_FALLBACK: dict[str, str] = {
    "language": "eval/score.py",
    "embodiment-valence-recipe": "eval/score.py",
}


# ---------------------------------------------------------------------------
# JS extractor (tolerant regex — not JSON)
# ---------------------------------------------------------------------------

def _extract_js_entries(js_path: Path) -> dict[int, dict[str, Any]]:
    """Parse AM_EXPERIMENTS array body and return {n: {kind, chapter, script, output}}."""
    text = js_path.read_text(encoding="utf-8")

    start_m = re.search(r"window\.AM_EXPERIMENTS\s*=\s*\[", text)
    if not start_m:
        return {}
    body_start = start_m.end()
    rest = text[body_start:]
    end_m = re.search(r"^\s*\]\s*;", rest, re.MULTILINE)
    if not end_m:
        return {}
    body = rest[: end_m.start()]

    result: dict[int, dict[str, Any]] = {}

    # Split on entry openers: { n:<digits>,
    entry_starts = [
        (m.start(), int(m.group(1)))
        for m in re.finditer(r"\{\s*n\s*:\s*(\d+)\s*,", body)
    ]

    for i, (pos, n) in enumerate(entry_starts):
        end_pos = entry_starts[i + 1][0] if i + 1 < len(entry_starts) else len(body)
        chunk = body[pos:end_pos]

        kind_m = re.search(r"kind\s*:\s*\"(\w+)\"", chunk)
        chapter_m = re.search(r"chapter\s*:\s*\"(\w+)\"", chunk)

        # Extract trace block content
        trace_m = re.search(r"trace\s*:\s*\{([^}]*)\}", chunk, re.DOTALL)
        script = None
        output = None
        if trace_m:
            trace_body = trace_m.group(1)
            sm = re.search(r'script\s*:\s*"([^"]+)"', trace_body)
            om = re.search(r'output\s*:\s*"([^"]+)"', trace_body)
            if sm:
                script = sm.group(1)
            if om:
                output = om.group(1)

        result[n] = {
            "kind": kind_m.group(1) if kind_m else "",
            "chapter": chapter_m.group(1) if chapter_m else "",
            "script": script,
            "output": output,
        }

    return result


# ---------------------------------------------------------------------------
# Script detection helpers
# ---------------------------------------------------------------------------

def _find_scripts(n: int, repo_root: Path) -> list[str]:
    """Return list of matched script paths (relative to repo_root) for experiment n."""
    found: list[str] = []

    # Patterns to check: exp02, exp002, exp2_ etc.
    patterns = [
        f"exp{n:02d}*.py",
        f"exp{n:03d}*.py",
        f"exp{n}_*.py",
    ]

    search_dirs = [
        repo_root / "experiments",
    ]
    # For n <= 40, also check recovered/
    if n <= 40:
        search_dirs.append(repo_root / "experiments" / "recovered")

    seen: set[str] = set()
    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in patterns:
            for p in sorted(d.glob(pattern)):
                rel = str(p.relative_to(repo_root))
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)

    return found


def _find_output_txts(n: int, repo_root: Path) -> list[str]:
    """Return list of matched human-readable output txt paths for experiment n."""
    found: list[str] = []
    outputs_dir = repo_root / "experiments" / "outputs"
    recovered_outputs = repo_root / "experiments" / "recovered" / "outputs"

    patterns = [f"exp{n:03d}*.txt", f"exp{n:02d}*.txt"]

    seen: set[str] = set()
    for d in [outputs_dir] + ([recovered_outputs] if n <= 40 else []):
        if not d.exists():
            continue
        for pattern in patterns:
            for p in sorted(d.glob(pattern)):
                rel = str(p.relative_to(repo_root))
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)

    return found


def _find_metrics_json(n: int, repo_root: Path) -> list[str]:
    """Return structured *.json files for experiment n under experiments/outputs/."""
    outputs_dir = repo_root / "experiments" / "outputs"
    if not outputs_dir.exists():
        return []

    found: list[str] = []
    patterns = [f"exp{n:03d}*.json", f"exp{n:02d}*.json", f"exp{n}_*.json"]
    seen: set[str] = set()
    for pattern in patterns:
        for p in sorted(outputs_dir.glob(pattern)):
            rel = str(p.relative_to(repo_root))
            if rel not in seen:
                seen.add(rel)
                found.append(rel)

    # Also check subdirs: exp{n}_*/verdict.json
    for pattern in [f"exp{n}_*/verdict.json", f"exp{n:03d}_*/verdict.json"]:
        for p in sorted(outputs_dir.glob(pattern)):
            rel = str(p.relative_to(repo_root))
            if rel not in seen:
                seen.add(rel)
                found.append(rel)

    return found


def _find_trajectory_subdirs(n: int, repo_root: Path) -> list[str]:
    """Return output subdirectory paths containing trajectory files for experiment n."""
    outputs_dir = repo_root / "experiments" / "outputs"
    if not outputs_dir.exists():
        return []

    found: list[str] = []
    # Match exp{n}_* or exp{n:03d}_* subdirectory names
    patterns = [f"exp{n}_*", f"exp{n:03d}_*", f"exp{n:02d}_*"]
    seen: set[str] = set()

    for pattern in patterns:
        for subdir in sorted(outputs_dir.glob(pattern)):
            if not subdir.is_dir():
                continue
            # Check for trajectory files inside
            has_traj = (
                any(subdir.glob("traj_*.json"))
                or (subdir / "trajectories.json").exists()
                or (subdir / "events.jsonl").exists()
            )
            if has_traj:
                rel = str(subdir.relative_to(repo_root))
                if rel not in seen:
                    seen.add(rel)
                    found.append(rel)

    return found


def _git_tracked(path_rel: str, repo_root: Path) -> bool:
    """Return True if the file is tracked by git (exit code 0)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", path_rel],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_artifact_manifests(repo_root: Path) -> dict[int, list[str]]:
    """Scan all artifacts/*/manifest.json for source_experiments. Return {n: [artifact_id]}."""
    artifacts_dir = repo_root / "artifacts"
    result: dict[int, list[str]] = {}
    if not artifacts_dir.exists():
        return result
    for manifest_path in sorted(artifacts_dir.glob("*/manifest.json")):
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        art_id = data.get("artifact_id") or manifest_path.parent.name
        # Check both top-level and nested provenance keys
        exp_list: list[int] = []
        for key in ("source_experiments",):
            val = data.get(key)
            if isinstance(val, list):
                exp_list.extend(int(x) for x in val if isinstance(x, (int, float, str)))
        prov = data.get("provenance", {})
        if isinstance(prov, dict):
            for key in ("source_experiments",):
                val = prov.get(key)
                if isinstance(val, list):
                    exp_list.extend(int(x) for x in val if isinstance(x, (int, float, str)))
        for en in exp_list:
            result.setdefault(en, [])
            if art_id not in result[en]:
                result[en].append(art_id)
    return result


# ---------------------------------------------------------------------------
# Build a single experiment record
# ---------------------------------------------------------------------------

def _build_experiment_record(
    n: int,
    md_entry: dict,
    js_entry: dict | None,
    artifact_exp_map: dict[int, list[str]],
    scorer_files_exist: dict[str, str | None],
    repo_root: Path,
) -> dict:
    """Build one inventory record for experiment n."""
    direction = _direction_for(n)

    # --- Chapter from JS (coarse), direction from range map (fine) ---
    chapter = js_entry.get("chapter", "") if js_entry else ""

    # --- Status + verdict ---
    tag = md_entry.get("tag", "")
    status = "completed" if tag else "unknown"
    verdict = tag if tag else "unknown"

    # --- Script detection ---
    script_paths = _find_scripts(n, repo_root)
    has_script = len(script_paths) > 0

    # --- Output txt detection ---
    output_paths = _find_output_txts(n, repo_root)
    has_output = len(output_paths) > 0

    # --- Metrics JSON detection ---
    metrics_paths = _find_metrics_json(n, repo_root)
    has_metrics = len(metrics_paths) > 0

    # --- Raw trajectory subdirectory detection ---
    traj_dirs = _find_trajectory_subdirs(n, repo_root)
    has_raw_trajectories = len(traj_dirs) > 0

    # --- Scorer detection ---
    scorer_path_for_dir = _SCORER_MAP.get(direction)
    has_scorer = False
    scorer_path: str | None = None
    if scorer_path_for_dir is not None:
        resolved = scorer_files_exist.get(scorer_path_for_dir)
        if resolved is not None:
            has_scorer = True
            scorer_path = resolved
        else:
            # Try fallback for language/embodiment
            fallback = _SCORER_FALLBACK.get(direction)
            if fallback:
                resolved_fb = scorer_files_exist.get(fallback)
                if resolved_fb is not None:
                    has_scorer = True
                    scorer_path = resolved_fb

    # --- Checkpoint detection ---
    # From hardcoded SOURCE_EXPERIMENTS (authoritative)
    checkpoint_artifact_ids: list[str] = []
    if n in SOURCE_EXPERIMENTS:
        checkpoint_artifact_ids.append(DEFAULT_ARTIFACT_ID)
    # From manifest scan
    for art_id in artifact_exp_map.get(n, []):
        if art_id not in checkpoint_artifact_ids:
            checkpoint_artifact_ids.append(art_id)
    has_checkpoint = len(checkpoint_artifact_ids) > 0

    # --- Repro command ---
    if has_script:
        tracked = _git_tracked(script_paths[0], repo_root)
        has_repro_command = "script_committed" if tracked else "unknown"
    else:
        has_repro_command = "unknown"

    # --- Confidence ---
    if has_raw_trajectories or has_checkpoint or (has_script and has_metrics):
        confidence = "high"
    elif has_metrics or (has_script and has_output):
        confidence = "medium"
    elif has_output:
        confidence = "low"
    else:
        confidence = "unknown"

    # --- Backfill level (highest achievable) ---
    if has_checkpoint:
        backfill_level_possible = "checkpoint_bundle"
    elif has_raw_trajectories:
        backfill_level_possible = "trajectory_bundle"
    elif has_script:
        backfill_level_possible = "repro_bundle"
    elif has_metrics:
        backfill_level_possible = "metrics_bundle"
    elif has_output:
        backfill_level_possible = "summary_bundle"
    else:
        backfill_level_possible = "index_only"

    # --- Available sources (honest: only actually present) ---
    available_sources: list[str] = ["experiments-data.js"]
    for sp in script_paths:
        if sp not in available_sources:
            available_sources.append(sp)
    for op in output_paths:
        if op not in available_sources:
            available_sources.append(op)
    for mp in metrics_paths:
        if mp not in available_sources:
            available_sources.append(mp)
    for td in traj_dirs:
        if td not in available_sources:
            available_sources.append(td)
    if scorer_path:
        if scorer_path not in available_sources:
            available_sources.append(scorer_path)
    for art_id in checkpoint_artifact_ids:
        label = f"artifact:{art_id}"
        if label not in available_sources:
            available_sources.append(label)

    # --- Notes (honest) ---
    notes: list[str] = []
    if has_raw_trajectories:
        notes.append(f"raw per-seed trajectory JSON present in: {', '.join(traj_dirs)}")
    if has_checkpoint:
        notes.append(f"checkpoint artifact {checkpoint_artifact_ids[0]} covers this exp")
    if has_metrics and not has_raw_trajectories:
        notes.append("metrics JSON present, no raw trajectories — do not infer trajectories")
    if has_script and has_output and not has_metrics and not has_raw_trajectories:
        committed = has_repro_command == "script_committed"
        notes.append(
            "summary txt only; script committed + deterministic (rerunnable)"
            if committed
            else "summary txt only; script present but git tracking unconfirmed"
        )
    if not has_script and not has_output:
        notes.append("mention-only — no script or output found on disk")

    return {
        "experiment_id": f"exp{n:03d}" if n >= 100 else f"exp{n:02d}",
        "n": n,
        "direction": direction,
        "chapter": chapter,
        "status": status,
        "verdict": verdict,
        "available_sources": available_sources,
        "has_script": has_script,
        "has_output": has_output,
        "has_metrics": has_metrics,
        "has_raw_trajectories": has_raw_trajectories,
        "has_scorer": has_scorer,
        "has_checkpoint": has_checkpoint,
        "has_repro_command": has_repro_command,
        "confidence": confidence,
        "backfill_level_possible": backfill_level_possible,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def build_inventory(repo_root: str | Path = ".") -> dict:
    """Build the full experiment inventory.

    Returns a dict with keys:
        schema_version, repo_commit, count, counts_by_direction,
        counts_by_confidence, experiments (sorted by n).
    """
    repo_root = Path(repo_root).resolve()
    js_path = repo_root / "site" / "data" / "experiments-data.js"

    # --- Load MD data (authoritative for verdict) ---
    md_path = repo_root / "EXPERIMENTS.md"
    md_list = parse_experiments(md_path)
    md_map: dict[int, dict] = {e["n"]: e for e in md_list}

    # --- Load JS data (chapter, kind, script, output hints) ---
    js_map: dict[int, dict] = {}
    if js_path.exists():
        js_map = _extract_js_entries(js_path)

    # --- Union of all known experiment numbers ---
    all_ns: set[int] = set(md_map.keys()) | set(js_map.keys())

    # --- Pre-check which scorer files exist ---
    all_scorer_paths = set(_SCORER_MAP.values()) | set(_SCORER_FALLBACK.values())
    scorer_files_exist: dict[str, str | None] = {}
    for sp in all_scorer_paths:
        full = repo_root / sp
        scorer_files_exist[sp] = sp if full.exists() else None

    # --- Pre-scan artifact manifests ---
    artifact_exp_map = _check_artifact_manifests(repo_root)

    # --- Build records ---
    records: list[dict] = []
    discrepancy_ns: list[int] = []

    for n in sorted(all_ns):
        in_md = n in md_map
        in_js = n in js_map
        if in_md and not in_js:
            discrepancy_ns.append(n)
        elif in_js and not in_md:
            discrepancy_ns.append(n)

        # Prefer MD data; use empty dict if absent
        md_entry = md_map.get(n, {"n": n, "title": "", "tag": "", "is_breakthrough": False})
        js_entry = js_map.get(n)

        record = _build_experiment_record(
            n=n,
            md_entry=md_entry,
            js_entry=js_entry,
            artifact_exp_map=artifact_exp_map,
            scorer_files_exist=scorer_files_exist,
            repo_root=repo_root,
        )

        # Annotate discrepancies
        if n in discrepancy_ns:
            if n in md_map and n not in js_map:
                record["notes"].append(
                    "discrepancy: present in EXPERIMENTS.md but not in experiments-data.js"
                )
            else:
                record["notes"].append(
                    "discrepancy: present in experiments-data.js but not in EXPERIMENTS.md"
                )

        records.append(record)

    # --- Counts ---
    counts_by_direction: dict[str, int] = {}
    counts_by_confidence: dict[str, int] = {lvl: 0 for lvl in CONFIDENCE_LEVELS}

    for rec in records:
        d = rec["direction"]
        counts_by_direction[d] = counts_by_direction.get(d, 0) + 1
        c = rec["confidence"]
        counts_by_confidence[c] = counts_by_confidence.get(c, 0) + 1

    return {
        "schema_version": SCHEMA_VERSION,
        "repo_commit": repo_commit(repo_root),
        "count": len(records),
        "counts_by_direction": counts_by_direction,
        "counts_by_confidence": counts_by_confidence,
        "experiments": records,
    }


def inventory_json(repo_root: str | Path = ".") -> str:
    """Return deterministic canonical JSON of the inventory."""
    return canonical_json(build_inventory(repo_root))
