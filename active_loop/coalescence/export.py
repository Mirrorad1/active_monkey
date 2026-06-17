"""Export bundles from the coalescence layer.

Builds a self-contained bundle directory for one experiment at a given
backfill level.  HONESTY GATE: refuses to export at a level higher than
the evidence supports.  Raw data is NEVER copied — original committed paths
are referenced in place.
"""
from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from active_loop.coalescence import schema
from active_loop.coalescence.inventory import build_inventory
from active_loop.coalescence import validate


def export_bundle(
    experiment_id: str,
    level: str,
    out_dir: Any,
    repo_root: str = ".",
    created_at: Optional[str] = None,
) -> dict:
    """Build and write a bundle for one experiment at the given backfill level.

    Parameters
    ----------
    experiment_id:
        e.g. "exp222".
    level:
        One of BACKFILL_LEVEL_NAMES.
    out_dir:
        Directory to write the bundle into (created if absent).
    repo_root:
        Path to the repository root.  Defaults to cwd.
    created_at:
        ISO-8601 timestamp string; defaults to datetime.now(UTC).

    Returns
    -------
    The manifest dict.

    Raises
    ------
    ValueError
        If experiment_id is not found, or if level exceeds the evidence.
    """
    out_dir = Path(out_dir)
    repo_root_path = Path(repo_root).resolve()

    # ── 1. Look up the experiment ────────────────────────────────────────────
    inv = build_inventory(repo_root)
    record = None
    for exp in inv["experiments"]:
        if exp["experiment_id"] == experiment_id:
            record = exp
            break
    if record is None:
        raise ValueError(f"experiment {experiment_id!r} not found in inventory")

    # ── 2. HONESTY GATE ──────────────────────────────────────────────────────
    requested_idx = schema.backfill_level_index(level)
    possible_idx = schema.backfill_level_index(record["backfill_level_possible"])
    if requested_idx > possible_idx:
        raise ValueError(
            f"cannot export {experiment_id} at {level}: "
            f"evidence only supports {record['backfill_level_possible']}"
        )

    # ── 3. Resolve paths ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)

    # Source files: scripts from available_sources (those under experiments/)
    source_files: list[str] = []
    for src in record["available_sources"]:
        if src.startswith("experiments/") and src.endswith(".py"):
            fp = repo_root_path / src
            if fp.exists():
                source_files.append(src)

    # Metrics refs: only if level >= metrics_bundle and metrics exist
    metrics_refs: list[str] = []
    if requested_idx >= schema.backfill_level_index("metrics_bundle") and record["has_metrics"]:
        for src in record["available_sources"]:
            if src.startswith("experiments/") and src.endswith(".json"):
                fp = repo_root_path / src
                if fp.exists():
                    metrics_refs.append(src)

    # Raw data refs: only if level >= trajectory_bundle and has_raw_trajectories
    raw_data_refs: list[str] = []
    if (
        requested_idx >= schema.backfill_level_index("trajectory_bundle")
        and record["has_raw_trajectories"]
    ):
        for src in record["available_sources"]:
            if not src.startswith("experiments/"):
                continue
            fp = repo_root_path / src
            if fp.exists() and fp.is_dir():
                raw_data_refs.append(src)

    # Scorer refs: bundle-local reference if has_scorer
    scorer_refs: list[str] = ["scorer_card.json"] if record["has_scorer"] else []

    # State refs: artifact dirs if level >= checkpoint_bundle and has_checkpoint
    state_refs: list[str] = []
    if (
        requested_idx >= schema.backfill_level_index("checkpoint_bundle")
        and record["has_checkpoint"]
    ):
        for src in record["available_sources"]:
            if src.startswith("artifact:"):
                art_id = src[len("artifact:"):]
                art_dir = f"artifacts/{art_id}"
                fp = repo_root_path / art_dir
                if fp.exists():
                    state_refs.append(art_dir)

    # Reproduction command
    reproduction_command: Optional[str] = None
    if record["has_script"] and source_files:
        reproduction_command = (
            f"uv run --python .venv python {source_files[0]}"
        )

    # Created-at timestamp
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    # Question from inventory title (md title) if available; fall back to experiment_id
    # inventory records don't carry title — use a note-derived summary
    question = (
        "(see EXPERIMENTS.md for the experiment title/question)"
        if True  # always fall back since title isn't in record
        else experiment_id
    )
    # We can pull title from site_data directly
    try:
        from active_loop.site_data import parse_experiments
        md_path = repo_root_path / "EXPERIMENTS.md"
        if md_path.exists():
            md_list = parse_experiments(md_path)
            md_map = {e["n"]: e for e in md_list}
            n = record["n"]
            if n in md_map and md_map[n].get("title"):
                question = md_map[n]["title"]
    except Exception:
        pass

    # ── 4. Build ExperimentBundle ────────────────────────────────────────────
    bundle = schema.ExperimentBundle(
        experiment_id=experiment_id,
        direction=record["direction"],
        question=question,
        hypothesis=(
            "(not separately structured; see source_files and EXPERIMENTS.md)"
        ),
        status=record["status"],
        verdict=record["verdict"],
        repo_commit=schema.repo_commit(repo_root_path),
        created_at=created_at,
        confidence=record["confidence"],
        backfill_level=level,
        source_files=source_files,
        metrics_refs=metrics_refs,
        raw_data_refs=raw_data_refs,
        scorer_refs=scorer_refs,
        state_refs=state_refs,
        caveats=list(record["notes"]),
        reproduction_command=reproduction_command,
    )

    manifest_dict = bundle.to_dict()

    # ── 5. Write bundle files ────────────────────────────────────────────────

    # manifest.json
    schema.write_json(manifest_dict, out_dir / "manifest.json")

    # README.md
    readme_lines = [
        f"# Bundle: {experiment_id}",
        "",
        f"**Direction:** {record['direction']}",
        f"**Status:** {record['status']}",
        f"**Verdict:** {record['verdict']}",
        f"**Confidence:** {record['confidence']}",
        f"**Backfill level:** {level}",
        "",
        "## Question",
        "",
        question,
        "",
        "## Evidence included",
        "",
        f"- Source scripts: {len(source_files)}",
        f"- Metrics refs: {len(metrics_refs)}",
        f"- Raw data refs: {len(raw_data_refs)}",
        f"- Scorer refs: {len(scorer_refs)}",
        f"- State refs: {len(state_refs)}",
        "",
        "## Evidence NOT included at this level",
        "",
    ]
    all_levels = schema.BACKFILL_LEVEL_NAMES
    excluded = [lvl for lvl in all_levels if schema.backfill_level_index(lvl) > requested_idx]
    if excluded:
        for lvl in excluded:
            readme_lines.append(f"- {lvl} (not exported at this level)")
    else:
        readme_lines.append("- (all levels included)")
    readme_lines += [
        "",
        "## Reproduction",
        "",
        "Raw data is referenced in place, not copied.",
        "Re-runs are new reproduction runs, not replays of the original data.",
        "",
    ]
    if reproduction_command:
        readme_lines += [
            f"    {reproduction_command}",
            "",
        ]
    else:
        readme_lines += [
            "No committed script found for this experiment.",
            "",
        ]
    readme_lines.append(f"Repo commit: {manifest_dict['repo_commit']}")
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    # metrics.json (only if level >= metrics_bundle and has_metrics)
    if requested_idx >= schema.backfill_level_index("metrics_bundle") and record["has_metrics"]:
        # Try to copy a single small metrics json verbatim; else write refs + note
        if len(metrics_refs) == 1:
            src_path = repo_root_path / metrics_refs[0]
            try:
                import json as _json
                raw_metrics = _json.loads(src_path.read_text(encoding="utf-8"))
                # Copy verbatim if it's a small dict/list (not a huge file)
                if src_path.stat().st_size < 1_000_000:
                    schema.write_json(raw_metrics, out_dir / "metrics.json")
                else:
                    schema.write_json(
                        {"metrics_refs": metrics_refs, "note": "see referenced files"},
                        out_dir / "metrics.json",
                    )
            except Exception:
                schema.write_json(
                    {"metrics_refs": metrics_refs, "note": "see referenced files"},
                    out_dir / "metrics.json",
                )
        else:
            schema.write_json(
                {"metrics_refs": metrics_refs, "note": "see referenced files"},
                out_dir / "metrics.json",
            )

    # reproduction.sh
    repro_sh_path = out_dir / "reproduction.sh"
    if record["has_script"] and source_files:
        repro_content = (
            "#!/usr/bin/env bash\n"
            "# Reproduction script — re-run is a NEW reproduction, not a replay.\n"
            f"# Original script: {source_files[0]}\n"
            f"cd \"$(git rev-parse --show-toplevel)\"\n"
            f"{reproduction_command}\n"
        )
    else:
        repro_content = (
            "#!/usr/bin/env bash\n"
            "# No committed script was found for this experiment.\n"
            "# Consult EXPERIMENTS.md and source_files in manifest.json.\n"
        )
    repro_sh_path.write_text(repro_content, encoding="utf-8")
    # chmod +x (read/execute, no write for group/other)
    current = repro_sh_path.stat().st_mode
    repro_sh_path.chmod(current | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    # scorer_card.json (if has_scorer)
    if record["has_scorer"]:
        scorer_card = schema.ScorerCard.from_file(
            "eval/affect_score.py",
            scorer_id="affect-score",
            scorer_version="affect-score-1e",
            repo=repo_root_path,
            metrics=["mean_last"],
            required_controls=["constant-response"],
            pass_conditions=["mean_last>1/3", "improvement>=0.10", "genuine_fraction>=0.5"],
            limitations=[
                "symbolic codes not language",
                "long session load-bearing",
            ],
        )
        schema.write_json(scorer_card.to_dict(), out_dir / "scorer_card.json")

    # ── 6. Validate ──────────────────────────────────────────────────────────
    validate.validate_bundle(out_dir)

    return manifest_dict
