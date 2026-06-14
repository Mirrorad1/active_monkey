"""
ecology.evolvability.io — filesystem I/O helpers for preflight runs.

Stdlib only; no engine imports.  The single hard requirement is NO OVERWRITE:
new_run_dir() raises FileExistsError rather than silently clobbering an
existing run directory.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
from typing import Optional


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git_commit(repo_root: "str | pathlib.Path | None" = None) -> Optional[str]:
    """Return the current HEAD commit hash, or None on any failure."""
    try:
        cwd = str(repo_root) if repo_root is not None else None
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def config_hash(d: dict) -> str:
    """SHA-256 hex digest of a canonical JSON representation of d."""
    raw = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Directory management
# ---------------------------------------------------------------------------

def _timestamp() -> str:
    """Filesystem-safe UTC timestamp string like '20260614T120000Z'."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def new_run_dir(
    base: "str | pathlib.Path",
    slug: str,
    run_id: Optional[str] = None,
) -> pathlib.Path:
    """Create and return Path(base)/slug/(run_id or timestamp).

    Also creates a `raw/` subdirectory inside the run directory.

    Raises FileExistsError if the target already exists — never overwrites.
    """
    target = pathlib.Path(base) / slug / (run_id if run_id is not None else _timestamp())
    if target.exists():
        raise FileExistsError(f"Run directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    (target / "raw").mkdir()
    return target


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def write_json(path: "str | pathlib.Path", obj) -> None:
    """Write obj to path as pretty-printed JSON (indent=2, default=str)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def write_jsonl(path: "str | pathlib.Path", rows: list) -> None:
    """Write a list of dicts to path, one compact JSON object per line."""
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def append_jsonl(path: "str | pathlib.Path", row: dict) -> None:
    """Append one JSON object line to path."""
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: "str | pathlib.Path") -> list:
    """Parse a JSONL file and return a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def write_csv(
    path: "str | pathlib.Path",
    rows: list,
    fieldnames: Optional[list] = None,
) -> None:
    """Write rows (list of dicts) to a CSV file.

    - If rows is empty and fieldnames is given, writes just the header.
    - If rows is empty and no fieldnames, creates an empty file.
    - fieldnames defaults to sorted union of all row keys.
    """
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            if fieldnames:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
        return

    if fieldnames is None:
        all_keys: set = set()
        for row in rows:
            all_keys.update(row.keys())
        fieldnames = sorted(all_keys)

    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Text helper
# ---------------------------------------------------------------------------

def write_text(path: "str | pathlib.Path", text: str) -> None:
    """Write text to path (UTF-8)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
