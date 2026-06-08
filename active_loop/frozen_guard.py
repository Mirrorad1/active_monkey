"""Enforce the trust boundary: reject proposals that touch FROZEN paths."""
from __future__ import annotations

from pathlib import Path


def load_frozen(repo: Path | str) -> list[str]:
    """Read the FROZEN manifest: one path prefix per non-empty, non-comment line."""
    text = (Path(repo) / "FROZEN").read_text()
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def is_frozen_touched(changed_paths: list[str], repo: Path | str) -> bool:
    """True if any changed path falls under a FROZEN prefix (dir prefix or exact file)."""
    frozen = load_frozen(repo)
    for path in changed_paths:
        for prefix in frozen:
            if prefix.endswith("/"):
                if path == prefix.rstrip("/") or path.startswith(prefix):
                    return True
            elif path == prefix:
                return True
    return False
