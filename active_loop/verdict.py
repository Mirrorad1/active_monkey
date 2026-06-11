"""Machine-readable per-experiment verdict helper (T6, rigor-fairness-upgrade spec).

Call ``write_verdict`` at the end of a future experiment script to emit a
structured JSON record alongside the experiment's row outputs.  Do NOT
retrofit past experiments — their committed outputs are frozen history.

JSON schema
-----------
::

    {
        "schema_version": 1,
        "experiment":     <str>   # e.g. "exp154"
        "verdict":        <str>   # one of: "POSITIVE", "NEGATIVE", "MIXED"
        "halted":         <bool>  # True when a predeclared halting rule fired
        "arms": {
            "<name>": {
                "pass":   <bool>
                "reason": <str>   # brief human-readable justification
            },
            ...
        },
        "notes":          <str>   # free-form; empty string if omitted
    }

The output filename follows the convention ``exp<N>_verdict.json`` (see
``write_verdict`` for how ``path`` resolves the output location).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_VALID_VERDICTS = {"POSITIVE", "NEGATIVE", "MIXED"}


def _validate_arms(arms: Any) -> None:
    """Raise ValueError if *arms* is not a well-formed arm mapping."""
    if not isinstance(arms, dict):
        raise ValueError(f"arms must be a dict, got {type(arms).__name__}")
    for name, entry in arms.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"arms[{name!r}] must be a dict with 'pass' and 'reason' keys"
            )
        if "pass" not in entry or "reason" not in entry:
            raise ValueError(
                f"arms[{name!r}] is missing required keys ('pass', 'reason'); "
                f"got {set(entry)}"
            )
        if not isinstance(entry["pass"], bool):
            raise ValueError(
                f"arms[{name!r}]['pass'] must be bool, got "
                f"{type(entry['pass']).__name__}"
            )
        if not isinstance(entry["reason"], str):
            raise ValueError(
                f"arms[{name!r}]['reason'] must be str, got "
                f"{type(entry['reason']).__name__}"
            )


def write_verdict(
    path: str | Path,
    experiment: str,
    arms: dict[str, dict[str, Any]],
    verdict: str,
    halted: bool,
    notes: str = "",
) -> Path:
    """Write a machine-readable verdict JSON for *experiment* to *path*.

    Parameters
    ----------
    path:
        Destination file path (including filename).  Typically
        ``experiments/outputs/exp<N>_verdict.json`` relative to the repo root.
        Parent directories must already exist.
    experiment:
        Short experiment identifier, e.g. ``"exp154"``.
    arms:
        Mapping of arm name → ``{"pass": bool, "reason": str}``.  At least one
        entry required.
    verdict:
        Overall verdict string.  Must be one of ``"POSITIVE"``, ``"NEGATIVE"``,
        or ``"MIXED"``.
    halted:
        Whether a predeclared halting rule fired for this experiment.
    notes:
        Optional free-form text (defaults to empty string).

    Returns
    -------
    Path
        Absolute path of the written file.

    Raises
    ------
    ValueError
        If *verdict* is not one of the three allowed strings, or if *arms* is
        malformed (missing keys, wrong types, or not a dict).
    """
    if verdict not in _VALID_VERDICTS:
        raise ValueError(
            f"verdict must be one of {sorted(_VALID_VERDICTS)!r}, got {verdict!r}"
        )
    _validate_arms(arms)
    if not arms:
        raise ValueError("arms must contain at least one entry")

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment,
        "verdict": verdict,
        "halted": halted,
        "arms": arms,
        "notes": notes,
    }

    out = Path(path)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return out.resolve()
