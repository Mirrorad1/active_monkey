"""Tests for the lab_status() builder in active_loop/site_data.py.

Three concerns:
  1. Staleness guard — regenerating in-memory must produce bytes identical to
     the committed site/data/lab-status.js (same pattern as test_directions_index.py).
  2. Schema sanity — the generated structure contains the right values.
  3. Direction states — every direction in the output has a known/valid state.

Run:  uv run --python .venv pytest tests/test_lab_status.py -q
Remediation if staleness fails:
  uv run --python .venv python -m active_loop.site_data --lab-status
  then commit site/data/lab-status.js.
"""
from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).parent.parent
LAB_STATUS = ROOT / "site" / "data" / "lab-status.js"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_committed() -> str:
    return LAB_STATUS.read_text(encoding="utf-8")


def _regenerate_in_memory() -> str:
    """Call lab_status() without touching the filesystem."""
    # Import late so test collection doesn't fail if the module has a syntax error.
    from active_loop.site_data import lab_status  # noqa: PLC0415
    return lab_status()


def _max_exp_n_from_md() -> int:
    """Return the highest ## Exp N header number from EXPERIMENTS.md."""
    text = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    ns = [int(m) for m in re.findall(r"^## Exp (\d+) ", text, re.MULTILINE)]
    return max(ns)


def _total_exp_count_from_md() -> int:
    """Count ## Exp N headers in EXPERIMENTS.md."""
    text = (ROOT / "EXPERIMENTS.md").read_text(encoding="utf-8")
    return len(re.findall(r"^## Exp \d+ ", text, re.MULTILINE))


# ---------------------------------------------------------------------------
# Known valid state values (must stay in sync with _STATE_ORDER in site_data)
# ---------------------------------------------------------------------------

_KNOWN_STATES = {
    "active",
    "halted",
    "flagship-candidate",
    "exploratory",
    "closed-positive",
    "closed-negative",
    "published",
    "TBD-human",
}


def _state_base(state: str) -> str:
    """Normalize 'active (prereq build)' -> 'active', etc."""
    return state.split("(")[0].strip().split()[0] if state else state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_lab_status_not_stale():
    """In-memory regeneration must be byte-identical to the committed lab-status.js.

    If this fails: uv run --python .venv python -m active_loop.site_data --lab-status
    and commit the updated site/data/lab-status.js.
    """
    committed = _read_committed()
    regenerated = _regenerate_in_memory()
    assert regenerated == committed, (
        "site/data/lab-status.js is stale — regenerate with:\n"
        "  uv run --python .venv python -m active_loop.site_data --lab-status\n"
        "and commit site/data/lab-status.js.\n"
        f"Committed length: {len(committed)}, regenerated length: {len(regenerated)}"
    )


def test_latest_exp_n_matches_experiments_md():
    """latest_exp.n must equal the highest ## Exp N in EXPERIMENTS.md."""
    from active_loop.site_data import parse_experiments  # noqa: PLC0415
    exps = parse_experiments()
    max_n = max(e["n"] for e in exps)
    expected = _max_exp_n_from_md()
    assert max_n == expected, (
        f"latest_exp.n={max_n} does not match max Exp in EXPERIMENTS.md={expected}"
    )


def test_tally_total_equals_experiment_count():
    """tally.total must equal the number of ## Exp N entries in EXPERIMENTS.md."""
    from active_loop.site_data import parse_experiments  # noqa: PLC0415
    exps = parse_experiments()
    total = len(exps)
    expected = _total_exp_count_from_md()
    assert total == expected, (
        f"tally.total={total} does not match experiment count in EXPERIMENTS.md={expected}"
    )


def test_tally_components_sum_to_at_most_total():
    """positive + negative + mixed <= total (un-tagged early exps account for the rest)."""
    from active_loop.site_data import parse_experiments  # noqa: PLC0415
    exps = parse_experiments()
    total = len(exps)
    positive = sum(1 for e in exps if e["tag"] == "POSITIVE")
    negative = sum(1 for e in exps if e["tag"] == "NEGATIVE")
    mixed = sum(1 for e in exps if e["tag"] == "MIXED")
    assert positive + negative + mixed <= total, (
        f"Tag counts ({positive}+{negative}+{mixed}) exceed total ({total})"
    )
    assert positive >= 0 and negative >= 0 and mixed >= 0


def test_every_direction_has_known_state():
    """Every direction entry must have a recognized state value."""
    from active_loop.site_data import parse_directions  # noqa: PLC0415
    dirs = parse_directions()
    assert dirs, "No directions found — check loop/directions/*.md"
    unknown = []
    for d in dirs:
        base = _state_base(d["state"])
        if base not in _KNOWN_STATES:
            unknown.append(f'{d["name"]}: {d["state"]!r}')
    assert not unknown, (
        "Directions with unknown state values:\n" + "\n".join(unknown)
    )


def test_lab_status_js_is_valid_structure():
    """Spot-check the generated JS for required structural elements."""
    content = _regenerate_in_memory()
    assert content.startswith("/* GENERATED"), "Missing generated header comment"
    assert "window.AM_LAB_STATUS = {" in content
    assert "latest_exp:" in content
    assert "tally:" in content
    assert "directions:" in content
    assert "flagship:" in content
    # Flagship should reference the correct page
    assert "worldview-too-small.md" in content


def test_lab_status_latest_exp_tag_is_valid():
    """The latest experiment's tag must be POSITIVE, NEGATIVE, or MIXED."""
    from active_loop.site_data import parse_experiments  # noqa: PLC0415
    exps = parse_experiments()
    latest = max(exps, key=lambda e: e["n"])
    assert latest["tag"] in ("POSITIVE", "NEGATIVE", "MIXED"), (
        f"latest_exp (Exp {latest['n']}) has unrecognized tag: {latest['tag']!r}"
    )


def test_directions_sorted_active_before_exploratory():
    """Active directions must appear before exploratory ones in the output."""
    from active_loop.site_data import parse_directions  # noqa: PLC0415
    dirs = parse_directions()
    names = [d["name"] for d in dirs]
    states = [d["state"] for d in dirs]
    # Find last 'active' and first 'exploratory'
    active_indices = [i for i, s in enumerate(states) if _state_base(s) == "active"]
    exploratory_indices = [i for i, s in enumerate(states) if _state_base(s) == "exploratory"]
    if active_indices and exploratory_indices:
        assert max(active_indices) < min(exploratory_indices), (
            f"Active direction appears after exploratory direction.\n"
            f"Directions: {list(zip(names, states))}"
        )
