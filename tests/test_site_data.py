"""Fast, stdlib-only tests that assert the curated site snapshot
(experiments-data.js) is consistent with the ground-truth log
(EXPERIMENTS.md).  No network, no jax, no pymdp imports.
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).parent.parent


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Parse EXPERIMENTS.md
# ---------------------------------------------------------------------------

def _exp_md_numbers():
    """Return sorted list of experiment numbers from EXPERIMENTS.md."""
    text = _read("EXPERIMENTS.md")
    return sorted(int(m) for m in re.findall(r"^## Exp (\d+) ", text, re.MULTILINE))


# ---------------------------------------------------------------------------
# Parse experiments-data.js  (regex over plain text — no JS eval)
# ---------------------------------------------------------------------------

def _js_text():
    return _read("experiments-data.js")


def _am_experiments_body():
    """Return the text of the AM_EXPERIMENTS array body.

    Extracts the region between 'window.AM_EXPERIMENTS = [' and the closing
    '];' by finding the first top-level '];' after the opening.  This avoids
    false matches on nested brackets inside string literals.
    """
    text = _js_text()
    start_m = re.search(r"window\.AM_EXPERIMENTS\s*=\s*\[", text)
    assert start_m, "AM_EXPERIMENTS not found"
    body_start = start_m.end()
    # Walk forward and find the ]; that closes the top-level array.
    # We do this by looking for the sentinel pattern on a line of its own
    # (as produced by the formatter: just "];" at the start of a line).
    rest = text[body_start:]
    # Find "];" at the start of a line (possibly with whitespace before it)
    end_m = re.search(r"^\s*\]\s*;", rest, re.MULTILINE)
    assert end_m, "AM_EXPERIMENTS closing ]; not found"
    return rest[: end_m.start()]


def _exp_js_numbers():
    """Return sorted list of { n:<int>, entries in AM_EXPERIMENTS."""
    return sorted(int(m) for m in re.findall(r"\{\s*n\s*:\s*(\d+)\s*,", _am_experiments_body()))


def _tally_fields():
    """Return dict of AM_TALLY fields: total, win, wall, partial."""
    text = _js_text()
    m = re.search(
        r"AM_TALLY\s*=\s*\{([^}]*)\}",
        text,
    )
    assert m, "AM_TALLY not found in experiments-data.js"
    body = m.group(1)
    fields = {}
    for key in ("total", "win", "wall", "partial"):
        km = re.search(rf"\b{key}\s*:\s*(\d+)", body)
        assert km, f"AM_TALLY.{key} not found"
        fields[key] = int(km.group(1))
    return fields


def _surprise_values():
    """Return list of floats from AM_SURPRISE array."""
    text = _js_text()
    m = re.search(r"AM_SURPRISE\s*=\s*\[([^\]]*)\]", text)
    assert m, "AM_SURPRISE not found"
    return [float(v.strip()) for v in m.group(1).split(",") if v.strip()]


def _metric_from_to_values():
    """Return all numeric values that appear inside metric:{...} blocks
    as from: or to: fields."""
    text = _js_text()
    # find metric:{...} blocks (may span a line or two)
    metric_blocks = re.findall(r"metric\s*:\s*\{([^}]*)\}", text)
    values = []
    for blk in metric_blocks:
        for m in re.findall(r"\b(?:from|to)\s*:\s*([\d.]+)", blk):
            values.append(m)
    return values


def _kind_counts_in_js():
    """Count win/wall/partial kind values in AM_EXPERIMENTS entries."""
    body = _am_experiments_body()
    return {
        "win":     len(re.findall(r'kind\s*:\s*"win"',     body)),
        "wall":    len(re.findall(r'kind\s*:\s*"wall"',    body)),
        "partial": len(re.findall(r'kind\s*:\s*"partial"', body)),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_md_no_gaps():
    """Experiment numbers in EXPERIMENTS.md are consecutive starting at 1."""
    nums = _exp_md_numbers()
    assert nums, "No experiments found in EXPERIMENTS.md"
    assert nums == list(range(1, len(nums) + 1)), (
        f"Gaps in EXPERIMENTS.md numbering: {nums}"
    )


def test_js_count_matches_md():
    """Number of curated entries in AM_EXPERIMENTS == number in EXPERIMENTS.md."""
    md_count = len(_exp_md_numbers())
    js_count = len(_exp_js_numbers())
    assert js_count == md_count, (
        f"AM_EXPERIMENTS has {js_count} entries but EXPERIMENTS.md has {md_count}"
    )


def test_tally_total_matches_count():
    """AM_TALLY.total equals the number of experiments."""
    expected = len(_exp_md_numbers())
    tally = _tally_fields()
    assert tally["total"] == expected, (
        f"AM_TALLY.total={tally['total']} but experiment count={expected}"
    )


def test_tally_kinds_sum_to_total():
    """AM_TALLY win + wall + partial == total."""
    t = _tally_fields()
    assert t["win"] + t["wall"] + t["partial"] == t["total"], (
        f"AM_TALLY win({t['win']}) + wall({t['wall']}) + partial({t['partial']}) "
        f"!= total({t['total']})"
    )


def test_tally_kinds_match_entries():
    """AM_TALLY win/wall/partial match counted kind values in AM_EXPERIMENTS."""
    tally = _tally_fields()
    counts = _kind_counts_in_js()
    for k in ("win", "wall", "partial"):
        assert tally[k] == counts[k], (
            f"AM_TALLY.{k}={tally[k]} but {counts[k]} entries have kind:\"{k}\""
        )


def test_surprise_values_in_md():
    """Every value in AM_SURPRISE appears as a substring in EXPERIMENTS.md."""
    md = _read("EXPERIMENTS.md")
    for v in _surprise_values():
        # represent as string the same way it appears in the source
        s = str(v)
        # also try without trailing zero: 4.0 -> "4.0" and "4.00"
        alternatives = {s, f"{v:.2f}"}
        assert any(a in md for a in alternatives), (
            f"AM_SURPRISE value {v} ({alternatives}) not found in EXPERIMENTS.md"
        )


def test_metric_values_in_md():
    """Every from:/to: value inside metric:{} blocks appears in EXPERIMENTS.md."""
    md = _read("EXPERIMENTS.md")
    for raw in _metric_from_to_values():
        v = float(raw)
        alternatives = {raw, str(v), f"{v:.2f}"}
        assert any(a in md for a in alternatives), (
            f"metric value {raw} not found in EXPERIMENTS.md"
        )
