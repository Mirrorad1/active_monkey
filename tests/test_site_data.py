"""Fast, stdlib-only tests that assert the curated site snapshot
(experiments-data.js) is consistent with the ground-truth log
(EXPERIMENTS.md).  No network, no jax, no pymdp imports.
"""
import re
import pathlib

ROOT = pathlib.Path(__file__).parent.parent

VALID_KINDS = {"breakthrough", "positive", "wall", "partial"}


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
    """Return dict of AM_TALLY fields: total, breakthrough, positive, wall, partial."""
    text = _js_text()
    m = re.search(
        r"AM_TALLY\s*=\s*\{([^}]*)\}",
        text,
    )
    assert m, "AM_TALLY not found in experiments-data.js"
    body = m.group(1)
    fields = {}
    for key in ("total", "breakthrough", "positive", "wall", "partial"):
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
    """Count breakthrough/positive/wall/partial kind values in AM_EXPERIMENTS entries."""
    body = _am_experiments_body()
    return {
        "breakthrough": len(re.findall(r'kind\s*:\s*"breakthrough"', body)),
        "positive":     len(re.findall(r'kind\s*:\s*"positive"',     body)),
        "wall":         len(re.findall(r'kind\s*:\s*"wall"',         body)),
        "partial":      len(re.findall(r'kind\s*:\s*"partial"',      body)),
    }


def _split_entries():
    """Split the AM_EXPERIMENTS body into per-entry blocks.

    Each entry starts with a line like '  { n:NN,' and ends just before
    the next such line (or the array close).  Returns list of (n, block_text).
    Handles nested objects (metric:{}, trace:{}) correctly by using line-based
    splits rather than brace-matching regex.
    """
    body = _am_experiments_body()
    # Each entry starts at a line matching the pattern `  { n:<digits>,`
    entry_start = re.compile(r"^\s*\{\s*n\s*:\s*(\d+)\s*,", re.MULTILINE)
    positions = [(m.start(), int(m.group(1))) for m in entry_start.finditer(body)]
    entries = []
    for i, (pos, n) in enumerate(positions):
        end = positions[i+1][0] if i+1 < len(positions) else len(body)
        entries.append((n, body[pos:end]))
    return entries


def _all_entry_kinds():
    """Return list of (n, kind) pairs from AM_EXPERIMENTS."""
    results = []
    for n, blk in _split_entries():
        m = re.search(r'kind\s*:\s*"([^"]+)"', blk)
        if m:
            results.append((n, m.group(1)))
    return results


def _breakthrough_stories():
    """Return list of (n, story_text) for entries with kind:breakthrough."""
    results = []
    for n, blk in _split_entries():
        km = re.search(r'kind\s*:\s*"([^"]+)"', blk)
        if not km or km.group(1) != "breakthrough":
            continue
        story_m = re.search(r'story\s*:\s*"((?:[^"\\]|\\.)*)"', blk, re.DOTALL)
        story = story_m.group(1) if story_m else ""
        results.append((n, story))
    return results


def _trace_paths():
    """Return list of (n, script, output, rerun) from trace:{...} blocks."""
    results = []
    for n, blk in _split_entries():
        trace_m = re.search(r'trace\s*:\s*\{([^}]*)\}', blk)
        if not trace_m:
            results.append((n, None, None, None))
            continue
        tb = trace_m.group(1)
        script_m  = re.search(r'script\s*:\s*"([^"]+)"', tb)
        output_m  = re.search(r'output\s*:\s*"([^"]+)"', tb)
        rerun_m   = re.search(r'rerun\s*:\s*"([^"]+)"', tb)
        results.append((
            n,
            script_m.group(1)  if script_m  else None,
            output_m.group(1)  if output_m  else None,
            rerun_m.group(1)   if rerun_m   else None,
        ))
    return results


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
    """AM_TALLY breakthrough + positive + wall + partial == total."""
    t = _tally_fields()
    kind_sum = t["breakthrough"] + t["positive"] + t["wall"] + t["partial"]
    assert kind_sum == t["total"], (
        f"AM_TALLY breakthrough({t['breakthrough']}) + positive({t['positive']}) "
        f"+ wall({t['wall']}) + partial({t['partial']}) != total({t['total']})"
    )


def test_tally_kinds_match_entries():
    """AM_TALLY breakthrough/positive/wall/partial match counted kind values in AM_EXPERIMENTS."""
    tally = _tally_fields()
    counts = _kind_counts_in_js()
    for k in ("breakthrough", "positive", "wall", "partial"):
        assert tally[k] == counts[k], (
            f"AM_TALLY.{k}={tally[k]} but {counts[k]} entries have kind:\"{k}\""
        )


def test_all_kinds_valid():
    """Every kind value in AM_EXPERIMENTS is one of the four valid kinds."""
    entries = _all_entry_kinds()
    invalid = [(n, k) for n, k in entries if k not in VALID_KINDS]
    assert not invalid, (
        f"Entries with invalid kind values: {invalid}. "
        f"Valid kinds are: {VALID_KINDS}"
    )


def test_breakthrough_entries_have_story():
    """Every kind:'breakthrough' entry has a non-empty story field (> 200 chars)."""
    stories = _breakthrough_stories()
    assert stories, "No breakthrough entries found — expected 5"
    for n, story in stories:
        assert story, f"Exp {n} is 'breakthrough' but has no story field"
        assert len(story) > 200, (
            f"Exp {n} story is too short ({len(story)} chars, expected >200): {story!r}"
        )


def test_all_entries_have_trace():
    """Every entry in AM_EXPERIMENTS has a trace block with script/output/rerun paths."""
    traces = _trace_paths()
    for n, script, output, rerun in traces:
        assert script is not None, f"Exp {n} missing trace.script"
        assert output is not None, f"Exp {n} missing trace.output"
        assert rerun  is not None, f"Exp {n} missing trace.rerun"


def test_trace_paths_exist_on_disk():
    """Every trace script/output/rerun path exists on disk in the repo."""
    traces = _trace_paths()
    missing = []
    for n, script, output, rerun in traces:
        for label, path in [("script", script), ("output", output), ("rerun", rerun)]:
            if path is None:
                missing.append(f"Exp {n}: trace.{label} is None")
                continue
            full = ROOT / path
            if not full.exists():
                missing.append(f"Exp {n}: trace.{label} = {path!r} not found at {full}")
    assert not missing, "Missing trace files:\n" + "\n".join(missing)


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
