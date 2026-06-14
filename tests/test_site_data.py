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


def _journey_inline_style():
    """Return the Journey page's inline style block."""
    text = _read("journey.html")
    m = re.search(r"<style>\s*(.*?)\s*</style>", text, re.DOTALL)
    assert m, "journey.html inline <style> block not found"
    return m.group(1)


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


def _entry_plain_fields():
    """Return list of (n, plain_text) for every AM_EXPERIMENTS entry.

    plain is the layman's "in plain terms" — what we're really testing, no jargon —
    rendered above the technical setup in the journey. Required on every entry.
    """
    results = []
    for n, blk in _split_entries():
        m = re.search(r'plain\s*:\s*"((?:[^"\\]|\\.)*)"', blk, re.DOTALL)
        results.append((n, m.group(1) if m else None))
    return results


def test_all_entries_have_plain():
    """Every AM_EXPERIMENTS entry has a non-empty, substantive `plain` field.

    Guards the convention (loop/PROTOCOL.md step 5/6): every experiment carries a
    plain-language "what are we really testing" line alongside the in-depth setup.
    """
    missing = []
    for n, plain in _entry_plain_fields():
        if plain is None:
            missing.append(f"Exp {n}: no `plain` field")
        elif len(plain.strip()) < 20:
            missing.append(f"Exp {n}: `plain` too short ({len(plain.strip())} chars): {plain!r}")
    assert not missing, "Entries missing a plain-language field:\n" + "\n".join(missing)


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
    """Every entry in AM_EXPERIMENTS has a trace block with script and output paths.
    rerun is optional: absent/None is fine for fresh experiments not yet re-verified;
    when present it must also exist on disk (enforced by test_trace_paths_exist_on_disk).
    """
    traces = _trace_paths()
    for n, script, output, rerun in traces:
        assert script is not None, f"Exp {n} missing trace.script"
        assert output is not None, f"Exp {n} missing trace.output"
        # rerun is intentionally optional — omit it until an independent re-run is done


def test_trace_paths_exist_on_disk():
    """Every trace script/output path exists on disk in the repo.
    rerun is optional: when absent/None it is fine; when present it must exist on disk.
    """
    traces = _trace_paths()
    missing = []
    for n, script, output, rerun in traces:
        for label, path in [("script", script), ("output", output)]:
            if path is None:
                missing.append(f"Exp {n}: trace.{label} is None")
                continue
            full = ROOT / path
            if not full.exists():
                missing.append(f"Exp {n}: trace.{label} = {path!r} not found at {full}")
        # rerun is optional: only check if present
        if rerun is not None:
            full = ROOT / rerun
            if not full.exists():
                missing.append(f"Exp {n}: trace.rerun = {rerun!r} not found at {full}")
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


def test_metric_units_are_captions_not_sentences():
    """Every metric unit string is a short caption (<= 80 chars).

    Guard for the Exp 151 layout bug: the journey card header puts the metric in
    a nowrap-ish auto grid column; a sentence-length unit ("colors mapped of
    sixteen at first save (loc 0.12 cells; ...)") ballooned that column and
    crushed the title into a one-word-per-line strip. The CSS now clamps the
    metric box, but the unit is a caption by design — detail belongs in the
    entry's `one`/`result` text, not the unit.
    """
    offenders = []
    for n, blk in _split_entries():
        metric_m = re.search(r"metric\s*:\s*\{([^}]*)\}", blk)
        if not metric_m:
            continue
        unit_m = re.search(r'unit\s*:\s*"((?:[^"\\]|\\.)*)"', metric_m.group(1))
        if unit_m and len(unit_m.group(1)) > 80:
            offenders.append(f"Exp {n}: unit is {len(unit_m.group(1))} chars: {unit_m.group(1)!r}")
    assert not offenders, (
        "Metric units must stay caption-length (<= 80 chars):\n" + "\n".join(offenders)
    )


def test_journey_long_entries_have_reader_width_and_mobile_stack():
    """The Journey timeline must stay readable for long frontier entries.

    Regression guard for Exp 206-style cards: desktop should give text a real
    reader width, while narrow screens should stack the header and hide the
    decorative rail so content does not compress into a thin column.
    """
    css = re.sub(r"\s+", "", _journey_inline_style())
    assert ".tl-inner{max-width:min(1180px,calc(100vw-56px))" in css
    assert "@media(max-width:640px)" in css
    assert ".card-head{grid-template-columns:1fr" in css
    assert ".timeline{padding-top:16px" in css
    assert ".exp,.beat,.actdiv{padding-left:0" in css
    assert ".exp.node,.beat.node,.actdiv.node{display:none" in css


def _surprise_segments_points():
    """Return list of (exp, float) pairs from AM_SURPRISE_SEGMENTS points arrays."""
    text = _js_text()
    # Find the AM_SURPRISE_SEGMENTS array body
    m = re.search(r"window\.AM_SURPRISE_SEGMENTS\s*=\s*\[", text)
    assert m, "AM_SURPRISE_SEGMENTS not found in experiments-data.js"
    body_start = m.end()
    rest = text[body_start:]
    # Find the closing ]; at start of a line
    end_m = re.search(r"^\s*\]\s*;", rest, re.MULTILINE)
    assert end_m, "AM_SURPRISE_SEGMENTS closing ]; not found"
    body = rest[:end_m.start()]

    results = []
    # Find each segment object: { exp:N, ..., points:[v, v, ...], ... }
    for seg_m in re.finditer(r"\{\s*exp\s*:\s*(\d+)", body):
        exp_n = int(seg_m.group(1))
        # Find the points array within the following text
        seg_rest = body[seg_m.start():]
        # Find points:[...] — values are floats
        pts_m = re.search(r"points\s*:\s*\[([^\]]*)\]", seg_rest)
        if pts_m:
            for val_s in pts_m.group(1).split(","):
                val_s = val_s.strip()
                if val_s:
                    results.append((exp_n, float(val_s)))
    return results


def test_surprise_segments_points_in_md():
    """Every number in every AM_SURPRISE_SEGMENTS points array appears in EXPERIMENTS.md."""
    md = _read("EXPERIMENTS.md")
    for exp_n, v in _surprise_segments_points():
        s = str(v)
        alternatives = {s, f"{v:.2f}"}
        assert any(a in md for a in alternatives), (
            f"AM_SURPRISE_SEGMENTS exp:{exp_n} value {v} ({alternatives}) not found in EXPERIMENTS.md"
        )


# ---------------------------------------------------------------------------
# Caveat fields: T11 (rigor-fairness-upgrade) — honest caveats on public cards.
# Every AM_EXPERIMENTS entry must carry a `caveat` key; breakthrough/positive
# entries that have a non-empty caveat in EXPERIMENTS.md must surface it.
# ---------------------------------------------------------------------------

def _unescape_js(s: str) -> str:
    """Unescape a JS string literal value (handle \\", \\\\, \\n etc.)."""
    return s.replace('\\"', '"').replace("\\\\", "\\").replace("\\n", "\n")


def _caveat_fields():
    """Return list of (n, caveat_text_or_None) from AM_EXPERIMENTS entries.

    The returned text is unescaped from JS (\\\" → ") so it can be compared
    directly against the raw text from EXPERIMENTS.md.
    """
    results = []
    for n, blk in _split_entries():
        m = re.search(r'caveat\s*:\s*"((?:[^"\\]|\\.)*)"', blk)
        results.append((n, _unescape_js(m.group(1)) if m else None))
    return results


def test_all_entries_have_caveat_field():
    """Every AM_EXPERIMENTS entry has a `caveat` field (may be empty string).

    Guards: the public timeline never shows an experiment card without the
    caveat key present — front-end code can rely on it always existing.
    """
    missing = [n for n, cav in _caveat_fields() if cav is None]
    assert not missing, (
        f"Entries missing a caveat field: {missing}. "
        "Run `uv run --python .venv python -m active_loop.site_data` to regenerate."
    )


def test_breakthrough_positive_entries_have_non_empty_caveat_when_md_has_one():
    """BREAKTHROUGH/POSITIVE entries whose EXPERIMENTS.md section contains a
    caveat line must carry a non-empty caveat field in experiments-data.js.

    This is the core fairness invariant: the public timeline never shows a
    positive card while silently omitting the recorded honest caveat.
    """
    from active_loop.site_data import parse_caveats_from_md  # type: ignore

    md_caveats = parse_caveats_from_md()
    js_entries = {n: cav for n, cav in _caveat_fields()}

    offenders = []
    for n, kind in _all_entry_kinds():
        if kind not in ("breakthrough", "positive"):
            continue
        md_caveat = md_caveats.get(n, "")
        js_caveat = js_entries.get(n, None) or ""
        if md_caveat and not js_caveat:
            offenders.append(
                f"Exp {n} ({kind}): EXPERIMENTS.md has caveat but card has none"
            )
    assert not offenders, (
        "Positive/breakthrough cards missing their honest caveats:\n"
        + "\n".join(offenders)
    )


def test_caveat_field_is_first_sentence_from_md():
    """Spot-check: Exp 20's caveat in the JS matches the first sentence
    extracted by parse_caveats_from_md, confirming the generator and the
    committed file agree.
    """
    from active_loop.site_data import parse_caveats_from_md  # type: ignore

    md_caveats = parse_caveats_from_md()
    js_entries = {n: cav for n, cav in _caveat_fields()}

    exp20_md = md_caveats.get(20, "")
    exp20_js = js_entries.get(20, "")
    assert exp20_md, "Exp 20 must have a caveat in EXPERIMENTS.md"
    assert exp20_js, "Exp 20 caveat must be non-empty in experiments-data.js"
    assert exp20_js == exp20_md, (
        f"Exp 20 caveat mismatch.\n"
        f"  MD (generator):  {exp20_md!r}\n"
        f"  JS (committed):  {exp20_js!r}\n"
        "Run `uv run --python .venv python -m active_loop.site_data` to resync."
    )


def test_caveat_js_is_in_sync_with_generator():
    """Staleness guard: experiments-data.js caveat fields must match what the
    generator would produce from EXPERIMENTS.md right now.

    If this fails, run:
      uv run --python .venv python -m active_loop.site_data
    and commit the result.
    """
    from active_loop.site_data import parse_caveats_from_md  # type: ignore

    md_caveats = parse_caveats_from_md()
    js_entries = {n: cav for n, cav in _caveat_fields()}

    drift = []
    for n, md_cav in md_caveats.items():
        js_cav = js_entries.get(n, None)
        if js_cav is None:
            drift.append(f"Exp {n}: no caveat field in JS")
        elif js_cav != md_cav:
            drift.append(
                f"Exp {n}: JS={js_cav!r:.60} vs MD={md_cav!r:.60}"
            )
    assert not drift, (
        "experiments-data.js caveat fields are out of sync with EXPERIMENTS.md.\n"
        "Regenerate: uv run --python .venv python -m active_loop.site_data\n"
        + "\n".join(drift[:10])
    )


# ---------------------------------------------------------------------------
# Asset cache-busting: all ?v=N versions must agree across the three pages.
# Guard for the recurring stale-cache failure (a shared asset like am.css
# changes but a page is left referencing the old ?v= → returning visitors get
# mixed old-JS/new-HTML and the page breaks). See loop/META.md.
# ---------------------------------------------------------------------------

_PAGES = ["index.html", "journey.html", "open_problem.html"]


def test_asset_cache_versions_are_consistent():
    versions = {}
    for page in _PAGES:
        found = re.findall(r"\?v=(\d+)", _read(page))
        for v in found:
            versions.setdefault(int(v), []).append(page)
    assert len(versions) <= 1, (
        "Mixed ?v= asset versions across pages (bump them together): "
        + "; ".join(f"v={v}: {sorted(set(p))}" for v, p in sorted(versions.items()))
    )


# ---------------------------------------------------------------------------
# Experiment-number uniqueness: cross-branch collision guard (Exp 155
# incident, 2026-06-11). Two agents working in parallel — the local loop and
# a cloud branch (claude/n3-bounded-map-design-7705gt) — both claimed
# "Exp 155" for different experiments. Numbers are stable citations
# (LESSONS.md ground rule); a merge that lands a duplicate must trip CI here
# instead of silently double-numbering the log.
# ---------------------------------------------------------------------------


def test_experiment_numbers_are_unique():
    text = _read("EXPERIMENTS.md")
    nums = [int(m) for m in re.findall(r"^## Exp (\d+) ", text, re.MULTILINE)]
    dupes = sorted({n for n in nums if nums.count(n) > 1})
    assert not dupes, (
        f"duplicate experiment numbers in EXPERIMENTS.md: {dupes} — "
        "a parallel branch's expNN must be renumbered to the next free "
        "number at merge time (script, outputs, entry, and site data all move together)"
    )


# ---------------------------------------------------------------------------
# Shell-escape leak guard (Exp 191-era incident, 2026-06-12). Curated entries
# written to experiments-data.js through a shell pipeline leaked POSIX
# single-quote escapes ("'\''") into the file content; in a JS double-quoted
# string the embedded \' collapses to ' at parse time, so the public cards
# rendered possessives as triple apostrophes ("CALM-H6000'''s"). The site JS
# uses double-quoted strings throughout, so a backslash-apostrophe sequence
# is NEVER legitimate — its presence means a writer escaped for the wrong
# layer. Write these files from Python (or the site_data generators), not
# via single-quoted shell programs.
# ---------------------------------------------------------------------------


def test_site_js_has_no_shell_quote_escape_leaks():
    for fname in ("experiments-data.js", "lab-status.js"):
        text = _read(fname)
        leaks = [
            f"{fname}:{i}: …{line.strip()[:90]}"
            for i, line in enumerate(text.splitlines(), 1)
            if "\\'" in line
        ]
        assert not leaks, (
            "backslash-apostrophe (shell-escape leak) in site JS — fix the "
            "content AND the writer that produced it:\n" + "\n".join(leaks)
        )
