"""site_data — caveat injection for experiments-data.js and lab-status.js generation.

Reads EXPERIMENTS.md, extracts the "Honest caveat" (or "- Caveat:") text for
each experiment, then injects a ``caveat`` field into every matching entry in
``experiments-data.js``.

Caveat rule (applied uniformly to every experiment):
  The first sentence of the caveat text from EXPERIMENTS.md, where a sentence
  boundary is detected as a period/!/?  followed by whitespace and an uppercase
  letter or '('.  If the caveat text contains no such boundary the entire text
  is used.  Experiments that have no caveat line in EXPERIMENTS.md receive an
  empty string ("") as their caveat field.

Usage (regenerate experiments-data.js in-place):
  uv run --python .venv python -m active_loop.site_data

Usage (generate lab-status.js):
  uv run --python .venv python -m active_loop.site_data --lab-status
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Shared STATUS-block parser (imported from tools/gen_directions_index.py
# to avoid duplication — this is the single source of truth for that logic).
# ---------------------------------------------------------------------------

def _get_directions_parser():
    """Load and return the gen_directions_index module (lazy, cached)."""
    spec = importlib.util.spec_from_file_location(
        "gen_directions_index",
        ROOT / "tools" / "gen_directions_index.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# EXPERIMENTS.md parser
# ---------------------------------------------------------------------------

def parse_caveats_from_md(md_path: pathlib.Path | None = None) -> dict[int, str]:
    """Return {exp_n: first_sentence_of_caveat} for every experiment in EXPERIMENTS.md.

    Experiments with no caveat line map to "".
    """
    if md_path is None:
        md_path = ROOT / "EXPERIMENTS.md"
    text = md_path.read_text(encoding="utf-8")

    # Split into per-experiment blocks on "## Exp N " headings.
    header_re = re.compile(r"^## Exp (\d+) ", re.MULTILINE)
    headers = [(m.start(), int(m.group(1))) for m in header_re.finditer(text)]

    caveats: dict[int, str] = {}
    for i, (pos, n) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        block = text[pos:end]

        # Match "- Honest caveat[...]:" or "- Caveat[...]:"
        m = re.search(
            r"-\s*Honest caveat[^:]*:\s*(.+?)(?=\n-|\n##|\Z)",
            block,
            re.DOTALL,
        )
        if not m:
            m = re.search(
                r"-\s*Caveat[^:]*:\s*(.+?)(?=\n-|\n##|\Z)",
                block,
                re.DOTALL,
            )

        if m:
            # Flatten continuation lines (indented with spaces) into one line.
            raw = m.group(1).strip()
            raw = re.sub(r"\n[ \t]+", " ", raw)
            raw = raw.replace("\n", " ")
            # First sentence: split at  ./?/!  followed by whitespace + capital or "(".
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", raw)
            caveats[n] = parts[0].strip()
        else:
            caveats[n] = ""

    return caveats


# ---------------------------------------------------------------------------
# experiments-data.js patcher
# ---------------------------------------------------------------------------

def inject_caveats(
    js_path: pathlib.Path | None = None,
    caveats: dict[int, str] | None = None,
) -> str:
    """Return the updated text of experiments-data.js with caveat fields injected.

    For each entry ``{ n:N, ...  trace:{...} }`` in AM_EXPERIMENTS, a
    ``caveat:"<text>"`` field is inserted immediately before the ``trace:``
    field.  If the caveat for that experiment is the empty string the field is
    still written (``caveat:""``), so front-end code can rely on the key always
    being present.

    Already-existing ``caveat:`` fields are replaced (idempotent).

    The function returns the new file text; it does NOT write to disk.
    """
    if js_path is None:
        js_path = ROOT / "experiments-data.js"
    if caveats is None:
        caveats = parse_caveats_from_md()

    text = js_path.read_text(encoding="utf-8")

    def _escape_js(s: str) -> str:
        """Escape a Python string for use inside a JS double-quoted string."""
        return s.replace("\\", "\\\\").replace('"', '\\"')

    # Strategy: locate each entry's `trace:` line (which is the last field
    # before the closing `}`).  Just before it, remove any existing caveat
    # field, then insert the new one.
    #
    # We process the AM_EXPERIMENTS body only — identified by the array bounds
    # so we never accidentally touch other arrays.

    start_m = re.search(r"window\.AM_EXPERIMENTS\s*=\s*\[", text)
    if not start_m:
        raise ValueError("AM_EXPERIMENTS not found in experiments-data.js")
    body_start = start_m.end()
    rest = text[body_start:]
    end_m = re.search(r"^\s*\]\s*;", rest, re.MULTILINE)
    if not end_m:
        raise ValueError("AM_EXPERIMENTS closing ]; not found")

    prefix = text[: body_start]
    body = rest[: end_m.start()]
    suffix = rest[end_m.start():]

    # We'll walk entry-by-entry.  Each entry starts at `  { n:N,` and the
    # `trace:` line is the last field.  We insert/replace the caveat field
    # on the line immediately before `trace:`.

    # Remove any pre-existing caveat lines (idempotent re-runs).
    body = re.sub(
        r"\n[ \t]+caveat\s*:[ \t]*\"(?:[^\"\\]|\\.)*\"[ \t]*,?\n",
        "\n",
        body,
    )

    # Now insert caveat before each trace: line.
    def _replace_trace(m: re.Match) -> str:
        # m captures: (indent)(n_value)(indent_trace)(trace_line)
        indent = m.group(1)
        n_val = int(m.group(2))
        indent_trace = m.group(3)
        trace_line = m.group(4)
        caveat_text = _escape_js(caveats.get(n_val, ""))
        return (
            f"\n{indent}n:{n_val},"
            f"\n{indent_trace}caveat:\"{caveat_text}\","
            f"\n{indent_trace}{trace_line}"
        )

    # Pattern: capture  "  { n:N,"  ...  "    trace:"
    # We need to match each occurrence of n:<digits> followed (eventually on the
    # same or a later line) by the trace: field.  Since entries don't nest, we
    # do a two-step: (1) locate n: values in entry-opener lines, (2) find the
    # matching trace: in the next few lines.
    #
    # Simpler approach: a single regex that matches the opener `n:N,` on its
    # line AND the `trace:` token on the immediately-preceding line of the
    # closing block, inserting caveat before trace:.
    #
    # Because entries vary in field count we target trace: directly.

    # Match:  newline + indent + "trace:"
    # and capture the n: for the enclosing entry from the preceding text.
    # We'll do a positional scan instead.

    new_body_parts: list[str] = []
    pos = 0
    # Find each "n:<digits>," in entry-opener lines to build a map of
    # (offset_of_n_line, n_value).
    n_positions: list[tuple[int, int]] = [
        (m.start(), int(m.group(1)))
        for m in re.finditer(r"(?m)^\s*\{\s*n\s*:\s*(\d+)\s*,", body)
    ]
    # Find each trace: position.
    trace_positions: list[int] = [
        m.start() for m in re.finditer(r"(?m)^\s*trace\s*:", body)
    ]

    if not n_positions or not trace_positions:
        # Nothing to patch — return unchanged.
        return text

    # For each trace: find the n-value of its enclosing entry (the largest
    # n_position that is still before this trace: position).
    # Build list of (trace_start, n_value, full_trace_line_end).
    trace_patches: list[tuple[int, int, int]] = []
    for tp in trace_positions:
        # find the entry n-value: last n_pos before tp
        enclosing_n = None
        for np, nv in n_positions:
            if np < tp:
                enclosing_n = nv
            else:
                break
        if enclosing_n is None:
            continue
        # find end of the trace line
        line_end = body.find("\n", tp)
        if line_end == -1:
            line_end = len(body)
        trace_patches.append((tp, enclosing_n, line_end))

    # Now rebuild the body, inserting caveat lines before each trace:.
    cursor = 0
    for tp, nv, te in trace_patches:
        # Text from cursor up to (but not including) the trace: line.
        segment = body[cursor:tp]
        # tp IS the start of the trace: line (because the regex uses ^ anchor).
        # Extract leading whitespace directly from that position.
        indent_m = re.match(r"(\s*)", body[tp:])
        indent = indent_m.group(1) if indent_m else "    "

        caveat_text = _escape_js(caveats.get(nv, ""))
        caveat_line = f"{indent}caveat:\"{caveat_text}\",\n"
        new_body_parts.append(segment)
        new_body_parts.append(caveat_line)
        cursor = tp  # trace line follows

    new_body_parts.append(body[cursor:])
    new_body = "".join(new_body_parts)

    return prefix + new_body + suffix


def regenerate(
    js_path: pathlib.Path | None = None,
    md_path: pathlib.Path | None = None,
) -> None:
    """Re-inject caveats and write experiments-data.js in place."""
    if js_path is None:
        js_path = ROOT / "experiments-data.js"
    if md_path is None:
        md_path = ROOT / "EXPERIMENTS.md"
    caveats = parse_caveats_from_md(md_path)
    new_text = inject_caveats(js_path, caveats)
    js_path.write_text(new_text, encoding="utf-8")


# ---------------------------------------------------------------------------
# lab_status() builder
# ---------------------------------------------------------------------------

# Valid direction states, in the sort order specified in the API contract.
_STATE_ORDER = [
    "active",
    "halted",
    "flagship-candidate",
    "exploratory",
    "closed-positive",
    "closed-negative",
    "published",
]

# Regex to find experiment section headers:  ## Exp N — <title>
_EXP_HEADER_RE = re.compile(
    r"^## Exp (\d+) — (.+)$",  # U+2014 em-dash
    re.MULTILINE,
)

# Regex to find an explicit "- Verdict:" line within an experiment block.
_VERDICT_RE = re.compile(
    r"^\s*-\s*Verdict\s*:\s*(.+?)(?=\n\s*-|\n##|\Z)",
    re.MULTILINE | re.DOTALL,
)

# Regex to find BREAKTHROUGH in a verdict line or in the section header.
_BREAKTHROUGH_RE = re.compile(r"\bBREAKTHROUGH\b")


def _js_escape(s: str) -> str:
    """Escape a string for safe embedding in a JS double-quoted string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _extract_tag(text: str) -> str:
    """Extract the verdict tag (POSITIVE/NEGATIVE/MIXED) from a block of text.

    Looks for an explicit '- Verdict:' line first; falls back to the header
    text.  Returns the first match of POSITIVE, NEGATIVE, or MIXED (case-
    insensitive match against the all-caps canonical forms).  Returns "" if
    no tag is found (conservative — the caller treats "" as untagged).
    """
    for m in _VERDICT_RE.finditer(text):
        val = m.group(1)
        for tag in ("POSITIVE", "NEGATIVE", "MIXED"):
            if tag in val:
                return tag
        # Lower-case variants in older entries
        lower = val.lower()
        for tag in ("positive", "negative", "mixed"):
            if tag in lower:
                return tag.upper()
    # Fallback: scan the whole block (catches header-embedded tags in early exps)
    for tag in ("POSITIVE", "NEGATIVE", "MIXED"):
        if tag in text:
            return tag
    lower = text.lower()
    for tag in ("positive", "negative", "mixed"):
        if tag in lower:
            return tag.upper()
    return ""


def _is_breakthrough(block_text: str) -> bool:
    """Return True if BREAKTHROUGH appears in the block (header or verdict line)."""
    return bool(_BREAKTHROUGH_RE.search(block_text))


def _first_sentence(text: str) -> str:
    """Return the first sentence of *text* (split on .!? + whitespace + uppercase)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip())
    return parts[0].strip() if parts else text.strip()


def parse_experiments(md_path: pathlib.Path | None = None) -> list[dict]:
    """Parse EXPERIMENTS.md and return a list of experiment dicts.

    Each dict has keys: n (int), title (str), tag (str), is_breakthrough (bool).

    Title is the headline text after the em-dash (first sentence, JS-safe).
    Tag is POSITIVE / NEGATIVE / MIXED or "" if undetermined.
    Breakthroughs are counted conservatively: must contain the word BREAKTHROUGH.
    """
    if md_path is None:
        md_path = ROOT / "EXPERIMENTS.md"
    text = md_path.read_text(encoding="utf-8")

    headers = list(_EXP_HEADER_RE.finditer(text))
    results: list[dict] = []

    for i, m in enumerate(headers):
        n = int(m.group(1))
        raw_headline = m.group(2).strip()

        # Title = text after the em-dash in "## Exp N — <title>", first sentence.
        # Strip the trailing parenthetical verdict/note, e.g. "(POSITIVE; ...)",
        # then take the first sentence (split at .!? + whitespace + uppercase).
        headline = re.sub(r"\s*\([^)]*\)\s*$", "", raw_headline).strip()
        title = _first_sentence(headline)

        # Get the block for this experiment
        block_start = m.start()
        block_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        block = text[block_start:block_end]

        tag = _extract_tag(block)
        breakthrough = _is_breakthrough(block)

        results.append({
            "n": n,
            "title": title,
            "tag": tag,
            "is_breakthrough": breakthrough,
        })

    return results


def parse_directions(directions_dir: pathlib.Path | None = None) -> list[dict]:
    """Parse all direction cards and return a list of direction dicts.

    Each dict has keys: name (str), state (str), latest (str), next (str).

    Uses parse_status_block from tools/gen_directions_index.py to avoid
    duplicating the STATUS-block parsing logic.
    """
    if directions_dir is None:
        directions_dir = ROOT / "loop" / "directions"

    gen = _get_directions_parser()

    cards = sorted(directions_dir.glob("*.md"))
    directions: list[dict] = []

    for card in cards:
        if card.name == "_TEMPLATE.md":
            continue
        text = card.read_text(encoding="utf-8")
        fields = gen.parse_status_block(text)
        if fields is None:
            continue
        name = card.stem
        state = fields.get("state", "TBD-human")
        latest = fields.get("latest", "TBD")
        next_step = fields.get("next-falsifiable", "")
        # Take first sentence of next_step for the JS output
        next_line = _first_sentence(next_step) if next_step else ""
        directions.append({
            "name": name,
            "state": state,
            "latest": latest,
            "next": next_line,
        })

    # Sort by state order then by name
    def _state_key(d: dict) -> tuple[int, str]:
        state = d["state"]
        # Normalize: strip any trailing parenthetical annotation
        normalized = state.split("(")[0].strip()
        # Also handle "active (prereq build)" -> "active"
        normalized = normalized.split()[0] if " " in normalized else normalized
        try:
            idx = _STATE_ORDER.index(normalized)
        except ValueError:
            idx = len(_STATE_ORDER)
        return (idx, d["name"])

    directions.sort(key=_state_key)
    return directions


# The flagship metadata is static (defined by the spec).
_FLAGSHIP = {
    "title": "Agents That Know When Their Worldview Is Too Small",
    "page": "docs/flagship/worldview-too-small.md",
}


def lab_status(
    md_path: pathlib.Path | None = None,
    directions_dir: pathlib.Path | None = None,
) -> str:
    """Build and return the full lab-status.js content (no I/O).

    Deterministic: regenerating twice from the same inputs produces byte-identical
    output.  No wall-clock timestamps or environment-dependent content.
    """
    exps = parse_experiments(md_path)
    if not exps:
        raise ValueError("No experiments found in EXPERIMENTS.md")

    # Latest experiment = highest n
    latest = max(exps, key=lambda e: e["n"])

    # Tally
    total = len(exps)
    positive = sum(1 for e in exps if e["tag"] == "POSITIVE")
    negative = sum(1 for e in exps if e["tag"] == "NEGATIVE")
    mixed = sum(1 for e in exps if e["tag"] == "MIXED")
    breakthroughs = sum(1 for e in exps if e["is_breakthrough"])

    # Directions
    dirs = parse_directions(directions_dir)

    # Build JS
    lines: list[str] = []
    lines.append(
        "/* GENERATED — do not hand-edit."
        " Regenerate: uv run --python .venv python -m active_loop.site_data --lab-status */"
    )
    lines.append("window.AM_LAB_STATUS = {")

    # latest_exp
    title_js = _js_escape(latest["title"])
    tag = latest["tag"] or "MIXED"
    lines.append(
        f'  latest_exp: {{ n: {latest["n"]}, title: "{title_js}", tag: "{tag}" }},'
    )

    # tally
    lines.append(
        f"  tally: {{ total: {total}, positive: {positive},"
        f" negative: {negative}, mixed: {mixed}, breakthroughs: {breakthroughs} }},"
    )

    # directions
    lines.append("  directions: [")
    for d in dirs:
        name_js = _js_escape(d["name"])
        state_js = _js_escape(d["state"])
        latest_js = _js_escape(d["latest"])
        next_js = _js_escape(d["next"])
        lines.append(
            f'    {{ name: "{name_js}", state: "{state_js}",'
            f' latest: "{latest_js}", next: "{next_js}" }},'
        )
    lines.append("  ],")

    # flagship
    flagship_title_js = _js_escape(_FLAGSHIP["title"])
    flagship_page_js = _js_escape(_FLAGSHIP["page"])
    lines.append(
        f'  flagship: {{ title: "{flagship_title_js}",'
        f' page: "{flagship_page_js}" }}'
    )

    lines.append("};")
    return "\n".join(lines) + "\n"


def generate_lab_status(
    out_path: pathlib.Path | None = None,
    md_path: pathlib.Path | None = None,
    directions_dir: pathlib.Path | None = None,
) -> None:
    """Generate lab-status.js and write it to *out_path* (default: repo root)."""
    if out_path is None:
        out_path = ROOT / "lab-status.js"
    content = lab_status(md_path, directions_dir)
    out_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    if "--lab-status" in sys.argv:
        generate_lab_status()
        print("lab-status.js regenerated.", file=sys.stderr)
    else:
        regenerate()
        print("experiments-data.js regenerated with caveat fields.", file=sys.stderr)
