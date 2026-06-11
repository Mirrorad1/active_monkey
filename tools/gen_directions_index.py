"""Generate DIRECTIONS.md — a one-table index of all direction cards.

Usage (from repo root):
    uv run --python .venv python tools/gen_directions_index.py

The script parses the **STATUS.** block at the end of each
loop/directions/*.md file (skipping _TEMPLATE.md) and writes
DIRECTIONS.md at the repo root with a deterministic, sorted table:

    direction · state · latest Exp · next falsifiable step

Parsing tolerates both single-line and wrapped multi-line STATUS blocks,
and passes through TBD-human values unchanged.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent.parent
DIRECTIONS_DIR = ROOT / "loop" / "directions"
OUTPUT_PATH = ROOT / "DIRECTIONS.md"

HEADER = "GENERATED — edit the cards in loop/directions/, then re-run tools/gen_directions_index.py"

# The STATUS line format (after T12):
#   **STATUS.** state: <value> · latest: <value> · depends-on: <value> ·
#               reusable: <value> · why: <value> · next-falsifiable: <value>
# The block may be wrapped across two lines (the middle dot · may be followed
# by a newline before the next field key).
_STATUS_RE = re.compile(r"\*\*STATUS\.\*\*\s+(.+)", re.DOTALL)

# Field extraction: key followed by ': ' then everything up to the next
# ' · <key>:' boundary or end of the block text.
_FIELD_RE = re.compile(r"(\w[\w-]*)\s*:\s*(.*?)(?=\s+·\s+\w[\w-]*\s*:|$)", re.DOTALL)


def _clean(text: str) -> str:
    """Collapse whitespace/newlines inside a field value."""
    return " ".join(text.split())


def parse_status_block(card_text: str) -> dict[str, str] | None:
    """Return a dict of STATUS fields from the LAST **STATUS.** block in *card_text*.

    Returns None if no STATUS block is found.
    """
    # Find all matches; take the last one (some cards have an earlier narrative
    # STATUS block before the structured one added by T12).
    matches = list(_STATUS_RE.finditer(card_text))
    if not matches:
        return None
    raw = matches[-1].group(1)

    fields: dict[str, str] = {}
    for m in _FIELD_RE.finditer(raw):
        key = m.group(1).strip()
        val = _clean(m.group(2))
        fields[key] = val
    return fields if fields else None


def direction_name(path: pathlib.Path) -> str:
    """Return the display name: the stem of the filename."""
    return path.stem


def build_table(rows: list[tuple[str, dict[str, str]]]) -> str:
    """Build a Markdown table from (name, fields) rows."""
    col_headers = ["direction", "state", "latest Exp", "next falsifiable step"]
    data = []
    for name, fields in rows:
        state = fields.get("state", "TBD-human")
        latest = fields.get("latest", "TBD-human")
        # Strip leading "Exp " prefix duplication if the value already starts with "Exp"
        # (keep as-is — it's what the card says)
        nf = fields.get("next-falsifiable", "TBD-human")
        data.append([name, state, latest, nf])

    # Compute column widths
    widths = [len(h) for h in col_headers]
    for row in data:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    def sep_row() -> str:
        return "| " + " | ".join("-" * widths[i] for i in range(len(col_headers))) + " |"

    lines = [fmt_row(col_headers), sep_row()]
    for row in data:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def generate() -> str:
    """Parse all direction cards and return the full DIRECTIONS.md content."""
    cards = sorted(DIRECTIONS_DIR.glob("*.md"))
    rows: list[tuple[str, dict[str, str]]] = []

    for card in cards:
        if card.name == "_TEMPLATE.md":
            continue
        text = card.read_text(encoding="utf-8")
        fields = parse_status_block(text)
        if fields is None:
            print(f"WARNING: no STATUS block found in {card.name}", file=sys.stderr)
            continue
        rows.append((direction_name(card), fields))

    # rows are already sorted by filename (glob + sorted)
    table = build_table(rows)
    content = f"<!-- {HEADER} -->\n\n# Directions Index\n\n{table}\n"
    return content


def main() -> None:
    content = generate()
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
