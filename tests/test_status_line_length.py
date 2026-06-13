"""Guard: every direction card's STATUS line must be <= 800 characters.

A STATUS line is the single line starting with '**STATUS.**' in a
loop/directions/*.md file.  It is HOT context (fed into DIRECTIONS.md each
iteration), so runaway length is a concrete cost.  See loop/EFFICIENCY.md.
"""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
DIRECTIONS_DIR = ROOT / "loop" / "directions"
MAX_LEN = 800


def test_status_line_length():
    """All direction card STATUS lines fit within 800 characters."""
    violations = []
    for md in sorted(DIRECTIONS_DIR.glob("*.md")):
        for line in md.read_text(encoding="utf-8").splitlines():
            if line.startswith("**STATUS.**"):
                length = len(line)
                if length > MAX_LEN:
                    violations.append((md.name, length))
                break  # only one STATUS line per file

    assert not violations, (
        "STATUS line(s) exceed 800 chars — trim the narrative detail into the card BODY "
        "(see loop/EFFICIENCY.md):\n"
        + "\n".join(f"  {name}: {length} chars" for name, length in violations)
    )
