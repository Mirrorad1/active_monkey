"""Staleness guard for DIRECTIONS.md.

Asserts that re-running tools/gen_directions_index.py would produce
byte-identical output to the committed DIRECTIONS.md.  Any edit to a
direction card's STATUS block that is not followed by re-running the
generator will be caught here.

Pattern follows tests/test_site_data.py — stdlib only, no heavy imports.
"""
import pathlib
import importlib.util
import sys

ROOT = pathlib.Path(__file__).parent.parent


def _load_generator():
    """Import gen_directions_index without executing main()."""
    spec = importlib.util.spec_from_file_location(
        "gen_directions_index",
        ROOT / "tools" / "gen_directions_index.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_directions_md_is_not_stale():
    """Re-running the generator produces bytes identical to the committed DIRECTIONS.md.

    If this test fails, run:  uv run --python .venv python tools/gen_directions_index.py
    and commit the updated DIRECTIONS.md.
    """
    committed = (ROOT / "DIRECTIONS.md").read_text(encoding="utf-8")
    gen = _load_generator()
    regenerated = gen.generate()
    assert regenerated == committed, (
        "DIRECTIONS.md is stale — re-run tools/gen_directions_index.py and commit.\n"
        f"Committed length: {len(committed)}, regenerated length: {len(regenerated)}"
    )
