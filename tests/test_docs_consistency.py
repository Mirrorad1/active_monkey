"""Regression tests for the lightweight Markdown docs consistency checker.

Run: /Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_docs_consistency.py -q
"""
from __future__ import annotations

import importlib.util
import pathlib
import textwrap


ROOT = pathlib.Path(__file__).parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "check_docs", ROOT / "tools" / "check_docs.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_markdown_link_checker_ignores_urls_and_flags_missing_local_files(tmp_path):
    checker = _load()
    docs = tmp_path / "docs"
    docs.mkdir()
    (tmp_path / "README.md").write_text(
        textwrap.dedent(
            """
            # Lab

            [claims](docs/CLAIMS.md)
            [external](https://example.com)
            [missing](docs/MISSING.md)
            """
        ),
        encoding="utf-8",
    )
    (docs / "CLAIMS.md").write_text("# Claims\n", encoding="utf-8")

    errors = checker.check_markdown_links(
        tmp_path, [tmp_path / "README.md", docs / "CLAIMS.md"]
    )

    assert errors == ["README.md links to missing local file docs/MISSING.md"]


def test_claim_experiment_checker_requires_cited_ids_to_exist(tmp_path):
    checker = _load()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "CLAIMS.md").write_text(
        "| Evidence |\n|---|\n| Exp 1, Exp 99 |\n",
        encoding="utf-8",
    )
    (tmp_path / "EXPERIMENTS.md").write_text(
        "## Exp 1 - present\nbody\n",
        encoding="utf-8",
    )

    errors = checker.check_claim_experiments(tmp_path)

    assert errors == ["docs/CLAIMS.md cites missing experiment IDs: 99"]
