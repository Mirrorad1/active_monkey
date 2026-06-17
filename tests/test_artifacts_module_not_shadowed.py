"""Guard: active_loop.artifacts must resolve to the MODULE, never a package shadow.

`active_loop/artifacts.py` (the artifact API) and a stray `active_loop/artifacts/` directory
share the import name. CPython prefers the .py module today, but if anyone ever added
`active_loop/artifacts/__init__.py` the directory would shadow the module and silently break
every `from active_loop.artifacts import ...`. This fast test fails loudly if that happens.
"""
from __future__ import annotations

import pathlib

import active_loop.artifacts as artifacts

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_artifacts_resolves_to_module_file():
    assert pathlib.Path(artifacts.__file__).name == "artifacts.py"


def test_no_artifacts_package_init():
    # A package __init__ here would shadow the module — must never exist.
    assert not (ROOT / "active_loop" / "artifacts" / "__init__.py").exists()
