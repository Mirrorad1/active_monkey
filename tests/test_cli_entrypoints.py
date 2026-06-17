"""Structural guards for runnable command entrypoints.

Run:  /Users/mirro/Projects/active-loop/.venv/bin/python -m pytest tests/test_cli_entrypoints.py -q
"""
from __future__ import annotations

import importlib
import pathlib
import tomllib

ROOT = pathlib.Path(__file__).parent.parent

EXPECTED_CONSOLE_SCRIPTS = {
    "active-monkey": "active_loop.cli.main:main",
    "active-monkey-loop": "active_loop.cli.run_loop:main",
    "active-monkey-pr-loop": "active_loop.cli.run_pr_loop:main",
    "active-monkey-affect-loop": "active_loop.cli.run_affect_loop:main",
    "active-monkey-life": "active_loop.cli.run_life:main",
    "active-monkey-m1": "active_loop.cli.run_m1:main",
    "active-monkey-talk": "active_loop.cli.talk:main",
    "active-monkey-converse": "active_loop.cli.converse:main",
    "active-monkey-converse-demo": "active_loop.cli.converse_demo:main",
}


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_no_ad_hoc_python_entrypoints_at_repo_root():
    root_python = sorted(path.name for path in ROOT.glob("*.py"))

    assert root_python == []


def test_pyproject_exposes_console_scripts_for_command_entrypoints():
    scripts = _pyproject()["project"]["scripts"]

    for name, target in EXPECTED_CONSOLE_SCRIPTS.items():
        assert scripts[name] == target


def test_pyproject_has_build_backend_for_console_scripts():
    pyproject = _pyproject()

    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"


def test_console_script_targets_are_importable_callables():
    for target in EXPECTED_CONSOLE_SCRIPTS.values():
        module_name, _, attr = target.partition(":")
        module = importlib.import_module(module_name)

        assert callable(getattr(module, attr))
