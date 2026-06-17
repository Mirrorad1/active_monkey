"""Tests for the coalesce CLI subcommands."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from active_loop.cli.coalesce import build_coalesce_parser


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse(argv):
    """Parse args without the leading prog name."""
    return build_coalesce_parser().parse_args(argv)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_inventory_parses_and_runs():
    args = _parse(["coalesce", "inventory", "--json"])
    assert args.func(args) == 0


def test_backfill_plan_writes_file(tmp_path):
    out = tmp_path / "plan.json"
    args = _parse(["coalesce", "backfill-plan", "--out", str(out)])
    rc = args.func(args)
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "plans" in data


def test_export_then_validate(tmp_path):
    exp_dir = tmp_path / "exp222_bundle"
    # export
    args_export = _parse([
        "coalesce", "export",
        "--experiment", "exp222",
        "--level", "metrics_bundle",
        "--out", str(exp_dir),
    ])
    rc = args_export.func(args_export)
    assert rc == 0, "export should succeed"
    assert (exp_dir / "manifest.json").exists()

    # validate the exported bundle
    args_val = _parse(["coalesce", "validate", str(exp_dir)])
    rc_val = args_val.func(args_val)
    assert rc_val == 0, "validate should pass on the exported bundle"


def test_validate_all_returns_zero_on_clean_repo():
    args = _parse(["coalesce", "validate", "--all"])
    rc = args.func(args)
    assert rc == 0


def test_mechanisms_list_runs():
    args = _parse(["coalesce", "mechanisms", "list"])
    assert args.func(args) == 0


def test_mechanisms_list_runs_json():
    args = _parse(["coalesce", "mechanisms", "list", "--json"])
    assert args.func(args) == 0


def test_geometry_list_runs():
    args = _parse(["coalesce", "geometry", "list"])
    assert args.func(args) == 0


def test_geometry_list_runs_json():
    args = _parse(["coalesce", "geometry", "list", "--json"])
    assert args.func(args) == 0
