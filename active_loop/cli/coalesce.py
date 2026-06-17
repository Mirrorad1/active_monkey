"""Coalescence CLI subcommands for the active-monkey CLI.

Subcommands under "coalesce":
  inventory    [--json]
  backfill-plan [--out PATH] [--json]
  export        --experiment EXPID --level LEVEL --out DIR
  validate      PATH | --all
  mechanisms    list [--json]
  geometry      list [--json]

Usage (after registration in build_parser):
  active-monkey coalesce inventory --json
  active-monkey coalesce backfill-plan --out /tmp/plan.json
  active-monkey coalesce export --experiment exp222 --level metrics_bundle --out /tmp/exp222
  active-monkey coalesce validate /tmp/exp222
  active-monkey coalesce validate --all
  active-monkey coalesce mechanisms list
  active-monkey coalesce geometry list
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _print_json(obj) -> None:
    print(json.dumps(obj, sort_keys=True, indent=2))


# ── inventory ─────────────────────────────────────────────────────────────────

def _cmd_inventory(args) -> int:
    from active_loop.coalescence.inventory import build_inventory, inventory_json

    inv = build_inventory()
    if args.json:
        print(inventory_json())
    else:
        print(f"experiments: {inv['count']}")
        print(f"repo_commit: {inv['repo_commit']}")
        print()
        print("counts by direction:")
        for direction, cnt in sorted(inv["counts_by_direction"].items()):
            print(f"  {direction:<40} {cnt}")
        print()
        print("counts by confidence:")
        for conf, cnt in inv["counts_by_confidence"].items():
            print(f"  {conf:<12} {cnt}")
    return 0


# ── backfill-plan ─────────────────────────────────────────────────────────────

def _cmd_backfill_plan(args) -> int:
    from active_loop.coalescence.backfill import backfill_plan, backfill_plan_json
    from active_loop.coalescence.schema import write_json

    plan = backfill_plan()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_json(plan, out)

    if args.json:
        print(backfill_plan_json())
    else:
        summary = plan.get("summary", {})
        cbl = summary.get("counts_by_current_level", {})
        immediate = len(summary.get("can_backfill_immediately", []))
        rerun = len(summary.get("need_rerun_for_trajectories", []))
        print("backfill-plan summary:")
        print(f"  can_backfill_immediately: {immediate}")
        print(f"  need_rerun_for_trajectories: {rerun}")
        print(f"  need_checkpoint_export: {len(summary.get('need_checkpoint_export', []))}")
        print()
        print("counts by current level:")
        for lvl, cnt in cbl.items():
            print(f"  {lvl:<25} {cnt}")
        if args.out:
            print()
            print(f"written to: {args.out}")
    return 0


# ── export ────────────────────────────────────────────────────────────────────

def _cmd_export(args) -> int:
    from active_loop.coalescence.export import export_bundle

    try:
        manifest = export_bundle(
            experiment_id=args.experiment,
            level=args.level,
            out_dir=args.out,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    manifest_path = Path(args.out) / "manifest.json"
    print(f"manifest: {manifest_path}")
    print(f"backfill_level: {manifest.get('backfill_level', args.level)}")
    return 0


# ── validate ──────────────────────────────────────────────────────────────────

def _cmd_validate(args) -> int:
    from active_loop.coalescence.validate import (
        validate_artifact_file,
        validate_bundle,
        validate_all,
    )

    if args.all:
        result = validate_all(".")
        passed = result["passed"]
        failed = result["failed"]
        skipped = result["skipped"]
        print(f"passed: {len(passed)}  failed: {len(failed)}  skipped: {len(skipped)}")
        if failed:
            print()
            print("failures:")
            for f in failed:
                print(f"  {f['path']}: {f['error']}")
            return 1
        return 0

    if not args.path:
        print("error: PATH required unless --all is set", file=sys.stderr)
        return 2

    p = Path(args.path)
    try:
        if p.is_dir():
            result = validate_bundle(p)
        else:
            result = validate_artifact_file(p)
        print(f"OK: {p}")
        return 0
    except (FileNotFoundError, ValueError) as e:
        print(f"FAIL: {p}: {e}", file=sys.stderr)
        return 1


# ── mechanisms list ───────────────────────────────────────────────────────────

def _cmd_mechanisms_list(args) -> int:
    from active_loop.coalescence.schema import load

    root = Path(".")
    mechanisms_dir = root / "mechanisms"
    cards = []
    if mechanisms_dir.exists():
        for pattern in ("*/mechanism_card.yaml", "*/mechanism_card.yml", "*/mechanism_card.json"):
            for p in sorted(mechanisms_dir.glob(pattern)):
                try:
                    d = load(p)
                    cards.append({
                        "mechanism_id": d.get("mechanism_id", p.parent.name),
                        "status": d.get("status", "unknown"),
                        "mechanism_type": d.get("mechanism_type", "unknown"),
                        "source_experiments": d.get("source_experiments", []),
                    })
                except Exception as e:
                    print(f"warning: could not load {p}: {e}", file=sys.stderr)

    if args.json:
        _print_json(cards)
    else:
        if not cards:
            print("(none)")
        else:
            for card in cards:
                print(
                    f"{card['mechanism_id']}  status={card['status']}"
                    f"  type={card['mechanism_type']}"
                    f"  source_experiments={card['source_experiments']}"
                )
    return 0


# ── geometry list ─────────────────────────────────────────────────────────────

def _cmd_geometry_list(args) -> int:
    from active_loop.coalescence.schema import load

    root = Path(".")
    items = []

    # geometry_maps/*.{yaml,yml,json}
    geo_dir = root / "geometry_maps"
    if geo_dir.exists():
        for suffix in ("*.yaml", "*.yml", "*.json"):
            for p in sorted(geo_dir.glob(suffix)):
                try:
                    d = load(p)
                    items.append({
                        "geometry_id": d.get("geometry_id", p.stem),
                        "mechanism_id": d.get("mechanism_id"),
                        "source": "geometry_maps",
                    })
                except Exception as e:
                    print(f"warning: could not load {p}: {e}", file=sys.stderr)

    # boundary_notes/*.{yaml,yml,json}
    bn_dir = root / "boundary_notes"
    if bn_dir.exists():
        for suffix in ("*.yaml", "*.yml", "*.json"):
            for p in sorted(bn_dir.glob(suffix)):
                try:
                    d = load(p)
                    items.append({
                        "boundary_id": d.get("boundary_id", p.stem),
                        "failed_mechanism": d.get("failed_mechanism"),
                        "source": "boundary_notes",
                    })
                except Exception as e:
                    print(f"warning: could not load {p}: {e}", file=sys.stderr)

    if args.json:
        _print_json(items)
    else:
        if not items:
            print("(none)")
        else:
            for item in items:
                if item.get("source") == "geometry_maps":
                    print(
                        f"geometry_id={item.get('geometry_id')}  mechanism_id={item.get('mechanism_id')}"
                    )
                else:
                    print(
                        f"boundary_id={item.get('boundary_id')}  failed_mechanism={item.get('failed_mechanism')}"
                    )
    return 0


# ── subparser registration ────────────────────────────────────────────────────

def register_coalesce_subparser(top_sub) -> None:
    """Add the 'coalesce' parser + its subcommands to *top_sub* (an argparse subparsers object)."""
    coal = top_sub.add_parser("coalesce", help="coalescence layer: inventory, backfill, export, validate")
    coal_sub = coal.add_subparsers(dest="coalesce_subcommand", required=True)

    # inventory
    inv_p = coal_sub.add_parser("inventory", help="show experiment inventory")
    inv_p.add_argument("--json", action="store_true", help="output as JSON")
    inv_p.set_defaults(func=_cmd_inventory)

    # backfill-plan
    bp_p = coal_sub.add_parser("backfill-plan", help="show/write backfill plan")
    bp_p.add_argument("--out", default=None, metavar="PATH", help="write canonical JSON to PATH")
    bp_p.add_argument("--json", action="store_true", help="print canonical JSON to stdout")
    bp_p.set_defaults(func=_cmd_backfill_plan)

    # export
    ex_p = coal_sub.add_parser("export", help="export one experiment bundle")
    ex_p.add_argument("--experiment", required=True, metavar="EXPID", help="e.g. exp222")
    ex_p.add_argument("--level", required=True, metavar="LEVEL", help="backfill level name")
    ex_p.add_argument("--out", required=True, metavar="DIR", help="output directory")
    ex_p.set_defaults(func=_cmd_export)

    # validate
    val_p = coal_sub.add_parser("validate", help="validate an artifact file, bundle dir, or all")
    val_p.add_argument("path", nargs="?", default=None, metavar="PATH",
                       help="artifact file or bundle directory (omit with --all)")
    val_p.add_argument("--all", action="store_true", help="validate all artifacts in repo")
    val_p.set_defaults(func=_cmd_validate)

    # mechanisms
    mech_p = coal_sub.add_parser("mechanisms", help="inspect mechanism cards")
    mech_sub = mech_p.add_subparsers(dest="mechanisms_subcommand", required=True)
    ml_p = mech_sub.add_parser("list", help="list mechanism cards")
    ml_p.add_argument("--json", action="store_true")
    ml_p.set_defaults(func=_cmd_mechanisms_list)

    # geometry
    geo_p = coal_sub.add_parser("geometry", help="inspect geometry maps and boundary notes")
    geo_sub = geo_p.add_subparsers(dest="geometry_subcommand", required=True)
    gl_p = geo_sub.add_parser("list", help="list geometry maps and boundary notes")
    gl_p.add_argument("--json", action="store_true")
    gl_p.set_defaults(func=_cmd_geometry_list)


def build_coalesce_parser() -> argparse.ArgumentParser:
    """Return a standalone parser with the 'coalesce' subcommand (for testing)."""
    p = argparse.ArgumentParser(prog="coalesce-test")
    top_sub = p.add_subparsers(dest="command", required=True)
    register_coalesce_subparser(top_sub)
    return p
