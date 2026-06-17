"""Stable `active-monkey` CLI: export / inspect / score / converse over frozen artifacts.

Subcommands:
  active-monkey artifact export --preset affect-dyad-v0 --out <dir>
  active-monkey artifact inspect <dir> [--json]
  active-monkey score <dir> [--json] [--quick]
  active-monkey converse <dir> [--demo | --interactive]

Design rules:
  - Exit nonzero on invalid artifacts / structured failures.
  - `--json` prints machine-readable JSON; otherwise a human-readable summary.
  - Never uploads anything; never requires the network.
  - Agent-runtime imports (pymdp/JAX) are lazy: export/inspect of init tensors work
    without the inference engine; score/converse import it on demand with clear errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print_json(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


# ── artifact export ──────────────────────────────────────────────────────────

def _cmd_artifact_export(args) -> int:
    from active_loop.artifacts import export_affect_dyad_artifact

    if args.preset != "affect-dyad-v0":
        print(f"error: unknown preset {args.preset!r} (known: affect-dyad-v0)", file=sys.stderr)
        return 2
    manifest = export_affect_dyad_artifact(
        args.out,
        seed=args.seed,
        learn_turns=args.learn_turns,
        run_learned=not args.no_learned,
        run_eval=not args.no_eval,
    )
    if args.json:
        _print_json(manifest)
    else:
        print(f"exported {manifest['artifact_id']} -> {args.out}")
        print(f"  scorer_hash:        {manifest['scorer_hash']}")
        print(f"  init_checkpoint:    {manifest['init_checkpoint_hash']}")
        print(f"  learned_checkpoint: {manifest.get('learned_checkpoint_hash')}")
        if manifest.get("learned_unavailable"):
            print(f"  (learned skipped:   {manifest['learned_unavailable']})")
        print(f"  eval_results:       {manifest.get('eval_results_status')}")
    return 0


# ── artifact inspect ─────────────────────────────────────────────────────────

def _cmd_artifact_inspect(args) -> int:
    from active_loop.artifacts import inspect_artifact

    try:
        summary = inspect_artifact(args.path, verify=not args.no_verify)
    except (FileNotFoundError, ValueError) as e:
        msg = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
        if args.json:
            _print_json(msg)
        else:
            print(f"error: invalid artifact: {e}", file=sys.stderr)
        return 1

    checks = summary.get("checks", {})
    failed = [k for k, v in checks.items() if v is False]
    if args.json:
        _print_json(summary)
    else:
        print(f"artifact: {summary['artifact_id']}  schema {summary['schema_version']}")
        print(f"  agent_class:   {summary['agent_class']}")
        print(f"  architecture:  {summary['architecture']}")
        print(f"  repo_commit:   {summary['repo_commit']}")
        print(f"  source_exps:   {summary['source_experiments']}")
        print(f"  runtime_claim: {summary['runtime_claim']}")
        print(f"  frozen_scorer: {summary['frozen_scorer']} (sha256 {summary['scorer_hash']})")
        print(f"  files:         {len(summary['files'])}")
        if checks:
            print(f"  checks:        {checks}")
        if summary.get("known_limitations"):
            print("  known_limitations:")
            for lim in summary["known_limitations"]:
                print(f"    - {lim}")
    return 1 if failed else 0


# ── score ────────────────────────────────────────────────────────────────────

def _cmd_score(args) -> int:
    from active_loop.artifacts import score_artifact

    try:
        if args.quick:
            result = score_artifact(args.path, seeds=(20, 21), turns=40)
        else:
            result = score_artifact(args.path)
    except (FileNotFoundError, ValueError) as e:
        result = {"status": "error", "reason": f"{type(e).__name__}: {e}", "scorable": False}

    if args.json:
        _print_json(result)
    else:
        if result.get("status") == "ok":
            print(f"score: {result.get('artifact_id')}")
            print(f"  metric (mean last-third POS): {result['metric']:.4f}")
            print(f"  improvement:                  {result['improvement']:.4f}")
            print(f"  genuine_fraction:             {result['genuine_fraction']:.4f}")
            print(f"  verdict:                      {result['verdict']}")
            print(f"  guardrails:                   {result['guardrails']}")
            print(f"  scorer_hash:                  {result['scorer_hash']}")
        else:
            print(f"score FAILED: {result.get('reason')}", file=sys.stderr)
    return 0 if result.get("status") == "ok" else 1


# ── converse ─────────────────────────────────────────────────────────────────

def _cmd_converse(args) -> int:
    from active_loop.artifacts import load_agent_from_artifact, load_manifest
    from active_loop.cli import converse as converse_mod

    # validate the artifact first (clear nonzero exit on a bad path)
    try:
        load_manifest(args.path)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: invalid artifact: {e}", file=sys.stderr)
        return 1

    def factory(seed, turns):
        return load_agent_from_artifact(args.path, which="init", seed=seed)

    if args.interactive:
        converse_mod.run_interactive(seed=args.seed, turns=args.turns, agent_factory=factory)
    else:
        turns = args.turns if args.turns != converse_mod.TURNS_DEFAULT else 60
        converse_mod.run_demo(seed=args.seed, turns=turns, agent_factory=factory)
    return 0


# ── parser ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="active-monkey",
        description="Export, inspect, score, and converse with frozen active_monkey artifacts.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # artifact <export|inspect>
    art = sub.add_parser("artifact", help="export or inspect an artifact")
    artsub = art.add_subparsers(dest="subcommand", required=True)

    exp = artsub.add_parser("export", help="export a preset artifact")
    exp.add_argument("--preset", default="affect-dyad-v0", help="artifact preset (affect-dyad-v0)")
    exp.add_argument("--out", required=True, help="output directory")
    exp.add_argument("--seed", type=int, default=20)
    exp.add_argument("--learn-turns", type=int, default=60, dest="learn_turns")
    exp.add_argument("--no-learned", action="store_true", help="skip the learned checkpoint (no pymdp)")
    exp.add_argument("--no-eval", action="store_true", help="skip bundled eval_results (no pymdp)")
    exp.add_argument("--json", action="store_true")
    exp.set_defaults(func=_cmd_artifact_export)

    ins = artsub.add_parser("inspect", help="inspect + verify an artifact")
    ins.add_argument("path")
    ins.add_argument("--json", action="store_true")
    ins.add_argument("--no-verify", action="store_true", help="skip hash verification")
    ins.set_defaults(func=_cmd_artifact_inspect)

    # score
    sc = sub.add_parser("score", help="score an artifact with the frozen scorer")
    sc.add_argument("path")
    sc.add_argument("--json", action="store_true")
    sc.add_argument("--quick", action="store_true", help="abbreviated config (2 seeds x 40 turns)")
    sc.set_defaults(func=_cmd_score)

    # converse
    cv = sub.add_parser("converse", help="converse with an artifact (demo or interactive)")
    cv.add_argument("path")
    cv.add_argument("--demo", action="store_true", help="non-interactive scripted-partner demo")
    cv.add_argument("--interactive", action="store_true", help="interactive REPL")
    cv.add_argument("--seed", type=int, default=0)
    cv.add_argument("--turns", type=int, default=300)
    cv.set_defaults(func=_cmd_converse)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc = args.func(args)
    if isinstance(rc, int) and (argv is None):
        sys.exit(rc)
    return rc or 0


if __name__ == "__main__":
    main()
