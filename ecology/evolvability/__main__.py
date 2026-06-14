"""
ecology.evolvability.__main__ — CLI entry point for the Evolvability Preflight.

Usage
-----
    python -m ecology.evolvability --config PATH [--run-id ID] [--output-dir DIR]

Exits 0 on any outcome, including a negative scientific verdict.
Exits 1 only on genuine infrastructure errors (file not found, gate crash, etc.).
"""
from __future__ import annotations

import argparse
import dataclasses as D
import sys
import traceback


def main(argv=None) -> int:
    """Entry point; returns exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m ecology.evolvability",
        description="Run an Evolvability Preflight from a config file.",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to a .json or .yaml config file (PreflightConfig).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        metavar="ID",
        help="Explicit run ID (default: UTC timestamp).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Override output_dir from the config.",
    )

    args = parser.parse_args(argv)

    try:
        from ecology.evolvability.config import load_config
        from ecology.evolvability.runner import run_preflight

        cfg = load_config(args.config)

        if args.output_dir is not None:
            cfg = D.replace(cfg, output_dir=args.output_dir)

        result = run_preflight(cfg, run_id=args.run_id)

        run_dir = result.artifact_paths.get("run_dir", "?")
        report_md = result.artifact_paths.get("report_md", "?")
        summary_json = result.artifact_paths.get("summary_json", "?")

        print(f"Preflight complete.")
        print(f"  slug            : {result.slug}")
        print(f"  run_id          : {result.run_id}")
        print(f"  aggregate_verdict: {result.aggregate_verdict}")
        print(f"  failure_reason  : {result.failure_reason}")
        print(f"  run_dir         : {run_dir}")
        print(f"  report.md       : {report_md}")
        print(f"  summary.json    : {summary_json}")

        return 0

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
