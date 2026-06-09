"""Entry point for the parallel-track persistent-creature life runner.

Runs one or more live(steps)+save sessions for a named creature whose snapshot
lives in <state-dir>/<name>.  The runner NEVER births a creature — birth happens
in a logged experiment (see loop/directions/persistent-creature.md).

Examples:
    uv run --python .venv python run_life.py --steps 200
    uv run --python .venv python run_life.py --name mirro --steps 500 --sessions 3
    uv run --python .venv python run_life.py --steps 100 --commit
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from active_loop.creature import Creature


def _status_line(c: Creature) -> str:
    fav = c.favorite()
    conv = c.conviction()
    return (
        f"age={c.age_steps} | "
        f"map={c.map_accuracy():.3f} ({sum(l == t for l, t in zip(c.sensory_map(), c.world.cmap))}/{c.world.n_cells}) | "
        f"favorite=color-{fav} conviction={conv:.3f} | "
        f"localize={c.localize_bits():.3f} bits"
    )


def _git_commit(state_dir: Path, name: str, age: int, map_acc: float, repo_root: Path) -> None:
    """Stage creature state and commit."""
    creature_path = state_dir / name
    rel = creature_path.relative_to(repo_root)
    subprocess.run(
        ["git", "add", str(rel)],
        cwd=repo_root,
        check=True,
    )
    msg = f"life: {name} lived to age {age} (map {map_acc:.3f})"
    subprocess.run(
        ["git", "commit", "-m", msg],
        cwd=repo_root,
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Continue a persistent creature's life (parallel track)."
    )
    ap.add_argument("--name", default="mirro", help="Creature name (default: mirro)")
    ap.add_argument(
        "--steps", type=int, default=500, help="Steps per session (default: 500)"
    )
    ap.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="Number of live+save cycles (default: 1)",
    )
    ap.add_argument(
        "--state-dir",
        default="creature/state",
        help="Directory holding creature snapshots (default: creature/state)",
    )
    ap.add_argument(
        "--commit",
        action="store_true",
        help="After each save, git-commit the updated snapshot",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent
    state_dir = repo_root / args.state_dir
    creature_path = state_dir / args.name

    if not creature_path.exists():
        raise SystemExit(
            f"no such creature — birth happens in a logged experiment, "
            f"see loop/directions/persistent-creature.md\n"
            f"(looked for: {creature_path})"
        )

    print(f"Loading {args.name} from {creature_path} ...")
    c = Creature.load(creature_path)
    print(f"  loaded: {c!r}")

    for session_idx in range(1, args.sessions + 1):
        if args.sessions > 1:
            print(f"\n--- session {session_idx}/{args.sessions} ---")
        c.live(args.steps)
        c.save(creature_path)
        print(_status_line(c))

        if args.commit:
            _git_commit(state_dir, args.name, c.age_steps, c.map_accuracy(), repo_root)
            print(f"  committed: age={c.age_steps}")

    print(f"\nDone. {args.name} age={c.age_steps} state_hash={c._state_hash()[:16]}...")


if __name__ == "__main__":
    main()
