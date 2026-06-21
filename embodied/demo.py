"""embodied.demo — watch the trained creature: rollout -> render -> metrics."""
import subprocess
from pathlib import Path

from embodied.rollout import rollout
from embodied.render import render

OUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints" / "quadruped_forage" / "params"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def run(checkpoint=DEFAULT_CKPT, steps: int = 400) -> None:
    """Rollout a checkpoint, render two videos, write a metrics file.

    OUT_DIR is read at call time (not as a default parameter) so that
    monkeypatch.setattr(demo, "OUT_DIR", ...) takes effect in tests.
    """
    out = OUT_DIR  # read the live module global — must NOT be a default param
    out.mkdir(parents=True, exist_ok=True)

    traj = rollout(checkpoint, n_steps=steps)
    vids = render(traj, out)

    lines = [
        "embodied pipeline demo (Phase 1 — substrate+tooling, no exp number)",
        f"checkpoint: {checkpoint}",
        f"git: {_git_sha()}",
        f"n_steps: {len(traj.qpos)}",
        f"traj_hash: {traj.traj_hash}",
        f"final_dist_to_food: {traj.dist_to_food[-1]:.4f}",
        f"total_reached_steps: {sum(traj.reached):.0f}",
        f"thirdperson: {vids['thirdperson']}",
        f"firstperson: {vids['firstperson']}",
    ]
    text = "\n".join(lines) + "\n"
    (out / "embodied_pipeline.txt").write_text(text)
    print(text, end="")


def main() -> None:
    """CLI: python -m embodied.demo [--checkpoint PATH] [--steps N] [--train-smoke]."""
    import argparse

    p = argparse.ArgumentParser(description="Embodied demo: rollout -> render -> metrics.")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--train-smoke", action="store_true",
                   help="Train a tiny checkpoint first, then run.")
    a = p.parse_args()

    ckpt = a.checkpoint
    if a.train_smoke:
        from embodied.train import train
        ckpt = train(num_timesteps=2048, seed=0, out_dir=Path(a.checkpoint).parent)

    run(checkpoint=ckpt, steps=a.steps)


if __name__ == "__main__":
    main()
