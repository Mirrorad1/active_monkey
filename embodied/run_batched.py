"""embodied.run_batched — CLI to run the batched substrate (CPU dev / GPU sweep) + certify."""
import argparse
from pathlib import Path

import numpy as np

from embodied.batched_population import run, BatchedPopConfig
from embodied.foodfield import FoodFieldConfig
from embodied.run_population import certify

OUT = Path(__file__).resolve().parent / "outputs" / "embodied_batched.txt"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run the batched embodied substrate (CPU dev / GPU sweep) + certify each seed",
    )
    p.add_argument("--founders", type=int, default=30,
                   help="Number of founding creatures (default: 30)")
    p.add_argument("--horizon", type=int, default=200,
                   help="Steps per run (default: 200)")
    p.add_argument("--max-pop", type=int, default=256,
                   help="Max population / buffer size (default: 256)")
    p.add_argument("--bout", type=int, default=8,
                   help="Physics bout steps per simulation step (default: 8)")
    p.add_argument("--capacity", type=float, default=5.0,
                   help="Food field per-cell capacity (default: 5.0)")
    p.add_argument("--regen", type=float, default=0.2,
                   help="Food field regeneration rate (default: 0.2)")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="RNG seeds to run (default: 0 1 2)")
    a = p.parse_args()

    lines = [
        f"batched substrate run  founders={a.founders} horizon={a.horizon}"
        f" max_pop={a.max_pop} bout={a.bout} cap={a.capacity} regen={a.regen}",
    ]

    for s in a.seeds:
        r = run(BatchedPopConfig(
            n_founders=a.founders,
            horizon=a.horizon,
            bout_steps=a.bout,
            max_pop=a.max_pop,
            seed=s,
            field=FoodFieldConfig(capacity=a.capacity, regen=a.regen),
        ))
        v = certify(r)
        lines += [
            f"--- seed {s} ---",
            f"  N min/max/mean: {min(r.n_series)}/{max(r.n_series)}/{np.mean(r.n_series):.1f}",
            f"  births/deaths/capped: {r.births}/{r.deaths}/{r.capped}",
            f"  events_hash: {r.events_hash}",
            f"  certify: {v}",
        ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
