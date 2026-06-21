"""embodied.sweep_phase2p5 — the Phase-2.5 calibration sweep on the batched substrate.

Question: is there a food-field calibration where the embodied population is BOTH stable (certifies
under the FROZEN evolvability stability gate) AND competitive (density-dependent: per-capita intake
falls as N rises)? The Phase-2.5 preflight bracketed it — poor field -> collapse, rich field -> runaway
growth — so the stable+competitive equilibrium, if it exists, is in between (the prior arc's
"stability-vs-strong-competition" tension). This sweeps a (capacity, regen) grid x seeds using the
MJX-batched substrate (fast on GPU), certifies each cell, and reports the honest verdict.

Run on a GPU (RunPod) — see runpod/. Example:
  uv run --python .venv python -m embodied.sweep_phase2p5 \
    --capacities 8 12 16 22 30 --regens 0.3 0.5 --seeds 0 1 2 --horizon 300 --max-pop 256
"""
import argparse
import itertools
from pathlib import Path

from embodied.batched_population import run, BatchedPopConfig
from embodied.foodfield import FoodFieldConfig
from embodied.run_population import certify

OUT = Path(__file__).resolve().parent / "outputs" / "sweep_phase2p5.txt"


def cell_verdict(capacity, regen, seeds, founders, horizon, max_pop, bout):
    """Run a (capacity, regen) cell across seeds; aggregate the per-seed certify verdicts."""
    per_seed = []
    for s in seeds:
        r = run(BatchedPopConfig(n_founders=founders, horizon=horizon, bout_steps=bout,
                                 max_pop=max_pop, seed=s,
                                 field=FoodFieldConfig(capacity=capacity, regen=regen)))
        per_seed.append(certify(r))
    n = len(seeds)
    n_stable = sum(1 for v in per_seed if v["stable"])
    n_dd = sum(1 for v in per_seed if v["density_dependent"])
    # A cell is STABLE+COMPETITIVE iff a strict majority of seeds individually certify stable AND are
    # density-dependent (the binding conjunction; majority guards a lucky single seed).
    win = n_stable >= (n // 2 + 1) and n_dd >= (n // 2 + 1)
    return {"capacity": capacity, "regen": regen, "n": n, "n_stable": n_stable, "n_dd": n_dd,
            "n_eq": [round(v["n_eq"], 1) for v in per_seed],
            "corr": [round(v["intake_vs_N_corr"], 2) for v in per_seed], "win": win}


def main():
    p = argparse.ArgumentParser(description="Phase-2.5 stability-vs-competition calibration sweep.")
    p.add_argument("--capacities", type=float, nargs="+", default=[8, 12, 16, 22, 30])
    p.add_argument("--regens", type=float, nargs="+", default=[0.3, 0.5])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--founders", type=int, default=30)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--max-pop", type=int, default=256)
    p.add_argument("--bout", type=int, default=8)
    a = p.parse_args()

    lines = ["Phase-2.5 calibration sweep (MJX-batched substrate)",
             f"grid: capacity={a.capacities} regen={a.regens} seeds={a.seeds} "
             f"founders={a.founders} horizon={a.horizon} max_pop={a.max_pop} bout={a.bout}",
             "(a cell wins iff a majority of seeds certify STABLE under the FROZEN gate AND are density-dependent)",
             ""]
    winners = []
    for cap, reg in itertools.product(a.capacities, a.regens):
        cv = cell_verdict(cap, reg, a.seeds, a.founders, a.horizon, a.max_pop, a.bout)
        lines.append(f"  cap={cap:<5g} regen={reg:.2f}:  stable {cv['n_stable']}/{cv['n']}  "
                     f"density-dep {cv['n_dd']}/{cv['n']}  n_eq={cv['n_eq']}  corr={cv['corr']}  "
                     f"=> {'STABLE+COMPETITIVE' if cv['win'] else '-'}")
        if cv["win"]:
            winners.append(cv)
    lines.append("")
    if winners:
        cells = ", ".join(f"(cap={w['capacity']:g},regen={w['regen']:g})" for w in winners)
        lines.append(f"VERDICT: POSITIVE — {len(winners)} cell(s) STABLE + COMPETITIVE: {cells}. "
                     "A stable, density-dependent embodied population EXISTS — the wall does not hold here.")
    else:
        lines.append("VERDICT: NEGATIVE / NEW INSIGHT — no cell is both stable and density-dependent. "
                     "The stability-vs-strong-competition wall recurs on the embodied substrate "
                     "(stable only where food is too rich to compete; competitive only where it collapses).")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
