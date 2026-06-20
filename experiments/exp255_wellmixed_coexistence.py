"""experiments/exp255_wellmixed_coexistence.py — Exp 255: well-mixed Bazykin substrate.

Thin runner for the well-mixed Bazykin coexistence check (Exp 255).
The real substrate lives in ecology/wellmixed.py; this script runs the default
config across 4 seeds and writes the canonical output file.

PREDICTION: the same logistic-prey + logistic-predator ecology that collapsed
spatially (Exp 248-254c) COEXISTS stably when well-mixed (no spatial encounter
stochasticity, no energy-buffer demographic lag).
PREDECLARED FALSIFIER: if the well-mixed substrate ALSO collapses (either prey
or predator extinct at t=1500 on any seed), the Exp-250 diagnosis was wrong
(the destabilizer is NOT the spatial substrate; the Bazykin ODE itself is
unstable at these parameters).

RAW NUMBERS — controller judges.
"""
import os
import sys

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.wellmixed import WellMixedSim, WellMixedConfig  # noqa: E402

OUTPUT_PATH = os.path.join(_repo_root, "experiments", "outputs", "exp255.txt")

SEEDS = [0, 1, 2, 3]
HORIZON = 1500


def main():
    lines = []
    lines.append("=" * 80)
    lines.append("Exp 255 — Well-mixed Bazykin coexistence check. RAW — controller judges.")
    lines.append(f"Default WellMixedConfig, horizon={HORIZON}, seeds={SEEDS}")
    lines.append("=" * 80)
    lines.append(
        f"{'seed':>6}  {'t_end':>6}  {'prey_eq':>8}  {'pred_eq':>8}  {'persisted':>10}"
    )
    lines.append("-" * 50)

    all_persisted = True
    for seed in SEEDS:
        cfg = WellMixedConfig(horizon=HORIZON)
        sim = WellMixedSim(cfg, seed=seed)
        result = sim.run()

        t_end = result["t_end"]
        prey_series = result["prey_series"]
        pred_series = result["pred_series"]

        # Equilibrium estimate: mean of last 20% of series
        tail_start = max(0, len(prey_series) - int(0.2 * HORIZON))
        prey_eq = sum(prey_series[tail_start:]) / max(1, len(prey_series[tail_start:]))
        pred_eq = sum(pred_series[tail_start:]) / max(1, len(pred_series[tail_start:]))
        persisted = (not result["extinct"]) and (not result["exploded"])
        if not persisted:
            all_persisted = False

        line = (
            f"{seed:>6}  {t_end:>6}  {prey_eq:>8.1f}  {pred_eq:>8.1f}  {str(persisted):>10}"
        )
        lines.append(line)
        print(line)

    lines.append("")
    verdict = "POSITIVE" if all_persisted else "NEGATIVE"
    lines.append(
        f"VERDICT: {verdict} — "
        + ("all seeds coexist to t=1500 (bounded limit cycle)."
           if all_persisted
           else "at least one seed collapsed — falsifier triggered.")
    )
    lines.append(
        "FALSIFIER check: any extinction at t=1500 -> Exp-250 diagnosis WRONG."
    )
    lines.append(
        f"FALSIFIER {'NOT triggered' if all_persisted else 'TRIGGERED'}: "
        + ("diagnosis confirmed (spatial substrate was the destabilizer)."
           if all_persisted
           else "diagnosis refuted.")
    )

    output = "\n".join(lines) + "\n"
    print()
    print(lines[-3])
    print(lines[-2])
    print(lines[-1])

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nOutput written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
