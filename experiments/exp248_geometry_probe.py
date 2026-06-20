"""experiments/exp248_geometry_probe.py — Exp 248 Rung 1: Static-geometry expressibility probe.

CHEAP EXPRESSIBILITY PROBE — RAW NUMBERS ONLY. NO VERDICT EMITTED.
The controller reads these numbers and decides go/abort.

Design:
  - STATIC, NO-CO-EVOLUTION, NO-INVASION measurement.
  - 6 cells: prey_speed ∈ {1.0, 1.1} × predator_bracket ∈ {1.2, 1.4, 1.6}
  - Both sides FROZEN — no mutation drift:
      freeze_prey_speed=True   → prey speed copies verbatim (engine.py L1211:
                                   `_mut_speed = (role=="predator" and mutate_predator_speed)
                                                  or (role=="prey" and not freeze_prey_speed)`
                                   → False for prey when freeze_prey_speed=True)
      mutate_predator_speed=False → predator speed also copies verbatim (same expression,
                                   False for predator when mutate_predator_speed=False)
  - R=10 distinct seeds per cell; horizon=300 steps
  - Founders: 30 prey + 6 predators (in a 12×12 arena, ~0.2 predators/prey ratio)

Metrics per cell:
  1. Per-capita capture hazard = total_captures / total_prey_alive_steps
     where total_prey_alive_steps = Σ_t (# prey alive at step t).
     Denominator normalizes out population size and extinction timing.
     Captures counted via death events with cause="predation".
  2. Capture survival = 1 − hazard (fraction of prey-steps NOT ending in capture).
  3. Mean post-move predator→nearest-prey distance, averaged over all predator-steps
     (using pos_cont positions).

Headline output: marginal capture-survival delta = survival(s_prey=1.1) − survival(s_prey=1.0)
per predator bracket. Positive delta = faster prey escape better (expressibility signal).
Delta ≈ 0 = speed bump not expressed (CAN'T-POSE signal).

Freeze confirmation (see comment below in build_config): engine.py L1209-1213 computes
  _mut_speed = ((role=="predator" and mutate_predator_speed)
               or (role=="prey" and not freeze_prey_speed))
With mutate_predator_speed=False and freeze_prey_speed=True:
  - predators: False and False → _mut_speed=False → locomotor_speed COPIED VERBATIM
  - prey: False or not True → False → locomotor_speed COPIED VERBATIM
Both speeds are constant throughout the run (confirmed in mutate() in genotype.py:
LOCOMOTION_CONTINUOUS_TRAITS skips the rng.normal draw when mutate_continuous_locomotion=False,
so NO rng is consumed for speed and the genotype value is the parent's value).
"""
import sys
import os
import dataclasses as D

import numpy as np

# Add repo root to path (needed when run as a script, not as a module)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.engine import Ecology, EcologyConfig
from ecology.genotype import Genotype, founder
from ecology.scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Probe parameters (stated up front)
# ---------------------------------------------------------------------------
PREY_SPEEDS = [1.0, 1.1, 1.5]   # 1.1 = LOCAL marginal step (the invasion gradient);
                                # 1.5 = WIDE-contrast sensitivity check (can the instrument
                                #       detect an escape benefit when a large one exists?)
PRED_SPEEDS = [1.2, 1.4, 1.6]
R = 30            # replicates per cell (controller bumped 10→30 for delta-CI power)
HORIZON = 300     # steps per run
N_PREY = 30       # founder prey count
N_PRED = 6        # founder predator count


def build_config(s_prey: float, s_pred: float) -> EcologyConfig:
    """Build a frozen monomorphic two-role config for the given speed cell.

    Freeze confirmation (see module docstring):
      - freeze_prey_speed=True  → prey locomotor_speed never mutated
      - mutate_predator_speed=False → predator locomotor_speed never mutated
    Both enforced via engine.py L1209-1213 → mutate() LOCOMOTION_CONTINUOUS_TRAITS guard.
    """
    prey_geno = D.replace(founder(), locomotor_speed=s_prey, role="prey")
    pred_geno = D.replace(founder(), locomotor_speed=s_pred, role="predator")

    return D.replace(
        SCENARIOS["balanced"],
        # --- continuous world (same flags as _pred_cfg) ---
        enable_continuous_locomotion=True,
        continuous_layout="bump",
        continuous_dt=1.0,
        speed_cost_floor=0.0,
        speed_cost_slope=0.0,
        # --- depletion-aware intake (same as _pred_cfg) ---
        enable_continuous_depletion_intake=True,
        # --- predation gate ON ---
        enable_predation=True,
        # --- FREEZE BOTH SIDES (no co-evolution, no drift) ---
        freeze_prey_speed=True,
        mutate_predator_speed=False,
        # --- mutation rate: still ON for non-speed traits (lifelike dynamics) ---
        mutation_rate=0.05,
        # --- horizon ---
        horizon=HORIZON,
        # --- founder cohorts ---
        founder_mix=((prey_geno, N_PREY), (pred_geno, N_PRED)),
        # give predators reasonable start energy
        pred_start_energy_frac=0.75,
    )


def run_cell(s_prey: float, s_pred: float, seeds: list[int]) -> dict:
    """Run R replicates for a single (s_prey, s_pred) cell.

    Returns per-replicate and aggregated metrics.

    Metrics:
      - total_captures: predation deaths across the run.
      - total_prey_alive_steps: Σ_t prey alive at step t (denominator for hazard).
      - hazard: total_captures / total_prey_alive_steps.
      - survival: 1 - hazard.
      - mean_pred_prey_dist: mean post-move predator→nearest-prey distance
          averaged over all predator-steps with at least one prey alive.
    """
    cfg = build_config(s_prey, s_pred)

    per_rep = []
    for seed in seeds:
        eco = Ecology(cfg, seed=seed)

        total_captures = 0
        total_prey_alive_steps = 0
        pred_dist_sum = 0.0
        pred_dist_count = 0

        # Step-by-step instrumentation
        while True:
            if not eco.has_alive():
                break
            if eco.exploded:
                break
            if eco.t >= cfg.horizon:
                break

            # Count prey alive at START of this step (before step() modifies state)
            snap = eco.alive_snapshot()
            prey_alive = [c for c in snap if c.genotype.role == "prey"]
            preds_alive = [c for c in snap if c.genotype.role == "predator"]

            n_prey_this_step = len(prey_alive)
            total_prey_alive_steps += n_prey_this_step

            # Measure predator→nearest-prey distance (post-move from previous step;
            # at t=0 this is founder positions — included for completeness)
            if preds_alive and prey_alive:
                prey_positions = [
                    c.phenotype.pos_cont for c in prey_alive
                    if c.phenotype.pos_cont is not None
                ]
                for pred in preds_alive:
                    if pred.phenotype.pos_cont is None:
                        continue
                    px, py = pred.phenotype.pos_cont
                    min_d = None
                    for (qx, qy) in prey_positions:
                        d = ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5
                        if min_d is None or d < min_d:
                            min_d = d
                    if min_d is not None:
                        pred_dist_sum += min_d
                        pred_dist_count += 1

            # Run one step
            eco.step()

        # Count captures from events
        for ev in eco.events:
            if ev["event_type"] == "death" and ev.get("details", {}).get("cause") == "predation":
                total_captures += 1

        hazard = total_captures / total_prey_alive_steps if total_prey_alive_steps > 0 else float("nan")
        survival = 1.0 - hazard if total_prey_alive_steps > 0 else float("nan")
        mean_dist = pred_dist_sum / pred_dist_count if pred_dist_count > 0 else float("nan")

        per_rep.append({
            "seed": seed,
            "total_captures": total_captures,
            "total_prey_alive_steps": total_prey_alive_steps,
            "hazard": hazard,
            "survival": survival,
            "mean_pred_prey_dist": mean_dist,
        })

    # Aggregate (skip NaN replicates with no exposure)
    valid_surv = [r["survival"] for r in per_rep if not (r["survival"] != r["survival"])]
    valid_dist = [r["mean_pred_prey_dist"] for r in per_rep if not (r["mean_pred_prey_dist"] != r["mean_pred_prey_dist"])]

    mean_survival = float(np.mean(valid_surv)) if valid_surv else float("nan")
    std_survival = float(np.std(valid_surv)) if len(valid_surv) > 1 else float("nan")
    mean_hazard = 1.0 - mean_survival if not (mean_survival != mean_survival) else float("nan")
    mean_dist_agg = float(np.mean(valid_dist)) if valid_dist else float("nan")
    total_prey_steps_all = sum(r["total_prey_alive_steps"] for r in per_rep)
    total_caps_all = sum(r["total_captures"] for r in per_rep)

    return {
        "s_prey": s_prey,
        "s_pred": s_pred,
        "per_rep": per_rep,
        "mean_survival": mean_survival,
        "std_survival": std_survival,
        "mean_hazard": mean_hazard,
        "mean_pred_prey_dist": mean_dist_agg,
        "total_prey_alive_steps": total_prey_steps_all,
        "total_captures": total_caps_all,
    }


def _surv_array(cell: dict) -> np.ndarray:
    """Per-seed survival values for a cell, NaN-filtered (NaN = zero-exposure seed)."""
    vals = [r["survival"] for r in cell["per_rep"] if r["survival"] == r["survival"]]
    return np.asarray(vals, dtype=float)


def delta_stats(cell_fast: dict, cell_base: dict) -> dict:
    """Across-seed survival delta = mean(fast) − mean(base), with Welch SE and t.

    The unit of replication is the SEED (captures are correlated within a run), so the
    standard error is across-seed, NOT across prey-steps. t = delta / SE; |t| >~ 2 is the
    rough significance bar. Positive delta ⇒ faster prey survive better (expressibility).
    """
    a = _surv_array(cell_fast)
    b = _surv_array(cell_base)
    if len(a) < 2 or len(b) < 2:
        return {"delta": float("nan"), "se": float("nan"), "t": float("nan"),
                "na": len(a), "nb": len(b)}
    delta = float(a.mean() - b.mean())
    # sample variance (ddof=1), Welch combination
    se = float((a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 0.5)
    t = delta / se if se > 0 else float("nan")
    return {"delta": delta, "se": se, "t": t, "na": len(a), "nb": len(b)}


def main():
    seeds = list(range(R))  # seeds 0..R-1

    print(f"Exp 248 geometry probe | R={R} seeds | horizon={HORIZON} | "
          f"prey={N_PREY} + pred={N_PRED} founders")
    print(f"Prey speeds: {PREY_SPEEDS}  |  Predator brackets: {PRED_SPEEDS}")
    print()

    # Collect results for all 6 cells
    results: dict[tuple, dict] = {}
    for s_pred in PRED_SPEEDS:
        for s_prey in PREY_SPEEDS:
            print(f"  Running cell s_prey={s_prey} s_pred={s_pred} ...", end=" ", flush=True)
            r = run_cell(s_prey, s_pred, seeds)
            results[(s_prey, s_pred)] = r
            print(f"survival={r['mean_survival']:.6f} ± {r['std_survival']:.6f}  "
                  f"hazard={r['mean_hazard']:.6f}  "
                  f"pred→prey_dist={r['mean_pred_prey_dist']:.4f}  "
                  f"prey-steps={r['total_prey_alive_steps']}")

    # Build output
    lines = []
    lines.append("=" * 78)
    lines.append("Exp 248 — Red Queen predator-prey: STATIC GEOMETRY EXPRESSIBILITY PROBE")
    lines.append("RAW NUMBERS — NO VERDICT — controller judges go/abort")
    lines.append("=" * 78)
    lines.append(f"R={R} replicates | horizon={HORIZON} | "
                 f"founders: {N_PREY} prey + {N_PRED} predators")
    lines.append(f"Prey speeds: {PREY_SPEEDS}  |  Predator brackets: {PRED_SPEEDS}")
    lines.append("")
    lines.append("Metrics:")
    lines.append("  capture_hazard    = total_captures / total_prey_alive_steps")
    lines.append("  capture_survival  = 1 − capture_hazard")
    lines.append("  prey_alive_steps  = Σ_t (# prey alive at step t)  [exposure denominator]")
    lines.append("  pred→prey dist    = mean over predator-steps of (predator→nearest-prey dist)")
    lines.append("  Captures counted via events with event_type='death', cause='predation'")
    lines.append("")

    # Full table
    lines.append("-" * 78)
    lines.append(f"{'s_prey':>7} {'s_pred':>7} {'survival':>12} {'±std':>8} "
                 f"{'hazard':>10} {'pred→dist':>10} {'prey-steps':>11} {'captures':>9}")
    lines.append("-" * 78)
    for s_pred in PRED_SPEEDS:
        for s_prey in PREY_SPEEDS:
            r = results[(s_prey, s_pred)]
            lines.append(
                f"{s_prey:>7.1f} {s_pred:>7.1f} "
                f"{r['mean_survival']:>12.6f} {r['std_survival']:>8.6f} "
                f"{r['mean_hazard']:>10.6f} {r['mean_pred_prey_dist']:>10.4f} "
                f"{r['total_prey_alive_steps']:>11d} {r['total_captures']:>9d}"
            )
        lines.append("")

    # Delta table (the headline expressibility signal) — with across-seed SE + t.
    # LOCAL contrast (1.1 − 1.0) is the invasion-from-rarity gradient (the gate).
    # WIDE contrast (1.5 − 1.0) is the instrument-sensitivity sanity check.
    lines.append("=" * 78)
    lines.append("HEADLINE: marginal capture-survival delta = survival(s_prey) − survival(1.0)")
    lines.append("          delta>0 ⇒ faster prey survive better (escape EXPRESSED).")
    lines.append("          se,t are ACROSS-SEED (unit of replication = run); |t|>~2 ≈ significant.")
    lines.append("          LOCAL (1.1−1.0) = the invasion gradient (decides the gate).")
    lines.append("          WIDE  (1.5−1.0) = sensitivity check (can the probe see a BIG escape?).")
    lines.append("-" * 78)
    lines.append(f"{'s_pred':>7} {'contrast':>10} {'surv_base':>10} {'surv_fast':>10} "
                 f"{'delta':>11} {'se':>9} {'t':>8}")
    lines.append("-" * 78)
    for s_pred in PRED_SPEEDS:
        base = results[(1.0, s_pred)]
        for fast_speed, label in ((1.1, "1.1-1.0"), (1.5, "1.5-1.0")):
            fast = results[(fast_speed, s_pred)]
            st = delta_stats(fast, base)
            lines.append(
                f"{s_pred:>7.1f} {label:>10} "
                f"{base['mean_survival']:>10.6f} {fast['mean_survival']:>10.6f} "
                f"{st['delta']:>+11.6f} {st['se']:>9.6f} {st['t']:>8.3f}"
            )
        lines.append("")

    # Prey-alive-step exposure summary (power check)
    lines.append("Exposure counts (total prey-alive-steps across all R seeds per cell):")
    hdr = f"{'s_pred':>7}" + "".join(f"{'s_prey='+str(s):>14}" for s in PREY_SPEEDS)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for s_pred in PRED_SPEEDS:
        row = f"{s_pred:>7.1f}" + "".join(
            f"{results[(s, s_pred)]['total_prey_alive_steps']:>14d}" for s in PREY_SPEEDS)
        lines.append(row)
    lines.append("")

    # Per-replicate detail for transparency
    lines.append("=" * 78)
    lines.append("Per-replicate detail:")
    lines.append("-" * 78)
    for s_pred in PRED_SPEEDS:
        for s_prey in PREY_SPEEDS:
            r = results[(s_prey, s_pred)]
            lines.append(f"  s_prey={s_prey} s_pred={s_pred}:")
            for rep in r["per_rep"]:
                lines.append(
                    f"    seed={rep['seed']:2d}  survival={rep['survival']:.6f}  "
                    f"hazard={rep['hazard']:.6f}  captures={rep['total_captures']:4d}  "
                    f"prey-steps={rep['total_prey_alive_steps']:6d}  "
                    f"pred→dist={rep['mean_pred_prey_dist']:.4f}"
                )
        lines.append("")

    output_text = "\n".join(lines)

    # Print to stdout
    print()
    print(output_text)

    # Write to file
    out_dir = os.path.join(_repo_root, "experiments", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp248_geometry.txt")
    with open(out_path, "w") as f:
        f.write(output_text)
        f.write("\n")
    print(f"\n[Output written to {out_path}]")


if __name__ == "__main__":
    main()
