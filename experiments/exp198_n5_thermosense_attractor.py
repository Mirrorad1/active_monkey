"""Exp 198 — N5: the thermosense intensity ATTRACTOR + clean de-novo emergence
(pre-registered in loop/directions/population-ecology.md, commit 5b09427, BEFORE any data).

Hypothesis: under temperature the thermosense organ EMERGES de novo from intensity 0 to a low
gene-pool equilibrium (clean heritable selection — no survivor-bias confound, since a trait nobody
has cannot be survivor-biased), reaching the SAME convergent attractor a seeded-high founder (0.20)
RELAXES down to — a low, near-activation-threshold equilibrium that never runs away to a functional
organ (the fitness valley + neutral-sub-threshold drift cap it); without temperature the equilibrium
is lower. This resolves Exp 197's survivor-bias confound (which used a seeded founder).

Design: 2x2 temperature x seeding, fresh seeds {13,14,15,16,17}, horizon 12000, the Exp 197
thermosense regime (floor 0.02, active_threshold 0.05, noise 0.2, stress 1.0, tol_cost 1.5, comfort
amp 0.4/period 1500, thermal_weight 2.5; founder tolerance 0.10):
  A = temperature ON, founder intensity 0.0 (de-novo); B = ON, founder 0.20 (seeded);
  C = temperature OFF, founder 0.0; D = OFF, founder 0.20.
Metric = NEWBORN (gene-pool) mean thermosense intensity; end = mean over creatures born in [10000,12000].

Pilots (disclosed, per L7): 2-seed pilots {100,101} (deleted) gave end newborn intensity A 0.040-0.044,
B 0.048-0.050, C 0.027-0.030, D 0.033-0.035 (A>C ~0.013; A~B, C~D within ~0.01; all <0.05). Params +
thresholds FIXED; final on FRESH seeds {13-17}.

Predictions if TRUE (5 fresh seeds):
  P1 determinism: same seed -> identical event hash, both code paths.
  P2 validity: all 4 arms persist to horizon (no extinction/explosion), >=4/5 seeds.
  P3 (CORE) clean de-novo heritable temperature effect: A_end - C_end >= 0.008 AND A_end > 0.02, >=4/5
     seeds (both start at 0 -> PURE heritable, no survivor bias).
  P4 attractor/convergence: |A-B| <= 0.02 AND |C-D| <= 0.02, >=4/5 seeds.
  P5 primitive ceiling: all four arms' end newborn intensity < 0.10, 5/5 seeds (no functional organ).

Falsifiers: F1 non-determinism -> NEGATIVE. F2 (CORE) A-C gap < 0.008 in a majority -> NEGATIVE (Exp 197
heritable signal was survivor bias). F3 |A-B|>0.02 or |C-D|>0.02 in a majority -> MIXED (not an
attractor). F4 any arm end > 0.15 -> reinterpret (functional organ). F5 arms fail to persist -> INCONCLUSIVE.

Verdict rule: POSITIVE iff P1^P2^P3^P4^P5; P3 fail -> NEGATIVE; P4 fail -> MIXED; P5 fail (runaway) ->
reinterpret. Repo token POSITIVE/NEGATIVE/MIXED. The CORE is the survivor-bias-FREE heritable
temperature effect (A vs C, both from 0); the attractor + primitive ceiling characterize why a costed
sensor evolves only to a primitive level. Params fixed pre-run; the effect is small (~0.013) but clean.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is on sys.path when script is run directly.
_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

# ---------------------------------------------------------------------------
# Experiment constants (pre-registered / fixed from pilots {100,101})
# ---------------------------------------------------------------------------
SEEDS = [13, 14, 15, 16, 17]
HORIZON = 12000
MAX_POP = 5000
CHECKPOINT_STRIDE = 1000
NEWBORN_WINDOW_START = 10000
NEWBORN_WINDOW_END = HORIZON
ACTIVE_THRESHOLD = 0.05

# Fixed regime from pilots
REG = dict(
    enable_thermosense=True,
    temperature_stress_scale=1.0,
    tolerance_cost_scale=1.5,
    comfort_amplitude=0.4,
    comfort_period=1500.0,
    temperature_comfort=0.5,
    thermal_avoidance_weight=2.5,
    thermosense_upkeep_floor=0.02,
    thermosense_active_threshold=0.05,
    thermosense_noise_base=0.2,
)


# ---------------------------------------------------------------------------
# Founder + config factories (importable for unit tests)
# ---------------------------------------------------------------------------

def make_founder(intensity: float) -> Any:
    """Return a founder genotype with given thermosense_intensity and tolerance 0.10."""
    return dataclasses.replace(
        FOUNDER,
        thermosense_intensity=intensity,
        temperature_tolerance=0.10,
    )


def make_cfg(temp_on: bool, intensity: float) -> EcologyConfig:
    """Return EcologyConfig for arm with temperature flag and founder intensity."""
    return dataclasses.replace(
        SCENARIOS["balanced"],
        horizon=HORIZON,
        max_population=MAX_POP,
        founder=make_founder(intensity),
        enable_temperature=temp_on,
        **REG,
    )


# Arm configs (pre-registered)
def make_arm_A() -> EcologyConfig:
    return make_cfg(True, 0.0)


def make_arm_B() -> EcologyConfig:
    return make_cfg(True, 0.20)


def make_arm_C() -> EcologyConfig:
    return make_cfg(False, 0.0)


def make_arm_D() -> EcologyConfig:
    return make_cfg(False, 0.20)


# ---------------------------------------------------------------------------
# Newborn window metric (importable for unit tests)
# ---------------------------------------------------------------------------

def newborn_mean_intensity_in_window(
    creatures: list,
    window_start: int,
    window_end: int,
) -> float:
    """Return mean thermosense_intensity for non-founder creatures with
    birth_t in [window_start, window_end].

    Returns NaN if no newborns in the window.
    This is DISTINCT from a living snapshot — it reads the birth cohort
    from the gene pool regardless of whether creatures are still alive.
    """
    newborns = [
        c for c in creatures
        if c.parent_id is not None
        and window_start <= c.phenotype.birth_t <= window_end
    ]
    if not newborns:
        return float("nan")
    intensities = [c.genotype.thermosense_intensity for c in newborns]
    return float(np.mean(intensities))


# ---------------------------------------------------------------------------
# Core runner: one arm x one seed, with checkpoint sampling
# ---------------------------------------------------------------------------

def run_arm(cfg: EcologyConfig, seed: int) -> dict[str, Any]:
    """Run one arm with one seed; return per-checkpoint trajectory + end summary."""
    eco = Ecology(cfg, seed=seed)

    trajectory: list[dict[str, Any]] = []
    checkpoints = list(range(CHECKPOINT_STRIDE, HORIZON + 1, CHECKPOINT_STRIDE))

    while eco.t < HORIZON and not eco.exploded:
        eco.step()

        if eco.t in checkpoints:
            alive = eco._alive()
            # Newborn mean intensity over creatures born in (t - CHECKPOINT_STRIDE, t]
            window_start = eco.t - CHECKPOINT_STRIDE
            window_end = eco.t
            nb_intensity = newborn_mean_intensity_in_window(
                eco._creatures, window_start, window_end
            )
            trajectory.append({
                "t": eco.t,
                "pop": len(alive),
                "newborn_mean_intensity": nb_intensity,
            })

        if not eco._alive():
            break

    alive = eco._alive()
    # End metric: newborn mean intensity for creatures born in [10000, 12000]
    end_nb_intensity = newborn_mean_intensity_in_window(
        eco._creatures, NEWBORN_WINDOW_START, NEWBORN_WINDOW_END
    )
    reached_horizon = eco.t >= HORIZON and not eco.exploded and len(alive) > 0

    summary = {
        "seed": seed,
        "steps_run": eco.t,
        "reached_horizon": reached_horizon,
        "extinct": len(alive) == 0,
        "exploded": eco.exploded,
        "final_pop": len(alive),
        "end_newborn_intensity": end_nb_intensity,
        "events_hash": eco.events_hash(),
    }
    return {"trajectory": trajectory, "summary": summary}


# ---------------------------------------------------------------------------
# Determinism check (P1): rerun seed 13 for arms A and C
# ---------------------------------------------------------------------------

def check_p1_determinism(
    hash_A_seed13: str,
    hash_C_seed13: str,
) -> bool:
    """Rerun seed 13 for arms A (temp-ON) and C (temp-OFF); confirm hashes match."""
    r2_A = run_arm(make_arm_A(), seed=13)
    r2_C = run_arm(make_arm_C(), seed=13)
    match_A = r2_A["summary"]["events_hash"] == hash_A_seed13
    match_C = r2_C["summary"]["events_hash"] == hash_C_seed13
    return match_A and match_C


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    out_dir = _REPO / "experiments" / "outputs" / "exp198_n5_thermosense_attractor"
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_defs = [
        ("A", make_arm_A),
        ("B", make_arm_B),
        ("C", make_arm_C),
        ("D", make_arm_D),
    ]

    # results[arm_label][seed] = run result dict
    results: dict[str, dict[int, dict[str, Any]]] = {label: {} for label, _ in arm_defs}

    print("Exp 198: running 4 arms x 5 seeds (horizon=12000, max_pop=5000) ...")
    for arm_label, arm_factory in arm_defs:
        cfg = arm_factory()
        for seed in SEEDS:
            print(f"  arm {arm_label} seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = run_arm(cfg, seed=seed)
            dt = time.time() - t0
            pop = result["summary"]["final_pop"]
            reached = result["summary"]["reached_horizon"]
            end_nb = result["summary"]["end_newborn_intensity"]
            print(f"done in {dt:.1f}s  pop={pop}  reached={reached}  end_nb={end_nb:.4f}")
            results[arm_label][seed] = result

            # Persist per-arm/seed trajectory
            traj_path = out_dir / f"traj_arm{arm_label}_seed{seed}.json"
            with open(traj_path, "w") as f:
                json.dump({
                    "arm": arm_label,
                    "seed": seed,
                    "trajectory": result["trajectory"],
                    "summary": result["summary"],
                }, f, indent=2)

    # ---------------------------------------------------------------------------
    # P1: determinism check (seed 13, arms A and C)
    # ---------------------------------------------------------------------------
    print("\nP1 determinism check (seed 13, arms A and C) ...", end=" ", flush=True)
    hash_A_13 = results["A"][13]["summary"]["events_hash"]
    hash_C_13 = results["C"][13]["summary"]["events_hash"]
    p1 = check_p1_determinism(hash_A_13, hash_C_13)
    print(f"{'PASS' if p1 else 'FAIL'}")

    # ---------------------------------------------------------------------------
    # Collect per-seed end metrics
    # ---------------------------------------------------------------------------
    # For each seed: all 4 arms must reach horizon for the seed to be "valid"
    A_end: dict[int, float] = {}
    B_end: dict[int, float] = {}
    C_end: dict[int, float] = {}
    D_end: dict[int, float] = {}
    valid_seeds: list[int] = []

    for seed in SEEDS:
        a_s = results["A"][seed]["summary"]
        b_s = results["B"][seed]["summary"]
        c_s = results["C"][seed]["summary"]
        d_s = results["D"][seed]["summary"]
        A_end[seed] = a_s["end_newborn_intensity"]
        B_end[seed] = b_s["end_newborn_intensity"]
        C_end[seed] = c_s["end_newborn_intensity"]
        D_end[seed] = d_s["end_newborn_intensity"]
        all_valid = (
            a_s["reached_horizon"]
            and b_s["reached_horizon"]
            and c_s["reached_horizon"]
            and d_s["reached_horizon"]
        )
        if all_valid:
            valid_seeds.append(seed)

    # ---------------------------------------------------------------------------
    # Verdict logic (EXACTLY as pre-registered)
    # ---------------------------------------------------------------------------
    A = A_end
    B = B_end
    C = C_end
    Dn = D_end

    P2 = len(valid_seeds) >= 4
    P3 = sum(
        (A[s] - C[s] >= 0.008) and (A[s] > 0.02)
        for s in valid_seeds
    ) >= 4
    P4 = sum(
        (abs(A[s] - B[s]) <= 0.02) and (abs(C[s] - Dn[s]) <= 0.02)
        for s in valid_seeds
    ) >= 4
    P5 = all(
        v < 0.10
        for s in valid_seeds
        for v in (A[s], B[s], C[s], Dn[s])
    )
    runaway = any(
        v > 0.15
        for s in valid_seeds
        for v in (A[s], B[s], C[s], Dn[s])
    )

    if not p1:
        verdict, token = "NEGATIVE", "NEGATIVE"
    elif not P2:
        verdict, token = "INCONCLUSIVE", "MIXED"
    elif not P3:
        verdict, token = "NEGATIVE", "NEGATIVE"
    elif runaway or (not P5):
        verdict, token = "REINTERPRET-RUNAWAY", "MIXED"
    elif not P4:
        verdict, token = "MIXED", "MIXED"
    else:
        verdict, token = "POSITIVE", "POSITIVE"

    # ---------------------------------------------------------------------------
    # Build SUMMARY block
    # ---------------------------------------------------------------------------
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("EXP 198 SUMMARY — N5: thermosense intensity ATTRACTOR + de-novo emergence")
    lines.append("=" * 72)
    lines.append(f"Seeds: {SEEDS}  |  Valid seeds: {valid_seeds}  |  Horizon: {HORIZON}")
    lines.append("")

    lines.append("Per-seed end newborn thermosense_intensity (born in [10000,12000]):")
    lines.append(f"  {'Seed':>6}  {'A(ON/0.0)':>10}  {'B(ON/0.20)':>10}  {'C(OFF/0.0)':>10}  {'D(OFF/0.20)':>10}  {'A-C gap':>8}  {'|A-B|':>7}  {'|C-D|':>7}  {'valid?':>7}")
    for seed in SEEDS:
        a = A_end[seed]
        b = B_end[seed]
        c = C_end[seed]
        d = Dn[seed]
        gap_ac = a - c
        ab = abs(a - b)
        cd = abs(c - d)
        is_v = seed in valid_seeds
        lines.append(
            f"  {seed:>6}  {a:>10.4f}  {b:>10.4f}  {c:>10.4f}  {d:>10.4f}  {gap_ac:>8.4f}  {ab:>7.4f}  {cd:>7.4f}  {'YES' if is_v else 'NO':>7}"
        )
    lines.append("")

    # Final populations
    lines.append("Final populations per arm/seed:")
    lines.append(f"  {'Seed':>6}  {'A':>8}  {'B':>8}  {'C':>8}  {'D':>8}")
    for seed in SEEDS:
        pops = {arm: results[arm][seed]["summary"]["final_pop"] for arm in ("A", "B", "C", "D")}
        lines.append(f"  {seed:>6}  {pops['A']:>8}  {pops['B']:>8}  {pops['C']:>8}  {pops['D']:>8}")
    lines.append("")

    # Representative trajectory: seed 13, arms A vs C newborn intensity over time
    lines.append("Representative trajectory (seed 13): arm A vs C newborn mean intensity over time:")
    lines.append(f"  {'t':>6}  {'A newborn':>12}  {'C newborn':>12}  {'A pop':>8}  {'C pop':>8}")
    traj_A13 = results["A"][13]["trajectory"]
    traj_C13 = results["C"][13]["trajectory"]
    # Align by t
    c13_by_t = {snap["t"]: snap for snap in traj_C13}
    for snap_a in traj_A13:
        t = snap_a["t"]
        snap_c = c13_by_t.get(t, {})
        nb_a = snap_a["newborn_mean_intensity"]
        nb_c = snap_c.get("newborn_mean_intensity", float("nan"))
        pop_a = snap_a["pop"]
        pop_c = snap_c.get("pop", 0)
        nb_a_str = f"{nb_a:.4f}" if not (isinstance(nb_a, float) and nb_a != nb_a) else "  NaN"
        nb_c_str = f"{nb_c:.4f}" if not (isinstance(nb_c, float) and nb_c != nb_c) else "  NaN"
        lines.append(f"  {t:>6}  {nb_a_str:>12}  {nb_c_str:>12}  {pop_a:>8}  {pop_c:>8}")
    lines.append("")

    # Predictions
    lines.append("Predictions:")
    lines.append(f"  P1 (determinism, seed 13 arms A+C):  {'PASS' if p1 else 'FAIL'}")
    lines.append(f"  P2 (validity >=4/5 seeds all arms):  {'PASS' if P2 else 'FAIL'}  (valid seeds: {valid_seeds})")
    lines.append(f"  P3 (CORE de-novo A-C>=0.008, A>0.02, >=4/5 valid): {'PASS' if P3 else 'FAIL'}")
    lines.append(f"  P4 (attractor |A-B|<=0.02 & |C-D|<=0.02, >=4/5):  {'PASS' if P4 else 'FAIL'}")
    lines.append(f"  P5 (ceiling all arms <0.10, 5/5):   {'PASS' if P5 else 'FAIL'}")
    lines.append(f"  Runaway flag (any arm >0.15):        {runaway}")
    lines.append("")
    lines.append(f"VERDICT: {verdict} (repo token: {token})")
    lines.append("=" * 72)

    runtime = time.time() - t_start
    lines.append(f"runtime: {runtime:.1f}s")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    # Write exp198.txt
    txt_path = _REPO / "experiments" / "outputs" / "exp198.txt"
    with open(txt_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary written to {txt_path}")

    # ---------------------------------------------------------------------------
    # Write verdict.json
    # ---------------------------------------------------------------------------
    per_seed_data: dict[str, Any] = {}
    for seed in SEEDS:
        a_s = results["A"][seed]["summary"]
        b_s = results["B"][seed]["summary"]
        c_s = results["C"][seed]["summary"]
        d_s = results["D"][seed]["summary"]
        a = A_end[seed]
        b = B_end[seed]
        c = C_end[seed]
        d = Dn[seed]
        per_seed_data[str(seed)] = {
            "A_end": a,
            "B_end": b,
            "C_end": c,
            "D_end": d,
            "A_minus_C": a - c,
            "abs_A_minus_B": abs(a - b),
            "abs_C_minus_D": abs(c - d),
            "valid": seed in valid_seeds,
            "A_reached_horizon": a_s["reached_horizon"],
            "B_reached_horizon": b_s["reached_horizon"],
            "C_reached_horizon": c_s["reached_horizon"],
            "D_reached_horizon": d_s["reached_horizon"],
            "A_final_pop": a_s["final_pop"],
            "B_final_pop": b_s["final_pop"],
            "C_final_pop": c_s["final_pop"],
            "D_final_pop": d_s["final_pop"],
            "P3_this_seed": (a - c >= 0.008) and (a > 0.02) if seed in valid_seeds else None,
            "P4_this_seed": (abs(a - b) <= 0.02) and (abs(c - d) <= 0.02) if seed in valid_seeds else None,
            "P5_this_seed": all(v < 0.10 for v in (a, b, c, d)) if seed in valid_seeds else None,
        }

    verdict_json = {
        "experiment": "exp198_n5_thermosense_attractor",
        "seeds": SEEDS,
        "valid_seeds": valid_seeds,
        "per_seed": per_seed_data,
        "P1": p1,
        "P2": P2,
        "P3": P3,
        "P4": P4,
        "P5": P5,
        "runaway": runaway,
        "verdict": verdict,
        "token": token,
    }
    verdict_path = out_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict_json, f, indent=2)
    print(f"Verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
