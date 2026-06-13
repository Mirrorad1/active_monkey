"""Exp 196 — N5: does senescence's complexity-frailty trade-off SELECT against complexity
at a LONGER horizon? (pre-registered in loop/directions/population-ecology.md, commit
01be7a4, BEFORE any data).

Hypothesis: Exp 195 found no complexity-selection from senescence at 600 steps (P6 null);
this tests whether that was a HORIZON limit, not an absence — over many more generations,
senescence's complexity-frailty trade-off progressively selects the population toward LOWER
complexity, so against a senescence-OFF control (mean complexity ~flat) the senescence-ON
population's mean complexity diverges measurably and progressively below it.

Reconnaissance (disclosed, per L7): a throwaway 2-seed probe (seeds 0,1; not committed) at
horizon 5000 with the cap raised showed control complexity flat (~0.54/~0.43) and treatment
drifting to ~0.34/~0.29 (gap 0.13-0.20 at t=5000 vs <0.03 at t=600). The claim is fixed from
the probe and TESTED on FRESH seeds {3,4,5,6,7} never used in the probe.

Setup: balanced regime with max_population raised to ~5000 so the population equilibrates at
its resource carrying capacity (~530) instead of halting at the safety guard (disclosed; the
L19 fix removing artificial truncation). horizon 5000; both ARMS (OFF=control, ON=treatment)
on the SAME fresh seeds {3,4,5,6,7}; sample population mean complexity every 500 steps;
senescence is rng-free.

Predictions if TRUE (5 seeds, report ALL):
  P1 determinism: same seed -> identical event hash, both arms.
  P2 validity (input gate): both arms reach t=5000 without explosion/extinction, >=4/5 seeds.
  P3 (CORE) selection emerges: treatment final mean complexity < control final by >= 0.05,
     >=4/5 seeds.
  P4 progressive: gap(control-treatment) at t=5000 exceeds gap at t=600 by >= 0.05, >=4/5
     seeds; AND control mean complexity ~flat (|control@5000 - control@600| < 0.05).
  P5 direction: treatment < control at horizon in ALL valid seeds.

Falsifiers:
  F1 non-determinism -> NEGATIVE.
  F2 (CORE) no emergence: treatment-control gap < 0.05 at horizon in a majority of valid
     seeds -> NEGATIVE (Exp 195 P6 null is a real absence).
  F3 not progressive: horizon gap does not exceed the t=600 gap by >=0.05 in a majority -> MIXED.
  F4 wrong direction: treatment > control in any valid seed -> NEGATIVE.
  F5 invalid: arms fail to persist full-length in a majority -> NO_VERDICT.

Verdict rule: POSITIVE iff P1 ^ P2 ^ P3 ^ P4 ^ P5 and none of F1/F4/F5 fire; F2 (P3 fail) ->
NEGATIVE; F3 (P4 fail) -> MIXED. Honesty: the raised cap is a disclosed design change (the
L19 fix), not tuning-to-metric; the selection direction is expected from the imposed frailty
cost but its magnitude/timescale/whether it beats complexity's benefits is the emergent,
falsifiable question; the control isolates senescence's effect from drift; probe seeds (0,1)
disclosed, tested on fresh seeds.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from statistics import mean

import numpy as np

# Ensure repo root is in path
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS
from ecology.genotype import complexity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [3, 4, 5, 6, 7]
HORIZON = 5000
MAX_POP = 5000
SAMPLE_STEPS = list(range(500, HORIZON + 1, 500))  # 500, 1000, ..., 5000
EXTRA_SAMPLE = 600  # also capture t=600

# Senescence params (identical to exp195 treatment)
SENES_PARAMS = dict(
    enable_senescence=True,
    senescence_onset0=155.0,
    senescence_onset_frailty=0.65,
    senescence_rate_frailty=2.0,
    senescence_base=0.002,
    senescence_self_maintenance=1.5,
    senescence_exp=1.5,
)

OUT_BASE = _REPO / "experiments" / "outputs" / "exp196_n5_senescence_selection"
OUT_BASE.mkdir(parents=True, exist_ok=True)
EXP196_TXT = _REPO / "experiments" / "outputs" / "exp196.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mean_complexity_alive(eco: Ecology) -> float | None:
    """Mean complexity of currently alive creatures; None if none alive."""
    alive = eco._alive()
    if not alive:
        return None
    return mean(complexity(c.genotype) for c in alive)


def run_arm_with_trajectory(
    arm: str, seed: int
) -> tuple[list[dict], dict]:
    """
    Run one arm (control or treatment) for a single seed.
    Returns (trajectory, run_meta).

    trajectory: list of {t, pop, mean_complexity} sampled at EXTRA_SAMPLE and every 500 steps.
    run_meta: dict with summary stats and events_hash.
    """
    base_cfg = SCENARIOS["balanced"]
    if arm == "treatment":
        cfg = replace(base_cfg, horizon=HORIZON, max_population=MAX_POP, **SENES_PARAMS)
    else:
        cfg = replace(base_cfg, horizon=HORIZON, max_population=MAX_POP)

    eco = Ecology(cfg, seed=seed)

    # Determine all sample points: extra + regular, sorted
    sample_set = set(SAMPLE_STEPS) | {EXTRA_SAMPLE}
    sample_sorted = sorted(sample_set)

    trajectory: list[dict] = []
    mc_at_600: float | None = None
    mc_at_final: float | None = None
    extinct = False

    while eco.t < cfg.horizon and not eco.exploded:
        eco.step()
        t = eco.t
        if t in sample_set:
            alive = eco._alive()
            if not alive:
                extinct = True
                trajectory.append({"t": t, "pop": 0, "mean_complexity": None})
                break
            mc = mean(complexity(c.genotype) for c in alive)
            trajectory.append({"t": t, "pop": len(alive), "mean_complexity": mc})
            if t == EXTRA_SAMPLE:
                mc_at_600 = mc

    # If we completed without extinction, grab final state
    if not extinct:
        alive = eco._alive()
        final_pop = len(alive)
        if alive:
            mc_at_final = mean(complexity(c.genotype) for c in alive)
        else:
            extinct = True
            final_pop = 0
    else:
        final_pop = 0

    reached_horizon = eco.t == HORIZON and not eco.exploded and not extinct

    # If the final sample wasn't the last sampled traj point, add it
    if reached_horizon and (not trajectory or trajectory[-1]["t"] != HORIZON):
        if mc_at_final is not None:
            trajectory.append({"t": HORIZON, "pop": final_pop, "mean_complexity": mc_at_final})

    # mc_at_final from last valid traj entry if not already set
    if mc_at_final is None and trajectory:
        last = trajectory[-1]
        mc_at_final = last["mean_complexity"]
        final_pop = last["pop"] if last["pop"] is not None else 0

    run_meta = {
        "arm": arm,
        "seed": seed,
        "reached_horizon": reached_horizon,
        "exploded": eco.exploded,
        "extinct": extinct,
        "final_pop": final_pop,
        "mc_at_600": mc_at_600,
        "mc_at_final": mc_at_final,
        "events_hash": eco.events_hash(),
    }
    return trajectory, run_meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 72)
    emit("Exp 196 — N5: Senescence complexity-selection over long horizon")
    emit("=" * 72)
    emit()

    # -----------------------------------------------------------------------
    # Run all arms x seeds; collect trajectories
    # -----------------------------------------------------------------------
    # results[(arm, seed)] = (trajectory, run_meta)
    results: dict[tuple[str, int], tuple[list[dict], dict]] = {}

    emit("--- Running arms (control=OFF, treatment=ON) x seeds {3,4,5,6,7} ---")
    for arm in ["control", "treatment"]:
        for seed in SEEDS:
            traj, meta = run_arm_with_trajectory(arm, seed)
            results[(arm, seed)] = (traj, meta)
            mc_600 = f"{meta['mc_at_600']:.4f}" if meta["mc_at_600"] is not None else "N/A"
            mc_fin = f"{meta['mc_at_final']:.4f}" if meta["mc_at_final"] is not None else "N/A"
            emit(
                f"  {arm}/seed{seed}: reached={meta['reached_horizon']}, "
                f"pop={meta['final_pop']}, mc@600={mc_600}, mc@final={mc_fin}, "
                f"exploded={meta['exploded']}, extinct={meta['extinct']}, "
                f"hash={meta['events_hash'][:12]}..."
            )
            # Persist per-run outputs
            out_dir = OUT_BASE / f"{arm}_seed{seed}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "trajectory.json", "w") as f:
                json.dump(traj, f, indent=2)
            with open(out_dir / "summary.json", "w") as f:
                json.dump(meta, f, indent=2)

    # -----------------------------------------------------------------------
    # P1: Determinism — rerun seed 3 both arms, check hash
    # -----------------------------------------------------------------------
    emit()
    emit("--- P1 Determinism (seed 3 only, both arms) ---")
    p1_ok = True
    for arm in ["control", "treatment"]:
        _, meta2 = run_arm_with_trajectory(arm, 3)
        orig_hash = results[(arm, 3)][1]["events_hash"]
        match = meta2["events_hash"] == orig_hash
        if not match:
            p1_ok = False
        emit(f"  {arm}/seed3 rerun: {'PASS' if match else 'FAIL'} "
             f"(hash {orig_hash[:12]}...)")

    # -----------------------------------------------------------------------
    # Per-seed analysis: gap, validity
    # -----------------------------------------------------------------------
    valid_seeds: list[int] = []
    seed_data: dict[int, dict] = {}

    for seed in SEEDS:
        _, ctrl_meta = results[("control", seed)]
        _, trt_meta = results[("treatment", seed)]
        valid = ctrl_meta["reached_horizon"] and trt_meta["reached_horizon"]
        if valid:
            valid_seeds.append(seed)

        mc_ctrl_600 = ctrl_meta["mc_at_600"]
        mc_trt_600 = trt_meta["mc_at_600"]
        mc_ctrl_fin = ctrl_meta["mc_at_final"]
        mc_trt_fin = trt_meta["mc_at_final"]

        gap_600 = (mc_ctrl_600 - mc_trt_600) if (mc_ctrl_600 is not None and mc_trt_600 is not None) else None
        gap_5000 = (mc_ctrl_fin - mc_trt_fin) if (mc_ctrl_fin is not None and mc_trt_fin is not None) else None
        prog_gap = (gap_5000 - gap_600) if (gap_5000 is not None and gap_600 is not None) else None
        ctrl_flat = abs(mc_ctrl_fin - mc_ctrl_600) < 0.05 if (mc_ctrl_fin is not None and mc_ctrl_600 is not None) else None

        seed_data[seed] = {
            "valid": valid,
            "mc_ctrl_600": mc_ctrl_600,
            "mc_trt_600": mc_trt_600,
            "mc_ctrl_fin": mc_ctrl_fin,
            "mc_trt_fin": mc_trt_fin,
            "gap_600": gap_600,
            "gap_5000": gap_5000,
            "prog_gap": prog_gap,
            "ctrl_flat": ctrl_flat,
            "ctrl_final_pop": ctrl_meta["final_pop"],
            "trt_final_pop": trt_meta["final_pop"],
        }

    n_valid = len(valid_seeds)

    # -----------------------------------------------------------------------
    # Evaluate P2..P5 and Falsifiers on valid seeds
    # -----------------------------------------------------------------------

    # P2: both arms reach horizon >=4/5 seeds
    p2_ok = n_valid >= 4

    # P3 (CORE): treatment final < control final by >=0.05, >=4/5 valid seeds
    p3_per_seed = {}
    for seed in valid_seeds:
        d = seed_data[seed]
        gap = d["gap_5000"]
        p3_per_seed[seed] = gap is not None and gap >= 0.05

    n_p3_pass = sum(1 for v in p3_per_seed.values() if v)
    p3_ok = n_p3_pass >= 4

    # P4: progressive: gap@5000 > gap@600 by >=0.05, >=4/5 valid seeds; AND ctrl flat
    p4_per_seed = {}
    p4_ctrl_flat_per_seed = {}
    for seed in valid_seeds:
        d = seed_data[seed]
        prog = d["prog_gap"]
        flat = d["ctrl_flat"]
        p4_per_seed[seed] = prog is not None and prog >= 0.05
        p4_ctrl_flat_per_seed[seed] = flat is True

    n_p4_pass = sum(1 for v in p4_per_seed.values() if v)
    n_ctrl_flat = sum(1 for v in p4_ctrl_flat_per_seed.values() if v)
    p4_ok = (n_p4_pass >= 4) and (n_ctrl_flat >= 4)

    # P5: treatment < control at horizon in ALL valid seeds
    p5_per_seed = {}
    for seed in valid_seeds:
        d = seed_data[seed]
        mc_ctrl = d["mc_ctrl_fin"]
        mc_trt = d["mc_trt_fin"]
        p5_per_seed[seed] = (mc_trt is not None and mc_ctrl is not None and mc_trt < mc_ctrl)

    p5_ok = all(p5_per_seed.values()) if p5_per_seed else False

    # Falsifiers
    f1_fires = not p1_ok
    # F2: gap < 0.05 at horizon in majority of valid seeds
    f2_fires = (sum(1 for v in p3_per_seed.values() if not v) > n_valid / 2) if n_valid > 0 else True
    # F3: not progressive (P4 fails)
    f3_fires = not p4_ok
    # F4: treatment > control in any valid seed
    f4_fires = any(not v for v in p5_per_seed.values())
    # F5: majority fail to persist
    f5_fires = n_valid < 3  # majority of 5 seeds fail

    # Verdict rule
    if f1_fires or f4_fires or f5_fires:
        if f5_fires:
            verdict = "NO_VERDICT"
        else:
            verdict = "NEGATIVE"
    elif not p3_ok:
        verdict = "NEGATIVE"  # F2 fires
    elif p2_ok and p3_ok and p4_ok and p5_ok:
        verdict = "POSITIVE"
    elif p3_ok and not p4_ok:
        verdict = "MIXED"
    else:
        verdict = "MIXED"

    # -----------------------------------------------------------------------
    # Print SUMMARY
    # -----------------------------------------------------------------------
    emit()
    emit("=" * 72)
    emit("SUMMARY")
    emit("=" * 72)
    emit()

    emit(f"Seeds: {SEEDS}  Horizon: {HORIZON}  max_pop: {MAX_POP}")
    emit(f"Valid seeds (both arms reached t={HORIZON}): {valid_seeds}  (n={n_valid})")
    emit()

    emit("Per-seed: control vs treatment mean complexity at t=600 and t=5000")
    emit(f"{'seed':>6}  {'ctrl@600':>10}  {'trt@600':>10}  {'gap@600':>9}  "
         f"{'ctrl@5000':>10}  {'trt@5000':>10}  {'gap@5000':>9}  "
         f"{'prog_gap':>9}  {'ctrl_flat':>9}  {'valid':>6}  "
         f"{'ctrl_pop':>8}  {'trt_pop':>8}")
    for seed in SEEDS:
        d = seed_data[seed]
        def _f(v): return f"{v:.4f}" if v is not None else "  N/A  "
        def _b(v): return str(v) if v is not None else "  N/A"
        emit(f"{seed:>6}  {_f(d['mc_ctrl_600']):>10}  {_f(d['mc_trt_600']):>10}  "
             f"{_f(d['gap_600']):>9}  "
             f"{_f(d['mc_ctrl_fin']):>10}  {_f(d['mc_trt_fin']):>10}  "
             f"{_f(d['gap_5000']):>9}  "
             f"{_f(d['prog_gap']):>9}  {_b(d['ctrl_flat']):>9}  "
             f"{'Y' if d['valid'] else 'N':>6}  "
             f"{d['ctrl_final_pop']:>8}  {d['trt_final_pop']:>8}")

    emit()
    # Representative trajectory (seed 3)
    emit("Representative trajectory — seed 3 (control vs treatment mean_complexity):")
    ctrl_traj, _ = results[("control", 3)]
    trt_traj, _ = results[("treatment", 3)]
    ctrl_by_t = {e["t"]: e["mean_complexity"] for e in ctrl_traj}
    trt_by_t = {e["t"]: e["mean_complexity"] for e in trt_traj}
    emit(f"{'t':>6}  {'ctrl_mc':>8}  {'trt_mc':>8}  {'gap':>8}")
    for t in sorted(set(ctrl_by_t) | set(trt_by_t)):
        cmc = ctrl_by_t.get(t)
        tmc = trt_by_t.get(t)
        gap_str = f"{cmc - tmc:.4f}" if (cmc is not None and tmc is not None) else "  N/A"
        emit(f"{t:>6}  "
             f"{f'{cmc:.4f}' if cmc is not None else '  N/A':>8}  "
             f"{f'{tmc:.4f}' if tmc is not None else '  N/A':>8}  "
             f"{gap_str:>8}")

    emit()
    emit("-- PREDICTIONS --")

    # P1
    emit(f"P1 determinism: {'PASS' if p1_ok else 'FAIL'}")

    emit()
    # P2
    emit(f"P2 validity (both arms reach horizon): {'PASS' if p2_ok else 'FAIL'}  "
         f"[{n_valid}/5 seeds valid, threshold >=4]")

    emit()
    # P3
    emit(f"P3 (CORE) selection emerges (gap@5000 >=0.05): {'PASS' if p3_ok else 'FAIL'}  "
         f"[{n_p3_pass}/{n_valid} valid seeds pass, threshold >=4]")
    for seed in valid_seeds:
        d = seed_data[seed]
        gap = d["gap_5000"]
        emit(f"   seed={seed}: gap@5000={f'{gap:.4f}' if gap is not None else 'N/A'}  "
             f"(>=0.05: {gap is not None and gap >= 0.05})")

    emit()
    # P4
    emit(f"P4 progressive gap (gap@5000 - gap@600 >=0.05): {'PASS' if p4_ok else 'FAIL'}  "
         f"[{n_p4_pass}/{n_valid} valid seeds pass threshold, {n_ctrl_flat}/{n_valid} ctrl flat]")
    for seed in valid_seeds:
        d = seed_data[seed]
        prog = d["prog_gap"]
        flat = d["ctrl_flat"]
        g600 = f"{d['gap_600']:.4f}" if d["gap_600"] is not None else "N/A"
        g5000 = f"{d['gap_5000']:.4f}" if d["gap_5000"] is not None else "N/A"
        pgap = f"{prog:.4f}" if prog is not None else "N/A"
        emit(f"   seed={seed}: gap@600={g600}, gap@5000={g5000}, "
             f"prog_gap={pgap} (>=0.05: {prog is not None and prog >= 0.05}), "
             f"ctrl_flat={flat}")

    emit()
    # P5
    emit(f"P5 direction (treatment < control in ALL valid seeds): "
         f"{'PASS' if p5_ok else 'FAIL'}")
    for seed in valid_seeds:
        d = seed_data[seed]
        trt_s = f"{d['mc_trt_fin']:.4f}" if d["mc_trt_fin"] is not None else "N/A"
        ctrl_s = f"{d['mc_ctrl_fin']:.4f}" if d["mc_ctrl_fin"] is not None else "N/A"
        emit(f"   seed={seed}: trt={trt_s} < ctrl={ctrl_s}: {p5_per_seed.get(seed, False)}")

    emit()
    emit("-- FALSIFIERS --")
    emit(f"F1 non-determinism: {'FIRES' if f1_fires else 'CLEAR'}")
    emit(f"F2 (CORE) no emergence: {'FIRES' if f2_fires else 'CLEAR'}")
    emit(f"F3 not progressive: {'FIRES' if f3_fires else 'CLEAR'}")
    emit(f"F4 wrong direction (treatment > control any valid seed): {'FIRES' if f4_fires else 'CLEAR'}")
    emit(f"F5 invalid (majority fail to persist): {'FIRES' if f5_fires else 'CLEAR'}")

    emit()
    emit(f"VERDICT: {verdict}")

    # -----------------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------------
    runtime = time.time() - t_start
    emit()
    emit(f"runtime: {runtime:.1f}s")

    # -----------------------------------------------------------------------
    # Write exp196.txt
    # -----------------------------------------------------------------------
    with open(EXP196_TXT, "w") as f:
        f.write("\n".join(lines) + "\n")

    # -----------------------------------------------------------------------
    # Write verdict.json
    # -----------------------------------------------------------------------
    # Build per-seed numbers for JSON
    per_seed_json: dict = {}
    for seed in SEEDS:
        d = seed_data[seed]
        ctrl_traj_j, _ = results[("control", seed)]
        trt_traj_j, _ = results[("treatment", seed)]
        per_seed_json[str(seed)] = {
            "valid": d["valid"],
            "mc_ctrl_600": d["mc_ctrl_600"],
            "mc_trt_600": d["mc_trt_600"],
            "mc_ctrl_final": d["mc_ctrl_fin"],
            "mc_trt_final": d["mc_trt_fin"],
            "gap_600": d["gap_600"],
            "gap_5000": d["gap_5000"],
            "prog_gap": d["prog_gap"],
            "ctrl_flat": d["ctrl_flat"],
            "ctrl_final_pop": d["ctrl_final_pop"],
            "trt_final_pop": d["trt_final_pop"],
            "P3_pass": p3_per_seed.get(seed),
            "P4_pass": p4_per_seed.get(seed),
            "P5_pass": p5_per_seed.get(seed),
            "control_trajectory": ctrl_traj_j,
            "treatment_trajectory": trt_traj_j,
        }

    verdict_data = {
        "verdict": verdict,
        "seeds": SEEDS,
        "horizon": HORIZON,
        "max_population": MAX_POP,
        "valid_seeds": valid_seeds,
        "n_valid": n_valid,
        "P1": {"pass": p1_ok},
        "P2": {"pass": p2_ok, "n_valid": n_valid, "threshold": 4},
        "P3": {
            "pass": p3_ok,
            "n_pass": n_p3_pass,
            "n_valid": n_valid,
            "threshold_gap": 0.05,
            "threshold_seeds": 4,
            "per_seed": {str(s): p3_per_seed.get(s) for s in valid_seeds},
        },
        "P4": {
            "pass": p4_ok,
            "n_prog_pass": n_p4_pass,
            "n_ctrl_flat": n_ctrl_flat,
            "n_valid": n_valid,
            "threshold_prog_gap": 0.05,
            "threshold_seeds": 4,
            "per_seed_prog": {str(s): p4_per_seed.get(s) for s in valid_seeds},
            "per_seed_flat": {str(s): p4_ctrl_flat_per_seed.get(s) for s in valid_seeds},
        },
        "P5": {
            "pass": p5_ok,
            "per_seed": {str(s): p5_per_seed.get(s) for s in valid_seeds},
        },
        "F1": {"fires": f1_fires},
        "F2": {"fires": f2_fires},
        "F3": {"fires": f3_fires},
        "F4": {"fires": f4_fires},
        "F5": {"fires": f5_fires},
        "per_seed": per_seed_json,
        "senescence_params": SENES_PARAMS,
        "runtime_s": round(runtime, 1),
    }

    with open(OUT_BASE / "verdict.json", "w") as f:
        json.dump(verdict_data, f, indent=2)

    emit()
    emit(f"Outputs written to {OUT_BASE}/")
    emit(f"exp196.txt written to {EXP196_TXT}")


if __name__ == "__main__":
    main()
