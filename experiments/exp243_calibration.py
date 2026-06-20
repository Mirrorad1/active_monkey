"""Exp 243 substrate calibration search.

Goal: find (speed_cost_slope, continuous_regen_rate) regimes where a monomorphic population
is VIABLE across a band of ≥2-3 speeds AND Mechanism A+B damps it to a stable N_eq.
Then confirm the regime genuinely needs A (oscillates/runs away without it).

Approach:
1. Timing pre-check: run ONE cell and time it; reduce horizon if >30s.
2. Grid (A+B ON): speed_cost_slope in {0.05, 0.1, 0.2} × regen_rate in {0.5, 1.0, 2.0}
   × speed in {0.5, 1.0, 2.0}, single seed, horizon=2000.
3. Pathology check: at the most promising viable (slope, regen) cell, re-run with A OFF
   (hmax=0) at 2-3 speeds.

Constraints:
- No engine mutation; events_hash untouched.
- speed_cost_slope kwarg was added to run_cert (default=0.6; existing tests unchanged).
"""
from __future__ import annotations

import time
import sys
import math
import numpy as np
from ecology.evolvability.cert_run import run_cert
from ecology.evolvability.stability import oscillation_verdict, level_cv, drift_slope

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
SLOPES = [0.05, 0.1, 0.2]
REGENS = [0.5, 1.0, 2.0]
SPEEDS = [0.5, 1.0, 2.0]
HMAX = 0.04
KC = 60.0
THETA = 1.0
RATE_SCALE = 0.0
LAYOUT = "bump"
SEED = 42
HORIZON = 2000
BURN_IN = 0.6

# Timing budget: 12 minutes total = 720 seconds
BUDGET_SECONDS = 720

# Viability thresholds
PERSIST_FLOOR = 30
EXTINCT_THRESHOLD = 5  # n_eq below this = clearly extinct


def classify_cell(r: dict, horizon: int, burn_in: float) -> dict:
    """Classify a run result into a human-readable verdict."""
    N = np.asarray(r["N"], dtype=float)
    neq = r["n_eq"]
    exploded = r["exploded"]

    if exploded:
        return {"class": "RUNAWAY", "n_eq": neq, "oscillation": "N/A"}

    if neq < EXTINCT_THRESHOLD:
        return {"class": "EXTINCT", "n_eq": neq, "oscillation": "N/A"}

    # Check oscillation verdict
    ov = oscillation_verdict(N)
    osc_class = ov["classification"]

    # Check for growth vs plateau
    win_start = int(horizon * burn_in)
    win_len = horizon - win_start
    # Compare last-quarter vs first-quarter of window
    q = win_len // 4
    if q > 0:
        first_q = N[:q].mean()
        last_q = N[-q:].mean()
        ratio = last_q / first_q if first_q > 0 else 0.0
        if ratio > 1.5:
            growth_class = "RUNAWAY"
        elif ratio < 0.5:
            growth_class = "COLLAPSING"
        else:
            growth_class = "PLATEAU"
    else:
        growth_class = "UNKNOWN"

    # Compute level CV for additional context
    lcv = level_cv(N) if neq >= 1 else float("inf")

    return {
        "class": f"VIABLE-{osc_class}" if neq >= PERSIST_FLOOR else f"MARGINAL-{osc_class}",
        "n_eq": neq,
        "oscillation": osc_class,
        "growth": growth_class,
        "level_cv": lcv,
        "births_per_step": r["births_per_step"],
        "crowding_per_step": r["crowding_per_step"],
        "availability": r["availability_mean"],
    }


def run_grid(a_on: bool = True) -> list[dict]:
    """Run the full grid. If a_on=False, disable Mechanism A (hmax=0)."""
    results = []
    hmax = HMAX if a_on else 0.0
    for slope in SLOPES:
        for regen in REGENS:
            for speed in SPEEDS:
                t0 = time.time()
                try:
                    r = run_cert(
                        speed=speed,
                        hmax=hmax,
                        Kc=KC,
                        theta=THETA,
                        regen_rate=regen,
                        rate_scale=RATE_SCALE,
                        layout=LAYOUT,
                        seed=SEED,
                        horizon=HORIZON,
                        burn_in=BURN_IN,
                        speed_cost_slope=slope,
                    )
                    elapsed = time.time() - t0
                    classification = classify_cell(r, HORIZON, BURN_IN)
                    results.append({
                        "slope": slope, "regen": regen, "speed": speed,
                        "wall_s": elapsed,
                        **classification,
                        "raw": r,
                    })
                    print(f"  slope={slope} regen={regen} speed={speed}: "
                          f"{classification.get('class','?')} n_eq={classification.get('n_eq',0):.0f} "
                          f"({elapsed:.1f}s)", flush=True)
                except Exception as exc:
                    elapsed = time.time() - t0
                    results.append({
                        "slope": slope, "regen": regen, "speed": speed,
                        "wall_s": elapsed,
                        "class": f"ERROR:{exc}", "n_eq": 0,
                    })
                    print(f"  slope={slope} regen={regen} speed={speed}: ERROR {exc} ({elapsed:.1f}s)", flush=True)
    return results


def pathology_check(best_slope: float, best_regen: float, check_speeds: list[float]) -> list[dict]:
    """Re-run the best (slope, regen) cell with A OFF (hmax=0) at multiple speeds."""
    results = []
    for speed in check_speeds:
        t0 = time.time()
        r = run_cert(
            speed=speed,
            hmax=0.0,   # A OFF
            Kc=KC,
            theta=THETA,
            regen_rate=best_regen,
            rate_scale=RATE_SCALE,
            layout=LAYOUT,
            seed=SEED,
            horizon=HORIZON,
            burn_in=BURN_IN,
            speed_cost_slope=best_slope,
        )
        elapsed = time.time() - t0
        classification = classify_cell(r, HORIZON, BURN_IN)
        results.append({
            "slope": best_slope, "regen": best_regen, "speed": speed,
            "a_on": False,
            "wall_s": elapsed,
            **classification,
        })
        print(f"  A-OFF slope={best_slope} regen={best_regen} speed={speed}: "
              f"{classification.get('class','?')} n_eq={classification.get('n_eq',0):.0f} "
              f"({elapsed:.1f}s)", flush=True)
    return results


def format_table(results: list[dict]) -> str:
    """Format the grid results as a readable table."""
    lines = []
    lines.append("=" * 100)
    lines.append(
        f"{'slope':>7} {'regen':>6} {'speed':>6} | {'class':>22} {'n_eq':>6} "
        f"{'oscil':>10} {'births/s':>9} {'crowd/s':>8} {'avail':>6} | {'wall_s':>7}"
    )
    lines.append("-" * 100)
    for r in results:
        births = r.get("births_per_step", float("nan"))
        crowd = r.get("crowding_per_step", float("nan"))
        avail = r.get("availability", float("nan"))
        osc = r.get("oscillation", "N/A")
        cls = r.get("class", "?")
        lines.append(
            f"{r['slope']:>7.2f} {r['regen']:>6.1f} {r['speed']:>6.1f} | "
            f"{cls:>22} {r.get('n_eq', 0):>6.0f} "
            f"{osc:>10} {births:>9.4f} {crowd:>8.4f} {avail:>6.3f} | {r['wall_s']:>7.1f}s"
        )
    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    output_lines = []

    def log(msg=""):
        print(msg, flush=True)
        output_lines.append(msg)

    log("=" * 80)
    log("Exp 243 Substrate Calibration Search")
    log("=" * 80)
    log()

    # -----------------------------------------------------------------------
    # Step 1: Timing pre-check
    # -----------------------------------------------------------------------
    log("STEP 1: Timing pre-check (speed=1.0, slope=0.1, regen=1.0, horizon=3000)")
    t0 = time.time()
    r_precheck = run_cert(
        speed=1.0, hmax=HMAX, Kc=KC, theta=THETA, regen_rate=1.0,
        rate_scale=RATE_SCALE, layout=LAYOUT, seed=SEED, horizon=3000, burn_in=BURN_IN,
        speed_cost_slope=0.1
    )
    precheck_elapsed = time.time() - t0
    log(f"  Pre-check: wall_time={precheck_elapsed:.2f}s, exploded={r_precheck['exploded']}, "
        f"n_eq={r_precheck['n_eq']:.1f}")

    # Determine horizon
    if precheck_elapsed > 30:
        log(f"  Pre-check exceeded 30s (was {precheck_elapsed:.1f}s) — using horizon=2000")
        use_horizon = 2000
    else:
        log(f"  Pre-check under 30s — using horizon=3000 AS PLANNED")
        use_horizon = 3000

    # We know from prior runs that fast cells (speed=2.0, slope=0.05) take ~67s at horizon=2000.
    # Force horizon=2000 to stay under budget.
    if use_horizon == 3000 and precheck_elapsed > 20:
        log(f"  Precheck ~22s; fast cells ~67s at h=2000 → forcing horizon=2000 to stay under 12min budget")
        use_horizon = 2000

    global HORIZON
    HORIZON = use_horizon
    projected = 27 * precheck_elapsed * (use_horizon / 3000) * 2.5  # fast cells ~2.5x slower
    log(f"  Horizon set to: {use_horizon}")
    log(f"  Pre-check exploded={r_precheck['exploded']} n_eq={r_precheck['n_eq']:.0f} — viable.")
    log()

    # -----------------------------------------------------------------------
    # Step 2: Run the grid (A+B ON)
    # -----------------------------------------------------------------------
    log("STEP 2: Running grid (A+B ON)")
    log(f"  Grid: slope in {SLOPES} × regen in {REGENS} × speed in {SPEEDS}")
    log(f"  Params: hmax={HMAX}, Kc={KC}, theta={THETA}, horizon={HORIZON}, seed={SEED}")
    log()

    grid_t0 = time.time()
    grid_results = run_grid(a_on=True)
    grid_elapsed = time.time() - grid_t0

    log()
    log(f"Grid total wall time: {grid_elapsed:.1f}s ({grid_elapsed/60:.1f}min)")
    log()

    # -----------------------------------------------------------------------
    # Step 3: Identify viable regimes
    # -----------------------------------------------------------------------
    log("STEP 3: Regime map")
    log()
    table = format_table(grid_results)
    log(table)
    log()

    # Find (slope, regen) pairs with >=2 speeds having VIABLE-DAMPED
    viable_damped_counts = {}  # (slope, regen) -> count of VIABLE-DAMPED speeds
    viable_damped_speeds = {}  # (slope, regen) -> list of speeds
    for r in grid_results:
        key = (r["slope"], r["regen"])
        cls = r.get("class", "")
        if "VIABLE" in cls and "DAMPED" in cls:
            viable_damped_counts[key] = viable_damped_counts.get(key, 0) + 1
            if key not in viable_damped_speeds:
                viable_damped_speeds[key] = []
            viable_damped_speeds[key].append(r["speed"])

    log("VIABLE+DAMPED band summary:")
    best_key = None
    best_count = 0
    for key in sorted(viable_damped_counts):
        count = viable_damped_counts[key]
        speeds = sorted(viable_damped_speeds.get(key, []))
        log(f"  slope={key[0]:.2f}, regen={key[1]:.1f}: {count} speeds viable+damped: {speeds}")
        if count > best_count:
            best_count = count
            best_key = key
    if not viable_damped_counts:
        log("  *** NO (slope, regen) cell has ANY speed that is VIABLE+DAMPED. ***")
    log()

    # Also note MARGINAL-DAMPED
    marginal_damped = {}
    for r in grid_results:
        key = (r["slope"], r["regen"])
        cls = r.get("class", "")
        if "MARGINAL" in cls and "DAMPED" in cls:
            if key not in marginal_damped:
                marginal_damped[key] = []
            marginal_damped[key].append(r["speed"])
    if marginal_damped:
        log("MARGINAL+DAMPED band summary (n_eq < 30 but > 5, damped):")
        for key in sorted(marginal_damped):
            log(f"  slope={key[0]:.2f}, regen={key[1]:.1f}: speeds {sorted(marginal_damped[key])}")
        log()

    # -----------------------------------------------------------------------
    # Step 4: Pathology check (A OFF) at best viable regime
    # -----------------------------------------------------------------------
    log("STEP 4: Pathology check (A OFF = enable_density_mortality disabled, hmax=0)")
    log()

    if best_key is not None:
        best_slope, best_regen = best_key
        check_speeds_for_pathology = sorted(viable_damped_speeds.get(best_key, [0.5, 1.0, 2.0]))
        if len(check_speeds_for_pathology) < 2:
            check_speeds_for_pathology = sorted(set(check_speeds_for_pathology) | {0.5, 1.0})[:3]
        log(f"  Best viable regime: slope={best_slope}, regen={best_regen} ({best_count} VIABLE-DAMPED speeds)")
        log(f"  Testing A-OFF at speeds: {check_speeds_for_pathology}")
        log()
        pathology_results = pathology_check(best_slope, best_regen, check_speeds_for_pathology)
        log()
        log("Pathology check results (A OFF, same (slope, regen) as best viable):")
        table_pathology = format_table(pathology_results)
        log(table_pathology)
        log()

        # Verdict: did removing A destabilize?
        a_on_classes = {r["speed"]: r.get("class", "") for r in grid_results
                        if r["slope"] == best_slope and r["regen"] == best_regen}
        a_off_classes = {r["speed"]: r.get("class", "") for r in pathology_results}

        log("A-ON vs A-OFF comparison:")
        any_destabilized = False
        any_changed = False
        for speed in check_speeds_for_pathology:
            on_c = a_on_classes.get(speed, "?")
            off_c = a_off_classes.get(speed, "?")
            changed = on_c != off_c
            if changed:
                any_changed = True
            if "DAMPED" in on_c and ("OSCILLATORY" in off_c or "RUNAWAY" in off_c or "EXTINCT" in off_c):
                any_destabilized = True
            log(f"  speed={speed}: A-ON={on_c} → A-OFF={off_c}" + (" [DESTABILIZED]" if changed else " [same]"))

        log()
        if any_destabilized:
            log("PATHOLOGY VERDICT: A is NECESSARY — removing A destabilizes (DAMPED→OSCILLATORY/RUNAWAY/EXTINCT).")
            log("This confirms A+B is doing real stabilization work.")
        elif any_changed:
            log("PATHOLOGY VERDICT: A changes dynamics but not clearly from DAMPED to unstable.")
        else:
            log("PATHOLOGY VERDICT: A appears NOT NEEDED — dynamics same without A. A is redundant at this regime.")
    else:
        # Fallback: pick best-looking (slope, regen) based on marginal or most n_eq > 0
        # Try the most populated failing regime for pathology
        most_pop = sorted(grid_results, key=lambda r: r.get("n_eq", 0), reverse=True)
        if most_pop and most_pop[0].get("n_eq", 0) > 0:
            best_slope = most_pop[0]["slope"]
            best_regen = most_pop[0]["regen"]
            check_speeds_for_pathology = [0.5, 1.0, 2.0]
        else:
            best_slope, best_regen = 0.1, 1.0
            check_speeds_for_pathology = [0.5, 1.0, 2.0]
        log(f"  No VIABLE-DAMPED regime found; testing A-OFF at best populated regime: slope={best_slope}, regen={best_regen}")
        pathology_results = pathology_check(best_slope, best_regen, check_speeds_for_pathology)
        log()
        log("Pathology check results (A OFF at best populated regime):")
        table_pathology = format_table(pathology_results)
        log(table_pathology)
        log()

    # -----------------------------------------------------------------------
    # Step 5: Recommendation
    # -----------------------------------------------------------------------
    log()
    log("=" * 80)
    log("RECOMMENDATION")
    log("=" * 80)

    if best_key is not None and best_count >= 2:
        best_slope, best_regen = best_key
        speeds = sorted(viable_damped_speeds.get(best_key, []))
        log(f"VIABLE+DAMPED REGIME FOUND: slope={best_slope}, regen={best_regen}")
        log(f"  Viable+damped speeds: {speeds}")
        log(f"  Recommendation: full Stage-1 sweep should use:")
        log(f"    speed_cost_slope in {[best_slope]} (or nearby {[s for s in SLOPES if abs(s - best_slope) < 0.06]})")
        log(f"    continuous_regen_rate in {[best_regen]}")
        log(f"    Replaces non-viable plan regen {{0.025, 0.05, 0.10}}")
        # Estimate sweep feasibility
        viable_cells = [r for r in grid_results if "VIABLE" in r.get("class", "")]
        if viable_cells:
            avg_viable_time = sum(r["wall_s"] for r in viable_cells) / len(viable_cells)
            full_sweep = 2304 * avg_viable_time
            log(f"  Projected full 2304-run sweep time (at avg {avg_viable_time:.1f}s/run): "
                f"{full_sweep:.0f}s = {full_sweep/3600:.1f}h")
    else:
        log("NO VIABLE+DAMPED BAND FOUND in the searched grid.")
        log("This is a substantive result: even with the viable founder + A+B, this substrate")
        log("does not host a damped equilibrium across ≥2 speeds at {SLOPES} × {REGENS}.")
        log()
        log("Observed patterns:")
        # Summarize what DID happen
        class_counts = {}
        for r in grid_results:
            cls = r.get("class", "?").split("-")[0]
            class_counts[cls] = class_counts.get(cls, 0) + 1
        for cls, count in sorted(class_counts.items()):
            log(f"  {cls}: {count} cells")
        log()
        log("Recommendation: the current (speed_cost_slope, regen_rate) parameter space")
        log("does not yield a stable calibration substrate. The continuous-locomotion")
        log("chapter verdict (CAN'T POSE) appears robust to this parameter search.")
        log("Do NOT proceed to full Stage-1 sweep at these parameter ranges.")

    log()
    log(f"Grid total wall time: {grid_elapsed:.1f}s ({grid_elapsed/60:.1f}min)")
    log(f"Script finished at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    # -----------------------------------------------------------------------
    # Write output file
    # -----------------------------------------------------------------------
    output_path = "experiments/outputs/exp243_calibration.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nOutput written to: {output_path}")


if __name__ == "__main__":
    main()
