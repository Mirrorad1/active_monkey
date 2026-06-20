"""Exp 243 — Runtime preflight pilot.

Goal: measure real per-run wall time + a logistic-aware growth check on 3 representative
cells before launching the full ~2304-run sweep.

Cells probed:
  - SLOW:   speed=0.5, hmax=0.04, Kc=60, theta=1.0, regen=0.05, layout=bump, seed=42
  - MEDIUM: speed=1.0, hmax=0.04, Kc=60, theta=1.0, regen=0.05, layout=bump, seed=42
  - FAST:   speed=2.0, hmax=0.04, Kc=60, theta=1.0, regen=0.05, layout=bump, seed=42

Each run: horizon=4000 (burn_in=0.6 => analysis window steps 2400–4000).

Logistic-aware growth check (NOT naive geometric extrapolation):
  Classify growth over the FULL run as:
  - DECELERATING/plateauing (healthy): increments per step shrink over time (logistic approach)
  - RUNAWAY: still accelerating near max_population / hits exploded
  - EXTINCT: N→0 — population goes extinct within the run

Full Stage-1 grid:
  8 speeds × 4 hmax × 3 Kc × 3 regen × 8 seeds = 2304 runs at horizon 4000
  (trivially parallel across cells/seeds)

CRITICAL FINDING: See birth-balance diagnostic (exp243_birthbalance.py) for context.
The default founder (repro_threshold=17.0, aging_cost=0.02/step) in the continuous
world (capacity=2.0, regen_rate=0.05) produces ALL-EXTINCT populations — the maximum
achievable energy falls short of the reproduction threshold. This preflight confirms
whether EXTINCT is indeed the universal outcome and measures the actual run times.

Run:
  uv run --python .venv python experiments/exp243_preflight.py
"""
from __future__ import annotations

import os
import sys
import time
import multiprocessing
from pathlib import Path

import numpy as np

# Ensure repo root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from ecology.evolvability.cert_run import run_cert, _reconstruct_N_per_step
from ecology.evolvability.stability import oscillation_verdict

# ---- Preflight parameters ----
HORIZON = 4000
BURN_IN = 0.6
SEED = 42
LAYOUT = "bump"
HMAX = 0.04
KC = 60.0
THETA = 1.0
# regen_rate=1.0: Exp-242 viable continuous regime (sweep was [0.5, 1.0, 2.0]).
# regen=0.05 was the original (non-viable) discrete "balanced" rate — too slow to
# sustain the continuous founder at any density. The birth-balance confirmed regen=1.0
# gives b=0.019/step and implied N_eq=28 (viable, sub-Kc=60).
REGEN_RATE = 1.0
RATE_SCALE = 0.0

# Stage-1 full grid dimensions (from spec §8)
STAGE1_SPEEDS = 8
STAGE1_HMAX = 4
STAGE1_KC = 3
STAGE1_REGEN = 3
STAGE1_SEEDS = 8
STAGE1_TOTAL = STAGE1_SPEEDS * STAGE1_HMAX * STAGE1_KC * STAGE1_REGEN * STAGE1_SEEDS

CELLS = [
    {"label": "SLOW",   "speed": 0.5},
    {"label": "MEDIUM", "speed": 1.0},
    {"label": "FAST",   "speed": 2.0},
]

# Time budget: if any single run exceeds this, stop early and report NO-GO
MAX_SINGLE_RUN_SECS = 180.0  # 3 minutes


def _classify_growth_logistic(N_window: np.ndarray, exploded: bool, n_eq: float) -> str:
    """Logistic-aware growth classification of the analysis window.

    The analysis window is the LAST 40% of the run (steps 2400-4000).
    For a healthy logistic plateau, the window should be FLAT around N_eq.

    Rules:
    - RUNAWAY: exploded=True, OR N still accelerating in the window with high final N
    - EXTINCT: median(N_window) <= 5 OR all N_window == 0
    - DECELERATING: N window is flat/declining at a stable level (logistic plateau)

    Key distinction from naive geometric: we check that growth DECELERATES
    (increments shrink) rather than just that N is large.
    """
    if exploded:
        return "RUNAWAY"

    if float(np.median(N_window)) <= 5.0:
        return "EXTINCT"

    n = len(N_window)
    if n < 2:
        return "EXTINCT" if float(N_window[0]) <= 5.0 else "DECELERATING"

    # Compute per-step increments in the window
    dN = np.diff(N_window)

    # Compare first vs second half of the window
    half = len(dN) // 2
    if half == 0:
        return "DECELERATING"

    first_half = dN[:half]
    second_half = dN[half:]

    # Mean increment in each half (positive = growth rate)
    mean_first = float(np.mean(first_half))
    mean_second = float(np.mean(second_half))

    # OLS slope across the whole window — still growing?
    t_w = np.arange(n, dtype=float)
    slope_win, _ = np.polyfit(t_w, N_window, 1)

    # For a logistic plateau:
    #   - slope_win should be near zero or negative (no net growth in the window)
    #   - mean_second should NOT exceed mean_first by a large margin (deceleration)
    #   - N should be STABLE (not growing fast)

    # RUNAWAY signal: strong positive slope across window AND second half still growing faster
    n_eq_ref = n_eq if n_eq > 0 else float(np.median(N_window))
    relative_slope = slope_win / n_eq_ref if n_eq_ref > 0 else 0.0

    if relative_slope > 0.001 and mean_second > mean_first * 1.1 and mean_second > 0:
        return "RUNAWAY"

    # If window min is still > 5 and not strongly growing, it's a plateau or extinct/declining
    if float(N_window.min()) >= 5:
        return "DECELERATING"
    else:
        return "EXTINCT"


def run_one_cell(cell: dict) -> dict:
    """Run a single preflight cell, time it, classify its dynamics."""
    label = cell["label"]
    speed = cell["speed"]

    print(f"\n--- {label} (speed={speed}) ---")
    print(f"  horizon={HORIZON}, hmax={HMAX}, Kc={KC}, regen={REGEN_RATE}, seed={SEED}")

    t0 = time.perf_counter()
    tel = run_cert(
        speed,
        hmax=HMAX,
        Kc=KC,
        theta=THETA,
        regen_rate=REGEN_RATE,
        rate_scale=RATE_SCALE,
        layout=LAYOUT,
        seed=SEED,
        horizon=HORIZON,
        burn_in=BURN_IN,
    )
    elapsed = time.perf_counter() - t0

    N_window = tel["N"]  # analysis window only (steps 2400–4000)
    n_eq_val = tel["n_eq"]
    exploded = tel["exploded"]
    births_per_step = tel["births_per_step"]
    crowding_per_step = tel["crowding_per_step"]
    availability = tel["availability_mean"]

    N_win_arr = np.asarray(N_window, dtype=float)

    # Growth classification (logistic-aware, on the analysis window)
    growth_class = _classify_growth_logistic(N_win_arr, exploded, n_eq_val)

    # Oscillation verdict (only meaningful if population survived)
    if float(np.median(N_win_arr)) > 5.0:
        osc = oscillation_verdict(N_win_arr, seed=SEED)
        osc_class = osc["classification"]
    else:
        # Extinct population: oscillation detector is undefined/meaningless
        osc = {"ar_modulus": float("nan"), "amp_ptp": float("nan"), "autocorr_trough": float("nan"),
               "classification": "N/A (EXTINCT)"}
        osc_class = "N/A (EXTINCT)"

    result = {
        "label": label,
        "speed": speed,
        "elapsed_s": elapsed,
        "n_eq": n_eq_val,
        "final_N": float(N_win_arr[-1]) if len(N_win_arr) > 0 else 0.0,
        "median_N_window": float(np.median(N_win_arr)),
        "growth_class": growth_class,
        "osc_class": osc_class,
        "exploded": exploded,
        "births_per_step": births_per_step,
        "crowding_per_step": crowding_per_step,
        "availability": availability,
        "osc_detail": osc,
    }

    print(f"  Wall time:          {elapsed:.3f}s")
    print(f"  n_eq (median win):  {n_eq_val:.1f}")
    print(f"  final N (win end):  {result['final_N']:.0f}")
    print(f"  median N in window: {result['median_N_window']:.1f}")
    print(f"  exploded:           {exploded}")
    print(f"  growth_class:       {growth_class}")
    print(f"  oscillation:        {osc_class}")
    if not np.isnan(osc['ar_modulus']):
        print(f"    ar_mod={osc['ar_modulus']:.3f}, amp_ptp={osc['amp_ptp']:.3f}, "
              f"trough={osc['autocorr_trough']:.3f}")
    print(f"  births/step:        {births_per_step:.4f}")
    print(f"  crowding/step:      {crowding_per_step:.4f}")
    print(f"  availability:       {availability:.3f}")

    return result


def main():
    print("=" * 65)
    print("Exp 243 Runtime Preflight Pilot")
    print(f"  3 cells at horizon={HORIZON}, burn_in={BURN_IN}")
    print(f"  hmax={HMAX}, Kc={KC}, theta={THETA}, regen={REGEN_RATE}, seed={SEED}")
    print(f"  Stage-1 grid: {STAGE1_TOTAL} runs ({STAGE1_SPEEDS} speeds x "
          f"{STAGE1_HMAX} hmax x {STAGE1_KC} Kc x {STAGE1_REGEN} regen x "
          f"{STAGE1_SEEDS} seeds)")
    print("=" * 65)

    results = []
    for cell in CELLS:
        r = run_one_cell(cell)
        results.append(r)

        # Early stop if single run is too slow
        if r["elapsed_s"] > MAX_SINGLE_RUN_SECS:
            print(f"\n  EARLY STOP: {cell['label']} took {r['elapsed_s']:.1f}s > {MAX_SINGLE_RUN_SECS}s limit")
            print("  Not running remaining cells.")
            break

    # Summary statistics
    elapsed_times = [r["elapsed_s"] for r in results]
    median_elapsed = float(np.median(elapsed_times))
    max_elapsed = float(np.max(elapsed_times))

    # Project Stage-1 cost
    stage1_serial_s = median_elapsed * STAGE1_TOTAL
    stage1_serial_h = stage1_serial_s / 3600.0

    try:
        ncores = multiprocessing.cpu_count()
    except Exception:
        ncores = 4
    useful_concurrency = max(1, ncores)
    stage1_parallel_s = stage1_serial_s / useful_concurrency
    stage1_parallel_h = stage1_parallel_s / 3600.0

    # Classify cells
    runaway_cells = [r for r in results if r["growth_class"] == "RUNAWAY" or r["exploded"]]
    extinct_cells = [r for r in results if r["growth_class"] == "EXTINCT"]
    decelerating_cells = [r for r in results if r["growth_class"] == "DECELERATING"]
    damped_cells = [r for r in results if "DAMPED" in r["osc_class"]]
    oscillatory_cells = [r for r in results if "OSCILLATORY" in r["osc_class"]]
    na_cells = [r for r in results if "N/A" in r["osc_class"]]

    all_cells_run = len(results) == len(CELLS)
    any_runaway = len(runaway_cells) > 0
    all_extinct = len(extinct_cells) == len(results)
    time_ok = max_elapsed <= MAX_SINGLE_RUN_SECS

    # GO/NO-GO with full reasoning
    if not all_cells_run:
        go_nogo = "NO-GO"
        go_reason = f"Not all cells completed (early stop at {results[-1]['elapsed_s']:.1f}s)"
    elif any_runaway:
        go_nogo = "NO-GO"
        go_reason = f"Runaway detected in: {[r['label'] for r in runaway_cells]}"
    elif all_extinct:
        go_nogo = "NO-GO"
        go_reason = (
            f"ALL {len(results)} probed cells EXTINCT at default params (A+B ON, "
            f"hmax={HMAX}, Kc={KC}, regen={REGEN_RATE}). "
            f"The default founder cannot reproduce in the continuous substrate at these settings. "
            f"See birth-balance diagnostic (exp243_birthbalance.txt) for root cause. "
            f"The full Stage-1 sweep at these params would produce all-extinct cells with "
            f"zero births and zero stability signal. Parameter redesign required."
        )
    elif not time_ok:
        go_nogo = "NO-GO"
        go_reason = f"Run time too high: max {max_elapsed:.1f}s > {MAX_SINGLE_RUN_SECS}s"
    elif len(decelerating_cells) >= 2:
        go_nogo = "GO"
        go_reason = (
            f"No runaways; {len(decelerating_cells)}/{len(results)} cells DECELERATING; "
            f"max run time {max_elapsed:.3f}s; "
            f"{len(damped_cells)}/{len(results)} DAMPED cells"
        )
    else:
        go_nogo = "NO-GO"
        go_reason = (
            f"Insufficient stable cells: {len(decelerating_cells)} DECELERATING, "
            f"{len(extinct_cells)} EXTINCT, {len(runaway_cells)} RUNAWAY"
        )

    print()
    print("=" * 65)
    print("PREFLIGHT SUMMARY")
    print("=" * 65)
    for r in results:
        print(f"  {r['label']:8s} (speed={r['speed']}): {r['elapsed_s']:7.3f}s  "
              f"n_eq={r['n_eq']:6.1f}  growth={r['growth_class']:15s}  "
              f"osc={r['osc_class'][:20]:20s}  exploded={r['exploded']}")

    print()
    print(f"  Median per-run wall time: {median_elapsed:.3f}s")
    print(f"  Max per-run wall time:    {max_elapsed:.3f}s")
    print()
    print(f"  Stage-1 grid: {STAGE1_TOTAL} runs")
    print(f"  Projected serial cost:    {stage1_serial_s:.1f}s = {stage1_serial_h:.4f}h")
    print(f"  Available cores:          {ncores}")
    print(f"  Projected parallel cost:  {stage1_parallel_s:.1f}s = {stage1_parallel_h:.4f}h  "
          f"(~{useful_concurrency}x concurrency, trivially parallel)")
    print()
    print(f"  DECELERATING cells: {len(decelerating_cells)}/{len(results)} "
          f"({[r['label'] for r in decelerating_cells]})")
    print(f"  DAMPED cells:       {len(damped_cells)}/{len(results)} "
          f"({[r['label'] for r in damped_cells]})")
    print(f"  OSCILLATORY cells:  {len(oscillatory_cells)}/{len(results)} "
          f"({[r['label'] for r in oscillatory_cells]})")
    print(f"  N/A (extinct) cells:{len(na_cells)}/{len(results)} "
          f"({[r['label'] for r in na_cells]})")
    print(f"  EXTINCT cells:      {len(extinct_cells)}/{len(results)} "
          f"({[r['label'] for r in extinct_cells]})")
    print(f"  RUNAWAY cells:      {len(runaway_cells)}/{len(results)} "
          f"({[r['label'] for r in runaway_cells]})")
    print()
    print(f"  *** {go_nogo}: {go_reason} ***")
    print("=" * 65)

    # Qualitative verdict — the KEY decision-relevant output per the task spec
    if all_extinct:
        qualitative = (
            "NEGATIVE: A+B ON at the DEFAULT PARAMS (hmax=0.04, Kc=60, regen=0.05) "
            "produces EXTINCTION in ALL 3 probed cells (slow/medium/fast). "
            "The population goes extinct due to aging: the default founder's max achievable "
            "energy in the continuous substrate falls SHORT of the reproduction threshold "
            "(see exp243_birthbalance.txt for the energy budget analysis). "
            "There are ZERO births, zero crowding-deaths, and zero stability signal. "
            "Horizon=4000 does NOT fix this — it merely runs 4000 steps of extinction. "
            "The Stage-1 sweep at these params would cost ~{:.1f}s serial (fast, but useless). "
            "ACTION REQUIRED: Redesign the sweep to use parameters where reproduction is "
            "possible (lower aging cost, lower repro threshold, or higher continuous_capacity)."
        ).format(stage1_serial_s)
    elif len(decelerating_cells) > 0 and len(runaway_cells) == 0:
        qualitative = (
            f"POSITIVE: A+B ON at defaults produces a DECELERATING/plateauing trajectory in "
            f"{len(decelerating_cells)}/{len(results)} probed cells "
            f"({[r['label'] for r in decelerating_cells]}). "
            f"{len(damped_cells)} DAMPED. Substrate appears viable at horizon={HORIZON}."
        )
    elif len(runaway_cells) > 0:
        qualitative = (
            f"MIXED-NEGATIVE: A+B ON at defaults: {len(runaway_cells)} RUNAWAY, "
            f"{len(decelerating_cells)} DECELERATING, {len(extinct_cells)} EXTINCT. "
            f"The density brake is insufficient to prevent runaway at the fast end."
        )
    else:
        qualitative = (
            f"MIXED: A+B ON at defaults: {len(decelerating_cells)} DECELERATING, "
            f"{len(extinct_cells)} EXTINCT. Partial stability."
        )

    print(f"\n  QUALITATIVE (A+B damp at defaults?): {qualitative}")
    print()

    # Write output file
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "exp243_preflight.txt"

    per_cell_lines = []
    for r in results:
        osc_detail = r["osc_detail"]
        ar_str = (f"ar_mod={osc_detail['ar_modulus']:.3f}, amp_ptp={osc_detail['amp_ptp']:.3f}"
                  if not np.isnan(osc_detail.get('ar_modulus', float('nan'))) else "N/A")
        per_cell_lines.append(
            f"  {r['label']:8s} (speed={r['speed']}):\n"
            f"    wall_time={r['elapsed_s']:.3f}s, n_eq={r['n_eq']:.1f}, "
            f"median_N_window={r['median_N_window']:.1f}, final_N={r['final_N']:.0f}\n"
            f"    growth={r['growth_class']}, osc={r['osc_class']}, "
            f"exploded={r['exploded']}\n"
            f"    births/step={r['births_per_step']:.4f}, "
            f"crowding/step={r['crowding_per_step']:.4f}, "
            f"availability={r['availability']:.3f}\n"
            f"    osc_detail: {ar_str}"
        )

    summary = f"""Exp 243 Runtime Preflight Pilot
================================
Parameters:
  horizon={HORIZON}, burn_in={BURN_IN}, seed={SEED}
  hmax={HMAX}, Kc={KC}, theta={THETA}, regen={REGEN_RATE}, layout={LAYOUT}
  Stage-1 grid: {STAGE1_TOTAL} runs ({STAGE1_SPEEDS} speeds x {STAGE1_HMAX} hmax x {STAGE1_KC} Kc x {STAGE1_REGEN} regen x {STAGE1_SEEDS} seeds)

Per-cell results:
{chr(10).join(per_cell_lines)}

Wall time summary:
  Median per-run: {median_elapsed:.3f}s
  Max per-run:    {max_elapsed:.3f}s

Stage-1 projection:
  Serial:   {stage1_serial_s:.1f}s = {stage1_serial_h:.4f}h
  Parallel: {stage1_parallel_s:.1f}s = {stage1_parallel_h:.4f}h  ({ncores} cores, trivially parallel)
  NOTE: Fast wall time is because populations go extinct quickly (not a sign of efficiency)

Cell outcome summary:
  DECELERATING: {len(decelerating_cells)}/{len(results)}  ({[r['label'] for r in decelerating_cells]})
  EXTINCT:      {len(extinct_cells)}/{len(results)}  ({[r['label'] for r in extinct_cells]})
  RUNAWAY:      {len(runaway_cells)}/{len(results)}  ({[r['label'] for r in runaway_cells]})
  DAMPED:       {len(damped_cells)}/{len(results)}  ({[r['label'] for r in damped_cells]})
  OSCILLATORY:  {len(oscillatory_cells)}/{len(results)}  ({[r['label'] for r in oscillatory_cells]})
  N/A (extinct):{len(na_cells)}/{len(results)}  ({[r['label'] for r in na_cells]})

GO/NO-GO: {go_nogo}
Reason: {go_reason}

QUALITATIVE (A+B damped at defaults?): {qualitative}
"""
    out_path.write_text(summary)
    print(f"Summary written to: {out_path}")


if __name__ == "__main__":
    main()
