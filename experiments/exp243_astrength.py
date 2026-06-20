"""experiments/exp243_astrength.py — Exp 243: A-strength sweep.

Decisive test of Mechanism A (global density-dependent crowding mortality) as a
stabilizer. The calibration search (commit 61cac9c) found viable continuous
populations at speeds 1.0–2.0 but OSCILLATORY, with NO damped band — however it
fixed A's strength at hmax=0.04, Kc=60, rate_scale=0. A's strength is the actual
stabilizer knob (it substitutes for the resource feedback that vanishes when the
field saturates). This script sweeps A's strength to determine whether a stronger
/ lead-compensated brake DAMPS the oscillation across a speed band, or flips to
period-2 (the lagged-brake failure the design audit predicted).

FIXED REGIME (least-saturated viable from calibration):
  speed_cost_slope=0.05, regen_rate=0.5, layout=bump, theta=1, burn_in=0.6

SEARCH:
1. PRE-CHECK + pathology baseline: run A-OFF (hmax=0.0) at speeds {1.0,1.5,2.0},
   h=2500, seed=A. Confirm the pathology (OSCILLATORY/extinct without A?).
2. A-STRENGTH SWEEP (theta=1): hmax in {0.04,0.10,0.20} × rate_scale in {0.0,0.04}
   × speed in {1.0,1.5,2.0}, seed=A, h=2500. Record DAMPED/OSCILLATORY, n_eq,
   exploded, return_map_slope, marginal_brake, births/step, crowding/step.
3. BAND CHECK: is there ANY (hmax,rate_scale) that damps a contiguous band of >=2
   speeds? If yes, confirm on seed=B.
4. PERIOD-2 DIAGNOSIS: as hmax rises 0.04->0.20, does return_map_slope approach -1
   and |p+N*p'| exceed 0.5? Does rate_scale=0.04 rescue cells rate_scale=0 leaves
   oscillatory?
5. (If time remains and theta=1 finds nothing) probe theta=2 at best (hmax,rate_scale)
   x {1.0,1.5,2.0}, seed=A.

OUTCOME (reported honestly):
  POSITIVE: a (hmax, rate_scale[, theta]) damps a >=2-speed contiguous band on >=2
            seeds -> recommend that regime as the sweep target.
  NEGATIVE-period2: stronger A flips to period-2 (return_map_slope->-1 / marginal-
            brake>=0.5) before it damps -> A cannot damp this substrate.
  NEGATIVE-other: A neither damps nor period-2-flips.

CONSTRAINTS:
  - No engine mutation; events_hash untouched.
  - Bounded compute (~30 min); skip/note cells exceeding ~90s or that explode.
  - DO NOT stage active_loop/cli/converse_demo.py.
  - Commit: only this script + output.

Usage:
  uv run --python .venv python experiments/exp243_astrength.py
"""
from __future__ import annotations

import time
import sys
from pathlib import Path
import numpy as np

from ecology.evolvability.cert_run import run_cert
from ecology.evolvability.stability import (
    oscillation_verdict,
    return_map_slope as _return_map_slope,
    _marginal_brake,
    level_cv,
    n_eq as _n_eq,
)

# ---------------------------------------------------------------------------
# Fixed regime (least-saturated viable from calibration commit 61cac9c)
# calibration found slope=0.05, regen=0.5 gave VIABLE-DAMPED at speed=1.0
# ---------------------------------------------------------------------------
SPEED_COST_SLOPE = 0.05
REGEN_RATE = 0.5
LAYOUT = "bump"
THETA = 1
BURN_IN = 0.6

# Seed naming: A=42 (calibration seed), B=43
SEED_A = 42
SEED_B = 43

# Sweep axes
HMAXES = [0.04, 0.10, 0.20]
RATE_SCALES = [0.0, 0.04]
SPEEDS = [1.0, 1.5, 2.0]
KC = 60.0

# Horizon for main sweep (pre-check will time and potentially shrink)
HORIZON_MAIN = 2500
HORIZON_AOFF = 2500

# Per-cell timeout: flag if > 90s
CELL_TIMEOUT_WARN = 90.0

# Total budget guard: warn if projected to exceed 30 min
BUDGET_SECONDS = 1800

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    _lines.append(msg)


def save_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(_lines) + "\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
EXTINCT_THRESHOLD = 5   # n_eq below this = effectively extinct


def classify_N(N: np.ndarray, exploded: bool, n_eq_val: float) -> dict:
    """Return oscillation_verdict + classification string."""
    if exploded:
        return {"classification": "OSCILLATORY", "_exploded": True}
    if n_eq_val < EXTINCT_THRESHOLD:
        return {"classification": "EXTINCT", "_exploded": False}
    ov = oscillation_verdict(N)
    return ov


def run_cell(speed: float, *, hmax: float, rate_scale: float, seed: int,
             horizon: int, label: str = "") -> dict:
    """Run one cert_run cell and return enriched result dict."""
    t0 = time.monotonic()
    r = run_cert(
        speed,
        hmax=hmax,
        Kc=KC,
        theta=THETA,
        regen_rate=REGEN_RATE,
        rate_scale=rate_scale,
        layout=LAYOUT,
        seed=seed,
        horizon=horizon,
        burn_in=BURN_IN,
        speed_cost_slope=SPEED_COST_SLOPE,
    )
    wall = time.monotonic() - t0
    N = np.asarray(r["N"], dtype=float)
    neq = r["n_eq"]
    exploded = r["exploded"]
    ov = classify_N(N, exploded, neq)
    cls = ov.get("classification", "UNKNOWN")
    # return_map_slope on the window
    rms = float(_return_map_slope(N)) if len(N) > 2 and np.ptp(N) > 0 else 0.0
    # marginal brake at n_eq
    params = {"hmax": hmax, "Kc": KC, "theta": THETA}
    mb = float(_marginal_brake(neq, params))
    return {
        "speed": speed,
        "hmax": hmax,
        "rate_scale": rate_scale,
        "seed": seed,
        "horizon": horizon,
        "n_eq": neq,
        "exploded": exploded,
        "classification": cls,
        "oscillation_detail": ov,
        "return_map_slope": rms,
        "marginal_brake": mb,
        "births_per_step": r["births_per_step"],
        "crowding_per_step": r["crowding_per_step"],
        "wall_s": wall,
        "label": label,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_script_start = time.monotonic()
    output_path = Path("experiments/outputs/exp243_astrength.txt")

    log("=" * 80)
    log("Exp 243 A-strength sweep")
    log("=" * 80)
    log()
    log("Fixed regime: speed_cost_slope=0.05, regen_rate=0.5, layout=bump, theta=1, burn_in=0.6")
    log(f"Seed A={SEED_A}, Seed B={SEED_B}")
    log(f"Sweep: hmax in {HMAXES} × rate_scale in {RATE_SCALES} × speed in {SPEEDS}")
    log()

    # -----------------------------------------------------------------------
    # STEP 1: Pre-check timing and pathology baseline (A OFF)
    # -----------------------------------------------------------------------
    log("=" * 80)
    log("STEP 1: Timing pre-check + A-OFF pathology baseline")
    log("=" * 80)
    log()
    log("Running A-OFF at speeds {1.0, 1.5, 2.0}, horizon=2500, seed=A ...")
    log()

    aoff_results: list[dict] = []
    for spd in SPEEDS:
        log(f"  A-OFF speed={spd} ...")
        r = run_cell(spd, hmax=0.0, rate_scale=0.0, seed=SEED_A,
                     horizon=HORIZON_AOFF, label="A-OFF")
        aoff_results.append(r)
        cls = r["classification"]
        log(f"    -> {cls:15s}  n_eq={r['n_eq']:6.1f}  "
            f"exploded={r['exploded']}  wall={r['wall_s']:.1f}s")

    log()
    log("A-OFF baseline summary:")
    log(f"  {'speed':>6}  {'class':>15}  {'n_eq':>7}  {'exploded':>8}  {'wall_s':>7}")
    log(f"  {'-'*6}  {'-'*15}  {'-'*7}  {'-'*8}  {'-'*7}")
    for r in aoff_results:
        log(f"  {r['speed']:>6.1f}  {r['classification']:>15}  {r['n_eq']:>7.1f}  "
            f"{str(r['exploded']):>8}  {r['wall_s']:>7.1f}s")
    log()

    # Use timing to project budget
    aoff_times = [r["wall_s"] for r in aoff_results]
    mean_aoff_time = float(np.mean(aoff_times)) if aoff_times else 30.0
    n_main_cells = len(HMAXES) * len(RATE_SCALES) * len(SPEEDS)
    projected_s = mean_aoff_time * n_main_cells
    log(f"Timing: A-OFF mean cell time = {mean_aoff_time:.1f}s")
    log(f"Main sweep: {n_main_cells} cells × {mean_aoff_time:.1f}s = projected {projected_s:.0f}s "
        f"({projected_s/60:.1f} min)")

    # Adjust horizon if projected budget exceeds 30 min
    horizon = HORIZON_MAIN
    dropped: list[str] = []
    if projected_s > BUDGET_SECONDS:
        # Shrink horizon proportionally
        factor = BUDGET_SECONDS / projected_s
        horizon = max(1500, int(HORIZON_MAIN * factor))
        dropped.append(f"horizon shrunk from {HORIZON_MAIN} to {horizon} (budget protection)")
        log(f"  WARNING: projected {projected_s:.0f}s > {BUDGET_SECONDS}s budget; "
            f"shrinking horizon to {horizon}")
    else:
        log(f"  Within budget; using horizon={horizon}")
    log()

    # Pathology verdict
    aoff_all_damped = all(r["classification"] == "DAMPED" for r in aoff_results
                          if r["classification"] not in ("EXTINCT",))
    aoff_viable = [r for r in aoff_results if r["classification"] not in ("EXTINCT",)]
    if aoff_all_damped and aoff_viable:
        log("  PATHOLOGY: A-OFF cells are DAMPED — A is NOT needed to damp this substrate.")
        log("  (This would mean A is irrelevant. Report this as the primary finding.)")
        pathology_verdict = "A-NOT-NEEDED"
    elif not aoff_viable:
        log("  PATHOLOGY: All A-OFF speeds extinct — cannot confirm oscillatory without A.")
        pathology_verdict = "ALL-EXTINCT"
    else:
        osc_classes = [r["classification"] for r in aoff_viable]
        log(f"  PATHOLOGY: A-OFF viable cells -> {osc_classes}")
        if all(c == "OSCILLATORY" for c in osc_classes):
            pathology_verdict = "ALL-OSCILLATORY"
            log("  CONFIRMED: removing A leaves viable speeds OSCILLATORY — A is the stabilizer.")
        else:
            pathology_verdict = "MIXED"
            log(f"  MIXED: some A-OFF cells are {set(osc_classes)}.")
    log()

    # -----------------------------------------------------------------------
    # STEP 2: A-strength sweep (theta=1, seed=A)
    # -----------------------------------------------------------------------
    log("=" * 80)
    log("STEP 2: A-strength sweep (theta=1, seed=A)")
    log("=" * 80)
    log()
    log(f"Grid: hmax in {HMAXES} × rate_scale in {RATE_SCALES} × speed in {SPEEDS}")
    log(f"Horizon={horizon}, seed=A={SEED_A}")
    if dropped:
        log(f"DROPPED/ADJUSTED: {'; '.join(dropped)}")
    log()

    sweep_results: list[dict] = []
    total_cells = len(HMAXES) * len(RATE_SCALES) * len(SPEEDS)
    cell_n = 0
    for hmax in HMAXES:
        for rs in RATE_SCALES:
            for spd in SPEEDS:
                cell_n += 1
                log(f"  [{cell_n:02d}/{total_cells}] hmax={hmax:.2f}, rate_scale={rs:.2f}, speed={spd:.1f} ...")
                r = run_cell(spd, hmax=hmax, rate_scale=rs, seed=SEED_A,
                             horizon=horizon, label="sweep-seedA")
                sweep_results.append(r)
                cls = r["classification"]
                log(f"    -> {cls:15s}  n_eq={r['n_eq']:6.1f}  rms={r['return_map_slope']:+.3f}  "
                    f"mb={r['marginal_brake']:.4f}  "
                    f"b/s={r['births_per_step']:.4f}  c/s={r['crowding_per_step']:.4f}  "
                    f"wall={r['wall_s']:.1f}s")
                if r["wall_s"] > CELL_TIMEOUT_WARN:
                    log(f"    WARNING: cell exceeded {CELL_TIMEOUT_WARN}s (wall={r['wall_s']:.1f}s)")
    log()

    # -----------------------------------------------------------------------
    # STEP 3: Band check and full grid table
    # -----------------------------------------------------------------------
    log("=" * 80)
    log("STEP 3: Full grid table + band check")
    log("=" * 80)
    log()

    # Print grid table
    header = (f"{'hmax':>6}  {'rs':>5}  {'speed':>6}  "
              f"{'class':>15}  {'n_eq':>7}  {'rms':>7}  {'mb':>7}  "
              f"{'b/s':>7}  {'c/s':>7}  {'wall_s':>7}")
    log(header)
    log("-" * len(header))
    for r in sweep_results:
        log(f"  {r['hmax']:>4.2f}  {r['rate_scale']:>5.2f}  {r['speed']:>6.1f}  "
            f"  {r['classification']:>15}  {r['n_eq']:>7.1f}  "
            f"{r['return_map_slope']:>+7.3f}  {r['marginal_brake']:>7.4f}  "
            f"{r['births_per_step']:>7.4f}  {r['crowding_per_step']:>7.4f}  "
            f"{r['wall_s']:>7.1f}s")
    log()

    # Identify damped cells
    damped_cells: dict[tuple, list[float]] = {}  # (hmax, rate_scale) -> [damped speeds]
    for r in sweep_results:
        key = (r["hmax"], r["rate_scale"])
        if key not in damped_cells:
            damped_cells[key] = []
        if r["classification"] == "DAMPED":
            damped_cells[key].append(r["speed"])

    log("Band check (contiguous damped speeds per (hmax, rate_scale)):")
    best_band_key = None
    best_band_len = 0
    best_band_speeds: list[float] = []
    for key, damped_spds in sorted(damped_cells.items()):
        damped_spds_sorted = sorted(damped_spds)
        # Find longest contiguous run in {1.0,1.5,2.0}
        run_len = 0
        max_run = 0
        max_run_speeds: list[float] = []
        cur_run: list[float] = []
        for spd in SPEEDS:
            if spd in damped_spds_sorted:
                cur_run.append(spd)
                run_len += 1
                if run_len > max_run:
                    max_run = run_len
                    max_run_speeds = list(cur_run)
            else:
                run_len = 0
                cur_run = []
        hmax_v, rs_v = key
        log(f"  hmax={hmax_v:.2f}, rs={rs_v:.2f}: damped={damped_spds_sorted}, "
            f"max_contiguous={max_run} {max_run_speeds}")
        if max_run > best_band_len:
            best_band_len = max_run
            best_band_key = key
            best_band_speeds = max_run_speeds
    log()

    has_band = best_band_len >= 2
    log(f"Band verdict (seed A): {'BAND FOUND' if has_band else 'NO BAND FOUND'} "
        f"(best contiguous = {best_band_len} at key={best_band_key})")
    log()

    # -----------------------------------------------------------------------
    # STEP 4: Period-2 diagnosis
    # -----------------------------------------------------------------------
    log("=" * 80)
    log("STEP 4: Period-2 diagnosis")
    log("=" * 80)
    log()
    log("Trend of return_map_slope and marginal_brake vs hmax (rate_scale=0.0, by speed):")
    log()
    log(f"  {'hmax':>6}  {'rs':>5}  {'speed':>6}  {'class':>15}  {'rms':>8}  {'mb':>8}  period2?")
    log(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")

    period2_flag: dict[float, list] = {}  # speed -> list of (hmax, rs, rms, mb, p2)
    for r in sweep_results:
        spd = r["speed"]
        hmax_v = r["hmax"]
        rs = r["rate_scale"]
        rms = r["return_map_slope"]
        mb = r["marginal_brake"]
        cls = r["classification"]
        # Period-2 signature: return_map_slope near/below -1 OR marginal_brake >= 0.5
        p2 = (rms <= -0.80 or mb >= 0.5)
        if spd not in period2_flag:
            period2_flag[spd] = []
        period2_flag[spd].append((hmax_v, rs, rms, mb, p2))
        log(f"  {hmax_v:>6.2f}  {rs:>5.2f}  {spd:>6.1f}  {cls:>15}  "
            f"{rms:>+8.3f}  {mb:>8.4f}  {'YES' if p2 else 'no'}")
    log()

    # Summarize period-2 trend per speed
    log("Period-2 trend summary (does rms approach -1 and mb approach/exceed 0.5 as hmax rises?):")
    for spd in SPEEDS:
        entries = sorted(period2_flag.get(spd, []), key=lambda e: (e[0], e[1]))
        log(f"  speed={spd}:")
        for hmax_v, rs, rms, mb, p2 in entries:
            log(f"    hmax={hmax_v:.2f} rs={rs:.2f}: rms={rms:+.3f} mb={mb:.4f}  "
                f"-> {'PERIOD-2 SIGNATURE' if p2 else 'no period-2'}")
    log()

    # rate_scale effect: does rs=0.04 rescue cells rs=0.0 leaves oscillatory?
    log("rate_scale effect (does rs=0.04 rescue rs=0.0 oscillatory cells?):")
    rescued = []
    not_rescued = []
    for hmax_v in HMAXES:
        for spd in SPEEDS:
            r_rs0 = next((r for r in sweep_results
                          if r["hmax"] == hmax_v and r["speed"] == spd and r["rate_scale"] == 0.0), None)
            r_rs04 = next((r for r in sweep_results
                           if r["hmax"] == hmax_v and r["speed"] == spd and r["rate_scale"] == 0.04), None)
            if r_rs0 is None or r_rs04 is None:
                continue
            cls0 = r_rs0["classification"]
            cls04 = r_rs04["classification"]
            if cls0 == "OSCILLATORY" and cls04 == "DAMPED":
                rescued.append((hmax_v, spd))
                log(f"  hmax={hmax_v:.2f} speed={spd:.1f}: rs=0.0 OSCILLATORY -> rs=0.04 DAMPED  [RESCUED]")
            elif cls0 == "DAMPED" and cls04 == "OSCILLATORY":
                log(f"  hmax={hmax_v:.2f} speed={spd:.1f}: rs=0.0 DAMPED -> rs=0.04 OSCILLATORY  [DEGRADED]")
                not_rescued.append((hmax_v, spd))
            else:
                log(f"  hmax={hmax_v:.2f} speed={spd:.1f}: rs=0.0 {cls0} -> rs=0.04 {cls04}  [no change]")
    log()

    # -----------------------------------------------------------------------
    # STEP 3b: Seed-B confirmation (if a band exists)
    # -----------------------------------------------------------------------
    seedB_results: list[dict] = []
    if has_band and best_band_key is not None:
        log("=" * 80)
        log("STEP 3b: Seed-B confirmation of best band")
        log("=" * 80)
        log()
        hmax_best, rs_best = best_band_key
        log(f"Best band: hmax={hmax_best}, rate_scale={rs_best}, speeds={best_band_speeds}")
        log(f"Running seed B={SEED_B} at that (hmax, rate_scale) for all speeds ...")
        log()
        for spd in SPEEDS:
            log(f"  seed-B speed={spd:.1f} ...")
            r = run_cell(spd, hmax=hmax_best, rate_scale=rs_best,
                         seed=SEED_B, horizon=horizon, label="seedB-confirm")
            seedB_results.append(r)
            cls = r["classification"]
            log(f"    -> {cls:15s}  n_eq={r['n_eq']:6.1f}  rms={r['return_map_slope']:+.3f}  "
                f"mb={r['marginal_brake']:.4f}  wall={r['wall_s']:.1f}s")
        log()
        seedB_damped = [r["speed"] for r in seedB_results if r["classification"] == "DAMPED"]
        log(f"Seed-B damped speeds: {sorted(seedB_damped)}")
        seedB_band_ok = len(seedB_damped) >= 2
        log(f"Seed-B band confirmed: {seedB_band_ok} "
            f"(>=2 damped speeds on seed B: {sorted(seedB_damped)})")
        log()
    else:
        seedB_band_ok = False
        log("STEP 3b: Skipped (no band to confirm on seed B)")
        log()

    # -----------------------------------------------------------------------
    # STEP 5: theta=2 probe (only if theta=1 finds nothing and time permits)
    # -----------------------------------------------------------------------
    theta2_results: list[dict] = []
    elapsed_s = time.monotonic() - t_script_start
    if not has_band and (BUDGET_SECONDS - elapsed_s) > 120:
        log("=" * 80)
        log("STEP 5: theta=2 probe (theta=1 found nothing, time permits)")
        log("=" * 80)
        log()
        # Find best (hmax, rate_scale) cell by closest to damped (most damped_count or lowest rms)
        # Use the cell with most DAMPED classifications across speeds
        best_th2_key = None
        best_th2_score = -1
        for hmax_v in HMAXES:
            for rs in RATE_SCALES:
                damped_count = sum(1 for r in sweep_results
                                   if r["hmax"] == hmax_v and r["rate_scale"] == rs
                                   and r["classification"] == "DAMPED")
                if damped_count > best_th2_score:
                    best_th2_score = damped_count
                    best_th2_key = (hmax_v, rs)

        if best_th2_key is not None:
            hmax_th2, rs_th2 = best_th2_key
            log(f"Best (hmax, rate_scale) for theta=2 probe: hmax={hmax_th2}, rs={rs_th2} "
                f"(had {best_th2_score} damped cells at theta=1)")
            log()
            for spd in SPEEDS:
                remaining = BUDGET_SECONDS - (time.monotonic() - t_script_start)
                if remaining < 60:
                    log(f"  Budget nearly exhausted ({remaining:.0f}s left); skipping speed={spd}")
                    break
                log(f"  theta=2 speed={spd:.1f} ...")
                t0 = time.monotonic()
                r2 = run_cert(
                    spd,
                    hmax=hmax_th2,
                    Kc=KC,
                    theta=2,
                    regen_rate=REGEN_RATE,
                    rate_scale=rs_th2,
                    layout=LAYOUT,
                    seed=SEED_A,
                    horizon=horizon,
                    burn_in=BURN_IN,
                    speed_cost_slope=SPEED_COST_SLOPE,
                )
                wall = time.monotonic() - t0
                N2 = np.asarray(r2["N"], dtype=float)
                neq2 = r2["n_eq"]
                exploded2 = r2["exploded"]
                ov2 = classify_N(N2, exploded2, neq2)
                cls2 = ov2.get("classification", "UNKNOWN")
                rms2 = float(_return_map_slope(N2)) if len(N2) > 2 and np.ptp(N2) > 0 else 0.0
                mb2 = float(_marginal_brake(neq2, {"hmax": hmax_th2, "Kc": KC, "theta": 2}))
                theta2_results.append({
                    "speed": spd, "hmax": hmax_th2, "rate_scale": rs_th2, "theta": 2,
                    "n_eq": neq2, "exploded": exploded2, "classification": cls2,
                    "return_map_slope": rms2, "marginal_brake": mb2,
                    "births_per_step": r2["births_per_step"],
                    "crowding_per_step": r2["crowding_per_step"],
                    "wall_s": wall,
                })
                log(f"    -> {cls2:15s}  n_eq={neq2:6.1f}  rms={rms2:+.3f}  "
                    f"mb={mb2:.4f}  wall={wall:.1f}s")
            log()

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    log("=" * 80)
    log("VERDICT")
    log("=" * 80)
    log()

    # Determine overall outcome
    # POSITIVE: a (hmax, rate_scale) damps a >=2-speed contiguous band on >=2 seeds
    # NEGATIVE-period2: stronger A flips to period-2 before damping
    # NEGATIVE-other: neither

    if has_band and seedB_band_ok:
        outcome = "POSITIVE"
        hmax_best, rs_best = best_band_key
        log(f"OUTCOME: POSITIVE")
        log(f"  (hmax={hmax_best}, rate_scale={rs_best}) damps speeds {best_band_speeds} on seed A")
        seedB_damped_list = sorted(r["speed"] for r in seedB_results if r["classification"] == "DAMPED")
        log(f"  Seed B confirms damped: {seedB_damped_list}")
        log(f"  RECOMMENDED regime: hmax={hmax_best}, Kc={KC}, theta={THETA}, "
            f"rate_scale={rs_best}, regen_rate={REGEN_RATE}, speed_cost_slope={SPEED_COST_SLOPE}")
    elif has_band and not seedB_band_ok:
        outcome = "POSITIVE-SEED-A-ONLY"
        hmax_best, rs_best = best_band_key
        log(f"OUTCOME: POSITIVE (seed A only — seed B does not confirm)")
        log(f"  Band at (hmax={hmax_best}, rs={rs_best}): speeds {best_band_speeds} on seed A")
        if seedB_results:
            seedB_damped_list = sorted(r["speed"] for r in seedB_results
                                       if r["classification"] == "DAMPED")
            log(f"  Seed B damped: {seedB_damped_list} (insufficient for >=2-seed confirmation)")
    else:
        # Check period-2 signature
        p2_cells = [(r["hmax"], r["rate_scale"], r["speed"], r["return_map_slope"], r["marginal_brake"])
                    for r in sweep_results
                    if r["return_map_slope"] <= -0.80 or r["marginal_brake"] >= 0.5]
        # Check monotone trend of rms toward -1 as hmax rises
        # Aggregate across speeds: mean rms at each hmax
        rms_by_hmax = {}
        for hmax_v in HMAXES:
            rms_vals = [r["return_map_slope"] for r in sweep_results
                        if r["hmax"] == hmax_v and r["classification"] not in ("EXTINCT",)]
            rms_by_hmax[hmax_v] = float(np.mean(rms_vals)) if rms_vals else float("nan")

        rms_trend_descending = all(
            rms_by_hmax.get(HMAXES[i], 0) >= rms_by_hmax.get(HMAXES[i+1], 0)
            for i in range(len(HMAXES)-1)
            if not (np.isnan(rms_by_hmax.get(HMAXES[i], float("nan")))
                    or np.isnan(rms_by_hmax.get(HMAXES[i+1], float("nan"))))
        )
        mb_by_hmax = {}
        for hmax_v in HMAXES:
            mb_vals = [r["marginal_brake"] for r in sweep_results
                       if r["hmax"] == hmax_v and r["classification"] not in ("EXTINCT",)]
            mb_by_hmax[hmax_v] = float(np.mean(mb_vals)) if mb_vals else float("nan")
        mb_trend_rising = all(
            mb_by_hmax.get(HMAXES[i], 0) <= mb_by_hmax.get(HMAXES[i+1], 0)
            for i in range(len(HMAXES)-1)
            if not (np.isnan(mb_by_hmax.get(HMAXES[i], float("nan")))
                    or np.isnan(mb_by_hmax.get(HMAXES[i+1], float("nan"))))
        )

        log("rms trend by hmax (mean across speeds/rate_scales):")
        for hmax_v in HMAXES:
            log(f"  hmax={hmax_v:.2f}: mean_rms={rms_by_hmax.get(hmax_v, float('nan')):+.3f}  "
                f"mean_mb={mb_by_hmax.get(hmax_v, float('nan')):.4f}")
        log()

        any_p2 = bool(p2_cells)
        has_period2_trend = (rms_trend_descending or mb_trend_rising) and any_p2

        if has_period2_trend:
            outcome = "NEGATIVE-period2"
            log("OUTCOME: NEGATIVE-period2")
            log("  Stronger A shows return_map_slope trending toward -1 and/or marginal_brake "
                "approaching/exceeding 0.5")
            log("  Period-2 signature cells:")
            for hmax_v, rs, spd, rms, mb in p2_cells:
                log(f"    hmax={hmax_v:.2f} rs={rs:.2f} speed={spd:.1f}: rms={rms:+.3f} mb={mb:.4f}")
            log("  CONCLUSION: A cannot damp this substrate — lag is binding; "
                "stronger A flips to period-2 before damping.")
        else:
            outcome = "NEGATIVE-other"
            log("OUTCOME: NEGATIVE-other")
            log("  A neither damps the oscillation nor clearly shows period-2 flip.")
            log("  Summary: A shifts n_eq but does not stabilize dynamics in this regime.")
            if p2_cells:
                log(f"  Weak period-2 signatures: {len(p2_cells)} cells (but trend not monotone).")
            log("  rms trend descending:", rms_trend_descending)
            log("  mb trend rising:", mb_trend_rising)

    log()
    log("A-OFF pathology baseline: " + pathology_verdict)
    if pathology_verdict == "A-NOT-NEEDED":
        log("  IMPORTANT: A-OFF cells are DAMPED — the fixed regime damps WITHOUT A.")
        log("  This undermines the A-as-stabilizer hypothesis in this regime.")
    log()

    # Print period-2 trend table
    log("-" * 80)
    log("Period-2 trend table (return_map_slope at each hmax, for rate_scale=0.0):")
    log(f"  {'hmax':>6}  {'speed':>6}  {'class':>15}  {'rms':>8}  {'mb':>8}  period2?")
    log(f"  {'-'*6}  {'-'*6}  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in sorted(sweep_results, key=lambda r: (r["rate_scale"], r["hmax"], r["speed"])):
        if r["rate_scale"] != 0.0:
            continue
        rms = r["return_map_slope"]
        mb = r["marginal_brake"]
        cls = r["classification"]
        p2 = rms <= -0.80 or mb >= 0.5
        log(f"  {r['hmax']:>6.2f}  {r['speed']:>6.1f}  {cls:>15}  "
            f"{rms:>+8.3f}  {mb:>8.4f}  {'YES' if p2 else 'no'}")
    log()

    if theta2_results:
        log("-" * 80)
        log("theta=2 probe results (Step 5):")
        log(f"  {'hmax':>6}  {'rs':>5}  {'speed':>6}  {'class':>15}  {'rms':>8}  {'mb':>8}")
        log(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*15}  {'-'*8}  {'-'*8}")
        for r in theta2_results:
            log(f"  {r['hmax']:>6.2f}  {r['rate_scale']:>5.2f}  {r['speed']:>6.1f}  "
                f"  {r['classification']:>15}  {r['return_map_slope']:>+8.3f}  "
                f"{r['marginal_brake']:>8.4f}")
        log()

    total_wall = time.monotonic() - t_script_start
    log(f"Total wall time: {total_wall:.1f}s ({total_wall/60:.1f} min)")
    log(f"Script finished.")
    log()

    save_output(output_path)
    log(f"Output written to: {output_path}")
    print(f"\nFinal outcome: {outcome}", flush=True)


if __name__ == "__main__":
    main()
