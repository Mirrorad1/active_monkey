"""Exp 189 — N4 Variable-Length Separation Test (RUNG-6 FALSIFIABLE).

PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit 5818b22) BEFORE any data.
Status: FALSIFIABLE — predeclared falsifiers below; fresh seeds 280-287.

Question: does online tempo learning finally BIND — and beat every constant — where
burst lengths VARY such that no revision-safe constant covers?  Controller: REG-TB
VERBATIM from exp188 (KAPPA=1.5, T0=2800, pressure-gated, gap-spanning calm 2600;
NO retuning of any kind).

predeclar (5818b22):
  P1 (THE SEPARATION): in >= 2 cells (predicted E1 and E3) REG-TB passes BOTH bars while
     NO constant arm (C2900, C3500, CALM2600-H3000) passes both. F1: zero separation
     cells — the ambiguity bound extends to variable-L.
  P2 (revision invariance): REG-TB Phase-R latency ~2875 <= baseline+3000 (no completed
     stretches in permanent pressure -> threshold stays T0=2800). F2: REG-TB fails revision.
  P3 (the kappa-reach law): REG-TB's six cell-level defense signs land exactly as
     predicted (E1 pass, E3 pass, C1 pass, E2 fail, E4 fail, D1 fail). F3: >= 2 cells off.

falsifier bindings (ordered per pre-registration):
  F1: zero separation cells.
  F2: REG-TB fails revision (revision_pass_normal == False for REG-TB).
  F3: >= 2 predicted cell signs are wrong.

Cells (6; settle 6000, exogenous attack color = argmin(v) at step 6000 fixed for all
bursts, 2500-step tail, captivity mechanics verbatim):
  E1 escalating: L-seq (1200, 2400, 3200), G=600 — REG-TB PASS predicted.
  E2 doubling-jump: (800, 1600, 3200), G=600 — REG-TB FAIL predicted.
  E3 escalating wider gaps: (1200, 2400, 3200), G=1200 — REG-TB PASS predicted.
  E4 doubling train: (600, 1200, 2400, 4800), G=600 — REG-TB FAIL predicted.
  D1 descending: (3200, 2400, 1200), G=600 — REG-TB FAIL predicted.
  C1 fixed control: (2400, 2400, 2400), G=600 — REG-TB PASS and C2900 PASS predicted.

Arms (6): baseline; oracle (exact-train freeze, witness); INT-C2900 (predicted: fails
  defense in E1-E4 by mid-long-burst concession, passes C1, fails D1); INT-C3500 (the
  deliberately super-tolerance constant — predicted: DEFENDS E1/E3/D1-first-burst cells
  where max stretch 3325 < 3500 but FAILS revision: latency ~3575 > baseline+3000);
  CALM2600-H3000 (frozen-time best — predicted: concedes mid-train wherever total train
  frozen time > 3000); REG-TB.

Seeds FRESH 280-287. W: 6 cells x 6 arms x 8 = 288; R: 5 non-oracle arms x 8 = 40.
Bars VERBATIM exp186/188: defense >= 6/8 (final-window displacement-rejection over
[last_bend+1500, last_bend+2000)); revision >= 6/8 (latency <= same-seed baseline +
3000 normal; +1500 tight and +6000 loose secondary); oracle >= 7/8; baseline deficit
>= 7/8; PC1 < 0.15 gated; NO mid-run adjustments.

Outputs: experiments/outputs/exp189_rows.json (JSONL), exp189.txt (gate evidence
table; per-cell x arm table; REG-TB stretch/threshold traces per cell; the
separation table; P1/P2/P3 verdicts; tight/loose secondary), script
experiments/exp189_n4_variable_length.py.
"""
from __future__ import annotations

import collections
import copy
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Import the exp188 runner and helpers via importlib (zero drift from upstream)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec188 = _ilu.spec_from_file_location(
    "exp188",
    str(REPO_ROOT / "experiments" / "exp188_n4_regulated_controller.py"),
)
_mod188 = _ilu.module_from_spec(_spec188)
_spec188.loader.exec_module(_mod188)  # type: ignore[union-attr]

# Pull the runner and helpers from exp188
run_fork_schedule_188 = _mod188.run_fork_schedule_188
compute_defense = _mod188.compute_defense
compute_final_gap = _mod188.compute_final_gap
phase_r_latency = _mod188.phase_r_latency
to_plain = _mod188.to_plain
pi_of = _mod188.pi_of
run_equivalence_gate_188 = _mod188.run_equivalence_gate_188

# Constants from exp188/exp185
LAMBDA = _mod188.LAMBDA
INIT_MASS = _mod188.INIT_MASS
PHASE_R_START = _mod188.PHASE_R_START
BURST_SEED_OFFSET_R = _mod188.BURST_SEED_OFFSET_R
BURST_SEED_OFFSET_W_EXP183 = _mod188.BURST_SEED_OFFSET_W_EXP183
EVAL = _mod188.EVAL
FINE_EVAL = _mod188.FINE_EVAL
N_STEPS_PHASE_R = _mod188.N_STEPS_PHASE_R
CHUNK_SIZE = _mod188.CHUNK_SIZE
P6_HOLD = _mod188.P6_HOLD
DEFENSE_FRAC_THRESH = _mod188.DEFENSE_FRAC_THRESH
DEFENSE_WINDOW_OFFSET_START = _mod188.DEFENSE_WINDOW_OFFSET_START
DEFENSE_WINDOW_OFFSET_END = _mod188.DEFENSE_WINDOW_OFFSET_END
PC1_AHAT_DRIFT_MAX = _mod188.PC1_AHAT_DRIFT_MAX

DEFAULT_THETA = _mod188.DEFAULT_THETA
DEFAULT_RELEASE_CALM_SNAPS = _mod188.DEFAULT_RELEASE_CALM_SNAPS
CALM2600_SNAPS = _mod188.CALM2600_SNAPS
THETA3_THETA = _mod188.THETA3_THETA
E_STAR = _mod188.E_STAR
FLOAT_ATOL = _mod188.FLOAT_ATOL

EXP183_BURST_WINDOWS = _mod188.EXP183_BURST_WINDOWS
EXP183_N_STEPS = _mod188.EXP183_N_STEPS
EXP183_GATE_SEEDS = _mod188.EXP183_GATE_SEEDS
EXP183_GATE_ARMS = _mod188.EXP183_GATE_ARMS

PRESSURE_WINDOW = _mod188.PRESSURE_WINDOW   # 200
PRESSURE_FRAC = _mod188.PRESSURE_FRAC       # 0.6

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Exp 189 configuration
# ---------------------------------------------------------------------------

SEEDS_189 = [280, 281, 282, 283, 284, 285, 286, 287]
N_SEEDS = len(SEEDS_189)

# Confirmation bars (verbatim exp186/188)
DEFENSE_MIN_PASS = 6    # >= 6/8 seeds
ORACLE_DEFENSE_MIN = 7  # >= 7/8
BASELINE_FAIL_MIN = 7   # >= 7/8 must fail for deficit

# Revision modes: primary = normal (+3000); tight/loose secondary diagnostics
REVISION_MODES = [
    ("normal", 3000),
    ("tight",  1500),
    ("loose",  6000),
]

# REG-TB constants (verbatim exp188 — NO retuning)
REG_TB_KAPPA = 1.5
REG_TB_T0 = 2800
INT_CALM_SNAPS = CALM2600_SNAPS   # 104 snaps = 2600 steps

# CALM2600-H3000 parameters
CALM2600_H3000_H = 3000

# ---------------------------------------------------------------------------
# Variable-length cells (6)
# Each cell has a list of burst lengths L_seq and a gap G.
# L_seq defines the burst durations in order.
# ---------------------------------------------------------------------------

def make_cell(name: str, L_seq: list, G: int, regtb_pred: str) -> dict:
    """Build a cell descriptor for variable-length bursts."""
    n_bursts = len(L_seq)
    total_burst = sum(L_seq)
    total_gaps = (n_bursts - 1) * G
    n_steps = 6000 + total_burst + total_gaps + 2500
    # Build burst windows
    windows = []
    cur = 6000
    for L in L_seq:
        windows.append((cur, cur + L))
        cur += L + G  # advance by L then gap (last gap doesn't matter)
    # Remove the trailing gap from the last burst's endpoint tracking
    # (windows already computed correctly above)
    cell_tag = f"{name}:({','.join(str(l) for l in L_seq)},G={G})"
    return {
        "name": name,
        "L_seq": L_seq,
        "G": G,
        "n_bursts": n_bursts,
        "n_steps": n_steps,
        "burst_windows": windows,
        "cell_tag": cell_tag,
        "regtb_pred": regtb_pred,  # "pass" or "fail"
    }


CELLS_189 = [
    make_cell("E1", [1200, 2400, 3200], 600,  "pass"),
    make_cell("E2", [800,  1600, 3200], 600,  "fail"),
    make_cell("E3", [1200, 2400, 3200], 1200, "pass"),
    make_cell("E4", [600,  1200, 2400, 4800], 600, "fail"),
    make_cell("D1", [3200, 2400, 1200], 600,  "fail"),
    make_cell("C1", [2400, 2400, 2400], 600,  "pass"),
]

assert len(CELLS_189) == 6, f"Expected 6 cells, got {len(CELLS_189)}"

# Verify n_steps are divisible by CHUNK_SIZE
for _c in CELLS_189:
    assert _c["n_steps"] % CHUNK_SIZE == 0, (
        f"Cell {_c['name']}: n_steps={_c['n_steps']} not divisible by CHUNK_SIZE={CHUNK_SIZE}"
    )

# ---------------------------------------------------------------------------
# Arm definitions (6 arms total)
# arm_mode conventions reuse exp188 conventions:
#   "baseline"                              — standard
#   "oracle"                                — oracle freeze (exact burst windows)
#   ("reg_tb", kappa, T0, calm_snaps)      — REG-TB regulated controller
#   ("int_c", C, calm_snaps)               — continuous stopwatch (INT-C2900)
#   ("freeze_time", H)                     — fixed-H frozen-time arm (for CALM2600-H3000)
#
# For CALM2600-H3000: freeze_time mode with H=3000 and release_calm_snaps=CALM2600_SNAPS
# For INT-C3500: int_c mode with C=3500 and INT_CALM_SNAPS
# ---------------------------------------------------------------------------

ARM_DEFS = []
ARM_DEFS.append(("baseline",        "baseline",                                             DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("oracle",          "oracle",                                               DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("INT-C2900",       ("int_c", 2900, INT_CALM_SNAPS),                        DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("INT-C3500",       ("int_c", 3500, INT_CALM_SNAPS),                        DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("CALM2600-H3000",  ("freeze_time", CALM2600_H3000_H),                      DEFAULT_THETA, CALM2600_SNAPS))
ARM_DEFS.append(("REG-TB",          ("reg_tb", REG_TB_KAPPA, REG_TB_T0, INT_CALM_SNAPS),    DEFAULT_THETA, INT_CALM_SNAPS))

assert len(ARM_DEFS) == 6, f"Expected 6 arms, got {len(ARM_DEFS)}"

# R arms: all 5 non-oracle arms
R_ARM_DEFS = [(name, mode, theta, rcs) for name, mode, theta, rcs in ARM_DEFS
              if name != "oracle"]
assert len(R_ARM_DEFS) == 5, f"Expected 5 R arms, got {len(R_ARM_DEFS)}"


def tv_func(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 80)
    print("Exp 189 — N4 Variable-Length Separation Test (RUNG-6 FALSIFIABLE)")
    print("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit 5818b22)")
    print(f"Seeds: {SEEDS_189} | 6 cells x 6 arms x 8 seeds = 288 W | 5 R-arms x 8 = 40 R")
    print("=" * 80)
    print()

    # Spine safety (L14)
    mirro = Creature.load("creature/state/mirro")
    mirro_root = copy.deepcopy(mirro)
    mirro_root._state_dir = None

    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    n_cells_world = mirro.world.n_cells
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={n_colors}, n_cells={n_cells_world}")

    vc = mirro.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v_spine = (vc / vc_sum) * INIT_MASS
    else:
        v_spine = np.ones(n_colors) * (INIT_MASS / n_colors)
    spine_argmax = int(np.argmax(v_spine))
    spine_argmin = int(np.argmin(v_spine))
    assert spine_argmax != spine_argmin, "argmin == argmax — degenerate spine"
    print(f"Spine standing favorite: color {spine_argmax}, attack candidate: color {spine_argmin}")
    print()

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows_path = out_dir / "exp189_rows.json"
    out_txt_path = out_dir / "exp189.txt"
    committed_183_path = out_dir / "exp183_rows.json"

    print("CELLS (6 variable-length cells):")
    for ci, cell in enumerate(CELLS_189):
        print(f"  [{ci}] {cell['name']:3s}  L_seq={cell['L_seq']}  G={cell['G']}  "
              f"n_steps={cell['n_steps']}  pred={cell['regtb_pred']}")
    print()
    print("ARMS (6):")
    for arm_name, arm_mode, theta, rcs in ARM_DEFS:
        print(f"  {arm_name:20s}  theta={theta}  rcs={rcs}  mode={arm_mode}")
    print()
    print(f"REG-TB constants (verbatim exp188): KAPPA={REG_TB_KAPPA}, T0={REG_TB_T0}, calm_snaps={INT_CALM_SNAPS}")
    print()

    # ====================================================================
    # STEP 1: EQUIVALENCE GATE (L15)
    # Re-run exp183 (baseline x s229) and (H1200 x s229) through exp188 code path
    # (which this script delegates to verbatim)
    # ====================================================================
    print("=" * 80)
    print("STEP 1: EQUIVALENCE GATE (L15)")
    print("Reproducing exp183 (baseline x s229) and (H1200 x s229) via exp188 code path")
    print("=" * 80)
    t_gate = time.time()

    gate_pass, gate_detail = run_equivalence_gate_188(
        mirro_root, base_cmap, n_colors, committed_183_path
    )

    print()
    print(gate_detail)
    print(f"Gate time: {time.time()-t_gate:.1f}s")
    print()

    if not gate_pass:
        print("EQUIVALENCE GATE FAIL — aborting. Grid NOT run.")
        with open(out_txt_path, "w") as f:
            f.write("EQUIVALENCE GATE FAIL\n\n")
            f.write(gate_detail + "\n")
        return

    print("EQUIVALENCE GATE PASS — proceeding to grid.")
    print()

    # ====================================================================
    # STEP 2: W SESSIONS (6 cells x 6 arms x 8 seeds = 288)
    # ====================================================================
    print("=" * 80)
    print("STEP 2: W SESSIONS (6 cells x 6 arms x 8 seeds = 288 sessions)")
    print("=" * 80)

    defense_by: dict = {}
    pc1_flags: dict = {}
    rows_buffer: list[dict] = []

    # REG-TB per-session stretch/threshold diagnostic storage
    # regtb_traces[(cell_name, seed)] = list of per-burst dicts
    regtb_traces: dict = {}

    t_grid = time.time()
    n_total = len(CELLS_189) * len(ARM_DEFS) * N_SEEDS
    n_done = 0

    for ci, cell in enumerate(CELLS_189):
        cell_name = cell["name"]
        cell_tag = cell["cell_tag"]
        L_seq = cell["L_seq"]
        G = cell["G"]
        n_steps = cell["n_steps"]
        burst_windows = cell["burst_windows"]
        n_bursts = cell["n_bursts"]

        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in SEEDS_189:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

                reloc_rng_seed = 280_000 + 10_000 * ci + seed

                rr = run_fork_schedule_188(
                    mirro=root,
                    fork_seed=seed,
                    base_cmap=base_cmap,
                    n_colors=n_colors,
                    arm_name=arm_name,
                    arm_mode=arm_mode,
                    phase="W",
                    burst_windows=burst_windows,
                    color_mode="exogenous_fixed",
                    reloc_rng_seed=reloc_rng_seed,
                    n_steps=n_steps,
                    theta=theta_val,
                    release_calm_snaps=rc_snaps,
                )

                defense, frac, attack_color = compute_defense(
                    rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps
                )
                defense_by[(cell_tag, arm_name, seed)] = bool(defense)
                pc1_flag = rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX
                pc1_flags[(cell_tag, arm_name, seed)] = pc1_flag

                final_gap = compute_final_gap(rr["v_traj"], rr["expressed_arr"], n_steps)
                per_burst = []
                for bi, (bstart, bend) in enumerate(burst_windows):
                    pb = {
                        "bi": bi,
                        "L": L_seq[bi],
                        "bstart": bstart,
                        "bend": bend,
                        "gap_start": rr["gap_start"][bi],
                        "gap_end": rr["gap_end"][bi],
                        "expr_frac_or_null": rr["per_burst_expr_frac"][bi],
                    }
                    per_burst.append(pb)

                quiet_check = int(rr.get("quiet_accumulation_check", 0))
                assert quiet_check == 0, (
                    f"QUIET ACCUMULATION VIOLATION: arm={arm_name} cell={cell_tag} seed={seed} "
                    f"violations={quiet_check}"
                )

                # REG-TB per-session stretch tracing (for kappa-reach law audit)
                if arm_name == "REG-TB":
                    events_summary_regtb = rr.get("events", [])
                    stretch_log_sess = rr.get("stretch_log", [])
                    concession_events = [e for e in events_summary_regtb
                                        if e.get("label") == "concession"]
                    burst_trace = []
                    for bi, (bstart, bend) in enumerate(burst_windows):
                        # Find concession events that started inside this burst
                        burst_step_set = set(range(bstart, bend))
                        burst_concessions = [e for e in concession_events
                                             if e.get("entry_step") in burst_step_set]
                        burst_trace.append({
                            "burst_idx": bi,
                            "L": L_seq[bi],
                            "bstart": bstart,
                            "bend": bend,
                            "n_concessions_in_burst": len(burst_concessions),
                            "concession_int_acc": [e.get("int_acc_at_release") for e in burst_concessions],
                            "S_max_at_concession": [e.get("reg_S_max_at_release") for e in burst_concessions],
                        })
                    regtb_traces[(cell_name, seed)] = {
                        "stretch_log": stretch_log_sess,
                        "S_max_final": rr.get("S_max_final", 0.0),
                        "max_current_stretch": rr.get("max_current_stretch", 0.0),
                        "n_completed_stretches": rr.get("n_completed_stretches", 0),
                        "burst_trace": burst_trace,
                        "defended": bool(defense),
                    }

                row = {
                    "exp": 189,
                    "kind": "W",
                    "cell": {
                        "name": cell_name,
                        "L_seq": L_seq,
                        "G": G,
                        "tag": cell_tag,
                        "regtb_pred": cell["regtb_pred"],
                    },
                    "arm": arm_name,
                    "seed": int(seed),
                    "attack_color": to_plain(attack_color),
                    "defense": bool(defense),
                    "final_expr_frac": to_plain(frac),
                    "final_gap": to_plain(final_gap),
                    "per_burst": to_plain(per_burst),
                    "n_events": int(len(rr["events"])),
                    "events_summary": to_plain([
                        {
                            "label": e["label"],
                            "entry_step": e["entry_step"],
                            "frozen_steps": e["frozen_steps"],
                            "E_blocked": e["E_blocked"],
                            "c_star": e["c_star"],
                            "trigger_latency": e.get("trigger_latency"),
                            "int_acc_at_release": e.get("int_acc_at_release"),
                            "reg_S_max_at_release": e.get("reg_S_max_at_release"),
                        }
                        for e in rr["events"]
                    ]),
                    "ahat_drift": float(rr["ahat_drift"]),
                    "settle_tv": to_plain(rr.get("settle_tv")),
                    "flags": ["PC1_DRIFT"] if pc1_flag else [],
                    "theta": theta_val,
                    "release_calm_snaps": rc_snaps,
                    "n_resets": int(rr.get("n_resets", 0)),
                    "quiet_accumulation_check": quiet_check,
                    "concession_events_in_train": bool(rr.get("concession_events_in_train", False)),
                    "mean_concession_step": to_plain(rr.get("mean_concession_step")),
                    "n_completed_stretches": int(rr.get("n_completed_stretches", 0)),
                    "S_max_final": float(rr.get("S_max_final", 0.0)),
                    "max_current_stretch": float(rr.get("max_current_stretch", 0.0)),
                    "stretch_log": to_plain(rr.get("stretch_log", [])),
                }
                rows_buffer.append(row)
                n_done += 1

        elapsed = time.time() - t_grid
        rate = elapsed / max(1, n_done)
        remaining = rate * (n_total - n_done)
        print(
            f"  cell {ci+1:02d}/{len(CELLS_189)} ({cell_name}) "
            f"| elapsed {elapsed:.0f}s | ETA {remaining:.0f}s",
            flush=True,
        )

    t_grid_done = time.time()
    print(f"W grid done: {(t_grid_done - t_grid)/60:.1f} min total")
    print()

    # ====================================================================
    # STEP 3: R SESSIONS (5 non-oracle arms x 8 seeds = 40)
    # ====================================================================
    print("=" * 80)
    print("STEP 3: R SESSIONS (5 non-oracle arms x 8 seeds = 40 sessions)")
    print("baseline R latencies are the tolerance reference")
    print("=" * 80)

    r_latencies: dict = {}
    r_rows_buffer: list[dict] = []

    t_r = time.time()
    for arm_label, arm_mode_r, theta_val, rc_snaps in R_ARM_DEFS:
        for seed in SEEDS_189:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None

            rr = run_fork_schedule_188(
                mirro=root,
                fork_seed=seed,
                base_cmap=base_cmap,
                n_colors=n_colors,
                arm_name=arm_label,
                arm_mode=arm_mode_r,
                phase="R",
                burst_windows=[],
                color_mode="endogenous",
                reloc_rng_seed=BURST_SEED_OFFSET_R + seed,
                n_steps=N_STEPS_PHASE_R,
                theta=theta_val,
                release_calm_snaps=rc_snaps,
            )

            rc = rr["regime_color"]
            lat = None
            if rc is not None:
                lat = phase_r_latency(rr["expressed_arr"], rc, N_STEPS_PHASE_R)
            r_latencies[(arm_label, seed)] = lat

            quiet_check_r = int(rr.get("quiet_accumulation_check", 0))
            assert quiet_check_r == 0, (
                f"QUIET ACCUMULATION VIOLATION in R: arm={arm_label} seed={seed} "
                f"violations={quiet_check_r}"
            )

            print(
                f"  arm={arm_label:22s} seed={seed}  "
                f"regime_color={rc}  latency={lat}  "
                f"n_resets={rr.get('n_resets', 0)}  "
                f"quiet_check={quiet_check_r}  "
                f"ahat_drift={rr['ahat_drift']:.5f}",
                flush=True,
            )

            r_row = {
                "exp": 189,
                "kind": "R",
                "arm": arm_label,
                "seed": int(seed),
                "latency": to_plain(lat),
                "n_events": int(len(rr["events"])),
                "ahat_drift": float(rr["ahat_drift"]),
                "flags": [],
                "theta": theta_val,
                "release_calm_snaps": rc_snaps,
                "n_resets": int(rr.get("n_resets", 0)),
                "quiet_accumulation_check": quiet_check_r,
                "n_completed_stretches": int(rr.get("n_completed_stretches", 0)),
                "S_max_final": float(rr.get("S_max_final", 0.0)),
                "max_current_stretch": float(rr.get("max_current_stretch", 0.0)),
                "stretch_log": to_plain(rr.get("stretch_log", [])),
            }
            r_rows_buffer.append(r_row)

    print(f"Phase-R done: {time.time()-t_r:.1f}s")
    print()

    # ====================================================================
    # STEP 4: WRITE W+R ROWS TO JSONL
    # ====================================================================
    with open(out_rows_path, "w") as fh:
        for row in rows_buffer:
            fh.write(json.dumps(row) + "\n")
        for row in r_rows_buffer:
            fh.write(json.dumps(row) + "\n")
    print(f"W+R rows written to {out_rows_path}")
    print()

    # ====================================================================
    # STEP 5: ANALYSIS
    # ====================================================================
    print("=" * 80)
    print("STEP 5: ANALYSIS")
    print("=" * 80)

    # ---- 5a. Defense counts per (cell_tag, arm_name) ----
    defense_counts: dict = {}
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        for arm_name, _, _, _ in ARM_DEFS:
            n_pass = sum(
                1 for seed in SEEDS_189
                if defense_by.get((cell_tag, arm_name, seed), False)
            )
            defense_counts[(cell_tag, arm_name)] = n_pass

    defense_pass_map: dict = {}
    for (cell_tag, arm_name), cnt in defense_counts.items():
        defense_pass_map[(cell_tag, arm_name)] = cnt >= DEFENSE_MIN_PASS

    # ---- 5b. Revision pass per arm ----
    revision_pass: dict = {}
    for arm_name, _, _, _ in R_ARM_DEFS:
        for mode_name, tolerance in REVISION_MODES:
            n_pass = 0
            for seed in SEEDS_189:
                bl_lat = r_latencies.get(("baseline", seed))
                arm_lat = r_latencies.get((arm_name, seed))
                if arm_lat is None or bl_lat is None:
                    continue
                if arm_lat <= bl_lat + tolerance:
                    n_pass += 1
            revision_pass[(arm_name, mode_name)] = n_pass >= DEFENSE_MIN_PASS

    # ---- 5c. Per cell: both_pass per arm ----
    cell_arm_results: dict = {}
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        for arm_name, _, _, _ in ARM_DEFS:
            if arm_name == "oracle":
                cell_arm_results[(cell_tag, arm_name)] = {
                    "defense_count": defense_counts[(cell_tag, arm_name)],
                    "defense_pass": defense_pass_map.get((cell_tag, arm_name), False),
                }
                continue
            d_ok = defense_pass_map.get((cell_tag, arm_name), False)
            r_ok_normal = revision_pass.get((arm_name, "normal"), False)
            r_ok_tight  = revision_pass.get((arm_name, "tight"),  False)
            r_ok_loose  = revision_pass.get((arm_name, "loose"),  False)
            cell_arm_results[(cell_tag, arm_name)] = {
                "defense_count": defense_counts[(cell_tag, arm_name)],
                "defense_pass": d_ok,
                "revision_pass_normal": r_ok_normal,
                "revision_pass_tight":  r_ok_tight,
                "revision_pass_loose":  r_ok_loose,
                "both_pass_normal": d_ok and r_ok_normal,
                "both_pass_tight":  d_ok and r_ok_tight,
                "both_pass_loose":  d_ok and r_ok_loose,
            }

    # ---- 5d. PC1 summary ----
    pc1_total = sum(1 for v in pc1_flags.values() if v)
    pc1_note = f"{pc1_total}/{len(pc1_flags)} sessions flagged ahat_drift >= {PC1_AHAT_DRIFT_MAX}"

    # ---- 5e. Oracle + baseline checks per cell ----
    oracle_ok: dict = {}
    baseline_deficit: dict = {}
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        oracle_cnt = defense_counts[(cell_tag, "oracle")]
        bl_pass_cnt = defense_counts[(cell_tag, "baseline")]
        oracle_ok[cell_tag] = oracle_cnt >= ORACLE_DEFENSE_MIN
        baseline_deficit[cell_tag] = (N_SEEDS - bl_pass_cnt) >= BASELINE_FAIL_MIN

    # ---- 5f. REG-TB per-cell results ----
    regtb_cells_pass_normal: list[str] = []
    regtb_cells_pass_tight: list[str] = []
    regtb_cells_pass_loose: list[str] = []
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        ar = cell_arm_results.get((cell_tag, "REG-TB"), {})
        if ar.get("both_pass_normal", False):
            regtb_cells_pass_normal.append(cell_tag)
        if ar.get("both_pass_tight", False):
            regtb_cells_pass_tight.append(cell_tag)
        if ar.get("both_pass_loose", False):
            regtb_cells_pass_loose.append(cell_tag)

    # Map cell names for display
    regtb_cell_names_pass = [cell["name"] for cell in CELLS_189
                              if cell["cell_tag"] in regtb_cells_pass_normal]

    # ---- 5g. Constant arms that pass both bars per cell ----
    constant_arms = ["INT-C2900", "INT-C3500", "CALM2600-H3000"]

    def any_constant_passes(cell_tag: str) -> bool:
        """Returns True if any constant arm passes both bars (normal) in this cell."""
        for carm in constant_arms:
            ar = cell_arm_results.get((cell_tag, carm), {})
            if ar.get("both_pass_normal", False):
                return True
        return False

    # ---- 5h. P1: SEPARATION — cells where REG-TB passes both and no constant does ----
    separation_cells: list[str] = []
    separation_cell_names: list[str] = []
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        regtb_passes = cell_arm_results.get((cell_tag, "REG-TB"), {}).get("both_pass_normal", False)
        if regtb_passes and not any_constant_passes(cell_tag):
            separation_cells.append(cell_tag)
            separation_cell_names.append(cell["name"])

    p1_n_separation = len(separation_cells)
    p1_pass = p1_n_separation >= 2
    p1_f1_fired = p1_n_separation == 0

    # ---- 5i. P2: revision invariance ----
    regtb_rev_normal = revision_pass.get(("REG-TB", "normal"), False)
    regtb_rev_tight  = revision_pass.get(("REG-TB", "tight"),  False)
    regtb_rev_loose  = revision_pass.get(("REG-TB", "loose"),  False)
    p2_pass = regtb_rev_normal
    p2_f2_fired = not regtb_rev_normal

    # REG-TB Phase-R latencies for all seeds
    regtb_r_lats = [r_latencies.get(("REG-TB", s)) for s in SEEDS_189]
    bl_lats = [r_latencies.get(("baseline", s)) for s in SEEDS_189]
    regtb_phase_r_latencies_all = [l for l in regtb_r_lats if l is not None]

    # Revision pass counts (how many seeds pass)
    regtb_rev_n_normal = sum(
        1 for seed in SEEDS_189
        if (r_latencies.get(("REG-TB", seed)) is not None and
            r_latencies.get(("baseline", seed)) is not None and
            r_latencies.get(("REG-TB", seed)) <= r_latencies.get(("baseline", seed)) + 3000)
    )

    # ---- 5j. P3: kappa-reach law — predicted vs actual per cell ----
    # Predictions: E1=pass, E2=fail, E3=pass, E4=fail, D1=fail, C1=pass
    kappa_reach_predictions = {c["name"]: c["regtb_pred"] for c in CELLS_189}
    kappa_reach_actuals = {}
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        ar = cell_arm_results.get((cell_tag, "REG-TB"), {})
        # defense_pass is what we compare to prediction
        kappa_reach_actuals[cell["name"]] = "pass" if ar.get("defense_pass", False) else "fail"

    p3_cell_results = {}
    n_cells_correct = 0
    for cname, pred in kappa_reach_predictions.items():
        actual = kappa_reach_actuals[cname]
        correct = (pred == actual)
        p3_cell_results[cname] = {"predicted": pred, "actual": actual, "correct": correct}
        if correct:
            n_cells_correct += 1

    n_cells_wrong = 6 - n_cells_correct
    p3_pass = n_cells_wrong < 2  # F3: >= 2 cells off
    p3_f3_fired = n_cells_wrong >= 2

    # ---- 5k. REG-TB stretch diagnostics per cell ----
    regtb_stretch_summary: dict = {}
    for cell in CELLS_189:
        cell_name = cell["name"]
        cell_tag = cell["cell_tag"]
        cell_rows = [row for row in rows_buffer
                     if row["cell"]["tag"] == cell_tag and row["arm"] == "REG-TB"]

        # Collect per-seed stretch logs
        all_stretch_logs = [row.get("stretch_log", []) for row in cell_rows]
        all_S_max = [row.get("S_max_final", 0.0) for row in cell_rows]
        all_max_stretch = [row.get("max_current_stretch", 0.0) for row in cell_rows]
        n_seeds_concede = sum(
            1 for row in cell_rows
            if any(e.get("label") == "concession" for e in row.get("events_summary", []))
        )

        # Per-burst trace (aggregate across seeds)
        per_burst_concessions = collections.defaultdict(list)
        for row in cell_rows:
            for e in row.get("events_summary", []):
                if e.get("label") == "concession":
                    # Find which burst this is in
                    es = e.get("entry_step", 0)
                    for bi, (bs, be) in enumerate(cell["burst_windows"]):
                        if bs <= es < be:
                            per_burst_concessions[bi].append({
                                "int_acc": e.get("int_acc_at_release"),
                                "S_max": e.get("reg_S_max_at_release"),
                            })

        regtb_stretch_summary[cell_name] = {
            "all_S_max": all_S_max,
            "mean_S_max": statistics.mean(all_S_max) if all_S_max else 0.0,
            "max_current_stretch_per_seed": all_max_stretch,
            "max_of_max_stretch": max(all_max_stretch) if all_max_stretch else 0.0,
            "n_seeds_concede": n_seeds_concede,
            "per_burst_concessions": dict(per_burst_concessions),
        }

    # Build human-readable stretch traces summary for each cell
    def build_stretch_trace_summary(cell: dict) -> str:
        """Build a compact summary of REG-TB stretch/threshold dynamics per cell."""
        cell_name = cell["name"]
        L_seq = cell["L_seq"]
        G = cell["G"]
        diag = regtb_stretch_summary.get(cell_name, {})

        lines = []
        lines.append(f"  {cell_name}: L_seq={L_seq} G={G} pred={cell['regtb_pred']}")
        lines.append(f"    max_stretch_per_seed: {[f'{s:.0f}' for s in diag.get('max_current_stretch_per_seed', [])]}")
        lines.append(f"    mean_S_max_final: {diag.get('mean_S_max', 0.0):.0f}  max_of_max: {diag.get('max_of_max_stretch', 0.0):.0f}")
        lines.append(f"    n_seeds_with_concession: {diag.get('n_seeds_concede', 0)}/8")

        # Expected threshold per burst based on kappa-reach law
        # Stretch of burst bi ~ L_seq[bi] + ~125 (pressure tail)
        expected_stretches = [L + 125 for L in L_seq]
        lines.append(f"    kappa-reach: expected_stretches~{expected_stretches}")
        lines.append(f"    threshold at each burst: T0={REG_TB_T0}, kappa*S_max grows with each completed burst")

        per_burst_conc = diag.get("per_burst_concessions", {})
        for bi, (bs, be) in enumerate(cell["burst_windows"]):
            concs = per_burst_conc.get(bi, [])
            s_vals = [c.get("S_max") for c in concs if c.get("S_max") is not None]
            acc_vals = [c.get("int_acc") for c in concs if c.get("int_acc") is not None]
            if concs:
                lines.append(f"    burst[{bi}] L={L_seq[bi]}: {len(concs)} concessions; "
                              f"int_acc~{[f'{a:.0f}' for a in acc_vals]}; "
                              f"S_max@concede~{[f'{s:.0f}' for s in s_vals]}")
            else:
                lines.append(f"    burst[{bi}] L={L_seq[bi]}: defended (no concessions)")
        return "\n".join(lines)

    stretch_traces_all = []
    for cell in CELLS_189:
        stretch_traces_all.append(build_stretch_trace_summary(cell))
    stretch_traces_summary_str = "\n".join(stretch_traces_all)

    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0
    print(f"Total runtime: {runtime_min:.1f} min")
    print()

    # ====================================================================
    # Write analysis row to JSONL
    # ====================================================================
    analysis_row = {
        "exp": 189,
        "kind": "analysis",
        "seeds": SEEDS_189,
        "p1_n_separation": p1_n_separation,
        "separation_cells": separation_cells,
        "separation_cell_names": separation_cell_names,
        "p1_pass": p1_pass,
        "p1_f1_fired": p1_f1_fired,
        "p2_pass": p2_pass,
        "p2_f2_fired": p2_f2_fired,
        "regtb_rev_normal": regtb_rev_normal,
        "regtb_rev_tight": regtb_rev_tight,
        "regtb_rev_loose": regtb_rev_loose,
        "regtb_rev_n_normal": regtb_rev_n_normal,
        "p3_pass": p3_pass,
        "p3_f3_fired": p3_f3_fired,
        "p3_n_wrong": n_cells_wrong,
        "p3_cell_results": p3_cell_results,
        "kappa_reach_actuals": kappa_reach_actuals,
        "pc1_total_flagged": pc1_total,
        "runtime_min": runtime_min,
        "regtb_cells_pass_normal": regtb_cells_pass_normal,
        "regtb_cell_names_pass": regtb_cell_names_pass,
    }
    with open(out_rows_path, "a") as fh:
        fh.write(json.dumps(to_plain(analysis_row)) + "\n")

    # ====================================================================
    # Write text report (exp189.txt)
    # ====================================================================
    lines: list[str] = []

    def p(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("EXP 189 — N4 VARIABLE-LENGTH SEPARATION TEST (RUNG-6 FALSIFIABLE)")
    p("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit 5818b22)")
    p(f"Runtime: {runtime_min:.1f} min | Seeds: {SEEDS_189}")
    p(f"6 cells x 6 arms x 8 seeds = 288 W | 5 arms x 8 seeds = 40 R")
    p(f"REG-TB VERBATIM exp188: KAPPA={REG_TB_KAPPA}, T0={REG_TB_T0}, calm_snaps={INT_CALM_SNAPS}")
    p("=" * 80)
    p()

    p("EQUIVALENCE GATE (L15)")
    p("-" * 60)
    p(gate_detail)
    p()

    p("CELLS (6 variable-length cells)")
    p("-" * 60)
    p(f"  {'name':4s}  {'L_seq':30s}  {'G':6s}  {'n_steps':8s}  {'regtb_pred':10s}")
    p("-" * 65)
    for cell in CELLS_189:
        p(f"  {cell['name']:4s}  {str(cell['L_seq']):30s}  {cell['G']:6d}  {cell['n_steps']:8d}  {cell['regtb_pred']:10s}")
    p()

    p("PHASE-R LATENCY TABLE (5 non-oracle arms x 8 seeds)")
    p("-" * 60)
    p(f"  {'arm':25s} " + " ".join(f"s{s}" for s in SEEDS_189))
    p(f"  {'baseline':25s} " + " ".join(str(r_latencies.get(('baseline', s))) for s in SEEDS_189))
    for arm_name, _, _, _ in R_ARM_DEFS:
        if arm_name == "baseline":
            continue
        lats = " ".join(
            f"{str(r_latencies.get((arm_name, s))):>6}" for s in SEEDS_189
        )
        p(f"  {arm_name:25s} {lats}")
    p()

    p("REVISION PASS TABLE (arm x mode; primary = normal +3000)")
    p("-" * 60)
    p(f"  {'arm':25s} " + " ".join(f"{m:8s}" for m, _ in REVISION_MODES))
    for arm_name, _, _, _ in R_ARM_DEFS:
        passes = " ".join(
            f"{'PASS' if revision_pass.get((arm_name, m), False) else 'fail':8s}"
            for m, _ in REVISION_MODES
        )
        p(f"  {arm_name:25s} {passes}")
    p()

    p("=" * 80)
    p("PER-CELL x PER-ARM DEFENSE/REVISION TABLE (primary mode = normal +3000)")
    p("=" * 80)
    p()
    arm_order = [name for name, _, _, _ in ARM_DEFS]
    p(f"  {'cell':>30}  {'arm':>20}  {'def_n/8':>8}  {'rev_norm':>9}  {'both_norm':>10}  {'both_tight':>11}  {'both_loose':>11}")
    p("-" * 110)
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        cell_display = f"{cell['name']}:{cell['L_seq']}G{cell['G']}"
        for arm_name in arm_order:
            ar = cell_arm_results.get((cell_tag, arm_name), {})
            dc = ar.get("defense_count", 0)
            if arm_name == "oracle":
                p(f"  {cell_display:>30}  {arm_name:>20}  {dc:>8}/8  {'(diag)':>9}  {'(diag)':>10}  {'(diag)':>11}  {'(diag)':>11}")
            else:
                rn = "PASS" if ar.get("revision_pass_normal", False) else "fail"
                bn = "PASS" if ar.get("both_pass_normal", False) else "fail"
                bt = "PASS" if ar.get("both_pass_tight",  False) else "fail"
                bl = "PASS" if ar.get("both_pass_loose",  False) else "fail"
                p(f"  {cell_display:>30}  {arm_name:>20}  {dc:>8}/8  {rn:>9}  {bn:>10}  {bt:>11}  {bl:>11}")
        p("-" * 110)
    p()

    p("=" * 80)
    p("CONJUNCT TABLE (oracle/deficit per cell)")
    p("=" * 80)
    p()
    p(f"  {'cell':>25}  {'oracle_n/8':>11}  {'oracle_ok':>10}  {'bl_fail_n/8':>12}  {'deficit_ok':>11}")
    p("-" * 70)
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        cell_display = f"{cell['name']}:{cell['L_seq']}G{cell['G']}"
        o_cnt = defense_counts[(cell_tag, "oracle")]
        b_cnt = defense_counts[(cell_tag, "baseline")]
        b_fail = N_SEEDS - b_cnt
        o_ok = oracle_ok[cell_tag]
        d_ok = baseline_deficit[cell_tag]
        p(f"  {cell_display:>25}  {o_cnt:>11}/8  {'OK' if o_ok else 'FAIL':>10}  {b_fail:>12}/8  {'OK' if d_ok else 'FAIL':>11}")
    p()

    p("=" * 80)
    p("SEPARATION TABLE — P1: cells where REG-TB passes both, no constant does")
    p("  Constant arms: INT-C2900, INT-C3500, CALM2600-H3000")
    p("=" * 80)
    p()
    p(f"  {'cell':>25}  {'REG-TB_both':>12}  {'C2900_both':>11}  {'C3500_both':>11}  {'CALM2600-H3000_both':>20}  {'separated':>10}")
    p("-" * 100)
    for cell in CELLS_189:
        cell_tag = cell["cell_tag"]
        cell_display = f"{cell['name']}:{cell['L_seq']}G{cell['G']}"
        rt_both = "PASS" if cell_arm_results.get((cell_tag, "REG-TB"), {}).get("both_pass_normal", False) else "fail"
        c29_both = "PASS" if cell_arm_results.get((cell_tag, "INT-C2900"), {}).get("both_pass_normal", False) else "fail"
        c35_both = "PASS" if cell_arm_results.get((cell_tag, "INT-C3500"), {}).get("both_pass_normal", False) else "fail"
        calm_both = "PASS" if cell_arm_results.get((cell_tag, "CALM2600-H3000"), {}).get("both_pass_normal", False) else "fail"
        sep = "YES" if cell_tag in separation_cells else "no"
        p(f"  {cell_display:>25}  {rt_both:>12}  {c29_both:>11}  {c35_both:>11}  {calm_both:>20}  {sep:>10}")
    p()
    p(f"  Separation cells: {separation_cell_names} (n={p1_n_separation})")
    p()

    p("=" * 80)
    p("REG-TB STRETCH / THRESHOLD TRACES PER CELL")
    p(f"  KAPPA={REG_TB_KAPPA}, T0={REG_TB_T0}")
    p("  Stretch ~ L + ~125 pressure-window tail per burst")
    p("  Threshold = max(KAPPA * S_max_completed, T0)")
    p("=" * 80)
    p()
    for cell in CELLS_189:
        p(build_stretch_trace_summary(cell))
        p()

    p("=" * 80)
    p("KAPPA-REACH LAW AUDIT (P3)")
    p(f"  REG-TB defends iff (i) first burst stretch <= T0={REG_TB_T0} AND")
    p(f"  (ii) every later burst stretch <= KAPPA x longest COMPLETED stretch so far")
    p("=" * 80)
    p()
    p(f"  {'cell':>8}  {'predicted':>10}  {'actual_def':>11}  {'def_count':>10}  {'correct':>8}")
    p("-" * 60)
    for cell in CELLS_189:
        cname = cell["name"]
        r = p3_cell_results[cname]
        cell_tag = cell["cell_tag"]
        dc = defense_counts.get((cell_tag, "REG-TB"), 0)
        p(f"  {cname:>8}  {r['predicted']:>10}  {r['actual']:>11}  {dc:>10}/8  {'OK' if r['correct'] else 'WRONG':>8}")
    p()
    p(f"  Cells wrong: {n_cells_wrong}/6  (F3 threshold: >= 2)")
    p()

    p("=" * 80)
    p("PHASE-R REG-TB CONCESSION DETAILS")
    p("=" * 80)
    p()
    p(f"  {'seed':>7}  {'baseline_lat':>13}  {'regtb_lat':>10}  {'diff':>8}")
    p("-" * 45)
    for seed in SEEDS_189:
        bl_l = r_latencies.get(("baseline", seed))
        rb_l = r_latencies.get(("REG-TB", seed))
        diff_str = str(rb_l - bl_l) if (rb_l is not None and bl_l is not None) else "N/A"
        p(f"  {seed:>7}  {str(bl_l):>13}  {str(rb_l):>10}  {diff_str:>8}")
    p()

    p("=" * 80)
    p("VERDICT LINES")
    p("=" * 80)
    p()

    # P1
    p(f"P1 (THE SEPARATION): in >= 2 cells REG-TB passes both bars while no constant does.")
    p(f"    Separation cells found: {p1_n_separation} (predicted >= 2; predicted: E1, E3)")
    p(f"    Separation cell names: {separation_cell_names}")
    p(f"    F1 fired (zero separation cells): {p1_f1_fired}")
    if p1_pass:
        p(f"    => P1 VERDICT: POSITIVE — REG-TB earns separation in {p1_n_separation} cell(s).")
    else:
        if p1_f1_fired:
            p(f"    => P1 VERDICT: NEGATIVE — F1 FIRED. Zero separation cells.")
            p(f"       The ambiguity bound extends to variable-L geometry at this richness.")
        else:
            p(f"    => P1 VERDICT: PARTIAL — only {p1_n_separation}/2 required separation cells.")
    p()

    # P2
    p(f"P2 (revision invariance): REG-TB Phase-R latency <= baseline + 3000.")
    p(f"    REG-TB revision (normal): {regtb_rev_normal} ({regtb_rev_n_normal}/8 seeds pass)")
    p(f"    REG-TB revision (tight):  {regtb_rev_tight}")
    p(f"    REG-TB revision (loose):  {regtb_rev_loose}")
    p(f"    Phase-R latencies: {regtb_r_lats}")
    p(f"    Baseline latencies: {bl_lats}")
    p(f"    F2 fired (REG-TB fails revision): {p2_f2_fired}")
    if p2_pass:
        p(f"    => P2 VERDICT: POSITIVE — REG-TB passes revision; tempo learning did not")
        p(f"       contaminate concession (threshold stays T0 in permanent pressure).")
    else:
        p(f"    => P2 VERDICT: NEGATIVE — F2 FIRED. REG-TB fails revision.")
    p()

    # P3
    p(f"P3 (kappa-reach law): REG-TB's 6 cell defense signs land as predicted.")
    p(f"    Cells correct: {n_cells_correct}/6 (F3 threshold: < 2 wrong)")
    for cname, r in p3_cell_results.items():
        cell_tag = next(c["cell_tag"] for c in CELLS_189 if c["name"] == cname)
        dc = defense_counts.get((cell_tag, "REG-TB"), 0)
        status = "OK" if r["correct"] else "WRONG"
        p(f"    {cname}: predicted={r['predicted']:4s}  actual={r['actual']:4s}  def={dc}/8  {status}")
    p(f"    F3 fired (>= 2 wrong): {p3_f3_fired}")
    if p3_pass:
        p(f"    => P3 VERDICT: CONFIRMED — kappa-reach law lands correctly ({n_cells_correct}/6 cells).")
    else:
        p(f"    => P3 VERDICT: FAILED — {n_cells_wrong} cells off (F3 fired).")
    p()

    p(f"PC1: {pc1_note}")
    p()

    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)
    p()
    p(f"  Gate:              PASS")
    p(f"  Seeds:             {SEEDS_189}")
    p(f"  Runtime:           {runtime_min:.1f} min")
    p()
    p(f"  P1 (separation: REG-TB beats all constants in >= 2 cells): {'POSITIVE' if p1_pass else 'NEGATIVE'}")
    p(f"     separation_cells: {separation_cell_names}  n={p1_n_separation}")
    p(f"  P2 (revision invariance): {'POSITIVE' if p2_pass else 'NEGATIVE'}")
    p(f"     rev_normal: {regtb_rev_normal} ({regtb_rev_n_normal}/8)")
    p(f"  P3 (kappa-reach law): {'CONFIRMED' if p3_pass else 'FAILED'}")
    p(f"     cells_correct: {n_cells_correct}/6")
    p()
    p(f"  F1 (zero separation):  {p1_f1_fired}")
    p(f"  F2 (REG-TB fails rev): {p2_f2_fired}")
    p(f"  F3 (>= 2 cells wrong): {p3_f3_fired}")
    p()

    if p1_pass and p2_pass and p3_pass:
        p("  OVERALL: REGULATION EARNS ITS KEEP — P1+P2+P3 all POSITIVE/CONFIRMED.")
        p("  Online tempo learning binds in exactly the predicted cells; the kappa-reach")
        p("  law maps the boundary; revision invariant. Synthesis appendix + consult.")
    elif p1_pass and p2_pass and not p3_pass:
        p("  OVERALL: SEPARATION ACHIEVED but kappa-reach law partial — consult.")
    elif p1_f1_fired and p2_pass:
        p("  OVERALL: NEGATIVE — F1 FIRED. Ambiguity bound is geometry-general; consult.")
    elif p2_f2_fired:
        p("  OVERALL: F2 FIRED — REG-TB fails revision; tempo learning contaminates concession.")
    else:
        p("  OVERALL: PARTIAL — review individual verdicts above.")
    p("=" * 80)

    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nText written to {out_txt_path}")
    print(f"Rows written to {out_rows_path}")
    print(f"Total runtime: {runtime_min:.1f} min")


if __name__ == "__main__":
    main()
