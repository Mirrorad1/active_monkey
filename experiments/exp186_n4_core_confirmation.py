"""Exp 186 — N4 Core Confirmation (CONFIRMATION-grade).

PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit e3e282e) BEFORE any data.
Status: CONFIRMATION (fresh seeds 250-257; 8 seeds/cell; predeclared falsifiers).

This experiment is the rung-4 fresh-seed confirmation of the surviving N4 crack core
from Exp 185. The predeclared falsifiers are:

  F1: core fails to replicate (< 4/6 core cells CONFIRMED) -> NEGATIVE result;
      config-sufficient closure with refined domain mapping.
  F2: oracle fails >= 2/8 in a core cell -> class-E contamination; cell excluded + flagged.
  F3: baseline deficit absent (< 7/8 fail) in a core cell -> precondition failure;
      cell excluded + flagged.
  PC1: ahat_drift < 0.15 gated per session (NOT mid-run bar adjustment).

CRACK CONFIRMED per core cell iff ALL THREE conjuncts hold:
  (i)   NO non-oracle arm passes BOTH bars (defense >= 6/8 AND revision >= 6/8);
  (ii)  oracle defends >= 7/8 seeds;
  (iii) baseline fails defense >= 7/8 seeds (the displacement deficit exists).

n4_freeze (E*=600) is included in the extended family for conjunct (i)
(i.e. if n4_freeze covers a cell, the cell is NOT confirmed as a crack).
oracle is EXCLUDED from the no-arm-covers conjunct (diagnostic witness only).

CORE REPLICATES iff >= 4/6 core cells confirm.

V-sample cells (8):
  - Four (1200,1,G) cells: cell_idx 32/33/34/35 (G=200/600/1200/2400).
  - First four other normal-mode V cells by cell_idx (from committed Exp 185
    classification, excluding (1200,1,*)): cell_idx 5,13,15,29.
V-sample verdicts: covered / crack / variance-dominated (best-arm defense [3,5]/8).

Arms (16):
  baseline; H600,H900,H1200,H1800,H2400,H3000,H4200,H6000;
  n4_freeze (E*=600, reference); oracle (diagnostic witness);
  CALM2600 x {H1200,H3000,H6000}; THETA3.0 x {H1200,H3000}.

W sessions: 14 cells x 16 arms x 8 seeds = 1792.
R sessions: 15 non-oracle arms x 8 seeds = 120.
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
# Import the exp185 runner via importlib (zero drift)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_spec185 = _ilu.spec_from_file_location(
    "exp185",
    str(REPO_ROOT / "experiments" / "exp185_n4_crack_classification.py"),
)
_mod185 = _ilu.module_from_spec(_spec185)
_spec185.loader.exec_module(_mod185)  # type: ignore[union-attr]

# Pull everything we need from exp185
run_fork_schedule_185 = _mod185.run_fork_schedule_185
compute_defense = _mod185.compute_defense
compute_final_gap = _mod185.compute_final_gap
phase_r_latency = _mod185.phase_r_latency
to_plain = _mod185.to_plain
pi_of = _mod185.pi_of

# Constants from exp185 (verbatim)
LAMBDA = _mod185.LAMBDA
INIT_MASS = _mod185.INIT_MASS
PHASE_R_START = _mod185.PHASE_R_START
BURST_SEED_OFFSET_R = _mod185.BURST_SEED_OFFSET_R
BURST_SEED_OFFSET_W_EXP183 = _mod185.BURST_SEED_OFFSET_W_EXP183
EVAL = _mod185.EVAL
FINE_EVAL = _mod185.FINE_EVAL
N_STEPS_PHASE_R = _mod185.N_STEPS_PHASE_R
CHUNK_SIZE = _mod185.CHUNK_SIZE
P6_HOLD = _mod185.P6_HOLD
DEFENSE_FRAC_THRESH = _mod185.DEFENSE_FRAC_THRESH
DEFENSE_WINDOW_OFFSET_START = _mod185.DEFENSE_WINDOW_OFFSET_START
DEFENSE_WINDOW_OFFSET_END = _mod185.DEFENSE_WINDOW_OFFSET_END
PC1_AHAT_DRIFT_MAX = _mod185.PC1_AHAT_DRIFT_MAX

DEFAULT_THETA = _mod185.DEFAULT_THETA
DEFAULT_RELEASE_CALM_SNAPS = _mod185.DEFAULT_RELEASE_CALM_SNAPS
CALM2600_SNAPS = _mod185.CALM2600_SNAPS
THETA3_THETA = _mod185.THETA3_THETA
E_STAR = _mod185.E_STAR
FLOAT_ATOL = _mod185.FLOAT_ATOL

EXP183_BURST_WINDOWS = _mod185.EXP183_BURST_WINDOWS
EXP183_N_STEPS = _mod185.EXP183_N_STEPS
EXP183_GATE_SEEDS = _mod185.EXP183_GATE_SEEDS
EXP183_GATE_ARMS = _mod185.EXP183_GATE_ARMS

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Exp 186 configuration
# ---------------------------------------------------------------------------

SEEDS_186 = [250, 251, 252, 253, 254, 255, 256, 257]
N_SEEDS = len(SEEDS_186)

# Confirmation bars
DEFENSE_MIN_PASS = 6   # >= 6/8 seeds (confirmation-grade)
ORACLE_DEFENSE_MIN = 7  # >= 7/8
BASELINE_FAIL_MIN = 7   # >= 7/8 must fail for deficit to exist
CORE_CONFIRM_MIN = 4    # >= 4/6 core cells must confirm

# Revision modes: primary = normal (+3000); tight/loose are secondary diagnostics
REVISION_MODES = [
    ("normal", 3000),
    ("tight",  1500),
    ("loose",  6000),
]

# Core-6 cells (surviving normal-mode A and B from exp185)
# A: (1600,2,200), (1600,3,200), (2400,3,200)
# B: (1600,4,200), (2400,4,200), (2400,4,600)
CORE_CELLS = [
    {"L": 1600, "K": 2, "G": 200, "cell_idx": 52, "exp185_class": "A"},
    {"L": 1600, "K": 3, "G": 200, "cell_idx": 56, "exp185_class": "A"},
    {"L": 1600, "K": 4, "G": 200, "cell_idx": 60, "exp185_class": "B"},
    {"L": 2400, "K": 3, "G": 200, "cell_idx": 72, "exp185_class": "A"},
    {"L": 2400, "K": 4, "G": 200, "cell_idx": 76, "exp185_class": "B"},
    {"L": 2400, "K": 4, "G": 600, "cell_idx": 77, "exp185_class": "B"},
]

# V-sample-8:
#   (1200,1,G) cells: cell_idx 32,33,34,35 (G=200/600/1200/2400)
#   First 4 other normal-mode V cells (excluding 1200,1,*): cell_idx 5,13,15,29
VSAMPLE_CELLS = [
    {"L": 1200, "K": 1, "G": 200,  "cell_idx": 32, "v_rule": "(1200,1,G) group"},
    {"L": 1200, "K": 1, "G": 600,  "cell_idx": 33, "v_rule": "(1200,1,G) group"},
    {"L": 1200, "K": 1, "G": 1200, "cell_idx": 34, "v_rule": "(1200,1,G) group"},
    {"L": 1200, "K": 1, "G": 2400, "cell_idx": 35, "v_rule": "(1200,1,G) group"},
    {"L": 400,  "K": 2, "G": 600,  "cell_idx":  5, "v_rule": "first-4-other normal-V by cell_idx (excl. 1200,1,*)"},
    {"L": 400,  "K": 4, "G": 600,  "cell_idx": 13, "v_rule": "first-4-other normal-V by cell_idx (excl. 1200,1,*)"},
    {"L": 400,  "K": 4, "G": 2400, "cell_idx": 15, "v_rule": "first-4-other normal-V by cell_idx (excl. 1200,1,*)"},
    {"L": 800,  "K": 4, "G": 600,  "cell_idx": 29, "v_rule": "first-4-other normal-V by cell_idx (excl. 1200,1,*)"},
]

ALL_CELLS_186 = CORE_CELLS + VSAMPLE_CELLS   # 14 cells total

# Arms (16):
#   baseline
#   H600..H6000 (8)
#   n4_freeze (E*=600, reference — included in extended family for conjunct i)
#   oracle (diagnostic witness, excluded from conjunct i)
#   CALM2600 x {H1200, H3000, H6000}
#   THETA3.0 x {H1200, H3000}
FIXED_HORIZONS_186 = [600, 900, 1200, 1800, 2400, 3000, 4200, 6000]

# arm definitions: (arm_name, arm_mode, theta, release_calm_snaps)
# arm_mode: "baseline", "oracle", "freeze_evidence", ("freeze_time", H)
ARM_DEFS = []

# baseline
ARM_DEFS.append(("baseline", "baseline", DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))

# 8 fixed-H arms
for H in FIXED_HORIZONS_186:
    ARM_DEFS.append((f"H{H}", ("freeze_time", H), DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))

# n4_freeze (evidence-based concession E*=600)
ARM_DEFS.append(("n4_freeze", "freeze_evidence", DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))

# oracle (diagnostic, excluded from family)
ARM_DEFS.append(("oracle", "oracle", DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))

# CALM2600 x {H1200, H3000, H6000}
for H in [1200, 3000, 6000]:
    ARM_DEFS.append((f"CALM2600-H{H}", ("freeze_time", H), DEFAULT_THETA, CALM2600_SNAPS))

# THETA3.0 x {H1200, H3000}
for H in [1200, 3000]:
    ARM_DEFS.append((f"THETA3-H{H}", ("freeze_time", H), THETA3_THETA, DEFAULT_RELEASE_CALM_SNAPS))

assert len(ARM_DEFS) == 16, f"Expected 16 arms, got {len(ARM_DEFS)}"

# Arms that are part of the extended family (for conjunct i — no-arm-covers check)
# All arms EXCEPT oracle
FAMILY_ARM_NAMES = {name for name, _, _, _ in ARM_DEFS if name != "oracle"}

# R arms: all 15 non-oracle arms
R_ARM_DEFS = [(name, mode, theta, rcs) for name, mode, theta, rcs in ARM_DEFS
              if name != "oracle"]
assert len(R_ARM_DEFS) == 15, f"Expected 15 R arms, got {len(R_ARM_DEFS)}"


def compute_n_steps(L: int, K: int, G: int) -> int:
    return 6000 + K * L + (K - 1) * G + 2500


def compute_burst_windows(L: int, K: int, G: int) -> list:
    windows = []
    cur = 6000
    for ki in range(K):
        bstart = cur
        bend = cur + L
        windows.append((bstart, bend))
        cur = bend
        if ki < K - 1:
            cur += G
    return windows


# ---------------------------------------------------------------------------
# Equivalence gate (L15) — same pattern as exp185, re-run through THIS code path
# ---------------------------------------------------------------------------

def load_committed_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_equivalence_gate_186(
    mirro_root, base_cmap, n_colors, committed_183_path: Path
) -> tuple[bool, str]:
    """Run equivalence gate: reproduce exp183 baseline x s229 and H1200 x s229.

    This runs through exp186's code path (which calls exp185's imported runner).
    Emits full got-vs-committed per-field table (L15 pattern).
    Returns (gate_pass, detail_string).
    """
    committed_rows = load_committed_rows(committed_183_path)
    committed_w = {}
    for row in committed_rows:
        if row.get("phase") == "W":
            key = (row["arm"], row["fork_seed"], row["burst_idx"])
            committed_w[key] = row

    detail_lines = []
    all_pass = True

    arm_lookup = {
        "baseline": ("baseline", "baseline"),
        "H1200": ("H1200", ("freeze_time", 1200)),
    }

    for arm_name in EXP183_GATE_ARMS:
        _, arm_mode = arm_lookup[arm_name]
        for seed in EXP183_GATE_SEEDS:
            print(f"  [gate] arm={arm_name} seed={seed} ...", flush=True)
            root = copy.deepcopy(mirro_root)
            root._state_dir = None

            rr = run_fork_schedule_185(
                mirro=root,
                fork_seed=seed,
                base_cmap=base_cmap,
                n_colors=n_colors,
                arm_name=arm_name,
                arm_mode=arm_mode,
                phase="W",
                burst_windows=EXP183_BURST_WINDOWS,
                color_mode="endogenous",
                reloc_rng_seed=BURST_SEED_OFFSET_W_EXP183 + seed,
                n_steps=EXP183_N_STEPS,
                theta=DEFAULT_THETA,
                release_calm_snaps=DEFAULT_RELEASE_CALM_SNAPS,
            )

            session_pass = True
            detail_lines.append(f"\n=== arm={arm_name} seed={seed} ===")
            for bi in range(len(EXP183_BURST_WINDOWS)):
                key = (arm_name, seed, bi)
                if key not in committed_w:
                    msg = f"  MISS: committed row not found for {key}"
                    detail_lines.append(msg)
                    session_pass = False
                    all_pass = False
                    continue

                cr = committed_w[key]

                bstart, bend = EXP183_BURST_WINDOWS[bi]
                win_start = bend + 1500
                win_end = bend + 2000
                if win_end <= EXP183_N_STEPS and rr["burst_onset_color"][bi] is not None:
                    bc = rr["burst_onset_color"][bi]
                    gate_frac = float(np.mean(rr["expressed_arr"][win_start:win_end] == bc))
                    gate_recovered = gate_frac < 0.5
                else:
                    gate_recovered = None

                checks = [
                    ("gap_start",  rr["gap_start"][bi],   cr["gap_start"]),
                    ("gap_end",    rr["gap_end"][bi],     cr["gap_end"]),
                    ("d_b",        rr["d_b"][bi],         cr["d_b"]),
                    ("tv_b",       rr["tv_b"][bi],        cr["tv_b"]),
                    ("recovered",  gate_recovered,        cr["recovered"]),
                    ("n_events",   len(rr["events"]),     cr["n_events"]),
                ]

                for field_name, got, expected in checks:
                    if got is None and expected is None:
                        detail_lines.append(
                            f"  b{bi} {field_name:<14} got={got!r:<40} "
                            f"committed={expected!r:<40} OK (both None)"
                        )
                        continue
                    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
                        ok = abs(float(got) - float(expected)) <= FLOAT_ATOL
                    elif type(got) == type(expected):
                        ok = got == expected
                    else:
                        ok = str(got) == str(expected)

                    status = "OK" if ok else "MISMATCH"
                    detail_lines.append(
                        f"  b{bi} {field_name:<14} got={str(got):<40} "
                        f"committed={str(expected):<40} {status}"
                    )
                    if not ok:
                        session_pass = False
                        all_pass = False

                got_evs = rr["events"]
                exp_evs = cr.get("events_summary", [])
                if len(got_evs) != len(exp_evs):
                    detail_lines.append(
                        f"  b{bi} event count      got={len(got_evs):<40} "
                        f"committed={len(exp_evs):<40} MISMATCH"
                    )
                    session_pass = False
                    all_pass = False
                else:
                    for ei, (ge, ee) in enumerate(zip(got_evs, exp_evs)):
                        for ef in ["label", "entry_step", "frozen_steps", "c_star", "trigger_latency"]:
                            gv = ge.get(ef)
                            ev = ee.get(ef)
                            ok = gv == ev
                            status = "OK" if ok else "MISMATCH"
                            detail_lines.append(
                                f"  ev{ei} {ef:<14} got={str(gv):<40} "
                                f"committed={str(ev):<40} {status}"
                            )
                            if not ok:
                                session_pass = False
                                all_pass = False
                        gv = ge.get("E_blocked")
                        ev = ee.get("E_blocked")
                        if gv is not None and ev is not None:
                            ok = abs(float(gv) - float(ev)) <= FLOAT_ATOL
                            status = "OK" if ok else "MISMATCH"
                            detail_lines.append(
                                f"  ev{ei} {'E_blocked':<14} got={str(gv):<40} "
                                f"committed={str(ev):<40} {status}"
                            )
                            if not ok:
                                session_pass = False
                                all_pass = False

            status = "PASS" if session_pass else "FAIL"
            detail_lines.append(f"  gate arm={arm_name} seed={seed}: {status}")
            print(f"  [gate] arm={arm_name} seed={seed}: {status}", flush=True)

    overall = "PASS" if all_pass else "FAIL"
    detail_str = f"EQUIVALENCE GATE: {overall}\n" + "\n".join(detail_lines)
    return all_pass, detail_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 80)
    print("Exp 186 — N4 Core Confirmation (CONFIRMATION-grade)")
    print("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit e3e282e)")
    print("Seeds: 250-257 (8 fresh seeds); 14 cells x 16 arms x 8 seeds = 1792 W")
    print("=" * 80)
    print()

    # V-sample cell selection trace
    print("V-SAMPLE CELL SELECTION TRACE")
    print("-" * 60)
    print("Rule: four (1200,1,G) cells (G in 200/600/1200/2400) PLUS first four OTHER")
    print("      normal-mode V cells by cell_idx from committed exp185 classification")
    print("      (lowest cell_idx among normal-mode V cells, excluding (1200,1,*)).")
    print()
    print("(1200,1,G) group (cell_idx 32-35):")
    for c in VSAMPLE_CELLS[:4]:
        print(f"  cell_idx={c['cell_idx']} L={c['L']} K={c['K']} G={c['G']}")
    print()
    print("First-4-other normal-V from exp185.txt classification table:")
    print("  Normal-mode V cells (not 1200,1,*), sorted by cell_idx:")
    print("  [5,13,15,29,30,31,40,43,44,45,46,47,57,58,61,62,63,70,71,73,75,79]")
    print("  => first 4: cell_idx 5, 13, 15, 29")
    for c in VSAMPLE_CELLS[4:]:
        print(f"  cell_idx={c['cell_idx']} L={c['L']} K={c['K']} G={c['G']}  [{c['v_rule']}]")
    print()

    print("CORE CELLS (6):")
    for c in CORE_CELLS:
        print(f"  cell_idx={c['cell_idx']} L={c['L']} K={c['K']} G={c['G']} exp185_class={c['exp185_class']}")
    print()
    print("ALL 14 CELLS:")
    for c in ALL_CELLS_186:
        kind = "CORE" if c in CORE_CELLS else "V-sample"
        print(f"  cell_idx={c['cell_idx']:>3} L={c['L']} K={c['K']} G={c['G']}  [{kind}]")
    print()
    print("ARMS (16):")
    for name, mode, theta, rcs in ARM_DEFS:
        fam = "(family)" if name in FAMILY_ARM_NAMES else "(oracle-only)"
        print(f"  {name:20s}  theta={theta}  release_calm_snaps={rcs}  {fam}")
    print()
    print(f"R arms (15 non-oracle): {[n for n,_,_,_ in R_ARM_DEFS]}")
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
    out_rows_path = out_dir / "exp186_rows.json"
    out_txt_path = out_dir / "exp186.txt"
    committed_183_path = out_dir / "exp183_rows.json"

    # ====================================================================
    # STEP 1: EQUIVALENCE GATE (L15)
    # ====================================================================
    print("=" * 80)
    print("STEP 1: EQUIVALENCE GATE (L15)")
    print("Reproducing exp183 (baseline x s229) and (H1200 x s229) via exp186 code path")
    print("Default params: theta=3.5, release_calm_snaps=8")
    print("=" * 80)
    t_gate = time.time()

    gate_pass, gate_detail = run_equivalence_gate_186(
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
    # STEP 2: W SESSIONS (14 cells x 16 arms x 8 seeds = 1792)
    # ====================================================================
    print("=" * 80)
    print("STEP 2: W SESSIONS (14 cells x 16 arms x 8 seeds = 1792 sessions)")
    print("=" * 80)

    # defense_by[(cell_idx, arm_name, seed)] -> bool
    defense_by: dict = {}
    # pc1_flag[(cell_idx, arm_name, seed)] -> bool (ahat_drift >= 0.15)
    pc1_flags: dict = {}
    # raw defense counts for partial reporting
    # defense_counts[(cell_idx, arm_name)] -> int
    # stored after full seed sweep

    rows_buffer: list[dict] = []

    t_grid = time.time()
    n_total = len(ALL_CELLS_186) * len(ARM_DEFS) * N_SEEDS
    n_done = 0

    for ci, cell in enumerate(ALL_CELLS_186):
        L, K, G = cell["L"], cell["K"], cell["G"]
        cell_idx = cell["cell_idx"]
        n_steps = compute_n_steps(L, K, G)
        burst_windows = compute_burst_windows(L, K, G)

        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in SEEDS_186:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

                reloc_rng_seed = 195_000 + 1000 * cell_idx + seed

                rr = run_fork_schedule_185(
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
                defense_by[(cell_idx, arm_name, seed)] = bool(defense)
                pc1_flag = rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX
                pc1_flags[(cell_idx, arm_name, seed)] = pc1_flag

                final_gap = compute_final_gap(rr["v_traj"], rr["expressed_arr"], n_steps)
                per_burst = []
                for bi, (bstart, bend) in enumerate(burst_windows):
                    pb = {
                        "bi": bi,
                        "gap_start": rr["gap_start"][bi],
                        "gap_end": rr["gap_end"][bi],
                        "expr_frac_or_null": rr["per_burst_expr_frac"][bi],
                    }
                    per_burst.append(pb)

                row = {
                    "exp": 186,
                    "kind": "W",
                    "cell": {"L": L, "K": K, "G": G, "idx": cell_idx},
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
                        }
                        for e in rr["events"]
                    ]),
                    "ahat_drift": float(rr["ahat_drift"]),
                    "settle_tv": to_plain(rr.get("settle_tv")),
                    "flags": ["PC1_DRIFT"] if pc1_flag else [],
                    "theta": theta_val,
                    "release_calm_snaps": rc_snaps,
                }
                rows_buffer.append(row)
                n_done += 1

        elapsed = time.time() - t_grid
        if ci == 0 or (ci + 1) % 4 == 0 or ci == len(ALL_CELLS_186) - 1:
            rate = elapsed / max(1, n_done)
            remaining = rate * (n_total - n_done)
            print(
                f"  cell {ci+1:02d}/{len(ALL_CELLS_186)} (L={L},K={K},G={G},idx={cell_idx}) "
                f"| elapsed {elapsed:.0f}s | ETA {remaining:.0f}s",
                flush=True,
            )

    t_grid_done = time.time()
    print(f"W grid done: {(t_grid_done - t_grid)/60:.1f} min total")
    print()

    # ====================================================================
    # STEP 3: R SESSIONS (15 non-oracle arms x 8 seeds = 120 sessions)
    # ====================================================================
    print("=" * 80)
    print("STEP 3: R SESSIONS (15 non-oracle arms x 8 seeds = 120 sessions)")
    print("baseline R latencies are the tolerance reference")
    print("=" * 80)

    # r_latencies[(arm_name, seed)] -> int|None
    r_latencies: dict = {}
    r_rows_buffer: list[dict] = []

    t_r = time.time()
    for arm_label, arm_mode_r, theta_val, rc_snaps in R_ARM_DEFS:
        for seed in SEEDS_186:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None

            rr = run_fork_schedule_185(
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

            print(
                f"  arm={arm_label:22s} seed={seed}  "
                f"regime_color={rc}  latency={lat}  "
                f"ahat_drift={rr['ahat_drift']:.5f}",
                flush=True,
            )

            r_row = {
                "exp": 186,
                "kind": "R",
                "arm": arm_label,
                "seed": int(seed),
                "latency": to_plain(lat),
                "n_events": int(len(rr["events"])),
                "ahat_drift": float(rr["ahat_drift"]),
                "flags": [],
                "theta": theta_val,
                "release_calm_snaps": rc_snaps,
            }
            r_rows_buffer.append(r_row)

    print(f"Phase-R done: {time.time()-t_r:.1f}s")
    print()

    # ====================================================================
    # STEP 4: WRITE W+R ROWS TO JSONL (streaming)
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

    # ---- 5a. Defense counts per (cell_idx, arm_name) ----
    defense_counts: dict = {}   # (cell_idx, arm_name) -> int (count passing seeds)
    for cell in ALL_CELLS_186:
        cell_idx = cell["cell_idx"]
        for arm_name, _, _, _ in ARM_DEFS:
            n_pass = sum(
                1 for seed in SEEDS_186
                if defense_by.get((cell_idx, arm_name, seed), False)
            )
            defense_counts[(cell_idx, arm_name)] = n_pass

    # defense_pass[(cell_idx, arm_name)] -> bool (>= 6/8)
    defense_pass: dict = {}
    for (cell_idx, arm_name), cnt in defense_counts.items():
        defense_pass[(cell_idx, arm_name)] = cnt >= DEFENSE_MIN_PASS

    # ---- 5b. Revision pass per arm (normal mode = primary) ----
    # revision_pass[(arm_name, mode_name)] -> bool (>= 6/8 seeds with lat <= bl+tol)
    revision_pass: dict = {}
    for arm_name, _, _, _ in R_ARM_DEFS:
        for mode_name, tolerance in REVISION_MODES:
            n_pass = 0
            for seed in SEEDS_186:
                bl_lat = r_latencies.get(("baseline", seed))
                arm_lat = r_latencies.get((arm_name, seed))
                if arm_lat is None or bl_lat is None:
                    continue
                if arm_lat <= bl_lat + tolerance:
                    n_pass += 1
            revision_pass[(arm_name, mode_name)] = n_pass >= DEFENSE_MIN_PASS

    # ---- 5c. Per core cell: compute the three conjuncts ----
    # Primary mode = normal (+3000)
    PRIMARY_MODE = "normal"

    core_results: dict = {}  # cell_idx -> dict

    for cell in CORE_CELLS:
        cell_idx = cell["cell_idx"]

        # Conjunct (ii): oracle defends >= 7/8
        oracle_count = defense_counts[(cell_idx, "oracle")]
        conjunct_ii = oracle_count >= ORACLE_DEFENSE_MIN

        # F2 check: oracle fails >= 2/8 => exclude + flag
        oracle_fail_count = N_SEEDS - oracle_count
        f2_fired = oracle_fail_count >= 2

        # Conjunct (iii): baseline fails defense >= 7/8
        baseline_pass_count = defense_counts[(cell_idx, "baseline")]
        baseline_fail_count = N_SEEDS - baseline_pass_count
        conjunct_iii = baseline_fail_count >= BASELINE_FAIL_MIN

        # F3 check: baseline deficit absent (< 7/8 fail) => exclude + flag
        f3_fired = not conjunct_iii  # no deficit

        # Conjunct (i): no family arm passes both bars (primary mode = normal)
        # Family = all arms except oracle
        # For each family arm: both_pass iff defense_pass AND revision_pass (normal)
        arm_results = {}
        for arm_name in FAMILY_ARM_NAMES:
            d_ok = defense_pass.get((cell_idx, arm_name), False)
            # baseline is in family but has no revision in the traditional sense;
            # for the "both bars" check, baseline revision is trivially true (H=0 arm,
            # baseline latency = baseline latency + 0 <= baseline + tol always);
            # however per pre-reg the check is specifically defense+revision as defined.
            # For baseline arm there is no Phase-R arm under its own name
            # (it runs as "baseline"), so:
            if arm_name == "baseline":
                r_ok = revision_pass.get(("baseline", PRIMARY_MODE), False)
            else:
                r_ok = revision_pass.get((arm_name, PRIMARY_MODE), False)
            both_ok = d_ok and r_ok
            arm_results[arm_name] = {
                "defense_count": defense_counts[(cell_idx, arm_name)],
                "defense_pass": d_ok,
                "revision_pass_normal": r_ok,
                "both_pass_normal": both_ok,
                # secondary diagnostics
                "revision_pass_tight": revision_pass.get((arm_name, "tight"), False),
                "revision_pass_loose": revision_pass.get((arm_name, "loose"), False),
                "both_pass_tight": d_ok and revision_pass.get((arm_name, "tight"), False),
                "both_pass_loose": d_ok and revision_pass.get((arm_name, "loose"), False),
            }

        # Conjunct (i): no family arm passes both bars in normal mode
        any_arm_covers = any(v["both_pass_normal"] for v in arm_results.values())
        conjunct_i = not any_arm_covers

        # Determine CONFIRMED status
        # If F2 or F3: exclude + flag, don't confirm
        excluded = f2_fired or f3_fired
        confirmed = (not excluded) and conjunct_i and conjunct_ii and conjunct_iii

        # Binding conjunct (which one failed first, if not confirmed)
        binding = None
        if not confirmed and not excluded:
            if not conjunct_i:
                binding = "conjunct_i (arm covers)"
            elif not conjunct_ii:
                binding = "conjunct_ii (oracle weak)"
            elif not conjunct_iii:
                binding = "conjunct_iii (no deficit)"

        # A/B relabel
        # A: some family arm defends >= 6/8 but every such arm fails revision
        # B: no family arm defends >= 6/8 at all
        defending_arms = [
            name for name in FAMILY_ARM_NAMES
            if defense_pass.get((cell_idx, name), False)
        ]
        if defending_arms:
            if all(not revision_pass.get((name, PRIMARY_MODE), False) for name in defending_arms):
                relabel = "A"
            else:
                relabel = "covered"  # shouldn't happen if conjunct_i holds
        else:
            relabel = "B"

        # Best near-miss: for each non-confirmed core cell, which arm had highest defense count
        # and by how much did it miss the bars
        near_miss = None
        if not confirmed:
            best_arm_name = None
            best_def = -1
            for arm_name in FAMILY_ARM_NAMES:
                dc = defense_counts[(cell_idx, arm_name)]
                if dc > best_def:
                    best_def = dc
                    best_arm_name = arm_name
            if best_arm_name is not None:
                near_miss = {
                    "arm": best_arm_name,
                    "defense_count": best_def,
                    "defense_gap": DEFENSE_MIN_PASS - best_def,
                    "revision_pass_normal": revision_pass.get((best_arm_name, PRIMARY_MODE), False),
                }

        core_results[cell_idx] = {
            "cell": cell,
            "oracle_count": oracle_count,
            "oracle_fail_count": oracle_fail_count,
            "baseline_pass_count": baseline_pass_count,
            "baseline_fail_count": baseline_fail_count,
            "conjunct_i": conjunct_i,
            "conjunct_ii": conjunct_ii,
            "conjunct_iii": conjunct_iii,
            "f2_fired": f2_fired,
            "f3_fired": f3_fired,
            "excluded": excluded,
            "confirmed": confirmed,
            "binding_conjunct": binding,
            "relabel": relabel if confirmed else "—",
            "arm_results": arm_results,
            "near_miss": near_miss,
        }

    # ---- 5d. Replication verdict ----
    n_confirmed = sum(1 for r in core_results.values() if r["confirmed"])
    n_excluded = sum(1 for r in core_results.values() if r["excluded"])
    n_eligible = len(CORE_CELLS) - n_excluded
    replicates = n_confirmed >= CORE_CONFIRM_MIN

    # ---- 5e. V-sample verdicts ----
    vsample_results: dict = {}  # cell_idx -> verdict

    for cell in VSAMPLE_CELLS:
        cell_idx = cell["cell_idx"]

        # Best-arm defense count (across family arms, primary mode check)
        best_def = max(
            defense_counts.get((cell_idx, name), 0)
            for name in FAMILY_ARM_NAMES
        )

        # oracle check
        oracle_count_v = defense_counts[(cell_idx, "oracle")]
        # baseline deficit check
        bl_fail_v = N_SEEDS - defense_counts[(cell_idx, "baseline")]

        # covered: some family arm passes both bars (normal mode)
        is_covered = any(
            defense_pass.get((cell_idx, name), False)
            and revision_pass.get((name, PRIMARY_MODE), False)
            for name in FAMILY_ARM_NAMES
        )

        # crack conjuncts (same as core)
        conjunct_i_v = not is_covered
        conjunct_ii_v = oracle_count_v >= ORACLE_DEFENSE_MIN
        conjunct_iii_v = bl_fail_v >= BASELINE_FAIL_MIN
        is_crack = conjunct_i_v and conjunct_ii_v and conjunct_iii_v

        # variance-dominated: best-arm defense count in [3,5]/8
        is_variance_dominated = 3 <= best_def <= 5

        if is_covered:
            verdict = "covered"
        elif is_crack:
            verdict = "crack"
        elif is_variance_dominated:
            verdict = "variance-dominated"
        else:
            # Neither clearly covered nor crack — report best_def
            verdict = f"indeterminate (best_def={best_def}/8)"

        vsample_results[cell_idx] = {
            "cell": cell,
            "best_arm_def": best_def,
            "oracle_count": oracle_count_v,
            "baseline_fail_count": bl_fail_v,
            "is_covered": is_covered,
            "is_crack": is_crack,
            "is_variance_dominated": is_variance_dominated,
            "verdict": verdict,
        }

    # ---- 5f. F1/F2/F3/PC1 status ----
    f1_fired = not replicates
    f2_cells = [cell["cell_idx"] for cell in CORE_CELLS
                if core_results[cell["cell_idx"]]["f2_fired"]]
    f3_cells = [cell["cell_idx"] for cell in CORE_CELLS
                if core_results[cell["cell_idx"]]["f3_fired"]]
    f2_fired = len(f2_cells) > 0
    f3_fired = len(f3_cells) > 0

    pc1_total = sum(1 for v in pc1_flags.values() if v)
    pc1_note = (f"{pc1_total}/{len(pc1_flags)} sessions flagged ahat_drift >= {PC1_AHAT_DRIFT_MAX}")

    # ---- 5g. Secondary diagnostics (tight/loose) for core cells ----
    # same rows, different tolerances
    core_secondary: dict = {}
    for cell in CORE_CELLS:
        cell_idx = cell["cell_idx"]
        for mode_name in ["tight", "loose"]:
            any_covers = any(
                defense_pass.get((cell_idx, name), False)
                and revision_pass.get((name, mode_name), False)
                for name in FAMILY_ARM_NAMES
            )
            core_secondary[(cell_idx, mode_name)] = not any_covers  # conjunct_i for that mode

    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0
    print(f"Total runtime: {runtime_min:.1f} min")
    print()

    # ====================================================================
    # Write analysis row to JSONL
    # ====================================================================
    analysis_row = {
        "exp": 186,
        "kind": "analysis",
        "seeds": SEEDS_186,
        "n_confirmed": n_confirmed,
        "n_excluded": n_excluded,
        "replicates": replicates,
        "core_results": {
            str(k): {
                "cell": to_plain(v["cell"]),
                "confirmed": v["confirmed"],
                "relabel": v["relabel"],
                "binding_conjunct": v["binding_conjunct"],
                "conjunct_i": v["conjunct_i"],
                "conjunct_ii": v["conjunct_ii"],
                "conjunct_iii": v["conjunct_iii"],
                "oracle_count": v["oracle_count"],
                "baseline_fail_count": v["baseline_fail_count"],
                "f2_fired": v["f2_fired"],
                "f3_fired": v["f3_fired"],
                "excluded": v["excluded"],
                "near_miss": v["near_miss"],
            }
            for k, v in core_results.items()
        },
        "vsample_results": {
            str(k): {
                "cell": to_plain(v["cell"]),
                "verdict": v["verdict"],
                "best_arm_def": v["best_arm_def"],
                "oracle_count": v["oracle_count"],
                "baseline_fail_count": v["baseline_fail_count"],
            }
            for k, v in vsample_results.items()
        },
        "f1_fired": f1_fired,
        "f2_fired": f2_fired,
        "f3_fired": f3_fired,
        "f2_cells": f2_cells,
        "f3_cells": f3_cells,
        "pc1_total_flagged": pc1_total,
        "runtime_min": runtime_min,
    }
    with open(out_rows_path, "a") as fh:
        fh.write(json.dumps(to_plain(analysis_row)) + "\n")

    # ====================================================================
    # Write text report (exp186.txt)
    # ====================================================================
    lines: list[str] = []

    def p(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("EXP 186 — N4 CORE CONFIRMATION (CONFIRMATION-grade)")
    p("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit e3e282e)")
    p(f"Runtime: {runtime_min:.1f} min | Seeds: {SEEDS_186}")
    p(f"14 cells x 16 arms x 8 seeds = 1792 W | 15 arms x 8 seeds = 120 R")
    p("=" * 80)
    p()

    p("V-SAMPLE CELL SELECTION TRACE")
    p("-" * 60)
    p("Rule: four (1200,1,G) cells (G in 200/600/1200/2400) PLUS")
    p("      first four OTHER normal-mode V cells by cell_idx (excl. 1200,1,*)")
    p("      from committed exp185 classification.")
    p()
    p("(1200,1,G) group:")
    for c in VSAMPLE_CELLS[:4]:
        p(f"  cell_idx={c['cell_idx']} L={c['L']} K={c['K']} G={c['G']}")
    p()
    p("Other normal-V (first 4 by cell_idx):")
    for c in VSAMPLE_CELLS[4:]:
        p(f"  cell_idx={c['cell_idx']} L={c['L']} K={c['K']} G={c['G']}")
    p()

    p("EQUIVALENCE GATE (L15)")
    p("-" * 60)
    p(gate_detail)
    p()

    p("PHASE-R LATENCY TABLE (15 non-oracle arms x 8 seeds)")
    p("-" * 60)
    bl_lats = [r_latencies.get(("baseline", s)) for s in SEEDS_186]
    p(f"  {'arm':25s} " + " ".join(f"s{s}" for s in SEEDS_186))
    p(f"  {'baseline':25s} " + " ".join(str(r_latencies.get(('baseline', s))) for s in SEEDS_186))
    for arm_name, _, _, _ in R_ARM_DEFS:
        if arm_name == "baseline":
            continue
        lats = " ".join(
            f"{str(r_latencies.get((arm_name, s))):>6}" for s in SEEDS_186
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
    p("CORE CELL CONFIRMATION TABLE (primary mode = normal +3000)")
    p("=" * 80)
    p()
    for cell in CORE_CELLS:
        cell_idx = cell["cell_idx"]
        r = core_results[cell_idx]
        p(f"Cell ({cell['L']},{cell['K']},{cell['G']}) cell_idx={cell_idx}  exp185_class={cell['exp185_class']}")
        p(f"  Oracle defense:    {r['oracle_count']}/8  (conjunct ii: {'PASS' if r['conjunct_ii'] else 'FAIL'})")
        p(f"  Baseline fail:     {r['baseline_fail_count']}/8  (conjunct iii: {'PASS' if r['conjunct_iii'] else 'FAIL'})")
        if r["f2_fired"]:
            p(f"  *** F2 FIRED: oracle fails {r['oracle_fail_count']}/8 — cell EXCLUDED ***")
        if r["f3_fired"]:
            p(f"  *** F3 FIRED: baseline deficit absent — cell EXCLUDED ***")
        p()
        p(f"  Per-arm table (defense_count/8 | rev_pass_normal | both_pass_normal):")
        p(f"  {'arm':25s} {'def_n/8':>8} {'rev_norm':>9} {'both_norm':>10} {'def_tight':>10} {'def_loose':>10}")

        # Sort arms: baseline first, then H-arms, then family specials
        arm_order = ["baseline"] + [f"H{h}" for h in FIXED_HORIZONS_186] + \
                    ["n4_freeze"] + [f"CALM2600-H{h}" for h in [1200,3000,6000]] + \
                    [f"THETA3-H{h}" for h in [1200,3000]]
        for arm_name in arm_order:
            if arm_name not in r["arm_results"]:
                continue
            ar = r["arm_results"][arm_name]
            dc = ar["defense_count"]
            rn = "PASS" if ar["revision_pass_normal"] else "fail"
            bn = "PASS" if ar["both_pass_normal"] else "fail"
            rt = "PASS" if ar["revision_pass_tight"] else "fail"
            bt = "PASS" if ar["both_pass_tight"] else "fail"
            rl = "PASS" if ar["revision_pass_loose"] else "fail"
            bl_ = "PASS" if ar["both_pass_loose"] else "fail"
            p(f"  {arm_name:25s} {dc:>8}/8 {rn:>9} {bn:>10} {ar['defense_count']:>10}/8 {ar['defense_count']:>10}/8")

        # oracle (defense count only, not in family)
        oracle_dc = defense_counts[(cell_idx, "oracle")]
        p(f"  {'oracle (diag)':25s} {oracle_dc:>8}/8  [not in family]")
        p()

        # Secondary tight/loose conjunct_i
        ci_tight = core_secondary.get((cell_idx, "tight"), None)
        ci_loose = core_secondary.get((cell_idx, "loose"), None)
        p(f"  Conjunct i (no-arm-covers): normal={r['conjunct_i']}  tight={ci_tight}  loose={ci_loose}")
        p()

        status = "CONFIRMED" if r["confirmed"] else ("EXCLUDED" if r["excluded"] else "NOT CONFIRMED")
        binding_str = f"  binding conjunct: {r['binding_conjunct']}" if r["binding_conjunct"] else ""
        relabel_str = f"  A/B relabel: {r['relabel']}" if r["confirmed"] else ""
        p(f"  => VERDICT: {status}{binding_str}{relabel_str}")
        if r["near_miss"] and not r["confirmed"]:
            nm = r["near_miss"]
            p(f"     Best near-miss: arm={nm['arm']}  def={nm['defense_count']}/8 "
              f"(gap={nm['defense_gap']})  rev_norm={nm['revision_pass_normal']}")
        p()
        p("-" * 60)
    p()

    p("=" * 80)
    p("V-SAMPLE TABLE")
    p("=" * 80)
    p()
    p(f"  {'cell_idx':>8} {'L':>5} {'K':>3} {'G':>5} {'best_def':>9} {'oracle':>7} {'bl_fail':>8} {'verdict'}")
    for cell in VSAMPLE_CELLS:
        cell_idx = cell["cell_idx"]
        v = vsample_results[cell_idx]
        p(f"  {cell_idx:>8} {cell['L']:>5} {cell['K']:>3} {cell['G']:>5} "
          f"{v['best_arm_def']:>9}/8 {v['oracle_count']:>7}/8 {v['baseline_fail_count']:>8}/8 "
          f"{v['verdict']}")
    p()

    p("=" * 80)
    p("REPLICATION VERDICT")
    p("=" * 80)
    p()
    p(f"  Core cells: {len(CORE_CELLS)}")
    p(f"  Excluded (F2/F3): {n_excluded}")
    p(f"  Confirmed: {n_confirmed}/{len(CORE_CELLS)}")
    p(f"  REPLICATION THRESHOLD: >= {CORE_CONFIRM_MIN}/6")
    p()
    rep_str = "REPLICATES" if replicates else "DOES NOT REPLICATE"
    p(f"  => REPLICATION VERDICT: {n_confirmed}/6  {rep_str}")
    p()

    p("=" * 80)
    p("FALSIFIER STATUS")
    p("=" * 80)
    p()
    p(f"  F1 (core < 4/6 -> NEGATIVE):            {'FIRED' if f1_fired else 'not fired'}")
    p(f"     confirmed={n_confirmed}/6, threshold=4")
    p(f"  F2 (oracle fails >= 2/8 in core cell):  {'FIRED — cells: ' + str(f2_cells) if f2_fired else 'not fired'}")
    p(f"  F3 (baseline deficit absent in core):   {'FIRED — cells: ' + str(f3_cells) if f3_fired else 'not fired'}")
    p(f"  PC1 (ahat_drift gate, per-session):      {pc1_note}")
    p()

    p("=" * 80)
    p("SECONDARY TIGHT/LOOSE DIAGNOSTICS (conjunct_i per core cell)")
    p("  (same arm-coverage check; different revision tolerance)")
    p("=" * 80)
    p()
    p(f"  {'cell':>30} {'normal':>8} {'tight':>7} {'loose':>7}")
    for cell in CORE_CELLS:
        cell_idx = cell["cell_idx"]
        ci_norm = core_results[cell_idx]["conjunct_i"]
        ci_tight = core_secondary.get((cell_idx, "tight"), "?")
        ci_loose = core_secondary.get((cell_idx, "loose"), "?")
        label = f"({cell['L']},{cell['K']},{cell['G']}) idx={cell_idx}"
        p(f"  {label:>30} {str(ci_norm):>8} {str(ci_tight):>7} {str(ci_loose):>7}")
    p()

    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)
    p()
    p(f"  Gate:                PASS")
    p(f"  Seeds:               {SEEDS_186}")
    p(f"  Core cells:          {len(CORE_CELLS)}")
    p(f"  Confirmed:           {n_confirmed}/6")
    p(f"  Excluded:            {n_excluded}")
    p(f"  Replication:         {rep_str}  ({n_confirmed}/6 >= {CORE_CONFIRM_MIN}/6 required)")
    p()
    p("  Core confirmations:")
    for cell in CORE_CELLS:
        cell_idx = cell["cell_idx"]
        r = core_results[cell_idx]
        stat = "CONFIRMED" if r["confirmed"] else ("EXCLUDED" if r["excluded"] else "NOT CONFIRMED")
        relabel = f"  relabel={r['relabel']}" if r["confirmed"] else ""
        binding = f"  binding={r['binding_conjunct']}" if r["binding_conjunct"] else ""
        p(f"    ({cell['L']},{cell['K']},{cell['G']}) idx={cell_idx}: {stat}{relabel}{binding}")
    p()
    p("  V-sample verdicts:")
    for cell in VSAMPLE_CELLS:
        cell_idx = cell["cell_idx"]
        v = vsample_results[cell_idx]
        p(f"    ({cell['L']},{cell['K']},{cell['G']}) idx={cell_idx}: {v['verdict']}")
    p()
    p("  Falsifier status:")
    p(f"    F1: {'FIRED' if f1_fired else 'not fired'}")
    p(f"    F2: {'FIRED (' + str(f2_cells) + ')' if f2_fired else 'not fired'}")
    p(f"    F3: {'FIRED (' + str(f3_cells) + ')' if f3_fired else 'not fired'}")
    p(f"    PC1: {pc1_note}")
    p()
    p(f"  Runtime: {runtime_min:.1f} min")
    p()
    p("  NOTE: CONFIRMATION-grade. Pre-registered commit e3e282e.")
    if replicates:
        p("  CRACK REPLICATES: Exp 187 controller re-test licensed in confirmed cells.")
    else:
        p("  CRACK DOES NOT REPLICATE: Config-sufficient closure; chapter appendix to be filed.")
    p("=" * 80)

    # Write text file
    with open(out_txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nText written to {out_txt_path}")
    print(f"Rows written to {out_rows_path}")
    print(f"Total runtime: {runtime_min:.1f} min")


if __name__ == "__main__":
    main()
