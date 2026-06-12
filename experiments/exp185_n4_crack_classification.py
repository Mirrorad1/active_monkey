"""Exp 185 — N4 Crack Classification (EXPLORATORY-classification).

PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit 8cc5cdb) BEFORE any data.
This script is EXPLORATORY-classification status: 3 seeds/cell, same seeds/schedules as
Exp 184 (diagnosis, not confirmation). No crack CLAIM — surviving A/B cells are candidates
pending rung-4 fresh-seed confirmation.

Predeclared shape predictions + FALSIFIERS (copied verbatim-in-substance from the
pre-registration card, commit 8cc5cdb, committed before any data):
P1 (CALM2600 unified coverage law): defense_pass(cell, H) iff H >= K*L+(K-1)*G-75 under
   CALM2600; F1 = >20% of CALM2600 cell x H outcomes on the wrong side.
P2 (H9000 latency law at scale): H9000 Phase-R latency = 9000 + baseline +- 50 and fails
   revision in all modes; F2 = deviation > 50 steps.
P3 (no config regression): no cell flips exists_H_both true->false when new arms added;
   F3 = >5 such flips (bookkeeping bug check).
F-cls: any candidate cell left unclassified is a classification failure.

New arms (6 configs x 80 cells x 3 seeds = 1440 W + 18 Phase-R):
  H9000 (THETA=3.5, RELEASE_CALM_SNAPS=8)
  THETA3-H1200 (THETA=3.0, H=1200)
  THETA3-H3000 (THETA=3.0, H=3000)
  CALM2600-H1200 (RELEASE_CALM_SNAPS=26, H=1200)  [2600/100 = 26 snaps]
  CALM2600-H3000 (RELEASE_CALM_SNAPS=26, H=3000)
  CALM2600-H6000 (RELEASE_CALM_SNAPS=26, H=6000)

EQUIVALENCE GATE (L15): with default params, re-run exp183-equivalence config
(baseline x s229, H1200 x s229) and emit the full got-vs-committed per-field table.
Gate FAIL => stop, report.

Classes:
  A = route-1 interval-empty (provable via latency=H+baseline law)
  B = route-2 dose-bound (no H defends; undissolved by all simple configs)
  D = dissolved (some new simple-config arm passes BOTH bars in that cell x mode)
  V = seed-variance (one seed flip would change candidate status)
  E = excluded (oracle defended 80/80 — all cells, same as exp184)
  C = excluded (min L=400 >> d=75 detection floor pure contribution)
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

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Configuration (verbatim Exp 184 where unchanged)
# ---------------------------------------------------------------------------

SEEDS = [240, 241, 242]

N_STEPS_PHASE_R = 15_000
CHUNK_SIZE = 100

LAMBDA = 0.9997
INIT_MASS = 1.0 / (1.0 - LAMBDA)   # ~3333.3

PHASE_R_START = 6000

BURST_SEED_OFFSET_R = 165_000
BURST_SEED_OFFSET_W_EXP183 = 160_000

EVAL = 100
FINE_EVAL = 25
MON_W_SNAPS = 40
CTRL_MBAR_WINDOW = 120
CTRL_MIN_SNAPS = 40
# Default RELEASE_CALM_SNAPS = 8 (Exp 184 verbatim)
DEFAULT_RELEASE_CALM_SNAPS = 8
REFRACTORY_CHECKS = 8

# Default THETA = 3.5 (Exp 184 verbatim)
DEFAULT_THETA = 3.5
E_STAR = 600.0
PRESSURE_WINDOW = 200
PRESSURE_FRAC = 0.6

# Exp 184 fixed horizons (for loading committed rows)
FIXED_HORIZONS_184 = [600, 900, 1200, 1800, 2400, 3000, 4200, 6000]

# New arms for Exp 185
# CALM2600: release_calm_steps=2600; snaps = 2600 / FINE_EVAL = 104 ... wait:
# RELEASE_CALM_SNAPS is checked every FINE_EVAL=25 steps, so 2600/25 = 104 snaps.
# Pre-reg says "release-calm lengthened to 2600 steps so freezes span every gap G<=2400"
# G_VALS = [200, 600, 1200, 2400]; need release_calm >= G+25 for spanning.
# 2600 steps / 25 = 104 snaps.
CALM2600_SNAPS = 104  # 2600 steps / 25 steps per snap

# THETA3 uses theta=3.0 with default release_calm
THETA3_THETA = 3.0

# H9000 uses default theta and default release_calm
H9000_H = 9000

# Grid definition (verbatim Exp 184)
L_VALS = [400, 800, 1200, 1600, 2400]
K_VALS = [1, 2, 3, 4]
G_VALS = [200, 600, 1200, 2400]

REVISION_MODES = [
    ("normal", 3000),
    ("tight",  1500),
    ("loose",  6000),
]

P6_HOLD = 2000

PC1_AHAT_DRIFT_MAX = 0.15
SETTLE_TV_FLAG_MAX = 0.05

DEFENSE_FRAC_THRESH = 0.5
DEFENSE_WINDOW_OFFSET_START = 1500
DEFENSE_WINDOW_OFFSET_END = 2000

MIN_PASS_FRACTION = 2   # >= 2/3 seeds

# Exp 183 equivalence gate constants
EXP183_BURST_WINDOWS = [
    (6000, 6800),
    (9000, 9800),
    (12000, 12800),
]
EXP183_N_STEPS = 15_000
EXP183_GATE_SEEDS = [229]
EXP183_GATE_ARMS = ["baseline", "H1200"]

FLOAT_ATOL = 1e-9

# New arms for exp185: (config_name, theta, release_calm_snaps, H, arm_mode_for_runner)
# arm_mode_for_runner: ("freeze_time", H) for all new H-arms; "baseline" for baseline
EXP185_NEW_CONFIGS = [
    # (config_name, theta, release_calm_snaps, arm_name_suffix, H_value)
    ("H9000",        DEFAULT_THETA,  DEFAULT_RELEASE_CALM_SNAPS, "H9000",   9000),
    ("THETA3-H1200", THETA3_THETA,   DEFAULT_RELEASE_CALM_SNAPS, "H1200",   1200),
    ("THETA3-H3000", THETA3_THETA,   DEFAULT_RELEASE_CALM_SNAPS, "H3000",   3000),
    ("CALM2600-H1200", DEFAULT_THETA, CALM2600_SNAPS,            "H1200",   1200),
    ("CALM2600-H3000", DEFAULT_THETA, CALM2600_SNAPS,            "H3000",   3000),
    ("CALM2600-H6000", DEFAULT_THETA, CALM2600_SNAPS,            "H6000",   6000),
]
# Full arm name in rows = config_name (e.g. "H9000", "THETA3-H1200", "CALM2600-H3000")

# Exp 184 arm names for combined sweep (the 8 fixed-H arms)
EXP184_H_ARMS = [f"H{h}" for h in FIXED_HORIZONS_184]


def enumerate_cells() -> list[dict]:
    cells = []
    idx = 0
    for L in L_VALS:
        for K in K_VALS:
            for G in G_VALS:
                n_steps = 6000 + K * L + (K - 1) * G + 2500
                cells.append({"L": L, "K": K, "G": G, "idx": idx, "n_steps": n_steps})
                idx += 1
    return cells

CELLS = enumerate_cells()
assert len(CELLS) == 80


# ---------------------------------------------------------------------------
# Helper functions (verbatim Exp 184)
# ---------------------------------------------------------------------------

def pi_of(v_row: np.ndarray) -> np.ndarray:
    s = v_row.sum()
    return v_row / s if s > 0 else np.ones_like(v_row) / len(v_row)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def phase_r_latency(expressed_arr: np.ndarray, regime_color: int, n_steps: int) -> int | None:
    max_t = n_steps - P6_HOLD
    for t in range(PHASE_R_START, max_t + 1):
        if expressed_arr[t] == regime_color:
            if np.all(expressed_arr[t:t + P6_HOLD] == regime_color):
                return t - PHASE_R_START
    return None


def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_plain(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_defense(
    expressed_arr: np.ndarray,
    burst_windows: list,
    burst_onset_color: list,
    n_steps: int,
) -> tuple[bool, float | None, int | None]:
    if not burst_windows:
        return False, None, None
    last_bi = len(burst_windows) - 1
    _, last_bend = burst_windows[last_bi]
    attack_color = burst_onset_color[last_bi]
    if attack_color is None:
        return False, None, None
    win_start = last_bend + DEFENSE_WINDOW_OFFSET_START
    win_end = last_bend + DEFENSE_WINDOW_OFFSET_END
    if win_end > n_steps:
        return False, None, attack_color
    frac = float(np.mean(expressed_arr[win_start:win_end] == attack_color))
    return frac < DEFENSE_FRAC_THRESH, frac, attack_color


def compute_final_gap(v_traj: np.ndarray, expressed_arr: np.ndarray, n_steps: int) -> float | None:
    if len(v_traj) == 0:
        return None
    last_v = v_traj[-1]
    fav = int(np.argmax(last_v))
    attack = int(np.argmin(last_v))
    return float(last_v[fav] - last_v[attack])


def defense_pass_for_cell(defense_by_arm_seed: dict, arm_name: str, seeds: list) -> bool:
    n_pass = sum(1 for seed in seeds if defense_by_arm_seed.get((arm_name, seed), False))
    return n_pass >= MIN_PASS_FRACTION


def revision_pass_fn(
    h_arm_label: str,
    tolerance: int,
    latencies: dict,
    seeds: list,
) -> bool:
    """revision_pass = latency <= baseline+tolerance in >=2/3 seeds."""
    n_pass = 0
    for seed in seeds:
        bl_lat = latencies.get(("baseline", seed))
        arm_lat = latencies.get((h_arm_label, seed))
        if arm_lat is None or bl_lat is None:
            continue
        if arm_lat <= bl_lat + tolerance:
            n_pass += 1
    return n_pass >= MIN_PASS_FRACTION


# ---------------------------------------------------------------------------
# Generalized runner: run_fork_schedule_185
# Copied from exp184 with THETA and RELEASE_CALM_SNAPS exposed as parameters.
# ALL defaults identical to exp184 values (DEFAULT_THETA=3.5, DEFAULT_RELEASE_CALM_SNAPS=8).
# ---------------------------------------------------------------------------

def run_fork_schedule_185(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    arm_name: str,
    arm_mode: object,
    phase: str,           # "W" | "R"
    burst_windows: list,
    color_mode: str,      # 'endogenous' | 'exogenous_fixed'
    reloc_rng_seed: int,
    n_steps: int,
    n_actions: int = 4,
    theta: float = DEFAULT_THETA,
    release_calm_snaps: int = DEFAULT_RELEASE_CALM_SNAPS,
) -> dict:
    """Run one fork for n_steps steps under the given arm and phase.

    Identical to exp184's run_fork_schedule with theta and release_calm_snaps
    as exposed parameters (defaults unchanged from exp184).

    Phase W: bursts at burst_windows with captivity; color determined by color_mode.
    Phase R: PERMANENT captivity from PHASE_R_START onward; color = argmin(v) at onset.
    """
    n_chunks = n_steps // CHUNK_SIZE
    assert n_chunks * CHUNK_SIZE == n_steps

    fork_name = f"exp185_{arm_name}_{phase}_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    B = c.world.transition_matrix()

    vc = c.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v = (vc / vc_sum) * INIT_MASS
    else:
        v = np.ones(n_colors) * (INIT_MASS / n_colors)

    color_cells: list[list[int]] = [[] for _ in range(n_colors)]
    for cell_idx, color in enumerate(base_cmap):
        color_cells[color].append(cell_idx)
    color_cells_arr = [np.array(lst, dtype=np.int32) for lst in color_cells]

    if phase == "W":
        burst_rng = np.random.default_rng(reloc_rng_seed)
    else:
        burst_rng = np.random.default_rng(BURST_SEED_OFFSET_R + fork_seed)

    burst_step_set: set[int] = set()
    if phase == "W":
        for bstart, bend in burst_windows:
            burst_step_set.update(range(bstart, bend))

    oracle_burst_set: set[int] = burst_step_set if arm_mode == "oracle" else set()

    expressed_arr = np.empty(n_steps, dtype=np.int32)
    true_pos_arr = np.empty(n_steps, dtype=np.int32)
    obs_arr = np.empty(n_steps, dtype=np.int32)
    state_arr = np.empty(n_steps, dtype=np.int32)

    v_traj: list[np.ndarray] = []

    v_fine: list[np.ndarray] = []
    k_since_reset: int = 0

    mismatch_history: list[float] = []

    m_bar_floor: float | None = None

    events: list[dict] = []

    burst_preburst_fav: list[int | None] = [None] * len(burst_windows)
    burst_onset_color: list[int | None] = [None] * len(burst_windows)
    current_burst_color: int | None = None
    current_burst_idx: int | None = None

    exogenous_attack_color: int | None = None

    regime_color: int | None = None
    regime_rng_active = False

    mbar_phase_r_snapshots: list[float] = []

    freeze_state = "NORMAL"
    v_ref: np.ndarray | None = None
    pi_ref: np.ndarray | None = None
    m_bar_frozen: float = 0.0
    blocked_w_by_color: np.ndarray = np.zeros(n_colors, dtype=np.float64)
    calm_count: int = 0
    frozen_steps: int = 0
    entry_step: int = 0
    directional_pressure_acc: float = 0.0
    resist_live_mismatch: list[float] = []
    resist_mbar_frozen_check: list[float] = []

    checks_since_release: int = REFRACTORY_CHECKS

    is_freeze_arm = arm_mode in ("freeze_evidence", "oracle") or (
        isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time"
    )

    gap_start: list[float | None] = [None] * len(burst_windows)
    gap_end: list[float | None] = [None] * len(burst_windows)
    d_b: list[float | None] = [None] * len(burst_windows)
    tv_b: list[float | None] = [None] * len(burst_windows)
    pi_at_burst_start: list[np.ndarray | None] = [None] * len(burst_windows)

    head_dose: list[float] = [0.0] * len(burst_windows)
    gap_plateau: list[float | None] = [None] * len(burst_windows)
    _burst_head_active: list[bool] = [False] * len(burst_windows)
    _burst_first_freeze_step: list[int | None] = [None] * len(burst_windows)
    _burst_plateau_v_prev: list[np.ndarray | None] = [None] * len(burst_windows)
    _burst_plateau_found: list[bool] = [False] * len(burst_windows)

    per_burst_expr_frac: list[float | None] = [None] * len(burst_windows)

    settle_tv_val: float | None = None
    settle_pi_5000: np.ndarray | None = None

    A_hat_start = c._A_hat().copy()

    global_step = 0

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(CHUNK_SIZE):
            t = global_step

            if phase == "W" and color_mode == "exogenous_fixed" and t == 6000:
                exogenous_attack_color = int(np.argmin(v))

            if t == 5000:
                settle_pi_5000 = pi_of(v.copy())
            if t == 6000 and settle_pi_5000 is not None:
                settle_tv_val = tv(settle_pi_5000, pi_of(v.copy()))

            in_burst = False
            burst_idx_now: int | None = None

            if phase == "W":
                in_burst = t in burst_step_set
                for bi, (bstart, bend) in enumerate(burst_windows):
                    if bstart <= t < bend:
                        burst_idx_now = bi
                        break

                if burst_idx_now is not None and t == burst_windows[burst_idx_now][0]:
                    pre_fav = int(np.argmax(v))
                    burst_preburst_fav[burst_idx_now] = pre_fav

                    if color_mode == "endogenous":
                        bc = int(np.argmin(v))
                    else:
                        bc = exogenous_attack_color
                        if bc is None:
                            bc = int(np.argmin(v))

                    burst_onset_color[burst_idx_now] = bc
                    current_burst_color = bc
                    current_burst_idx = burst_idx_now
                    pi_at_burst_start[burst_idx_now] = pi_of(v.copy())
                    gap_start[burst_idx_now] = float(v[pre_fav] - v[bc])
                    _burst_head_active[burst_idx_now] = True
                    _burst_first_freeze_step[burst_idx_now] = None
                    _burst_plateau_found[burst_idx_now] = False

                if burst_idx_now is None and current_burst_idx is not None:
                    for bi, (bstart, bend) in enumerate(burst_windows):
                        if t == bend:
                            if burst_preburst_fav[bi] is not None and burst_onset_color[bi] is not None:
                                pi_end = pi_of(v.copy())
                                gap_end[bi] = float(v[burst_preburst_fav[bi]] - v[burst_onset_color[bi]])
                                d_b[bi] = float(pi_end[burst_onset_color[bi]] - pi_at_burst_start[bi][burst_onset_color[bi]])
                                tv_b[bi] = tv(pi_end, pi_at_burst_start[bi])
                            current_burst_color = None
                            current_burst_idx = None
                            break

            if phase == "R" and t == PHASE_R_START:
                regime_color = int(np.argmin(v))
                regime_rng_active = True

            in_regime = phase == "R" and regime_rng_active and t >= PHASE_R_START

            obs = int(base_cmap[c.true_pos])
            obs_arr[t] = obs
            true_pos_arr[t] = c.true_pos

            A_hat = c._A_hat()

            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom_qs = qs_updated.sum()
            if denom_qs > 0:
                qs_updated = qs_updated / denom_qs
            else:
                qs_updated = np.ones(n_cells) / n_cells

            c.pA[obs, :] += qs_updated

            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            w = math.exp(-h_predicted)

            if arm_mode == "oracle":
                resist_now = (phase == "W") and (t in oracle_burst_set)
                if resist_now:
                    state_arr[t] = 1
                    if in_burst and phase == "W":
                        cells_of_bc = color_cells_arr[current_burst_color]
                        if len(cells_of_bc) > 0:
                            c.true_pos = int(burst_rng.choice(cells_of_bc))
                        qs_next = np.zeros(n_cells)
                        qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                        c.qs = qs_next
                    elif in_regime:
                        cells_of_rc = color_cells_arr[regime_color]
                        if len(cells_of_rc) > 0:
                            c.true_pos = int(burst_rng.choice(cells_of_rc))
                        qs_next = np.zeros(n_cells)
                        qs_next[cells_of_rc] = 1.0 / len(cells_of_rc)
                        c.qs = qs_next
                    else:
                        action = int(rng.integers(0, n_actions))
                        c.true_pos = c.world.move(c.true_pos, action)
                        c.qs = B[:, :, action] @ qs_updated
                    expressed = int(np.argmax(v))
                    expressed_arr[t] = expressed
                    c.age_steps += 1
                    global_step += 1
                    if global_step % EVAL == 0:
                        v_traj.append(v.copy())
                    continue
                else:
                    state_arr[t] = 0

            elif is_freeze_arm and arm_mode != "oracle":
                if freeze_state == "RESIST":
                    frozen_steps += 1
                    blocked_w_by_color[obs] += w
                    c_star_now = int(np.argmax(blocked_w_by_color))
                    E_blocked_now = float(blocked_w_by_color[c_star_now])

                    e_obs_vec = np.zeros(n_colors)
                    e_obs_vec[obs] = 1.0
                    e_cstar_vec = np.zeros(n_colors)
                    e_cstar_vec[c_star_now] = 1.0
                    dir_score = float(np.dot(e_obs_vec - pi_ref, e_cstar_vec - pi_ref))
                    directional_pressure_acc += dir_score

                    state_arr[t] = 1

                    if in_burst and phase == "W":
                        cells_of_bc = color_cells_arr[current_burst_color]
                        if len(cells_of_bc) > 0:
                            c.true_pos = int(burst_rng.choice(cells_of_bc))
                        qs_next = np.zeros(n_cells)
                        qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                        c.qs = qs_next
                    elif in_regime:
                        cells_of_rc = color_cells_arr[regime_color]
                        if len(cells_of_rc) > 0:
                            c.true_pos = int(burst_rng.choice(cells_of_rc))
                        qs_next = np.zeros(n_cells)
                        qs_next[cells_of_rc] = 1.0 / len(cells_of_rc)
                        c.qs = qs_next
                    else:
                        action = int(rng.integers(0, n_actions))
                        c.true_pos = c.world.move(c.true_pos, action)
                        c.qs = B[:, :, action] @ qs_updated

                    expressed = int(np.argmax(v))
                    expressed_arr[t] = expressed
                    c.age_steps += 1
                    global_step += 1

                    if global_step % EVAL == 0:
                        v_traj.append(v.copy())

                    if global_step % FINE_EVAL == 0:
                        checks_since_release += 1

                        resist_mbar_frozen_check.append(m_bar_frozen)

                        t_now = global_step
                        t_pressure_start = max(0, t_now - PRESSURE_WINDOW)
                        obs_window = obs_arr[t_pressure_start:t_now]
                        c_star_pressure = int(np.argmax(blocked_w_by_color))
                        if len(obs_window) > 0:
                            pressure_active = (
                                float(np.sum(obs_window == c_star_pressure)) / len(obs_window)
                            ) >= PRESSURE_FRAC
                        else:
                            pressure_active = False

                        if not pressure_active:
                            calm_count += 1
                        else:
                            calm_count = 0

                        c_star_final = int(np.argmax(blocked_w_by_color))
                        E_blocked_final = float(blocked_w_by_color[c_star_final])

                        released = False
                        release_label = None

                        # Use parameter release_calm_snaps (key change vs exp184)
                        if calm_count >= release_calm_snaps:
                            release_label = "transient"
                            released = True

                        if not released:
                            if arm_mode == "freeze_evidence":
                                if pressure_active and E_blocked_final >= E_STAR:
                                    release_label = "concession"
                                    released = True
                            elif isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time":
                                H = arm_mode[1]
                                if frozen_steps >= H:
                                    release_label = "concession"
                                    released = True

                        if released:
                            trigger_latency_ev: int | None = None
                            for bi, (bstart, bend) in enumerate(burst_windows):
                                if bstart <= entry_step < bend:
                                    trigger_latency_ev = entry_step - bstart
                                    break

                            events.append({
                                "entry_step": entry_step,
                                "exit_step": t_now,
                                "label": release_label,
                                "E_blocked": E_blocked_final,
                                "c_star": c_star_final,
                                "frozen_steps": frozen_steps,
                                "directional_pressure_acc": directional_pressure_acc,
                                "trigger_latency": trigger_latency_ev,
                            })
                            freeze_state = "NORMAL"
                            mismatch_history.clear()
                            v_fine.clear()
                            k_since_reset = 0
                            calm_count = 0
                            frozen_steps = 0
                            directional_pressure_acc = 0.0
                            resist_live_mismatch.clear()
                            resist_mbar_frozen_check.clear()
                            checks_since_release = 0
                    continue

                else:
                    state_arr[t] = 0

            else:
                state_arr[t] = 0

            # Value update (NORMAL state)
            if arm_mode == "baseline":
                g_t = 1.0
            elif arm_mode in ("freeze_evidence",) or (
                isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time"
            ):
                g_t = 1.0
            elif arm_mode == "oracle":
                g_t = 1.0
            else:
                g_t = float(arm_mode)

            v *= LAMBDA
            v[obs] += g_t * w
            c.value_counts[obs] += g_t * w

            expressed = int(np.argmax(v))
            expressed_arr[t] = expressed

            if in_burst and phase == "W":
                cells_of_bc = color_cells_arr[current_burst_color]
                if len(cells_of_bc) > 0:
                    c.true_pos = int(burst_rng.choice(cells_of_bc))
                qs_next = np.zeros(n_cells)
                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                c.qs = qs_next
            elif in_regime:
                cells_of_rc = color_cells_arr[regime_color]
                if len(cells_of_rc) > 0:
                    c.true_pos = int(burst_rng.choice(cells_of_rc))
                qs_next = np.zeros(n_cells)
                qs_next[cells_of_rc] = 1.0 / len(cells_of_rc)
                c.qs = qs_next
            else:
                action = int(rng.integers(0, n_actions))
                c.true_pos = c.world.move(c.true_pos, action)
                c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1
            global_step += 1

            if global_step % EVAL == 0:
                v_traj.append(v.copy())

            if global_step % FINE_EVAL == 0:
                checks_since_release += 1

                if is_freeze_arm and arm_mode != "oracle" and freeze_state == "NORMAL":
                    v_fine.append(v.copy())
                    k_since_reset += 1
                    k = k_since_reset - 1

                    if k_since_reset >= 7:
                        Lf = min(MON_W_SNAPS, k_since_reset - 5)
                        if Lf >= 2:
                            v_base = v_fine[k - 4]
                            v_drift_ref = v_fine[k - 4 - Lf]
                            slope = (v_base - v_drift_ref) / Lf
                            v_hat = v_base + slope * 4
                            m_k = float(np.linalg.norm(v_hat - v_fine[k]))
                            mismatch_history.append(m_k)

                            n_hist = len(mismatch_history)

                            live_m_bar: float | None = None
                            if n_hist >= CTRL_MIN_SNAPS:
                                tail = mismatch_history[-CTRL_MBAR_WINDOW:]
                                live_m_bar = statistics.median(tail)
                                m_bar_floor = live_m_bar

                                if phase == "R" and global_step >= PHASE_R_START:
                                    mbar_phase_r_snapshots.append(live_m_bar)

                            if live_m_bar is not None:
                                trigger_denom = live_m_bar
                            elif m_bar_floor is not None:
                                trigger_denom = m_bar_floor
                            else:
                                trigger_denom = None

                            # Use parameter theta (key change vs exp184)
                            if (
                                trigger_denom is not None
                                and m_k > 0
                                and checks_since_release >= REFRACTORY_CHECKS
                                and (m_k / max(trigger_denom, 1e-12)) >= theta
                            ):
                                freeze_state = "RESIST"
                                entry_step = global_step
                                v_ref = v.copy()
                                pi_ref = pi_of(v_ref)
                                m_bar_frozen = trigger_denom
                                blocked_w_by_color = np.zeros(n_colors, dtype=np.float64)
                                calm_count = 0
                                frozen_steps = 0
                                directional_pressure_acc = 0.0
                                resist_live_mismatch = []
                                resist_mbar_frozen_check = []

    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    n_bursts = len(burst_windows)
    for bi in range(n_bursts):
        bstart, bend = burst_windows[bi]
        win_start = bend + DEFENSE_WINDOW_OFFSET_START
        win_end = bend + DEFENSE_WINDOW_OFFSET_END
        fits = True
        if bi + 1 < n_bursts:
            next_bstart = burst_windows[bi + 1][0]
            if win_end > next_bstart:
                fits = False
        if win_end > n_steps:
            fits = False
        if fits and burst_onset_color[bi] is not None:
            bc = burst_onset_color[bi]
            per_burst_expr_frac[bi] = float(
                np.mean(expressed_arr[win_start:win_end] == bc)
            )

    return {
        "fork_seed": fork_seed,
        "arm_name": arm_name,
        "arm_mode": arm_mode,
        "phase": phase,
        "ahat_drift": ahat_drift,
        "expressed_arr": expressed_arr,
        "true_pos_arr": true_pos_arr,
        "obs_arr": obs_arr,
        "state_arr": state_arr,
        "v_traj": np.array(v_traj) if v_traj else np.zeros((0, n_colors)),
        "burst_preburst_fav": burst_preburst_fav,
        "burst_onset_color": burst_onset_color,
        "regime_color": regime_color,
        "events": events,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "d_b": d_b,
        "tv_b": tv_b,
        "per_burst_expr_frac": per_burst_expr_frac,
        "settle_tv": settle_tv_val,
        "mbar_phase_r_snapshots": list(mbar_phase_r_snapshots),
        "exogenous_attack_color": exogenous_attack_color,
    }


# ---------------------------------------------------------------------------
# Equivalence gate (L15)
# ---------------------------------------------------------------------------

def load_committed_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_equivalence_gate_185(
    mirro_root, base_cmap, n_colors, committed_rows_path: Path
) -> tuple[bool, str]:
    """Run equivalence gate: reproduce exp183 baseline x s229 and H1200 x s229.

    Emits full got-vs-committed per-field table (L15 pattern).
    Returns (gate_pass, detail_string).
    """
    committed_rows = load_committed_rows(committed_rows_path)
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
                    bc_exp183 = rr["burst_onset_color"][bi]
                    gate_frac = float(np.mean(rr["expressed_arr"][win_start:win_end] == bc_exp183))
                    gate_recovered = gate_frac < 0.5
                else:
                    gate_recovered = None

                checks = [
                    ("gap_start", rr["gap_start"][bi], cr["gap_start"]),
                    ("gap_end", rr["gap_end"][bi], cr["gap_end"]),
                    ("d_b", rr["d_b"][bi], cr["d_b"]),
                    ("tv_b", rr["tv_b"][bi], cr["tv_b"]),
                    ("recovered", gate_recovered, cr["recovered"]),
                    ("n_events", len(rr["events"]), cr["n_events"]),
                ]

                for field_name, got, expected in checks:
                    if got is None and expected is None:
                        detail_lines.append(f"  b{bi} {field_name:<14} got={got!r:<40} committed={expected!r:<40} OK (both None)")
                        continue
                    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
                        ok = abs(float(got) - float(expected)) <= FLOAT_ATOL
                    elif type(got) == type(expected):
                        ok = got == expected
                    else:
                        ok = str(got) == str(expected)

                    status = "OK" if ok else "MISMATCH"
                    detail_lines.append(
                        f"  b{bi} {field_name:<14} got={str(got):<40} committed={str(expected):<40} {status}"
                    )
                    if not ok:
                        session_pass = False
                        all_pass = False

                got_evs = rr["events"]
                exp_evs = cr.get("events_summary", [])
                if len(got_evs) != len(exp_evs):
                    detail_lines.append(
                        f"  b{bi} event count      got={len(got_evs):<40} committed={len(exp_evs):<40} MISMATCH"
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
                                f"  ev{ei} {ef:<14} got={str(gv):<40} committed={str(ev):<40} {status}"
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
                                f"  ev{ei} {'E_blocked':<14} got={str(gv):<40} committed={str(ev):<40} {status}"
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
    print("Exp 185 — N4 Crack Classification (EXPLORATORY-classification)")
    print("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit 8cc5cdb)")
    print("Seeds: 240-242; 6 new configs x 80 cells x 3 seeds = 1440 W + 18 R")
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
    assert spine_argmax != spine_argmin, "argmin == argmax — degenerate"
    print(f"Spine standing favorite: color {spine_argmax}, attack candidate: color {spine_argmin}")
    print()

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows_path = out_dir / "exp185_rows.json"
    out_txt_path = out_dir / "exp185.txt"
    committed_183_path = out_dir / "exp183_rows.json"
    committed_184_path = out_dir / "exp184_rows.json"

    # ====================================================================
    # STEP 1: EQUIVALENCE GATE (L15)
    # ====================================================================
    print("=" * 80)
    print("STEP 1: EQUIVALENCE GATE (L15)")
    print("Reproducing exp183 (baseline x s229) and (H1200 x s229) via exp185 runner")
    print("Default params: theta=3.5, release_calm_snaps=8")
    print("=" * 80)
    t_gate = time.time()

    gate_pass, gate_detail = run_equivalence_gate_185(
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
    # STEP 2: NEW W SESSIONS (6 configs x 80 cells x 3 seeds = 1440)
    # ====================================================================
    print("=" * 80)
    print("STEP 2: NEW W SESSIONS (6 configs x 80 cells x 3 seeds = 1440 sessions)")
    print("=" * 80)

    # new_w_defense: (config_name, cell_idx, seed) -> bool
    new_w_defense: dict = {}
    new_w_final_expr_frac: dict = {}
    new_w_ahat_drift: dict = {}

    t_grid = time.time()

    for ci, cell in enumerate(CELLS):
        L, K, G = cell["L"], cell["K"], cell["G"]
        cell_idx = cell["idx"]
        n_steps = cell["n_steps"]

        burst_windows = []
        cur = 6000
        for ki in range(K):
            bstart = cur
            bend = cur + L
            burst_windows.append((bstart, bend))
            cur = bend
            if ki < K - 1:
                cur += G

        for config_name, theta_val, rc_snaps, h_suffix, h_val in EXP185_NEW_CONFIGS:
            arm_mode = ("freeze_time", h_val)
            for seed in SEEDS:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

                reloc_rng_seed = 190_000 + 1000 * cell_idx + seed

                rr = run_fork_schedule_185(
                    mirro=root,
                    fork_seed=seed,
                    base_cmap=base_cmap,
                    n_colors=n_colors,
                    arm_name=config_name,
                    arm_mode=arm_mode,
                    phase="W",
                    burst_windows=burst_windows,
                    color_mode="exogenous_fixed",
                    reloc_rng_seed=reloc_rng_seed,
                    n_steps=n_steps,
                    theta=theta_val,
                    release_calm_snaps=rc_snaps,
                )
                rr["burst_windows_used"] = burst_windows

                defense, frac, attack_color = compute_defense(
                    rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps
                )
                new_w_defense[(config_name, cell_idx, seed)] = bool(defense)
                new_w_final_expr_frac[(config_name, cell_idx, seed)] = frac
                new_w_ahat_drift[(config_name, cell_idx, seed)] = rr["ahat_drift"]

                # Write row immediately
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
                    "exp": 185,
                    "kind": "W",
                    "cell": {"L": L, "K": K, "G": G, "idx": cell_idx},
                    "arm": config_name,
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
                    "flags": [],
                    "theta": theta_val,
                    "release_calm_snaps": rc_snaps,
                }
                # Accumulate rows for writing after loop
                # (store as list to write all at once)
                if not hasattr(main, '_rows_buffer'):
                    main._rows_buffer = []
                main._rows_buffer.append(row)

        elapsed = time.time() - t_grid
        if ci == 0:
            projected = elapsed * len(CELLS)
            print(f"  cell {ci:02d} (L={L},K={K},G={G},idx={cell_idx}) done in {elapsed:.1f}s "
                  f"| projected total: {projected/60:.1f} min", flush=True)
        elif (ci + 1) % 10 == 0 or ci == len(CELLS) - 1:
            rate = elapsed / (ci + 1)
            remaining = rate * (len(CELLS) - ci - 1)
            print(f"  cell {ci:02d}/{len(CELLS)-1} (L={L},K={K},G={G}) | "
                  f"elapsed {elapsed:.0f}s | ETA {remaining:.0f}s", flush=True)

    t_grid_done = time.time()
    print(f"Grid W done: {(t_grid_done - t_grid)/60:.1f} min total")
    print()

    # ====================================================================
    # STEP 3: PHASE-R SESSIONS (6 configs + baseline x 3 seeds = 21 sessions)
    # Each new config needs its own R latencies; baseline reused from exp184
    # We run baseline + the 6 new config arms (7 arms x 3 seeds = 21 sessions)
    # ====================================================================
    print("=" * 80)
    print("STEP 3: PHASE-R SESSIONS (6 new configs + baseline x 3 seeds = 21 R-sessions)")
    print("=" * 80)

    # R latencies: (arm_label, seed) -> latency
    # arm_label for R: config_name (e.g. "H9000") for freeze-time arms,
    # "baseline" for the baseline
    # For each new config, the R arm uses the same theta/release_calm_snaps
    r_latencies_185: dict = {}  # (arm_label, seed) -> int|None
    r_results_buffer: list[dict] = []

    # Build R arms: baseline + 6 new configs
    r_arms_185 = [("baseline", "baseline", DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS)]
    for config_name, theta_val, rc_snaps, h_suffix, h_val in EXP185_NEW_CONFIGS:
        r_arms_185.append((config_name, ("freeze_time", h_val), theta_val, rc_snaps))

    t_r = time.time()
    for arm_label, arm_mode_r, theta_val, rc_snaps in r_arms_185:
        for seed in SEEDS:
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
            r_latencies_185[(arm_label, seed)] = lat

            print(f"  arm={arm_label:20s} seed={seed}  regime_color={rc}  "
                  f"latency={lat}  ahat_drift={rr['ahat_drift']:.5f}", flush=True)

            # Store R row
            r_row = {
                "exp": 185,
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
            r_results_buffer.append(r_row)

    print(f"Phase-R done: {time.time()-t_r:.1f}s")
    print()

    # ====================================================================
    # STEP 4: WRITE W+R ROWS TO JSONL
    # ====================================================================
    with open(out_rows_path, "w") as fh:
        # W rows
        if hasattr(main, '_rows_buffer'):
            for row in main._rows_buffer:
                fh.write(json.dumps(row) + "\n")
        # R rows
        for row in r_results_buffer:
            fh.write(json.dumps(row) + "\n")

    print(f"Rows written to {out_rows_path}")
    print()

    # ====================================================================
    # STEP 5: ANALYSIS
    # ====================================================================
    print("=" * 80)
    print("STEP 5: ANALYSIS")
    print("=" * 80)

    # ---- 5a. Load committed exp184 rows; recompute exists_H_both + candidate sets ----
    print("5a. Loading committed exp184 rows and recomputing candidate sets...")
    e184_rows = load_committed_rows(committed_184_path)

    # Build defense_results_184: (cell_idx, arm_name, seed) -> bool
    defense_results_184: dict = {}
    # R latencies from exp184
    r_latencies_184: dict = {}  # (arm, seed) -> int|None

    for row in e184_rows:
        if row["kind"] == "W":
            cell_idx = row["cell"]["idx"]
            arm = row["arm"]
            seed = row["seed"]
            defense_results_184[(cell_idx, arm, seed)] = bool(row["defense"])
        elif row["kind"] == "R":
            arm = row["arm"]
            seed = row["seed"]
            r_latencies_184[(arm, seed)] = row.get("latency")

    # defense_pass_184: (cell_idx, arm) -> bool
    defense_pass_184: dict = {}
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        for arm_name in EXP184_H_ARMS + ["oracle", "baseline", "n4_freeze"]:
            n_pass = sum(
                1 for seed in SEEDS
                if defense_results_184.get((cell_idx, arm_name, seed), False)
            )
            defense_pass_184[(cell_idx, arm_name)] = n_pass >= MIN_PASS_FRACTION

    # Revision pass from exp184 latencies
    rev_pass_184: dict = {}  # (h_arm, mode_name) -> bool
    for h in FIXED_HORIZONS_184:
        h_arm = f"H{h}"
        for mode_name, tolerance in REVISION_MODES:
            rev_pass_184[(h_arm, mode_name)] = revision_pass_fn(
                h_arm, tolerance, r_latencies_184, SEEDS
            )

    # exists_H_both from exp184 only
    exists_h_both_184: dict = {}  # (cell_idx, mode_name) -> bool
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        for mode_name, tolerance in REVISION_MODES:
            found = False
            for h in FIXED_HORIZONS_184:
                h_arm = f"H{h}"
                if (defense_pass_184.get((cell_idx, h_arm), False)
                        and rev_pass_184.get((h_arm, mode_name), False)):
                    found = True
                    break
            exists_h_both_184[(cell_idx, mode_name)] = found

    # Oracle pass
    oracle_pass_184: dict = {}  # cell_idx -> bool
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        n_pass = sum(
            1 for seed in SEEDS
            if defense_results_184.get((cell_idx, "oracle", seed), False)
        )
        oracle_pass_184[cell_idx] = n_pass >= MIN_PASS_FRACTION

    # Candidate cells from exp184 only
    candidate_cells_184: dict = {}  # mode_name -> list of cell_idx
    for mode_name, _ in REVISION_MODES:
        candidates = []
        for ci, cell in enumerate(CELLS):
            cell_idx = cell["idx"]
            if (not exists_h_both_184.get((cell_idx, mode_name), True)
                    and oracle_pass_184.get(cell_idx, False)):
                candidates.append(cell_idx)
        candidate_cells_184[mode_name] = candidates

    n_normal = len(candidate_cells_184["normal"])
    n_tight = len(candidate_cells_184["tight"])
    n_loose = len(candidate_cells_184["loose"])
    print(f"  Sanity check — exp184 candidate counts:")
    print(f"    normal: {n_normal} (expected 36)")
    print(f"    tight:  {n_tight} (expected 57)")
    print(f"    loose:  {n_loose} (expected 32)")
    sanity_ok = (n_normal == 36 and n_tight == 57 and n_loose == 32)
    if sanity_ok:
        print("  SANITY PASS: candidate counts match committed values")
    else:
        print("  SANITY FAIL: candidate counts do not match committed values!")
    print()

    # ---- 5b. Combined sweep: compute revision passes for new configs ----
    print("5b. Computing revision passes for new configs from Phase-R latencies...")

    # For new configs, their revision labeling uses config_name as the arm_label in R
    # Revision_pass for new config: compare r_latencies_185[(config_name, seed)]
    # vs r_latencies_185[("baseline", seed)]
    rev_pass_185: dict = {}  # (config_name, mode_name) -> bool
    for config_name, theta_val, rc_snaps, h_suffix, h_val in EXP185_NEW_CONFIGS:
        for mode_name, tolerance in REVISION_MODES:
            n_pass = 0
            for seed in SEEDS:
                bl_lat = r_latencies_185.get(("baseline", seed))
                arm_lat = r_latencies_185.get((config_name, seed))
                if arm_lat is None or bl_lat is None:
                    continue
                if arm_lat <= bl_lat + tolerance:
                    n_pass += 1
            rev_pass_185[(config_name, mode_name)] = n_pass >= MIN_PASS_FRACTION

    # defense_pass for new configs: (config_name, cell_idx) -> bool
    defense_pass_185: dict = {}
    for config_name, _, _, _, _ in EXP185_NEW_CONFIGS:
        for ci, cell in enumerate(CELLS):
            cell_idx = cell["idx"]
            n_pass = sum(
                1 for seed in SEEDS
                if new_w_defense.get((config_name, cell_idx, seed), False)
            )
            defense_pass_185[(config_name, cell_idx)] = n_pass >= MIN_PASS_FRACTION

    # Combined exists_H_both (exp184 arms + new exp185 configs)
    exists_h_both_combined: dict = {}  # (cell_idx, mode_name) -> bool
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        for mode_name, tolerance in REVISION_MODES:
            # Start with exp184 result
            found = exists_h_both_184.get((cell_idx, mode_name), False)
            if not found:
                # Check new configs
                for config_name, _, _, _, _ in EXP185_NEW_CONFIGS:
                    def_ok = defense_pass_185.get((config_name, cell_idx), False)
                    rev_ok = rev_pass_185.get((config_name, mode_name), False)
                    if def_ok and rev_ok:
                        found = True
                        break
            exists_h_both_combined[(cell_idx, mode_name)] = found

    # ---- Dissolution map ----
    # dissolved_by[(cell_idx, mode_name)] = config_name (first dissolving config)
    dissolved_by_map: dict = {}  # (cell_idx, mode_name) -> config_name|None
    for mode_name, tolerance in REVISION_MODES:
        for cell_idx_cand in candidate_cells_184[mode_name]:
            first_dissolving = None
            for config_name, _, _, _, _ in EXP185_NEW_CONFIGS:
                def_ok = defense_pass_185.get((config_name, cell_idx_cand), False)
                rev_ok = rev_pass_185.get((config_name, mode_name), False)
                if def_ok and rev_ok:
                    first_dissolving = config_name
                    break
            dissolved_by_map[(cell_idx_cand, mode_name)] = first_dissolving

    # ---- 5c. Classification (A/B/D/V) ----
    print("5c. Classifying candidate cells...")
    print("  V rule: V overrides A/B (not D) when any revision-passing arm has defense")
    print("          count == 1 among the combined sweep arms for that cell x mode.")
    print("  A rule: interval-empty (provable by latency=H+baseline law):")
    print("          ALL H in FIXED_HORIZONS_184 that defend in that cell fail revision,")
    print("          OR: min defending H > max revision-passing H (interval empty).")
    print("  B rule: no H defends in combined sweep (all configs fail defense); dose-bound.")
    print()

    # For V: check if any arm (in combined sweep) that passes revision for this mode
    # has defense count == 1 (one seed flip changes candidate status)
    def is_v_cell(cell_idx, mode_name, tolerance):
        """Return True if some revision-passing arm has defense count == 1."""
        # Check exp184 H arms
        for h in FIXED_HORIZONS_184:
            h_arm = f"H{h}"
            if not rev_pass_184.get((h_arm, mode_name), False):
                continue
            n_def = sum(
                1 for seed in SEEDS
                if defense_results_184.get((cell_idx, h_arm, seed), False)
            )
            if n_def == 1:
                return True
        # Check new configs
        for config_name, _, _, _, _ in EXP185_NEW_CONFIGS:
            if not rev_pass_185.get((config_name, mode_name), False):
                continue
            n_def = sum(
                1 for seed in SEEDS
                if new_w_defense.get((config_name, cell_idx, seed), False)
            )
            if n_def == 1:
                return True
        return False

    def classify_cell(cell_idx, mode_name, tolerance):
        """Classify a candidate cell into A/B/D/V.

        D: dissolved by a new config (both bars pass).
        V: one seed flip changes candidate status (seed-variance; overrides A/B, not D).
        A: route-1 interval-empty (some H defends but no H passes revision in that mode,
           provable by latency law).
        B: no H defends in full combined sweep (dose-bound).
        """
        # First check dissolution
        dissolving = dissolved_by_map.get((cell_idx, mode_name))
        if dissolving is not None:
            return "D"

        # Check V
        if is_v_cell(cell_idx, mode_name, tolerance):
            return "V"

        # Check if any H (exp184) defends the cell
        any_h_defends = any(
            defense_pass_184.get((cell_idx, f"H{h}"), False)
            for h in FIXED_HORIZONS_184
        )
        any_new_defends = any(
            defense_pass_185.get((config_name, cell_idx), False)
            for config_name, _, _, _, _ in EXP185_NEW_CONFIGS
        )

        if not any_h_defends and not any_new_defends:
            return "B"

        # Some H defends but none passes both bars in this mode => route-1 (A)
        return "A"

    # Classify all candidate cells per mode
    classification: dict = {}  # (cell_idx, mode_name) -> class letter
    for mode_name, tolerance in REVISION_MODES:
        for cell_idx in candidate_cells_184[mode_name]:
            cls = classify_cell(cell_idx, mode_name, tolerance)
            classification[(cell_idx, mode_name)] = cls

    # ---- 5d. P1: CALM2600 coverage law ----
    print("5d. P1: CALM2600 coverage law check...")
    # P1: defense_pass(cell, H) iff H >= K*L+(K-1)*G-75 under CALM2600
    # Check for all 3 CALM2600 arms x 80 cells
    calm2600_configs = [cn for cn, _, rc, _, _ in EXP185_NEW_CONFIGS if rc == CALM2600_SNAPS]
    p1_outcomes: list[dict] = []  # per (config, cell, predicted, actual)
    p1_misfits: list[dict] = []

    for config_name, theta_val, rc_snaps, h_suffix, h_val in EXP185_NEW_CONFIGS:
        if rc_snaps != CALM2600_SNAPS:
            continue
        H = h_val
        for ci, cell in enumerate(CELLS):
            L, K, G = cell["L"], cell["K"], cell["G"]
            cell_idx = cell["idx"]
            # Predicted: defense_pass iff H >= K*L + (K-1)*G - 75
            predicted_pass = H >= K * L + (K - 1) * G - 75
            actual_pass = defense_pass_185.get((config_name, cell_idx), False)
            match = (predicted_pass == actual_pass)
            outcome = {
                "config": config_name,
                "cell_idx": cell_idx,
                "L": L, "K": K, "G": G, "H": H,
                "predicted": predicted_pass,
                "actual": actual_pass,
                "match": match,
            }
            p1_outcomes.append(outcome)
            if not match:
                p1_misfits.append(outcome)

    p1_total = len(p1_outcomes)
    p1_n_misfits = len(p1_misfits)
    p1_misfit_frac = p1_n_misfits / p1_total if p1_total > 0 else 0.0
    p1_f1_fired = p1_misfit_frac > 0.20
    print(f"  P1: {p1_n_misfits}/{p1_total} misfits = {p1_misfit_frac:.3f} "
          f"({'F1 FIRED' if p1_f1_fired else 'PASS'})")
    print()

    # ---- 5e. P2: H9000 Phase-R latency ----
    print("5e. P2: H9000 Phase-R latency vs 9000+baseline +-50...")
    p2_deviations: list[dict] = []
    p2_f2_fired = False
    for seed in SEEDS:
        bl_lat = r_latencies_185.get(("baseline", seed))
        h9000_lat = r_latencies_185.get(("H9000", seed))
        if bl_lat is None or h9000_lat is None:
            print(f"  s{seed}: baseline={bl_lat}, H9000={h9000_lat} (None — cannot check)")
            p2_deviations.append({
                "seed": seed, "baseline": bl_lat, "h9000": h9000_lat,
                "predicted": None, "deviation": None, "ok": False
            })
            continue
        predicted = 9000 + bl_lat
        deviation = abs(h9000_lat - predicted)
        ok = deviation <= 50
        if not ok:
            p2_f2_fired = True
        p2_deviations.append({
            "seed": seed, "baseline": bl_lat, "h9000": h9000_lat,
            "predicted": predicted, "deviation": deviation, "ok": ok
        })
        print(f"  s{seed}: baseline={bl_lat}, H9000={h9000_lat}, predicted={predicted}, "
              f"deviation={deviation} {'OK' if ok else 'F2 FIRED'}")
    # Check revision fails all modes (expected from latency law)
    for mode_name, tolerance in REVISION_MODES:
        rev_ok = rev_pass_185.get(("H9000", mode_name), False)
        print(f"  H9000 revision_pass({mode_name}, tol={tolerance}): {rev_ok} "
              f"(expected False by law)")
    print()

    # ---- 5f. P3: no cell flips exists_H_both true->false ----
    print("5f. P3: checking no cell flips exists_H_both true->false...")
    p3_regressions: list[dict] = []
    for mode_name, _ in REVISION_MODES:
        for ci, cell in enumerate(CELLS):
            cell_idx = cell["idx"]
            was_true = exists_h_both_184.get((cell_idx, mode_name), False)
            is_now = exists_h_both_combined.get((cell_idx, mode_name), False)
            if was_true and not is_now:
                p3_regressions.append({
                    "cell_idx": cell_idx,
                    "L": cell["L"], "K": cell["K"], "G": cell["G"],
                    "mode": mode_name,
                })
    n_regressions = len(p3_regressions)
    p3_f3_fired = n_regressions > 5
    print(f"  P3: {n_regressions} regressions (F3 threshold: >5) "
          f"({'F3 FIRED' if p3_f3_fired else 'PASS'})")
    print()

    # ---- 5g. Build final classification tables ----
    print("5g. Building final classification tables...")

    # Class counts per mode
    class_counts: dict = {}  # mode_name -> {A, B, D, V}
    for mode_name, _ in REVISION_MODES:
        counts = {"A": 0, "B": 0, "D": 0, "V": 0}
        for cell_idx in candidate_cells_184[mode_name]:
            cls = classification.get((cell_idx, mode_name), "?")
            if cls in counts:
                counts[cls] += 1
        class_counts[mode_name] = counts

    # Dissolution map: config -> n cells dissolved (across all modes, deduplicated by cell)
    dissolved_by_config: dict = {}  # config_name -> set of (cell_idx, mode_name)
    for (cell_idx, mode_name), config_name in dissolved_by_map.items():
        if config_name is not None:
            dissolved_by_config.setdefault(config_name, set()).add((cell_idx, mode_name))

    # Surviving candidates (A and B classes) per mode
    surviving: dict = {}  # mode_name -> list of (L, K, G, class)
    for mode_name, _ in REVISION_MODES:
        surv = []
        for cell_idx in candidate_cells_184[mode_name]:
            cls = classification.get((cell_idx, mode_name), "?")
            if cls in ("A", "B"):
                cell = CELLS[cell_idx]
                surv.append({"L": cell["L"], "K": cell["K"], "G": cell["G"],
                             "cell_idx": cell_idx, "class": cls})
        surviving[mode_name] = surv

    # Check F-cls: any candidate cell unclassified?
    f_cls_failed = any(
        classification.get((cell_idx, mode_name), "?") not in ("A", "B", "D", "V")
        for mode_name, _ in REVISION_MODES
        for cell_idx in candidate_cells_184[mode_name]
    )

    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0
    print(f"Total runtime: {runtime_min:.1f} min")
    print()

    # ====================================================================
    # Write analysis row to JSONL
    # ====================================================================
    analysis_row = {
        "exp": 185,
        "kind": "analysis",
        "sanity_ok": sanity_ok,
        "sanity_counts": {"normal": n_normal, "tight": n_tight, "loose": n_loose},
        "class_counts": {mode: counts for mode, counts in class_counts.items()},
        "dissolution_map": {
            f"{cell_idx}_{mode}": config
            for (cell_idx, mode), config in dissolved_by_map.items()
            if config is not None
        },
        "p1_misfit_frac": p1_misfit_frac,
        "p1_n_misfits": p1_n_misfits,
        "p1_total": p1_total,
        "p1_f1_fired": p1_f1_fired,
        "p2_deviations": p2_deviations,
        "p2_f2_fired": p2_f2_fired,
        "p3_regressions": len(p3_regressions),
        "p3_f3_fired": p3_f3_fired,
        "f_cls_failed": f_cls_failed,
        "surviving": {
            mode: [{"L": s["L"], "K": s["K"], "G": s["G"],
                    "cell_idx": s["cell_idx"], "class": s["class"]}
                   for s in cells]
            for mode, cells in surviving.items()
        },
        "runtime_min": runtime_min,
    }
    with open(out_rows_path, "a") as fh:
        fh.write(json.dumps(to_plain(analysis_row)) + "\n")

    # ====================================================================
    # Write text report
    # ====================================================================
    with open(out_txt_path, "w") as f:
        def p(*args, **kwargs):
            print(*args, **kwargs, file=f)
            print(*args, **kwargs)

        p("=" * 80)
        p("EXP 185 — N4 CRACK CLASSIFICATION (EXPLORATORY-classification)")
        p("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit 8cc5cdb)")
        p(f"Runtime: {runtime_min:.1f} min | Seeds: {SEEDS} | 6 new configs x 80 cells x 3 seeds")
        p("=" * 80)
        p()

        p("EQUIVALENCE GATE (L15)")
        p("-" * 60)
        p(gate_detail)
        p()

        p("SANITY CHECK — EXP 184 CANDIDATE COUNTS (reproduced from committed rows)")
        p("-" * 60)
        p(f"  normal: {n_normal} (expected 36) {'OK' if n_normal == 36 else 'FAIL'}")
        p(f"  tight:  {n_tight} (expected 57) {'OK' if n_tight == 57 else 'FAIL'}")
        p(f"  loose:  {n_loose} (expected 32) {'OK' if n_loose == 32 else 'FAIL'}")
        p(f"  Overall: {'SANITY PASS' if sanity_ok else 'SANITY FAIL'}")
        p()

        p("PHASE-R LATENCY TABLE — NEW CONFIGS (7 arms x 3 seeds)")
        p("-" * 60)
        bl_lats_185 = {seed: r_latencies_185.get(("baseline", seed)) for seed in SEEDS}
        hdr = f"  {'arm':22s} " + " ".join(f"s{s:3d}" for s in SEEDS)
        hdr += "  [baseline: " + "/".join(str(bl_lats_185[s]) for s in SEEDS) + "]"
        p(hdr)
        for arm_label, _, theta_val, rc_snaps in r_arms_185:
            lats_str = " ".join(
                f"{str(r_latencies_185.get((arm_label, s), 'N/A')):>8}"
                for s in SEEDS
            )
            p(f"  {arm_label:22s} {lats_str}")
        p()

        p("REVISION PASS TABLE — NEW CONFIGS (arm x mode)")
        p("-" * 60)
        p(f"  {'arm':22s} " + " ".join(f"{m:8s}" for m, _ in REVISION_MODES))
        for config_name, theta_val, rc_snaps, h_suffix, h_val in EXP185_NEW_CONFIGS:
            passes = " ".join(
                f"{'PASS' if rev_pass_185.get((config_name, m), False) else 'fail':8s}"
                for m, _ in REVISION_MODES
            )
            p(f"  {config_name:22s} {passes}")
        p()

        p("P1 — CALM2600 COVERAGE LAW CHECK")
        p("  Predicted: defense_pass(cell, H) iff H >= K*L + (K-1)*G - 75")
        p("-" * 60)
        p(f"  Total CALM2600 cell x H outcomes: {p1_total}")
        p(f"  Misfits: {p1_n_misfits} ({p1_misfit_frac:.3f})")
        p(f"  F1 threshold: >20% misfits")
        p(f"  P1 VERDICT: {'F1 FIRED — model wrong' if p1_f1_fired else 'PASS — spanning model holds'}")
        if p1_misfits:
            p(f"  Misfit cells (up to 20):")
            for m in p1_misfits[:20]:
                p(f"    config={m['config']} cell_idx={m['cell_idx']} L={m['L']} K={m['K']} "
                  f"G={m['G']} H={m['H']}: predicted={m['predicted']} actual={m['actual']}")
        p()

        p("P2 — H9000 LATENCY LAW CHECK")
        p("  Predicted: H9000 Phase-R latency = 9000 + baseline +- 50; fails all modes")
        p("-" * 60)
        for d in p2_deviations:
            if d["deviation"] is None:
                p(f"  s{d['seed']}: baseline={d['baseline']} H9000={d['h9000']} (cannot check)")
            else:
                p(f"  s{d['seed']}: baseline={d['baseline']} H9000={d['h9000']} "
                  f"predicted={d['predicted']} deviation={d['deviation']} "
                  f"{'OK' if d['ok'] else 'F2 FIRED'}")
        for mode_name, tolerance in REVISION_MODES:
            rev_ok = rev_pass_185.get(("H9000", mode_name), False)
            p(f"  H9000 revision_pass({mode_name}, tol={tolerance}): {rev_ok} "
              f"(expected False by law)")
        p(f"  P2 VERDICT: {'F2 FIRED — deviation >50' if p2_f2_fired else 'PASS — law holds'}")
        p()

        p("P3 — NO CONFIG REGRESSION CHECK")
        p("  F3 threshold: >5 cells flip exists_H_both true->false")
        p("-" * 60)
        p(f"  Regressions: {n_regressions} {'(F3 FIRED)' if p3_f3_fired else '(PASS)'}")
        if p3_regressions:
            for r in p3_regressions:
                p(f"    cell_idx={r['cell_idx']} L={r['L']} K={r['K']} G={r['G']} mode={r['mode']}")
        else:
            p("  (none — structurally impossible as new arms can only add options)")
        p()

        p("CLASSIFICATION TABLE — PER MODE, PER CANDIDATE CELL")
        p("  V rule: V assigned (overriding A/B, not D) when any revision-passing arm has")
        p("          defense count == 1 in the combined sweep (one flip changes status)")
        p("  A rule: some H defends; none pass both bars (interval-empty, provable by latency law)")
        p("  B rule: no H defends in full combined sweep (dose-bound)")
        p("  D rule: some new simple-config arm passes BOTH bars")
        p("-" * 60)
        for mode_name, tolerance in REVISION_MODES:
            counts = class_counts[mode_name]
            p(f"\nMode={mode_name} (tolerance={tolerance}):")
            p(f"  Class counts: A={counts['A']} B={counts['B']} D={counts['D']} V={counts['V']} "
              f"total={sum(counts.values())}")
            p(f"  {'cell_idx':>8} {'L':>5} {'K':>3} {'G':>5} {'class':>6} {'dissolved_by':>20}")
            for cell_idx in sorted(candidate_cells_184[mode_name]):
                cell = CELLS[cell_idx]
                cls = classification.get((cell_idx, mode_name), "?")
                dby = dissolved_by_map.get((cell_idx, mode_name)) or ""
                p(f"  {cell_idx:>8} {cell['L']:>5} {cell['K']:>3} {cell['G']:>5} {cls:>6} {dby:>20}")
        p()

        p("DISSOLUTION MAP (config -> n cell x mode pairs dissolved)")
        p("-" * 60)
        for config_name, _, _, _, _ in EXP185_NEW_CONFIGS:
            n_dissolved = len(dissolved_by_config.get(config_name, set()))
            p(f"  {config_name:22s}: {n_dissolved} cell x mode pairs dissolved")
        p()

        p("SURVIVING CRACK CANDIDATES (A and B classes only)")
        p("-" * 60)
        for mode_name, _ in REVISION_MODES:
            surv = surviving[mode_name]
            p(f"\nMode={mode_name}: {len(surv)} surviving (A+B)")
            for s in surv:
                p(f"  cell_idx={s['cell_idx']} L={s['L']} K={s['K']} G={s['G']} class={s['class']}")
        p()

        p("F-CLS CHECK (any candidate unclassified?)")
        p("-" * 60)
        p(f"  F-cls: {'FIRED — some cell unclassified!' if f_cls_failed else 'PASS — all cells classified'}")
        p()

        p("=" * 80)
        p("SUMMARY")
        p("=" * 80)
        p(f"  Gate: PASS")
        p(f"  Sanity: {'PASS' if sanity_ok else 'FAIL'} ({n_normal}/36 normal, {n_tight}/57 tight, {n_loose}/32 loose)")
        for mode_name, _ in REVISION_MODES:
            counts = class_counts[mode_name]
            n_surv = counts["A"] + counts["B"]
            p(f"  mode={mode_name}: A={counts['A']} B={counts['B']} D={counts['D']} V={counts['V']} "
              f"| surviving={n_surv}")
        p(f"  P1 CALM2600 spanning law: {'F1 FIRED' if p1_f1_fired else 'PASS'} "
          f"({p1_misfit_frac:.3f} misfit fraction)")
        p(f"  P2 H9000 latency law: {'F2 FIRED' if p2_f2_fired else 'PASS'}")
        p(f"  P3 no regression: {'F3 FIRED' if p3_f3_fired else 'PASS'} ({n_regressions} regressions)")
        p(f"  F-cls: {'FIRED' if f_cls_failed else 'PASS'}")
        p(f"  Runtime: {runtime_min:.1f} min")
        p()
        p("NOTE: EXPLORATORY-classification. No crack CLAIM.")
        p("Surviving A/B cells are candidates pending rung-4 fresh-seed confirmation.")
        p("=" * 80)

    print(f"Text written to {out_txt_path}")
    print(f"Rows written to {out_rows_path}")
    print()
    print(f"Total runtime: {runtime_min:.1f} min")


if __name__ == "__main__":
    main()
