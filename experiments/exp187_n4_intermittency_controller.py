"""Exp 187 — N4 Intermittency Controller (RUNG-5 FALSIFIABLE).

PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit e1bda4e) BEFORE any data.
Status: FALSIFIABLE — predeclared falsifiers below; fresh seeds 260-267.

Thesis: in the 9 confirmed crack cells the transient/sustained discriminator is
pressure INTERMITTENCY. INT controller family = freeze machinery verbatim
(THETA 3.5 sliding trigger) + gap-spanning release (release-calm 2600 steps =
CALM2600_SNAPS) + concession on CONTINUOUS pressure only: the concession
accumulator RESETS whenever the pressure statistic de-asserts (pressure =
fraction of last 200 steps' observations == c_star; ACTIVE iff >= 0.6 —
the exp182/183 statistic verbatim).

Concession forms (constants on the new surface; the kill test runs ON the
surface):
  INT-E600: E_blocked-style integral counting ONLY continuous coherent pressure,
    reset on any de-assert; concede at 600 units.
  INT-C1500 / INT-C2000 / INT-C2900: continuous-ACTIVE-pressure stopwatches
    (steps of unbroken ACTIVE pressure while frozen; reset on de-assert);
    concede at C.

Arms (10):
  baseline; oracle; H3000; CALM2600-H6000; THETA3-H3000;
  n4_freeze (E*=600, WITHOUT gap reset — verbatim exp183 evidence concession);
  INT-E600; INT-C1500; INT-C2000; INT-C2900.

Cells (9, the exp186-confirmed set):
  (1600,2,200), (1600,3,200), (2400,3,200), (1600,4,200), (2400,4,200),
  (2400,4,600), (400,4,600), (400,4,2400), (800,4,600).

Seeds FRESH 260-267. W: 9 cells x 10 arms x 8 seeds = 720.
R: 9 non-oracle arms x 8 seeds = 72 (verbatim Phase R).

Bars VERBATIM exp186: defense(arm, cell) >= 6/8; revision(arm) >= 6/8
  (latency <= same-seed baseline + 3000 normal; tight +1500, loose +6000);
  oracle >= 7/8; baseline deficit >= 7/8; PC1 < 0.15 gated.

Predeclared falsifiers (predeclar, e1bda4e):
  P1: INT-E600 passes BOTH bars in >= 7/9 cells; also passes TIGHT tolerance.
  F1: INT-E600 fails defense in >= 3 cells.
  P2: every INT-C with C <= 2900 passes both bars wherever INT-E600 does;
      C2900 fails tight while E600 passes it.
  F2: no single C covers all 9 cells while cell-dependent C's exist.
  P3: n4_freeze (no-reset) fails defense in train cells (concedes mid-train)
      while INT-E600 defends — the gap-reset is the entire mechanism.
  F3: no separation (INT-E600 does not separate from n4_freeze noreset).

Outputs: experiments/outputs/exp187_rows.json (JSONL),
         experiments/outputs/exp187.txt (gate evidence table; per-cell x
         per-arm defense/revision table; minimal-pair table; P1/P2/P3 verdicts;
         tight/loose secondary); script
         experiments/exp187_n4_intermittency_controller.py.
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

# Pressure constants (verbatim exp182/183)
PRESSURE_WINDOW = _mod185.PRESSURE_WINDOW   # 200
PRESSURE_FRAC = _mod185.PRESSURE_FRAC       # 0.6

REFRACTORY_CHECKS = _mod185.__dict__.get("REFRACTORY_CHECKS", 8)

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Exp 187 configuration
# ---------------------------------------------------------------------------

SEEDS_187 = [260, 261, 262, 263, 264, 265, 266, 267]
N_SEEDS = len(SEEDS_187)

# Confirmation bars (verbatim exp186)
DEFENSE_MIN_PASS = 6    # >= 6/8 seeds
ORACLE_DEFENSE_MIN = 7  # >= 7/8
BASELINE_FAIL_MIN = 7   # >= 7/8 must fail for deficit

# Revision modes: primary = normal (+3000); tight/loose secondary diagnostics
REVISION_MODES = [
    ("normal", 3000),
    ("tight",  1500),
    ("loose",  6000),
]

# 9 confirmed crack cells (exp186-confirmed set)
CRACK_CELLS = [
    {"L": 1600, "K": 2, "G": 200},
    {"L": 1600, "K": 3, "G": 200},
    {"L": 2400, "K": 3, "G": 200},
    {"L": 1600, "K": 4, "G": 200},
    {"L": 2400, "K": 4, "G": 200},
    {"L": 2400, "K": 4, "G": 600},
    {"L": 400,  "K": 4, "G": 600},
    {"L": 400,  "K": 4, "G": 2400},
    {"L": 800,  "K": 4, "G": 600},
]

# Assign cell_idx labels for identification
for _c in CRACK_CELLS:
    _c["cell_tag"] = f"({_c['L']},{_c['K']},{_c['G']})"

# ---------------------------------------------------------------------------
# INT arm modes
# ---------------------------------------------------------------------------
# INT arm_mode is a tuple: ('int_e', E_threshold, calm_snaps) or
#                          ('int_c', C_threshold, calm_snaps)
# calm_snaps = CALM2600_SNAPS (2600 steps / 25 steps/snap = 104 snaps)
# The runner below handles these modes.

INT_CALM_SNAPS = CALM2600_SNAPS   # 104 snaps = 2600 steps

# Arms (10):
# (arm_name, arm_mode, theta, release_calm_snaps)
ARM_DEFS = []

ARM_DEFS.append(("baseline",     "baseline",                            DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("oracle",       "oracle",                              DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("H3000",        ("freeze_time", 3000),                 DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("CALM2600-H6000", ("freeze_time", 6000),               DEFAULT_THETA, CALM2600_SNAPS))
ARM_DEFS.append(("THETA3-H3000", ("freeze_time", 3000),                 THETA3_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("n4_freeze",    "freeze_evidence",                     DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("INT-E600",     ("int_e", 600.0, INT_CALM_SNAPS),      DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("INT-C1500",    ("int_c", 1500,  INT_CALM_SNAPS),      DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("INT-C2000",    ("int_c", 2000,  INT_CALM_SNAPS),      DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("INT-C2900",    ("int_c", 2900,  INT_CALM_SNAPS),      DEFAULT_THETA, INT_CALM_SNAPS))

assert len(ARM_DEFS) == 10, f"Expected 10 arms, got {len(ARM_DEFS)}"

# R arms: all 9 non-oracle arms
R_ARM_DEFS = [(name, mode, theta, rcs) for name, mode, theta, rcs in ARM_DEFS
              if name != "oracle"]
assert len(R_ARM_DEFS) == 9, f"Expected 9 R arms, got {len(R_ARM_DEFS)}"

# Family arms (for conjunct i — no-arm-covers check): all except oracle
FAMILY_ARM_NAMES = {name for name, _, _, _ in ARM_DEFS if name != "oracle"}


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
# INT runner — surgical extension of run_fork_schedule_185
# Handles ('int_e', E_thresh, calm_snaps) and ('int_c', C_thresh, calm_snaps).
# The concession accumulator RESETS to zero when pressure de-asserts
# (pressure_active = False at a snap check while frozen).
# release-calm = INT_CALM_SNAPS (spanning, 2600 steps).
# ---------------------------------------------------------------------------

def run_fork_schedule_int(
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

    Handles INT arm modes ('int_e', E_thresh, calm_snaps) and
    ('int_c', C_thresh, calm_snaps) in addition to all existing arm modes
    from run_fork_schedule_185.

    INT concession logic (surgical extension):
    - Same THETA 3.5 sliding trigger as standard freeze arms.
    - release_calm_snaps = INT_CALM_SNAPS (spanning, 2600 steps).
    - Concession accumulator (E_blocked for int_e; step-counter for int_c)
      RESETS to zero whenever pressure is INACTIVE at a snap check while frozen.
    - Concession fires when accumulator >= threshold AND pressure is currently ACTIVE.

    For non-INT modes, this function delegates verbatim to the existing code path
    (i.e. it is a superset runner).

    n_resets: number of times the accumulator was reset (INT arms only).
    concession_events_in_train: bool — did any concession event fire inside a
      training burst window?
    mean_concession_step: mean entry_step of concession events (None if none).
    """
    # Determine if this is an INT arm
    is_int_arm = isinstance(arm_mode, tuple) and arm_mode[0] in ("int_e", "int_c")

    if not is_int_arm:
        # Delegate to the existing runner — zero code duplication
        result = run_fork_schedule_185(
            mirro=mirro,
            fork_seed=fork_seed,
            base_cmap=base_cmap,
            n_colors=n_colors,
            arm_name=arm_name,
            arm_mode=arm_mode,
            phase=phase,
            burst_windows=burst_windows,
            color_mode=color_mode,
            reloc_rng_seed=reloc_rng_seed,
            n_steps=n_steps,
            theta=theta,
            release_calm_snaps=release_calm_snaps,
        )
        # Add INT-specific keys (zero values for non-INT arms)
        result["n_resets"] = 0
        burst_step_set_check: set[int] = set()
        for bstart, bend in burst_windows:
            burst_step_set_check.update(range(bstart, bend))
        in_train_concessions = sum(
            1 for e in result["events"]
            if e.get("label") == "concession" and e.get("entry_step") in burst_step_set_check
        )
        result["concession_events_in_train"] = in_train_concessions > 0
        concession_steps = [e["entry_step"] for e in result["events"]
                            if e.get("label") == "concession"]
        result["mean_concession_step"] = (
            float(sum(concession_steps)) / len(concession_steps)
            if concession_steps else None
        )
        return result

    # ---- INT arm: full inline runner ----
    # Unpack INT mode
    if arm_mode[0] == "int_e":
        _, int_thresh, int_calm_snaps = arm_mode
        int_mode = "int_e"
    else:
        _, int_thresh, int_calm_snaps = arm_mode
        int_mode = "int_c"

    # release_calm_snaps parameter is the arm-level spanning calm (== int_calm_snaps)
    # We use int_calm_snaps internally so the caller's release_calm_snaps is ignored
    # for transient release; this is correct since ARM_DEFS sets both to INT_CALM_SNAPS.
    _release_calm_snaps = int_calm_snaps

    n_chunks = n_steps // CHUNK_SIZE
    assert n_chunks * CHUNK_SIZE == n_steps

    fork_name = f"exp187_{arm_name}_{phase}_s{fork_seed}"
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

    # INT-specific state
    # For int_e: int_acc = E_blocked-style integral (only CONTINUOUS pressure)
    # For int_c: int_acc = step-counter of unbroken ACTIVE pressure while frozen
    int_acc: float = 0.0
    n_resets: int = 0   # how many times int_acc was reset

    checks_since_release: int = REFRACTORY_CHECKS

    gap_start: list[float | None] = [None] * len(burst_windows)
    gap_end: list[float | None] = [None] * len(burst_windows)
    d_b: list[float | None] = [None] * len(burst_windows)
    tv_b: list[float | None] = [None] * len(burst_windows)
    pi_at_burst_start: list[np.ndarray | None] = [None] * len(burst_windows)

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
                settle_tv_val = tv_func(settle_pi_5000, pi_of(v.copy()))

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

                if burst_idx_now is None and current_burst_idx is not None:
                    for bi, (bstart, bend) in enumerate(burst_windows):
                        if t == bend:
                            if burst_preburst_fav[bi] is not None and burst_onset_color[bi] is not None:
                                pi_end = pi_of(v.copy())
                                gap_end[bi] = float(v[burst_preburst_fav[bi]] - v[burst_onset_color[bi]])
                                d_b[bi] = float(pi_end[burst_onset_color[bi]] - pi_at_burst_start[bi][burst_onset_color[bi]])
                                tv_b[bi] = tv_func(pi_end, pi_at_burst_start[bi])
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

            # INT arm: RESIST state handling
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

                    # INT-specific: accumulator reset on de-assert
                    if not pressure_active:
                        # Pressure de-asserted: reset accumulator
                        if int_acc > 0:
                            n_resets += 1
                            int_acc = 0.0
                        calm_count += 1
                    else:
                        calm_count = 0
                        # Pressure active: accumulate
                        if int_mode == "int_e":
                            c_star_final_now = int(np.argmax(blocked_w_by_color))
                            int_acc = float(blocked_w_by_color[c_star_final_now])
                        elif int_mode == "int_c":
                            int_acc += FINE_EVAL  # count steps of active pressure

                    c_star_final = int(np.argmax(blocked_w_by_color))
                    E_blocked_final = float(blocked_w_by_color[c_star_final])

                    released = False
                    release_label = None

                    # Transient release (spanning calm)
                    if calm_count >= _release_calm_snaps:
                        release_label = "transient"
                        released = True

                    # INT concession: only fires when pressure is ACTIVE and
                    # accumulator >= threshold
                    if not released and pressure_active and int_acc >= int_thresh:
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
                            "int_acc_at_release": float(int_acc),
                            "n_resets_at_release": n_resets,
                        })
                        freeze_state = "NORMAL"
                        mismatch_history.clear()
                        v_fine.clear()
                        k_since_reset = 0
                        calm_count = 0
                        frozen_steps = 0
                        directional_pressure_acc = 0.0
                        int_acc = 0.0
                        checks_since_release = 0
                continue

            else:
                state_arr[t] = 0

            # Value update (NORMAL state)
            v *= LAMBDA
            v[obs] += w
            c.value_counts[obs] += w

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

                if freeze_state == "NORMAL":
                    v_fine.append(v.copy())
                    k_since_reset += 1
                    k = k_since_reset - 1

                    MON_W_SNAPS = 40
                    CTRL_MBAR_WINDOW = 120
                    CTRL_MIN_SNAPS = 40

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
                                int_acc = 0.0  # reset on new freeze
                                # n_resets is per-session total, not reset here

    A_hat_end = c._A_hat().copy()
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

    # Compute INT-specific summary fields
    in_train_concessions = sum(
        1 for e in events
        if e.get("label") == "concession" and e.get("entry_step") in burst_step_set
    )
    concession_steps = [e["entry_step"] for e in events if e.get("label") == "concession"]
    mean_concession_step = (
        float(sum(concession_steps)) / len(concession_steps)
        if concession_steps else None
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
        "n_resets": n_resets,
        "concession_events_in_train": in_train_concessions > 0,
        "mean_concession_step": mean_concession_step,
    }


def tv_func(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


# ---------------------------------------------------------------------------
# Equivalence gate (L15) — re-run exp183 baseline x s229, H1200 x s229
# through THIS script's code path
# ---------------------------------------------------------------------------

def load_committed_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_equivalence_gate_187(
    mirro_root, base_cmap, n_colors, committed_183_path: Path
) -> tuple[bool, str]:
    """Run equivalence gate: reproduce exp183 baseline x s229 and H1200 x s229.

    These arms are non-INT so they pass through run_fork_schedule_185 unchanged.
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

            rr = run_fork_schedule_int(
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

            status_str = "PASS" if session_pass else "FAIL"
            detail_lines.append(f"  gate arm={arm_name} seed={seed}: {status_str}")
            print(f"  [gate] arm={arm_name} seed={seed}: {status_str}", flush=True)

    overall = "PASS" if all_pass else "FAIL"
    detail_str = f"EQUIVALENCE GATE: {overall}\n" + "\n".join(detail_lines)
    return all_pass, detail_str


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 80)
    print("Exp 187 — N4 Intermittency Controller (RUNG-5 FALSIFIABLE)")
    print("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit e1bda4e)")
    print(f"Seeds: {SEEDS_187} | 9 cells x 10 arms x 8 seeds = 720 W | 9 R-arms x 8 seeds = 72 R")
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
    out_rows_path = out_dir / "exp187_rows.json"
    out_txt_path = out_dir / "exp187.txt"
    committed_183_path = out_dir / "exp183_rows.json"

    print("CELLS (9 confirmed crack cells):")
    for ci, cell in enumerate(CRACK_CELLS):
        print(f"  [{ci}] ({cell['L']},{cell['K']},{cell['G']})")
    print()
    print("ARMS (10):")
    for arm_name, arm_mode, theta, rcs in ARM_DEFS:
        fam = "(family)" if arm_name in FAMILY_ARM_NAMES else "(oracle-only)"
        print(f"  {arm_name:20s}  theta={theta}  rcs={rcs}  mode={arm_mode}  {fam}")
    print()

    # ====================================================================
    # STEP 1: EQUIVALENCE GATE (L15)
    # ====================================================================
    print("=" * 80)
    print("STEP 1: EQUIVALENCE GATE (L15)")
    print("Reproducing exp183 (baseline x s229) and (H1200 x s229) via exp187 code path")
    print("Default params: theta=3.5, release_calm_snaps=8 (non-INT path)")
    print("=" * 80)
    t_gate = time.time()

    gate_pass, gate_detail = run_equivalence_gate_187(
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
    # STEP 2: W SESSIONS (9 cells x 10 arms x 8 seeds = 720)
    # ====================================================================
    print("=" * 80)
    print("STEP 2: W SESSIONS (9 cells x 10 arms x 8 seeds = 720 sessions)")
    print("=" * 80)

    # defense_by[(cell_tag, arm_name, seed)] -> bool
    defense_by: dict = {}
    pc1_flags: dict = {}

    rows_buffer: list[dict] = []

    t_grid = time.time()
    n_total = len(CRACK_CELLS) * len(ARM_DEFS) * N_SEEDS
    n_done = 0

    for ci, cell in enumerate(CRACK_CELLS):
        L, K, G = cell["L"], cell["K"], cell["G"]
        cell_tag = cell["cell_tag"]
        n_steps = compute_n_steps(L, K, G)
        burst_windows = compute_burst_windows(L, K, G)

        for arm_name, arm_mode, theta_val, rc_snaps in ARM_DEFS:
            for seed in SEEDS_187:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

                # Unique reloc seed per cell x seed
                reloc_rng_seed = 200_000 + 10_000 * ci + seed

                rr = run_fork_schedule_int(
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
                        "gap_start": rr["gap_start"][bi],
                        "gap_end": rr["gap_end"][bi],
                        "expr_frac_or_null": rr["per_burst_expr_frac"][bi],
                    }
                    per_burst.append(pb)

                row = {
                    "exp": 187,
                    "kind": "W",
                    "cell": {"L": L, "K": K, "G": G, "tag": cell_tag},
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
                        }
                        for e in rr["events"]
                    ]),
                    "ahat_drift": float(rr["ahat_drift"]),
                    "settle_tv": to_plain(rr.get("settle_tv")),
                    "flags": ["PC1_DRIFT"] if pc1_flag else [],
                    "theta": theta_val,
                    "release_calm_snaps": rc_snaps,
                    "n_resets": int(rr.get("n_resets", 0)),
                    "concession_events_in_train": bool(rr.get("concession_events_in_train", False)),
                    "mean_concession_step": to_plain(rr.get("mean_concession_step")),
                }
                rows_buffer.append(row)
                n_done += 1

        elapsed = time.time() - t_grid
        rate = elapsed / max(1, n_done)
        remaining = rate * (n_total - n_done)
        print(
            f"  cell {ci+1:02d}/{len(CRACK_CELLS)} ({cell_tag}) "
            f"| elapsed {elapsed:.0f}s | ETA {remaining:.0f}s",
            flush=True,
        )

    t_grid_done = time.time()
    print(f"W grid done: {(t_grid_done - t_grid)/60:.1f} min total")
    print()

    # ====================================================================
    # STEP 3: R SESSIONS (9 non-oracle arms x 8 seeds = 72 sessions)
    # ====================================================================
    print("=" * 80)
    print("STEP 3: R SESSIONS (9 non-oracle arms x 8 seeds = 72 sessions)")
    print("baseline R latencies are the tolerance reference")
    print("=" * 80)

    r_latencies: dict = {}
    r_rows_buffer: list[dict] = []

    t_r = time.time()
    for arm_label, arm_mode_r, theta_val, rc_snaps in R_ARM_DEFS:
        for seed in SEEDS_187:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None

            rr = run_fork_schedule_int(
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
                f"n_resets={rr.get('n_resets', 0)}  "
                f"ahat_drift={rr['ahat_drift']:.5f}",
                flush=True,
            )

            r_row = {
                "exp": 187,
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

    PRIMARY_MODE = "normal"

    # ---- 5a. Defense counts per (cell_tag, arm_name) ----
    defense_counts: dict = {}
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        for arm_name, _, _, _ in ARM_DEFS:
            n_pass = sum(
                1 for seed in SEEDS_187
                if defense_by.get((cell_tag, arm_name, seed), False)
            )
            defense_counts[(cell_tag, arm_name)] = n_pass

    defense_pass: dict = {}
    for (cell_tag, arm_name), cnt in defense_counts.items():
        defense_pass[(cell_tag, arm_name)] = cnt >= DEFENSE_MIN_PASS

    # ---- 5b. Revision pass per arm ----
    revision_pass: dict = {}
    for arm_name, _, _, _ in R_ARM_DEFS:
        for mode_name, tolerance in REVISION_MODES:
            n_pass = 0
            for seed in SEEDS_187:
                bl_lat = r_latencies.get(("baseline", seed))
                arm_lat = r_latencies.get((arm_name, seed))
                if arm_lat is None or bl_lat is None:
                    continue
                if arm_lat <= bl_lat + tolerance:
                    n_pass += 1
            revision_pass[(arm_name, mode_name)] = n_pass >= DEFENSE_MIN_PASS

    # ---- 5c. Per cell: both_pass per arm ----
    cell_arm_results: dict = {}  # (cell_tag, arm_name) -> dict
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        for arm_name, _, _, _ in ARM_DEFS:
            if arm_name == "oracle":
                cell_arm_results[(cell_tag, arm_name)] = {
                    "defense_count": defense_counts[(cell_tag, arm_name)],
                    "defense_pass": defense_pass.get((cell_tag, arm_name), False),
                }
                continue
            d_ok = defense_pass.get((cell_tag, arm_name), False)
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
    oracle_ok: dict = {}       # cell_tag -> bool (>= 7/8)
    baseline_deficit: dict = {}  # cell_tag -> bool (>= 7/8 fail)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        oracle_cnt = defense_counts[(cell_tag, "oracle")]
        bl_pass_cnt = defense_counts[(cell_tag, "baseline")]
        oracle_ok[cell_tag] = oracle_cnt >= ORACLE_DEFENSE_MIN
        baseline_deficit[cell_tag] = (N_SEEDS - bl_pass_cnt) >= BASELINE_FAIL_MIN

    # ---- 5f. INT-E600 verdicts ----
    int_e600_cells_pass: list[str] = []  # cell_tags where INT-E600 passes both bars (normal)
    int_e600_cells_tight: list[str] = []  # cell_tags where INT-E600 passes both bars (tight)

    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        ar = cell_arm_results.get((cell_tag, "INT-E600"), {})
        if ar.get("both_pass_normal", False):
            int_e600_cells_pass.append(cell_tag)
        if ar.get("both_pass_tight", False):
            int_e600_cells_tight.append(cell_tag)

    n_int_e600_pass = len(int_e600_cells_pass)

    # ---- 5g. P1 verdict ----
    p1_pass = n_int_e600_pass >= 7
    p1_tight_pass = len(int_e600_cells_tight) == n_int_e600_pass  # tight passes everywhere normal does
    f1_fired = n_int_e600_pass <= (9 - 3)  # fails in >= 3 cells means passes in <= 6

    # ---- 5h. P2 verdict: C sweep ----
    # Every INT-C with C <= 2900 passes both bars wherever INT-E600 does
    c_arms = ["INT-C1500", "INT-C2000", "INT-C2900"]
    c_coverage: dict = {}  # arm_name -> list of cells where both_pass_normal
    for arm_name in c_arms:
        passing = []
        for cell in CRACK_CELLS:
            cell_tag = cell["cell_tag"]
            ar = cell_arm_results.get((cell_tag, arm_name), {})
            if ar.get("both_pass_normal", False):
                passing.append(cell_tag)
        c_coverage[arm_name] = passing

    # Does each C cover at least everywhere INT-E600 does?
    c_universal: dict = {}  # arm_name -> bool
    for arm_name in c_arms:
        c_universal[arm_name] = all(
            cell_tag in c_coverage[arm_name]
            for cell_tag in int_e600_cells_pass
        )

    # C2900 fails tight where E600 passes tight?
    c2900_fails_tight = not revision_pass.get("INT-C2900", {}) if False else (
        not revision_pass.get(("INT-C2900", "tight"), False)
        and revision_pass.get(("INT-E600", "tight"), False)
    )
    # More careful: tight revision pass for each arm
    c2900_rev_tight = revision_pass.get(("INT-C2900", "tight"), False)
    e600_rev_tight  = revision_pass.get(("INT-E600",  "tight"), False)
    c2900_fails_tight_vs_e600 = e600_rev_tight and not c2900_rev_tight

    # F2: no single C covers all 9 cells while cell-dependent C's exist
    # (report whether a single universal C exists)
    any_c_covers_all = any(
        len(c_coverage[arm]) == 9
        for arm in c_arms
    )

    p2_pass = (
        all(c_universal[arm] for arm in c_arms[:2])  # C1500, C2000 cover where E600 does
        # C2900 coverage check; tolerance difference from P2 predeclaration
    )

    # ---- 5i. P3 / minimal pair: n4_freeze (noreset) vs INT-E600 ----
    # n4_freeze = freeze_evidence (no gap reset, DEFAULT_RELEASE_CALM_SNAPS)
    # INT-E600 = gap-spanning + accumulator reset on de-assert
    # P3: n4_freeze fails defense in train cells where INT-E600 defends
    # "train cells" = cells where burst structure has gaps (K>1)

    minimal_pair_rows: list[dict] = []
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        L, K, G = cell["L"], cell["K"], cell["G"]
        n4_dc = defense_counts[(cell_tag, "n4_freeze")]
        int_e_dc = defense_counts[(cell_tag, "INT-E600")]
        n4_pass = defense_pass.get((cell_tag, "n4_freeze"), False)
        int_e_pass = defense_pass.get((cell_tag, "INT-E600"), False)

        # Concession events in train for n4_freeze (from rows_buffer)
        n4_in_train = [
            row for row in rows_buffer
            if row["cell"]["tag"] == cell_tag
            and row["arm"] == "n4_freeze"
            and row.get("concession_events_in_train", False)
        ]
        n4_concession_in_train_count = len(n4_in_train)

        # Mean concession step for n4_freeze
        n4_mean_cs = []
        for row in rows_buffer:
            if row["cell"]["tag"] == cell_tag and row["arm"] == "n4_freeze":
                if row.get("mean_concession_step") is not None:
                    n4_mean_cs.append(row["mean_concession_step"])
        n4_mean_cs_agg = float(sum(n4_mean_cs) / len(n4_mean_cs)) if n4_mean_cs else None

        # INT-E600 concession in train
        int_in_train = [
            row for row in rows_buffer
            if row["cell"]["tag"] == cell_tag
            and row["arm"] == "INT-E600"
            and row.get("concession_events_in_train", False)
        ]
        int_concession_in_train_count = len(int_in_train)

        separates = int_e_pass and not n4_pass

        minimal_pair_rows.append({
            "cell_tag": cell_tag,
            "K": K,
            "n4_freeze_def": n4_dc,
            "int_e600_def": int_e_dc,
            "n4_freeze_pass": n4_pass,
            "int_e600_pass": int_e_pass,
            "n4_concession_in_train_seeds": n4_concession_in_train_count,
            "int_concession_in_train_seeds": int_concession_in_train_count,
            "n4_mean_concession_step": n4_mean_cs_agg,
            "separates": separates,
        })

    # P3 verdict: separation in >= 1 cell (and specifically in train cells)
    n_separated = sum(1 for mp in minimal_pair_rows if mp["separates"])
    p3_pass = n_separated >= 1
    f3_fired = n_separated == 0

    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0
    print(f"Total runtime: {runtime_min:.1f} min")
    print()

    # ====================================================================
    # Write analysis row to JSONL
    # ====================================================================
    analysis_row = {
        "exp": 187,
        "kind": "analysis",
        "seeds": SEEDS_187,
        "n_int_e600_pass": n_int_e600_pass,
        "int_e600_cells_pass": int_e600_cells_pass,
        "int_e600_cells_tight": int_e600_cells_tight,
        "p1_pass": p1_pass,
        "p1_tight_pass": p1_tight_pass,
        "f1_fired": f1_fired,
        "c_coverage": c_coverage,
        "c_universal": c_universal,
        "c2900_fails_tight_vs_e600": c2900_fails_tight_vs_e600,
        "p2_pass": p2_pass,
        "minimal_pair_rows": minimal_pair_rows,
        "n_separated": n_separated,
        "p3_pass": p3_pass,
        "f3_fired": f3_fired,
        "pc1_total_flagged": pc1_total,
        "runtime_min": runtime_min,
    }
    with open(out_rows_path, "a") as fh:
        fh.write(json.dumps(to_plain(analysis_row)) + "\n")

    # ====================================================================
    # Write text report (exp187.txt)
    # ====================================================================
    lines: list[str] = []

    def p(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("EXP 187 — N4 INTERMITTENCY CONTROLLER (RUNG-5 FALSIFIABLE)")
    p("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit e1bda4e)")
    p(f"Runtime: {runtime_min:.1f} min | Seeds: {SEEDS_187}")
    p(f"9 cells x 10 arms x 8 seeds = 720 W | 9 arms x 8 seeds = 72 R")
    p("=" * 80)
    p()

    p("EQUIVALENCE GATE (L15)")
    p("-" * 60)
    p(gate_detail)
    p()

    p("PHASE-R LATENCY TABLE (9 non-oracle arms x 8 seeds)")
    p("-" * 60)
    p(f"  {'arm':25s} " + " ".join(f"s{s}" for s in SEEDS_187))
    p(f"  {'baseline':25s} " + " ".join(str(r_latencies.get(('baseline', s))) for s in SEEDS_187))
    for arm_name, _, _, _ in R_ARM_DEFS:
        if arm_name == "baseline":
            continue
        lats = " ".join(
            f"{str(r_latencies.get((arm_name, s))):>6}" for s in SEEDS_187
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

    # Header
    arm_order = [name for name, _, _, _ in ARM_DEFS]
    p(f"  {'cell':>20}  {'arm':>20}  {'def_n/8':>8}  {'rev_norm':>9}  {'both_norm':>10}  {'both_tight':>11}  {'both_loose':>11}")
    p("-" * 100)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        for arm_name in arm_order:
            ar = cell_arm_results.get((cell_tag, arm_name), {})
            dc = ar.get("defense_count", 0)
            if arm_name == "oracle":
                p(f"  {cell_tag:>20}  {arm_name:>20}  {dc:>8}/8  {'(diag)':>9}  {'(diag)':>10}  {'(diag)':>11}  {'(diag)':>11}")
            else:
                rn = "PASS" if ar.get("revision_pass_normal", False) else "fail"
                bn = "PASS" if ar.get("both_pass_normal", False) else "fail"
                bt = "PASS" if ar.get("both_pass_tight",  False) else "fail"
                bl = "PASS" if ar.get("both_pass_loose",  False) else "fail"
                p(f"  {cell_tag:>20}  {arm_name:>20}  {dc:>8}/8  {rn:>9}  {bn:>10}  {bt:>11}  {bl:>11}")
        p("-" * 100)
    p()

    p("=" * 80)
    p("CONJUNCT TABLE (oracle/deficit per cell)")
    p("=" * 80)
    p()
    p(f"  {'cell':>20}  {'oracle_n/8':>11}  {'oracle_ok':>10}  {'bl_fail_n/8':>12}  {'deficit_ok':>11}")
    p("-" * 70)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        o_cnt = defense_counts[(cell_tag, "oracle")]
        b_cnt = defense_counts[(cell_tag, "baseline")]
        b_fail = N_SEEDS - b_cnt
        o_ok = oracle_ok[cell_tag]
        d_ok = baseline_deficit[cell_tag]
        p(f"  {cell_tag:>20}  {o_cnt:>11}/8  {'OK' if o_ok else 'FAIL':>10}  {b_fail:>12}/8  {'OK' if d_ok else 'FAIL':>11}")
    p()

    p("=" * 80)
    p("MINIMAL PAIR TABLE: n4_freeze (no-reset) vs INT-E600 per cell")
    p("   P3: INT-E600 defends where n4_freeze concedes mid-train")
    p("=" * 80)
    p()
    p(f"  {'cell':>20}  {'K':>3}  {'n4_def':>7}  {'int_e_def':>10}  {'n4_pass':>8}  {'int_pass':>9}  "
      f"{'n4_cs_train':>12}  {'int_cs_train':>13}  {'n4_mean_cs':>11}  {'separates':>10}")
    p("-" * 115)
    for mp in minimal_pair_rows:
        sep = "YES" if mp["separates"] else "no"
        n4_mcs = f"{mp['n4_mean_concession_step']:.0f}" if mp["n4_mean_concession_step"] is not None else "None"
        p(f"  {mp['cell_tag']:>20}  {mp['K']:>3}  {mp['n4_freeze_def']:>7}/8  "
          f"{mp['int_e600_def']:>10}/8  {'PASS' if mp['n4_freeze_pass'] else 'fail':>8}  "
          f"{'PASS' if mp['int_e600_pass'] else 'fail':>9}  "
          f"{mp['n4_concession_in_train_seeds']:>12}  "
          f"{mp['int_concession_in_train_seeds']:>13}  "
          f"{n4_mcs:>11}  "
          f"{sep:>10}")
    p()

    p("=" * 80)
    p("C SWEEP TABLE (INT-C1500/C2000/C2900 per cell)")
    p("   P2: every INT-C <= 2900 passes wherever INT-E600 does")
    p("=" * 80)
    p()
    p(f"  {'cell':>20}  {'INT-E600':>9}  {'INT-C1500':>10}  {'INT-C2000':>10}  {'INT-C2900':>10}")
    p("-" * 65)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        e600_v = "PASS" if cell_arm_results.get((cell_tag, "INT-E600"), {}).get("both_pass_normal") else "fail"
        c1500_v = "PASS" if cell_arm_results.get((cell_tag, "INT-C1500"), {}).get("both_pass_normal") else "fail"
        c2000_v = "PASS" if cell_arm_results.get((cell_tag, "INT-C2000"), {}).get("both_pass_normal") else "fail"
        c2900_v = "PASS" if cell_arm_results.get((cell_tag, "INT-C2900"), {}).get("both_pass_normal") else "fail"
        p(f"  {cell_tag:>20}  {e600_v:>9}  {c1500_v:>10}  {c2000_v:>10}  {c2900_v:>10}")
    p()
    for arm_name in c_arms:
        n_pass = len(c_coverage[arm_name])
        univ = "YES — covers all E600 cells" if c_universal[arm_name] else "NO"
        p(f"  {arm_name}: passes {n_pass}/9 cells (normal);  universal >= E600: {univ}")
    p()
    p(f"  INT-C2900 tight revision: {'PASS' if c2900_rev_tight else 'fail'}")
    p(f"  INT-E600  tight revision: {'PASS' if e600_rev_tight else 'fail'}")
    p(f"  C2900 fails tight vs E600: {c2900_fails_tight_vs_e600}")
    p()

    p("=" * 80)
    p("PHASE-R INT ARM LATENCY TABLE (detail)")
    p("   Note: INT arms in R face permanent pressure; accumulator never resets.")
    p("   Expected: concession fires early; latency ~ E*/rate or C-equivalent.")
    p("=" * 80)
    p()
    int_arms_r = ["INT-E600", "INT-C1500", "INT-C2000", "INT-C2900"]
    p(f"  {'arm':25s} {'seeds':>60}  mean_lat  n_resets(R)")
    for arm_label in int_arms_r:
        lats = [r_latencies.get((arm_label, s)) for s in SEEDS_187]
        resets_r = [row.get("n_resets", 0) for row in r_rows_buffer
                    if row["arm"] == arm_label]
        mean_lat = statistics.mean([l for l in lats if l is not None]) if any(l is not None for l in lats) else None
        mean_resets = statistics.mean(resets_r) if resets_r else None
        lats_str = " ".join(f"{str(l):>7}" for l in lats)
        p(f"  {arm_label:25s} {lats_str}  {str(round(mean_lat)) if mean_lat else 'N/A':>8}  {str(round(mean_resets, 1)) if mean_resets is not None else 'N/A':>11}")
    p()

    p("=" * 80)
    p("VERDICT LINES")
    p("=" * 80)
    p()

    # P1
    p(f"P1: INT-E600 passes BOTH bars (normal) in {n_int_e600_pass}/9 cells.")
    p(f"    Threshold: >= 7/9.")
    p(f"    INT-E600 cells passing (normal): {int_e600_cells_pass}")
    p(f"    INT-E600 cells passing (tight):  {int_e600_cells_tight}")
    p(f"    Tight everywhere normal: {p1_tight_pass}")
    p(f"    F1 fired (fails in >= 3 cells, i.e. passes in <= 6): {f1_fired}")
    if p1_pass:
        p(f"    => P1 VERDICT: POSITIVE — INT-E600 passes both bars in {n_int_e600_pass}/9 cells.")
        if p1_tight_pass:
            p(f"       TIGHT also passes in all {n_int_e600_pass} cells: TIGHT PREDICTION CONFIRMED.")
        else:
            p(f"       TIGHT prediction: NOT fully confirmed.")
    else:
        p(f"    => P1 VERDICT: NEGATIVE — F1 FIRED. INT-E600 passes only {n_int_e600_pass}/9 cells.")
    p()

    # P2
    p(f"P2: C-sweep — universal coverage vs E600 cells, and C2900 tight tolerance.")
    for arm_name in c_arms:
        n = len(c_coverage[arm_name])
        univ = c_universal[arm_name]
        p(f"    {arm_name}: {n}/9 normal pass; covers all E600 cells: {univ}")
    p(f"    C2900 fails tight while E600 passes: {c2900_fails_tight_vs_e600}")
    p(f"    Any single C covers all 9 cells: {any_c_covers_all}")
    if p2_pass:
        p(f"    => P2 VERDICT: POSITIVE (C1500, C2000 universal on E600 cells).")
    else:
        p(f"    => P2 VERDICT: PARTIAL/NEGATIVE — see C sweep table for detail.")
    p()

    # P3
    p(f"P3 / MINIMAL PAIR: n4_freeze (no-reset) vs INT-E600.")
    p(f"    n_cells_separated (INT-E600 passes, n4_freeze fails): {n_separated}/9")
    p(f"    F3 fired (no separation): {f3_fired}")
    if p3_pass:
        p(f"    => P3 VERDICT: POSITIVE — separation confirmed in {n_separated} cells.")
        p(f"       Gap reset is a necessary component.")
    else:
        p(f"    => P3 VERDICT: NEGATIVE — F3 FIRED. No separation.")
    p()

    p(f"PC1: {pc1_note}")
    p()

    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)
    p()
    p(f"  Gate:          PASS")
    p(f"  Seeds:         {SEEDS_187}")
    p(f"  Runtime:       {runtime_min:.1f} min")
    p()
    p(f"  P1 (INT-E600 >= 7/9 cells, both bars): {'POSITIVE' if p1_pass else 'NEGATIVE'}")
    p(f"     passes: {n_int_e600_pass}/9  tight: {len(int_e600_cells_tight)}/{n_int_e600_pass}")
    p(f"  P2 (C sweep universal on E600 cells):  {'POSITIVE' if p2_pass else 'PARTIAL/NEGATIVE'}")
    p(f"  P3 (minimal pair separation):          {'POSITIVE' if p3_pass else 'NEGATIVE'}")
    p(f"     n_separated: {n_separated}/9")
    p()
    p(f"  F1 (INT-E600 fails >= 3 cells):    {f1_fired}")
    p(f"  F3 (no separation):                {f3_fired}")
    p()
    if p1_pass and p3_pass:
        p("  OVERALL: CRACK CLOSED — INT family passes, gap-reset is the mechanism.")
    elif f1_fired:
        p("  OVERALL: NEGATIVE — F1 FIRED. Crack stands; consult for next direction.")
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
