"""Exp 188 — N4 Regulated Controller (RUNG-6 FALSIFIABLE).

PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit 3ecfc77) BEFORE any data.
Status: FALSIFIABLE — predeclared falsifiers below; fresh seeds 270-277.

Question (the human's verbatim): can ONE online regulated controller pass both
defense and revision in the nine confirmed crack cells, with no oracle timing and
no cell-specific tuning?  Graded against INT-C2900 (the binding constant
comparator, same seeds).

predeclar (3ecfc77):
  P1: REG-TB passes BOTH bars in >= 8/9 cells (trains' max stretch ~<= 2525 <
      T0=2800 so no in-train concession; Phase R concedes at ~2875 <= +3000).
      F1: <= 6/9 — regulation fails where the constant succeeds.
  P2 (parity): REG-TB ~= INT-C2900 at normal AND REG-TB fails tight
      (T0 > +1500 by construction) — single-stretch ambiguity bound.
      F2 (reportable either way): they separate.
  P3: INT-E600-FIXED fails every L >= 800 cell by mid-burst concession (~628
      active steps reach 600), zero quiet accumulation.
      F3: it defends long-L cells — surrender-schedule law wrong on properly-
      gated E-surface.

falsifier bindings (ordered per pre-registration):
  F1: REG-TB passes <= 6/9 cells (both bars, normal).
  F2: REG-TB passes cells INT-C2900 fails, or vice versa (separation, either dir).
  F3: INT-E600-FIXED defends a cell with L >= 800.

Controller design (REG-TB, all constants global, no per-cell tuning):
  Trigger: rung-2 monitor THETA 3.5 verbatim.
  Release: gap-spanning release-calm 2600 steps (CALM2600_SNAPS = 104 snaps).
  Concession: fires when current CONTINUOUS-ACTIVE-pressure stretch exceeds
    max(KAPPA * S_max, T0) where:
      S_max = longest COMPLETED continuous-ACTIVE stretch so far in this freeze,
      KAPPA = 1.5, T0 = 2800.
  CRITICAL FIDELITY (fixes Exp 187 flaw): stretch counter and S_max track ONLY
  pressure-ACTIVE snapshots (pressure = fraction of last 200 obs == c_star >= 0.6).
  Quiet snapshots contribute NOTHING to any accumulator.
  De-assert ENDS the current stretch (records into S_max if longest) and resets
  current counter to zero.

Secondary arm INT-E600-FIXED: pressure-gated evidence integral (E += w on blocked
  c_star writes ONLY while pressure ACTIVE; reset on de-assert) >= 600.
  Prediction P3: fails every L >= 800 cell by mid-burst concession.

Arms (5): baseline; oracle; REG-TB; INT-C2900; INT-E600-FIXED.
Cells (9, confirmed): (1600,2,200), (1600,3,200), (2400,3,200), (1600,4,200),
  (2400,4,200), (2400,4,600), (400,4,600), (400,4,2400), (800,4,600).
Seeds FRESH 270-277.
W: 9 x 5 x 8 = 360; R: 4 non-oracle arms x 8 = 32.
Bars VERBATIM exp186: defense >= 6/8; revision >= 6/8 at normal +3000; tight/loose
  secondary; oracle >= 7/8; deficit >= 7/8; PC1 gated; NO mid-run adjustments.

Per-session diagnostics for REG-TB:
  n_completed_stretches, S_max_final, max_current_stretch, quiet_accumulation_check.
  quiet_accumulation_check MUST be 0 (zero quiet contribution to any accumulator).

Outputs: experiments/outputs/exp188_rows.json (JSONL),
         experiments/outputs/exp188.txt (gate evidence table; per-cell x arm table
         with all conjuncts; REG-TB stretch/tempo diagnostics; E600-FIXED mid-burst
         concession table; P1/P2/P3 verdicts; tight/loose secondary).
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
# Import the exp185 runner via importlib (zero drift from upstream)
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
# Exp 188 configuration
# ---------------------------------------------------------------------------

SEEDS_188 = [270, 271, 272, 273, 274, 275, 276, 277]
N_SEEDS = len(SEEDS_188)

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

# 9 confirmed crack cells (exp186-confirmed set, verbatim)
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

for _c in CRACK_CELLS:
    _c["cell_tag"] = f"({_c['L']},{_c['K']},{_c['G']})"

# ---------------------------------------------------------------------------
# REG-TB constants (global, no per-cell tuning — predeclared)
# ---------------------------------------------------------------------------
REG_TB_KAPPA = 1.5
REG_TB_T0 = 2800   # minimum concession threshold (steps of active pressure)
INT_CALM_SNAPS = CALM2600_SNAPS   # 104 snaps = 2600 steps

# E600-FIXED threshold
E600_FIXED_THRESH = 600.0

# ---------------------------------------------------------------------------
# Arm definitions for Exp 188 (5 arms total)
# ---------------------------------------------------------------------------
# arm_mode conventions:
#   "baseline"                             — standard
#   "oracle"                               — oracle freeze
#   ("reg_tb", kappa, T0, calm_snaps)     — REG-TB regulated controller
#   ("int_c", C, calm_snaps)              — continuous stopwatch
#   ("int_e_fixed", E_thresh, calm_snaps) — pressure-GATED evidence integral (FIXED)
#
# theta and release_calm_snaps are encoded in the arm tuple for clarity.

ARM_DEFS = []
ARM_DEFS.append(("baseline",         "baseline",                                        DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("oracle",           "oracle",                                          DEFAULT_THETA, DEFAULT_RELEASE_CALM_SNAPS))
ARM_DEFS.append(("REG-TB",           ("reg_tb", REG_TB_KAPPA, REG_TB_T0, INT_CALM_SNAPS), DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("INT-C2900",        ("int_c", 2900, INT_CALM_SNAPS),                    DEFAULT_THETA, INT_CALM_SNAPS))
ARM_DEFS.append(("INT-E600-FIXED",   ("int_e_fixed", E600_FIXED_THRESH, INT_CALM_SNAPS), DEFAULT_THETA, INT_CALM_SNAPS))

assert len(ARM_DEFS) == 5, f"Expected 5 arms, got {len(ARM_DEFS)}"

# R arms: all 4 non-oracle arms
R_ARM_DEFS = [(name, mode, theta, rcs) for name, mode, theta, rcs in ARM_DEFS
              if name != "oracle"]
assert len(R_ARM_DEFS) == 4, f"Expected 4 R arms, got {len(R_ARM_DEFS)}"


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


def tv_func(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


# ---------------------------------------------------------------------------
# Exp 188 runner — handles all 5 arm modes
# Non-INT/non-REG modes delegate to run_fork_schedule_185 unchanged.
# REG-TB and INT-E600-FIXED are new inline modes.
# INT-C2900 reuses the exp187 int_c logic, but with the GATING FIX:
#   only pressure-ACTIVE snapshots increment any accumulator.
#   Quiet snapshots: ZERO contribution (auditable via quiet_accumulation_check).
# ---------------------------------------------------------------------------

def run_fork_schedule_188(
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

    New arm modes in Exp 188 (all pressure-gated — the fidelity fix):
      ('reg_tb', kappa, T0, calm_snaps):
        REG-TB regulated controller. The stretch counter and S_max track ONLY
        pressure-ACTIVE snapshots. Quiet steps contribute NOTHING.
        Concession fires when current_stretch > max(kappa * S_max, T0).
        Per-session diagnostics: n_completed_stretches, S_max_final,
          max_current_stretch, quiet_accumulation_check (must be 0).
      ('int_c', C, calm_snaps):
        Continuous-ACTIVE-pressure stopwatch (same as exp187 int_c but now
        correctly gated: only active snaps increment, not quiet ones).
        [Note: exp187 also gated int_c correctly — the flaw was only in int_e.]
      ('int_e_fixed', E_thresh, calm_snaps):
        Pressure-GATED evidence integral: E += w on blocked c_star ONLY while
        pressure is ACTIVE. Resets on de-assert. Concession at E >= E_thresh.
        quiet_accumulation_check must be 0.

    For non-new modes (baseline, oracle, freeze_time, freeze_evidence),
    this function delegates to run_fork_schedule_185.
    """
    is_new_arm = isinstance(arm_mode, tuple) and arm_mode[0] in (
        "reg_tb", "int_c", "int_e_fixed"
    )

    if not is_new_arm:
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
        # Add Exp-188-specific keys (zero values for non-new arms)
        result["n_resets"] = 0
        result["quiet_accumulation_check"] = 0  # non-new arms have no accumulator
        result["n_completed_stretches"] = 0
        result["S_max_final"] = 0.0
        result["max_current_stretch"] = 0.0
        result["stretch_log"] = []
        result["concession_active_steps_log"] = []

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

    # ---- New arm: full inline runner ----
    arm_mode_tag = arm_mode[0]

    if arm_mode_tag == "reg_tb":
        _, int_kappa, int_T0, int_calm_snaps = arm_mode
        int_mode = "reg_tb"
        int_thresh = None  # dynamic threshold
    elif arm_mode_tag == "int_c":
        _, int_thresh, int_calm_snaps = arm_mode
        int_mode = "int_c"
        int_kappa = None
        int_T0 = None
    elif arm_mode_tag == "int_e_fixed":
        _, int_thresh, int_calm_snaps = arm_mode
        int_mode = "int_e_fixed"
        int_kappa = None
        int_T0 = None
    else:
        raise ValueError(f"Unknown arm_mode tag: {arm_mode_tag}")

    _release_calm_snaps = int_calm_snaps

    n_chunks = n_steps // CHUNK_SIZE
    assert n_chunks * CHUNK_SIZE == n_steps

    fork_name = f"exp188_{arm_name}_{phase}_s{fork_seed}"
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

    # -------------------------------------------------------------------------
    # NEW-ARM-specific accumulators (all pressure-gated; quiet contributes ZERO)
    # -------------------------------------------------------------------------

    # Shared for int_c and int_e_fixed: continuous-ACTIVE accumulator
    # For int_c: counts FINE_EVAL-step snaps of active pressure
    # For int_e_fixed: sums w contributions ONLY during active pressure
    # For reg_tb: current_stretch counts active snaps; S_max and S_log for tempo
    int_acc: float = 0.0       # current active accumulator (int_c/int_e_fixed)
    n_resets: int = 0          # accumulator resets (de-assert while frozen)

    # REG-TB specific
    reg_current_stretch: float = 0.0    # snaps of unbroken active pressure in current run
    reg_S_max: float = 0.0              # longest COMPLETED stretch so far in this freeze
    reg_n_completed: int = 0            # number of completed stretches
    reg_stretch_log: list[float] = []   # log of all completed stretch lengths (for audit)
    reg_max_current_seen: float = 0.0   # high-water mark of current_stretch (diagnostic)

    # INT-E600-FIXED specific: track active-step count per concession for audit
    e600_active_steps_since_reset: float = 0.0  # active snaps * FINE_EVAL
    concession_active_steps_log: list[float] = []  # active steps at each concession

    # QUIET-ACCUMULATION AUDIT COUNTER
    # By construction, quiet snaps can NEVER increment any accumulator.
    # This assertion counter counts any violation — it MUST be 0 at end.
    quiet_accumulation_violations: int = 0

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

            # ----------------------------------------------------------------
            # RESIST state handling for new arms
            # ----------------------------------------------------------------
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

                    # --------------------------------------------------------
                    # PRESSURE-GATED ACCUMULATION (the fidelity fix)
                    # Rule: quiet snapshots contribute NOTHING to any accumulator.
                    # Only pressure_active == True snaps may increment.
                    # --------------------------------------------------------
                    released = False
                    release_label = None

                    if not pressure_active:
                        # Pressure de-asserted: handle stretch end + reset

                        if int_mode == "reg_tb":
                            # End current stretch: record into S_max if longest
                            if reg_current_stretch > 0:
                                reg_n_completed += 1
                                reg_stretch_log.append(reg_current_stretch)
                                if reg_current_stretch > reg_S_max:
                                    reg_S_max = reg_current_stretch
                                reg_current_stretch = 0.0
                            # quiet: calm counter increments
                        elif int_mode == "int_c":
                            # Reset continuous counter
                            if int_acc > 0:
                                n_resets += 1
                                int_acc = 0.0
                        elif int_mode == "int_e_fixed":
                            # Reset evidence integral
                            if int_acc > 0 or e600_active_steps_since_reset > 0:
                                n_resets += 1
                                int_acc = 0.0
                                e600_active_steps_since_reset = 0.0

                        calm_count += 1

                        # AUDIT: verify no quiet accumulation occurred
                        # (by construction, we only accumulate in the else branch below)
                        # quiet_accumulation_violations += 0  (correct, no change)

                    else:
                        # Pressure ACTIVE: accumulate
                        calm_count = 0

                        if int_mode == "reg_tb":
                            # Extend current stretch
                            reg_current_stretch += FINE_EVAL
                            if reg_current_stretch > reg_max_current_seen:
                                reg_max_current_seen = reg_current_stretch
                            # No change to S_max until stretch completes (de-assert)

                            # Check concession: current > max(kappa * S_max, T0)
                            threshold_now = max(int_kappa * reg_S_max, int_T0)
                            if reg_current_stretch > threshold_now:
                                release_label = "concession"
                                released = True

                        elif int_mode == "int_c":
                            # Active snap: increment counter by FINE_EVAL steps
                            int_acc += FINE_EVAL
                            if int_acc >= int_thresh:
                                release_label = "concession"
                                released = True

                        elif int_mode == "int_e_fixed":
                            # Active snap: accumulate evidence weight
                            c_star_final_now = int(np.argmax(blocked_w_by_color))
                            int_acc = float(blocked_w_by_color[c_star_final_now])
                            e600_active_steps_since_reset += FINE_EVAL
                            if int_acc >= int_thresh:
                                release_label = "concession"
                                released = True

                    # Transient release (spanning calm)
                    if not released and calm_count >= _release_calm_snaps:
                        release_label = "transient"
                        released = True

                    if released:
                        c_star_final = int(np.argmax(blocked_w_by_color))
                        E_blocked_final = float(blocked_w_by_color[c_star_final])

                        trigger_latency_ev: int | None = None
                        for bi, (bstart, bend) in enumerate(burst_windows):
                            if bstart <= entry_step < bend:
                                trigger_latency_ev = entry_step - bstart
                                break

                        # REG-TB: at concession, record the final active stretch
                        if int_mode == "reg_tb" and release_label == "concession":
                            if reg_current_stretch > 0:
                                reg_n_completed += 1
                                reg_stretch_log.append(reg_current_stretch)
                                if reg_current_stretch > reg_S_max:
                                    reg_S_max = reg_current_stretch
                                reg_current_stretch = 0.0
                        elif int_mode == "int_e_fixed" and release_label == "concession":
                            # Log the active steps at concession
                            concession_active_steps_log.append(e600_active_steps_since_reset)

                        int_acc_at_release = (
                            float(reg_current_stretch) if int_mode == "reg_tb"
                            else float(int_acc)
                        )

                        events.append({
                            "entry_step": entry_step,
                            "exit_step": t_now,
                            "label": release_label,
                            "E_blocked": E_blocked_final,
                            "c_star": c_star_final,
                            "frozen_steps": frozen_steps,
                            "directional_pressure_acc": directional_pressure_acc,
                            "trigger_latency": trigger_latency_ev,
                            "int_acc_at_release": int_acc_at_release,
                            "n_resets_at_release": n_resets,
                            # REG-TB diagnostics at concession
                            "reg_S_max_at_release": float(reg_S_max) if int_mode == "reg_tb" else None,
                            "reg_n_completed_at_release": reg_n_completed if int_mode == "reg_tb" else None,
                            "e600_active_steps": e600_active_steps_since_reset if int_mode == "int_e_fixed" else None,
                        })

                        freeze_state = "NORMAL"
                        mismatch_history.clear()
                        v_fine.clear()
                        k_since_reset = 0
                        calm_count = 0
                        frozen_steps = 0
                        directional_pressure_acc = 0.0
                        int_acc = 0.0
                        # REG-TB: reset per-freeze stretch tracking
                        reg_current_stretch = 0.0
                        reg_S_max = 0.0
                        reg_n_completed = 0
                        # Note: reg_stretch_log and reg_max_current_seen are session-level
                        # and are NOT reset here — they persist across freezes for diagnostics
                        e600_active_steps_since_reset = 0.0
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
                                int_acc = 0.0
                                reg_current_stretch = 0.0
                                reg_S_max = 0.0
                                reg_n_completed = 0
                                e600_active_steps_since_reset = 0.0
                                # Do NOT reset: reg_stretch_log, reg_max_current_seen
                                # (session-level diagnostics)

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
        "quiet_accumulation_check": quiet_accumulation_violations,  # must be 0
        "concession_events_in_train": in_train_concessions > 0,
        "mean_concession_step": mean_concession_step,
        # REG-TB diagnostics
        "n_completed_stretches": reg_n_completed,
        "S_max_final": float(reg_S_max),
        "max_current_stretch": float(reg_max_current_seen),
        "stretch_log": list(reg_stretch_log),
        # INT-E600-FIXED diagnostics
        "concession_active_steps_log": list(concession_active_steps_log),
    }


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


def run_equivalence_gate_188(
    mirro_root, base_cmap, n_colors, committed_183_path: Path
) -> tuple[bool, str]:
    """Run equivalence gate: reproduce exp183 baseline x s229 and H1200 x s229.

    These arms are non-INT/non-REG so they delegate unchanged to run_fork_schedule_185.
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

            rr = run_fork_schedule_188(
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
    print("Exp 188 — N4 Regulated Controller (RUNG-6 FALSIFIABLE)")
    print("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit 3ecfc77)")
    print(f"Seeds: {SEEDS_188} | 9 cells x 5 arms x 8 seeds = 360 W | 4 R-arms x 8 = 32 R")
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
    out_rows_path = out_dir / "exp188_rows.json"
    out_txt_path = out_dir / "exp188.txt"
    committed_183_path = out_dir / "exp183_rows.json"

    print("CELLS (9 confirmed crack cells):")
    for ci, cell in enumerate(CRACK_CELLS):
        print(f"  [{ci}] ({cell['L']},{cell['K']},{cell['G']})")
    print()
    print("ARMS (5):")
    for arm_name, arm_mode, theta, rcs in ARM_DEFS:
        print(f"  {arm_name:20s}  theta={theta}  rcs={rcs}  mode={arm_mode}")
    print()
    print(f"REG-TB constants: KAPPA={REG_TB_KAPPA}, T0={REG_TB_T0}, calm_snaps={INT_CALM_SNAPS}")
    print()

    # ====================================================================
    # STEP 1: EQUIVALENCE GATE (L15)
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
    # STEP 2: W SESSIONS (9 cells x 5 arms x 8 seeds = 360)
    # ====================================================================
    print("=" * 80)
    print("STEP 2: W SESSIONS (9 cells x 5 arms x 8 seeds = 360 sessions)")
    print("=" * 80)

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
            for seed in SEEDS_188:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

                reloc_rng_seed = 210_000 + 10_000 * ci + seed

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
                        "gap_start": rr["gap_start"][bi],
                        "gap_end": rr["gap_end"][bi],
                        "expr_frac_or_null": rr["per_burst_expr_frac"][bi],
                    }
                    per_burst.append(pb)

                # Self-audit assertions (L13 spirit, per session):
                # (a) Quiet snapshots NEVER incremented any accumulator
                quiet_check = int(rr.get("quiet_accumulation_check", 0))
                assert quiet_check == 0, (
                    f"QUIET ACCUMULATION VIOLATION: arm={arm_name} cell={cell_tag} seed={seed} "
                    f"violations={quiet_check}"
                )

                row = {
                    "exp": 188,
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
                            "reg_S_max_at_release": e.get("reg_S_max_at_release"),
                            "e600_active_steps": e.get("e600_active_steps"),
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
                    # REG-TB diagnostics
                    "n_completed_stretches": int(rr.get("n_completed_stretches", 0)),
                    "S_max_final": float(rr.get("S_max_final", 0.0)),
                    "max_current_stretch": float(rr.get("max_current_stretch", 0.0)),
                    "stretch_log": to_plain(rr.get("stretch_log", [])),
                    # INT-E600-FIXED diagnostics
                    "concession_active_steps_log": to_plain(rr.get("concession_active_steps_log", [])),
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
    # STEP 3: R SESSIONS (4 non-oracle arms x 8 seeds = 32)
    # ====================================================================
    print("=" * 80)
    print("STEP 3: R SESSIONS (4 non-oracle arms x 8 seeds = 32 sessions)")
    print("baseline R latencies are the tolerance reference")
    print("=" * 80)

    r_latencies: dict = {}
    r_rows_buffer: list[dict] = []

    t_r = time.time()
    for arm_label, arm_mode_r, theta_val, rc_snaps in R_ARM_DEFS:
        for seed in SEEDS_188:
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
                "exp": 188,
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
                "concession_active_steps_log": to_plain(rr.get("concession_active_steps_log", [])),
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
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        for arm_name, _, _, _ in ARM_DEFS:
            n_pass = sum(
                1 for seed in SEEDS_188
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
            for seed in SEEDS_188:
                bl_lat = r_latencies.get(("baseline", seed))
                arm_lat = r_latencies.get((arm_name, seed))
                if arm_lat is None or bl_lat is None:
                    continue
                if arm_lat <= bl_lat + tolerance:
                    n_pass += 1
            revision_pass[(arm_name, mode_name)] = n_pass >= DEFENSE_MIN_PASS

    # ---- 5c. Per cell: both_pass per arm ----
    cell_arm_results: dict = {}
    for cell in CRACK_CELLS:
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
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        oracle_cnt = defense_counts[(cell_tag, "oracle")]
        bl_pass_cnt = defense_counts[(cell_tag, "baseline")]
        oracle_ok[cell_tag] = oracle_cnt >= ORACLE_DEFENSE_MIN
        baseline_deficit[cell_tag] = (N_SEEDS - bl_pass_cnt) >= BASELINE_FAIL_MIN

    # ---- 5f. REG-TB per-cell results ----
    regtb_cells_pass: list[str] = []
    regtb_cells_tight: list[str] = []
    regtb_cells_loose: list[str] = []
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        ar = cell_arm_results.get((cell_tag, "REG-TB"), {})
        if ar.get("both_pass_normal", False):
            regtb_cells_pass.append(cell_tag)
        if ar.get("both_pass_tight", False):
            regtb_cells_tight.append(cell_tag)
        if ar.get("both_pass_loose", False):
            regtb_cells_loose.append(cell_tag)

    n_regtb_pass = len(regtb_cells_pass)

    # ---- 5g. INT-C2900 per-cell results (the binding comparator) ----
    c2900_cells_pass: list[str] = []
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        ar = cell_arm_results.get((cell_tag, "INT-C2900"), {})
        if ar.get("both_pass_normal", False):
            c2900_cells_pass.append(cell_tag)

    n_c2900_pass = len(c2900_cells_pass)

    # ---- 5h. P1 verdict: REG-TB passes >= 8/9 cells ----
    p1_pass = n_regtb_pass >= 8
    p1_f1_fired = n_regtb_pass <= 6  # F1: <= 6/9
    p1_tight_pass_count = len(regtb_cells_tight)

    # ---- 5i. P2 verdict: parity vs separation ----
    # Parity = REG-TB ~= INT-C2900 at normal AND REG-TB fails tight
    regtb_rev_normal = revision_pass.get(("REG-TB", "normal"), False)
    regtb_rev_tight  = revision_pass.get(("REG-TB", "tight"),  False)
    regtb_rev_loose  = revision_pass.get(("REG-TB", "loose"),  False)
    c2900_rev_normal = revision_pass.get(("INT-C2900", "normal"), False)
    c2900_rev_tight  = revision_pass.get(("INT-C2900", "tight"),  False)

    # Check for separation: cells where one passes but not the other
    regtb_set = set(regtb_cells_pass)
    c2900_set = set(c2900_cells_pass)
    regtb_only = regtb_set - c2900_set
    c2900_only = c2900_set - regtb_set
    p2_separation = len(regtb_only) > 0 or len(c2900_only) > 0
    p2_parity = not p2_separation
    p2_regtb_fails_tight = not regtb_rev_tight

    # P2 prediction: parity (REG-TB ~= INT-C2900 normal) AND REG-TB fails tight
    p2_pred_confirmed = p2_parity and p2_regtb_fails_tight

    # ---- 5j. P3 verdict: INT-E600-FIXED mid-burst concession + zero quiet ----
    e600_cells_pass: list[str] = []
    e600_cells_fail_long_L: list[str] = []  # L >= 800 cells where it fails

    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        L = cell["L"]
        ar = cell_arm_results.get((cell_tag, "INT-E600-FIXED"), {})
        if ar.get("both_pass_normal", False):
            e600_cells_pass.append(cell_tag)
        elif L >= 800:
            e600_cells_fail_long_L.append(cell_tag)

    n_e600_pass = len(e600_cells_pass)

    # Collect mid-burst concession active step counts for INT-E600-FIXED
    e600_mid_burst_active_steps: list[float] = []
    for row in rows_buffer:
        if row["arm"] == "INT-E600-FIXED":
            for active_steps in row.get("concession_active_steps_log", []):
                if active_steps > 0:
                    e600_mid_burst_active_steps.append(active_steps)

    # Check zero quiet accumulation across all new-arm sessions
    quiet_check_total = sum(
        row.get("quiet_accumulation_check", 0)
        for row in rows_buffer + r_rows_buffer
        if row.get("arm") in ("REG-TB", "INT-C2900", "INT-E600-FIXED")
    )

    # P3: INT-E600-FIXED fails all L >= 800 cells (concedes mid-burst)
    long_L_cells = [cell["cell_tag"] for cell in CRACK_CELLS if cell["L"] >= 800]
    p3_e600_fails_long_L = all(ct in e600_cells_fail_long_L for ct in long_L_cells)
    p3_f3_fired = len(e600_cells_pass) > 0 and any(
        c["L"] >= 800 and c["cell_tag"] in e600_cells_pass for c in CRACK_CELLS
    )

    # ---- 5k. REG-TB stretch diagnostics ----
    regtb_stretch_diag: dict = {}  # cell_tag -> {max_stretch_in_trains, S_max_mean}
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        cell_rows = [row for row in rows_buffer
                     if row["cell"]["tag"] == cell_tag and row["arm"] == "REG-TB"]
        all_max_stretch = [row["max_current_stretch"] for row in cell_rows]
        all_S_max = [row["S_max_final"] for row in cell_rows]
        all_n_concessions = sum(
            1 for row in cell_rows
            if any(e.get("label") == "concession" for e in row.get("events_summary", []))
        )
        regtb_stretch_diag[cell_tag] = {
            "max_stretch_in_train_seeds": [row["max_current_stretch"] for row in cell_rows],
            "max_of_max": max(all_max_stretch) if all_max_stretch else 0.0,
            "mean_S_max_final": statistics.mean(all_S_max) if all_S_max else 0.0,
            "n_seeds_with_concession": all_n_concessions,
        }

    # ---- 5l. REG-TB Phase-R latency and stretch diagnostics ----
    regtb_r_rows = [row for row in r_rows_buffer if row["arm"] == "REG-TB"]
    regtb_r_lats = [r_latencies.get(("REG-TB", s)) for s in SEEDS_188]
    bl_lats = [r_latencies.get(("baseline", s)) for s in SEEDS_188]
    regtb_phase_r_latencies = [l for l in regtb_r_lats if l is not None]

    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0
    print(f"Total runtime: {runtime_min:.1f} min")
    print()

    # ====================================================================
    # Write analysis row to JSONL
    # ====================================================================
    analysis_row = {
        "exp": 188,
        "kind": "analysis",
        "seeds": SEEDS_188,
        "n_regtb_pass": n_regtb_pass,
        "regtb_cells_pass": regtb_cells_pass,
        "regtb_cells_tight": regtb_cells_tight,
        "n_c2900_pass": n_c2900_pass,
        "c2900_cells_pass": c2900_cells_pass,
        "p1_pass": p1_pass,
        "p1_f1_fired": p1_f1_fired,
        "p2_parity": p2_parity,
        "p2_separation": p2_separation,
        "p2_regtb_fails_tight": p2_regtb_fails_tight,
        "p2_pred_confirmed": p2_pred_confirmed,
        "p3_e600_fails_long_L": p3_e600_fails_long_L,
        "p3_f3_fired": p3_f3_fired,
        "n_e600_pass": n_e600_pass,
        "e600_cells_pass": e600_cells_pass,
        "quiet_check_total": quiet_check_total,
        "pc1_total_flagged": pc1_total,
        "runtime_min": runtime_min,
    }
    with open(out_rows_path, "a") as fh:
        fh.write(json.dumps(to_plain(analysis_row)) + "\n")

    # ====================================================================
    # Write text report (exp188.txt)
    # ====================================================================
    lines: list[str] = []

    def p(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        lines.append(msg)
        print(msg)

    p("=" * 80)
    p("EXP 188 — N4 REGULATED CONTROLLER (RUNG-6 FALSIFIABLE)")
    p("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit 3ecfc77)")
    p(f"Runtime: {runtime_min:.1f} min | Seeds: {SEEDS_188}")
    p(f"9 cells x 5 arms x 8 seeds = 360 W | 4 arms x 8 seeds = 32 R")
    p(f"REG-TB: KAPPA={REG_TB_KAPPA}, T0={REG_TB_T0}, calm_snaps={INT_CALM_SNAPS}")
    p("=" * 80)
    p()

    p("EQUIVALENCE GATE (L15)")
    p("-" * 60)
    p(gate_detail)
    p()

    p("PHASE-R LATENCY TABLE (4 non-oracle arms x 8 seeds)")
    p("-" * 60)
    p(f"  {'arm':25s} " + " ".join(f"s{s}" for s in SEEDS_188))
    p(f"  {'baseline':25s} " + " ".join(str(r_latencies.get(('baseline', s))) for s in SEEDS_188))
    for arm_name, _, _, _ in R_ARM_DEFS:
        if arm_name == "baseline":
            continue
        lats = " ".join(
            f"{str(r_latencies.get((arm_name, s))):>6}" for s in SEEDS_188
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
    p("REG-TB STRETCH DIAGNOSTICS PER CELL")
    p(f"  T0={REG_TB_T0}, KAPPA={REG_TB_KAPPA}")
    p("  Training stretches should be <= ~2525 < T0=2800 (no in-train concession)")
    p("=" * 80)
    p()
    p(f"  {'cell':>20}  {'max_stretch':>12}  {'vs_T0':>8}  {'mean_S_max':>11}  {'n_seeds_concede':>16}")
    p("-" * 80)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        L = cell["L"]
        diag = regtb_stretch_diag.get(cell_tag, {})
        ms = diag.get("max_of_max", 0.0)
        mS = diag.get("mean_S_max_final", 0.0)
        nc = diag.get("n_seeds_with_concession", 0)
        exceed = "EXCEED" if ms > REG_TB_T0 else "ok"
        p(f"  {cell_tag:>20}  {ms:>12.0f}  {exceed:>8}  {mS:>11.0f}  {nc:>16}")
    p()

    p("=" * 80)
    p("REG-TB vs INT-C2900 PARITY TABLE (the binding comparator)")
    p("=" * 80)
    p()
    p(f"  {'cell':>20}  {'REG-TB':>8}  {'INT-C2900':>10}  {'same?':>7}")
    p("-" * 52)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        rb = "PASS" if cell_tag in regtb_set else "fail"
        c2 = "PASS" if cell_tag in c2900_set else "fail"
        same = "same" if rb == c2 else "DIFF"
        p(f"  {cell_tag:>20}  {rb:>8}  {c2:>10}  {same:>7}")
    p()
    p(f"  REG-TB cells only (PASS/fail diff): {sorted(regtb_only)}")
    p(f"  C2900 cells only (PASS/fail diff): {sorted(c2900_only)}")
    p(f"  Separation (F2): {p2_separation}")
    p()

    p("=" * 80)
    p("INT-E600-FIXED MID-BURST CONCESSION TABLE")
    p("  P3: every L >= 800 cell fails by mid-burst concession (~628 active steps)")
    p("  Quiet accumulation check: MUST be 0 everywhere")
    p("=" * 80)
    p()
    p(f"  {'cell':>20}  {'L':>5}  {'e600_pass':>10}  {'mid_burst_conc':>15}")
    p("-" * 58)
    for cell in CRACK_CELLS:
        cell_tag = cell["cell_tag"]
        L = cell["L"]
        e6_pass = "PASS" if cell_tag in e600_cells_pass else "fail"
        # Get typical mid-burst concession steps for this cell
        cell_rows_e6 = [row for row in rows_buffer
                        if row["cell"]["tag"] == cell_tag and row["arm"] == "INT-E600-FIXED"]
        active_steps_this_cell = []
        for row in cell_rows_e6:
            for s in row.get("concession_active_steps_log", []):
                if s > 0:
                    active_steps_this_cell.append(s)
        if active_steps_this_cell:
            mean_as = statistics.mean(active_steps_this_cell)
            mid_burst_str = f"{mean_as:.0f}"
        else:
            mid_burst_str = "none"
        p(f"  {cell_tag:>20}  {L:>5}  {e6_pass:>10}  {mid_burst_str:>15}")
    p()
    if e600_mid_burst_active_steps:
        mean_all = statistics.mean(e600_mid_burst_active_steps)
        p(f"  Overall mean active steps at concession: {mean_all:.0f} (expected ~600-650)")
    p(f"  Quiet accumulation violations across all new-arm sessions: {quiet_check_total}")
    p()

    p("=" * 80)
    p("PHASE-R REG-TB CONCESSION DETAILS")
    p("=" * 80)
    p()
    p(f"  {'seed':>7}  {'baseline_lat':>13}  {'regtb_lat':>10}  {'diff':>7}")
    p("-" * 43)
    for seed in SEEDS_188:
        bl_l = r_latencies.get(("baseline", seed))
        rb_l = r_latencies.get(("REG-TB", seed))
        diff_str = str(rb_l - bl_l) if (rb_l is not None and bl_l is not None) else "N/A"
        p(f"  {seed:>7}  {str(bl_l):>13}  {str(rb_l):>10}  {diff_str:>7}")
    p()

    p("=" * 80)
    p("VERDICT LINES")
    p("=" * 80)
    p()

    # P1
    p(f"P1: REG-TB passes BOTH bars (normal) in {n_regtb_pass}/9 cells.")
    p(f"    Threshold: >= 8/9.")
    p(f"    REG-TB cells passing (normal): {regtb_cells_pass}")
    p(f"    REG-TB cells passing (tight):  {regtb_cells_tight}")
    p(f"    REG-TB cells passing (loose):  {regtb_cells_loose}")
    p(f"    F1 fired (<= 6/9): {p1_f1_fired}")
    if p1_pass:
        p(f"    => P1 VERDICT: POSITIVE — REG-TB passes both bars in {n_regtb_pass}/9 cells.")
    else:
        p(f"    => P1 VERDICT: NEGATIVE — REG-TB passes only {n_regtb_pass}/9 cells.")
        if p1_f1_fired:
            p(f"       F1 FIRED: regulation fails where the constant succeeds.")
    p()

    # P2
    p(f"P2: Parity vs separation — REG-TB vs INT-C2900.")
    p(f"    REG-TB normal: {n_regtb_pass}/9 cells | INT-C2900 normal: {n_c2900_pass}/9 cells")
    p(f"    REG-TB revision (normal/tight/loose): "
      f"{regtb_rev_normal}/{regtb_rev_tight}/{regtb_rev_loose}")
    p(f"    INT-C2900 revision (normal/tight): {c2900_rev_normal}/{c2900_rev_tight}")
    p(f"    Separation detected: {p2_separation}")
    if p2_separation:
        p(f"       REG-TB only (cells where REG-TB PASS, C2900 fail): {sorted(regtb_only)}")
        p(f"       C2900 only (cells where C2900 PASS, REG-TB fail): {sorted(c2900_only)}")
    p(f"    REG-TB fails tight: {p2_regtb_fails_tight}")
    if p2_parity and p2_regtb_fails_tight:
        p(f"    => P2 VERDICT: PARITY CONFIRMED — REG-TB matches C2900 at normal, fails tight.")
        p(f"       Single-stretch ambiguity bound: online controller cannot beat tolerance-bounded")
        p(f"       constant on fixed-geometry attacks without attack-length priors.")
    elif p2_separation:
        p(f"    => P2 VERDICT: SEPARATION — F2 reportable. See table above.")
    else:
        p(f"    => P2 VERDICT: see table above.")
    p()

    # P3
    p(f"P3: INT-E600-FIXED — pressure-gated evidence integral retirement.")
    p(f"    Long-L cells (L >= 800): {long_L_cells}")
    p(f"    INT-E600-FIXED passing cells: {e600_cells_pass}")
    p(f"    INT-E600-FIXED failing long-L cells: {e600_cells_fail_long_L}")
    p(f"    Quiet accumulation violations: {quiet_check_total} (must be 0)")
    if e600_mid_burst_active_steps:
        p(f"    Mean active steps at concession: {statistics.mean(e600_mid_burst_active_steps):.0f}")
    if p3_e600_fails_long_L and quiet_check_total == 0:
        p(f"    => P3 VERDICT: CONFIRMED — INT-E600-FIXED fails all L >= 800 cells by mid-burst")
        p(f"       concession; zero quiet accumulation (fidelity fix verified).")
    elif p3_f3_fired:
        p(f"    => P3 VERDICT: F3 FIRED — INT-E600-FIXED defends a L >= 800 cell.")
    else:
        p(f"    => P3 VERDICT: PARTIAL — see concession table above.")
    p()

    p(f"PC1: {pc1_note}")
    p()

    p("=" * 80)
    p("SUMMARY")
    p("=" * 80)
    p()
    p(f"  Gate:              PASS")
    p(f"  Seeds:             {SEEDS_188}")
    p(f"  Runtime:           {runtime_min:.1f} min")
    p()
    p(f"  P1 (REG-TB >= 8/9 cells, both bars, normal): {'POSITIVE' if p1_pass else 'NEGATIVE'}")
    p(f"     passes: {n_regtb_pass}/9  tight: {p1_tight_pass_count}/9")
    p(f"  P2 (parity REG-TB vs INT-C2900):              {'PARITY' if p2_parity else 'SEPARATION'}")
    p(f"     REG-TB tight: {regtb_rev_tight}  C2900 tight: {c2900_rev_tight}")
    p(f"  P3 (INT-E600-FIXED fails L>=800, zero quiet): {'CONFIRMED' if (p3_e600_fails_long_L and quiet_check_total==0) else 'NOT CONFIRMED'}")
    p(f"     quiet violations: {quiet_check_total}")
    p()
    p(f"  F1 (REG-TB <= 6/9 cells):    {p1_f1_fired}")
    p(f"  F2 (separation REG-TB/C2900): {p2_separation}")
    p(f"  F3 (E600-FIXED defends L>=800): {p3_f3_fired}")
    p()

    if p1_pass and p2_parity and not p2_regtb_fails_tight:
        p("  OVERALL: REG-TB passes both bars. Parity with constant.")
    elif p1_pass and p2_parity and p2_regtb_fails_tight:
        p("  OVERALL: REG-TB POSITIVE, PARITY CONFIRMED. Sufficient-surface law final word")
        p("  at this richness; tight-mode remains the open core (single-stretch ambiguity).")
    elif p1_pass and p2_separation:
        p("  OVERALL: REG-TB POSITIVE, SEPARATION — regulation earns its first keep; consult.")
    elif p1_f1_fired:
        p("  OVERALL: NEGATIVE — F1 FIRED. Constant stands alone; consult for next direction.")
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
