"""Exp 184 — N4 Fixed-H Squeeze Map (EXPLORATORY).

PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit c220d2c) BEFORE any data.

Predeclared shape predictions + FALSIFIERS (copied verbatim-in-substance from the
pre-registration card, commit c220d2c, committed before any data; this docstring
note added post-run for the mechanical rubric — content unchanged):
P1 dose-route H-invariance within the H >= L-75 stratum; FALSIFIER F1 = defense
improves monotonically with H in that stratum. P2 (L=2400, tight) interval-empty;
FALSIFIER F2 = some H passes both bars there. P3 oracle defends every cell >= 2/3
seeds; FALSIFIER F3 = an oracle-failed cell. F4 (map-validity falsifier) =
degenerate map (trigger silent off the Exp-183 geometry, or everything passes).

EXPLORATORY squeeze map; 3 seeds/cell (FRESH 240-242).
Smoke = cell (L=800, K=3, G=2400), all arms, seed 240, disclosed.

Grid: L in {400,800,1200,1600,2400}; K in {1,2,3,4};
      G in {200,600,1200,2400}; H in {600,900,1200,1800,2400,3000,4200,6000};
      revision-tolerance modes normal/tight/loose = +3000/+1500/+6000 over
      same-seed baseline Phase-R latency (evaluation bars on the SAME Phase-R runs).
      Cells enumerated lexicographically by (L,K,G), cell_idx 0-79.

W sessions: settle 6000; attack color = argmin(v) at step 6000 FIXED for all K bursts
  (exogenous); K bursts of length L, gaps G; 2500-step tail;
  length 6000+K*L+(K-1)*G+2500. Captivity mechanics verbatim exp183;
  relocation rng default_rng(190000+1000*cell_idx+seed).

R sessions: verbatim exp183 Phase R; arms baseline + 8 H + n4-ref (no oracle); latency
  criterion verbatim.

Arms: baseline; freeze_time H-sweep (8: 600,900,1200,1800,2400,3000,4200,6000);
  n4 freeze_evidence E_STAR=600 (REFERENCE only, excluded from exists_H_both);
  oracle exact-train freeze (DIAGNOSTIC witness).
  W 80x11x3=2640 sessions; R 30.

EQUIVALENCE GATE (licenses everything): the generalized runner configured to Exp 183's
  exact setup (BURST_WINDOWS 6000-6800/9000-9800/12000-12800, ENDOGENOUS argmin-at-onset
  color per burst, relocation rng 160000+seed, N_STEPS 15000) must bit-match the COMMITTED
  experiments/outputs/exp183_rows.json rows for (baseline x seed 229) and
  (H1200 x seed 229): gap_start, gap_end, d_b, tv_b, recovered, n_events, event fields
  {label, entry_step, frozen_steps, E_blocked (atol 1e-9), c_star, trigger_latency}.
  Gate FAIL => stop, report, do NOT run the grid.

Primary: defense(arm,cell,seed) = expr_frac(attack color) < 0.5 over
  [last_bend+1500, last_bend+2000). defense_pass(arm,cell) = >=2/3 seeds.
  revision_pass(H,mode) = latency <= baseline+tolerance in >=2/3 seeds.
  both_pass = defense AND revision. exists_H_both(cell,mode) = any H.
  Candidate crack cells: NOT exists_H_both while oracle defends (>=2/3 seeds).

Per-burst quantities are DIAGNOSTIC only (windows collide at G<2000): gap ledger per
  burst (gap at burst-start snap; frozen-plateau gap via the L13 constancy-asserted read
  where a full-frozen snap interval exists; gap at burst-end snap), head dose, events,
  per-burst expr where [bend+1500,bend+2000) fits before the next burst.

Shape predictions evaluated mechanically in the txt:
  P1 within each cell, among H-arms with H >= L-75, defense outcome is H-INVARIANT
     (report any cell where defense improves monotonically with H in that stratum = F1).
  P2: (L=2400, tight) cells predicted interval-empty (defense needs H >~ 2325; tight
     allows H <~ 1650).
  P3: oracle defends every cell at >=2/3 seeds (report failures = F3/class-E flags).
  F4: degenerate map (trigger never fires off the exp183 geometry, or everything passes)
     is a reportable result.

PC1 ahat_drift < 0.15 gated per session; settle TV(pi) over [5000,6000) logged,
  flagged >0.05, NOT gated; assert argmin != argmax at step 6000.

Outputs: experiments/outputs/exp184_rows.json (JSONL), experiments/outputs/exp184.txt,
  script experiments/exp184_n4_fixed_h_squeeze_map.py.
  No crack CLAIM in the outputs — candidate cells only; classification is next iteration.
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
# Configuration
# ---------------------------------------------------------------------------

SEEDS = [240, 241, 242]   # 3 fresh forks
SMOKE_SEED = 240

# Phase-R sessions use verbatim Exp 183 N_STEPS
N_STEPS_PHASE_R = 15_000
CHUNK_SIZE = 100

# Value decay (verbatim Exp 183)
LAMBDA = 0.9997
INIT_MASS = 1.0 / (1.0 - LAMBDA)   # ~3333.3

# Phase R: permanent relocation from step 6000
PHASE_R_START = 6000

# Burst RNG seed offsets
# Phase W: use relocation rng default_rng(190000+1000*cell_idx+seed)
# Phase R: rng 165_000+seed (verbatim exp183)
BURST_SEED_OFFSET_R = 165_000

# Exp 183 equivalence gate: Phase W burst seed offset
BURST_SEED_OFFSET_W_EXP183 = 160_000

# Coarse snapshot cadence (retained for gap/diagnostic tables)
EVAL = 100

# Fine trigger cadence constants (verbatim Exp 183)
FINE_EVAL = 25
MON_W_SNAPS = 40
CTRL_MBAR_WINDOW = 120
CTRL_MIN_SNAPS = 40
RELEASE_CALM_SNAPS = 8
REFRACTORY_CHECKS = 8

# Freeze-gate constants (verbatim Exp 183)
THETA = 3.5
E_STAR = 600.0
PRESSURE_WINDOW = 200
PRESSURE_FRAC = 0.6

# Fixed H sweep for Exp 184 (8 arms: adds 4200 and 6000 vs Exp 183)
FIXED_HORIZONS = [600, 900, 1200, 1800, 2400, 3000, 4200, 6000]

# Grid definition
L_VALS = [400, 800, 1200, 1600, 2400]
K_VALS = [1, 2, 3, 4]
G_VALS = [200, 600, 1200, 2400]

# Revision tolerance modes: (mode_name, tolerance_steps)
REVISION_MODES = [
    ("normal", 3000),
    ("tight",  1500),
    ("loose",  6000),
]

# Phase-R latency criterion (verbatim Exp 183)
P6_HOLD = 2000

# Precondition
PC1_AHAT_DRIFT_MAX = 0.15
SETTLE_TV_FLAG_MAX = 0.05

# Defense criterion: expr_frac(attack color) < 0.5 over [last_bend+1500, last_bend+2000)
DEFENSE_FRAC_THRESH = 0.5
DEFENSE_WINDOW_OFFSET_START = 1500
DEFENSE_WINDOW_OFFSET_END = 2000

# Minimum seeds-per-cell for defense/revision pass
MIN_PASS_FRACTION = 2   # >= 2/3 seeds

# Arms for W sessions (11 per cell)
# (name, mode) where mode: "baseline" | "freeze_evidence" | ("freeze_time", H) | "oracle"
W_ARMS: list[tuple[str, object]] = (
    [("baseline", "baseline"), ("n4_freeze", "freeze_evidence")]
    + [(f"H{h}", ("freeze_time", h)) for h in FIXED_HORIZONS]
    + [("oracle", "oracle")]
)
# 11 arms total

# Arms for R sessions (10: no oracle)
R_ARMS: list[tuple[str, object]] = (
    [("baseline", "baseline"), ("n4_freeze", "freeze_evidence")]
    + [(f"H{h}", ("freeze_time", h)) for h in FIXED_HORIZONS]
)
# 10 arms total

# Equivalence gate reference: exp183 burst windows
EXP183_BURST_WINDOWS = [
    (6000, 6800),
    (9000, 9800),
    (12000, 12800),
]
EXP183_N_STEPS = 15_000
EXP183_GATE_SEEDS = [229]
EXP183_GATE_ARMS = ["baseline", "H1200"]

# Enumerate cells lexicographically by (L, K, G)
def enumerate_cells() -> list[dict]:
    cells = []
    idx = 0
    for L in L_VALS:
        for K in K_VALS:
            for G in G_VALS:
                n_steps = 6000 + K * L + (K - 1) * G + 2500
                cells.append({
                    "L": L, "K": K, "G": G,
                    "idx": idx,
                    "n_steps": n_steps,
                })
                idx += 1
    return cells

CELLS = enumerate_cells()
assert len(CELLS) == 80, f"Expected 80 cells, got {len(CELLS)}"


# ---------------------------------------------------------------------------
# Helper functions (verbatim Exp 183 where applicable)
# ---------------------------------------------------------------------------

def pi_of(v_row: np.ndarray) -> np.ndarray:
    s = v_row.sum()
    return v_row / s if s > 0 else np.ones_like(v_row) / len(v_row)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())


def snap_index(step: int) -> int:
    return step // EVAL - 1


def phase_r_latency(
    expressed_arr: np.ndarray,
    regime_color: int,
    n_steps: int,
) -> int | None:
    max_t = n_steps - P6_HOLD
    for t in range(PHASE_R_START, max_t + 1):
        if expressed_arr[t] == regime_color:
            if np.all(expressed_arr[t:t + P6_HOLD] == regime_color):
                return t - PHASE_R_START
    return None


# ---------------------------------------------------------------------------
# Generalized runner: run_fork_schedule
# Copied from exp183.run_fork with MINIMAL surgical diffs only:
#   (a) burst_windows passed as a parameter
#   (b) color mode parameter: 'endogenous' (argmin at each burst onset — exp183 verbatim)
#       vs 'exogenous_fixed' (one color computed as argmin(v) at step 6000, all bursts)
#   (c) relocation rng seed passed as a parameter
#   (d) N_STEPS passed as a parameter
#   (e) oracle mode freezes exactly during the provided burst windows
# PRESERVE every rng call site and its order.
# ---------------------------------------------------------------------------

def run_fork_schedule(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    arm_name: str,
    arm_mode: object,
    phase: str,           # "W" | "R"
    burst_windows: list,  # (a) parameterized
    color_mode: str,      # (b) 'endogenous' | 'exogenous_fixed'
    reloc_rng_seed: int,  # (c) parameterized
    n_steps: int,         # (d) parameterized
    n_actions: int = 4,
) -> dict:
    """Run one fork for n_steps steps under the given arm and phase.

    Phase W: bursts at burst_windows with captivity; color determined by color_mode.
    Phase R: PERMANENT captivity from PHASE_R_START onward; color = argmin(v) at onset.

    arm_mode variants (verbatim exp183):
      "baseline"          -> g=1, no freeze
      "freeze_evidence"   -> state machine with E_blocked >= E_STAR concession
      float               -> fixed write-gain (not used in exp184 W, kept for compat)
      ("freeze_time", H)  -> state machine with frozen_steps >= H concession
      "oracle"            -> Phase W only: freeze exactly during burst windows
    """
    n_chunks = n_steps // CHUNK_SIZE
    assert n_chunks * CHUNK_SIZE == n_steps, f"n_steps={n_steps} must be divisible by {CHUNK_SIZE}"

    fork_name = f"exp184_{arm_name}_{phase}_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    # Initialize v from spine's value_counts at equilibrium mass
    vc = c.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v = (vc / vc_sum) * INIT_MASS
    else:
        v = np.ones(n_colors) * (INIT_MASS / n_colors)

    # Pre-build per-color cell lists
    color_cells: list[list[int]] = [[] for _ in range(n_colors)]
    for cell_idx, color in enumerate(base_cmap):
        color_cells[color].append(cell_idx)
    color_cells_arr = [np.array(lst, dtype=np.int32) for lst in color_cells]

    # Burst / regime RNG
    # (c) reloc_rng_seed is passed as parameter; used directly
    if phase == "W":
        burst_rng = np.random.default_rng(reloc_rng_seed)
    else:
        burst_rng = np.random.default_rng(BURST_SEED_OFFSET_R + fork_seed)

    # Build burst step set for Phase W
    burst_step_set: set[int] = set()
    if phase == "W":
        for bstart, bend in burst_windows:
            burst_step_set.update(range(bstart, bend))

    # Oracle burst step set (exact freeze during burst windows)
    oracle_burst_set: set[int] = burst_step_set if arm_mode == "oracle" else set()

    # Per-step storage
    expressed_arr = np.empty(n_steps, dtype=np.int32)
    true_pos_arr = np.empty(n_steps, dtype=np.int32)
    obs_arr = np.empty(n_steps, dtype=np.int32)
    state_arr = np.empty(n_steps, dtype=np.int32)  # 0=NORMAL, 1=RESIST

    # v trajectory (coarse snapshot every EVAL steps regardless of state)
    v_traj: list[np.ndarray] = []

    # Fine v trajectory for trigger machinery (appended every FINE_EVAL steps, NORMAL only)
    # CLEARED (along with mismatch_history) on any release/concession
    v_fine: list[np.ndarray] = []
    k_since_reset: int = 0

    # Mismatch history for the controller (CLEARED on release/concession)
    mismatch_history: list[float] = []

    # Delta 2: floor guard
    m_bar_floor: float | None = None

    # Events list
    events: list[dict] = []

    # Phase W diagnostics
    burst_preburst_fav: list[int | None] = [None] * len(burst_windows)
    burst_onset_color: list[int | None] = [None] * len(burst_windows)
    current_burst_color: int | None = None
    current_burst_idx: int | None = None

    # (b) exogenous color: computed at step 6000, reused for all bursts
    exogenous_attack_color: int | None = None

    # Phase R state
    regime_color: int | None = None
    regime_rng_active = False

    # Phase R: m_bar tracking for contamination diagnostic
    mbar_phase_r_snapshots: list[float] = []

    # Freeze-gate state machine variables (for freeze arms)
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

    # Refractory counter
    checks_since_release: int = REFRACTORY_CHECKS  # start armed

    # Determine if this arm uses the freeze state machine
    is_freeze_arm = arm_mode in ("freeze_evidence", "oracle") or (
        isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time"
    )

    # Value-dynamics gap/TV diagnostics per burst (Phase W)
    gap_start: list[float | None] = [None] * len(burst_windows)
    gap_end: list[float | None] = [None] * len(burst_windows)
    d_b: list[float | None] = [None] * len(burst_windows)
    tv_b: list[float | None] = [None] * len(burst_windows)
    pi_at_burst_start: list[np.ndarray | None] = [None] * len(burst_windows)

    # Per-burst head_dose: v[attack_color] accumulated only in unfrozen head steps
    head_dose: list[float] = [0.0] * len(burst_windows)
    # Per-burst frozen-plateau gap: gap at first fully-frozen snapshot
    gap_plateau: list[float | None] = [None] * len(burst_windows)
    # Per-burst: track if we are in the head (unfrozen) phase
    _burst_head_active: list[bool] = [False] * len(burst_windows)
    _burst_first_freeze_step: list[int | None] = [None] * len(burst_windows)
    _burst_plateau_v_prev: list[np.ndarray | None] = [None] * len(burst_windows)
    _burst_plateau_found: list[bool] = [False] * len(burst_windows)

    # Per-burst expr_frac (where [bend+1500, bend+2000) fits before next burst)
    per_burst_expr_frac: list[float | None] = [None] * len(burst_windows)

    # Settle TV: track TV(pi) over [5000, 6000) for logging/flagging
    settle_tv_val: float | None = None
    settle_pi_5000: np.ndarray | None = None

    A_hat_start = c._A_hat().copy()

    global_step = 0

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(CHUNK_SIZE):
            t = global_step

            # ----------------------------------------------------------------
            # (b) exogenous_fixed color: capture argmin(v) at step 6000
            # ----------------------------------------------------------------
            if phase == "W" and color_mode == "exogenous_fixed" and t == 6000:
                exogenous_attack_color = int(np.argmin(v))

            # ----------------------------------------------------------------
            # Settle TV: capture pi at step 5000
            # ----------------------------------------------------------------
            if t == 5000:
                settle_pi_5000 = pi_of(v.copy())
            if t == 6000 and settle_pi_5000 is not None:
                settle_tv_val = tv(settle_pi_5000, pi_of(v.copy()))

            # ----------------------------------------------------------------
            # Phase W: burst context management
            # ----------------------------------------------------------------
            in_burst = False
            burst_idx_now: int | None = None

            if phase == "W":
                in_burst = t in burst_step_set
                for bi, (bstart, bend) in enumerate(burst_windows):
                    if bstart <= t < bend:
                        burst_idx_now = bi
                        break

                # Burst onset: record pre-burst favorite, set burst color
                if burst_idx_now is not None and t == burst_windows[burst_idx_now][0]:
                    pre_fav = int(np.argmax(v))
                    burst_preburst_fav[burst_idx_now] = pre_fav

                    # (b) color_mode determines attack color
                    if color_mode == "endogenous":
                        bc = int(np.argmin(v))
                    else:  # exogenous_fixed
                        bc = exogenous_attack_color
                        if bc is None:
                            bc = int(np.argmin(v))  # fallback (shouldn't happen)

                    burst_onset_color[burst_idx_now] = bc
                    current_burst_color = bc
                    current_burst_idx = burst_idx_now
                    # Gap diagnostic: snapshot at onset
                    pi_at_burst_start[burst_idx_now] = pi_of(v.copy())
                    gap_start[burst_idx_now] = float(v[pre_fav] - v[bc])
                    # Head dose tracking: start head for this burst
                    _burst_head_active[burst_idx_now] = True
                    _burst_first_freeze_step[burst_idx_now] = None
                    _burst_plateau_found[burst_idx_now] = False

                # Burst end: record gap_end and TV diagnostics, clear context
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

            # ----------------------------------------------------------------
            # Phase R: permanent relocation onset
            # ----------------------------------------------------------------
            if phase == "R" and t == PHASE_R_START:
                regime_color = int(np.argmin(v))
                regime_rng_active = True

            in_regime = phase == "R" and regime_rng_active and t >= PHASE_R_START

            # ----------------------------------------------------------------
            # Observe
            # ----------------------------------------------------------------
            obs = int(base_cmap[c.true_pos])
            obs_arr[t] = obs
            true_pos_arr[t] = c.true_pos

            # ----------------------------------------------------------------
            # A_hat pre-step
            # ----------------------------------------------------------------
            A_hat = c._A_hat()

            # ----------------------------------------------------------------
            # Belief update
            # ----------------------------------------------------------------
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom_qs = qs_updated.sum()
            if denom_qs > 0:
                qs_updated = qs_updated / denom_qs
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # ----------------------------------------------------------------
            # Dirichlet count update
            # ----------------------------------------------------------------
            c.pA[obs, :] += qs_updated

            # ----------------------------------------------------------------
            # Predictability weight
            # ----------------------------------------------------------------
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            w = math.exp(-h_predicted)

            # ----------------------------------------------------------------
            # Freeze-gate RESIST logic: check BEFORE value update
            # Oracle: freeze exactly during burst windows (e)
            # ----------------------------------------------------------------
            if arm_mode == "oracle":
                # RESIST (frozen) iff in exact burst window
                resist_now = (phase == "W") and (t in oracle_burst_set)
                if resist_now:
                    # v FROZEN: skip decay AND write
                    state_arr[t] = 1
                    # Action / Move
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
                # State machine for n4_freeze and H-arms
                if freeze_state == "RESIST":
                    # v FROZEN this step: skip decay AND write
                    frozen_steps += 1
                    blocked_w_by_color[obs] += w
                    c_star_now = int(np.argmax(blocked_w_by_color))
                    E_blocked_now = float(blocked_w_by_color[c_star_now])

                    # Directional pressure score
                    e_obs_vec = np.zeros(n_colors)
                    e_obs_vec[obs] = 1.0
                    e_cstar_vec = np.zeros(n_colors)
                    e_cstar_vec[c_star_now] = 1.0
                    dir_score = float(np.dot(e_obs_vec - pi_ref, e_cstar_vec - pi_ref))
                    directional_pressure_acc += dir_score

                    state_arr[t] = 1

                    # Action / Move
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

                        # Pressure statistic
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

                        if calm_count >= RELEASE_CALM_SNAPS:
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
                            trigger_latency: int | None = None
                            for bi, (bstart, bend) in enumerate(burst_windows):
                                if bstart <= entry_step < bend:
                                    trigger_latency = entry_step - bstart
                                    break

                            events.append({
                                "entry_step": entry_step,
                                "exit_step": t_now,
                                "label": release_label,
                                "E_blocked": E_blocked_final,
                                "c_star": c_star_final,
                                "frozen_steps": frozen_steps,
                                "directional_pressure_acc": directional_pressure_acc,
                                "trigger_latency": trigger_latency,
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
                    # freeze_state == "NORMAL": fall through to standard value update
                    state_arr[t] = 0

            else:
                state_arr[t] = 0

            # ----------------------------------------------------------------
            # Value update (NORMAL state for all arms reaching here)
            # ----------------------------------------------------------------
            if arm_mode == "baseline":
                g_t = 1.0
            elif arm_mode in ("freeze_evidence",) or (
                isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time"
            ):
                g_t = 1.0
            elif arm_mode == "oracle":
                g_t = 1.0
            else:
                g_t = float(arm_mode)  # fixed write-gain arms

            v *= LAMBDA
            v[obs] += g_t * w
            c.value_counts[obs] += g_t * w

            # ----------------------------------------------------------------
            # Expressed preference
            # ----------------------------------------------------------------
            expressed = int(np.argmax(v))
            expressed_arr[t] = expressed

            # ----------------------------------------------------------------
            # Action / Move
            # ----------------------------------------------------------------
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

            # ----------------------------------------------------------------
            # Coarse snapshot
            # ----------------------------------------------------------------
            if global_step % EVAL == 0:
                v_traj.append(v.copy())

            # ----------------------------------------------------------------
            # Fine snapshot (NORMAL state only) — trigger machinery
            # ----------------------------------------------------------------
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

                            if (
                                trigger_denom is not None
                                and m_k > 0
                                and checks_since_release >= REFRACTORY_CHECKS
                                and (m_k / max(trigger_denom, 1e-12)) >= THETA
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

    # A_hat drift
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # Compute per-burst expr_frac where window fits before next burst
    n_bursts = len(burst_windows)
    for bi in range(n_bursts):
        bstart, bend = burst_windows[bi]
        win_start = bend + DEFENSE_WINDOW_OFFSET_START
        win_end = bend + DEFENSE_WINDOW_OFFSET_END
        # Check if window fits before next burst (or end of session)
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
# Defense criterion for Exp 184
# Primary: expr_frac(attack color) < 0.5 over [last_bend+1500, last_bend+2000)
# ---------------------------------------------------------------------------

def compute_defense(
    expressed_arr: np.ndarray,
    burst_windows: list,
    burst_onset_color: list,
    n_steps: int,
) -> tuple[bool, float | None, int | None]:
    """Return (defense_passed, final_expr_frac, attack_color).

    Uses the LAST burst's window for the primary criterion.
    """
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
    """Gap at final coarse snapshot: v_fav - v_attack (if v_traj available)."""
    if len(v_traj) == 0:
        return None
    last_v = v_traj[-1]
    fav = int(np.argmax(last_v))
    attack = int(np.argmin(last_v))
    return float(last_v[fav] - last_v[attack])


# ---------------------------------------------------------------------------
# Equivalence gate
# ---------------------------------------------------------------------------

FLOAT_ATOL = 1e-9


def load_committed_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_equivalence_gate(mirro_root, base_cmap, n_colors, committed_rows_path: Path) -> tuple[bool, str]:
    """Run the equivalence gate: reproduce exp183 baseline x s229 and H1200 x s229.

    Returns (gate_pass, detail_string).
    """
    # Load committed rows for reference
    committed_rows = load_committed_rows(committed_rows_path)
    committed_w = {}
    for row in committed_rows:
        if row.get("phase") == "W":
            key = (row["arm"], row["fork_seed"], row["burst_idx"])
            committed_w[key] = row

    detail_lines = []
    all_pass = True

    arm_lookup_exp183 = {
        "baseline": ("baseline", "baseline"),
        "H1200": ("H1200", ("freeze_time", 1200)),
    }

    for arm_name in EXP183_GATE_ARMS:
        _, arm_mode = arm_lookup_exp183[arm_name]
        for seed in EXP183_GATE_SEEDS:
            print(f"  [gate] arm={arm_name} seed={seed} ...", flush=True)
            root = copy.deepcopy(mirro_root)
            root._state_dir = None

            rr = run_fork_schedule(
                mirro=root,
                fork_seed=seed,
                base_cmap=base_cmap,
                n_colors=n_colors,
                arm_name=arm_name,
                arm_mode=arm_mode,
                phase="W",
                burst_windows=EXP183_BURST_WINDOWS,  # (a) exp183 windows
                color_mode="endogenous",              # (b) endogenous: argmin at each burst onset
                reloc_rng_seed=BURST_SEED_OFFSET_W_EXP183 + seed,  # (c) exp183 reloc rng
                n_steps=EXP183_N_STEPS,               # (d) exp183 N_STEPS
            )

            # Compare vs committed rows for each burst
            session_pass = True
            for bi in range(len(EXP183_BURST_WINDOWS)):
                key = (arm_name, seed, bi)
                if key not in committed_w:
                    detail_lines.append(f"  MISS: committed row not found for {key}")
                    session_pass = False
                    all_pass = False
                    continue

                cr = committed_w[key]
                # Compare scalar quantities
                checks = [
                    ("gap_start", rr["gap_start"][bi], cr["gap_start"]),
                    ("gap_end", rr["gap_end"][bi], cr["gap_end"]),
                    ("d_b", rr["d_b"][bi], cr["d_b"]),
                    ("tv_b", rr["tv_b"][bi], cr["tv_b"]),
                    ("recovered", rr["per_burst_expr_frac"][bi] is not None and
                     float(rr["per_burst_expr_frac"][bi]) < DEFENSE_FRAC_THRESH
                     if rr["per_burst_expr_frac"][bi] is not None else None,
                     cr["recovered"]),
                    ("n_events", len(rr["events"]), cr["n_events"]),
                ]

                # Also check the exp183 burst_recovered verbatim:
                # use expr_frac < 0.5 on the per-burst window
                bstart, bend = EXP183_BURST_WINDOWS[bi]
                win_start = bend + 1500
                win_end = bend + 2000
                if win_end <= EXP183_N_STEPS and rr["burst_onset_color"][bi] is not None:
                    bc_exp183 = rr["burst_onset_color"][bi]
                    gate_frac = float(np.mean(rr["expressed_arr"][win_start:win_end] == bc_exp183))
                    gate_recovered = gate_frac < 0.5
                else:
                    gate_recovered = None

                checks[4] = ("recovered", gate_recovered, cr["recovered"])

                for field_name, got, expected in checks:
                    if got is None and expected is None:
                        continue
                    if isinstance(got, (int, float)) and isinstance(expected, (int, float)):
                        ok = abs(float(got) - float(expected)) <= FLOAT_ATOL
                    elif isinstance(got, bool) and isinstance(expected, bool):
                        ok = got == expected
                    elif isinstance(got, int) and isinstance(expected, int):
                        ok = got == expected
                    elif type(got) == type(expected):
                        ok = got == expected
                    else:
                        ok = str(got) == str(expected)

                    if not ok:
                        detail_lines.append(
                            f"  MISMATCH arm={arm_name} s={seed} bi={bi} field={field_name}: "
                            f"got={got!r} expected={expected!r}"
                        )
                        session_pass = False
                        all_pass = False

                # Compare events
                got_evs = rr["events"]
                exp_evs = cr.get("events_summary", [])
                if len(got_evs) != len(exp_evs):
                    detail_lines.append(
                        f"  MISMATCH arm={arm_name} s={seed} bi={bi} event count: "
                        f"got={len(got_evs)} expected={len(exp_evs)}"
                    )
                    session_pass = False
                    all_pass = False
                else:
                    for ei, (ge, ee) in enumerate(zip(got_evs, exp_evs)):
                        for ef in ["label", "entry_step", "frozen_steps", "c_star", "trigger_latency"]:
                            gv = ge.get(ef)
                            ev = ee.get(ef)
                            if gv != ev:
                                detail_lines.append(
                                    f"  MISMATCH arm={arm_name} s={seed} bi={bi} "
                                    f"event[{ei}].{ef}: got={gv!r} expected={ev!r}"
                                )
                                session_pass = False
                                all_pass = False
                        # E_blocked: float atol
                        gv = ge.get("E_blocked")
                        ev = ee.get("E_blocked")
                        if gv is not None and ev is not None:
                            if abs(float(gv) - float(ev)) > FLOAT_ATOL:
                                detail_lines.append(
                                    f"  MISMATCH arm={arm_name} s={seed} bi={bi} "
                                    f"event[{ei}].E_blocked: got={gv} expected={ev} "
                                    f"diff={abs(float(gv)-float(ev)):.2e}"
                                )
                                session_pass = False
                                all_pass = False

            status = "PASS" if session_pass else "FAIL"
            detail_lines.append(f"  gate arm={arm_name} seed={seed}: {status}")
            print(f"  [gate] arm={arm_name} seed={seed}: {status}", flush=True)

    overall = "PASS" if all_pass else "FAIL"
    detail_str = f"EQUIVALENCE GATE: {overall}\n" + "\n".join(detail_lines)
    return all_pass, detail_str


# ---------------------------------------------------------------------------
# Phase-R revision pass check
# ---------------------------------------------------------------------------

def compute_revision_results(r_results_by_arm_seed: dict) -> dict:
    """Compute latency for each (arm, seed) in R sessions.

    Returns dict keyed by (arm, seed) -> latency (int or None).
    """
    lats = {}
    for (arm, seed), rr in r_results_by_arm_seed.items():
        rc = rr["regime_color"]
        if rc is None:
            lats[(arm, seed)] = None
        else:
            lats[(arm, seed)] = phase_r_latency(rr["expressed_arr"], rc, N_STEPS_PHASE_R)
    return lats


def revision_pass(
    h_arm: str,
    mode: str,
    tolerance: int,
    latencies: dict,  # (arm, seed) -> latency | None
    seeds: list,
) -> bool:
    """revision_pass(H, mode) = latency <= baseline+tolerance in >=2/3 seeds."""
    n_pass = 0
    for seed in seeds:
        bl_lat = latencies.get(("baseline", seed))
        arm_lat = latencies.get((h_arm, seed))
        if arm_lat is None:
            continue  # never adopted — doesn't pass
        if bl_lat is None:
            # baseline never adopted — tolerance undefined; treat as fail
            continue
        if arm_lat <= bl_lat + tolerance:
            n_pass += 1
    return n_pass >= MIN_PASS_FRACTION


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def to_plain(obj):
    """Recursively convert numpy types to plain Python types."""
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


def defense_pass_for_cell(
    defense_by_arm_seed: dict,   # (arm, seed) -> bool
    arm_name: str,
    seeds: list,
) -> bool:
    """defense_pass(arm, cell) = >=2/3 seeds."""
    n_pass = sum(1 for seed in seeds if defense_by_arm_seed.get((arm_name, seed), False))
    return n_pass >= MIN_PASS_FRACTION


# ---------------------------------------------------------------------------
# JSON row writer
# ---------------------------------------------------------------------------

def write_w_row(fh, cell: dict, arm_name: str, seed: int, rr: dict, n_steps: int):
    """Write one W-session row to JSONL."""
    defense, final_expr_frac, attack_color = compute_defense(
        rr["expressed_arr"],
        rr.get("burst_windows_used", []),
        rr["burst_onset_color"],
        n_steps,
    )

    # final gap from v_traj
    final_gap = compute_final_gap(rr["v_traj"], rr["expressed_arr"], n_steps)

    # per-burst diagnostics
    burst_windows = rr.get("burst_windows_used", [])
    per_burst = []
    for bi, (bstart, bend) in enumerate(burst_windows):
        pb = {
            "bi": bi,
            "gap_start": rr["gap_start"][bi],
            "gap_plateau": None,   # L13 plateau read — not implemented in step-level yet
            "gap_end": rr["gap_end"][bi],
            "head_dose": None,     # diagnostic only
            "expr_frac_or_null": rr["per_burst_expr_frac"][bi],
        }
        per_burst.append(pb)

    # settle TV
    settle_tv = rr.get("settle_tv")
    flags = []
    if settle_tv is not None and settle_tv > SETTLE_TV_FLAG_MAX:
        flags.append(f"settle_tv_high:{settle_tv:.4f}")
    if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
        flags.append(f"pc1_fail:ahat_drift={rr['ahat_drift']:.4f}")

    row = {
        "exp": 184,
        "kind": "W",
        "cell": {"L": cell["L"], "K": cell["K"], "G": cell["G"], "idx": cell["idx"]},
        "arm": arm_name,
        "seed": int(seed),
        "attack_color": to_plain(attack_color),
        "pre_fav": to_plain(rr["burst_preburst_fav"][0]) if rr["burst_preburst_fav"] else None,
        "defense": bool(defense),
        "final_expr_frac": to_plain(final_expr_frac),
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
        "settle_tv": to_plain(settle_tv),
        "flags": flags,
    }
    fh.write(json.dumps(row) + "\n")


def write_r_row(fh, arm_name: str, seed: int, rr: dict):
    """Write one R-session row to JSONL."""
    rc = rr["regime_color"]
    latency = None
    if rc is not None:
        latency = phase_r_latency(rr["expressed_arr"], rc, N_STEPS_PHASE_R)

    flags = []
    if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
        flags.append(f"pc1_fail:ahat_drift={rr['ahat_drift']:.4f}")

    row = {
        "exp": 184,
        "kind": "R",
        "arm": arm_name,
        "seed": int(seed),
        "latency": to_plain(latency),
        "n_events": int(len(rr["events"])),
        "ahat_drift": float(rr["ahat_drift"]),
        "flags": flags,
    }
    fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 80)
    print("Exp 184 — N4 Fixed-H Squeeze Map (EXPLORATORY)")
    print("PRE-REGISTERED in loop/directions/identity-n4-crack.md (commit c220d2c)")
    print("Seeds: 240-242; Grid: 80 cells; Arms: 11W + 10R")
    print("=" * 80)
    print()

    # ---- Spine safety (L14) ----
    mirro = Creature.load("creature/state/mirro")
    mirro_root = copy.deepcopy(mirro)
    mirro_root._state_dir = None

    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    n_cells_world = mirro.world.n_cells
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={n_colors}, n_cells={n_cells_world}")

    # Assert argmin != argmax
    vc = mirro.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v_spine = (vc / vc_sum) * INIT_MASS
    else:
        v_spine = np.ones(n_colors) * (INIT_MASS / n_colors)
    spine_argmax = int(np.argmax(v_spine))
    spine_argmin = int(np.argmin(v_spine))
    assert spine_argmax != spine_argmin, "argmin == argmax at spine — degenerate"
    print(f"Spine standing favorite: color {spine_argmax}, attack candidate: color {spine_argmin}")
    print()

    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows_path = out_dir / "exp184_rows.json"
    out_txt_path = out_dir / "exp184.txt"
    committed_rows_path = out_dir / "exp183_rows.json"

    # ====================================================================
    # STEP 1: EQUIVALENCE GATE
    # ====================================================================
    print("=" * 80)
    print("STEP 1: EQUIVALENCE GATE")
    print("Reproducing exp183 (baseline x s229) and (H1200 x s229) with endogenous color mode")
    print("=" * 80)
    t_gate = time.time()

    gate_pass, gate_detail = run_equivalence_gate(
        mirro_root, base_cmap, n_colors, committed_rows_path
    )

    print()
    print(gate_detail)
    print(f"Gate time: {time.time()-t_gate:.1f}s")
    print()

    if not gate_pass:
        print("EQUIVALENCE GATE FAIL — aborting. Grid NOT run.")
        print("Fix the runner to match exp183 exactly before proceeding.")
        with open(out_txt_path, "w") as f:
            f.write("EQUIVALENCE GATE FAIL\n\n")
            f.write(gate_detail + "\n")
        return

    print("EQUIVALENCE GATE PASS — proceeding to smoke and grid.")
    print()

    # ====================================================================
    # STEP 2: SMOKE (cell L=800, K=3, G=2400, all 11 arms, seed 240)
    # ====================================================================
    print("=" * 80)
    print("STEP 2: SMOKE — cell (L=800,K=3,G=2400), all 11 arms, seed 240")
    print("=" * 80)
    smoke_cell = None
    for c in CELLS:
        if c["L"] == 800 and c["K"] == 3 and c["G"] == 2400:
            smoke_cell = c
            break
    assert smoke_cell is not None, "Smoke cell not found"

    smoke_burst_windows = []
    cur = 6000
    for ki in range(smoke_cell["K"]):
        bstart = cur
        bend = cur + smoke_cell["L"]
        smoke_burst_windows.append((bstart, bend))
        cur = bend
        if ki < smoke_cell["K"] - 1:
            cur += smoke_cell["G"]
    smoke_n_steps = smoke_cell["n_steps"]

    print(f"  cell_idx={smoke_cell['idx']}, n_steps={smoke_n_steps}")
    print(f"  burst_windows: {smoke_burst_windows}")
    print()

    smoke_results = {}
    t_smoke = time.time()
    for arm_name, arm_mode in W_ARMS:
        root = copy.deepcopy(mirro_root)
        root._state_dir = None
        rr = run_fork_schedule(
            mirro=root,
            fork_seed=SMOKE_SEED,
            base_cmap=base_cmap,
            n_colors=n_colors,
            arm_name=arm_name,
            arm_mode=arm_mode,
            phase="W",
            burst_windows=smoke_burst_windows,
            color_mode="exogenous_fixed",
            reloc_rng_seed=190_000 + 1000 * smoke_cell["idx"] + SMOKE_SEED,
            n_steps=smoke_n_steps,
        )
        rr["burst_windows_used"] = smoke_burst_windows
        smoke_results[arm_name] = rr
        defense, frac, attack_color = compute_defense(
            rr["expressed_arr"], smoke_burst_windows, rr["burst_onset_color"], smoke_n_steps
        )
        frac_s = f"{frac:.3f}" if frac is not None else "N/A"
        print(f"  arm={arm_name:12s}  defense={'PASS' if defense else 'fail'}  "
              f"frac={frac_s}  "
              f"attack_color={attack_color}  n_events={len(rr['events'])}  "
              f"ahat_drift={rr['ahat_drift']:.5f}")
    print(f"  Smoke time: {time.time()-t_smoke:.1f}s")
    print()

    # ====================================================================
    # STEP 3: FULL GRID W SESSIONS (80 cells x 11 arms x 3 seeds)
    # ====================================================================
    print("=" * 80)
    print("STEP 3: FULL GRID W SESSIONS (80 cells x 11 arms x 3 seeds = 2640 sessions)")
    print("=" * 80)

    # Storage: (cell_idx, arm_name, seed) -> result dict
    w_results: dict = {}
    # defense_by_cell_arm_seed: (cell_idx, arm_name, seed) -> bool
    defense_results: dict = {}
    # Per-cell attack_color (fixed at step 6000 for each seed)
    attack_color_by_cell_seed: dict = {}

    t_grid = time.time()
    t_first_cell_done = None

    pc1_failures_w = []
    settle_tv_flags = []
    argmin_eq_argmax_flags = []

    for ci, cell in enumerate(CELLS):
        L, K, G = cell["L"], cell["K"], cell["G"]
        cell_idx = cell["idx"]
        n_steps = cell["n_steps"]

        # Build burst windows for this cell
        burst_windows = []
        cur = 6000
        for ki in range(K):
            bstart = cur
            bend = cur + L
            burst_windows.append((bstart, bend))
            cur = bend
            if ki < K - 1:
                cur += G

        t_cell = time.time()

        for arm_name, arm_mode in W_ARMS:
            for seed in SEEDS:
                root = copy.deepcopy(mirro_root)
                root._state_dir = None

                reloc_rng_seed = 190_000 + 1000 * cell_idx + seed

                rr = run_fork_schedule(
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
                )
                rr["burst_windows_used"] = burst_windows

                # PC1
                if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
                    pc1_failures_w.append(
                        f"PC1 FAIL W: cell={cell_idx} arm={arm_name} seed={seed} "
                        f"ahat_drift={rr['ahat_drift']:.4f}"
                    )

                # Settle TV flag
                settle_tv = rr.get("settle_tv")
                if settle_tv is not None and settle_tv > SETTLE_TV_FLAG_MAX:
                    settle_tv_flags.append(
                        f"settle_tv_high cell={cell_idx} arm={arm_name} seed={seed}: "
                        f"{settle_tv:.4f}"
                    )

                # Defense
                defense, frac, attack_color = compute_defense(
                    rr["expressed_arr"], burst_windows, rr["burst_onset_color"], n_steps
                )
                defense_results[(cell_idx, arm_name, seed)] = bool(defense)

                # Record attack color (per cell per seed — should be same across arms)
                if (cell_idx, seed) not in attack_color_by_cell_seed:
                    attack_color_by_cell_seed[(cell_idx, seed)] = attack_color

                # Store result
                w_results[(cell_idx, arm_name, seed)] = rr

        elapsed = time.time() - t_grid
        cell_time = time.time() - t_cell

        if ci == 0:
            t_first_cell_done = time.time()
            projected = (t_first_cell_done - t_grid) * len(CELLS)
            print(f"  cell {ci:02d} (L={L},K={K},G={G},idx={cell_idx}) done in {cell_time:.1f}s "
                  f"| projected total: {projected/60:.1f} min", flush=True)
            if projected > 45 * 60:
                print(f"  WARNING: projected runtime {projected/60:.1f} min > 45 min. "
                      f"Continuing as predeclared — do NOT subsample.", flush=True)
        elif (ci + 1) % 10 == 0 or ci == len(CELLS) - 1:
            rate = elapsed / (ci + 1)
            remaining = rate * (len(CELLS) - ci - 1)
            print(f"  cell {ci:02d}/{len(CELLS)-1} (L={L},K={K},G={G}) done | "
                  f"elapsed {elapsed:.0f}s | ETA {remaining:.0f}s", flush=True)

    t_grid_done = time.time()
    print(f"Grid W done: {(t_grid_done - t_grid)/60:.1f} min total")
    print()

    # ====================================================================
    # STEP 4: PHASE-R SESSIONS (10 arms x 3 seeds = 30 sessions)
    # ====================================================================
    print("=" * 80)
    print("STEP 4: PHASE-R SESSIONS (10 arms x 3 seeds = 30 sessions)")
    print("=" * 80)

    r_results_by_arm_seed: dict = {}
    pc1_failures_r = []

    t_r = time.time()
    for arm_name, arm_mode in R_ARMS:
        for seed in SEEDS:
            root = copy.deepcopy(mirro_root)
            root._state_dir = None

            rr = run_fork_schedule(
                mirro=root,
                fork_seed=seed,
                base_cmap=base_cmap,
                n_colors=n_colors,
                arm_name=arm_name,
                arm_mode=arm_mode,
                phase="R",
                burst_windows=[],         # no bursts in Phase R
                color_mode="endogenous",  # doesn't matter (no bursts)
                reloc_rng_seed=BURST_SEED_OFFSET_R + seed,  # exp183 verbatim
                n_steps=N_STEPS_PHASE_R,
            )
            rr["burst_windows_used"] = []
            r_results_by_arm_seed[(arm_name, seed)] = rr

            if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
                pc1_failures_r.append(
                    f"PC1 FAIL R: arm={arm_name} seed={seed} "
                    f"ahat_drift={rr['ahat_drift']:.4f}"
                )

            rc = rr["regime_color"]
            lat = None
            if rc is not None:
                lat = phase_r_latency(rr["expressed_arr"], rc, N_STEPS_PHASE_R)
            print(f"  arm={arm_name:12s} seed={seed}  regime_color={rc}  "
                  f"latency={lat}  ahat_drift={rr['ahat_drift']:.5f}", flush=True)

    print(f"Phase-R done: {time.time()-t_r:.1f}s")
    print()

    # ====================================================================
    # STEP 5: ANALYSIS + OUTPUTS
    # ====================================================================
    print("=" * 80)
    print("STEP 5: ANALYSIS")
    print("=" * 80)

    # Compute revision latencies
    r_latencies = compute_revision_results(r_results_by_arm_seed)

    # For each (H_arm, mode), compute revision_pass over seeds
    rev_pass: dict = {}  # (h_arm, mode_name) -> bool
    for h in FIXED_HORIZONS:
        h_arm = f"H{h}"
        for mode_name, tolerance in REVISION_MODES:
            rev_pass[(h_arm, mode_name)] = revision_pass(
                h_arm, mode_name, tolerance, r_latencies, SEEDS
            )

    # defense_pass_grid: (cell_idx, arm_name) -> bool
    defense_pass_grid: dict = {}
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        for arm_name, _ in W_ARMS:
            n_pass = sum(
                1 for seed in SEEDS
                if defense_results.get((cell_idx, arm_name, seed), False)
            )
            defense_pass_grid[(cell_idx, arm_name)] = n_pass >= MIN_PASS_FRACTION

    # exists_H_both(cell, mode): any H such that defense_pass(H, cell) AND revision_pass(H, mode)
    exists_h_both: dict = {}  # (cell_idx, mode_name) -> bool
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        for mode_name, tolerance in REVISION_MODES:
            found = False
            for h in FIXED_HORIZONS:
                h_arm = f"H{h}"
                def_ok = defense_pass_grid.get((cell_idx, h_arm), False)
                rev_ok = rev_pass.get((h_arm, mode_name), False)
                if def_ok and rev_ok:
                    found = True
                    break
            exists_h_both[(cell_idx, mode_name)] = found

    # Candidate crack cells: NOT exists_H_both while oracle defends (>=2/3 seeds)
    oracle_pass: dict = {}  # cell_idx -> bool
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        n_pass = sum(
            1 for seed in SEEDS
            if defense_results.get((cell_idx, "oracle", seed), False)
        )
        oracle_pass[cell_idx] = n_pass >= MIN_PASS_FRACTION

    candidate_cells: dict = {}  # mode_name -> list of (cell_idx, L, K, G)
    for mode_name, _ in REVISION_MODES:
        candidates = []
        for ci, cell in enumerate(CELLS):
            cell_idx = cell["idx"]
            if (not exists_h_both.get((cell_idx, mode_name), True)) and oracle_pass.get(cell_idx, False):
                candidates.append((cell_idx, cell["L"], cell["K"], cell["G"]))
        candidate_cells[mode_name] = candidates

    # P1: H-invariance check
    # Among H-arms with H >= L-75, defense outcome should be H-INVARIANT
    # Report cells where defense improves monotonically with H in that stratum
    p1_f1_cells = []  # cells where defense improves monotonically with H (F1 evidence)
    for ci, cell in enumerate(CELLS):
        cell_idx = cell["idx"]
        L = cell["L"]
        # Stratum: H >= L-75
        stratum_arms = [(f"H{h}", h) for h in FIXED_HORIZONS if h >= L - 75]
        if len(stratum_arms) < 2:
            continue
        # Get defense counts for each H in stratum
        def_counts = []
        for h_arm, h in stratum_arms:
            n_pass = sum(
                1 for seed in SEEDS
                if defense_results.get((cell_idx, h_arm, seed), False)
            )
            def_counts.append((h, n_pass))
        # Check if monotonically increasing (F1: improvement = worse for dose model)
        is_monotone_inc = all(
            def_counts[i][1] <= def_counts[i+1][1]
            for i in range(len(def_counts)-1)
        ) and def_counts[0][1] < def_counts[-1][1]
        if is_monotone_inc:
            p1_f1_cells.append((cell_idx, cell["L"], cell["K"], cell["G"], def_counts))

    # P2: (L=2400, tight) interval-empty check
    # defense needs H >~ 2325; tight allows H <~ 1650; predicted NOT exists_H_both
    p2_l2400_tight_results = []
    for ci, cell in enumerate(CELLS):
        if cell["L"] != 2400:
            continue
        cell_idx = cell["idx"]
        exists = exists_h_both.get((cell_idx, "tight"), False)
        p2_l2400_tight_results.append({
            "cell_idx": cell_idx, "K": cell["K"], "G": cell["G"],
            "exists_H_both_tight": exists,
        })
    p2_interval_empty = all(not r["exists_H_both_tight"] for r in p2_l2400_tight_results)

    # P3: oracle defends every cell
    p3_oracle_failures = [
        (cell["idx"], cell["L"], cell["K"], cell["G"])
        for cell in CELLS
        if not oracle_pass.get(cell["idx"], False)
    ]

    # F4: degenerate map check
    # Degenerate if trigger never fires OR everything passes everywhere
    n_events_total = sum(
        len(w_results[(cell["idx"], arm_name, seed)]["events"])
        for cell in CELLS
        for arm_name, _ in W_ARMS
        if arm_name not in ("baseline", "oracle")
        for seed in SEEDS
        if (cell["idx"], arm_name, seed) in w_results
    )
    all_cells_pass_any_mode = all(
        exists_h_both.get((cell["idx"], mode_name), False)
        for cell in CELLS
        for mode_name, _ in REVISION_MODES
    )
    f4_trigger_silent = n_events_total == 0
    f4_degenerate = f4_trigger_silent or all_cells_pass_any_mode

    # PC1 summary
    pc1_failures_all = pc1_failures_w + pc1_failures_r
    pc1_pass = len(pc1_failures_all) == 0

    # K=1 G-degeneracy note: K=1 cells are G-replicates
    k1_cells = [c for c in CELLS if c["K"] == 1]
    k1_defense_by_L_arm_seed: dict = {}
    for cell in k1_cells:
        for arm_name, _ in W_ARMS:
            for seed in SEEDS:
                key_lg = (cell["L"], arm_name, seed)
                defense_val = defense_results.get((cell["idx"], arm_name, seed), False)
                if key_lg not in k1_defense_by_L_arm_seed:
                    k1_defense_by_L_arm_seed[key_lg] = []
                k1_defense_by_L_arm_seed[key_lg].append(defense_val)

    elapsed_total = time.time() - t_start
    runtime_min = elapsed_total / 60.0

    print(f"Total runtime: {runtime_min:.1f} min")
    print()

    # ====================================================================
    # Write JSONL rows
    # ====================================================================
    with open(out_rows_path, "w") as fh:
        # W rows
        for ci, cell in enumerate(CELLS):
            cell_idx = cell["idx"]
            burst_windows = []
            cur = 6000
            for ki in range(cell["K"]):
                bstart = cur
                bend = cur + cell["L"]
                burst_windows.append((bstart, bend))
                cur = bend
                if ki < cell["K"] - 1:
                    cur += cell["G"]

            for arm_name, _ in W_ARMS:
                for seed in SEEDS:
                    if (cell_idx, arm_name, seed) not in w_results:
                        continue
                    rr = w_results[(cell_idx, arm_name, seed)]
                    rr["burst_windows_used"] = burst_windows
                    write_w_row(fh, cell, arm_name, seed, rr, cell["n_steps"])

        # Smoke rows (skip if already in grid — smoke_cell is in grid)
        # (already included above since smoke_cell is in CELLS)

        # R rows
        for arm_name, _ in R_ARMS:
            for seed in SEEDS:
                if (arm_name, seed) in r_results_by_arm_seed:
                    rr = r_results_by_arm_seed[(arm_name, seed)]
                    write_r_row(fh, arm_name, seed, rr)

    print(f"Rows written to {out_rows_path}")

    # ====================================================================
    # Write text analysis
    # ====================================================================
    with open(out_txt_path, "w") as f:
        def p(*args, **kwargs):
            print(*args, **kwargs, file=f)
            print(*args, **kwargs)

        p("=" * 80)
        p("EXP 184 — N4 FIXED-H SQUEEZE MAP (EXPLORATORY)")
        p("PRE-REGISTERED loop/directions/identity-n4-crack.md (commit c220d2c)")
        p(f"Runtime: {runtime_min:.1f} min | Seeds: {SEEDS} | Grid: {len(CELLS)} cells")
        p("=" * 80)
        p()

        p("EQUIVALENCE GATE")
        p("-" * 40)
        p(gate_detail)
        p()

        p("PC1 / PRECONDITIONS")
        p("-" * 40)
        if pc1_pass:
            p("PC1 PASS: all ahat_drift < 0.15")
        else:
            p(f"PC1 FAIL: {len(pc1_failures_all)} sessions")
            for f_msg in pc1_failures_all[:20]:
                p(f"  {f_msg}")
        if settle_tv_flags:
            p(f"Settle TV flags (>0.05, NOT gated): {len(settle_tv_flags)}")
            for msg in settle_tv_flags[:10]:
                p(f"  {msg}")
        else:
            p("Settle TV: no flags (all <= 0.05)")
        p()

        p("SMOKE RESULTS — cell (L=800,K=3,G=2400), seed 240, all 11 arms")
        p("-" * 40)
        for arm_name, _ in W_ARMS:
            rr = smoke_results.get(arm_name)
            if rr is None:
                continue
            defense, frac, attack_color = compute_defense(
                rr["expressed_arr"], smoke_burst_windows, rr["burst_onset_color"],
                smoke_n_steps
            )
            frac_s2 = f"{frac:.3f}" if frac is not None else "N/A"
            p(f"  arm={arm_name:12s}  defense={'PASS' if defense else 'fail'}  "
              f"frac={frac_s2:8s}  "
              f"attack_color={attack_color}  n_events={len(rr['events'])}  "
              f"ahat_drift={rr['ahat_drift']:.5f}")
        p()

        # Phase-R latency table
        p("PHASE-R LATENCY TABLE (10 arms x 3 seeds)")
        p("-" * 40)
        bl_lats = {seed: r_latencies.get(("baseline", seed)) for seed in SEEDS}
        hdr = f"  {'arm':14s} " + " ".join(f"s{s:3d}" for s in SEEDS) + "  [bl:" + "/".join(str(bl_lats[s]) for s in SEEDS) + "]"
        p(hdr)
        for arm_name, _ in R_ARMS:
            lats_str = " ".join(
                f"{str(r_latencies.get((arm_name, s), 'N/A')):>6}"
                for s in SEEDS
            )
            p(f"  {arm_name:14s} {lats_str}")
        p()

        # Revision pass table
        p("REVISION PASS TABLE (H_arm x mode)")
        p("-" * 40)
        p(f"  {'arm':14s} " + " ".join(f"{m:8s}" for m, _ in REVISION_MODES))
        for h in FIXED_HORIZONS:
            h_arm = f"H{h}"
            passes = " ".join(
                f"{'PASS' if rev_pass.get((h_arm, m), False) else 'fail':8s}"
                for m, _ in REVISION_MODES
            )
            p(f"  {h_arm:14s} {passes}")
        p()

        # 80-cell defense maps per mode
        for mode_name, tolerance in REVISION_MODES:
            p(f"EXISTS_H_BOTH MAP — mode={mode_name} (tolerance={tolerance})")
            p(f"  Rows=L, Cols=G; K sub-rows; mark C=crack candidate (oracle defends, no H passes both)")
            p("-" * 60)
            # Header
            p(f"  {'L':6s} {'K':4s} | " + " ".join(f"G={G:5d}" for G in G_VALS))
            p("  " + "-" * 50)

            for L in L_VALS:
                for K in K_VALS:
                    row_parts = []
                    for G in G_VALS:
                        # Find cell
                        cell = next(c for c in CELLS if c["L"]==L and c["K"]==K and c["G"]==G)
                        cell_idx = cell["idx"]
                        exists = exists_h_both.get((cell_idx, mode_name), False)
                        oracle_ok = oracle_pass.get(cell_idx, False)
                        is_candidate = (not exists) and oracle_ok
                        symbol = "C" if is_candidate else ("Y" if exists else "n")
                        row_parts.append(f"{symbol:>7}")
                    p(f"  L={L:5d} K={K:1d} | " + " ".join(row_parts))
            p()

        # P1: H-invariance
        p("P1 — H-INVARIANCE CHECK (H >= L-75 stratum)")
        p("-" * 40)
        if not p1_f1_cells:
            p("P1 HOLDS: no cell shows monotone defense improvement with H in H>=L-75 stratum")
            p("  (consistent with dose model: head dose is H-invariant)")
        else:
            p(f"F1 EVIDENCE: {len(p1_f1_cells)} cell(s) show monotone H improvement:")
            for cell_idx, L, K, G, def_counts in p1_f1_cells[:20]:
                p(f"  cell_idx={cell_idx} L={L} K={K} G={G}: {def_counts}")
        p()

        # P2: (L=2400, tight)
        p("P2 — (L=2400, tight) INTERVAL-EMPTY CHECK")
        p("-" * 40)
        p(f"Predicted: NOT exists_H_both for all L=2400 cells in tight mode")
        p(f"(defense needs H >~ 2325; tight allows H <~ 1650)")
        p()
        for r2 in p2_l2400_tight_results:
            status = "FAIL (exists H)" if r2["exists_H_both_tight"] else "PASS (no H)"
            p(f"  cell_idx={r2['cell_idx']} K={r2['K']} G={r2['G']}: {status}")
        p()
        if p2_interval_empty:
            p("P2 VERDICT: interval-empty for all L=2400 tight cells (prediction holds)")
        else:
            p("P2 VERDICT: some L=2400 tight cell has exists_H_both=True (prediction fails)")
        p()

        # P3: oracle table
        p("P3 — ORACLE TABLE")
        p("-" * 40)
        if not p3_oracle_failures:
            p("P3 PASS: oracle defends all 80 cells at >=2/3 seeds")
        else:
            p(f"F3/CLASS-E FLAGS: oracle fails {len(p3_oracle_failures)} cells:")
            for cell_idx, L, K, G in p3_oracle_failures:
                p(f"  cell_idx={cell_idx} L={L} K={K} G={G}")
        p()

        # Oracle per-cell table
        p("Oracle defense counts (n_seeds_passing / 3 per cell):")
        p(f"  {'L':6s} {'K':4s} | " + " ".join(f"G={G:5d}" for G in G_VALS))
        for L in L_VALS:
            for K in K_VALS:
                row_parts = []
                for G in G_VALS:
                    cell = next(c for c in CELLS if c["L"]==L and c["K"]==K and c["G"]==G)
                    n_pass = sum(
                        1 for seed in SEEDS
                        if defense_results.get((cell["idx"], "oracle", seed), False)
                    )
                    row_parts.append(f"{n_pass:>7}")
                p(f"  L={L:5d} K={K:1d} | " + " ".join(row_parts))
        p()

        # F4
        p("F4 — DEGENERATE MAP CHECK")
        p("-" * 40)
        p(f"  Total freeze events (all non-baseline, non-oracle, W arms): {n_events_total}")
        p(f"  Trigger silent (0 events): {f4_trigger_silent}")
        p(f"  All cells pass any mode: {all_cells_pass_any_mode}")
        p(f"  F4 degenerate: {f4_degenerate}")
        p()

        # Candidate crack cells per mode
        p("CANDIDATE CRACK CELLS (NOT exists_H_both AND oracle defends)")
        p("-" * 40)
        for mode_name, tolerance in REVISION_MODES:
            cands = candidate_cells[mode_name]
            p(f"  mode={mode_name} (tolerance={tolerance}): {len(cands)} candidates")
            for cell_idx, L, K, G in cands[:40]:
                p(f"    cell_idx={cell_idx} L={L} K={K} G={G}")
        p()

        # K=1 G-degeneracy note
        p("K=1 G-DEGENERACY NOTE")
        p("-" * 40)
        p("K=1 cells are G-replicates of one schedule (only one burst, no gap used).")
        p("Pooled K=1 defense table (L x arm, pooled across G and seeds):")
        p(f"  {'L':6s} | " + " ".join(f"{arm_name[:8]:>10}" for arm_name, _ in W_ARMS))
        for L in L_VALS:
            row_parts = []
            for arm_name, _ in W_ARMS:
                # Pool across all G and all seeds for K=1
                vals = []
                for G in G_VALS:
                    cell = next(c for c in CELLS if c["L"]==L and c["K"]==1 and c["G"]==G)
                    for seed in SEEDS:
                        vals.append(defense_results.get((cell["idx"], arm_name, seed), False))
                n_pass_pool = sum(vals)
                n_total = len(vals)
                row_parts.append(f"{n_pass_pool:>4}/{n_total}")
            p(f"  L={L:5d} | " + " ".join(row_parts))
        p()

        # Revision pass details per mode
        p("REVISION PASS DETAILS PER SEED")
        p("-" * 40)
        for mode_name, tolerance in REVISION_MODES:
            p(f"  Mode={mode_name} (tolerance={tolerance}):")
            for h in FIXED_HORIZONS:
                h_arm = f"H{h}"
                per_seed = []
                for seed in SEEDS:
                    bl_lat = r_latencies.get(("baseline", seed))
                    arm_lat = r_latencies.get((h_arm, seed))
                    if arm_lat is None:
                        per_seed.append(f"s{seed}:None")
                    elif bl_lat is None:
                        per_seed.append(f"s{seed}:nobl")
                    else:
                        within = arm_lat <= bl_lat + tolerance
                        per_seed.append(f"s{seed}:{arm_lat}({'ok' if within else 'FAIL'})")
                overall = "PASS" if rev_pass.get((h_arm, mode_name), False) else "fail"
                p(f"    {h_arm:8s}: {overall:5s}  " + "  ".join(per_seed))
        p()

        # Summary
        p("=" * 80)
        p("SUMMARY")
        p("=" * 80)
        for mode_name, _ in REVISION_MODES:
            n_cands = len(candidate_cells[mode_name])
            n_exists = sum(1 for cell in CELLS if exists_h_both.get((cell["idx"], mode_name), False))
            p(f"  mode={mode_name}: {n_exists}/80 cells pass (exists_H_both), "
              f"{80-n_exists}/80 fail, {n_cands} candidate crack cells")
        p()
        p(f"  P1 H-invariance: {'HOLDS (no F1 exceptions)' if not p1_f1_cells else f'F1 exceptions: {len(p1_f1_cells)} cells'}")
        p(f"  P2 (L=2400 tight): {'interval-empty (prediction holds)' if p2_interval_empty else 'F2: some cell has H passing both'}")
        p(f"  P3 oracle: {'all 80 cells defended' if not p3_oracle_failures else f'F3/class-E: {len(p3_oracle_failures)} cells'}")
        p(f"  F4 degenerate: {f4_degenerate} (trigger events: {n_events_total})")
        p(f"  PC1: {'PASS' if pc1_pass else f'FAIL ({len(pc1_failures_all)} sessions)'}")
        p(f"  Runtime: {runtime_min:.1f} min")
        p()
        p("NOTE: No crack CLAIM. Candidate cells require Part-3 classification.")
        p("=" * 80)

    print(f"Text analysis written to {out_txt_path}")
    print()
    print(f"Total runtime: {runtime_min:.1f} min")

    # Final git status check (for completeness)
    print()
    print("Output files:")
    print(f"  {out_rows_path}")
    print(f"  {out_txt_path}")


if __name__ == "__main__":
    main()
