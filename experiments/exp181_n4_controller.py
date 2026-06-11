"""Exp 181 — N4 rung 3 controller: does regulating value-update inertia from the
rung-2 mismatch signal yield whipsaw resistance WITH sustained-evidence revision,
and is the regulation a LAYER (no fixed constant matches it on both arms)?

PRE-REGISTERED in loop/directions/identity-n4.md (commit 3a8bce0) BEFORE any data.

=== PREDECLARED MAP ===

HYPOTHESIS (binding, falsifiable):
  N4 is real iff an agent that predicts its own policy drift and regularizes
  toward its predicted self (commitment) survives transient manipulation that
  whipsaws an N4-less twin — while still revising under sustained contrary
  evidence at least as fast as the twin. Pure rigidity is NOT N4 (fails
  revision); pure recency-following is NOT N4 (fails resistance); reducible-
  to-config is NOT N4 (some fixed inertia constant matches — rung 3-iii kill).
  predeclar: this script is the pre-registered implementation; the controller
  FORM (squared-ratio law, 30-snapshot median window, +3000 revision tolerance)
  is PROVIDED; the value CONTENT, every mismatch, and every m_bar are
  self-formed.

CONTROLLER LAW (N4 arm):
  At each monitor snapshot (every EVAL=100 steps, after the step update),
  compute mismatch m_k = ||v_hat_k - v_snap[k]||_2
  where v_hat_k = v_snap[k-1] + (v_snap[k-1] - v_snap[k-11]) / 10
  (one-step-ahead prediction from the rung-2 linear-drift formula, evaluated
  at the CURRENT snapshot k from snapshots k-1 and k-11).
  m_bar = median of the trailing 30 snapshot-mismatches (all >= 10 if < 30 exist);
  g for the next 100 steps = min(1.0, (m_bar / m_k)^2) if m_k > 0 else 1.0.
  Until 10 snapshot-mismatches exist: g = 1.0.
  Fixed arms: g = c always. Baseline: g = 1 always.

ARMS (same-snapshot, same-schedule per phase — Exp 170 binding):
  "baseline"  : g = 1 always
  "n4"        : g regulated by controller law above
  "c0.3"      : g = 0.3 fixed
  "c0.1"      : g = 0.1 fixed
  "c0.05"     : g = 0.05 fixed
  "c0.02"     : g = 0.02 fixed
  "c0.01"     : g = 0.01 fixed
  7 arms total.

PHASES (separate 15000-step sessions per arm per seed):
  Phase W (whipsaw): displacement regime verbatim Exp 176.
    Bursts (6000-6800), (9000-9800), (12000-12800); burst color = argmin(v)
    at onset; relocation rng 160_000+seed; qs reset to uniform over burst-
    color cells.
  Phase R (revision): settle to 6000, then PERMANENT relocation regime
    steps [6000, 15000): same captivity mechanics; color = argmin(v) at onset
    of step 6000; relocation rng 165_000+seed; qs reset to uniform over
    regime-color cells. The world has genuinely changed; the correct response
    is to adopt the new favorite.

SEEDS: FRESH 210-217 (8 forks). Smoke seed [210], all 7 arms x both phases,
  disclosed.
  7 arms x 2 phases x 8 seeds = 112 sessions.

VALUE UPDATE (gated):
  Each step: v *= LAMBDA; v[obs] += g_t * predictability_weight
  g gates the INCREMENT only; multiplicative decay is untouched (order-
  preserving — full resistance freezes the ordering, not the mass).
  Also: c.value_counts[obs] += g_t * predictability_weight (bookkeeping).

PROPERTIES / FALSIFIERS (ALL of P5, P6, P7 required for POSITIVE):

  P5 (resistance/recovery):
    Phase W, N4 arm. Pre-burst favorite is expressed at burst_end+2000 having
    held >= 500 consecutive steps, for >= 2/3 bursts per fork, in >= 7/8 forks;
    AND baseline passes same criterion in <= 2/8 forks (deficit must exist).
    FALSIFIER F5: N4 passes in <= 4/8 forks.
    Operationalization (simpler equivalent, pre-declared): a burst is "recovered"
    iff expressed_arr[bend+1500 : bend+2000] are ALL == pre_fav. This implies
    pre_fav holds >= 500 consecutive steps ending at-or-before bend+2000 AND
    expressed_arr[bend+2000] == pre_fav. A fork passes Phase W iff >= 2/3
    bursts are recovered.

  P6 (revision):
    Phase R, N4 revision latency <= baseline latency + 3000 steps (same seed)
    in >= 6/8 forks. Latency = first step t >= 6000 with expressed_arr[t:t+2000]
    all == regime_color; None if never within N_STEPS-2000.
    FALSIFIER F6: latency > baseline+3000 in >= 4/8 forks, OR >= 3/8 forks
    never adopt within the phase (rigidity).

  P7 (the kill test):
    NO fixed constant c satisfies BOTH the P5 criterion (>= 7/8 forks recovering)
    AND the P6 criterion (>= 6/8 forks with latency <= baseline+3000).
    FALSIFIER F7: some constant satisfies both => N4 is config, not a layer
    (NEGATIVE-config, the chapter's central kill).

VERDICT MAP:
  POSITIVE     iff P5 AND P6 AND P7.
  NEGATIVE (config)       iff P5 AND P6 AND NOT P7.
  NEGATIVE (no-resistance) iff F5.
  NEGATIVE (rigidity)     iff P5 AND F6.
  MIXED (between-bands)   otherwise.
  NOT A FALSIFIER never counts toward POSITIVE.

PRECONDITIONS (gate verdict):
  PC1: ahat_drift < 0.15 (all arms, both phases).
  PC2': vector gate verbatim on Phase-W pre-burst windows (baseline arm);
    TV(pi(bstart-1000), pi(bstart)) <= 0.05 in >= 7/8 forks per window.
  PC3: confinement >= 0.90 on burst-color/regime-color cells (Phase-W bursts
    and Phase-R regime, all arms; mean over forks per burst/regime).

DIAGNOSTICS (printed, never gated):
  - g trajectories at burst onsets (first 4 snapshots per burst, all arms).
  - Leaked writing per resisted burst: sum over burst steps of g*w where
    obs == burst_color (N4 arm; computed online, stored per burst).
  - m_bar contamination during Phase R (the monitor's own adaptation).
  - Per-constant recovery-vs-revision frontier table.

HONEST CAVEATS (pre-registered):
  The controller form (squared-ratio, median-30) and the +3000 revision
  tolerance are PROVIDED; the revision regime reuses captivity mechanics
  (diet-driven revision, not free-roam evidence — a toy-scale concession,
  named); m_bar self-contamination during long regimes is a known confound
  the diagnostics must surface; a P5 failure may reflect the D5 absorption
  leak (resistance fades as the monitor adapts within the burst) — that
  outcome is a finding about monitor-based commitment, not an excuse.
  Cards designed AFTER Exp 176-180 data; mitigation = fresh seeds 210-217 +
  this card committed before any new data.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from active_loop.creature import Creature
from active_loop.verdict import write_verdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = list(range(210, 218))   # 8 fresh forks

N_STEPS = 15_000
CHUNK_SIZE = 100
N_CHUNKS = N_STEPS // CHUNK_SIZE  # 150

# Value decay
LAMBDA = 0.9997
INIT_MASS = 1.0 / (1.0 - LAMBDA)   # ~3333.3

# Phase W burst windows: [start, end)
BURST_WINDOWS = [
    (6000, 6800),
    (9000, 9800),
    (12000, 12800),
]

# Phase R: permanent relocation from step 6000 onward
PHASE_R_START = 6000

# Burst RNG seed offsets
BURST_SEED_OFFSET_W = 160_000   # Phase W: rng 160_000 + seed
BURST_SEED_OFFSET_R = 165_000   # Phase R: rng 165_000 + seed

# N4 controller parameters
EVAL = 100            # snapshot cadence (steps)
MON_W_SNAPS = 10      # drift window (snapshots) — rung-2 formula: k-1 and k-11
CTRL_MBAR_WINDOW = 30   # trailing snapshots for m_bar median
CTRL_MIN_SNAPS = 10     # g = 1 until this many mismatch samples exist

# Arms: (name, mode)
# mode: "baseline" | "n4" | float (fixed g)
FIXED_CONSTANTS = [0.3, 0.1, 0.05, 0.02, 0.01]
ARMS: list[tuple[str, object]] = (
    [("baseline", "baseline"), ("n4", "n4")]
    + [(f"c{c}", c) for c in FIXED_CONSTANTS]
)
# 7 arms total

# Precondition thresholds
PC1_AHAT_DRIFT_MAX = 0.15

# PC2' (vector-grade TV, baseline arm, Phase W only)
PC2P_TV_MAX = 0.05
PC2P_MIN_FORKS = 7    # >= 7/8 forks per window

# PC3 confinement
PC3_CONFINEMENT_MIN = 0.90

# P5 recovery criterion
P5_CONSEC = 500       # holds >= 500 consecutive steps
P5_LAG = 2000         # window after burst end
P5_BURSTS_PER_FORK = 2   # >= 2/3 bursts
P5_N4_MIN_FORKS = 7
P5_BASELINE_MAX_FORKS = 2
F5_N4_MAX_FORKS = 4

# P6 revision latency
P6_HOLD = 2000        # latency: holds >= 2000 consecutive steps
P6_TOLERANCE = 3000   # n4 latency <= baseline latency + 3000
P6_MIN_FORKS = 6
F6_LATENCY_FORKS = 4   # > baseline+3000 in >= 4/8 forks
F6_RIGIDITY_FORKS = 3  # never-adopt in >= 3/8 forks


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def pi_of(v_row: np.ndarray) -> np.ndarray:
    """Normalized identity vector."""
    s = v_row.sum()
    return v_row / s if s > 0 else np.ones_like(v_row) / len(v_row)


def tv(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance."""
    return 0.5 * float(np.abs(p - q).sum())


def snap_index(step: int) -> int:
    """v_traj index k such that v_traj[k] = v after step (k+1)*EVAL."""
    return step // EVAL - 1


# ---------------------------------------------------------------------------
# Phase W recovery criterion (P5, simpler equivalent form)
# ---------------------------------------------------------------------------

def burst_recovered(
    expressed_arr: np.ndarray,
    burst_idx: int,
    burst_preburst_fav: list,
) -> bool:
    """Pre-declared simpler operationalization:
    expressed_arr[bend+1500 : bend+2000] are ALL == pre_fav.

    This implies pre_fav holds >= 500 consecutive steps ending at-or-before
    bend+2000 AND expressed_arr[bend+2000] == pre_fav.
    """
    bstart, bend = BURST_WINDOWS[burst_idx]
    pre_fav = burst_preburst_fav[burst_idx]
    if pre_fav is None:
        return False
    window_start = bend + 1500
    window_end = bend + 2000
    if window_end > len(expressed_arr):
        return False
    return bool(np.all(expressed_arr[window_start:window_end] == pre_fav))


def fork_passes_phase_w(
    expressed_arr: np.ndarray,
    burst_preburst_fav: list,
) -> tuple[bool, list[bool]]:
    """Fork passes Phase W iff >= 2/3 bursts are recovered."""
    per_burst = [
        burst_recovered(expressed_arr, bi, burst_preburst_fav)
        for bi in range(len(BURST_WINDOWS))
    ]
    n_recovered = sum(per_burst)
    return n_recovered >= P5_BURSTS_PER_FORK, per_burst


# ---------------------------------------------------------------------------
# Phase R revision latency
# ---------------------------------------------------------------------------

def phase_r_latency(
    expressed_arr: np.ndarray,
    regime_color: int,
) -> int | None:
    """First step t >= PHASE_R_START with expressed_arr[t:t+P6_HOLD] all == regime_color.
    Returns the LATENCY t - PHASE_R_START per the card's definition
    ("latency = t - 6000"), or None if never (check t up to N_STEPS - P6_HOLD).
    """
    max_t = N_STEPS - P6_HOLD
    for t in range(PHASE_R_START, max_t + 1):
        if expressed_arr[t] == regime_color:
            if np.all(expressed_arr[t:t + P6_HOLD] == regime_color):
                return t - PHASE_R_START
    return None


# ---------------------------------------------------------------------------
# Core step-loop — run one fork for the full session (both phases)
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    arm_name: str,
    arm_mode: object,
    phase: str,    # "W" | "R"
    n_actions: int = 4,
) -> dict:
    """Run one fork for N_STEPS steps under the given arm and phase.

    Phase W: bursts at BURST_WINDOWS with captivity + burst color = argmin(v).
    Phase R: PERMANENT captivity from PHASE_R_START onward; color = argmin(v)
             at onset of step PHASE_R_START.

    N4 arm: g regulated online by controller law (computed at each snapshot).
    Fixed arms: g = constant.
    Baseline: g = 1.

    Returns a dict with all per-session recorded quantities.
    """
    fork_name = f"exp181_{arm_name}_{phase}_s{fork_seed}"
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
    if phase == "W":
        burst_rng = np.random.default_rng(BURST_SEED_OFFSET_W + fork_seed)
    else:
        burst_rng = np.random.default_rng(BURST_SEED_OFFSET_R + fork_seed)

    # Build burst step set for Phase W
    burst_step_set: set[int] = set()
    if phase == "W":
        for bstart, bend in BURST_WINDOWS:
            burst_step_set.update(range(bstart, bend))

    # Per-step storage
    expressed_arr = np.empty(N_STEPS, dtype=np.int32)
    true_pos_arr = np.empty(N_STEPS, dtype=np.int32)
    obs_arr = np.empty(N_STEPS, dtype=np.int32)

    # v trajectory (snapshot every EVAL steps)
    v_traj: list[np.ndarray] = []

    # g trajectory (per snapshot, value of g used for the next EVAL steps)
    g_traj: list[float] = []

    # N4 controller state
    n4_mismatch_history: list[float] = []
    g_current = 1.0   # current g (updated at each snapshot)

    # Phase W diagnostics
    burst_preburst_fav: list[int | None] = [None] * len(BURST_WINDOWS)
    burst_onset_color: list[int | None] = [None] * len(BURST_WINDOWS)
    current_burst_color: int | None = None
    current_burst_idx: int | None = None

    # Phase R state
    regime_color: int | None = None
    regime_rng_active = False

    # Phase R: leaked writing tracking — for N4 arm only in Phase W
    # (Phase W equivalent: per-burst leaked writing during bursts)
    # leaked_writing[bi] = sum of g*w where obs==burst_color during burst bi
    leaked_writing_w: list[float] = [0.0] * len(BURST_WINDOWS)

    # Phase R: m_bar tracking for contamination diagnostic
    mbar_phase_r_snapshots: list[float] = []

    A_hat_start = c._A_hat().copy()

    global_step = 0

    for chunk_idx in range(N_CHUNKS):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(CHUNK_SIZE):
            t = global_step

            # ----------------------------------------------------------------
            # Phase W: burst context management
            # ----------------------------------------------------------------
            in_burst = False
            burst_idx_now: int | None = None

            if phase == "W":
                in_burst = t in burst_step_set
                for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                    if bstart <= t < bend:
                        burst_idx_now = bi
                        break

                # Burst onset: record pre-burst favorite, set burst color
                if burst_idx_now is not None and t == BURST_WINDOWS[burst_idx_now][0]:
                    pre_fav = int(np.argmax(v))
                    burst_preburst_fav[burst_idx_now] = pre_fav
                    bc = int(np.argmin(v))
                    burst_onset_color[burst_idx_now] = bc
                    current_burst_color = bc
                    current_burst_idx = burst_idx_now

                # Burst end: clear context
                if burst_idx_now is None and current_burst_idx is not None:
                    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                        if t == bend:
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
            predictability_weight = math.exp(-h_predicted)

            # ----------------------------------------------------------------
            # Determine g_t for this step
            # ----------------------------------------------------------------
            if arm_mode == "baseline":
                g_t = 1.0
            elif arm_mode == "n4":
                g_t = g_current   # updated at each snapshot
            else:
                g_t = float(arm_mode)

            # ----------------------------------------------------------------
            # Value update (gated: g gates the increment only)
            # ----------------------------------------------------------------
            v *= LAMBDA
            v[obs] += g_t * predictability_weight
            c.value_counts[obs] += g_t * predictability_weight

            # ----------------------------------------------------------------
            # Expressed preference
            # ----------------------------------------------------------------
            expressed = int(np.argmax(v))
            expressed_arr[t] = expressed

            # ----------------------------------------------------------------
            # Leaked writing diagnostic (N4 arm, Phase W, during burst)
            # ----------------------------------------------------------------
            if arm_mode == "n4" and phase == "W" and burst_idx_now is not None:
                bc_now = burst_onset_color[burst_idx_now]
                if bc_now is not None and obs == bc_now:
                    leaked_writing_w[burst_idx_now] += g_t * predictability_weight

            # ----------------------------------------------------------------
            # Action / Move
            # ----------------------------------------------------------------
            if in_burst and phase == "W":
                # Phase W burst: relocate to random cell of burst color
                cells_of_bc = color_cells_arr[current_burst_color]
                if len(cells_of_bc) > 0:
                    c.true_pos = int(burst_rng.choice(cells_of_bc))
                qs_next = np.zeros(n_cells)
                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                c.qs = qs_next
            elif in_regime:
                # Phase R: permanent relocation to regime-color cells
                cells_of_rc = color_cells_arr[regime_color]
                if len(cells_of_rc) > 0:
                    c.true_pos = int(burst_rng.choice(cells_of_rc))
                qs_next = np.zeros(n_cells)
                qs_next[cells_of_rc] = 1.0 / len(cells_of_rc)
                c.qs = qs_next
            else:
                # Normal: random action + move + advance qs through B
                action = int(rng.integers(0, n_actions))
                c.true_pos = c.world.move(c.true_pos, action)
                c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1
            global_step += 1

            # ----------------------------------------------------------------
            # Snapshot v and update N4 controller (after increment of global_step)
            # ----------------------------------------------------------------
            if global_step % EVAL == 0:
                v_traj.append(v.copy())
                k = len(v_traj) - 1  # current snapshot index (0-based)

                # N4 controller: compute mismatch m_k if k >= MON_W_SNAPS
                if arm_mode == "n4":
                    if k >= MON_W_SNAPS:
                        # v_hat = v_snap[k-1] + (v_snap[k-1] - v_snap[k-11]) / 10
                        # = v_traj[k-1] + (v_traj[k-1] - v_traj[k-1-MON_W_SNAPS]) / MON_W_SNAPS
                        v_prev = v_traj[k - 1]
                        v_drift_ref = v_traj[k - 1 - MON_W_SNAPS]
                        v_hat = v_prev + (v_prev - v_drift_ref) / MON_W_SNAPS
                        m_k = float(np.linalg.norm(v_hat - v_traj[k]))
                        n4_mismatch_history.append(m_k)

                        n_hist = len(n4_mismatch_history)
                        if n_hist >= CTRL_MIN_SNAPS:
                            # m_bar = median of trailing 30 (or all if < 30)
                            tail = n4_mismatch_history[-CTRL_MBAR_WINDOW:]
                            m_bar = statistics.median(tail)
                            if m_k > 0:
                                g_new = min(1.0, (m_bar / m_k) ** 2)
                            else:
                                g_new = 1.0
                            g_current = g_new

                            # Phase R: track m_bar for contamination diagnostic
                            if phase == "R" and global_step >= PHASE_R_START:
                                mbar_phase_r_snapshots.append(m_bar)
                        # else: g remains 1.0 (fewer than CTRL_MIN_SNAPS mismatches)

                # Record g at this snapshot (for all arms)
                g_traj.append(g_current if arm_mode == "n4" else
                              (1.0 if arm_mode == "baseline" else float(arm_mode)))

    # A_hat drift
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    return {
        "fork_seed": fork_seed,
        "arm_name": arm_name,
        "arm_mode": arm_mode if not isinstance(arm_mode, float) else arm_mode,
        "phase": phase,
        "ahat_drift": ahat_drift,
        "expressed_arr": expressed_arr,
        "true_pos_arr": true_pos_arr,
        "obs_arr": obs_arr,
        "v_traj": np.array(v_traj),
        "g_traj": g_traj,
        "mismatch_history": list(n4_mismatch_history),
        "burst_preburst_fav": burst_preburst_fav,
        "burst_onset_color": burst_onset_color,
        "regime_color": regime_color,
        "leaked_writing_w": list(leaked_writing_w),
        "mbar_phase_r_snapshots": list(mbar_phase_r_snapshots),
    }


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------

def check_pc1(all_results: list[dict]) -> tuple[bool, list[str]]:
    """PC1: ahat_drift < 0.15 every session (all arms, both phases)."""
    failures: list[str] = []
    for rr in all_results:
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: arm={rr['arm_name']} phase={rr['phase']} "
                f"seed={rr['fork_seed']} ahat_drift={rr['ahat_drift']:.4f}"
            )
    return len(failures) == 0, failures


def check_pc2_prime(baseline_w_results: list[dict]) -> tuple[bool, list[str], list[dict]]:
    """PC2' (vector-grade TV, baseline arm, Phase W only).

    Per burst window, per fork:
      TV(pi(bstart-1000), pi(bstart)) <= 0.05 in >= 7/8 forks.
    """
    failures: list[str] = []
    rows: list[dict] = []

    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
        si_end = snap_index(bstart)
        si_start = snap_index(bstart - 1000) if bstart >= 1000 else 0
        forks_ok = 0
        for rr in baseline_w_results:
            seed = rr["fork_seed"]
            vt = rr["v_traj"]
            if si_end < 0 or si_end >= len(vt) or si_start < 0:
                d = float("nan")
            else:
                pi_pre = pi_of(vt[si_start])
                pi_onset = pi_of(vt[si_end])
                d = tv(pi_pre, pi_onset)

            ok = (not math.isnan(d)) and d <= PC2P_TV_MAX
            if ok:
                forks_ok += 1
            rows.append({
                "fork_seed": seed,
                "window_idx": bi,
                "bstart": bstart,
                "tv": None if math.isnan(d) else d,
                "passes": ok,
            })

        if forks_ok < PC2P_MIN_FORKS:
            failures.append(
                f"PC2' FAIL: window {bi} (bstart={bstart}) — "
                f"{forks_ok}/{len(baseline_w_results)} forks with "
                f"TV <= {PC2P_TV_MAX} (need >= {PC2P_MIN_FORKS})"
            )

    return len(failures) == 0, failures, rows


def check_pc3(all_results: list[dict], base_cmap: list) -> tuple[bool, list[str]]:
    """PC3: confinement >= 0.90 on burst-color/regime-color cells.

    Phase W: check all arms, all 3 bursts.
    Phase R: check all arms for the regime (steps PHASE_R_START onward).
    Mean over forks per burst/regime.
    """
    failures: list[str] = []

    # Phase W
    for bi in range(len(BURST_WINDOWS)):
        bstart, bend = BURST_WINDOWS[bi]
        w_results = [rr for rr in all_results if rr["phase"] == "W"]

        # Group by arm
        arm_names = list(dict.fromkeys(rr["arm_name"] for rr in w_results))
        for arm_name in arm_names:
            arm_forks = [rr for rr in w_results if rr["arm_name"] == arm_name]
            rates = []
            for rr in arm_forks:
                bc = rr["burst_onset_color"][bi]
                if bc is None:
                    continue
                tpa = rr["true_pos_arr"][bstart:bend]
                on_bc = sum(1 for p in tpa if base_cmap[p] == bc)
                rates.append(on_bc / len(tpa) if len(tpa) > 0 else 0.0)
            if rates:
                mean_rate = float(np.mean(rates))
                if mean_rate < PC3_CONFINEMENT_MIN:
                    failures.append(
                        f"PC3 FAIL: Phase W arm={arm_name} burst={bi} "
                        f"mean_confinement={mean_rate:.3f} < {PC3_CONFINEMENT_MIN}"
                    )

    # Phase R
    r_results = [rr for rr in all_results if rr["phase"] == "R"]
    arm_names_r = list(dict.fromkeys(rr["arm_name"] for rr in r_results))
    for arm_name in arm_names_r:
        arm_forks = [rr for rr in r_results if rr["arm_name"] == arm_name]
        rates = []
        for rr in arm_forks:
            rc = rr["regime_color"]
            if rc is None:
                continue
            tpa = rr["true_pos_arr"][PHASE_R_START:]
            on_rc = sum(1 for p in tpa if base_cmap[p] == rc)
            rates.append(on_rc / len(tpa) if len(tpa) > 0 else 0.0)
        if rates:
            mean_rate = float(np.mean(rates))
            if mean_rate < PC3_CONFINEMENT_MIN:
                failures.append(
                    f"PC3 FAIL: Phase R arm={arm_name} "
                    f"mean_confinement={mean_rate:.3f} < {PC3_CONFINEMENT_MIN}"
                )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# P5 / P6 / P7 evaluation
# ---------------------------------------------------------------------------

def eval_p5_arm(
    arm_results_w: list[dict],
) -> tuple[int, list[dict]]:
    """Return (n_forks_passing_p5, per_fork_rows) for Phase W results of one arm."""
    rows = []
    n_pass = 0
    for rr in arm_results_w:
        passed, per_burst = fork_passes_phase_w(rr["expressed_arr"], rr["burst_preburst_fav"])
        if passed:
            n_pass += 1
        rows.append({
            "fork_seed": rr["fork_seed"],
            "arm_name": rr["arm_name"],
            "per_burst_recovered": per_burst,
            "n_burst_recovered": sum(per_burst),
            "passes_p5": passed,
        })
    return n_pass, rows


def eval_p6_arm(
    arm_results_r: list[dict],
    baseline_r_results: list[dict],
) -> tuple[int, int, list[dict]]:
    """Return (n_forks_within_tolerance, n_forks_never_adopt, per_fork_rows).

    n_forks_within_tolerance: forks where arm_latency is not None AND
      arm_latency <= baseline_latency + P6_TOLERANCE (baseline same seed).
    n_forks_never_adopt: forks where arm_latency is None.
    """
    # Build baseline latency map by seed
    baseline_lat: dict[int, int | None] = {}
    for rr in baseline_r_results:
        rc = rr["regime_color"]
        if rc is None:
            baseline_lat[rr["fork_seed"]] = None
        else:
            baseline_lat[rr["fork_seed"]] = phase_r_latency(rr["expressed_arr"], rc)

    rows = []
    n_within = 0
    n_never = 0
    for rr in arm_results_r:
        seed = rr["fork_seed"]
        rc = rr["regime_color"]
        if rc is None:
            lat = None
        else:
            lat = phase_r_latency(rr["expressed_arr"], rc)

        if lat is None:
            n_never += 1

        bl = baseline_lat.get(seed)
        within_tol: bool | None = None
        if lat is not None:
            if bl is None:
                # Baseline never adopted — this fork satisfies neither P6 nor P7 comparison
                within_tol = None
            else:
                within_tol = lat <= bl + P6_TOLERANCE
                if within_tol:
                    n_within += 1
        rows.append({
            "fork_seed": seed,
            "arm_name": rr["arm_name"],
            "latency": lat,
            "baseline_latency": bl,
            "within_tolerance": within_tol,
        })
    return n_within, n_never, rows


def evaluate(
    results_by_arm_phase: dict,   # {(arm_name, phase): list[dict]}
    arm_names: list[str],
    seeds: list[int],
) -> dict:
    """Compute P5/P6/P7 and derive verdict.

    Returns dict with all evaluation quantities.
    """
    baseline_w = results_by_arm_phase.get(("baseline", "W"), [])
    baseline_r = results_by_arm_phase.get(("baseline", "R"), [])

    # P5: Phase W, per arm
    p5_by_arm: dict[str, dict] = {}
    for arm_name in arm_names:
        arm_w = results_by_arm_phase.get((arm_name, "W"), [])
        n_pass, rows = eval_p5_arm(arm_w)
        p5_by_arm[arm_name] = {"n_forks_passing": n_pass, "rows": rows}

    n4_p5_pass_n = p5_by_arm.get("n4", {}).get("n_forks_passing", 0)
    bl_p5_pass_n = p5_by_arm.get("baseline", {}).get("n_forks_passing", 0)

    p5_n4 = n4_p5_pass_n >= P5_N4_MIN_FORKS
    p5_baseline_deficit = bl_p5_pass_n <= P5_BASELINE_MAX_FORKS
    p5_pass = p5_n4 and p5_baseline_deficit
    f5 = n4_p5_pass_n <= F5_N4_MAX_FORKS

    # P6: Phase R, per arm
    p6_by_arm: dict[str, dict] = {}
    for arm_name in arm_names:
        arm_r = results_by_arm_phase.get((arm_name, "R"), [])
        n_within, n_never, rows = eval_p6_arm(arm_r, baseline_r)
        p6_by_arm[arm_name] = {
            "n_forks_within_tolerance": n_within,
            "n_forks_never": n_never,
            "rows": rows,
        }

    n4_p6 = p6_by_arm.get("n4", {})
    n4_p6_within = n4_p6.get("n_forks_within_tolerance", 0)
    n4_p6_never = n4_p6.get("n_forks_never", 0)

    p6_n4_within = n4_p6_within >= P6_MIN_FORKS
    p6_n4_no_rigidity = n4_p6_never < F6_RIGIDITY_FORKS
    p6_pass = p6_n4_within and p6_n4_no_rigidity
    f6 = (n4_p6_within <= (len(seeds) - F6_LATENCY_FORKS)) or (n4_p6_never >= F6_RIGIDITY_FORKS)

    # P7: NO constant arm has BOTH p5 >= 7/8 AND p6 >= 6/8
    constant_arm_names = [name for name, mode in ARMS if isinstance(mode, float)]
    p7_pass = True
    p7_kill_arms: list[str] = []
    for carm in constant_arm_names:
        c_p5_n = p5_by_arm.get(carm, {}).get("n_forks_passing", 0)
        c_p6_n = p6_by_arm.get(carm, {}).get("n_forks_within_tolerance", 0)
        if c_p5_n >= P5_N4_MIN_FORKS and c_p6_n >= P6_MIN_FORKS:
            p7_pass = False
            p7_kill_arms.append(carm)
    f7 = not p7_pass

    # Verdict map
    if p5_pass and p6_pass and p7_pass:
        verdict_str = "POSITIVE"
        tier = "positive"
    elif p5_pass and p6_pass and not p7_pass:
        verdict_str = "NEGATIVE"
        tier = "config"
    elif f5:
        verdict_str = "NEGATIVE"
        tier = "no-resistance"
    elif p5_pass and f6:
        verdict_str = "NEGATIVE"
        tier = "rigidity"
    else:
        verdict_str = "MIXED"
        tier = "between-bands"

    return {
        "p5_by_arm": p5_by_arm,
        "p6_by_arm": p6_by_arm,
        "n4_p5_pass_n": n4_p5_pass_n,
        "bl_p5_pass_n": bl_p5_pass_n,
        "p5_n4": p5_n4,
        "p5_baseline_deficit": p5_baseline_deficit,
        "p5_pass": p5_pass,
        "f5": f5,
        "n4_p6_within": n4_p6_within,
        "n4_p6_never": n4_p6_never,
        "p6_pass": p6_pass,
        "f6": f6,
        "p7_pass": p7_pass,
        "f7": f7,
        "p7_kill_arms": p7_kill_arms,
        "verdict_str": verdict_str,
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_p5_table(ev: dict, arm_names: list[str]) -> None:
    print("Phase W — P5 recovery table (per arm per fork):")
    header = (f"{'arm':>12} {'seed':>5} {'b0':>4} {'b1':>4} {'b2':>4} "
              f"{'n_rec':>6} {'passes':>7}")
    print(header)
    print("-" * len(header))
    for arm_name in arm_names:
        arm_data = ev["p5_by_arm"].get(arm_name, {})
        for row in arm_data.get("rows", []):
            pb = row["per_burst_recovered"]
            b0 = "Y" if pb[0] else "n"
            b1 = "Y" if pb[1] else "n"
            b2 = "Y" if pb[2] else "n"
            print(f"{arm_name:>12} {row['fork_seed']:>5} {b0:>4} {b1:>4} {b2:>4} "
                  f"{row['n_burst_recovered']:>6} {'PASS' if row['passes_p5'] else 'fail':>7}")
        arm_n = arm_data.get("n_forks_passing", 0)
        n_forks = len(arm_data.get("rows", []))
        print(f"  {'':>12} {arm_name} TOTAL: {arm_n}/{n_forks} forks pass P5")
    print()


def print_p6_table(ev: dict, arm_names: list[str]) -> None:
    print("Phase R — P6 revision latency table (per arm per fork):")
    header = (f"{'arm':>12} {'seed':>5} {'latency':>9} {'bl_lat':>8} "
              f"{'tol':>8} {'within?':>8}")
    print(header)
    print("-" * len(header))
    for arm_name in arm_names:
        arm_data = ev["p6_by_arm"].get(arm_name, {})
        for row in arm_data.get("rows", []):
            lat_s = str(row["latency"]) if row["latency"] is not None else "None"
            bl_s = str(row["baseline_latency"]) if row["baseline_latency"] is not None else "None"
            tol_s = "N/A" if row["within_tolerance"] is None else str(P6_TOLERANCE)
            within_s = (
                "N/A" if row["within_tolerance"] is None
                else ("PASS" if row["within_tolerance"] else "fail")
            )
            print(f"{arm_name:>12} {row['fork_seed']:>5} {lat_s:>9} {bl_s:>8} "
                  f"{tol_s:>8} {within_s:>8}")
        n_within = arm_data.get("n_forks_within_tolerance", 0)
        n_never = arm_data.get("n_forks_never", 0)
        n_forks = len(arm_data.get("rows", []))
        print(f"  {arm_name} TOTAL: {n_within}/{n_forks} within tolerance, "
              f"{n_never}/{n_forks} never adopted")
    print()


def print_g_at_burst_onsets(results_by_arm_phase: dict, arm_names: list[str]) -> None:
    """Print g values at burst onsets (first 4 snapshots of each burst), all arms, Phase W."""
    print("N4 g at Phase W burst onsets (first 4 snapshots per burst, all arms):")
    for arm_name in arm_names:
        arm_w = results_by_arm_phase.get((arm_name, "W"), [])
        if not arm_w:
            continue
        # Use first fork (seed 210 in smoke)
        rr0 = arm_w[0]
        g_traj = rr0["g_traj"]
        print(f"  arm={arm_name} seed={rr0['fork_seed']}:")
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            snap_start = bstart // EVAL  # first snapshot after bstart
            vals = g_traj[snap_start:snap_start + 4]
            val_strs = [f"{x:.4f}" for x in vals]
            print(f"    burst {bi} (step {bstart}): snaps {snap_start}-{snap_start+3}: {val_strs}")
    print()


def print_leaked_writing(results_by_arm_phase: dict) -> None:
    """Leaked writing per resisted burst (N4 arm, Phase W)."""
    n4_w = results_by_arm_phase.get(("n4", "W"), [])
    if not n4_w:
        return
    print("Leaked writing (N4 arm, Phase W) — sum of g*w where obs==burst_color per burst:")
    header = f"{'seed':>5} {'burst_0':>10} {'burst_1':>10} {'burst_2':>10}"
    print(header)
    print("-" * len(header))
    for rr in n4_w:
        lw = rr["leaked_writing_w"]
        print(f"{rr['fork_seed']:>5} {lw[0]:>10.4f} {lw[1]:>10.4f} {lw[2]:>10.4f}")
    print()


def print_mbar_contamination(results_by_arm_phase: dict) -> None:
    """m_bar contamination during Phase R (N4 arm diagnostic)."""
    n4_r = results_by_arm_phase.get(("n4", "R"), [])
    if not n4_r:
        return
    print("m_bar contamination during Phase R (N4 arm — monitor adaptation to regime):")
    for rr in n4_r:
        snaps = rr["mbar_phase_r_snapshots"]
        if snaps:
            early = snaps[:5]
            late = snaps[-5:]
            early_s = [f"{x:.4f}" for x in early]
            late_s = [f"{x:.4f}" for x in late]
            print(f"  seed={rr['fork_seed']}: n_snaps={len(snaps)}, "
                  f"early={early_s}, late={late_s}")
        else:
            print(f"  seed={rr['fork_seed']}: no Phase-R m_bar snapshots")
    print()


def print_constant_frontier(ev: dict) -> None:
    """Per-constant recovery-vs-revision frontier table."""
    constant_arm_names = [name for name, mode in ARMS if isinstance(mode, float)]
    print("Constant-arm recovery/revision frontier (Phase W P5 forks vs Phase R median latency):")
    header = f"{'arm':>8} {'p5_forks':>10} {'p6_within':>10} {'med_latency':>12}"
    print(header)
    print("-" * len(header))
    for arm_name in constant_arm_names:
        p5_n = ev["p5_by_arm"].get(arm_name, {}).get("n_forks_passing", 0)
        p6_data = ev["p6_by_arm"].get(arm_name, {})
        p6_n = p6_data.get("n_forks_within_tolerance", 0)
        lats = [
            row["latency"]
            for row in p6_data.get("rows", [])
            if row["latency"] is not None
        ]
        med_lat = statistics.median(lats) if lats else None
        med_s = str(int(med_lat)) if med_lat is not None else "None"
        print(f"{arm_name:>8} {p5_n:>10} {p6_n:>10} {med_s:>12}")
    print()


# ---------------------------------------------------------------------------
# JSON row writer
# ---------------------------------------------------------------------------

def write_json_rows(
    results_by_arm_phase: dict,
    arm_names: list[str],
    ev: dict | None,
    pc2p_rows: list[dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for arm_name in arm_names:
            for phase in ("W", "R"):
                for rr in results_by_arm_phase.get((arm_name, phase), []):
                    seed = rr["fork_seed"]
                    rc = rr["regime_color"]
                    lat = None
                    if phase == "R" and rc is not None:
                        lat = phase_r_latency(rr["expressed_arr"], rc)

                    # Phase W: per-burst recovery rows
                    if phase == "W":
                        for bi in range(len(BURST_WINDOWS)):
                            bstart, bend = BURST_WINDOWS[bi]
                            recovered = burst_recovered(
                                rr["expressed_arr"], bi, rr["burst_preburst_fav"]
                            )
                            row = {
                                "exp": 181,
                                "arm": arm_name,
                                "phase": "W",
                                "fork_seed": seed,
                                "burst_idx": bi,
                                "burst_start": bstart,
                                "burst_end": bend,
                                "pre_fav": rr["burst_preburst_fav"][bi],
                                "burst_color": rr["burst_onset_color"][bi],
                                "recovered": recovered,
                                "ahat_drift": rr["ahat_drift"],
                                "leaked_writing": rr["leaked_writing_w"][bi] if arm_name == "n4" else None,
                            }
                            fh.write(json.dumps(row) + "\n")
                    else:
                        # Phase R: latency row
                        row = {
                            "exp": 181,
                            "arm": arm_name,
                            "phase": "R",
                            "fork_seed": seed,
                            "regime_color": rc,
                            "latency": lat,
                            "ahat_drift": rr["ahat_drift"],
                        }
                        fh.write(json.dumps(row) + "\n")

        # PC2' rows
        for row in pc2p_rows:
            fh.write(json.dumps({"exp": 181, "phase": "pc2p", **row}) + "\n")

        # Summary row
        if ev is not None:
            summary = {
                "exp": 181,
                "phase": "summary",
                "verdict": ev["verdict_str"],
                "tier": ev["tier"],
                "n4_p5_forks": ev["n4_p5_pass_n"],
                "bl_p5_forks": ev["bl_p5_pass_n"],
                "p5_pass": ev["p5_pass"],
                "f5": ev["f5"],
                "n4_p6_within": ev["n4_p6_within"],
                "n4_p6_never": ev["n4_p6_never"],
                "p6_pass": ev["p6_pass"],
                "f6": ev["f6"],
                "p7_pass": ev["p7_pass"],
                "f7": ev["f7"],
                "p7_kill_arms": ev["p7_kill_arms"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 181 — N4 rung 3 controller"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke run: seed [210] only (all 7 arms x both phases = 14 sessions), no verdict"
    )
    args = parser.parse_args()

    seeds = [210] if args.smoke else SEEDS

    print("=" * 80)
    print("Exp 181 — N4 rung 3 controller: resistance WITH revision, no fixed-constant match")
    print("PRE-REGISTERED in loop/directions/identity-n4.md (commit 3a8bce0)")
    print("=" * 80)
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    n_cells = mirro.world.n_cells

    print(f"n_cells={n_cells}, n_colors={n_colors}")
    print(f"LAMBDA={LAMBDA}, INIT_MASS={INIT_MASS:.1f}")
    print(f"Phase W bursts: {BURST_WINDOWS}")
    print(f"Phase R: permanent captivity from step {PHASE_R_START}")
    print(f"Arms ({len(ARMS)}): {[a[0] for a in ARMS]}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Total sessions: {len(seeds) * len(ARMS) * 2}")
    print()

    per_color_cells = [sum(1 for cc in base_cmap if cc == col) for col in range(n_colors)]
    print(f"Cells per color: {per_color_cells}")
    print()

    vc = mirro.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        spine_fav = int(np.argmax(vc))
        print(f"Mirro spine value_counts: {vc}")
        print(f"Spine standing favorite: color {spine_fav}")
    print()

    arm_names = [a[0] for a in ARMS]

    # --- Run all sessions ---
    results_by_arm_phase: dict[tuple[str, str], list[dict]] = {}

    for arm_name, arm_mode in ARMS:
        for phase in ("W", "R"):
            key = (arm_name, phase)
            results_by_arm_phase[key] = []
            print(f"Running arm={arm_name} phase={phase} ({len(seeds)} forks) ...", flush=True)
            for seed in seeds:
                print(f"  seed={seed} ...", flush=True)
                rr = run_fork(
                    mirro, seed, base_cmap, n_colors,
                    arm_name=arm_name, arm_mode=arm_mode,
                    phase=phase,
                )
                results_by_arm_phase[key].append(rr)
                drift_s = f"ahat_drift={rr['ahat_drift']:.5f}"
                if phase == "W":
                    print(f"    {drift_s}  burst_favs={rr['burst_preburst_fav']}  "
                          f"burst_colors={rr['burst_onset_color']}")
                else:
                    rc = rr["regime_color"]
                    lat = None
                    if rc is not None:
                        lat = phase_r_latency(rr["expressed_arr"], rc)
                    print(f"    {drift_s}  regime_color={rc}  latency={lat}")
            print()

    # --- All arms collected. Print diagnostics ---
    print_g_at_burst_onsets(results_by_arm_phase, arm_names)

    if not args.smoke:
        print_leaked_writing(results_by_arm_phase)
        print_mbar_contamination(results_by_arm_phase)

    # --- Smoke: print Phase W and Phase R tables for seed 210, then return ---
    if args.smoke:
        print("=== SMOKE RESULTS (seed 210, all 7 arms x 2 phases) ===")
        print()
        # Phase W summary
        print("Phase W (whipsaw) — recovery per burst:")
        for arm_name, arm_mode in ARMS:
            arm_w = results_by_arm_phase.get((arm_name, "W"), [])
            for rr in arm_w:
                passed, per_burst = fork_passes_phase_w(
                    rr["expressed_arr"], rr["burst_preburst_fav"]
                )
                bc = rr["burst_onset_color"]
                pf = rr["burst_preburst_fav"]
                print(f"  arm={arm_name} seed={rr['fork_seed']}: "
                      f"burst_colors={bc}  pre_favs={pf}  "
                      f"recovered={per_burst}  passes_p5={'Y' if passed else 'n'}")
        print()

        # Phase R summary
        print("Phase R (revision) — latency:")
        for arm_name, arm_mode in ARMS:
            arm_r = results_by_arm_phase.get((arm_name, "R"), [])
            for rr in arm_r:
                rc = rr["regime_color"]
                lat = None
                if rc is not None:
                    lat = phase_r_latency(rr["expressed_arr"], rc)
                print(f"  arm={arm_name} seed={rr['fork_seed']}: "
                      f"regime_color={rc}  latency={lat}")
        print()

        # N4 arm controller diagnostics for smoke seed
        n4_w_smoke = results_by_arm_phase.get(("n4", "W"), [])
        if n4_w_smoke:
            rr = n4_w_smoke[0]
            print(f"N4 arm Phase W seed={rr['fork_seed']}: "
                  f"mismatch_history length={len(rr['mismatch_history'])}")
            if rr["mismatch_history"]:
                print(f"  first 10 mismatches: {[f'{x:.4f}' for x in rr['mismatch_history'][:10]]}")
                print(f"  last 10 mismatches: {[f'{x:.4f}' for x in rr['mismatch_history'][-10:]]}")
            print(f"  leaked_writing_w={[f'{x:.4f}' for x in rr['leaked_writing_w']]}")
            print()

        n4_r_smoke = results_by_arm_phase.get(("n4", "R"), [])
        if n4_r_smoke:
            rr = n4_r_smoke[0]
            print(f"N4 arm Phase R seed={rr['fork_seed']}: "
                  f"mbar_phase_r_snapshots length={len(rr['mbar_phase_r_snapshots'])}")
            if rr["mbar_phase_r_snapshots"]:
                snaps = rr["mbar_phase_r_snapshots"]
                print(f"  early (first 5): {[f'{x:.4f}' for x in snaps[:5]]}")
                print(f"  late (last 5): {[f'{x:.4f}' for x in snaps[-5:]]}")
            print()

        print("SMOKE ONLY — no verdict")
        return

    # --- Preconditions ---
    all_results = [rr for v in results_by_arm_phase.values() for rr in v]
    baseline_w = results_by_arm_phase.get(("baseline", "W"), [])

    pc1_pass, pc1_failures = check_pc1(all_results)
    pc2p_pass, pc2p_failures, pc2p_rows = check_pc2_prime(baseline_w)
    pc3_pass, pc3_failures = check_pc3(all_results, base_cmap)

    all_gate_failures = pc1_failures + pc2p_failures + pc3_failures
    gate_pass = pc1_pass and pc2p_pass and pc3_pass

    print("=== PRECONDITIONS ===")
    if all_gate_failures:
        print("PRECONDITION FAILED:")
        for f in all_gate_failures:
            print(f"  {f}")
    else:
        print("PC1 + PC2' + PC3: all PASS")
    print()

    # PC2' table (printed regardless)
    print("PC2' table (baseline arm, Phase W, per fork per window):")
    hdr = f"{'seed':>5} {'win':>4} {'bstart':>7} {'tv':>8} {'ok?':>5}"
    print(hdr)
    print("-" * len(hdr))
    for row in pc2p_rows:
        tv_s = f"{row['tv']:.5f}" if row["tv"] is not None else "nan"
        print(f"{row['fork_seed']:>5} {row['window_idx']:>4} {row['bstart']:>7} "
              f"{tv_s:>8} {'ok' if row['passes'] else 'FAIL':>5}")
    print()

    if not gate_pass:
        print("PRECONDITION FAILED — writing rows, halting before verdict.")
        out_rows = Path(__file__).parent / "outputs" / "exp181_rows.json"
        write_json_rows(results_by_arm_phase, arm_names, ev=None,
                        pc2p_rows=pc2p_rows, out_path=out_rows)
        print(f"Rows written to {out_rows}")
        return

    # --- Evaluate ---
    ev = evaluate(results_by_arm_phase, arm_names, seeds)

    # --- Print evaluation tables ---
    print("=" * 80)
    print("EVALUATION TABLES")
    print("=" * 80)
    print()
    print_p5_table(ev, arm_names)
    print_p6_table(ev, arm_names)
    print_leaked_writing(results_by_arm_phase)
    print_mbar_contamination(results_by_arm_phase)
    print_constant_frontier(ev)

    # --- Conjunct summary ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    n_seeds = len(seeds)
    print(f"P5 (resistance/recovery, Phase W):")
    print(f"  N4 forks passing: {ev['n4_p5_pass_n']}/{n_seeds} "
          f"(need >= {P5_N4_MIN_FORKS}) => {'PASS' if ev['p5_n4'] else 'fail'}")
    print(f"  Baseline forks passing: {ev['bl_p5_pass_n']}/{n_seeds} "
          f"(need <= {P5_BASELINE_MAX_FORKS} for deficit) => "
          f"{'deficit exists' if ev['p5_baseline_deficit'] else 'NO DEFICIT'}")
    print(f"  P5 overall: {'PASS' if ev['p5_pass'] else ('FALSIFIER F5' if ev['f5'] else 'fail')}")
    print()
    print(f"P6 (revision, Phase R):")
    print(f"  N4 forks within tolerance (latency <= baseline+{P6_TOLERANCE}): "
          f"{ev['n4_p6_within']}/{n_seeds} (need >= {P6_MIN_FORKS})")
    print(f"  N4 forks never adopting: {ev['n4_p6_never']}/{n_seeds} "
          f"(rigidity if >= {F6_RIGIDITY_FORKS})")
    print(f"  P6 overall: {'PASS' if ev['p6_pass'] else ('FALSIFIER F6' if ev['f6'] else 'fail')}")
    print(f"  NOTE: If baseline latency is None for a fork, that fork cannot satisfy P6 "
          f"comparison — listed as within_tolerance=None in the table above.")
    print()
    print(f"P7 (kill test — no constant arm matches both P5 and P6):")
    if ev["p7_pass"]:
        print(f"  PASS — no constant arm has both p5_forks >= {P5_N4_MIN_FORKS} "
              f"AND p6_within >= {P6_MIN_FORKS}")
    else:
        print(f"  FAIL (FALSIFIER F7) — constant arms matching both: {ev['p7_kill_arms']}")
        print(f"  => N4 is config, not a layer (NEGATIVE-config, the chapter's central kill)")
    print()

    tier_texts = {
        "positive": "LAYER CONFIRMED — N4 resists transient pressure, revises under sustained evidence, not reducible to a constant",
        "config": "NEGATIVE (config) — some fixed constant matches N4 on both arms; N4 is config not a layer",
        "no-resistance": "NEGATIVE (no-resistance) — N4 fails whipsaw resistance (F5); controller insufficient",
        "rigidity": "NEGATIVE (rigidity) — N4 passes resistance but not revision (F6); controller too rigid",
        "between-bands": "MIXED (between-bands) — neither POSITIVE nor a named falsifier; richer test needed",
    }
    tier_text = tier_texts.get(ev["tier"], ev["tier"])
    print(f"VERDICT: {ev['verdict_str']}  ({tier_text})")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows = Path(__file__).parent / "outputs" / "exp181_rows.json"
    write_json_rows(results_by_arm_phase, arm_names, ev=ev,
                    pc2p_rows=pc2p_rows, out_path=out_rows)
    print(f"Rows written to {out_rows}")

    # --- Write verdict JSON ---
    arms_dict = {
        "P5_resistance_recovery": {
            "pass": bool(ev["p5_pass"]),
            "reason": (
                f"N4 forks passing Phase W: {ev['n4_p5_pass_n']}/{n_seeds} "
                f"(need >= {P5_N4_MIN_FORKS}); "
                f"baseline forks passing: {ev['bl_p5_pass_n']}/{n_seeds} "
                f"(need <= {P5_BASELINE_MAX_FORKS}); "
                f"falsifier F5: N4 <= {F5_N4_MAX_FORKS}/{n_seeds}"
            ),
        },
        "P6_revision": {
            "pass": bool(ev["p6_pass"]),
            "reason": (
                f"N4 forks within latency tolerance (baseline+{P6_TOLERANCE}): "
                f"{ev['n4_p6_within']}/{n_seeds} (need >= {P6_MIN_FORKS}); "
                f"never-adopted: {ev['n4_p6_never']}/{n_seeds} "
                f"(rigidity-falsifier if >= {F6_RIGIDITY_FORKS}); "
                f"falsifier F6: latency > baseline+{P6_TOLERANCE} in >= {F6_LATENCY_FORKS}/{n_seeds} forks"
            ),
        },
        "P7_not_config": {
            "pass": bool(ev["p7_pass"]),
            "reason": (
                "no fixed constant arm matches both P5 and P6 criteria"
                if ev["p7_pass"]
                else f"FALSIFIER F7: constant arms satisfying both: {ev['p7_kill_arms']} — N4 is config"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp181_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp181",
        arms=arms_dict,
        verdict=ev["verdict_str"],
        halted=not gate_pass,
        notes=(
            "N4 rung 3 controller. PRE-REGISTERED in loop/directions/identity-n4.md "
            "(commit 3a8bce0) BEFORE any data. Controller form provided, content self-formed. "
            "7 arms x 2 phases x 8 seeds = 112 sessions. "
            "Operationalization: expressed_arr[bend+1500:bend+2000] all == pre_fav (simpler "
            "equivalent of >= 500 consecutive steps holding at bend+2000). "
            "Verdict map: POSITIVE iff P5 AND P6 AND P7; NEGATIVE-config iff P5 AND P6 AND NOT P7; "
            "NEGATIVE-no-resistance iff F5; NEGATIVE-rigidity iff P5 AND F6; MIXED otherwise."
        ),
    )
    print(f"Verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
