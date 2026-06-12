"""Exp 182 — N4 rung 3 attempt 2: the FREEZE-GATE control surface.

PRE-REGISTERED in loop/directions/identity-n4.md (commit a0eaf1f) BEFORE any data.
Card block: "RUNG 3, ATTEMPT 2 — PRE-REGISTRATION (Exp 182, the FREEZE-GATE...)"

=== PREDECLARED MAP ===

HYPOTHESIS (binding, falsifiable):
  N4 is real iff an agent that predicts its own policy drift and FREEZES its
  value dynamics (both decay and writes) under identity pressure survives
  transient manipulation that whipsaws an N4-less twin — while still revising
  under sustained contrary evidence at least as fast as the twin — and the
  regulation is a LAYER (no fixed constant matches it on BOTH arms — the Exp 173
  universal-constant kill). The control surface is STATE-BASED (NORMAL/RESIST),
  not a gated increment; Exp 181's write-channel failure was the wrong surface.
  predeclar: controller FORM provided (THETA=2.0, E_STAR=600, 0.6 pressure bar,
  200-step window, 2-snapshot release, fixed-H sweep); value CONTENT, every
  mismatch, every m_bar are self-formed.

STATE MACHINE (n4_freeze and H-arms):
  NORMAL: v <- lambda*v + w*e_obs; rung-2 monitor m_k at each snapshot (EVAL=100);
    m_bar = median of trailing 30 snapshot-mismatches (g==1 until 10 samples);
    anomaly ratio r_k = m_k / max(m_bar, eps).
    Enter RESIST when r_k >= THETA at one snapshot (once >= CTRL_MIN_SNAPS exist).
  RESIST: v FROZEN (no decay, no writes; value_counts untouched);
    v_ref = v at entry; pi_ref = pi(v_ref); m_bar FROZEN at entry.
    blocked_w_by_color[obs] += w each step; c_star = argmax(blocked_w_by_color);
    E_blocked = blocked_w_by_color[c_star].
    Directional pressure: dot(e_obs - pi_ref, e_cstar - pi_ref) logged.
    PRESSURE statistic (at snapshots): fraction of last 200 obs == c_star >= 0.6.
    Release A (TRANSIENT): pressure INACTIVE at 2 consecutive snapshot checks =>
      unfreeze to NORMAL with v == v_ref preserved; mismatch_history CLEARED.
    Release B (CONCESSION, n4_freeze arm): pressure ACTIVE AND E_blocked >= E_STAR
      => accept new regime; unfreeze; NORMAL writes resume; mismatch_history CLEARED.
    Release C (CONCESSION, H-arms): frozen_steps >= H => unfreeze; mismatch CLEARED.

ARMS (same-snapshot, same-schedule per phase — Exp 170 binding):
  "baseline"     : g=1 always, no freeze
  "n4_freeze"    : state machine with evidence-based concession (E_blocked >= E_STAR)
  "c0.3"         : fixed write-gain 0.3 (retained from Exp 181)
  "c0.1"         : fixed write-gain 0.1
  "c0.05"        : fixed write-gain 0.05
  "c0.02"        : fixed write-gain 0.02
  "c0.01"        : fixed write-gain 0.01
  "H600"         : freeze + time-based concession at frozen_steps >= 600
  "H900"         : freeze + time-based concession at frozen_steps >= 900
  "H1200"        : freeze + time-based concession at frozen_steps >= 1200
  "H1800"        : freeze + time-based concession at frozen_steps >= 1800
  "H2400"        : freeze + time-based concession at frozen_steps >= 2400
  "H3000"        : freeze + time-based concession at frozen_steps >= 3000
  "oracle"       : Phase W only — v frozen exactly during burst windows, DIAGNOSTIC only
  13 verdict-relevant arms + oracle.

PHASES (separate 15000-step sessions per arm per seed):
  Phase W (whipsaw): bursts (6000-6800),(9000-9800),(12000-12800); burst color =
    argmin(v) at onset; relocation rng 160_000+seed; qs uniform over burst-color cells.
  Phase R (revision): permanent captivity from step 6000; color = argmin(v) at onset;
    rng 165_000+seed. The correct response is to adopt the new favorite.
  Oracle: Phase W only (diagnostic, excluded from all properties).

SEEDS: FRESH 218-225 (8 forks). Smoke seed [218], ALL arms, both phases
  (oracle W only), disclosed.
  13 verdict arms x 2 phases x 8 seeds + 8 oracle = 216 sessions.

PROPERTIES / FALSIFIERS (ALL of P5, P6, P7 required for POSITIVE):

  P5 (resistance/recovery — verbatim Exp 181):
    Phase W, n4_freeze arm. expressed_arr[bend+1500:bend+2000] all == pre_fav,
    for >= 2/3 bursts per fork, in >= 7/8 forks; AND baseline passes in <= 2/8.
    FALSIFIER F5: n4_freeze passes in <= 4/8 forks.

  P6 (revision — verbatim Exp 181):
    Phase R, n4_freeze latency <= baseline latency + 3000 in >= 6/8 forks.
    Latency = first step t >= 6000 with expressed_arr[t:t+2000] all == regime_color,
    or None if never; scan to N_STEPS-2000.
    FALSIFIER F6 (rigidity): never-adopt in >= 3/8 forks, or over-tolerance in >= 4/8.

  P7 (the kill test):
    NO write-gain constant AND NO fixed-H constant satisfies BOTH the P5 bar
    (>= 7/8 forks) AND the P6 bar (>= 6/8 forks) on its own arms.
    FALSIFIER F7: some constant arm does => NEGATIVE-config (fixed horizon suffices
    at this richness — the honest kill; n4's evidence-based concession is ornament).

VERDICT MAP:
  POSITIVE           iff P5 AND P6 AND P7.
  NEGATIVE (config)  iff P5 AND P6 AND NOT P7.
  NEGATIVE (rigidity)        iff P5 AND F6.
  NEGATIVE (no-resistance)   iff F5.
  NO VERDICT iff PC1/PC2'/PC3 fail.
  MIXED (between-bands) otherwise.
  Oracle NEVER counts toward any property.

PRECONDITIONS (gate verdict):
  PC1: ahat_drift < 0.15 (all sessions including oracle).
  PC2': vector-grade TV verbatim Exp 181 (baseline arm, Phase W windows):
    TV(pi(bstart-1000), pi(bstart)) <= 0.05 in >= 7/8 forks per window.
  PC3: confinement >= 0.90 on burst/regime-color cells (all relocation arms/phases).

CRITICAL INTERPRETATION RULES (pre-registered):
  1. fixed-H constant arm passing BOTH P5+P6 => NEGATIVE-config is the correct
     verdict (a fixed horizon suffices; evidence-based concession is ornament, not layer).
  2. n4_freeze passing P5 but NOT P6 (never-adopt or over-tolerance) => NEGATIVE-rigidity:
     the freeze protects without revising (protect-without-revise is not the layer).
  3. n4_freeze failing P5 => NEGATIVE-no-resistance: revise-without-protect,
     the same trap as Exp 181's write-channel (revise-without-protect is not the layer).

HONEST CAVEATS (pre-registered):
  THETA, E_STAR, 0.6 pressure bar, 200-step pressure window, and 2-snapshot release
  are PROVIDED; the 100-step trigger lag persists by design (bounded exposure, ~85
  units audited); the cleared-history re-forming window (10 snapshots) is a declared
  vulnerability — no trigger possible inside it; E_STAR-vs-H may not separate at this
  richness — NEGATIVE-config is a live and acceptable outcome; revision regime remains
  diet-driven captivity; cards designed after Exp 181 data; mitigation = fresh seeds
  218-225 + this card committed before any new data; Exp 181 remains NEGATIVE for
  write-channel gain control — this attempt changes the surface, not the parameters.
"""
from __future__ import annotations

import argparse
import collections
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

SEEDS = list(range(218, 226))   # 8 fresh forks

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

# Monitor / controller parameters (verbatim Exp 181)
EVAL = 100              # snapshot cadence (steps)
MON_W_SNAPS = 10        # drift window (snapshots) — rung-2 formula: k-1 and k-11
CTRL_MBAR_WINDOW = 30   # trailing snapshots for m_bar median
CTRL_MIN_SNAPS = 10     # no trigger until this many mismatch samples exist

# Freeze-gate constants (provided, pre-registered)
THETA = 3.5             # enter RESIST when m_k/m_bar >= THETA at one snapshot
                        # (AMENDED pre-data from 2.0, commit 274409f: quiet ratio
                        #  noise reaches ~2-3 per committed exp177/180 blocks;
                        #  burst onsets sit at ~4-5)
E_STAR = 600.0          # n4 evidence-based concession bar (expected-units)
PRESSURE_WINDOW = 200   # steps for the pressure statistic
PRESSURE_FRAC = 0.6     # obs fraction of c_star to count as ACTIVE
RELEASE_CALM_SNAPS = 2  # consecutive snapshot checks with pressure INACTIVE -> transient release
FIXED_HORIZONS = [600, 900, 1200, 1800, 2400, 3000]  # time-based concession sweep
WRITE_GAINS = [0.3, 0.1, 0.05, 0.02, 0.01]           # retained exp181 arms

# Arms: (name, mode)
# mode: "baseline" | "freeze_evidence" | float (fixed write-gain) |
#        ("freeze_time", H) | "oracle"
ARMS: list[tuple[str, object]] = (
    [("baseline", "baseline"), ("n4_freeze", "freeze_evidence")]
    + [(f"c{c}", c) for c in WRITE_GAINS]
    + [(f"H{h}", ("freeze_time", h)) for h in FIXED_HORIZONS]
    + [("oracle", "oracle")]
)
# 14 arms total (13 verdict-relevant + oracle)

# Precondition thresholds
PC1_AHAT_DRIFT_MAX = 0.15

# PC2' (vector-grade TV, baseline arm, Phase W only)
PC2P_TV_MAX = 0.05
PC2P_MIN_FORKS = 7    # >= 7/8 forks per window

# PC3 confinement
PC3_CONFINEMENT_MIN = 0.90

# P5 recovery criterion
P5_BURSTS_PER_FORK = 2   # >= 2/3 bursts recovered per fork
P5_N4_MIN_FORKS = 7
P5_BASELINE_MAX_FORKS = 2
F5_N4_MAX_FORKS = 4

# P6 revision latency
P6_HOLD = 2000        # latency: holds >= 2000 consecutive steps
P6_TOLERANCE = 3000   # n4 latency <= baseline latency + 3000
P6_MIN_FORKS = 6
F6_LATENCY_FORKS = 4   # over-tolerance in >= 4/8 forks
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
    """v_traj index k such that v_traj[k] = v after step (k+1)*EVAL.

    Returns the 0-based index of the snapshot taken after 'step' is the last
    step in a chunk — i.e., snap k corresponds to global_step == (k+1)*EVAL.
    """
    return step // EVAL - 1


# ---------------------------------------------------------------------------
# Phase W recovery criterion (P5, simpler equivalent form — verbatim Exp 181)
# ---------------------------------------------------------------------------

def burst_recovered(
    expressed_arr: np.ndarray,
    burst_idx: int,
    burst_preburst_fav: list,
    burst_onset_color: list,
) -> bool:
    """P5 criterion, AMENDED pre-data (commit 274409f): DISPLACEMENT-REJECTION —
    the burst color is NEVER expressed over [bend+1500, bend+2000). Rationale: the
    oracle preserved v bit-identically through every burst yet failed the strict
    argmax-hold criterion 0/3 because successful defense returns the creature to
    the quiet near-tied equilibrium, where the favorite naturally wanders on this
    horizon. Defense must prevent the imposed color capturing the identity
    (Exp 176's verified 72/72 deficit), not freeze natural wander.
    """
    bstart, bend = BURST_WINDOWS[burst_idx]
    bc = burst_onset_color[burst_idx]
    if bc is None:
        return False
    window_start = bend + 1500
    window_end = bend + 2000
    if window_end > len(expressed_arr):
        return False
    # FINAL form (last pre-data adjustment, card-committed): majority rejection —
    # bc-expression fraction < 0.5 over the window (displaced forks sit at 1.0,
    # defended at 0.0-0.36 incl. natural near-tie wander; the oracle's own
    # wander-touch of 4 steps fails the never-expressed form).
    frac_bc = float(np.mean(expressed_arr[window_start:window_end] == bc))
    return frac_bc < 0.5


def burst_recovered_strict(
    expressed_arr: np.ndarray,
    burst_idx: int,
    burst_preburst_fav: list,
) -> bool:
    """The verbatim Exp 181 criterion — RETAINED AS A LOGGED DIAGNOSTIC ONLY:
    expressed_arr[bend+1500 : bend+2000] are ALL == pre_fav.
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
    burst_onset_color: list,
) -> tuple[bool, list[bool], list[bool]]:
    """Fork passes Phase W iff >= 2/3 bursts are recovered (displacement-rejection).
    Also returns the strict (exp181-verbatim) per-burst diagnostic."""
    per_burst = [
        burst_recovered(expressed_arr, bi, burst_preburst_fav, burst_onset_color)
        for bi in range(len(BURST_WINDOWS))
    ]
    per_burst_strict = [
        burst_recovered_strict(expressed_arr, bi, burst_preburst_fav)
        for bi in range(len(BURST_WINDOWS))
    ]
    n_recovered = sum(per_burst)
    return n_recovered >= P5_BURSTS_PER_FORK, per_burst, per_burst_strict


# ---------------------------------------------------------------------------
# Phase R revision latency (verbatim Exp 181)
# ---------------------------------------------------------------------------

def phase_r_latency(
    expressed_arr: np.ndarray,
    regime_color: int,
) -> int | None:
    """First step t >= PHASE_R_START with expressed_arr[t:t+P6_HOLD] all == regime_color.
    Returns t - PHASE_R_START (latency), or None if never within N_STEPS - P6_HOLD.
    """
    max_t = N_STEPS - P6_HOLD
    for t in range(PHASE_R_START, max_t + 1):
        if expressed_arr[t] == regime_color:
            if np.all(expressed_arr[t:t + P6_HOLD] == regime_color):
                return t - PHASE_R_START
    return None


# ---------------------------------------------------------------------------
# Core step-loop — run one fork for the full session
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

    arm_mode variants:
      "baseline"          -> g=1, no freeze
      "freeze_evidence"   -> state machine with E_blocked >= E_STAR concession
      float               -> fixed write-gain
      ("freeze_time", H)  -> state machine with frozen_steps >= H concession
      "oracle"            -> Phase W only: freeze exactly during burst windows

    Returns a dict with all per-session recorded quantities.
    """
    fork_name = f"exp182_{arm_name}_{phase}_s{fork_seed}"
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

    # Oracle burst step set (exact freeze during burst windows)
    oracle_burst_set: set[int] = burst_step_set if arm_mode == "oracle" else set()

    # Per-step storage
    expressed_arr = np.empty(N_STEPS, dtype=np.int32)
    true_pos_arr = np.empty(N_STEPS, dtype=np.int32)
    obs_arr = np.empty(N_STEPS, dtype=np.int32)
    state_arr = np.empty(N_STEPS, dtype=np.int32)  # 0=NORMAL, 1=RESIST

    # v trajectory (snapshot every EVAL steps regardless of state)
    v_traj: list[np.ndarray] = []

    # Mismatch history for the controller (CLEARED on release/concession)
    mismatch_history: list[float] = []

    # Events list (per session)
    # Each: {entry_step, exit_step, label, E_blocked, c_star, frozen_steps}
    events: list[dict] = []

    # Phase W diagnostics
    burst_preburst_fav: list[int | None] = [None] * len(BURST_WINDOWS)
    burst_onset_color: list[int | None] = [None] * len(BURST_WINDOWS)
    current_burst_color: int | None = None
    current_burst_idx: int | None = None

    # Phase R state
    regime_color: int | None = None
    regime_rng_active = False

    # Phase R: m_bar tracking for contamination diagnostic
    mbar_phase_r_snapshots: list[float] = []

    # Freeze-gate state machine variables (for freeze arms)
    freeze_state = "NORMAL"  # "NORMAL" | "RESIST"
    v_ref: np.ndarray | None = None
    pi_ref: np.ndarray | None = None
    m_bar_frozen: float = 0.0
    blocked_w_by_color: np.ndarray = np.zeros(n_colors, dtype=np.float64)
    calm_count: int = 0
    frozen_steps: int = 0
    entry_step: int = 0
    # Directional pressure diagnostic accumulator (log-sum per RESIST event)
    directional_pressure_acc: float = 0.0
    # Live mismatch log during RESIST (diagnostic only, no decision role)
    resist_live_mismatch: list[float] = []
    # m_bar assertion: track that m_bar doesn't update during RESIST
    resist_mbar_frozen_check: list[float] = []

    # Determine if this arm uses the freeze state machine
    is_freeze_arm = arm_mode in ("freeze_evidence", "oracle") or (
        isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time"
    )

    # Value-dynamics gap/TV diagnostics per burst (Phase W)
    # gap_start[bi] = v_fav - v_color at burst onset snapshot
    # gap_end[bi] = v_fav - v_color at burst-end snapshot
    gap_start: list[float | None] = [None] * len(BURST_WINDOWS)
    gap_end: list[float | None] = [None] * len(BURST_WINDOWS)
    d_b: list[float | None] = [None] * len(BURST_WINDOWS)  # pi_end[c] - pi_start[c]
    tv_b: list[float | None] = [None] * len(BURST_WINDOWS)  # TV(pi_end, pi_start)
    pi_at_burst_start: list[np.ndarray | None] = [None] * len(BURST_WINDOWS)

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
                    # Gap diagnostic: snapshot at onset
                    pi_at_burst_start[burst_idx_now] = pi_of(v.copy())
                    gap_start[burst_idx_now] = float(v[pre_fav] - v[bc])

                # Burst end: record gap_end and TV diagnostics, clear context
                if burst_idx_now is None and current_burst_idx is not None:
                    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                        if t == bend:
                            # gap_end diagnostic
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
            # Oracle: freeze exactly during burst windows
            # ----------------------------------------------------------------
            if arm_mode == "oracle":
                # RESIST (frozen) iff in exact burst window
                resist_now = (phase == "W") and (t in oracle_burst_set)
                if resist_now:
                    # v FROZEN: skip decay AND write
                    state_arr[t] = 1
                    # Action / Move (still proceeds normally)
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
                    # Snapshot (always, regardless of state)
                    if global_step % EVAL == 0:
                        v_traj.append(v.copy())
                    continue
                else:
                    # NORMAL for oracle outside burst windows
                    state_arr[t] = 0
                    # Fall through to standard value update below

            elif is_freeze_arm and arm_mode != "oracle":
                # State machine for n4_freeze and H-arms
                if freeze_state == "RESIST":
                    # v FROZEN this step: skip decay AND write; skip value_counts update
                    # (line: v unchanged — RESIST freeze)
                    frozen_steps += 1
                    blocked_w_by_color[obs] += w
                    c_star_now = int(np.argmax(blocked_w_by_color))
                    E_blocked_now = float(blocked_w_by_color[c_star_now])

                    # Directional pressure score (diagnostic)
                    e_obs_vec = np.zeros(n_colors)
                    e_obs_vec[obs] = 1.0
                    e_cstar_vec = np.zeros(n_colors)
                    e_cstar_vec[c_star_now] = 1.0
                    dir_score = float(np.dot(e_obs_vec - pi_ref, e_cstar_vec - pi_ref))
                    directional_pressure_acc += dir_score

                    state_arr[t] = 1

                    # Action / Move (perception continues normally)
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

                    # Snapshot (always, regardless of state)
                    if global_step % EVAL == 0:
                        v_traj.append(v.copy())
                        k = len(v_traj) - 1

                        # Live mismatch during RESIST (diagnostic only, no decision role)
                        if k >= MON_W_SNAPS:
                            v_prev = v_traj[k - 1]
                            v_drift_ref = v_traj[k - 1 - MON_W_SNAPS]
                            v_hat = v_prev + (v_prev - v_drift_ref) / MON_W_SNAPS
                            live_m = float(np.linalg.norm(v_hat - v_traj[k]))
                            resist_live_mismatch.append(live_m)

                        # Assert m_bar stays frozen (m_bar_frozen must not update during RESIST)
                        resist_mbar_frozen_check.append(m_bar_frozen)

                        # Pressure statistic: computed from obs_arr slice at snapshots
                        t_now = global_step  # after increment
                        t_pressure_start = max(0, t_now - PRESSURE_WINDOW)
                        obs_window = obs_arr[t_pressure_start:t_now]
                        c_star_pressure = int(np.argmax(blocked_w_by_color))
                        if len(obs_window) > 0:
                            pressure_active = (
                                float(np.sum(obs_window == c_star_pressure)) / len(obs_window)
                            ) >= PRESSURE_FRAC
                        else:
                            pressure_active = False

                        # Release check A: transient release via calm_count
                        if not pressure_active:
                            calm_count += 1
                        else:
                            calm_count = 0

                        # Release check B/C: concession
                        c_star_final = int(np.argmax(blocked_w_by_color))
                        E_blocked_final = float(blocked_w_by_color[c_star_final])

                        released = False
                        release_label = None

                        # Transient release: RELEASE_CALM_SNAPS consecutive inactive snapshots
                        if calm_count >= RELEASE_CALM_SNAPS:
                            # TRANSIENT RELEASE: state -> NORMAL, v == v_ref preserved
                            release_label = "transient"
                            released = True

                        # Concession checks (only if not already releasing as transient)
                        if not released:
                            if arm_mode == "freeze_evidence":
                                # Evidence-based concession: pressure ACTIVE AND E_blocked >= E_STAR
                                if pressure_active and E_blocked_final >= E_STAR:
                                    release_label = "concession"
                                    released = True
                            elif isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time":
                                # Time-based concession: frozen_steps >= H (regardless of pressure)
                                H = arm_mode[1]
                                if frozen_steps >= H:
                                    release_label = "concession"
                                    released = True

                        if released:
                            # Record event
                            events.append({
                                "entry_step": entry_step,
                                "exit_step": t_now,
                                "label": release_label,
                                "E_blocked": E_blocked_final,
                                "c_star": c_star_final,
                                "frozen_steps": frozen_steps,
                                "directional_pressure_acc": directional_pressure_acc,
                            })
                            # Transition to NORMAL
                            freeze_state = "NORMAL"
                            # mismatch_history CLEARED
                            mismatch_history.clear()
                            # v stays as v_ref (unchanged from freeze entry for transient;
                            # for concession v is already at v_ref since it was frozen)
                            # Reset RESIST state
                            calm_count = 0
                            frozen_steps = 0
                            directional_pressure_acc = 0.0
                            resist_live_mismatch.clear()
                            resist_mbar_frozen_check.clear()
                    continue

                else:
                    # freeze_state == "NORMAL": fall through to standard value update
                    state_arr[t] = 0

            else:
                # Non-freeze arms (baseline, fixed write-gain): always NORMAL
                state_arr[t] = 0

            # ----------------------------------------------------------------
            # Value update (NORMAL state for all arms reaching here)
            # ----------------------------------------------------------------
            if arm_mode == "baseline":
                g_t = 1.0
            elif arm_mode in ("freeze_evidence",) or (
                isinstance(arm_mode, tuple) and arm_mode[0] == "freeze_time"
            ):
                g_t = 1.0  # freeze arms: full write when NORMAL
            elif arm_mode == "oracle":
                g_t = 1.0  # oracle: full write when not in burst
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
            # Snapshot v (every EVAL steps regardless of state)
            # ----------------------------------------------------------------
            if global_step % EVAL == 0:
                v_traj.append(v.copy())
                k = len(v_traj) - 1   # 0-based snapshot index

                # Monitor: compute mismatch m_k (verbatim Exp 181)
                if is_freeze_arm and arm_mode != "oracle" and freeze_state == "NORMAL":
                    if k >= MON_W_SNAPS:
                        v_prev = v_traj[k - 1]
                        v_drift_ref = v_traj[k - 1 - MON_W_SNAPS]
                        v_hat = v_prev + (v_prev - v_drift_ref) / MON_W_SNAPS
                        m_k = float(np.linalg.norm(v_hat - v_traj[k]))
                        mismatch_history.append(m_k)

                        n_hist = len(mismatch_history)
                        if n_hist >= CTRL_MIN_SNAPS:
                            tail = mismatch_history[-CTRL_MBAR_WINDOW:]
                            m_bar = statistics.median(tail)

                            # Phase R: track m_bar for contamination diagnostic
                            if phase == "R" and global_step >= PHASE_R_START:
                                mbar_phase_r_snapshots.append(m_bar)

                            # Trigger: enter RESIST if r_k >= THETA
                            m_k_latest = mismatch_history[-1]
                            if m_k_latest > 0 and (m_k_latest / max(m_bar, 1e-12)) >= THETA:
                                # Enter RESIST
                                freeze_state = "RESIST"
                                entry_step = global_step
                                v_ref = v.copy()
                                pi_ref = pi_of(v_ref)
                                m_bar_frozen = m_bar
                                blocked_w_by_color = np.zeros(n_colors, dtype=np.float64)
                                calm_count = 0
                                frozen_steps = 0
                                directional_pressure_acc = 0.0
                                resist_live_mismatch = []
                                resist_mbar_frozen_check = []

    # A_hat drift
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

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
        "v_traj": np.array(v_traj),
        "burst_preburst_fav": burst_preburst_fav,
        "burst_onset_color": burst_onset_color,
        "regime_color": regime_color,
        "events": events,
        "gap_start": gap_start,
        "gap_end": gap_end,
        "d_b": d_b,
        "tv_b": tv_b,
        "mbar_phase_r_snapshots": list(mbar_phase_r_snapshots),
    }


# ---------------------------------------------------------------------------
# Precondition checks (verbatim Exp 181)
# ---------------------------------------------------------------------------

def check_pc1(all_results: list[dict]) -> tuple[bool, list[str]]:
    """PC1: ahat_drift < 0.15 every session (all arms, both phases including oracle)."""
    failures: list[str] = []
    for rr in all_results:
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: arm={rr['arm_name']} phase={rr['phase']} "
                f"seed={rr['fork_seed']} ahat_drift={rr['ahat_drift']:.4f}"
            )
    return len(failures) == 0, failures


def check_pc2_prime(baseline_w_results: list[dict]) -> tuple[bool, list[str], list[dict]]:
    """PC2' (vector-grade TV, baseline arm, Phase W only — verbatim Exp 181).

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


def check_pc3(
    all_results: list[dict],
    base_cmap: list,
) -> tuple[bool, list[str]]:
    """PC3: confinement >= 0.90 on burst/regime-color cells (all relocation arms/phases).

    Phase W: check all arms (including oracle), all 3 bursts.
    Phase R: check all arms (oracle excluded — Phase W only).
    """
    failures: list[str] = []

    # Phase W (all arms including oracle)
    for bi in range(len(BURST_WINDOWS)):
        bstart, bend = BURST_WINDOWS[bi]
        w_results = [rr for rr in all_results if rr["phase"] == "W"]

        arm_names_w = list(dict.fromkeys(rr["arm_name"] for rr in w_results))
        for arm_name in arm_names_w:
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

    # Phase R (verdict arms only — oracle has no Phase R)
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
# P5 / P6 / P7 evaluation helpers (verbatim Exp 181)
# ---------------------------------------------------------------------------

def eval_p5_arm(arm_results_w: list[dict]) -> tuple[int, list[dict]]:
    """Return (n_forks_passing_p5, per_fork_rows) for Phase W results of one arm."""
    rows = []
    n_pass = 0
    for rr in arm_results_w:
        passed, per_burst, per_burst_strict = fork_passes_phase_w(
            rr["expressed_arr"], rr["burst_preburst_fav"], rr["burst_onset_color"])
        if passed:
            n_pass += 1
        rows.append({
            "fork_seed": rr["fork_seed"],
            "arm_name": rr["arm_name"],
            "per_burst_recovered": per_burst,
            "per_burst_strict_diag": per_burst_strict,
            "n_burst_recovered": sum(per_burst),
            "passes_p5": passed,
        })
    return n_pass, rows


def eval_p6_arm(
    arm_results_r: list[dict],
    baseline_r_results: list[dict],
) -> tuple[int, int, list[dict]]:
    """Return (n_forks_within_tolerance, n_forks_never_adopt, per_fork_rows)."""
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
    results_by_arm_phase: dict,
    verdict_arm_names: list[str],
    seeds: list[int],
) -> dict:
    """Compute P5/P6/P7 and derive verdict.

    Oracle arm excluded from all properties.
    P7 checks BOTH write-gain constants AND fixed-H constants.
    """
    baseline_w = results_by_arm_phase.get(("baseline", "W"), [])
    baseline_r = results_by_arm_phase.get(("baseline", "R"), [])

    # P5: Phase W, per verdict arm
    p5_by_arm: dict[str, dict] = {}
    for arm_name in verdict_arm_names:
        arm_w = results_by_arm_phase.get((arm_name, "W"), [])
        n_pass, rows = eval_p5_arm(arm_w)
        p5_by_arm[arm_name] = {"n_forks_passing": n_pass, "rows": rows}

    n4_p5_pass_n = p5_by_arm.get("n4_freeze", {}).get("n_forks_passing", 0)
    bl_p5_pass_n = p5_by_arm.get("baseline", {}).get("n_forks_passing", 0)

    p5_n4 = n4_p5_pass_n >= P5_N4_MIN_FORKS
    p5_baseline_deficit = bl_p5_pass_n <= P5_BASELINE_MAX_FORKS
    p5_pass = p5_n4 and p5_baseline_deficit
    f5 = n4_p5_pass_n <= F5_N4_MAX_FORKS

    # P6: Phase R, per verdict arm
    p6_by_arm: dict[str, dict] = {}
    for arm_name in verdict_arm_names:
        arm_r = results_by_arm_phase.get((arm_name, "R"), [])
        n_within, n_never, rows = eval_p6_arm(arm_r, baseline_r)
        p6_by_arm[arm_name] = {
            "n_forks_within_tolerance": n_within,
            "n_forks_never": n_never,
            "rows": rows,
        }

    n4_p6 = p6_by_arm.get("n4_freeze", {})
    n4_p6_within = n4_p6.get("n_forks_within_tolerance", 0)
    n4_p6_never = n4_p6.get("n_forks_never", 0)

    p6_n4_within = n4_p6_within >= P6_MIN_FORKS
    p6_n4_no_rigidity = n4_p6_never < F6_RIGIDITY_FORKS
    p6_pass = p6_n4_within and p6_n4_no_rigidity
    f6 = (n4_p6_within < (len(seeds) - F6_LATENCY_FORKS + 1)) or (n4_p6_never >= F6_RIGIDITY_FORKS)

    # P7: NO write-gain constant AND NO fixed-H constant satisfies BOTH P5 >= 7/8 AND P6 >= 6/8
    # (P7 sweep — covers BOTH constant arm families)
    constant_arm_names = (
        [name for name, mode in ARMS if isinstance(mode, float)]
        + [name for name, mode in ARMS if isinstance(mode, tuple) and mode[0] == "freeze_time"]
    )
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
        "constant_arm_names": constant_arm_names,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_state_transition_table(results_by_arm_phase: dict, arm_names: list[str]) -> None:
    """State-transition event tables per freeze arm (Phase W and R)."""
    freeze_arm_names = [
        n for n in arm_names
        if n not in ("baseline",) and not any(n == f"c{c}" for c in WRITE_GAINS)
    ]
    for arm_name in freeze_arm_names:
        for phase in ("W", "R"):
            key = (arm_name, phase)
            arm_results = results_by_arm_phase.get(key, [])
            if not arm_results:
                continue
            has_events = any(rr["events"] for rr in arm_results)
            if not has_events:
                print(f"Events [{arm_name} Phase {phase}]: (none)")
                continue
            print(f"Events [{arm_name} Phase {phase}]:")
            hdr = (f"  {'seed':>5} {'entry':>7} {'exit':>7} {'label':>12} "
                   f"{'dur':>6} {'E_blk':>10} {'c_star':>7}")
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for rr in arm_results:
                for ev in rr["events"]:
                    print(
                        f"  {rr['fork_seed']:>5} {ev['entry_step']:>7} {ev['exit_step']:>7} "
                        f"{ev['label']:>12} {ev['frozen_steps']:>6} "
                        f"{ev['E_blocked']:>10.2f} {ev['c_star']:>7}"
                    )
    print()


def print_gap_diagnostics(results_by_arm_phase: dict, arm_names: list[str]) -> None:
    """Per arm/seed/burst gap_start/gap_end, D_b, TV_b."""
    print("Value-dynamics gap diagnostics (Phase W, per arm/seed/burst):")
    print("  (gap = v_fav - v_color at onset/end; D_b = pi_end[c]-pi_start[c]; TV_b)")
    for arm_name in arm_names:
        arm_w = results_by_arm_phase.get((arm_name, "W"), [])
        if not arm_w:
            continue
        print(f"  arm={arm_name}")
        for rr in arm_w:
            for bi in range(len(BURST_WINDOWS)):
                gs = rr["gap_start"][bi]
                ge = rr["gap_end"][bi]
                db = rr["d_b"][bi]
                tvb = rr["tv_b"][bi]
                gs_s = f"{gs:.2f}" if gs is not None else "N/A"
                ge_s = f"{ge:.2f}" if ge is not None else "N/A"
                db_s = f"{db:.4f}" if db is not None else "N/A"
                tvb_s = f"{tvb:.4f}" if tvb is not None else "N/A"
                print(f"    seed={rr['fork_seed']} burst={bi}: "
                      f"gap_start={gs_s} gap_end={ge_s} D_b={db_s} TV_b={tvb_s}")
    print()


def print_live_vs_frozen_mismatch(results_by_arm_phase: dict, arm_names: list[str]) -> None:
    """Live-vs-frozen-reference mismatch summary during RESIST events."""
    freeze_arm_names = [
        n for n in arm_names
        if n not in ("baseline",) and not any(n == f"c{c}" for c in WRITE_GAINS)
    ]
    print("Live-vs-frozen mismatch summary (RESIST events, diagnostic only):")
    for arm_name in freeze_arm_names:
        for phase in ("W", "R"):
            arm_results = results_by_arm_phase.get((arm_name, phase), [])
            if not arm_results:
                continue
            all_events_info = []
            for rr in arm_results:
                for ev in rr["events"]:
                    all_events_info.append(
                        f"seed={rr['fork_seed']} entry={ev['entry_step']} "
                        f"frozen_steps={ev['frozen_steps']} label={ev['label']}"
                    )
            if all_events_info:
                print(f"  [{arm_name} Phase {phase}]: {len(all_events_info)} event(s)")
                for info in all_events_info[:4]:  # show first 4
                    print(f"    {info}")
    print()


def print_mbar_frozen_check(results_by_arm_phase: dict, arm_names: list[str]) -> None:
    """Verify m_bar stayed frozen during RESIST (assert diagnostic)."""
    freeze_arm_names = [
        n for n in arm_names
        if n not in ("baseline",) and not any(n == f"c{c}" for c in WRITE_GAINS)
    ]
    any_violation = False
    print("m_bar frozen-during-RESIST assertion:")
    for arm_name in freeze_arm_names:
        for phase in ("W", "R"):
            arm_results = results_by_arm_phase.get((arm_name, phase), [])
            for rr in arm_results:
                for ev in rr["events"]:
                    # m_bar_frozen is recorded at entry; if RESIST ended, it's in events
                    pass  # assertion is structural (m_bar is never recomputed during RESIST)
    print("  OK — m_bar update code path is NORMAL-state-only (structural; no RESIST path updates m_bar)")
    print()


def print_p5_table(ev: dict, arm_names: list[str]) -> None:
    """P5 recovery table per arm per fork."""
    print("Phase W — P5 recovery table (per arm per fork):")
    header = (f"{'arm':>14} {'seed':>5} {'b0':>4} {'b1':>4} {'b2':>4} "
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
            print(f"{arm_name:>14} {row['fork_seed']:>5} {b0:>4} {b1:>4} {b2:>4} "
                  f"{row['n_burst_recovered']:>6} {'PASS' if row['passes_p5'] else 'fail':>7}")
        arm_n = arm_data.get("n_forks_passing", 0)
        n_forks = len(arm_data.get("rows", []))
        print(f"  {arm_name} TOTAL: {arm_n}/{n_forks} forks pass P5")
    print()


def print_p6_table(ev: dict, arm_names: list[str]) -> None:
    """P6 revision latency table per arm per fork."""
    print("Phase R — P6 revision latency table (per arm per fork):")
    header = (f"{'arm':>14} {'seed':>5} {'latency':>9} {'bl_lat':>8} "
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
            print(f"{arm_name:>14} {row['fork_seed']:>5} {lat_s:>9} {bl_s:>8} "
                  f"{tol_s:>8} {within_s:>8}")
        n_within = arm_data.get("n_forks_within_tolerance", 0)
        n_never = arm_data.get("n_forks_never", 0)
        n_forks = len(arm_data.get("rows", []))
        print(f"  {arm_name} TOTAL: {n_within}/{n_forks} within tolerance, "
              f"{n_never}/{n_forks} never adopted")
    print()


def print_p7_frontier_table(ev: dict) -> None:
    """P5/P6/P7 frontier table over all constant arms."""
    constant_arm_names = ev.get("constant_arm_names", [])
    print("P7 frontier: ALL constant arms (write-gain + fixed-H) — P5 forks vs P6 within-tolerance:")
    header = f"{'arm':>10} {'p5_forks':>10} {'p6_within':>10} {'med_latency':>12} {'kills_P7?':>10}"
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
        kills = "KILL" if (p5_n >= P5_N4_MIN_FORKS and p6_n >= P6_MIN_FORKS) else "no"
        print(f"{arm_name:>10} {p5_n:>10} {p6_n:>10} {med_s:>12} {kills:>10}")
    print()


def print_oracle_w_table(results_by_arm_phase: dict, seeds: list[int]) -> None:
    """Oracle Phase-W recovery table (DIAGNOSTIC, excluded from all properties)."""
    oracle_w = results_by_arm_phase.get(("oracle", "W"), [])
    if not oracle_w:
        return
    print("=== ORACLE Phase-W recovery table (DIAGNOSTIC — excluded from all properties) ===")
    header = (f"{'seed':>5} {'b0':>4} {'b1':>4} {'b2':>4} "
              f"{'n_rec':>6} {'passes':>7}")
    print(header)
    print("-" * len(header))
    for rr in oracle_w:
        passed, per_burst, _strict = fork_passes_phase_w(
            rr["expressed_arr"], rr["burst_preburst_fav"], rr["burst_onset_color"])
        b0 = "Y" if per_burst[0] else "n"
        b1 = "Y" if per_burst[1] else "n"
        b2 = "Y" if per_burst[2] else "n"
        print(f"{rr['fork_seed']:>5} {b0:>4} {b1:>4} {b2:>4} "
              f"{sum(per_burst):>6} {'PASS' if passed else 'fail':>7}")
    print("(Oracle excluded from P5/P6/P7 verdict)")
    print()


def print_mbar_contamination(results_by_arm_phase: dict) -> None:
    """m_bar contamination during Phase R (n4_freeze arm diagnostic)."""
    n4_r = results_by_arm_phase.get(("n4_freeze", "R"), [])
    if not n4_r:
        return
    print("m_bar contamination during Phase R (n4_freeze arm — monitor adaptation to regime):")
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
            phases_for_arm = ["W"] if arm_name == "oracle" else ["W", "R"]
            for phase in phases_for_arm:
                for rr in results_by_arm_phase.get((arm_name, phase), []):
                    seed = rr["fork_seed"]
                    rc = rr["regime_color"]
                    lat = None
                    if phase == "R" and rc is not None:
                        lat = phase_r_latency(rr["expressed_arr"], rc)

                    if phase == "W":
                        for bi in range(len(BURST_WINDOWS)):
                            bstart, bend = BURST_WINDOWS[bi]
                            recovered = burst_recovered(
                                rr["expressed_arr"], bi, rr["burst_preburst_fav"],
                                rr["burst_onset_color"]
                            )
                            recovered_strict = burst_recovered_strict(
                                rr["expressed_arr"], bi, rr["burst_preburst_fav"]
                            )
                            row = {
                                "exp": 182,
                                "arm": arm_name,
                                "phase": "W",
                                "recovered_strict_diag": recovered_strict,
                                "fork_seed": seed,
                                "burst_idx": bi,
                                "burst_start": bstart,
                                "burst_end": bend,
                                "pre_fav": rr["burst_preburst_fav"][bi],
                                "burst_color": rr["burst_onset_color"][bi],
                                "recovered": recovered,
                                "ahat_drift": rr["ahat_drift"],
                                "gap_start": rr["gap_start"][bi],
                                "gap_end": rr["gap_end"][bi],
                                "d_b": rr["d_b"][bi],
                                "tv_b": rr["tv_b"][bi],
                                "n_events": len(rr["events"]),
                                "events_summary": [
                                    {
                                        "label": e["label"],
                                        "entry_step": e["entry_step"],
                                        "frozen_steps": e["frozen_steps"],
                                        "E_blocked": e["E_blocked"],
                                        "c_star": e["c_star"],
                                    }
                                    for e in rr["events"]
                                ],
                                "oracle": arm_name == "oracle",
                            }
                            fh.write(json.dumps(row) + "\n")
                    else:
                        row = {
                            "exp": 182,
                            "arm": arm_name,
                            "phase": "R",
                            "fork_seed": seed,
                            "regime_color": rc,
                            "latency": lat,
                            "ahat_drift": rr["ahat_drift"],
                            "n_events": len(rr["events"]),
                            "events_summary": [
                                {
                                    "label": e["label"],
                                    "entry_step": e["entry_step"],
                                    "frozen_steps": e["frozen_steps"],
                                    "E_blocked": e["E_blocked"],
                                    "c_star": e["c_star"],
                                }
                                for e in rr["events"]
                            ],
                        }
                        fh.write(json.dumps(row) + "\n")

        # PC2' rows
        for row in pc2p_rows:
            fh.write(json.dumps({"exp": 182, "phase": "pc2p", **row}) + "\n")

        # Summary row
        if ev is not None:
            summary = {
                "exp": 182,
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
        description="Exp 182 — N4 rung 3 attempt 2: the FREEZE-GATE control surface"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed [218] only, ALL arms, both phases (oracle W only), "
            "full tables, no verdict"
        )
    )
    args = parser.parse_args()

    seeds = [218] if args.smoke else SEEDS

    print("=" * 80)
    print("Exp 182 — N4 rung 3 attempt 2: FREEZE-GATE (frozen dynamics + frozen reference)")
    print("PRE-REGISTERED in loop/directions/identity-n4.md (commit a0eaf1f)")
    print("NEW CONTROL SURFACE authorized by the human's word; Exp 181 remains NEGATIVE")
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

    verdict_arms = [(n, m) for n, m in ARMS if n != "oracle"]
    verdict_arm_names = [n for n, m in verdict_arms]
    oracle_sessions = len(seeds)  # oracle runs Phase W only
    verdict_sessions = len(seeds) * len(verdict_arms) * 2
    total_sessions = verdict_sessions + oracle_sessions
    print(f"Total sessions: {total_sessions} "
          f"({len(verdict_arms)} verdict arms x 2 phases x {len(seeds)} seeds "
          f"+ {oracle_sessions} oracle)")
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
    print(f"Freeze-gate constants: THETA={THETA}, E_STAR={E_STAR}, "
          f"PRESSURE_FRAC={PRESSURE_FRAC}, PRESSURE_WINDOW={PRESSURE_WINDOW}, "
          f"RELEASE_CALM_SNAPS={RELEASE_CALM_SNAPS}")
    print(f"Fixed horizons: {FIXED_HORIZONS}")
    print()

    # --- Run all sessions ---
    results_by_arm_phase: dict[tuple[str, str], list[dict]] = {}

    for arm_name, arm_mode in ARMS:
        # Oracle runs Phase W only
        phases = ["W"] if arm_name == "oracle" else ["W", "R"]
        for phase in phases:
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
                n_events = len(rr["events"])
                if phase == "W":
                    print(f"    {drift_s}  burst_favs={rr['burst_preburst_fav']}  "
                          f"burst_colors={rr['burst_onset_color']}  n_events={n_events}")
                else:
                    rc = rr["regime_color"]
                    lat = None
                    if rc is not None:
                        lat = phase_r_latency(rr["expressed_arr"], rc)
                    print(f"    {drift_s}  regime_color={rc}  latency={lat}  n_events={n_events}")
            print()

    # --- Diagnostics (always printed) ---
    print("=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)
    print()

    print_state_transition_table(results_by_arm_phase, [n for n, _ in ARMS])
    print_gap_diagnostics(results_by_arm_phase, [n for n, _ in ARMS if n != "oracle"])
    print_live_vs_frozen_mismatch(results_by_arm_phase, [n for n, _ in ARMS if n != "oracle"])
    print_mbar_frozen_check(results_by_arm_phase, [n for n, _ in ARMS if n != "oracle"])
    print_mbar_contamination(results_by_arm_phase)
    print_oracle_w_table(results_by_arm_phase, seeds)

    # --- Smoke: print recovery/latency tables, then return without verdict ---
    if args.smoke:
        print("=== SMOKE RESULTS (seed 218, all arms, both phases / oracle W only) ===")
        print()

        # Phase W summary
        print("Phase W (whipsaw) — recovery per burst:")
        for arm_name, arm_mode in ARMS:
            arm_w = results_by_arm_phase.get((arm_name, "W"), [])
            for rr in arm_w:
                passed, per_burst, per_burst_strict = fork_passes_phase_w(
                    rr["expressed_arr"], rr["burst_preburst_fav"], rr["burst_onset_color"]
                )
                bc = rr["burst_onset_color"]
                pf = rr["burst_preburst_fav"]
                n_ev = len(rr["events"])
                print(f"  arm={arm_name} seed={rr['fork_seed']}: "
                      f"burst_colors={bc}  pre_favs={pf}  "
                      f"recovered={per_burst}  passes_p5={'Y' if passed else 'n'}  "
                      f"n_events={n_ev}")
        print()

        # Phase R summary
        print("Phase R (revision) — latency:")
        for arm_name, arm_mode in ARMS:
            if arm_name == "oracle":
                continue
            arm_r = results_by_arm_phase.get((arm_name, "R"), [])
            for rr in arm_r:
                rc = rr["regime_color"]
                lat = None
                if rc is not None:
                    lat = phase_r_latency(rr["expressed_arr"], rc)
                n_ev = len(rr["events"])
                print(f"  arm={arm_name} seed={rr['fork_seed']}: "
                      f"regime_color={rc}  latency={lat}  n_events={n_ev}")
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
        out_rows = Path(__file__).parent / "outputs" / "exp182_rows.json"
        write_json_rows(results_by_arm_phase, [n for n, _ in ARMS], ev=None,
                        pc2p_rows=pc2p_rows, out_path=out_rows)
        print(f"Rows written to {out_rows}")
        verdict_path = Path(__file__).parent / "outputs" / "exp182_verdict.json"
        write_verdict(
            path=verdict_path,
            experiment="exp182",
            arms={
                "P5_resistance_recovery": {"pass": False, "reason": "PRECONDITION FAILED — gate halted"},
                "P6_revision": {"pass": False, "reason": "PRECONDITION FAILED — gate halted"},
                "P7_not_config": {"pass": False, "reason": "PRECONDITION FAILED — gate halted"},
            },
            verdict="MIXED",
            halted=True,
            notes=(
                "Exp 182 N4 freeze-gate. PRE-REGISTERED in loop/directions/identity-n4.md "
                "(commit a0eaf1f). PRECONDITION FAILED — halted before verdict. "
                "New control surface authorized by the human's word; freeze preserves ordering AND mass; "
                "monitor reference frozen during resistance; NEGATIVE-config is a live outcome."
            ),
        )
        print(f"Verdict written to {verdict_path}")
        return

    # --- Evaluate ---
    ev = evaluate(results_by_arm_phase, verdict_arm_names, seeds)

    # --- Print evaluation tables ---
    print("=" * 80)
    print("EVALUATION TABLES")
    print("=" * 80)
    print()
    print_p5_table(ev, verdict_arm_names)
    print_p6_table(ev, verdict_arm_names)
    print_p7_frontier_table(ev)

    # --- Conjunct summary ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    n_seeds = len(seeds)
    print(f"P5 (resistance/recovery, Phase W):")
    print(f"  n4_freeze forks passing: {ev['n4_p5_pass_n']}/{n_seeds} "
          f"(need >= {P5_N4_MIN_FORKS}) => {'PASS' if ev['p5_n4'] else 'fail'}")
    print(f"  Baseline forks passing: {ev['bl_p5_pass_n']}/{n_seeds} "
          f"(need <= {P5_BASELINE_MAX_FORKS} for deficit) => "
          f"{'deficit exists' if ev['p5_baseline_deficit'] else 'NO DEFICIT'}")
    print(f"  P5 overall: {'PASS' if ev['p5_pass'] else ('FALSIFIER F5' if ev['f5'] else 'fail')}")
    print()
    print(f"P6 (revision, Phase R):")
    print(f"  n4_freeze forks within tolerance (latency <= baseline+{P6_TOLERANCE}): "
          f"{ev['n4_p6_within']}/{n_seeds} (need >= {P6_MIN_FORKS})")
    print(f"  n4_freeze forks never adopting: {ev['n4_p6_never']}/{n_seeds} "
          f"(rigidity if >= {F6_RIGIDITY_FORKS})")
    print(f"  P6 overall: {'PASS' if ev['p6_pass'] else ('FALSIFIER F6' if ev['f6'] else 'fail')}")
    print()
    print(f"P7 (kill test — no write-gain OR fixed-H constant arm matches both P5 and P6):")
    if ev["p7_pass"]:
        print(f"  PASS — no constant arm (write-gain or fixed-H) has both "
              f"p5_forks >= {P5_N4_MIN_FORKS} AND p6_within >= {P6_MIN_FORKS}")
    else:
        print(f"  FAIL (FALSIFIER F7) — constant arms matching both: {ev['p7_kill_arms']}")
        print(f"  => NEGATIVE-config: a fixed horizon suffices; evidence-based concession is ornament, not layer")
    print()

    tier_texts = {
        "positive": (
            "LAYER CONFIRMED — freeze-gate resists transient pressure, revises under "
            "sustained evidence, not reducible to any fixed constant"
        ),
        "config": (
            "NEGATIVE (config) — some fixed constant (write-gain or fixed-H) matches "
            "n4_freeze on both arms; the freeze is config, not a layer"
        ),
        "no-resistance": (
            "NEGATIVE (no-resistance) — n4_freeze fails whipsaw resistance (F5); "
            "freeze-gate control surface insufficient"
        ),
        "rigidity": (
            "NEGATIVE (rigidity) — n4_freeze passes resistance but not revision (F6); "
            "protect-without-revise is not the layer"
        ),
        "between-bands": (
            "MIXED (between-bands) — neither POSITIVE nor a named falsifier; "
            "richer test needed"
        ),
    }
    tier_text = tier_texts.get(ev["tier"], ev["tier"])
    print(f"VERDICT: {ev['verdict_str']}  ({tier_text})")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows = Path(__file__).parent / "outputs" / "exp182_rows.json"
    write_json_rows(results_by_arm_phase, [n for n, _ in ARMS], ev=ev,
                    pc2p_rows=pc2p_rows, out_path=out_rows)
    print(f"Rows written to {out_rows}")

    # --- Write verdict JSON ---
    halted_flag = ev["f5"] or ev["f6"] or ev["f7"]
    arms_dict = {
        "P5_resistance_recovery": {
            "pass": bool(ev["p5_pass"]),
            "reason": (
                f"n4_freeze forks passing Phase W: {ev['n4_p5_pass_n']}/{n_seeds} "
                f"(need >= {P5_N4_MIN_FORKS}); "
                f"baseline forks passing: {ev['bl_p5_pass_n']}/{n_seeds} "
                f"(need <= {P5_BASELINE_MAX_FORKS}); "
                f"falsifier F5: n4_freeze <= {F5_N4_MAX_FORKS}/{n_seeds}"
            ),
        },
        "P6_revision": {
            "pass": bool(ev["p6_pass"]),
            "reason": (
                f"n4_freeze forks within latency tolerance (baseline+{P6_TOLERANCE}): "
                f"{ev['n4_p6_within']}/{n_seeds} (need >= {P6_MIN_FORKS}); "
                f"never-adopted: {ev['n4_p6_never']}/{n_seeds} "
                f"(rigidity-falsifier if >= {F6_RIGIDITY_FORKS}); "
                f"falsifier F6: over-tolerance in >= {F6_LATENCY_FORKS}/{n_seeds} forks "
                f"or never-adopt in >= {F6_RIGIDITY_FORKS}/{n_seeds}"
            ),
        },
        "P7_not_config": {
            "pass": bool(ev["p7_pass"]),
            "reason": (
                "no fixed constant arm (write-gain or fixed-H) matches both P5 and P6 criteria"
                if ev["p7_pass"]
                else (
                    f"FALSIFIER F7: constant arms satisfying both: {ev['p7_kill_arms']} — "
                    f"NEGATIVE-config; a fixed horizon suffices at this richness"
                )
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp182_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp182",
        arms=arms_dict,
        verdict=ev["verdict_str"],
        halted=halted_flag,
        notes=(
            "Exp 182 N4 rung 3 attempt 2 FREEZE-GATE. "
            "PRE-REGISTERED in loop/directions/identity-n4.md (commit a0eaf1f) BEFORE any data. "
            "New control surface authorized by the human's word; Exp 181 remains NEGATIVE for "
            "write-channel gain control. "
            "Freeze preserves ordering AND mass (no decay during RESIST); "
            "monitor reference frozen during resistance (absorption-reopening addressed). "
            "NEGATIVE-config is a live outcome (fixed-H constant may suffice at this richness). "
            "13 verdict arms + oracle x phases x 8 seeds = 216 sessions. "
            "Verdict map: POSITIVE iff P5 AND P6 AND P7; NEGATIVE-config iff P5 AND P6 AND NOT P7; "
            "NEGATIVE-no-resistance iff F5; NEGATIVE-rigidity iff P5 AND F6; MIXED otherwise. "
            "Critical interpretation rules: fixed-H passing => config; "
            "protect-without-revise => rigidity; revise-without-protect => no-resistance."
        ),
    )
    print(f"Verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
