"""Exp 178 — N4 rung 2 attempt 2: the identity monitor under PC2' with scramble + flicker controls.

Two-arm design on fresh seeds 186-193 (Exp 170 same-schedule binding):
  Arm A (identity): verbatim Exp 177 regime — burst color = argmin(v) at onset,
    relocation rng 160000+seed, qs reset to uniform over burst-color cells.
  Arm B (scramble, generic-surprise control): identical burst windows, but each
    burst-step relocation targets a uniformly random cell of the WHOLE grid,
    relocation rng 175000+seed, qs reset uniform over ALL cells.

New gate: PC2' (vector-grade TV of pi over the pre-burst quiet window, replacing
strict argmax-constancy); PC3b (scramble color-frequency TV) voids P3 only.

PRE-REGISTRATION: the full hypothesis, predeclared properties, and falsifiers are
the card block "RUNG 2, ATTEMPT 2 — PRE-REGISTRATION (Exp 178)" in
loop/directions/identity-n4.md, committed (1a70e6c) BEFORE any data was generated.
Summary of the predeclared map implemented below:
  HYPOTHESIS: pi = v/sum(v) is quiet-stable at vector grade even where the argmax
  flickers; on that baseline the monitor is sensitive (P2), specific (P3), and
  argmax-independent (P4).
  P2 (sensitivity): AUROC_A >= 0.8 in >= 7/8 forks. FALSIFIER F2: median
  AUROC_A <= 0.5.
  P3 (specificity, valid only if PC3b): AUROC_A - AUROC_B >= 0.2 in >= 6/8 pairs
  AND median AUROC_B <= 0.65. FALSIFIER F3: median AUROC_B >= median AUROC_A - 0.05.
  P4 (argmax-independence, conditional on >= 2 flickering forks); D5 diagnostic.
  DECISION RULE: rung-2 (P2+P3 valid) / rung-1.5 (P2 only or PC3b-voided P3) /
  NEGATIVE (F2) / NO VERDICT (PC1/PC2'/PC3 failure).

Properties: P2 (sensitivity), P3 (specificity), P4 (argmax-independence),
D5 (adaptation diagnostic).

Pre-registered card: loop/directions/identity-n4.md §RUNG 2 ATTEMPT 2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from active_loop.creature import Creature
from active_loop.verdict import write_verdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = list(range(186, 194))   # 8 fresh forks

N_STEPS = 15_000
CHUNK_SIZE = 100
N_CHUNKS = N_STEPS // CHUNK_SIZE   # 150

# Value decay
LAMBDA = 0.9997
INIT_MASS = 1.0 / (1.0 - LAMBDA)  # ~ 3333.3

# Burst windows: [start, end) — inclusive start, exclusive end
BURST_WINDOWS = [
    (6000, 6800),
    (9000, 9800),
    (12000, 12800),
]
BURST_LEN = 800

# Burst RNG seed offsets
BURST_SEED_OFFSET_IDENTITY = 160_000   # Arm A: rng 160_000 + fork_seed
BURST_SEED_OFFSET_SCRAMBLE  = 175_000  # Arm B: rng 175_000 + fork_seed

# Monitor — declared (verbatim Exp 177)
EVAL = 100               # v snapshot cadence (steps)
MON_W_SNAPS = 10         # drift window = 10 snapshots = 1000 steps
EXCLUDE_AFTER = 1000     # transient band after burst end (steps)

# P2 (sensitivity, verbatim 177)
P2_AUROC_MIN = 0.8
P2_MIN_FORKS = 7
F2_MEDIAN_MAX = 0.5

# PC1
PC1_AHAT_DRIFT_MAX = 0.15

# PC2 (original argmax-constancy — kept for Arm A PC3 confinement logic compat.)
PC2_STABILITY_WINDOW = 1000
PC2_MIN_STABLE_FORKS = 7    # legacy (not used as gate in 178)

# PC3 confinement (Arm A)
PC3_CONFINEMENT_MIN = 0.90

# PC2' (vector-grade quiet continuity) — pre-registered
PC2P_TV_MAX = 0.05        # TV(pi(bstart-1000), pi(bstart)) bar
PC2P_MIN_FORKS = 7        # >= 7/8 Arm-A forks per window

# PC3b (scramble control validity)
PC3B_TV_MAX = 0.05        # burst color-frequency TV vs own pre-burst window
PC3B_MIN_FORKS = 6        # >= 6/8 forks per burst

# P3 (specificity)
P3_DELTA_MIN = 0.2        # AUROC_A - AUROC_B per fork pair
P3_MIN_PAIRS = 6          # >= 6/8 pairs
P3_B_MEDIAN_MAX = 0.65    # median AUROC_B bar
F3_MARGIN = 0.05          # F3: median AUROC_B >= median AUROC_A - 0.05

# P4 (conditional argmax-independence)
P4_MIN_FLICKER_FORKS = 2
P4_QUIET_BAND = (0.35, 0.65)

# D5 (adaptation diagnostic)
D5_RATIO = 2.0

# P1' / F1' thresholds (kept for burst-table diagnostics)
P1_FLIP_FINAL_FRAC = 0.50
P1_FLIP_FINAL_WINDOW = 400
P1_RECOVERY_LAG = 2000
P1_RECOVERY_HOLD = 500
P1_MIN_BURSTS_PER_FORK = 2
P1_MIN_FORKS = 7
F1_MAX_FORKS = 4


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
    """v_traj[k] is v at step (k+1)*EVAL."""
    return step // EVAL - 1


# ---------------------------------------------------------------------------
# Core step-loop — run one fork for the full session
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    arm: str,  # "identity" | "scramble"
    n_actions: int = 4,
) -> dict:
    """Run one fork (loaded fresh from mirro) for N_STEPS steps.

    arm='identity': burst color = argmin(v) at onset, relocation rng 160000+seed,
        qs reset to uniform over burst-color cells (verbatim Exp 177).
    arm='scramble': bursts at the SAME windows, relocation targets a uniformly
        random cell of the WHOLE grid, relocation rng 175000+seed, qs reset to
        uniform over ALL cells. burst_onset_color/burst_preburst_fav still
        recorded as covariates (argmin/argmax of v at onset).
    """
    arm_tag = "a" if arm == "identity" else "b"
    fork_name = f"exp178{arm_tag}_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    # Initialize decayed value store v from spine's value_counts
    vc = c.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v = (vc / vc_sum) * INIT_MASS
    else:
        v = np.ones(n_colors) * (INIT_MASS / n_colors)

    # Pre-build per-color cell lists (used by identity arm)
    color_cells: list[list[int]] = [[] for _ in range(n_colors)]
    for cell_idx, color in enumerate(base_cmap):
        color_cells[color].append(cell_idx)
    color_cells_arr = [np.array(lst, dtype=np.int32) for lst in color_cells]

    # Burst RNG
    if arm == "identity":
        burst_rng = np.random.default_rng(BURST_SEED_OFFSET_IDENTITY + fork_seed)
    else:
        burst_rng = np.random.default_rng(BURST_SEED_OFFSET_SCRAMBLE + fork_seed)

    # Build set of burst step indices for fast lookup
    burst_step_set = set()
    for bstart, bend in BURST_WINDOWS:
        burst_step_set.update(range(bstart, bend))

    # Per-step storage
    expressed_arr = np.empty(N_STEPS, dtype=np.int32)
    true_pos_arr = np.empty(N_STEPS, dtype=np.int32)
    obs_arr = np.empty(N_STEPS, dtype=np.int32)
    v_snapshots: list[dict] = []

    # Monitor: v trajectory snapshots every EVAL steps
    v_traj = []

    # Track active burst context
    current_burst_color: int | None = None
    current_burst_idx: int | None = None
    burst_preburst_fav: list[int | None] = [None, None, None]
    burst_onset_color: list[int | None] = [None, None, None]

    A_hat_start = c._A_hat().copy()

    eps = 1e-300
    global_step = 0

    for chunk_idx in range(N_CHUNKS):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(CHUNK_SIZE):
            t = global_step

            # --- Burst context management ---
            in_burst = t in burst_step_set
            burst_idx_now: int | None = None
            for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                if bstart <= t < bend:
                    burst_idx_now = bi
                    break

            # On burst onset: record pre-burst favorite and set burst color
            if burst_idx_now is not None and t == BURST_WINDOWS[burst_idx_now][0]:
                pre_fav = int(np.argmax(v))
                burst_preburst_fav[burst_idx_now] = pre_fav
                # burst color = argmin(v) at onset (diagnostic covariate for scramble arm too)
                bc = int(np.argmin(v))
                burst_onset_color[burst_idx_now] = bc
                if arm == "identity":
                    current_burst_color = bc
                else:
                    current_burst_color = bc  # stored as covariate; not used for relocation
                current_burst_idx = burst_idx_now
                v_snapshots.append({
                    "t": t, "label": f"onset_b{burst_idx_now}",
                    "v": v.copy(), "burst_color": bc, "pre_fav": pre_fav,
                })
            elif burst_idx_now is None:
                if current_burst_idx is not None:
                    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                        if t == bend:
                            v_snapshots.append({
                                "t": t, "label": f"end_b{bi}",
                                "v": v.copy(),
                                "burst_color": burst_onset_color[bi],
                                "pre_fav": burst_preburst_fav[bi],
                            })
                    current_burst_color = None
                    current_burst_idx = None

            for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                if t == bend + 1000:
                    v_snapshots.append({
                        "t": t, "label": f"end_b{bi}_plus1000",
                        "v": v.copy(),
                        "burst_color": burst_onset_color[bi],
                        "pre_fav": burst_preburst_fav[bi],
                    })
                if t == bend + 2000:
                    v_snapshots.append({
                        "t": t, "label": f"end_b{bi}_plus2000",
                        "v": v.copy(),
                        "burst_color": burst_onset_color[bi],
                        "pre_fav": burst_preburst_fav[bi],
                    })

            # --- A_hat pre-step ---
            A_hat = c._A_hat()

            # --- Observe ---
            obs = int(base_cmap[c.true_pos])
            obs_arr[t] = obs
            true_pos_arr[t] = c.true_pos

            # --- Surprise ---
            p_o = float(A_hat[obs, :] @ c.qs)

            # --- Belief update ---
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom_qs = qs_updated.sum()
            if denom_qs > 0:
                qs_updated = qs_updated / denom_qs
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # --- Dirichlet count update ---
            c.pA[obs, :] += qs_updated

            # --- Value accumulation (Exp 26 rule on decayed store) ---
            v *= LAMBDA
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            predictability_weight = math.exp(-h_predicted)
            v[obs] += predictability_weight
            c.value_counts[obs] += predictability_weight

            # --- Expressed preference ---
            expressed = int(np.argmax(v))
            expressed_arr[t] = expressed

            # --- Action / Move ---
            if in_burst:
                if arm == "identity":
                    # Relocate to uniformly random cell of burst color
                    cells_of_bc = color_cells_arr[current_burst_color]
                    if len(cells_of_bc) > 0:
                        c.true_pos = int(burst_rng.choice(cells_of_bc))
                    # qs reset to uniform over burst-color cells
                    qs_next = np.zeros(n_cells)
                    qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                    c.qs = qs_next
                else:
                    # Scramble: relocate to uniformly random cell of whole grid
                    c.true_pos = int(burst_rng.integers(0, n_cells))
                    # qs reset to uniform over ALL cells
                    c.qs = np.ones(n_cells) / n_cells
            else:
                # Normal: random action + move + advance qs through B
                action = int(rng.integers(0, n_actions))
                c.true_pos = c.world.move(c.true_pos, action)
                c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1
            global_step += 1

            # Monitor: snapshot v after step update
            if global_step % EVAL == 0:
                v_traj.append(v.copy())

    # A_hat drift
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    return {
        "fork_seed": fork_seed,
        "arm": arm,
        "n_total": N_STEPS,
        "ahat_drift": ahat_drift,
        "expressed_arr": expressed_arr,
        "true_pos_arr": true_pos_arr,
        "obs_arr": obs_arr,
        "v_snapshots": v_snapshots,
        "burst_preburst_fav": burst_preburst_fav,
        "burst_onset_color": burst_onset_color,
        "v_traj": np.array(v_traj),
    }


# ---------------------------------------------------------------------------
# Per-burst analysis: flip, recovery
# ---------------------------------------------------------------------------

def analyze_burst(
    expressed_arr: np.ndarray,
    burst_idx: int,
    burst_preburst_fav: list,
    burst_onset_color: list,
) -> dict:
    bstart, bend = BURST_WINDOWS[burst_idx]
    bc = burst_onset_color[burst_idx]
    pre_fav = burst_preburst_fav[burst_idx]

    if bc is None or pre_fav is None:
        return {
            "flip": False, "flip_latency": None,
            "recovery": False, "recovery_latency": None,
            "flip_frac": 0.0,
        }

    final_window_start = bend - P1_FLIP_FINAL_WINDOW
    final_window = expressed_arr[final_window_start:bend]
    flip_frac = float(np.mean(final_window == bc))
    flip = flip_frac >= P1_FLIP_FINAL_FRAC

    flip_latency: int | None = None
    for t in range(bstart, bend):
        if expressed_arr[t] == bc:
            flip_latency = t - bstart
            break

    recovery = False
    recovery_latency: int | None = None
    recovery_end = min(bend + P1_RECOVERY_LAG, N_STEPS)

    t = bend
    while t < recovery_end:
        if expressed_arr[t] == pre_fav:
            hold_end = min(t + P1_RECOVERY_HOLD, N_STEPS)
            if hold_end > t and np.all(expressed_arr[t:hold_end] == pre_fav):
                recovery = True
                recovery_latency = t - bend
                break
        t += 1

    return {
        "flip": flip,
        "flip_frac": flip_frac,
        "flip_latency": flip_latency,
        "recovery": recovery,
        "recovery_latency": recovery_latency,
    }


# ---------------------------------------------------------------------------
# Monitor functions (verbatim Exp 177)
# ---------------------------------------------------------------------------

def label_sample(target_step: int) -> str:
    """BURST / EXCLUDED / QUIET label."""
    for bstart, bend in BURST_WINDOWS:
        if bstart < target_step <= bend:
            return "BURST"
    for bstart, bend in BURST_WINDOWS:
        if bend < target_step <= bend + EXCLUDE_AFTER:
            return "EXCLUDED"
    return "QUIET"


def compute_monitor(v_traj: np.ndarray) -> list[dict]:
    """Mismatch samples: linear-drift self-prediction, L2 mismatch."""
    samples = []
    n = len(v_traj)
    for k in range(MON_W_SNAPS, n - 1):
        v_hat = v_traj[k] + (v_traj[k] - v_traj[k - MON_W_SNAPS]) / MON_W_SNAPS
        mismatch = float(np.linalg.norm(v_hat - v_traj[k + 1]))
        target_step = (k + 2) * EVAL
        samples.append({
            "target_step": target_step,
            "mismatch": mismatch,
            "label": label_sample(target_step),
        })
    return samples


def auroc(pos: list[float], neg: list[float]) -> float:
    """Rank-based AUROC (Mann-Whitney with midranks)."""
    if not pos or not neg:
        return float("nan")
    combined = np.array(pos + neg)
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(len(combined))
    sorted_vals = combined[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0
        for idx in range(i, j + 1):
            ranks[order[idx]] = midrank
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------

def check_pc1(run_results_a: list[dict], run_results_b: list[dict]) -> tuple[bool, list[str]]:
    """PC1: ahat_drift < 0.15 in both arms."""
    failures: list[str] = []
    for rr in run_results_a + run_results_b:
        seed = rr["fork_seed"]
        arm_tag = rr["arm"]
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: arm={arm_tag} seed={seed} ahat_drift={rr['ahat_drift']:.4f} "
                f">= {PC1_AHAT_DRIFT_MAX}"
            )
    return len(failures) == 0, failures


def check_pc2_prime(run_results_a: list[dict]) -> tuple[bool, list[str], list[dict]]:
    """PC2' (vector-grade quiet continuity).

    Per burst window, per Arm-A fork:
      d = TV(pi(bstart-1000), pi(bstart))
    Window passes iff count of forks with d <= PC2P_TV_MAX is >= PC2P_MIN_FORKS.
    PC2' passes iff all three windows pass.

    Returns (passed, failure_strings, per_fork_window_rows).
    Each row: fork_seed, window_idx, tv, flicker_count, top2_margin (covariates).
    """
    failures: list[str] = []
    rows: list[dict] = []

    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
        si_end = snap_index(bstart)       # snapshot index at bstart
        si_start = snap_index(bstart - 1000) if bstart >= 1000 else 0
        forks_ok = 0
        for rr in run_results_a:
            seed = rr["fork_seed"]
            vt = rr["v_traj"]
            if si_end < 0 or si_end >= len(vt) or si_start < 0 or si_start >= len(vt):
                # Cannot compute — treat as fail
                d = float("nan")
                flicker_count = None
                top2_margin = None
            else:
                pi_pre = pi_of(vt[si_start])
                pi_onset = pi_of(vt[si_end])
                d = tv(pi_pre, pi_onset)

                # Flicker count: adjacent argmax changes in expressed_arr[bstart-1000:bstart]
                exp_win = rr["expressed_arr"][max(0, bstart - 1000):bstart]
                flicker_count = int(np.sum(np.diff(exp_win) != 0))

                # Top-2 margin of pi at bstart
                sorted_pi = np.sort(pi_onset)[::-1]
                top2_margin = float(sorted_pi[0] - sorted_pi[1]) if len(sorted_pi) >= 2 else float("nan")

            ok = (not math.isnan(d)) and d <= PC2P_TV_MAX
            if ok:
                forks_ok += 1

            rows.append({
                "fork_seed": seed,
                "window_idx": bi,
                "bstart": bstart,
                "tv": d if not math.isnan(d) else None,
                "flicker_count": flicker_count,
                "top2_margin": top2_margin,
                "passes": ok,
            })

        if forks_ok < PC2P_MIN_FORKS:
            failures.append(
                f"PC2' FAIL: window {bi} (bstart={bstart}) — "
                f"only {forks_ok}/{len(run_results_a)} forks with "
                f"TV(pi(bstart-1000), pi(bstart)) <= {PC2P_TV_MAX} "
                f"(need >= {PC2P_MIN_FORKS})"
            )

    passed = len(failures) == 0
    return passed, failures, rows


def check_pc3(run_results_a: list[dict], base_cmap: list) -> tuple[bool, list[str]]:
    """PC3: confinement >= 90% on burst-color cells (Arm A only)."""
    failures: list[str] = []
    for bi in range(3):
        bstart, bend = BURST_WINDOWS[bi]
        confinement_rates: list[float] = []
        for rr in run_results_a:
            bc = rr["burst_onset_color"][bi]
            if bc is None:
                continue
            true_pos_burst = rr["true_pos_arr"][bstart:bend]
            on_bc = sum(1 for p in true_pos_burst if base_cmap[p] == bc)
            rate = on_bc / len(true_pos_burst) if len(true_pos_burst) > 0 else 0.0
            confinement_rates.append(rate)
        if confinement_rates:
            mean_conf = float(np.mean(confinement_rates))
            if mean_conf < PC3_CONFINEMENT_MIN:
                failures.append(
                    f"PC3 FAIL: burst {bi} — mean confinement rate={mean_conf:.3f} "
                    f"< {PC3_CONFINEMENT_MIN}"
                )
    return len(failures) == 0, failures


def check_pc3b(run_results_b: list[dict]) -> tuple[bool, list[str], list[dict]]:
    """PC3b (scramble control validity).

    Per scramble burst, per Arm-B fork:
      freq_burst = normalized color histogram of obs_arr[bstart:bend]
      freq_pre   = normalized color histogram of obs_arr[bstart-1000:bstart]
      d = TV(freq_burst, freq_pre)
    Burst passes iff count of forks with d <= PC3B_TV_MAX >= PC3B_MIN_FORKS.
    PC3b passes iff all three bursts pass.

    PC3b failure does NOT block the verdict — it voids P3 only.

    Returns (valid, failure_strings, rows).
    """
    failures: list[str] = []
    rows: list[dict] = []
    n_colors_inferred = None

    for rr in run_results_b:
        obs = rr["obs_arr"]
        if n_colors_inferred is None:
            n_colors_inferred = int(obs.max()) + 1

    if n_colors_inferred is None:
        return True, [], []

    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
        forks_ok = 0
        for rr in run_results_b:
            seed = rr["fork_seed"]
            obs = rr["obs_arr"]
            n_c = n_colors_inferred

            obs_burst = obs[bstart:bend]
            obs_pre = obs[max(0, bstart - 1000):bstart]

            freq_burst = np.zeros(n_c)
            for o in obs_burst:
                freq_burst[o] += 1
            s_b = freq_burst.sum()
            freq_burst = freq_burst / s_b if s_b > 0 else np.ones(n_c) / n_c

            freq_pre = np.zeros(n_c)
            for o in obs_pre:
                freq_pre[o] += 1
            s_p = freq_pre.sum()
            freq_pre = freq_pre / s_p if s_p > 0 else np.ones(n_c) / n_c

            d = tv(freq_burst, freq_pre)
            ok = d <= PC3B_TV_MAX
            if ok:
                forks_ok += 1

            rows.append({
                "fork_seed": seed,
                "burst_idx": bi,
                "bstart": bstart,
                "tv_color_freq": d,
                "passes": ok,
            })

        if forks_ok < PC3B_MIN_FORKS:
            failures.append(
                f"PC3b FAIL: burst {bi} (bstart={bstart}) — "
                f"only {forks_ok}/{len(run_results_b)} forks with "
                f"color-freq TV(burst, pre-burst) <= {PC3B_TV_MAX} "
                f"(need >= {PC3B_MIN_FORKS}); P3 voided"
            )

    valid = len(failures) == 0
    return valid, failures, rows


# ---------------------------------------------------------------------------
# Evaluate P2 / P3 / P4 / D5
# ---------------------------------------------------------------------------

def evaluate(
    run_results_a: list[dict],
    run_results_b: list[dict],
    pc2p_rows: list[dict],
    pc3b_valid: bool,
) -> dict:
    """Compute per-fork AUROCs and derive verdict.

    P2 (sensitivity): Arm-A AUROC >= 0.8 in >= 7/8 forks.
    F2: median Arm-A AUROC <= 0.5.
    P3 (specificity, only if pc3b_valid): AUROC_A - AUROC_B >= 0.2 in >= 6/8 pairs
        AND median AUROC_B <= 0.65.
    F3 flag: median AUROC_B >= median AUROC_A - F3_MARGIN.
    P4 (argmax-independence, conditional): evaluated iff >= 2 Arm-A forks flicker pre-burst.
    D5 (adaptation diagnostic): onset mismatch (first 2 burst samples) >= 2x late-burst (last 2).

    Decision rule (pre-registered):
      if F2:                                   => NEGATIVE, tier "negative"
      elif P2 and pc3b_valid and P3:
        if p4_status == "fail" and P4i_failed: => MIXED, tier "mixed-argmax-dependent"
        else:                                  => POSITIVE, tier "rung-2"
      elif P2 and (pc3b_valid and not P3) or (not pc3b_valid):
                                               => MIXED, tier "rung-1.5"
      else:                                   => MIXED, tier "between-bands"
    """
    # --- Per Arm-A fork AUROC ---
    per_fork_a = []
    aurocs_a = []
    for rr in run_results_a:
        samples = compute_monitor(rr["v_traj"])
        burst_m = [s["mismatch"] for s in samples if s["label"] == "BURST"]
        quiet_m = [s["mismatch"] for s in samples if s["label"] == "QUIET"]
        a = auroc(burst_m, quiet_m)
        aurocs_a.append(a)
        per_fork_a.append({
            "fork_seed": rr["fork_seed"],
            "arm": "A",
            "auroc": a,
            "n_burst": len(burst_m),
            "n_quiet": len(quiet_m),
            "mean_mismatch_burst": float(np.mean(burst_m)) if burst_m else None,
            "mean_mismatch_quiet": float(np.mean(quiet_m)) if quiet_m else None,
            "fork_passes_p2": (not math.isnan(a)) and a >= P2_AUROC_MIN,
            # D5 burst samples for later
            "_samples": samples,
        })

    # --- Per Arm-B fork AUROC ---
    per_fork_b = []
    aurocs_b = []
    for rr in run_results_b:
        samples = compute_monitor(rr["v_traj"])
        burst_m = [s["mismatch"] for s in samples if s["label"] == "BURST"]
        quiet_m = [s["mismatch"] for s in samples if s["label"] == "QUIET"]
        a = auroc(burst_m, quiet_m)
        aurocs_b.append(a)
        per_fork_b.append({
            "fork_seed": rr["fork_seed"],
            "arm": "B",
            "auroc": a,
            "n_burst": len(burst_m),
            "n_quiet": len(quiet_m),
            "mean_mismatch_burst": float(np.mean(burst_m)) if burst_m else None,
            "mean_mismatch_quiet": float(np.mean(quiet_m)) if quiet_m else None,
            "_samples": samples,
        })

    # --- P2 / F2 ---
    forks_pass_p2 = sum(1 for f in per_fork_a if f["fork_passes_p2"])
    valid_aurocs_a = [a for a in aurocs_a if not math.isnan(a)]
    median_auroc_a = float(np.median(valid_aurocs_a)) if valid_aurocs_a else float("nan")
    p2_pass = forks_pass_p2 >= P2_MIN_FORKS
    f2 = (not math.isnan(median_auroc_a)) and median_auroc_a <= F2_MEDIAN_MAX

    # --- P3 (only if pc3b_valid) ---
    valid_aurocs_b = [a for a in aurocs_b if not math.isnan(a)]
    median_auroc_b = float(np.median(valid_aurocs_b)) if valid_aurocs_b else float("nan")

    # Build seed-aligned pairs (same order — seeds match by construction)
    seed_to_auroc_a = {f["fork_seed"]: f["auroc"] for f in per_fork_a}
    seed_to_auroc_b = {f["fork_seed"]: f["auroc"] for f in per_fork_b}
    p3_pairs = []
    for seed in [rr["fork_seed"] for rr in run_results_a]:
        aa = seed_to_auroc_a.get(seed, float("nan"))
        ab = seed_to_auroc_b.get(seed, float("nan"))
        if not math.isnan(aa) and not math.isnan(ab):
            p3_pairs.append((seed, aa, ab, aa - ab))

    p3_n_pass = sum(1 for _, aa, ab, delta in p3_pairs if delta >= P3_DELTA_MIN)
    p3_b_ok = (not math.isnan(median_auroc_b)) and median_auroc_b <= P3_B_MEDIAN_MAX
    p3_pass = pc3b_valid and (p3_n_pass >= P3_MIN_PAIRS) and p3_b_ok

    f3_flag = (
        not math.isnan(median_auroc_b)
        and not math.isnan(median_auroc_a)
        and median_auroc_b >= median_auroc_a - F3_MARGIN
    ) if pc3b_valid else False

    # --- P4 (conditional argmax-independence) ---
    # Flickering forks: Arm-A forks with >= 1 argmax switch in ANY pre-burst window
    flicker_by_seed: dict[int, bool] = {}
    for row in pc2p_rows:
        seed = row["fork_seed"]
        if row.get("flicker_count") is not None and row["flicker_count"] >= 1:
            flicker_by_seed[seed] = True
        elif seed not in flicker_by_seed:
            flicker_by_seed[seed] = False

    flickering_seeds = [s for s, v in flicker_by_seed.items() if v]
    p4_evaluable = len(flickering_seeds) >= P4_MIN_FLICKER_FORKS

    p4i_pass = True
    p4i_failed = False
    p4ii_pass = True
    p4_status = "not-evaluable"

    if p4_evaluable:
        # P4i: every flickering fork has AUROC_A >= 0.8
        for seed in flickering_seeds:
            a_val = seed_to_auroc_a.get(seed, float("nan"))
            if math.isnan(a_val) or a_val < P2_AUROC_MIN:
                p4i_pass = False
                p4i_failed = True
                break

        # P4ii: AUROC(pooled quiet mismatch in flickered pre-burst windows vs
        #        pooled quiet in non-flickered pre-burst windows, Arm A, pre-burst only)
        #
        # A "pre-burst window" for burst bi covers steps [bstart-1000, bstart).
        # For each burst bi, a fork either flickered in that window or not.
        # We gather quiet-labeled mismatch samples whose target_step falls in a
        # flickered vs non-flickered pre-burst window (across Arm-A, pre-burst only).
        flickered_quiet = []
        stable_quiet = []

        # Determine per-fork per-window flicker state
        flicker_state: dict[tuple[int, int], bool] = {}  # (fork_seed, window_idx) -> bool
        for row in pc2p_rows:
            key = (row["fork_seed"], row["window_idx"])
            flicker_state[key] = (row.get("flicker_count") is not None
                                  and row["flicker_count"] >= 1)

        for pf in per_fork_a:
            seed = pf["fork_seed"]
            samples = pf["_samples"]
            for s in samples:
                ts = s["target_step"]
                # Check if this quiet sample falls in a pre-burst window
                for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                    pre_start = bstart - 1000
                    pre_end = bstart
                    if pre_start <= ts <= pre_end and s["label"] == "QUIET":
                        is_flickered = flicker_state.get((seed, bi), False)
                        if is_flickered:
                            flickered_quiet.append(s["mismatch"])
                        else:
                            stable_quiet.append(s["mismatch"])

        if flickered_quiet and stable_quiet:
            p4ii_val = auroc(flickered_quiet, stable_quiet)
            p4ii_pass = (
                not math.isnan(p4ii_val)
                and P4_QUIET_BAND[0] <= p4ii_val <= P4_QUIET_BAND[1]
            )
        else:
            p4ii_val = float("nan")
            p4ii_pass = True  # not enough data to distinguish => no-penalty

        if p4i_pass and p4ii_pass:
            p4_status = "pass"
        else:
            p4_status = "fail"
    else:
        p4ii_val = float("nan")

    # --- D5 (adaptation diagnostic) ---
    d5_count = 0
    d5_total = 0
    for pf in per_fork_a:
        samples = pf["_samples"]
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            burst_samples = [s for s in samples
                             if s["label"] == "BURST"
                             and bstart < s["target_step"] <= bend]
            if len(burst_samples) >= 4:
                first2 = [s["mismatch"] for s in burst_samples[:2]]
                last2 = [s["mismatch"] for s in burst_samples[-2:]]
                ratio = np.mean(first2) / (np.mean(last2) + 1e-300)
                if ratio >= D5_RATIO:
                    d5_count += 1
                d5_total += 1

    d5_fraction = float(d5_count / d5_total) if d5_total > 0 else float("nan")

    # --- Decision rule ---
    if f2:
        verdict_str = "NEGATIVE"
        tier = "negative"
    elif p2_pass and pc3b_valid and p3_pass:
        if p4_status == "fail" and p4i_failed:
            verdict_str = "MIXED"
            tier = "mixed-argmax-dependent"
        else:
            verdict_str = "POSITIVE"
            tier = "rung-2"
    elif p2_pass and ((pc3b_valid and not p3_pass) or (not pc3b_valid)):
        verdict_str = "MIXED"
        tier = "rung-1.5"
    else:
        verdict_str = "MIXED"
        tier = "between-bands"

    return {
        "per_fork_a": per_fork_a,
        "per_fork_b": per_fork_b,
        "p3_pairs": p3_pairs,
        "forks_pass_p2": forks_pass_p2,
        "median_auroc_a": median_auroc_a,
        "median_auroc_b": median_auroc_b,
        "p2_pass": p2_pass,
        "f2": f2,
        "p3_pass": p3_pass,
        "p3_n_pass": p3_n_pass,
        "p3_b_ok": p3_b_ok,
        "f3_flag": f3_flag,
        "p4_status": p4_status,
        "p4_evaluable": p4_evaluable,
        "p4i_pass": p4i_pass,
        "p4i_failed": p4i_failed,
        "p4ii_pass": p4ii_pass,
        "flickering_seeds": flickering_seeds,
        "d5_fraction": d5_fraction,
        "verdict_str": verdict_str,
        "tier": tier,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_burst_table(run_results: list[dict], arm_label: str = "") -> None:
    label = f" (Arm {arm_label})" if arm_label else ""
    header = (f"{'seed':>5} {'b':>2} {'pre_fav':>8} {'bc':>4} "
              f"{'flip?':>6} {'flip_frac':>10} {'flip_lat':>9} "
              f"{'recovery?':>10} {'rec_lat':>8}")
    print(f"Burst analysis table{label}:")
    print(header)
    print("-" * len(header))
    for rr in run_results:
        seed = rr["fork_seed"]
        for bi in range(3):
            ba = analyze_burst(
                rr["expressed_arr"], bi,
                rr["burst_preburst_fav"], rr["burst_onset_color"],
            )
            pre_fav = rr["burst_preburst_fav"][bi]
            bc = rr["burst_onset_color"][bi]
            flip_lat_s = str(ba["flip_latency"]) if ba["flip_latency"] is not None else "—"
            rec_lat_s = str(ba["recovery_latency"]) if ba["recovery_latency"] is not None else "—"
            print(
                f"{seed:>5} {bi:>2} {str(pre_fav):>8} {str(bc):>4} "
                f"{'YES' if ba['flip'] else 'no':>6} {ba['flip_frac']:>10.3f} "
                f"{flip_lat_s:>9} "
                f"{'YES' if ba['recovery'] else 'no':>10} {rec_lat_s:>8}"
            )
    print()


def print_pc2p_table(pc2p_rows: list[dict]) -> None:
    print("PC2' table (per fork per window: tv, flicker, margin):")
    header = f"{'seed':>5} {'win':>4} {'bstart':>7} {'tv':>8} {'flicker':>8} {'margin':>8} {'ok?':>5}"
    print(header)
    print("-" * len(header))
    for row in pc2p_rows:
        tv_s = f"{row['tv']:.5f}" if row["tv"] is not None else "nan"
        flicker_s = str(row["flicker_count"]) if row["flicker_count"] is not None else "—"
        margin_s = f"{row['top2_margin']:.4f}" if row["top2_margin"] is not None else "—"
        print(f"{row['fork_seed']:>5} {row['window_idx']:>4} {row['bstart']:>7} "
              f"{tv_s:>8} {flicker_s:>8} {margin_s:>8} {'ok' if row['passes'] else 'FAIL':>5}")
    print()


def print_pc3b_table(pc3b_rows: list[dict]) -> None:
    print("PC3b table (scramble color-frequency TV):")
    header = f"{'seed':>5} {'burst':>6} {'bstart':>7} {'tv':>8} {'ok?':>5}"
    print(header)
    print("-" * len(header))
    for row in pc3b_rows:
        print(f"{row['fork_seed']:>5} {row['burst_idx']:>6} {row['bstart']:>7} "
              f"{row['tv_color_freq']:>8.5f} {'ok' if row['passes'] else 'FAIL':>5}")
    print()


def print_auroc_table(ev: dict) -> None:
    print("AUROC table (per seed: AUROC_A, AUROC_B, delta):")
    header = f"{'seed':>5} {'AUROC_A':>9} {'AUROC_B':>9} {'delta':>8} {'P2_ok?':>7} {'P3_ok?':>7}"
    print(header)
    print("-" * len(header))
    seed_to_b = {f["fork_seed"]: f["auroc"] for f in ev["per_fork_b"]}
    for pf in ev["per_fork_a"]:
        seed = pf["fork_seed"]
        aa = pf["auroc"]
        ab = seed_to_b.get(seed, float("nan"))
        delta = (aa - ab) if not (math.isnan(aa) or math.isnan(ab)) else float("nan")
        p2_ok = pf["fork_passes_p2"]
        p3_ok = (not math.isnan(delta)) and delta >= P3_DELTA_MIN
        aa_s = f"{aa:.4f}" if not math.isnan(aa) else "nan"
        ab_s = f"{ab:.4f}" if not math.isnan(ab) else "nan"
        delta_s = f"{delta:.4f}" if not math.isnan(delta) else "nan"
        print(f"{seed:>5} {aa_s:>9} {ab_s:>9} {delta_s:>8} "
              f"{'PASS' if p2_ok else 'fail':>7} {'PASS' if p3_ok else 'fail':>7}")
    print()


# ---------------------------------------------------------------------------
# JSON row writer
# ---------------------------------------------------------------------------

def write_json_rows(
    run_results_a: list[dict],
    run_results_b: list[dict],
    ev: dict | None,
    pc2p_rows: list[dict],
    pc3b_rows: list[dict],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        # Burst rows for both arms
        for arm_tag, run_results in [("A", run_results_a), ("B", run_results_b)]:
            for rr in run_results:
                seed = rr["fork_seed"]
                for bi in range(3):
                    ba = analyze_burst(
                        rr["expressed_arr"], bi,
                        rr["burst_preburst_fav"], rr["burst_onset_color"],
                    )
                    bstart, bend = BURST_WINDOWS[bi]
                    row = {
                        "exp": 178,
                        "arm": arm_tag,
                        "fork_seed": seed,
                        "burst_idx": bi,
                        "burst_start": bstart,
                        "burst_end": bend,
                        "pre_fav": rr["burst_preburst_fav"][bi],
                        "burst_color": rr["burst_onset_color"][bi],
                        "flip": ba["flip"],
                        "flip_frac": ba["flip_frac"],
                        "flip_latency": ba["flip_latency"],
                        "recovery": ba["recovery"],
                        "recovery_latency": ba["recovery_latency"],
                        "ahat_drift": rr["ahat_drift"],
                    }
                    fh.write(json.dumps(row) + "\n")

        # AUROC rows
        if ev is not None:
            for pf in ev["per_fork_a"]:
                row = {
                    "exp": 178,
                    "arm": "A",
                    "fork_seed": pf["fork_seed"],
                    "auroc": pf["auroc"],
                    "n_burst": pf["n_burst"],
                    "n_quiet": pf["n_quiet"],
                    "mean_mismatch_burst": pf["mean_mismatch_burst"],
                    "mean_mismatch_quiet": pf["mean_mismatch_quiet"],
                    "fork_passes_p2": pf["fork_passes_p2"],
                }
                fh.write(json.dumps(row) + "\n")
            for pf in ev["per_fork_b"]:
                row = {
                    "exp": 178,
                    "arm": "B",
                    "fork_seed": pf["fork_seed"],
                    "auroc": pf["auroc"],
                    "n_burst": pf["n_burst"],
                    "n_quiet": pf["n_quiet"],
                    "mean_mismatch_burst": pf["mean_mismatch_burst"],
                    "mean_mismatch_quiet": pf["mean_mismatch_quiet"],
                }
                fh.write(json.dumps(row) + "\n")

        # PC2' rows
        for row in pc2p_rows:
            fh.write(json.dumps({"exp": 178, "phase": "pc2p", **row}) + "\n")

        # PC3b rows
        for row in pc3b_rows:
            fh.write(json.dumps({"exp": 178, "phase": "pc3b", **row}) + "\n")

        # Summary row
        if ev is not None:
            summary = {
                "exp": 178,
                "phase": "summary",
                "verdict": ev["verdict_str"],
                "tier": ev["tier"],
                "median_auroc_a": ev["median_auroc_a"],
                "median_auroc_b": ev["median_auroc_b"],
                "forks_pass_p2": ev["forks_pass_p2"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "p3_pass": ev["p3_pass"],
                "p3_n_pass": ev["p3_n_pass"],
                "f3_flag": ev["f3_flag"],
                "p4_status": ev["p4_status"],
                "d5_fraction": ev["d5_fraction"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 178 — N4 rung 2 attempt 2: the identity monitor under PC2' with scramble + flicker controls"
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: seed [186] only (both arms), no verdict")
    args = parser.parse_args()

    seeds = [186] if args.smoke else SEEDS

    print("=" * 80)
    print("Exp 178 — N4 rung 2 attempt 2: the identity monitor under PC2' with scramble + flicker controls")
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
    print(f"Bursts: {BURST_WINDOWS}")
    print(f"Seeds: {seeds}  Arms: identity + scramble")
    print()

    per_color_cells = [sum(1 for c in base_cmap if c == col) for col in range(n_colors)]
    print(f"Cells per color: {per_color_cells}")
    print()

    vc = mirro.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        spine_fav = int(np.argmax(vc))
        print(f"Mirro spine value_counts: {vc}")
        print(f"Spine standing favorite: color {spine_fav}")
    print()

    # --- Run Arm-A forks ---
    run_results_a: list[dict] = []
    print("Running Arm A (identity) ...")
    for seed in seeds:
        print(f"  Arm A seed={seed} ...", flush=True)
        rr = run_fork(mirro, seed, base_cmap, n_colors, arm="identity")
        run_results_a.append(rr)
        print(f"    drift={rr['ahat_drift']:.5f}  "
              f"standing_fav={rr['burst_preburst_fav'][0]}  "
              f"burst_colors={rr['burst_onset_color']}")
    print()

    # --- Run Arm-B forks ---
    run_results_b: list[dict] = []
    print("Running Arm B (scramble) ...")
    for seed in seeds:
        print(f"  Arm B seed={seed} ...", flush=True)
        rr = run_fork(mirro, seed, base_cmap, n_colors, arm="scramble")
        run_results_b.append(rr)
        print(f"    drift={rr['ahat_drift']:.5f}  "
              f"burst_onset_color(covariate)={rr['burst_onset_color']}")
    print()

    # --- Burst diagnostic tables (both arms) ---
    print_burst_table(run_results_a, arm_label="A — identity")
    print_burst_table(run_results_b, arm_label="B — scramble")

    # --- Smoke: print partial monitor data and return early ---
    if args.smoke:
        rr0 = run_results_a[0]
        samples0 = compute_monitor(rr0["v_traj"])
        burst0_samples = [s for s in samples0 if s["label"] == "BURST"
                          and 6000 < s["target_step"] <= 6800]
        quiet0_samples = [s for s in samples0 if s["label"] == "QUIET"]
        print(f"Monitor smoke Arm A (seed={rr0['fork_seed']}): "
              f"total={len(samples0)}, "
              f"BURST={len([s for s in samples0 if s['label']=='BURST'])}, "
              f"EXCLUDED={len([s for s in samples0 if s['label']=='EXCLUDED'])}, "
              f"QUIET={len(quiet0_samples)}")
        if burst0_samples:
            print("  Arm-A burst-0 first 4 BURST samples:")
            for s in burst0_samples[:4]:
                print(f"    target_step={s['target_step']} mismatch={s['mismatch']:.5f}")
            print("  Arm-A burst-0 last 4 BURST samples:")
            for s in burst0_samples[-4:]:
                print(f"    target_step={s['target_step']} mismatch={s['mismatch']:.5f}")
        a_smoke_a = auroc(
            [s["mismatch"] for s in samples0 if s["label"] == "BURST"],
            [s["mismatch"] for s in samples0 if s["label"] == "QUIET"],
        )
        print(f"  Arm A smoke AUROC: {a_smoke_a:.4f}")

        rr0b = run_results_b[0]
        samples0b = compute_monitor(rr0b["v_traj"])
        a_smoke_b = auroc(
            [s["mismatch"] for s in samples0b if s["label"] == "BURST"],
            [s["mismatch"] for s in samples0b if s["label"] == "QUIET"],
        )
        print(f"  Arm B smoke AUROC: {a_smoke_b:.4f}")

        # PC2' smoke (Arm A, window 0)
        bi, (bstart, bend) = 0, BURST_WINDOWS[0]
        si_end = snap_index(bstart)
        si_start = snap_index(bstart - 1000)
        pi_pre = pi_of(rr0["v_traj"][si_start])
        pi_onset = pi_of(rr0["v_traj"][si_end])
        tv_smoke = tv(pi_pre, pi_onset)
        print(f"  PC2' smoke Arm A seed={rr0['fork_seed']} window 0: TV={tv_smoke:.5f} (bar={PC2P_TV_MAX})")
        print()
        print("SMOKE ONLY — no verdict")
        return

    # --- Preconditions ---
    pc1_pass, pc1_failures = check_pc1(run_results_a, run_results_b)
    pc2p_pass, pc2p_failures, pc2p_rows = check_pc2_prime(run_results_a)
    pc3_pass, pc3_failures = check_pc3(run_results_a, base_cmap)
    pc3b_valid, pc3b_failures, pc3b_rows = check_pc3b(run_results_b)

    all_gate_failures = pc1_failures + pc2p_failures + pc3_failures
    gate_pass = pc1_pass and pc2p_pass and pc3_pass

    print("=== PRECONDITIONS ===")
    if all_gate_failures:
        print("PRECONDITION FAILED:")
        for f in all_gate_failures:
            print(f"  {f}")
    else:
        print("PC1 + PC2' + PC3: all PASS")
    if pc3b_failures:
        print("PC3b (non-gating — voids P3 only):")
        for f in pc3b_failures:
            print(f"  {f}")
    else:
        print("PC3b: PASS (P3 eligible)")
    print()

    # PC2' table (covariate logging regardless of gate outcome)
    print_pc2p_table(pc2p_rows)
    print_pc3b_table(pc3b_rows)

    if not gate_pass:
        print("PRECONDITION FAILED — writing rows, halting before verdict.")
        out_rows = Path(__file__).parent / "outputs" / "exp178_rows.json"
        write_json_rows(run_results_a, run_results_b, ev=None,
                        pc2p_rows=pc2p_rows, pc3b_rows=pc3b_rows, out_path=out_rows)
        print(f"Rows written to {out_rows}")
        return

    # --- Evaluate ---
    ev = evaluate(run_results_a, run_results_b, pc2p_rows, pc3b_valid)

    # --- AUROC table ---
    print("=" * 80)
    print("MONITOR TABLE — per-fork AUROC")
    print("=" * 80)
    print()
    mon_header = (f"{'seed':>5} {'AUROC_A':>9} {'n_bA':>5} {'n_qA':>5} "
                  f"{'mean_m_bA':>11} {'pass_P2?':>9}")
    print(mon_header)
    print("-" * len(mon_header))
    for pf in ev["per_fork_a"]:
        mmb = f"{pf['mean_mismatch_burst']:.5f}" if pf["mean_mismatch_burst"] is not None else "—"
        aa_s = f"{pf['auroc']:.4f}" if not math.isnan(pf["auroc"]) else "nan"
        print(f"{pf['fork_seed']:>5} {aa_s:>9} {pf['n_burst']:>5} {pf['n_quiet']:>5} "
              f"{mmb:>11} {'PASS' if pf['fork_passes_p2'] else 'fail':>9}")
    print()
    print_auroc_table(ev)

    # D5 adaptation diagnostic (Arm A)
    print("D5 adaptation diagnostic (Arm A, first-2 vs last-2 burst samples):")
    for pf in ev["per_fork_a"]:
        seed = pf["fork_seed"]
        samples = pf["_samples"]
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            b_samples = [s for s in samples
                         if s["label"] == "BURST" and bstart < s["target_step"] <= bend]
            if len(b_samples) >= 4:
                first2 = [s["mismatch"] for s in b_samples[:2]]
                last2 = [s["mismatch"] for s in b_samples[-2:]]
                ratio = np.mean(first2) / (np.mean(last2) + 1e-300)
                print(f"  seed={seed} burst={bi} "
                      f"first2={[f'{x:.4f}' for x in first2]} "
                      f"last2={[f'{x:.4f}' for x in last2]} "
                      f"ratio={ratio:.2f} {'>=2' if ratio >= D5_RATIO else '<2'}")
    print(f"D5 fraction (bursts with ratio >= {D5_RATIO}): {ev['d5_fraction']:.3f}")
    print()

    # --- Summary + Verdict ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "fail")
    print(f"P2 (sensitivity): forks with AUROC_A >= {P2_AUROC_MIN}: "
          f"{ev['forks_pass_p2']}/8 (need >= {P2_MIN_FORKS}); "
          f"median AUROC_A = {ev['median_auroc_a']:.3f} "
          f"(F2 if <= {F2_MEDIAN_MAX}) => {p2_status}")
    p3_status = "PASS" if ev["p3_pass"] else ("voided-PC3b" if not pc3b_valid else "fail")
    print(f"P3 (specificity): pairs with delta >= {P3_DELTA_MIN}: "
          f"{ev['p3_n_pass']}/8 (need >= {P3_MIN_PAIRS}); "
          f"median AUROC_B = {ev['median_auroc_b']:.3f} (<= {P3_B_MEDIAN_MAX}?={ev['p3_b_ok']}); "
          f"F3_flag={ev['f3_flag']} => {p3_status}")
    print(f"P4 (argmax-independence): flickering_forks={ev['flickering_seeds']} "
          f"evaluable={ev['p4_evaluable']} => {ev['p4_status']}")
    print(f"D5 (adaptation fraction): {ev['d5_fraction']:.3f}")
    print()
    tier_texts = {
        "rung-2": "RUNG-2 EVIDENCE — monitor detects + specifies identity displacement",
        "negative": "NO METACOGNITIVE SENSITIVITY — rung-2 kill (F2)",
        "mixed-argmax-dependent": "MIXED — sensitive but argmax-dependent (P4i failed)",
        "rung-1.5": "MIXED — sensitive but specificity unresolved",
        "between-bands": "MIXED — between bands",
    }
    tier_text = tier_texts.get(ev["tier"], ev["tier"])
    print(f"VERDICT: {ev['verdict_str']}  ({tier_text})")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows = Path(__file__).parent / "outputs" / "exp178_rows.json"
    write_json_rows(run_results_a, run_results_b, ev=ev,
                    pc2p_rows=pc2p_rows, pc3b_rows=pc3b_rows, out_path=out_rows)
    print(f"Rows written to {out_rows}")

    # --- Write verdict JSON ---
    arms = {
        "P2_sensitivity": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"forks with AUROC_A>={P2_AUROC_MIN}: {ev['forks_pass_p2']}/8 "
                f"(need >={P2_MIN_FORKS}); "
                f"median AUROC_A={ev['median_auroc_a']:.3f} "
                f"(F2 if <={F2_MEDIAN_MAX})"
            ),
        },
        "P3_specificity": {
            "pass": bool(ev["p3_pass"]),
            "reason": (
                f"pairs with AUROC_A-AUROC_B>={P3_DELTA_MIN}: {ev['p3_n_pass']}/8 "
                f"(need >={P3_MIN_PAIRS}); "
                f"median AUROC_B={ev['median_auroc_b']:.3f} (<={P3_B_MEDIAN_MAX}?={ev['p3_b_ok']}); "
                f"PC3b_valid={pc3b_valid}"
            ),
        },
        "P4_argmax_independence": {
            "pass": ev["p4_status"] in ("pass", "not-evaluable"),
            "reason": (
                f"flickering_forks={ev['flickering_seeds']}; "
                f"evaluable={ev['p4_evaluable']}; status={ev['p4_status']}"
            ),
        },
        "PC3b_control_validity": {
            "pass": bool(pc3b_valid),
            "reason": (
                "; ".join(pc3b_failures) if pc3b_failures
                else "all bursts within TV bar"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp178_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp178",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=not gate_pass,
        notes=(
            f"N4 rung 2 attempt 2, tier={ev['tier']}. "
            "Pre-registered card in loop/directions/identity-n4.md; "
            "PC2' vector-grade gate (TV(pi_pre, pi_onset) <= 0.05, replaces argmax-constancy); "
            "scramble specificity control (Arm B, rng 175000+seed, whole-grid relocation); "
            "argmax covariate only (flicker count logged, never gated). "
            f"D5 adaptation fraction={ev['d5_fraction']:.3f}. "
            f"P4 status={ev['p4_status']} (flickering={ev['flickering_seeds']})."
        ),
    )
    print(f"Verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
