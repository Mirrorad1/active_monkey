"""Exp 181 DYNAMICS AUDIT — DIAGNOSTIC ONLY.

This script is a deterministic diagnostic audit of the committed Exp 181 result.
It does NOT and CANNOT alter the Exp 181 verdict (NEGATIVE, no-resistance).

PURPOSE: After the Exp 181 NEGATIVE result, the human requested a richer post-hoc
analysis of Phase-W dynamics to understand *why* resistance failed — specifically:
  - How much identity-relevant writing leaked into v during each burst
  - Whether the v-gap between favorite and burst-color widened or narrowed
  - What the onset-lag of the N4 controller's g-suppression looks like step-by-step
  - Whether the N4 arm's leak is truly larger or smaller than baseline (ratio)

METHOD: Re-runs Phase W for all 7 arms x 8 seeds with per-step instrumentation
(w_t, g_t, obs_t recorded at every step during bursts), then validates the recomputed
leaked_writing values against the committed exp181_rows.json to a tolerance of 1e-3.
Any disagreement is a fatal assertion (the committed record is ground truth).

The verdict section at the bottom of experiments/outputs/exp181_verdict.json is
unchanged; this script writes only to:
  - experiments/outputs/exp181_dynamics_audit.txt
  - experiments/outputs/exp181_dynamics_audit.json

It shares no state with the original exp181_n4_controller.py run and makes no
calls to write_verdict or any verdict-bearing path.

SELF-VALIDATION: For the n4 arm, recomputed L_raw per seed/burst is compared
against the committed leaked_writing column from exp181_rows.json. Agreement
within 1e-3 confirms the rerun is bit-for-bit consistent with the original.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import constants and helpers from exp181_n4_controller
# ---------------------------------------------------------------------------
from experiments.exp181_n4_controller import (
    SEEDS,
    ARMS,
    BURST_WINDOWS,
    LAMBDA,
    EVAL,
    MON_W_SNAPS,
    CTRL_MBAR_WINDOW,
    CTRL_MIN_SNAPS,
    BURST_SEED_OFFSET_W,
    pi_of,
    tv,
    snap_index,
)

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Derived constants (match exp181)
# ---------------------------------------------------------------------------
N_STEPS = 15_000
CHUNK_SIZE = 100
N_CHUNKS = N_STEPS // CHUNK_SIZE
INIT_MASS = 1.0 / (1.0 - LAMBDA)   # ~3333.3

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_OUT_DIR = Path(__file__).parent / "outputs"
_AUDIT_TXT = _OUT_DIR / "exp181_dynamics_audit.txt"
_AUDIT_JSON = _OUT_DIR / "exp181_dynamics_audit.json"
_ROWS_JSON = Path(__file__).parent / "outputs" / "exp181_rows.json"


# ---------------------------------------------------------------------------
# Instrumented Phase-W fork runner
# ---------------------------------------------------------------------------

def run_fork_audit(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    arm_name: str,
    arm_mode: object,
) -> dict:
    """Reproduce exp181's Phase-W run_fork EXACTLY (same seeding, same controller,
    same burst mechanics, same update v *= LAMBDA; v[obs] += g*w), but additionally
    records per-step w_t, g_t, obs_t during burst windows.

    Phase path: "W" only (this audit covers Phase W dynamics exclusively).

    Chunk seeding: fork_seed * 10_000 + chunk_idx (identical to exp181).
    Burst RNG: BURST_SEED_OFFSET_W + fork_seed (identical to exp181).

    Returns a dict containing:
      - v_traj, g_traj, expressed_arr (same as exp181)
      - burst_preburst_fav, burst_onset_color (same)
      - leaked_writing_w (same — used for self-validation)
      - per_burst_steps: list of 3 dicts, one per burst, each with arrays
        {w: [...], g: [...], obs: [...]} for every step in [bstart, bend)
    """
    phase = "W"
    fork_name = f"exp181_audit_{arm_name}_{phase}_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)
    n_actions = 4

    # Initialize v (identical to exp181)
    vc = c.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v = (vc / vc_sum) * INIT_MASS
    else:
        v = np.ones(n_colors) * (INIT_MASS / n_colors)

    # Per-color cell lists (identical to exp181)
    color_cells: list[list[int]] = [[] for _ in range(n_colors)]
    for cell_idx, color in enumerate(base_cmap):
        color_cells[color].append(cell_idx)
    color_cells_arr = [np.array(lst, dtype=np.int32) for lst in color_cells]

    # Burst RNG (identical to exp181)
    burst_rng = np.random.default_rng(BURST_SEED_OFFSET_W + fork_seed)

    # Build burst step set
    burst_step_set: set[int] = set()
    for bstart, bend in BURST_WINDOWS:
        burst_step_set.update(range(bstart, bend))

    # Per-step storage (identical to exp181)
    expressed_arr = np.empty(N_STEPS, dtype=np.int32)
    true_pos_arr = np.empty(N_STEPS, dtype=np.int32)
    obs_arr = np.empty(N_STEPS, dtype=np.int32)

    v_traj: list[np.ndarray] = []
    g_traj: list[float] = []

    # N4 controller state (identical to exp181)
    n4_mismatch_history: list[float] = []
    g_current = 1.0

    # Phase W diagnostics (identical to exp181)
    burst_preburst_fav: list[int | None] = [None] * len(BURST_WINDOWS)
    burst_onset_color: list[int | None] = [None] * len(BURST_WINDOWS)
    current_burst_color: int | None = None
    current_burst_idx: int | None = None

    # leaked_writing (identical to exp181) — used for self-validation
    leaked_writing_w: list[float] = [0.0] * len(BURST_WINDOWS)

    # ADDITIONAL instrumentation: per-burst per-step records
    # pre-allocate lists; we'll fill them during burst steps
    _burst_lens = [bend - bstart for bstart, bend in BURST_WINDOWS]
    per_burst_w: list[list[float]] = [[] for _ in range(len(BURST_WINDOWS))]
    per_burst_g: list[list[float]] = [[] for _ in range(len(BURST_WINDOWS))]
    per_burst_obs: list[list[int]] = [[] for _ in range(len(BURST_WINDOWS))]

    global_step = 0

    for chunk_idx in range(N_CHUNKS):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(CHUNK_SIZE):
            t = global_step

            # ----------------------------------------------------------------
            # Phase W: burst context management (identical to exp181)
            # ----------------------------------------------------------------
            in_burst = t in burst_step_set
            burst_idx_now: int | None = None

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
            # Observe (identical to exp181)
            # ----------------------------------------------------------------
            obs = int(base_cmap[c.true_pos])
            obs_arr[t] = obs
            true_pos_arr[t] = c.true_pos

            # ----------------------------------------------------------------
            # A_hat pre-step (identical to exp181)
            # ----------------------------------------------------------------
            A_hat = c._A_hat()

            # ----------------------------------------------------------------
            # Belief update (identical to exp181)
            # ----------------------------------------------------------------
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom_qs = qs_updated.sum()
            if denom_qs > 0:
                qs_updated = qs_updated / denom_qs
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # ----------------------------------------------------------------
            # Dirichlet count update (identical to exp181)
            # ----------------------------------------------------------------
            c.pA[obs, :] += qs_updated

            # ----------------------------------------------------------------
            # Predictability weight (identical to exp181)
            # ----------------------------------------------------------------
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            predictability_weight = math.exp(-h_predicted)

            # ----------------------------------------------------------------
            # Determine g_t for this step (identical to exp181)
            # ----------------------------------------------------------------
            if arm_mode == "baseline":
                g_t = 1.0
            elif arm_mode == "n4":
                g_t = g_current
            else:
                g_t = float(arm_mode)

            # ----------------------------------------------------------------
            # Value update (identical to exp181)
            # ----------------------------------------------------------------
            v *= LAMBDA
            v[obs] += g_t * predictability_weight
            c.value_counts[obs] += g_t * predictability_weight

            # ----------------------------------------------------------------
            # Expressed preference (identical to exp181)
            # ----------------------------------------------------------------
            expressed = int(np.argmax(v))
            expressed_arr[t] = expressed

            # ----------------------------------------------------------------
            # Leaked writing diagnostic (identical to exp181)
            # ----------------------------------------------------------------
            if arm_mode == "n4" and burst_idx_now is not None:
                bc_now = burst_onset_color[burst_idx_now]
                if bc_now is not None and obs == bc_now:
                    leaked_writing_w[burst_idx_now] += g_t * predictability_weight

            # ----------------------------------------------------------------
            # ADDITIONAL: record per-burst per-step instrumentation
            # ----------------------------------------------------------------
            if burst_idx_now is not None:
                per_burst_w[burst_idx_now].append(predictability_weight)
                per_burst_g[burst_idx_now].append(g_t)
                per_burst_obs[burst_idx_now].append(obs)

            # ----------------------------------------------------------------
            # Action / Move (identical to exp181)
            # ----------------------------------------------------------------
            if in_burst:
                cells_of_bc = color_cells_arr[current_burst_color]
                if len(cells_of_bc) > 0:
                    c.true_pos = int(burst_rng.choice(cells_of_bc))
                qs_next = np.zeros(n_cells)
                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                c.qs = qs_next
            else:
                action = int(rng.integers(0, n_actions))
                c.true_pos = c.world.move(c.true_pos, action)
                c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1
            global_step += 1

            # ----------------------------------------------------------------
            # Snapshot v and update N4 controller (identical to exp181)
            # ----------------------------------------------------------------
            if global_step % EVAL == 0:
                v_traj.append(v.copy())
                k = len(v_traj) - 1

                if arm_mode == "n4":
                    if k >= MON_W_SNAPS:
                        v_prev = v_traj[k - 1]
                        v_drift_ref = v_traj[k - 1 - MON_W_SNAPS]
                        v_hat = v_prev + (v_prev - v_drift_ref) / MON_W_SNAPS
                        m_k = float(np.linalg.norm(v_hat - v_traj[k]))
                        n4_mismatch_history.append(m_k)

                        n_hist = len(n4_mismatch_history)
                        if n_hist >= CTRL_MIN_SNAPS:
                            tail = n4_mismatch_history[-CTRL_MBAR_WINDOW:]
                            m_bar = statistics.median(tail)
                            if m_k > 0:
                                g_new = min(1.0, (m_bar / m_k) ** 2)
                            else:
                                g_new = 1.0
                            g_current = g_new

                g_traj.append(
                    g_current if arm_mode == "n4"
                    else (1.0 if arm_mode == "baseline" else float(arm_mode))
                )

    return {
        "fork_seed": fork_seed,
        "arm_name": arm_name,
        "arm_mode": arm_mode,
        "v_traj": np.array(v_traj),
        "g_traj": g_traj,
        "expressed_arr": expressed_arr,
        "burst_preburst_fav": burst_preburst_fav,
        "burst_onset_color": burst_onset_color,
        "leaked_writing_w": list(leaked_writing_w),
        # ADDITIONAL instrumentation
        "per_burst_w": [list(per_burst_w[bi]) for bi in range(len(BURST_WINDOWS))],
        "per_burst_g": [list(per_burst_g[bi]) for bi in range(len(BURST_WINDOWS))],
        "per_burst_obs": [list(per_burst_obs[bi]) for bi in range(len(BURST_WINDOWS))],
    }


# ---------------------------------------------------------------------------
# Load committed rows for self-validation
# ---------------------------------------------------------------------------

def load_committed_n4_leak(rows_path: Path) -> dict:
    """Load committed n4 Phase-W leaked_writing from exp181_rows.json.

    Returns dict keyed (seed, burst_idx) -> leaked_writing float.
    Also returns committed pre_fav and burst_color per (arm, seed, burst_idx).
    """
    leak: dict[tuple[int, int], float] = {}
    pre_fav_map: dict[tuple[str, int, int], int | None] = {}
    burst_color_map: dict[tuple[str, int, int], int | None] = {}

    with rows_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("phase") != "W":
                continue
            arm = row.get("arm", "")
            seed = row.get("fork_seed")
            bi = row.get("burst_idx")
            pre_fav_map[(arm, seed, bi)] = row.get("pre_fav")
            burst_color_map[(arm, seed, bi)] = row.get("burst_color")
            if arm == "n4" and row.get("leaked_writing") is not None:
                leak[(seed, bi)] = float(row["leaked_writing"])

    return leak, pre_fav_map, burst_color_map


# ---------------------------------------------------------------------------
# Compute identity dynamics for one arm/seed
# ---------------------------------------------------------------------------

def compute_identity_dynamics(rr: dict) -> list[dict]:
    """For each burst in a Phase-W result, compute identity dynamics.

    Returns list of 3 dicts (one per burst) with:
      pre_fav, burst_color, v_f_start, v_c_start, gap_start,
      v_f_end, v_c_end, gap_end, pi_start, pi_end,
      directed_drift_D_b (pi_end[c] - pi_start[c]),
      whole_vector_TV_b (0.5 * sum|pi_end - pi_start|)
    """
    v_traj = rr["v_traj"]
    burst_preburst_fav = rr["burst_preburst_fav"]
    burst_onset_color = rr["burst_onset_color"]
    results = []

    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
        f = burst_preburst_fav[bi]
        c_color = burst_onset_color[bi]

        # snap_index(bstart) is the snapshot JUST BEFORE the burst
        # (the last snapshot taken before step bstart)
        # snap_index(bstart) = bstart // EVAL - 1
        si_start = snap_index(bstart)   # index into v_traj
        si_end = snap_index(bend)       # index into v_traj after bend

        # Guard bounds
        n_snaps = len(v_traj)
        if si_start < 0 or si_start >= n_snaps:
            results.append({"burst_idx": bi, "error": f"si_start={si_start} OOB"})
            continue
        if si_end < 0 or si_end >= n_snaps:
            results.append({"burst_idx": bi, "error": f"si_end={si_end} OOB"})
            continue
        if f is None or c_color is None:
            results.append({"burst_idx": bi, "error": "missing fav or burst_color"})
            continue

        v_start = v_traj[si_start]
        v_end = v_traj[si_end]

        v_f_start = float(v_start[f])
        v_c_start = float(v_start[c_color])
        gap_start = v_f_start - v_c_start

        v_f_end = float(v_end[f])
        v_c_end = float(v_end[c_color])
        gap_end = v_f_end - v_c_end

        pi_s = pi_of(v_start)
        pi_e = pi_of(v_end)

        directed_drift = float(pi_e[c_color] - pi_s[c_color])
        tv_b = float(0.5 * np.abs(pi_e - pi_s).sum())

        results.append({
            "burst_idx": bi,
            "bstart": bstart,
            "bend": bend,
            "pre_fav": f,
            "burst_color": c_color,
            "v_f_start": v_f_start,
            "v_c_start": v_c_start,
            "gap_start": gap_start,
            "v_f_end": v_f_end,
            "v_c_end": v_c_end,
            "gap_end": gap_end,
            "pi_start": pi_s.tolist(),
            "pi_end": pi_e.tolist(),
            "directed_drift_D_b": directed_drift,
            "whole_vector_TV_b": tv_b,
        })

    return results


# ---------------------------------------------------------------------------
# Compute leak quantities for one arm/seed
# ---------------------------------------------------------------------------

def compute_leak(rr: dict, baseline_rr: dict | None) -> list[dict]:
    """Compute L_raw, L_disc, and leak_ratio_vs_baseline per burst.

    L_raw = sum over burst steps tau of g_tau * w_tau * 1[obs_tau == c]
    L_disc = sum over burst steps tau of LAMBDA^(bend-1-tau) * g_tau * w_tau * 1[obs_tau == c]
    leak_ratio_vs_baseline = L_raw(arm) / L_raw(baseline) for same seed/burst

    For baseline arm g==1 so L_raw(baseline) = sum w_tau * 1[obs_tau == c].
    """
    results = []
    burst_onset_color = rr["burst_onset_color"]

    # Pre-build baseline L_raw if available
    baseline_L_raw: dict[int, float] = {}
    if baseline_rr is not None:
        bl_onset_color = baseline_rr["burst_onset_color"]
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            bl_c = bl_onset_color[bi]
            if bl_c is None:
                baseline_L_raw[bi] = float("nan")
                continue
            bl_w_arr = baseline_rr["per_burst_w"][bi]
            bl_g_arr = baseline_rr["per_burst_g"][bi]
            bl_obs_arr = baseline_rr["per_burst_obs"][bi]
            L = 0.0
            for tau_idx, (obs_t, g_t, w_t) in enumerate(zip(bl_obs_arr, bl_g_arr, bl_w_arr)):
                if obs_t == bl_c:
                    L += g_t * w_t  # g==1 for baseline but keep general
            baseline_L_raw[bi] = L

    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
        c_color = burst_onset_color[bi]
        if c_color is None:
            results.append({"burst_idx": bi, "error": "no burst_color"})
            continue

        w_arr = rr["per_burst_w"][bi]
        g_arr = rr["per_burst_g"][bi]
        obs_arr_b = rr["per_burst_obs"][bi]

        L_raw = 0.0
        L_disc = 0.0
        burst_len = bend - bstart  # == len(w_arr)

        for tau_idx, (obs_t, g_t, w_t) in enumerate(zip(obs_arr_b, g_arr, w_arr)):
            if obs_t == c_color:
                L_raw += g_t * w_t
                # tau = bstart + tau_idx
                # discount factor = LAMBDA^(bend - 1 - tau) = LAMBDA^(burst_len - 1 - tau_idx)
                disc = LAMBDA ** (burst_len - 1 - tau_idx)
                L_disc += disc * g_t * w_t

        bl_L = baseline_L_raw.get(bi, float("nan"))
        if not math.isnan(bl_L) and bl_L > 0:
            ratio = L_raw / bl_L
        else:
            ratio = float("nan")

        results.append({
            "burst_idx": bi,
            "bstart": bstart,
            "bend": bend,
            "burst_color": c_color,
            "L_raw": L_raw,
            "L_disc": L_disc,
            "baseline_L_raw": bl_L,
            "leak_ratio_vs_baseline": ratio,
        })

    return results


# ---------------------------------------------------------------------------
# Compute onset-lag diagnostic (n4 arm only)
# ---------------------------------------------------------------------------

def compute_onset_lag(rr: dict) -> list[dict]:
    """For each burst in the n4 arm, report:
      - g at snapshot index bstart//EVAL - 1 (pre-onset snapshot)
      - g at snapshot indices bstart//EVAL .. bstart//EVAL + 3
      - first-100-step exposure: sum g_tau * w_tau * 1[obs_tau == c] for tau in [bstart, bstart+100)
    """
    results = []
    g_traj = rr["g_traj"]
    burst_onset_color = rr["burst_onset_color"]

    for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
        c_color = burst_onset_color[bi]
        si_onset = bstart // EVAL  # first snapshot index at or after bstart

        # pre-onset snapshot
        si_pre = si_onset - 1

        g_pre = g_traj[si_pre] if si_pre >= 0 else float("nan")
        g_at_onward = [
            g_traj[si_onset + delta] if (si_onset + delta) < len(g_traj) else float("nan")
            for delta in range(4)
        ]

        # first-100-step exposure: per_burst data, first 100 steps
        w_arr = rr["per_burst_w"][bi]
        g_arr = rr["per_burst_g"][bi]
        obs_arr_b = rr["per_burst_obs"][bi]

        first_100_exposure = 0.0
        if c_color is not None:
            for tau_idx in range(min(100, len(obs_arr_b))):
                if obs_arr_b[tau_idx] == c_color:
                    first_100_exposure += g_arr[tau_idx] * w_arr[tau_idx]

        results.append({
            "burst_idx": bi,
            "bstart": bstart,
            "burst_color": c_color,
            "si_pre": si_pre,
            "g_pre_onset": g_pre,
            "si_onset": si_onset,
            "g_at_onset_snaps": g_at_onward,   # snaps si_onset to si_onset+3
            "first_100_step_exposure": first_100_exposure,
        })

    return results


# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------

def self_validate(
    all_results: dict,   # {(arm_name, seed): run_fork_audit result}
    committed_leak: dict,   # {(seed, burst_idx): float}
    committed_pre_fav: dict,   # {(arm, seed, bi): int|None}
    committed_burst_color: dict,  # {(arm, seed, bi): int|None}
) -> tuple[bool, list[str]]:
    """Assert n4 L_raw matches committed leaked_writing to 1e-3.
    Also assert pre_fav and burst_color match for all arms.

    Returns (all_pass, list_of_failure_messages).
    """
    failures: list[str] = []

    # Check n4 leak
    for seed in SEEDS:
        key = ("n4", seed)
        if key not in all_results:
            failures.append(f"MISSING n4 seed {seed} from rerun")
            continue
        rr = all_results[key]
        for bi in range(len(BURST_WINDOWS)):
            L_raw_rerun = rr["leaked_writing_w"][bi]
            committed_val = committed_leak.get((seed, bi))
            if committed_val is None:
                failures.append(f"MISSING committed n4 leak seed={seed} burst={bi}")
                continue
            diff = abs(L_raw_rerun - committed_val)
            if diff > 1e-3:
                failures.append(
                    f"LEAK MISMATCH n4 seed={seed} burst={bi}: "
                    f"rerun={L_raw_rerun:.6f} committed={committed_val:.6f} diff={diff:.2e}"
                )

    # Check pre_fav and burst_color for all arms
    arm_names = [a[0] for a in ARMS]
    for arm_name in arm_names:
        for seed in SEEDS:
            key = (arm_name, seed)
            if key not in all_results:
                failures.append(f"MISSING arm={arm_name} seed={seed} from rerun")
                continue
            rr = all_results[key]
            for bi in range(len(BURST_WINDOWS)):
                rerun_pf = rr["burst_preburst_fav"][bi]
                rerun_bc = rr["burst_onset_color"][bi]
                committed_pf = committed_pre_fav.get((arm_name, seed, bi))
                committed_bc = committed_burst_color.get((arm_name, seed, bi))
                if committed_pf is not None and rerun_pf != committed_pf:
                    failures.append(
                        f"PRE_FAV MISMATCH arm={arm_name} seed={seed} burst={bi}: "
                        f"rerun={rerun_pf} committed={committed_pf}"
                    )
                if committed_bc is not None and rerun_bc != committed_bc:
                    failures.append(
                        f"BURST_COLOR MISMATCH arm={arm_name} seed={seed} burst={bi}: "
                        f"rerun={rerun_bc} committed={committed_bc}"
                    )

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_float(x: float | None, width: int = 10, decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan".rjust(width)
    return f"{x:.{decimals}f}".rjust(width)


def fmt_int_or_none(x: int | None, width: int = 5) -> str:
    if x is None:
        return "None".rjust(width)
    return str(x).rjust(width)


# ---------------------------------------------------------------------------
# Build output lines
# ---------------------------------------------------------------------------

def build_output(
    all_results: dict,
    baseline_results: dict,   # seed -> rr for baseline arm
    identity_rows: list[dict],   # flat rows
    leak_rows: list[dict],
    onset_lag_rows: list[dict],
    validation_pass: bool,
    validation_failures: list[str],
) -> list[str]:
    """Return lines for the human-readable text output."""
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    ln("=" * 80)
    ln("Exp 181 DYNAMICS AUDIT — DIAGNOSTIC ONLY")
    ln("Deterministic Phase-W rerun with per-step instrumentation")
    ln("Verdict unchanged: NEGATIVE (no-resistance)")
    ln("=" * 80)
    ln()

    # ---- Identity dynamics table ----
    ln("TABLE 1: Identity dynamics per arm/seed/burst")
    ln("  gap_start = v_fav - v_burst_color at burst onset snapshot")
    ln("  gap_end   = v_fav - v_burst_color at burst end snapshot")
    ln("  D_b       = pi_end[burst_color] - pi_start[burst_color]  (directed drift)")
    ln("  TV_b      = 0.5 * sum|pi_end - pi_start|  (whole-vector movement)")
    ln()
    hdr = (f"{'arm':>10} {'seed':>5} {'bi':>3} {'f':>3} {'c':>3} "
           f"{'gap_start':>10} {'gap_end':>10} {'D_b':>10} {'TV_b':>10}")
    ln(hdr)
    ln("-" * len(hdr))
    for row in identity_rows:
        if "error" in row:
            ln(f"{'':>10} {'':>5} {row['burst_idx']:>3} ERROR: {row['error']}")
            continue
        arm = row["arm_name"]
        seed = row["seed"]
        bi = row["burst_idx"]
        f = row.get("pre_fav", "?")
        c_c = row.get("burst_color", "?")
        gs = row.get("gap_start", float("nan"))
        ge = row.get("gap_end", float("nan"))
        db = row.get("directed_drift_D_b", float("nan"))
        tvb = row.get("whole_vector_TV_b", float("nan"))
        ln(f"{arm:>10} {seed:>5} {bi:>3} {str(f):>3} {str(c_c):>3} "
           f"{fmt_float(gs)} {fmt_float(ge)} {fmt_float(db)} {fmt_float(tvb)}")
    ln()

    # ---- Leak table ----
    ln("TABLE 2: Leak quantities per arm/seed/burst")
    ln("  L_raw  = sum g_tau*w_tau * 1[obs==burst_color]  over burst")
    ln("  L_disc = sum LAMBDA^(bend-1-tau) * g_tau*w_tau * 1[obs==burst_color]")
    ln("  ratio  = L_raw(arm) / L_raw(baseline) for same seed/burst")
    ln()
    hdr2 = (f"{'arm':>10} {'seed':>5} {'bi':>3} "
            f"{'L_raw':>12} {'L_disc':>12} {'bl_L_raw':>12} {'ratio':>8}")
    ln(hdr2)
    ln("-" * len(hdr2))
    for row in leak_rows:
        if "error" in row:
            ln(f"{'':>10} {'':>5} {row['burst_idx']:>3} ERROR: {row['error']}")
            continue
        arm = row["arm_name"]
        seed = row["seed"]
        bi = row["burst_idx"]
        L_raw = row.get("L_raw", float("nan"))
        L_disc = row.get("L_disc", float("nan"))
        bl_L = row.get("baseline_L_raw", float("nan"))
        ratio = row.get("leak_ratio_vs_baseline", float("nan"))
        ln(f"{arm:>10} {seed:>5} {bi:>3} "
           f"{fmt_float(L_raw, 12, 4)} {fmt_float(L_disc, 12, 4)} "
           f"{fmt_float(bl_L, 12, 4)} {fmt_float(ratio, 8, 4)}")
    ln()

    # ---- Onset-lag table (n4 arm only) ----
    ln("TABLE 3: N4 onset-lag diagnostic (n4 arm)")
    ln("  g_pre    = g at snapshot si_onset - 1 (the step just before burst onset)")
    ln("  g_on[0..3] = g at snapshots si_onset, si_onset+1, si_onset+2, si_onset+3")
    ln("  first100 = sum g*w*1[obs==burst_color] over first 100 burst steps")
    ln()
    hdr3 = (f"{'seed':>5} {'bi':>3} {'bc':>3} "
            f"{'g_pre':>8} {'g_on0':>8} {'g_on1':>8} {'g_on2':>8} {'g_on3':>8} "
            f"{'first100':>10}")
    ln(hdr3)
    ln("-" * len(hdr3))
    for row in onset_lag_rows:
        if "error" in row:
            ln(f"{'':>5} {row['burst_idx']:>3} ERROR: {row['error']}")
            continue
        seed = row["seed"]
        bi = row["burst_idx"]
        bc = row.get("burst_color", "?")
        g_pre = row.get("g_pre_onset", float("nan"))
        g_on = row.get("g_at_onset_snaps", [float("nan")] * 4)
        f100 = row.get("first_100_step_exposure", float("nan"))
        ln(f"{seed:>5} {bi:>3} {str(bc):>3} "
           f"{fmt_float(g_pre, 8, 5)} "
           f"{fmt_float(g_on[0], 8, 5)} {fmt_float(g_on[1], 8, 5)} "
           f"{fmt_float(g_on[2], 8, 5)} {fmt_float(g_on[3], 8, 5)} "
           f"{fmt_float(f100, 10, 4)}")
    ln()

    # ---- Self-validation ----
    ln("=" * 80)
    ln("SELF-VALIDATION")
    ln("=" * 80)
    if validation_pass:
        ln("SELF-VALIDATION PASS (n4 leak matches committed record)")
        ln("  All recomputed L_raw values agree with committed leaked_writing within 1e-3.")
        ln("  All arm/seed pre_fav and burst_color values match committed rows.")
    else:
        ln("SELF-VALIDATION FAIL — discrepancies found:")
        for f_msg in validation_failures:
            ln(f"  {f_msg}")
    ln()
    ln("Meta: verdict unchanged = NEGATIVE (no-resistance)")
    ln("Source: deterministic rerun of exp181 Phase W with per-step instrumentation")
    ln("=" * 80)

    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 181 DYNAMICS AUDIT — DIAGNOSTIC ONLY")
    print("Deterministic Phase-W rerun with per-step instrumentation")
    print("Verdict unchanged: NEGATIVE (no-resistance)")
    print("=" * 80)
    print()

    # Load committed rows
    committed_leak, committed_pre_fav, committed_burst_color = load_committed_n4_leak(
        _ROWS_JSON
    )
    n_committed_leak = len(committed_leak)
    print(f"Loaded {n_committed_leak} committed n4 Phase-W leak values from {_ROWS_JSON}")
    print()

    # Load mirro spine
    mirro = Creature.load("creature/state/mirro")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    print(f"Loaded mirro: age={mirro.age_steps}, n_colors={n_colors}")
    print()

    arm_names_ordered = [a[0] for a in ARMS]

    # Run all arms x seeds (Phase W only)
    all_results: dict[tuple[str, int], dict] = {}

    total_sessions = len(ARMS) * len(SEEDS)
    session_idx = 0
    for arm_name, arm_mode in ARMS:
        for seed in SEEDS:
            session_idx += 1
            print(f"  [{session_idx}/{total_sessions}] arm={arm_name} seed={seed} ...", flush=True)
            rr = run_fork_audit(mirro, seed, base_cmap, n_colors, arm_name, arm_mode)
            all_results[(arm_name, seed)] = rr
        print()

    # Build per-arm/seed baseline lookup
    baseline_by_seed: dict[int, dict] = {
        seed: all_results[("baseline", seed)]
        for seed in SEEDS
        if ("baseline", seed) in all_results
    }

    # Compute identity dynamics, leak, onset-lag
    identity_rows: list[dict] = []
    leak_rows: list[dict] = []
    onset_lag_rows: list[dict] = []

    for arm_name, arm_mode in ARMS:
        for seed in SEEDS:
            key = (arm_name, seed)
            if key not in all_results:
                continue
            rr = all_results[key]
            bl_rr = baseline_by_seed.get(seed)

            # Identity dynamics
            dyn = compute_identity_dynamics(rr)
            for d in dyn:
                row = {"arm_name": arm_name, "seed": seed, **d}
                identity_rows.append(row)

            # Leak
            lk = compute_leak(rr, bl_rr)
            for d in lk:
                row = {"arm_name": arm_name, "seed": seed, **d}
                leak_rows.append(row)

            # Onset-lag (n4 only)
            if arm_name == "n4":
                lag = compute_onset_lag(rr)
                for d in lag:
                    row = {"seed": seed, **d}
                    onset_lag_rows.append(row)

    # Self-validation
    val_pass, val_failures = self_validate(
        all_results, committed_leak, committed_pre_fav, committed_burst_color
    )

    if val_pass:
        print("SELF-VALIDATION PASS (n4 leak matches committed record)")
    else:
        print("SELF-VALIDATION FAIL:")
        for f_msg in val_failures:
            print(f"  {f_msg}")
        # Assert loudly — mismatch means the rerun drifted from the original
        assert False, (
            "SELF-VALIDATION FAILED — rerun does not reproduce committed Exp 181 values. "
            "Failures:\n" + "\n".join(val_failures)
        )

    # Build text output
    out_lines = build_output(
        all_results,
        baseline_by_seed,
        identity_rows,
        leak_rows,
        onset_lag_rows,
        val_pass,
        val_failures,
    )
    out_text = "\n".join(out_lines)

    # Print to stdout
    print()
    print(out_text)

    # Write txt
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _AUDIT_TXT.write_text(out_text + "\n")
    print(f"\nWrote {_AUDIT_TXT}")

    # Build JSON output
    # Collect flat row dicts (one per arm/seed/burst) — keep only serializable data
    json_rows: list[dict] = []
    for arm_name, arm_mode in ARMS:
        for seed in SEEDS:
            key = (arm_name, seed)
            if key not in all_results:
                continue
            rr = all_results[key]
            bl_rr = baseline_by_seed.get(seed)

            dyn = compute_identity_dynamics(rr)
            lk = compute_leak(rr, bl_rr)

            for bi in range(len(BURST_WINDOWS)):
                d = dyn[bi] if bi < len(dyn) else {}
                l = lk[bi] if bi < len(lk) else {}
                row: dict = {
                    "arm": arm_name,
                    "seed": seed,
                    "burst_idx": bi,
                }
                # identity dynamics fields
                for field in [
                    "pre_fav", "burst_color",
                    "v_f_start", "v_c_start", "gap_start",
                    "v_f_end", "v_c_end", "gap_end",
                    "directed_drift_D_b", "whole_vector_TV_b",
                ]:
                    row[field] = d.get(field)
                # leak fields
                for field in ["L_raw", "L_disc", "baseline_L_raw", "leak_ratio_vs_baseline"]:
                    val = l.get(field)
                    if isinstance(val, float) and math.isnan(val):
                        val = None
                    row[field] = val
                # error field
                if "error" in d or "error" in l:
                    row["error"] = d.get("error") or l.get("error")
                json_rows.append(row)

    # N4 onset-lag section
    n4_onset_lag_section: list[dict] = []
    for seed in SEEDS:
        key = ("n4", seed)
        if key not in all_results:
            continue
        rr = all_results[key]
        lag = compute_onset_lag(rr)
        for d in lag:
            n4_onset_lag_section.append({"seed": seed, **d})

    # Self-validation summary
    val_summary: dict = {
        "pass": val_pass,
        "n_committed_leak_entries": n_committed_leak,
        "failures": val_failures,
    }

    # Meta section
    meta: dict = {
        "diagnostic_only": True,
        "verdict_unchanged": "NEGATIVE (no-resistance)",
        "source": (
            "deterministic rerun of exp181 Phase W with per-step instrumentation"
        ),
        "self_validation": val_summary,
        "arms": arm_names_ordered,
        "seeds": SEEDS,
        "burst_windows": BURST_WINDOWS,
    }

    json_out = {
        "rows": json_rows,
        "n4_onset_lag": n4_onset_lag_section,
        "meta": meta,
    }

    # Convert any remaining numpy types
    def _np_convert(obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with _AUDIT_JSON.open("w") as fh:
        json.dump(json_out, fh, indent=2, default=_np_convert)

    print(f"Wrote {_AUDIT_JSON}")


if __name__ == "__main__":
    main()
