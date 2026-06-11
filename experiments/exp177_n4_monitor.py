"""Exp 177 — N4 rung 2: the IDENTITY MONITOR (read-only). On the verified
displacement regime (Exp 176: lambda=0.9997, 15000-step sessions, 800-step
captivity bursts at 6000/9000/12000), give the fork an N4 instrument that
predicts its OWN value vector by linear drift from its own history and ask
whether the mismatch "I am not who I predicted I would be" separates burst
windows from quiet windows (card: loop/directions/identity-n4.md, rung 2).

The monitor is READ-ONLY: it steers nothing (the Exp 157/163 instrument
discipline). Provided vs self-formed, named: the monitor FORM (linear
drift, window W, horizon DELTA, L2 mismatch) is PROVIDED; the value
content, drift history, and every mismatch value are SELF-FORMED.

Monitor (declared): v is snapshotted every EVAL=100 steps (after the step
update): v_snap[k] = v at step (k+1)*100, k = 0..148 (149 snapshots).
At each snapshot index k with k >= 10, predict the NEXT snapshot by linear
drift over the last W=1000 steps (10 snapshots):
    v_hat[k+1] = v_snap[k] + (v_snap[k] - v_snap[k-10]) / 10
    mismatch[k+1] = || v_hat[k+1] - v_snap[k+1] ||_2
Sample labels (declared): a mismatch sample at target snapshot k+1 covers
steps (s-100, s] where s = (k+2)*100... simplified: the sample's target
step is s = the step of v_snap[k+1]. Label BURST iff s lies inside a burst
window's interior-or-just-after, operationalized: any burst (bstart, bend]
with bstart < s <= bend. Label EXCLUDED (transient band, scored in
neither class) iff not BURST and s <= bend + 1000 for any burst. Label
QUIET otherwise. Per fork, AUROC of mismatch for BURST vs QUIET samples
is computed by the rank-based Mann-Whitney statistic (ties handled by
midranks).

Counts per fork: 138 scored-or-excluded samples (k+1 = 11..148); 24 BURST
(8 per burst), ~30 EXCLUDED, ~84 QUIET.

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 ahat_drift < 0.15; PC2 pre-burst stability
(constant expressed preference over the 1000 steps before each burst in
>= 7/8 forks); PC3 confinement >= 90% — all verbatim from Exp 176, which
passed them 8/8 on fresh seeds.

Predeclared property and falsifier:
  P2 (the monitor has metacognitive sensitivity over identity): per-fork
     AUROC(BURST vs QUIET mismatch) >= 0.8 in >= 7/8 forks (a declared
     margin of 0.3 over chance — the burst must be SEPARABLE, not merely
     above 0.5).
     FALSIFIER F2: median per-fork AUROC <= 0.5 — the mismatch signal does
     not separate burst from quiet at all (no metacognitive sensitivity
     over identity; the card's rung-2 kill).
  VERDICT: POSITIVE iff P2. NEGATIVE iff F2. Otherwise MIXED. "Not a
  falsifier" never counts toward POSITIVE.

Ungated diagnostics: per-fork AUROC; mean mismatch by label class;
mismatch trajectory at burst onsets (first/last burst samples — does the
drift estimator adapt within the burst?); flip/recovery table as in 176.
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

SEEDS = list(range(178, 186))   # 8 fresh forks (176 used 170-177)

N_STEPS = 15_000
CHUNK_SIZE = 100
N_CHUNKS = N_STEPS // CHUNK_SIZE   # 150

# Value decay
LAMBDA = 0.9997          # per-step decay; half-life ~2310 steps; equilibrium mass = 1/(1-lambda) ~ 3333
INIT_MASS = 1.0 / (1.0 - LAMBDA)  # ~ 3333.3

# Burst windows: [start, end) — inclusive start, exclusive end
BURST_WINDOWS = [
    (6000, 6800),
    (9000, 9800),
    (12000, 12800),
]
BURST_LEN = 800   # steps per burst

# Burst RNG seed offset
BURST_SEED_OFFSET = 160_000   # rng 160_000 + fork_seed

# Precondition thresholds
PC1_AHAT_DRIFT_MAX = 0.15     # horizon-scaled, Exp 166 law
PC2_STABILITY_WINDOW = 1000   # steps before each burst checked for stable preference
PC2_MIN_STABLE_FORKS = 7      # >= 7/8 forks must show pre-burst stability
PC3_CONFINEMENT_MIN = 0.90    # >= 90% of burst steps on burst-color cells

# P1' / F1' thresholds
P1_FLIP_FINAL_FRAC = 0.50      # >= 50% of burst's final 400 steps showing burst color
P1_FLIP_FINAL_WINDOW = 400     # final steps of burst to evaluate flip
P1_RECOVERY_LAG = 2000         # max steps after burst end to recover
P1_RECOVERY_HOLD = 500         # must hold consecutive steps after recovery
P1_MIN_BURSTS_PER_FORK = 2     # >= 2 of 3 bursts must show flip+recovery
P1_MIN_FORKS = 7               # >= 7/8 forks
F1_MAX_FORKS = 4               # <= 4/8 forks => falsifier

# Monitor (rung 2) — declared
EVAL = 100               # v snapshot cadence (steps)
MON_W_SNAPS = 10         # drift window = 10 snapshots = 1000 steps
EXCLUDE_AFTER = 1000     # transient band after burst end (steps), scored in neither class
P2_AUROC_MIN = 0.8       # per-fork AUROC bar
P2_MIN_FORKS = 7         # >= 7/8 forks
F2_MEDIAN_MAX = 0.5      # falsifier: median per-fork AUROC <= 0.5


# ---------------------------------------------------------------------------
# Core step-loop — run one fork for the full session
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    base_cmap: list,
    n_colors: int,
    n_actions: int = 4,
) -> dict:
    """Run one fork (loaded fresh from mirro) for N_STEPS steps.

    Step-loop semantics (per exp155 template):
      1. A_hat pre-step
      2. Observe cmap[true_pos]
      3. Bayes belief update
      4. pA Dirichlet count update
      5. Value accumulation: v *= lambda; v[obs] += predictability_weight
         (also update c.value_counts as usual — harmless)
      6. Action + Move (or, during burst: burst relocation + qs reset)
      7. Advance qs (or set to burst-color uniform during burst)
      8. age_steps += 1

    Returns per-step arrays and summary statistics.
    """
    fork_name = f"exp177_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    # Initialize decayed value store v from spine's value_counts
    vc = c.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        v = (vc / vc_sum) * INIT_MASS   # normalize to equilibrium mass
    else:
        v = np.ones(n_colors) * (INIT_MASS / n_colors)

    # Pre-build per-color cell lists for burst relocation
    color_cells: list[list[int]] = [[] for _ in range(n_colors)]
    for cell_idx, color in enumerate(base_cmap):
        color_cells[color].append(cell_idx)
    color_cells_arr = [np.array(lst, dtype=np.int32) for lst in color_cells]

    # Burst RNG (own, seeded — never touches chunk rng)
    burst_rng = np.random.default_rng(BURST_SEED_OFFSET + fork_seed)

    # Build set of burst step indices for fast lookup
    burst_step_set = set()
    for bstart, bend in BURST_WINDOWS:
        burst_step_set.update(range(bstart, bend))

    # Per-step storage
    expressed_arr = np.empty(N_STEPS, dtype=np.int32)   # argmax v
    true_pos_arr = np.empty(N_STEPS, dtype=np.int32)
    obs_arr = np.empty(N_STEPS, dtype=np.int32)
    # v snapshots at diagnostic points (burst onset, burst end, +1000, +2000)
    v_snapshots: list[dict] = []   # list of {t, label, v_copy}

    # Monitor: v trajectory snapshots every EVAL steps
    v_traj = []

    # Track active burst context
    current_burst_color: int | None = None
    current_burst_idx: int | None = None
    # Pre-burst favorite per burst (captured at onset)
    burst_preburst_fav: list[int | None] = [None, None, None]
    burst_onset_color: list[int | None] = [None, None, None]

    # Record A_hat at start for drift
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
            # Identify which burst (0-indexed)
            burst_idx_now: int | None = None
            for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                if bstart <= t < bend:
                    burst_idx_now = bi
                    break

            # On burst onset: record pre-burst favorite and set burst color
            if burst_idx_now is not None and t == BURST_WINDOWS[burst_idx_now][0]:
                pre_fav = int(np.argmax(v))
                burst_preburst_fav[burst_idx_now] = pre_fav
                # burst color = argmin of v at onset
                bc = int(np.argmin(v))
                burst_onset_color[burst_idx_now] = bc
                current_burst_color = bc
                current_burst_idx = burst_idx_now
                # Snapshot at onset
                v_snapshots.append({
                    "t": t, "label": f"onset_b{burst_idx_now}",
                    "v": v.copy(), "burst_color": bc, "pre_fav": pre_fav,
                })
            elif burst_idx_now is None:
                # Just exited a burst (or not in one)
                # On burst end: snapshot
                if current_burst_idx is not None:
                    # We exited the burst; check if this is the first step after
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

            # Also snapshot at +1000 and +2000 after each burst end
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
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Observe ---
            obs = int(base_cmap[c.true_pos])
            obs_arr[t] = obs
            true_pos_arr[t] = c.true_pos

            # --- Surprise (used for predictability_weight) ---
            p_o = float(A_hat[obs, :] @ c.qs)

            # --- Belief update: qs ∝ A_hat[obs,:] * qs ---
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
            # First decay v
            v *= LAMBDA
            # Compute predictability_weight
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            predictability_weight = math.exp(-h_predicted)
            v[obs] += predictability_weight
            # Also update c.value_counts (harmless; standard Exp 26 bookkeeping)
            c.value_counts[obs] += predictability_weight

            # --- Expressed preference ---
            expressed = int(np.argmax(v))
            expressed_arr[t] = expressed

            # --- Action / Move ---
            if in_burst and current_burst_color is not None:
                # Burst: relocate to uniformly random cell of burst color
                cells_of_bc = color_cells_arr[current_burst_color]
                if len(cells_of_bc) > 0:
                    c.true_pos = int(burst_rng.choice(cells_of_bc))
                # After relocation: set qs to uniform over burst-color cells
                # (captivity disorients dead reckoning — keeps Bayes update sane)
                qs_next = np.zeros(n_cells)
                qs_next[cells_of_bc] = 1.0 / len(cells_of_bc)
                c.qs = qs_next
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
    """Analyze one burst's flip and recovery.

    Returns dict with: flip (bool), flip_latency (int|None),
    recovery (bool), recovery_latency (int|None),
    confinement_rate_placeholder (None — computed separately).
    """
    bstart, bend = BURST_WINDOWS[burst_idx]
    bc = burst_onset_color[burst_idx]
    pre_fav = burst_preburst_fav[burst_idx]

    if bc is None or pre_fav is None:
        return {
            "flip": False, "flip_latency": None,
            "recovery": False, "recovery_latency": None,
        }

    # FLIP: >= 50% of the final 400 burst steps show expressed == burst color
    final_window_start = bend - P1_FLIP_FINAL_WINDOW
    final_window = expressed_arr[final_window_start:bend]
    flip_frac = float(np.mean(final_window == bc))
    flip = flip_frac >= P1_FLIP_FINAL_FRAC

    # Flip latency: first step t in [bstart, bend) where expressed == bc
    flip_latency: int | None = None
    for t in range(bstart, bend):
        if expressed_arr[t] == bc:
            flip_latency = t - bstart
            break

    # RECOVERY: pre-burst favorite restored within 2000 steps after burst end,
    # holding for >= 500 consecutive steps
    recovery = False
    recovery_latency: int | None = None
    recovery_end = min(bend + P1_RECOVERY_LAG, N_STEPS)

    t = bend
    while t < recovery_end:
        if expressed_arr[t] == pre_fav:
            # Check if it holds for 500 consecutive steps
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
# Monitor functions (rung 2)
# ---------------------------------------------------------------------------

def label_sample(target_step: int) -> str:
    """BURST / EXCLUDED / QUIET label for a mismatch sample whose target snapshot is at target_step."""
    for bstart, bend in BURST_WINDOWS:
        if bstart < target_step <= bend:
            return "BURST"
    for bstart, bend in BURST_WINDOWS:
        if bend < target_step <= bend + EXCLUDE_AFTER:
            return "EXCLUDED"
    return "QUIET"


def compute_monitor(v_traj: np.ndarray) -> list[dict]:
    """Mismatch samples: for k in [MON_W_SNAPS, len-2], predict snapshot k+1 by linear drift.

    v_traj[k] is v at step (k+1)*EVAL. Returns list of {target_step, mismatch, label}.
    """
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

def check_preconditions(
    run_results: list[dict],
    seeds: list[int],
    base_cmap: list,
) -> tuple[bool, list[str]]:
    """Check PC1 (drift), PC2 (pre-burst stability), PC3 (confinement)."""
    failures: list[str] = []

    # PC1: ahat_drift < 0.15
    for rr in run_results:
        seed = rr["fork_seed"]
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: seed={seed} ahat_drift={rr['ahat_drift']:.4f} "
                f">= {PC1_AHAT_DRIFT_MAX} (horizon-scaled, Exp 166 law)"
            )

    # PC2: pre-burst stability — expressed preference constant over 1000 steps before
    # each burst in >= 7/8 forks
    for bi in range(3):
        bstart, bend = BURST_WINDOWS[bi]
        win_start = max(0, bstart - PC2_STABILITY_WINDOW)
        stable_forks = 0
        for rr in run_results:
            exp_window = rr["expressed_arr"][win_start:bstart]
            if len(exp_window) > 0 and len(set(exp_window.tolist())) == 1:
                stable_forks += 1
        if stable_forks < PC2_MIN_STABLE_FORKS:
            failures.append(
                f"PC2 FAIL: burst {bi} — only {stable_forks}/{len(run_results)} forks "
                f"show constant preference in {PC2_STABILITY_WINDOW} steps before burst "
                f"(need >= {PC2_MIN_STABLE_FORKS})"
            )

    # PC3: confinement — >= 90% of burst steps land on burst-color cells
    for bi in range(3):
        bstart, bend = BURST_WINDOWS[bi]
        confinement_rates: list[float] = []
        for rr in run_results:
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


# ---------------------------------------------------------------------------
# Evaluate P2 / F2 (monitor AUROC)
# ---------------------------------------------------------------------------

def evaluate(run_results: list[dict]) -> dict:
    per_fork = []
    aurocs = []
    for rr in run_results:
        samples = compute_monitor(rr["v_traj"])
        burst_m = [s["mismatch"] for s in samples if s["label"] == "BURST"]
        quiet_m = [s["mismatch"] for s in samples if s["label"] == "QUIET"]
        a = auroc(burst_m, quiet_m)
        aurocs.append(a)
        per_fork.append({
            "fork_seed": rr["fork_seed"],
            "auroc": a,
            "n_burst": len(burst_m),
            "n_quiet": len(quiet_m),
            "mean_mismatch_burst": float(np.mean(burst_m)) if burst_m else None,
            "mean_mismatch_quiet": float(np.mean(quiet_m)) if quiet_m else None,
            "fork_passes": (not math.isnan(a)) and a >= P2_AUROC_MIN,
        })
    forks_pass = sum(1 for f in per_fork if f["fork_passes"])
    median_auroc = float(np.median([a for a in aurocs if not math.isnan(a)]))
    p2_pass = forks_pass >= P2_MIN_FORKS
    f2 = median_auroc <= F2_MEDIAN_MAX
    verdict_str = "POSITIVE" if p2_pass else ("NEGATIVE" if f2 else "MIXED")
    return {
        "per_fork": per_fork,
        "forks_pass": forks_pass,
        "median_auroc": median_auroc,
        "p2_pass": p2_pass,
        "f2": f2,
        "verdict_str": verdict_str,
    }


# ---------------------------------------------------------------------------
# Print burst table (smoke + full run)
# ---------------------------------------------------------------------------

def print_burst_table(run_results: list[dict]) -> None:
    """Print per-fork, per-burst summary: onset fav, burst color, flip, latencies."""
    header = (f"{'seed':>5} {'b':>2} {'pre_fav':>8} {'bc':>4} "
              f"{'flip?':>6} {'flip_frac':>10} {'flip_lat':>9} "
              f"{'recovery?':>10} {'rec_lat':>8}")
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


# ---------------------------------------------------------------------------
# JSON row writer
# ---------------------------------------------------------------------------

def write_json_rows(
    run_results: list[dict],
    ev: dict | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for rr in run_results:
            seed = rr["fork_seed"]
            for bi in range(3):
                ba = analyze_burst(
                    rr["expressed_arr"], bi,
                    rr["burst_preburst_fav"], rr["burst_onset_color"],
                )
                bstart, bend = BURST_WINDOWS[bi]
                # Confinement rate for this burst
                bc = rr["burst_onset_color"][bi]
                if bc is not None:
                    true_pos_burst = rr["true_pos_arr"][bstart:bend]
                    # Note: base_cmap is accessed via closure — not available here;
                    # confinement is computed in check_preconditions; leave as None
                row = {
                    "exp": 177,
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
                    "displaced": ba["flip"] and not ba["recovery"],
                    "ahat_drift": rr["ahat_drift"],
                }
                fh.write(json.dumps(row) + "\n")
        if ev is not None:
            # Per-fork AUROC rows
            for pf in ev["per_fork"]:
                auroc_row = {
                    "exp": 177,
                    "fork_seed": pf["fork_seed"],
                    "auroc": pf["auroc"],
                    "n_burst": pf["n_burst"],
                    "n_quiet": pf["n_quiet"],
                    "mean_mismatch_burst": pf["mean_mismatch_burst"],
                    "mean_mismatch_quiet": pf["mean_mismatch_quiet"],
                    "fork_passes": pf["fork_passes"],
                }
                fh.write(json.dumps(auroc_row) + "\n")
            summary = {
                "exp": 177,
                "phase": "summary",
                "forks_pass": ev["forks_pass"],
                "median_auroc": ev["median_auroc"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 177 — N4 rung 2, the IDENTITY MONITOR (read-only)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: seed [178] only, full 15000 steps, no verdict")
    args = parser.parse_args()

    seeds = [178] if args.smoke else SEEDS

    print("=" * 80)
    print("Exp 177 — N4 rung 2, the IDENTITY MONITOR (read-only)")
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
    print(f"Seeds: {seeds}")
    print()

    # Per-color cell counts (sanity)
    per_color_cells = [sum(1 for c in base_cmap if c == col) for col in range(n_colors)]
    print(f"Cells per color: {per_color_cells}")
    print()

    # Standing preference of mirro's spine
    vc = mirro.value_counts.copy()
    vc_sum = vc.sum()
    if vc_sum > 0:
        spine_fav = int(np.argmax(vc))
        print(f"Mirro spine value_counts: {vc}")
        print(f"Spine standing favorite: color {spine_fav}")
    print()

    # --- Run forks ---
    run_results: list[dict] = []
    for seed in seeds:
        print(f"  Running seed={seed} ...", flush=True)
        rr = run_fork(mirro, seed, base_cmap, n_colors)
        run_results.append(rr)
        print(f"    drift={rr['ahat_drift']:.5f}  "
              f"standing_fav={rr['burst_preburst_fav'][0]}  "
              f"burst_colors={rr['burst_onset_color']}")

    print()

    # --- Burst table (diagnostic) ---
    print("Burst analysis table (ungated diagnostic):")
    print_burst_table(run_results)
    print()

    # --- Preconditions ---
    pc_pass, pc_failures = check_preconditions(run_results, seeds, base_cmap)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
    else:
        print("Preconditions: all PASS")
        print()

    # --- V-snapshot diagnostics ---
    print("Value trajectory snapshots (v[fav] vs v[burst_color]):")
    for rr in run_results:
        seed = rr["fork_seed"]
        for snap in rr["v_snapshots"]:
            bc = snap.get("burst_color")
            pre_fav = snap.get("pre_fav")
            if bc is None or pre_fav is None:
                continue
            v = snap["v"]
            print(f"  seed={seed} t={snap['t']:>6} [{snap['label']:<24}] "
                  f"v[fav={pre_fav}]={v[pre_fav]:>8.3f}  "
                  f"v[bc={bc}]={v[bc]:>8.3f}  "
                  f"expressed={int(np.argmax(v))}")

    print()

    if args.smoke:
        # Smoke: print monitor table for seed 178 (first 4 and last 4 BURST samples of burst 0)
        rr0 = run_results[0]
        samples0 = compute_monitor(rr0["v_traj"])
        burst0_samples = [s for s in samples0 if s["label"] == "BURST"
                          and 6000 < s["target_step"] <= 6800]
        quiet0_samples = [s for s in samples0 if s["label"] == "QUIET"]
        print(f"Monitor smoke (seed={rr0['fork_seed']}): "
              f"total samples={len(samples0)}, "
              f"BURST={len([s for s in samples0 if s['label']=='BURST'])}, "
              f"EXCLUDED={len([s for s in samples0 if s['label']=='EXCLUDED'])}, "
              f"QUIET={len(quiet0_samples)}")
        if burst0_samples:
            first4 = burst0_samples[:4]
            last4 = burst0_samples[-4:]
            print("  Burst-0 first 4 BURST samples (mismatch):")
            for s in first4:
                print(f"    target_step={s['target_step']} mismatch={s['mismatch']:.5f}")
            print("  Burst-0 last 4 BURST samples (mismatch):")
            for s in last4:
                print(f"    target_step={s['target_step']} mismatch={s['mismatch']:.5f}")
        a_smoke = auroc(
            [s["mismatch"] for s in samples0 if s["label"] == "BURST"],
            [s["mismatch"] for s in samples0 if s["label"] == "QUIET"],
        )
        print(f"  Smoke AUROC (burst vs quiet): {a_smoke:.4f}")
        print()
        print("SMOKE ONLY — no verdict")
        return

    # --- Verdict (full run only) ---
    if not pc_pass:
        print("PRECONDITION FAILED — writing rows, halting before verdict.")
        out_rows = Path(__file__).parent / "outputs" / "exp177_rows.json"
        write_json_rows(run_results, ev=None, out_path=out_rows)
        print(f"Rows written to {out_rows}")
        return

    ev = evaluate(run_results)

    # --- Monitor table ---
    print("=" * 80)
    print("MONITOR TABLE — per-fork AUROC and mismatch by class")
    print("=" * 80)
    print()
    mon_header = (f"{'seed':>5} {'AUROC':>7} {'n_burst':>8} {'n_quiet':>8} "
                  f"{'mean_m_burst':>13} {'mean_m_quiet':>13} {'pass?':>6}")
    print(mon_header)
    print("-" * len(mon_header))
    for pf in ev["per_fork"]:
        mmb = f"{pf['mean_mismatch_burst']:.5f}" if pf["mean_mismatch_burst"] is not None else "—"
        mmq = f"{pf['mean_mismatch_quiet']:.5f}" if pf["mean_mismatch_quiet"] is not None else "—"
        auroc_s = f"{pf['auroc']:.4f}" if not math.isnan(pf["auroc"]) else "nan"
        print(f"{pf['fork_seed']:>5} {auroc_s:>7} {pf['n_burst']:>8} {pf['n_quiet']:>8} "
              f"{mmb:>13} {mmq:>13} {'PASS' if pf['fork_passes'] else 'fail':>6}")
    print()

    # Burst-0 adaptation diagnostic: first/last BURST samples of burst 0
    print("Burst-0 adaptation diagnostic (first 4 / last 4 BURST samples of burst 0):")
    for rr in run_results:
        seed = rr["fork_seed"]
        samples = compute_monitor(rr["v_traj"])
        b0_burst = [s for s in samples if s["label"] == "BURST"
                    and 6000 < s["target_step"] <= 6800]
        if b0_burst:
            first4 = b0_burst[:4]
            last4 = b0_burst[-4:]
            firsts = " ".join(f"{s['mismatch']:.4f}" for s in first4)
            lasts = " ".join(f"{s['mismatch']:.4f}" for s in last4)
            print(f"  seed={seed} burst-0 first4=[{firsts}] last4=[{lasts}]")
    print()

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    for pf in ev["per_fork"]:
        auroc_s = f"{pf['auroc']:.4f}" if not math.isnan(pf["auroc"]) else "nan"
        status = "PASS" if pf["fork_passes"] else "fail"
        print(f"  seed={pf['fork_seed']} AUROC={auroc_s} "
              f"n_burst={pf['n_burst']} n_quiet={pf['n_quiet']} => {status} (>={P2_AUROC_MIN})")

    print()
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED")
    print(f"P2 (monitor sensitivity): forks with AUROC >= {P2_AUROC_MIN}: "
          f"{ev['forks_pass']}/8 (need >= {P2_MIN_FORKS}); "
          f"median AUROC = {ev['median_auroc']:.3f} "
          f"(F2 if <= {F2_MEDIAN_MAX}) => {p2_status}")
    print()

    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P2 (monitor sensitivity): forks with AUROC >= {P2_AUROC_MIN}: "
          f"{ev['forks_pass']}/8 (need >= {P2_MIN_FORKS}); "
          f"median AUROC = {ev['median_auroc']:.3f} "
          f"(F2 if <= {F2_MEDIAN_MAX}) => {p2_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        gate_text = "MONITOR IS SENSITIVE — N4 mismatch separates identity perturbation from quiet; rung 3 (commitment regulation) is armed"
    elif ev["verdict_str"] == "NEGATIVE":
        gate_text = "NO METACOGNITIVE SENSITIVITY OVER IDENTITY — the rung-2 kill"
    else:
        gate_text = "between bands"
    print(f"VERDICT: {ev['verdict_str']}  ({gate_text})")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows = Path(__file__).parent / "outputs" / "exp177_rows.json"
    write_json_rows(run_results, ev=ev, out_path=out_rows)
    print(f"Rows written to {out_rows}")

    # --- Write verdict JSON ---
    arms = {
        "P2_monitor_sensitivity": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"forks with AUROC>={P2_AUROC_MIN}: {ev['forks_pass']}/8 "
                f"(need >={P2_MIN_FORKS}); "
                f"median AUROC={ev['median_auroc']:.3f} "
                f"(F2 if <={F2_MEDIAN_MAX})"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp177_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp177",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=not pc_pass,
        notes=(
            "N4 rung 2 identity monitor, read-only, on the Exp 176 regime. "
            "Linear-drift self-prediction (W=1000 steps, horizon 100 steps), "
            "L2 mismatch, AUROC burst-vs-quiet with a 1000-step post-burst exclusion band. "
            "Monitor FORM provided; value content self-formed. Forks of mirro only."
        ),
    )
    print(f"Verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
