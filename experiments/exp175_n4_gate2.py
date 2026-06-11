"""Exp 175 — N4 rung 1, attempt 2: the DISPLACEMENT GATE. A constructible
transient-pressure regime must persistently degrade the N4-less baseline's
identity — flip its expressed preference AND leave it unrecovered — the
anti-regress precondition for the identity layer (card:
loop/directions/identity-n4.md; re-aimed per Exp 174's portrait).

Exp 174 (attempt 1, NO VERDICT by PC2) showed that at lambda=0.999 the
baseline has no stable favorite even unpressured, and that captivity bursts
REWRITE identity rather than whipsaw it (flips 24/24, recoveries ~3/24).
Attempt 2 declares a slower decay from the window-theorem band —
lambda = 0.9997 (half-life ~2310 steps, equilibrium mass 1/(1-lambda)
~ 3333) — so a stable favorite exists to be displaced, and re-aims the
gate at DISPLACEMENT: the burst must flip the favorite and the baseline
must FAIL to recover. Recovery becomes rung 3's deliverable (the N4
controller's job), not the world's.

Design: identical to Exp 174 except lambda and seeds. Forks of mirro only
(spine never saved), FRESH seeds 162-169, one 12000-step session per fork.
Value system: v[c] decays by lambda each step, then v[obs] +=
predictability_weight (Exp 26 rule on the decayed store; v initialized
from the spine's normalized value_counts scaled to mass 1/(1-lambda) so
the standing preference is inherited at the decay equilibrium scale —
declared). Expressed preference at t = argmax v. Bursts at steps
3000-3799, 6000-6799, 9000-9799; burst color = argmin of v at burst onset
(per fork, per burst). During a burst the move step is replaced by
relocation to a uniformly random cell of the burst color (own rng
160_000 + fork_seed); outside bursts the standard random walk runs.

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 ahat_drift < 0.15 (horizon-scaled, Exp 166 law);
PC2 pre-burst stability: the expressed preference is constant over the
1000 steps before each burst in >= 7/8 forks (the thing being displaced
must exist — RETAINED from attempt 1; this is the conjunct lambda=0.999
failed); PC3 confinement: >= 90% of burst steps land on burst-color cells.

Predeclared property and falsifier:
  P1' (the displacement regime exists): in >= 7/8 forks, for >= 2 of the
     3 bursts, BOTH: (i) FLIP — the expressed preference equals the burst
     color for >= 50% of the burst's final 400 steps; and (ii) RECOVERY
     FAILURE — the pre-burst favorite is NOT restored-and-held (>= 500
     consecutive steps) within 2000 steps after burst end. A burst
     satisfying both is a DISPLACEMENT.
     FALSIFIER F1': displacement (per the >= 2/3-bursts rule) in <= 4/8
     forks — either the slower decay resists flips or the baseline
     recovers on its own; no constructible persistent-degradation regime
     at this richness (the Exp 173-class verdict; a real result).
  VERDICT: POSITIVE (GATE PASSES — the regime persistently degrades the
  layerless agent; rungs 2-3 are armed, recovery is the N4 controller's
  deliverable) iff P1'. NEGATIVE iff F1'. Otherwise MIXED. "Not a
  falsifier" never counts toward POSITIVE.

Ungated diagnostics: identical to Exp 174 — per-fork value trajectories
at burst boundaries; flip latency and (where it occurs) recovery latency
per burst; inherited standing favorite per fork; drift; confinement.
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

SEEDS = list(range(162, 170))   # 8 fresh forks (174 used 154-161)

N_STEPS = 12_000
CHUNK_SIZE = 100
N_CHUNKS = N_STEPS // CHUNK_SIZE   # 120

# Value decay
LAMBDA = 0.9997          # per-step decay; half-life ~2310 steps; equilibrium mass = 1/(1-lambda) ~ 3333
INIT_MASS = 1.0 / (1.0 - LAMBDA)  # ~ 3333.3

# Burst windows: [start, end) — inclusive start, exclusive end
BURST_WINDOWS = [
    (3000, 3800),
    (6000, 6800),
    (9000, 9800),
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
    fork_name = f"exp175_s{fork_seed}"
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
# Evaluate P1' / F1'
# ---------------------------------------------------------------------------

def evaluate(run_results: list[dict]) -> dict:
    """Evaluate P1' and F1' on run results."""
    forks_displaced = 0
    per_fork_details: list[dict] = []

    for rr in run_results:
        seed = rr["fork_seed"]
        expressed_arr = rr["expressed_arr"]
        burst_preburst_fav = rr["burst_preburst_fav"]
        burst_onset_color = rr["burst_onset_color"]

        burst_results: list[dict] = []
        bursts_displaced = 0

        for bi in range(3):
            ba = analyze_burst(expressed_arr, bi, burst_preburst_fav, burst_onset_color)
            burst_results.append(ba)
            if ba["flip"] and not ba["recovery"]:
                bursts_displaced += 1

        fork_passes = bursts_displaced >= P1_MIN_BURSTS_PER_FORK
        if fork_passes:
            forks_displaced += 1

        per_fork_details.append({
            "fork_seed": seed,
            "burst_results": burst_results,
            "bursts_displaced": bursts_displaced,
            "fork_passes": fork_passes,
            "standing_fav": rr["burst_preburst_fav"][0],
        })

    p1_pass = forks_displaced >= P1_MIN_FORKS
    f1 = forks_displaced <= F1_MAX_FORKS

    if p1_pass:
        verdict_str = "POSITIVE"
    elif f1:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    return {
        "forks_displaced": forks_displaced,
        "p1_pass": p1_pass,
        "f1": f1,
        "verdict_str": verdict_str,
        "per_fork_details": per_fork_details,
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
                    "exp": 175,
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
            summary = {
                "exp": 175,
                "phase": "summary",
                "forks_displaced": ev["forks_displaced"],
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 175 — N4 rung 1 attempt 2, the DISPLACEMENT GATE")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke run: seed [162] only, full 12000 steps, no verdict")
    args = parser.parse_args()

    seeds = [162] if args.smoke else SEEDS

    print("=" * 80)
    print("Exp 175 — N4 rung 1 attempt 2, the DISPLACEMENT GATE")
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

    # --- Burst table ---
    print("Burst analysis table:")
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
        print("SMOKE ONLY — no verdict")
        return

    # --- Verdict (full run only) ---
    if not pc_pass:
        print("PRECONDITION FAILED — writing rows, halting before verdict.")
        out_rows = Path(__file__).parent / "outputs" / "exp175_rows.json"
        write_json_rows(run_results, ev=None, out_path=out_rows)
        print(f"Rows written to {out_rows}")
        return

    ev = evaluate(run_results)

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    for fd in ev["per_fork_details"]:
        bsums = [
            f"b{bi}:{'DISPLACED' if br['flip'] and not br['recovery'] else ('FLIP+REC' if br['flip'] and br['recovery'] else 'none')}"
            for bi, br in enumerate(fd["burst_results"])
        ]
        status = "PASS" if fd["fork_passes"] else "fail"
        print(f"  seed={fd['fork_seed']} bursts_displaced={fd['bursts_displaced']}/3 "
              f"[{', '.join(bsums)}] => {status}")

    print()
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1'" if ev["f1"] else "MIXED")
    print(f"P1' (displacement regime exists): forks_displaced={ev['forks_displaced']}/8 "
          f"(need >={P1_MIN_FORKS} for PASS, <={F1_MAX_FORKS} for F1') => {p1_status}")
    print()

    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1' (displacement regime exists, >= {P1_MIN_FORKS}/8 forks, >= {P1_MIN_BURSTS_PER_FORK}/3 bursts): "
          f"{p1_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        gate_text = "GATE PASSES — the regime persistently degrades the layerless agent; recovery is rung 3's deliverable"
    elif ev["verdict_str"] == "NEGATIVE":
        gate_text = ("GATE FAILS — no constructible persistent-degradation regime at this richness (Exp 173-class result)")
    else:
        gate_text = "GATE INCONCLUSIVE — between PASS and FALSIFIER bands"
    print(f"VERDICT: {ev['verdict_str']}  ({gate_text})")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows = Path(__file__).parent / "outputs" / "exp175_rows.json"
    write_json_rows(run_results, ev=ev, out_path=out_rows)
    print(f"Rows written to {out_rows}")

    # --- Write verdict JSON ---
    arms = {
        "P1prime_displacement_regime_exists": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"forks_displaced={ev['forks_displaced']}/8 "
                f"(need >={P1_MIN_FORKS}; F1': <={F1_MAX_FORKS}); "
                f"flip AND recovery-failure in >= {P1_MIN_BURSTS_PER_FORK}/3 bursts per qualifying fork"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp175_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp175",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=not pc_pass,
        notes=(
            "N4 rung 1 attempt 2, displacement gate. LAMBDA=0.9997 (declared, window-theorem band); "
            "burst relocation to burst-color cells; burst color = argmin(v) at onset; "
            "displacement = flip + recovery-failure. Forks of mirro only."
        ),
    )
    print(f"Verdict written to {verdict_path}")


if __name__ == "__main__":
    main()
