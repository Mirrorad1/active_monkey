"""Exp 79 — the early-tax test: is the clumped color's deficit an early-life imprint?

Exp 78 corrected Exp 77's reading: clumping suppresses a color's value share NOT through
soft end-state gates (fresh-creature clump columns are the sharpest in the grid) but —
hypothesized — through an EARLY LOCALIZATION TAX: a young creature inside the look-alike
block cannot tell which cell it occupies, its gate weights there are suppressed exactly
while value totals are small, and the never-decaying ledger makes the early deficit
permanent. This is the direct test: per-window accrual trajectories on FRESH seeds
(500+s, never run — the out-of-sample rule), CLUMPED vs SCATTERED, matched per seed.

Protocol: per (arm, seed): newborn (separate root) at cell 12, live in 12 windows of 250
steps; record value_counts after each window; window share for color 2 = window accrual
of color 2 / total window accrual; D(w) = scattered share2(w) - clumped share2(w) per
matched seed. Diagnostic: localize_bits at each window end per arm (clumped should start
high and converge).

Predeclared predictions:
  P1 (early concentration): mean D over windows 1-4 > mean D over windows 9-12, in
     >= 4/5 seeds.
  P2 (late fairness): |mean D over windows 9-12| <= 0.04 in >= 4/5 seeds (the tax
     vanishes once localized).
  P3 (imprint accounting): the color-2 COUNTS gap between arms (scattered minus clumped
     cumulative color-2 accrual) at step 1000 is >= 50 percent of the same gap at step
     3000, in >= 3/5 seeds — at least half the final deficit is in place by a third of
     life. (Sign guard: a seed where the step-3000 gap is <= 0 is excluded from P3 and
     reported — no deficit to account for.)
Falsifiers:
  F1 = P1 fails -> the early-tax mechanism is wrong; Exp 78's correction needs
     re-correction (the clumping effect has yet another cause).
  F2 = P2 fails -> an ONGOING tax exists beyond early life — partially rehabilitating a
     structural reading; report the late-window D values.
  F3 = P1+P2 hold but P3 fails -> the early tax is real but does not account for the end
     gap; a second mechanism contributes.
Provided priors declared: the two layouts (Exp 78's, verbatim), raising length, window
size, start cell, fresh birth seeds 500+s. Newborns are separate roots; the spine and
vela are untouched (not loaded).
"""
from __future__ import annotations

import collections
import sys

import numpy as np

# Add repo root to path so active_loop is importable when run directly
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WINDOWS = 12
CHUNK = 250
SEEDS = list(range(5))          # 0..4
BIRTH_BASE = 500
START = 12
N_COLORS = 3
ROWS, COLS = 5, 5
N_CELLS = ROWS * COLS           # 25

# ---------------------------------------------------------------------------
# Layout construction helpers (verbatim from Exp 78)
# ---------------------------------------------------------------------------

def build_clumped_cmap() -> list:
    """Color 2 at rows 2-4 x cols 0-2; remaining cells alternate 0,1 in index order."""
    color2_cells = set()
    for r in range(2, 5):          # rows 2,3,4
        for c in range(0, 3):      # cols 0,1,2
            color2_cells.add(r * COLS + c)
    # cells 10,11,12,15,16,17,20,21,22

    cmap = [None] * N_CELLS
    for cell in color2_cells:
        cmap[cell] = 2

    alt = 0
    for cell in range(N_CELLS):
        if cmap[cell] is None:
            cmap[cell] = alt
            alt = 1 - alt          # toggle between 0 and 1

    return cmap


def build_scattered_cmap() -> list:
    """Color 2 at (r+c) even cells; remaining cells alternate 0,1 in index order."""
    color2_cells = set()
    for r in range(ROWS):
        for c in range(COLS):
            if (r + c) % 2 == 0:
                color2_cells.add(r * COLS + c)
    # cells 0,2,4,6,8,10,12,14,16,18,20,22,24 — 13 cells with (r+c) even overall
    # But docstring says cells 0,2,4,6,8,12,16,20,24 (nine cells)
    # The docstring specifies exactly: cells 0,2,4,6,8,12,16,20,24 — let's use that.
    color2_cells = {0, 2, 4, 6, 8, 12, 16, 20, 24}

    cmap = [None] * N_CELLS
    for cell in color2_cells:
        cmap[cell] = 2

    alt = 0
    for cell in range(N_CELLS):
        if cmap[cell] is None:
            cmap[cell] = alt
            alt = 1 - alt

    return cmap


def verify_counts(cmap: list, label: str) -> None:
    counts = collections.Counter(cmap)
    assert counts == {0: 8, 1: 8, 2: 9}, (
        f"{label}: expected {{0:8, 1:8, 2:9}}, got {dict(counts)}"
    )


# ---------------------------------------------------------------------------
# Window share helper
# ---------------------------------------------------------------------------

def window_share2(prev_counts: np.ndarray, curr_counts: np.ndarray) -> float:
    """Share of color-2 accrual in a single window (curr - prev)."""
    delta = curr_counts - prev_counts
    total = delta.sum()
    if total <= 0:
        return 0.0
    return float(delta[2] / total)


# ---------------------------------------------------------------------------
# Check
# ---------------------------------------------------------------------------

def check(
    # shape: [seed][window] for each arm
    share2_clumped: list,   # list of 5 lists of 12 floats
    share2_scattered: list,
    # cumulative color-2 accrual at window-4 and window-12 ends, per seed, per arm
    cum2_clumped_w4: list,
    cum2_clumped_w12: list,
    cum2_scattered_w4: list,
    cum2_scattered_w12: list,
) -> str:
    """Evaluate P1, P2, P3 and return verdict string."""

    n_seeds = len(SEEDS)

    # Per-seed early and late D
    early_D = []
    late_D = []
    for s in range(n_seeds):
        D = [share2_scattered[s][w] - share2_clumped[s][w] for w in range(WINDOWS)]
        early_D.append(float(np.mean(D[0:4])))
        late_D.append(float(np.mean(D[8:12])))

    # P1: early_D > late_D in >= 4/5 seeds
    p1_wins = sum(1 for s in range(n_seeds) if early_D[s] > late_D[s])
    p1_pass = p1_wins >= 4

    # P2: |late_D| <= 0.04 in >= 4/5 seeds
    p2_wins = sum(1 for s in range(n_seeds) if abs(late_D[s]) <= 0.04)
    p2_pass = p2_wins >= 4

    # P3: imprint accounting
    # gap1000 = scattered_cum2_w4 - clumped_cum2_w4  (end of window 4 = step 1000)
    # gap3000 = scattered_cum2_w12 - clumped_cum2_w12 (end of window 12 = step 3000)
    p3_eligible = []
    p3_excluded = []
    for s in range(n_seeds):
        gap1000 = cum2_scattered_w4[s] - cum2_clumped_w4[s]
        gap3000 = cum2_scattered_w12[s] - cum2_clumped_w12[s]
        if gap3000 <= 0:
            p3_excluded.append((s, gap3000))
        else:
            ratio = gap1000 / gap3000
            p3_eligible.append((s, gap1000, gap3000, ratio))

    print("\n--- Prediction checks ---")

    print(f"\nPer-seed early_D (mean windows 1-4) and late_D (mean windows 9-12):")
    for s in range(n_seeds):
        flag = "early>late" if early_D[s] > late_D[s] else "early<=late"
        print(f"  seed {s}: early_D={early_D[s]:.4f}  late_D={late_D[s]:.4f}  [{flag}]")

    print(f"\nP1 (early_D > late_D): {p1_wins}/5 seeds  -> {'PASS' if p1_pass else 'FAIL'}")
    print(f"P2 (|late_D| <= 0.04): {p2_wins}/5 seeds  -> {'PASS' if p2_pass else 'FAIL'}")

    if p3_excluded:
        print(f"\nP3 sign-guard exclusions (gap3000 <= 0): "
              f"{[(f'seed {s}', f'gap3000={g:.2f}') for s,g in p3_excluded]}")

    if len(p3_eligible) < 3:
        p3_result = "INVALID"
        print(f"P3: only {len(p3_eligible)} seeds have gap3000 > 0 (need >= 3) -> P3 INVALID")
    else:
        p3_wins = sum(1 for (s, g1, g3, r) in p3_eligible if r >= 0.5)
        p3_pass = p3_wins >= 3
        p3_result = "PASS" if p3_pass else "FAIL"
        print(f"\nP3 imprint accounting (gap1000 >= 0.5 * gap3000) among {len(p3_eligible)} eligible seeds:")
        for (s, g1, g3, r) in p3_eligible:
            print(f"  seed {s}: gap1000={g1:.2f}  gap3000={g3:.2f}  ratio={r:.3f}  "
                  f"[{'>=0.5' if r >= 0.5 else '<0.5'}]")
        print(f"P3: {p3_wins}/{len(p3_eligible)} eligible seeds  -> {p3_result}")

    # Verdict
    p3_note = " (P3-INVALID: too few eligible seeds)" if p3_result == "INVALID" else ""

    if p1_pass and p2_pass and (p3_result in ("PASS", "INVALID")):
        verdict = f"EXP79: EARLY TAX CONFIRMED{p3_note}"
    elif not p1_pass:
        verdict = "EXP79: F1 — mechanism wrong (re-correction needed)"
    elif not p2_pass:
        late_vals = [f"{late_D[s]:.4f}" for s in range(n_seeds)]
        verdict = f"EXP79: F2 — ongoing tax [late D values: {', '.join(late_vals)}]"
    else:
        # P1+P2 hold but P3 fails
        verdict = f"EXP79: F3 — early tax insufficient{p3_note}"

    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Build and validate layouts
    clumped_cmap = build_clumped_cmap()
    scattered_cmap = build_scattered_cmap()

    verify_counts(clumped_cmap, "CLUMPED")
    verify_counts(scattered_cmap, "SCATTERED")

    world_clumped = World(rows=ROWS, cols=COLS, cmap=clumped_cmap, n_colors=N_COLORS)
    world_scattered = World(rows=ROWS, cols=COLS, cmap=scattered_cmap, n_colors=N_COLORS)

    # Collect per-arm, per-seed: window shares, cumulative accrual, localize_bits
    # share2[arm][seed][window]  (0-indexed windows)
    # loc_bits[arm][seed][window]
    share2: dict[str, list] = {"clumped": [], "scattered": []}
    loc_bits: dict[str, list] = {"clumped": [], "scattered": []}
    cum2_w4: dict[str, list] = {"clumped": [], "scattered": []}
    cum2_w12: dict[str, list] = {"clumped": [], "scattered": []}

    for arm_name, world in [("clumped", world_clumped), ("scattered", world_scattered)]:
        print(f"\n=== Arm: {arm_name.upper()} ===")
        for seed_idx in SEEDS:
            birth_seed = BIRTH_BASE + seed_idx
            c = Creature.birth(
                f"exp79-{arm_name}-s{seed_idx}",
                world=world,
                seed=birth_seed,
            )
            c.true_pos = START

            prev_counts = c.value_counts.copy()
            seed_shares = []
            seed_locs = []
            _cum2_w4 = None
            _cum2_w12 = None

            for w in range(WINDOWS):
                c.live(CHUNK)
                curr_counts = c.value_counts.copy()
                ws2 = window_share2(prev_counts, curr_counts)
                seed_shares.append(ws2)
                seed_locs.append(c.localize_bits())
                prev_counts = curr_counts
                if w == 3:   # window 4 end = step 1000
                    _cum2_w4 = float(curr_counts[2])
                if w == 11:  # window 12 end = step 3000
                    _cum2_w12 = float(curr_counts[2])

            share2[arm_name].append(seed_shares)
            loc_bits[arm_name].append(seed_locs)
            cum2_w4[arm_name].append(_cum2_w4)
            cum2_w12[arm_name].append(_cum2_w12)

    # ---------------------------------------------------------------------------
    # Table 1: per (arm, seed) — 12 window share2 values
    # ---------------------------------------------------------------------------
    print("\n=== Table 1: window share2 for color 2 (per arm, per seed) ===")
    win_header = "  ".join(f"w{w+1:02d}" for w in range(WINDOWS))
    print(f"{'arm':<12} {'seed':>4}  {win_header}")
    print("-" * (16 + WINDOWS * 7))
    for arm_name in ("clumped", "scattered"):
        for seed_idx in SEEDS:
            vals = "  ".join(f"{share2[arm_name][seed_idx][w]:.3f}" for w in range(WINDOWS))
            print(f"{arm_name:<12} {seed_idx:>4}  {vals}")
        print()

    # ---------------------------------------------------------------------------
    # Table 2: per seed — D(w) = scattered share2(w) - clumped share2(w)
    # ---------------------------------------------------------------------------
    print("=== Table 2: D(w) = scattered_share2(w) - clumped_share2(w), per seed ===")
    print(f"{'seed':>4}  {win_header}")
    print("-" * (6 + WINDOWS * 7))
    for seed_idx in SEEDS:
        D_row = [share2["scattered"][seed_idx][w] - share2["clumped"][seed_idx][w]
                 for w in range(WINDOWS)]
        vals = "  ".join(f"{d:.3f}" for d in D_row)
        print(f"{seed_idx:>4}  {vals}")

    # ---------------------------------------------------------------------------
    # Table 3: localization diagnostic — mean localize_bits per window per arm
    # ---------------------------------------------------------------------------
    print("\n=== Table 3: mean localize_bits per window (across seeds), per arm ===")
    print(f"{'arm':<12}  {win_header}")
    print("-" * (14 + WINDOWS * 7))
    for arm_name in ("clumped", "scattered"):
        mean_locs = [float(np.mean([loc_bits[arm_name][s][w] for s in SEEDS]))
                     for w in range(WINDOWS)]
        vals = "  ".join(f"{v:.3f}" for v in mean_locs)
        print(f"{arm_name:<12}  {vals}")

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    verdict = check(
        share2_clumped=share2["clumped"],
        share2_scattered=share2["scattered"],
        cum2_clumped_w4=cum2_w4["clumped"],
        cum2_clumped_w12=cum2_w12["clumped"],
        cum2_scattered_w4=cum2_w4["scattered"],
        cum2_scattered_w12=cum2_w12["scattered"],
    )
    print(f"\n{verdict}")


if __name__ == "__main__":
    main()
