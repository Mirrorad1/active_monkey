"""Exp 57 — rung-5 cascade on Exp 56's deviation: 'perceptual rigidity grows with age'.

Candidate: under the 500-step drift schedule, mirro@6700 never re-reached map accuracy
0.92 in any segment (0/8). Cascade:
(a) REPRODUCTION: 3 forks of the committed pre-epoch snapshot (git 0467c26, age 6700,
    hash f3edac2a) live the identical schedule with fresh per-step action seeds (bases
    81/82/83). Reproduces if EACH fork fails the 0.92 target in >= 7/8 segments.
(b) DEFLATIONARY SWEEP:
    (1) counts-to-flip analytics: a recolored cell's argmax flips only after new-color
        soft-counts exceed the entrenched column mass; ~20 visits/cell/segment. Print
        mirro@6700's per-cell column masses (mean/min/max). If mean >> 20, failure is
        arithmetic necessity of non-decaying counts.
    (2) newborn control (seed 85, base 84): birth in the segment-0 world, 1000-step
        declared settling, then the same 8-segment schedule. If rigidity is
        count-driven, the newborn recovers in EARLY segments and rigidifies in LATE
        ones; predeclared: Spearman rho(segment index, lag with >500 coded as 501)
        > 0.6, with >= 2 early-segment recoveries.
Predeclared verdict: candidate DIES as novelty (classified: lawful consequence of
non-decaying Dirichlet counts -- the Exp 48 value-inertia law generalized to
perception) if (a) reproduces AND both (b) checks confirm. It SURVIVES as
NOVELTY-CANDIDATE only if (a) reproduces but (b) fails to explain it. If (a) does not
reproduce, the deviation was trajectory noise -- died-and-why logged.
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from active_loop.creature import Creature, World

ROWS, COLS, N_COLORS = 5, 5, 3
CORNER_CYCLE = [(0, 0), (0, 2), (2, 2), (2, 0)]
PRE_EPOCH_COMMIT = "db35582"  # age 6700, hash f3edac2a


def build_cmap(corner):
    cmap = [(i % 2) for i in range(ROWS * COLS)]
    r0, c0 = corner
    for dr in range(3):
        for dc in range(3):
            cmap[(r0 + dr) * COLS + (c0 + dc)] = 2
    return cmap


def run_segment(c, seg_idx, base_seed, global_step_offset):
    """Run 500 steps; return (lag, end_acc). lag=501 if 0.92 never reached."""
    corner = CORNER_CYCLE[seg_idx % 4]
    cmap = build_cmap(corner)
    c.world = World(rows=ROWS, cols=COLS, cmap=cmap, n_colors=N_COLORS)
    lag = None
    for step_i in range(500):
        gseed = base_seed * 1000003 + global_step_offset + step_i
        c.live(1, seed=gseed)
        if (step_i + 1) % 25 == 0 and lag is None:
            if c.map_accuracy() >= 0.92:
                lag = step_i + 1
    end_acc = c.map_accuracy()
    return (lag if lag is not None else 501), end_acc


# ── 1. Materialize pre-epoch snapshot ──────────────────────────────────────────
print("=== EXP 57: RIGIDITY CASCADE ===")
t0 = time.time()

tmpdir = tempfile.mkdtemp(prefix="exp57_mirro_")
files = ["arrays.npz", "manifest.json", "BIOGRAPHY.jsonl"]
for fname in files:
    git_path = f"{PRE_EPOCH_COMMIT}:creature/state/mirro/{fname}"
    out_path = Path(tmpdir) / fname
    if fname.endswith(".npz"):
        result = subprocess.run(
            ["git", "show", git_path],
            capture_output=True,
            cwd=str(Path(__file__).parent.parent),
        )
        out_path.write_bytes(result.stdout)
    else:
        result = subprocess.run(
            ["git", "show", git_path],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent),
        )
        out_path.write_text(result.stdout)

subject0 = Creature.load(tmpdir)
subject0._state_dir = None  # unbound — safe
print(f"subject0: age={subject0.age_steps} hash16={subject0._state_hash()[:16]}")

# Column masses: pA column sums (per-cell accumulated evidence)
col_masses = subject0.pA.sum(axis=0)  # shape (n_cells,)
mean_mass = float(col_masses.mean())
min_mass = float(col_masses.min())
max_mass = float(col_masses.max())
print(f"pA_column_masses: mean={mean_mass:.2f} min={min_mass:.2f} max={max_mass:.2f}")
print(f"flip_threshold: ~20 visits/segment (500 steps / 25 cells); "
      f"mass_vs_threshold ratio={mean_mass/20:.1f}x")

# ── 2. Reproduction: 3 forks ───────────────────────────────────────────────────
print("\n=== (a) REPRODUCTION ===")
fork_results = []
for base in (81, 82, 83):
    fork = subject0.fork(f"casc_{base}")
    fork._state_dir = None
    segs_failed = 0
    for seg in range(8):
        offset = (base - 81) * 4000 + seg * 500
        lag, end_acc = run_segment(fork, seg, base, offset)
        failed = lag >= 501
        if failed:
            segs_failed += 1
        print(f"  fork_{base} seg{seg} lag={lag} end_acc={end_acc:.4f} "
              f"{'FAIL' if failed else 'ok'}")
    fork_results.append(segs_failed)
    print(f"  fork_{base} segments_failed={segs_failed}/8")

reproduced = all(sf >= 7 for sf in fork_results)
print(f"REPRODUCTION: {'REPRODUCED' if reproduced else 'NOT REPRODUCED'} "
      f"(failed_segs per fork: {fork_results})")

# ── 3. Newborn control ─────────────────────────────────────────────────────────
print("\n=== (b2) NEWBORN CONTROL ===")
seg0_cmap = build_cmap(CORNER_CYCLE[0])
world_seg0 = World(rows=ROWS, cols=COLS, cmap=seg0_cmap, n_colors=N_COLORS)
nb = Creature.birth("casc_newborn", world_seg0, seed=85)
nb._state_dir = None

# 1000-step settling
BASE_NB = 84
for i in range(1000):
    nb.live(1, seed=BASE_NB * 1000003 + i)
print(f"newborn post-settling: age={nb.age_steps} map_acc={nb.map_accuracy():.4f}")

# 8-segment schedule (continuing seed index from 1000)
nb_lags = []
for seg in range(8):
    offset = 1000 + seg * 500
    lag, end_acc = run_segment(nb, seg, BASE_NB, offset)
    nb_lags.append(lag)
    print(f"  newborn seg{seg} lag={lag} end_acc={end_acc:.4f}")

# Spearman rho (inline, no scipy dependency)
def spearman_rho(xs, ys):
    n = len(xs)
    def ranks(arr):
        order = sorted(range(n), key=lambda i: arr[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and arr[order[j]] == arr[order[i]]:
                j += 1
            avg_rank = (i + j - 1) / 2.0
            for k in range(i, j):
                r[order[k]] = avg_rank
            i = j
        return r
    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)

seg_indices = list(range(8))
rho = spearman_rho(seg_indices, nb_lags)
early_recoveries = sum(1 for i in range(4) if nb_lags[i] <= 450)
print(f"newborn_early_recoveries={early_recoveries} (segments 0-3 with lag<=450)")
print(f"spearman_rho={rho:.4f}")
b2_confirms = rho > 0.6 and early_recoveries >= 2

# ── 4. Summary ─────────────────────────────────────────────────────────────────
print("\n=== SUMMARY ===")
b1_confirms = mean_mass > 20 * 5  # mean column mass >> 20 visits/segment
print(f"(a) reproduction: {'3/3 forks failed >=7/8' if reproduced else 'not all forks failed >=7/8'} "
      f"-> {'REPRODUCED' if reproduced else 'NOT REPRODUCED'}")
print(f"(b1) counts: mean_column_mass={mean_mass:.2f} vs ~20 visits/segment "
      f"-> {'CONFIRMS' if b1_confirms else 'DOES NOT CONFIRM'}")
print(f"(b2) newborn: rho={rho:.4f}, early_recoveries={early_recoveries} "
      f"-> {'CONFIRMS' if b2_confirms else 'DOES NOT CONFIRM'} "
      f"(rho>0.6={'yes' if rho>0.6 else 'no'}, >=2 early={'yes' if early_recoveries>=2 else 'no'})")

if reproduced and b1_confirms and b2_confirms:
    print("VERDICT: candidate DIES — lawful consequence of non-decaying counts "
          "(plasticity ~ 1/accumulated mass); Exp 48 inertia law generalized to perception")
elif reproduced and not (b1_confirms and b2_confirms):
    missing = []
    if not b1_confirms:
        missing.append("b1(counts)")
    if not b2_confirms:
        missing.append("b2(newborn)")
    print(f"VERDICT: NOVELTY-CANDIDATE survives (mechanism checks failed to explain): {missing}")
else:
    print("VERDICT: deviation was trajectory noise (not reproduced)")

print(f"\nruntime={time.time()-t0:.1f}s")
