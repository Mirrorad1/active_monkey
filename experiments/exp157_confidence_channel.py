"""Exp 157 — the internal confidence channel: per-place expected uncertainty
yields meta-d' > 0 where discrimination is possible (N2 prereq build, piece 1).

Hypothesis: a confidence channel that indexes the creature's own
prediction-outcome history by its believed place (per-cell EWMA of binary
correctness — 'expected uncertainty') tracks accuracy (type-2 AUROC > 0.55)
in a world whose evidence quality varies by place, while Exp 155's natural
channel (max predictive probability, A_hat effectively frozen at this mass)
stays near chance. Provided vs self-formed, named: the channel FORM (per-cell
EWMA, alpha=0.05, init=0.5) and the world's noise placement are PROVIDED;
the channel's VALUES are self-formed online from the creature's own lived
prediction outcomes; nothing is read from the generator.

World R-PLACE-NOISE (provided): checkerboard reliability on mirro's 5x5
grid — cells with (row + col) % 2 == 0 are NOISY (true color with
p_true = 0.55, else uniform among the other colors; per-fork seeded rng,
actions untouched), cells with (row + col) % 2 == 1 are CLEAN (true color
always). 13 noisy, 12 clean cells. This fixes Exp 155's malformed-demand
lesson: type-2 discrimination requires trial-level variation in evidence
quality.

Design: forks of mirro only (spine never saved), FRESH seeds 18-25 (Exp
155/156 used 0-17), 4000 steps = 40 chunks x 100, same step-loop semantics
as Exp 155/156. Both channels computed each step BEFORE the Bayes update:
  C-old: max(A_hat @ qs)  (Exp 155's channel)
  C-new: EWMA[bel_cell]   where bel_cell = argmax(qs) pre-update; after the
         step's correctness is known, EWMA[bel_cell] <- (1-alpha)*EWMA[bel_cell]
         + alpha*correct_t  (alpha = 0.05, init 0.5 for all cells).
correct_t = (argmax(A_hat @ qs) == obs). Type-2 AUROC per channel via
Mann-Whitney over correct vs incorrect trials.

Preconditions (instrument-grade; any failure on the confirm set => print
"PRECONDITION FAILED", write rows, no verdict):
  PC1: >= 50 correct AND >= 50 incorrect trials per fork.
  PC2: ahat_drift < 0.05 per fork.
  PC3 (placement validity, L5 input-side): pooled accuracy at clean cells
       minus pooled accuracy at noisy cells >= 0.3 (the world must actually
       create varying evidence quality; cells classified by TRUE position).

Predeclared properties and falsifiers:
  P1 (the channel works): C-new per-fork type-2 AUROC > 0.55 in >= 7/8 forks
     AND pooled C-new AUROC >= 0.60.
     FALSIFIER F1: <= 4/8 forks with C-new AUROC > 0.5, OR pooled <= 0.52.
  P2 (the contrast — built channel beats the natural one): C-new AUROC >
     C-old AUROC per fork in >= 7/8 forks AND (pooled C-new - pooled C-old)
     >= 0.10. FALSIFIER F2: pooled difference <= 0.02.
  VERDICT: POSITIVE iff P1 and P2 both pass. NEGATIVE iff F1 or F2 fires.
  Otherwise MIXED. "Not a falsifier" never counts toward POSITIVE.

Ungated diagnostics (logged, not gated): pooled C-old AUROC; per-fork
detector events from the reconstructed ceiling detector (window 200,
mean/slope, block-end checks — this world's noise is i.i.d. by place, so
firing is allowed and informative, not gated); terminal EWMA value per cell
vs the cell's true reliability (calibration: Pearson r over the 25 cells,
per fork); clean-cell and noisy-cell accuracy per fork.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque as _deque
from pathlib import Path

import numpy as np

from active_loop.creature import (
    Creature,
    SURPRISE_WINDOW,
    CEILING_MEAN_THRESH,
    CEILING_SLOPE_THRESH,
)
from active_loop.verdict import write_verdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = list(range(18, 26))   # fresh seeds — 155 used 0-9, 156 used 10-17

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

ALPHA = 0.05        # EWMA learning rate for the internal confidence channel
EWMA_INIT = 0.5     # initial EWMA value per cell (midpoint)

# NoisyCmap seeded offset: private rng, action stream untouched
NOISE_SEED_OFFSET = 60_000

P_TRUE_NOISY = 0.55    # noisy cells: true color with this probability

# Precondition thresholds
PRECONDITION_MIN_CLASS = 50         # >= 50 correct AND incorrect per fork (PC1)
PRECONDITION_AHAT_DRIFT = 0.05      # max abs A_hat change (PC2)
PC3_ACC_DIFF_MIN = 0.3              # pooled clean_acc - noisy_acc >= 0.3 (PC3)

# P1 thresholds
P1_FORKS_AUROC_MIN = 7              # >= 7/8 forks with C-new AUROC > 0.55
P1_PER_FORK_THRESH = 0.55           # per-fork AUROC threshold for P1 count
P1_POOLED_MIN = 0.60                # pooled C-new AUROC >= 0.60
F1_FORKS_MAX = 4                    # <= 4/8 forks with C-new AUROC > 0.5 => F1
F1_POOLED_MAX = 0.52                # pooled C-new <= 0.52 => F1

# P2 thresholds
P2_FORKS_MIN = 7                    # >= 7/8 forks C-new > C-old
P2_POOLED_DIFF_MIN = 0.10           # pooled C-new - pooled C-old >= 0.10
F2_POOLED_DIFF_MAX = 0.02           # pooled diff <= 0.02 => F2


# ---------------------------------------------------------------------------
# PlaceNoiseCmap — checkerboard reliability wrapper.
# Noisy cells: (row + col) % 2 == 0  =>  true color with p_true = 0.55.
# Clean cells: (row + col) % 2 == 1  =>  true color always.
# The rng is private to this cmap; only noisy cells draw from it so the
# noise stream is position-dependent by design (a noisy-cell lookup consumes
# one draw; a clean-cell lookup does not — declared below for determinism).
# The action stream is untouched (same guarantee as exp155 NoisyCmap).
# ---------------------------------------------------------------------------

class PlaceNoiseCmap:
    """Color map wrapper with per-cell checkerboard reliability.

    Noisy cells (row+col even): true color returned with p_true = 0.55;
    otherwise uniform draw among the other n_colors-1 colors.
    Clean cells (row+col odd): true color always returned (no rng draw).

    The rng is private and seeded per-fork; action stream is untouched.
    """

    def __init__(self, base_cmap, n_colors, rows, cols, p_true, seed):
        assert rows == 5 and cols == 5, (
            f"PlaceNoiseCmap requires 5x5 world, got {rows}x{cols}"
        )
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.rows = int(rows)
        self.cols = int(cols)
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(seed)

    def is_noisy(self, cell: int) -> bool:
        """True iff (row + col) % 2 == 0 for this cell."""
        r, c = divmod(cell, self.cols)
        return (r + c) % 2 == 0

    def __getitem__(self, s: int) -> int:
        true = self.base[s]
        if self.is_noisy(s):
            # Consume one rng draw (noisy path)
            if self.rng.random() < self.p_true:
                return true
            others = [c for c in range(self.n_colors) if c != true]
            return int(self.rng.choice(others))
        else:
            # Clean path: return true color directly, no rng draw
            return true

    def __len__(self) -> int:
        return len(self.base)


# ---------------------------------------------------------------------------
# Mann-Whitney AUROC — verbatim from exp155
# ---------------------------------------------------------------------------

def mannwhitney_auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUROC via Mann-Whitney U statistic.

    pos_scores: confidence values where correct_t = 1
    neg_scores: confidence values where correct_t = 0
    Returns float in [0, 1].
    """
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = 0.0
    for p in pos_scores:
        u += np.sum(p > neg_scores) + 0.5 * np.sum(p == neg_scores)
    return float(u / (n_pos * n_neg))


def pooled_auroc(conf_list: list[np.ndarray],
                 correct_list: list[np.ndarray]) -> float:
    """Compute AUROC on all trials concatenated across forks."""
    all_conf = np.concatenate(conf_list)
    all_correct = np.concatenate(correct_list)
    pos = all_conf[all_correct == 1]
    neg = all_conf[all_correct == 0]
    return mannwhitney_auroc(pos, neg)


# ---------------------------------------------------------------------------
# Run one fork — the core instrumented step loop.
# Mirrors exp156 run_fork step semantics EXACTLY; adds per-cell EWMA channel.
#
# Step semantics (verbatim from exp155/156):
#   - A_hat pre-step
#   - pred_probs = A_hat @ c.qs  (shape n_colors)
#   - o_hat = argmax(pred_probs)
#   - conf_old = float(pred_probs.max())
#   - bel_cell = argmax(c.qs) PRE-update
#   - conf_new = ewma[bel_cell]
#   - obs from cmap_obj[c.true_pos]
#   - correct_t = (o_hat == obs)
#   - surprise = -ln(A_hat[obs,:] @ qs) pre-update
#   - Bayes update
#   - pA update
#   - value accumulation
#   - random action from per-chunk rng (chunk_seed = (fork_seed*10_000+chunk_idx) & 0xFFFFFFFF)
#   - move
#   - advance qs through B
#   - age += 1
#   AFTER correctness known:
#     ewma[bel_cell] = (1-ALPHA)*ewma[bel_cell] + ALPHA*correct_t
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one R-PLACE-NOISE fork (fresh from mirro spine).

    Returns per-fork aggregates and per-step arrays needed for pooled AUROC.
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW={SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp157_s{fork_seed}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4
    rows = c.world.rows
    cols = c.world.cols
    n_colors = c.world.n_colors
    # 5x5 assertion inside PlaceNoiseCmap

    base_cmap = list(mirro.world.cmap)

    noise_seed = NOISE_SEED_OFFSET + fork_seed
    cmap_obj = PlaceNoiseCmap(
        base_cmap=base_cmap,
        n_colors=n_colors,
        rows=rows,
        cols=cols,
        p_true=P_TRUE_NOISY,
        seed=noise_seed,
    )

    # True reliability per cell for calibration diagnostics
    # noisy cells: p_true = 0.55 (if predicted = true color, hit prob depends on A_hat;
    # for calibration we use the raw observational reliability: P(obs==true_color))
    # = p_true for noisy, 1.0 for clean.
    cell_true_reliability = np.array(
        [P_TRUE_NOISY if cmap_obj.is_noisy(s) else 1.0 for s in range(n_cells)]
    )
    cell_is_noisy = np.array([cmap_obj.is_noisy(s) for s in range(n_cells)], dtype=bool)

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    n_total = n_chunks * chunk_size

    # --- Per-step storage ---
    conf_old_arr  = np.empty(n_total, dtype=np.float64)
    conf_new_arr  = np.empty(n_total, dtype=np.float64)
    correct_arr   = np.empty(n_total, dtype=np.int32)
    surprise_arr  = np.empty(n_total, dtype=np.float64)
    true_pos_arr  = np.empty(n_total, dtype=np.int32)
    bel_cell_arr  = np.empty(n_total, dtype=np.int32)

    # Per-cell EWMA state (the internal confidence channel)
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    # Ceiling detector replication (verbatim from exp155/156)
    surprise_win = _deque(maxlen=SURPRISE_WINDOW)
    ceiling_events = 0

    # A_hat drift check
    A_hat_start = c._A_hat().copy()

    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(chunk_size):
            t = global_step

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Predicted observation and confidence (BEFORE update) ---
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            o_hat = int(np.argmax(pred_probs))
            conf_old = float(pred_probs.max())

            # --- bel_cell and conf_new (BEFORE update) ---
            bel_cell = int(np.argmax(c.qs))
            conf_new = float(ewma[bel_cell])

            # --- Observe (from regime-specific cmap) ---
            true_pos_t = c.true_pos
            obs = int(cmap_obj[true_pos_t])

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0

            # --- Surprise: -ln(A_hat[obs,:] @ qs) pre-update ---
            p_o = float(A_hat[obs, :] @ c.qs)
            surprise_t = -math.log(p_o + eps)
            surprise_win.append(surprise_t)

            # --- Store per-step records ---
            conf_old_arr[t]  = conf_old
            conf_new_arr[t]  = conf_new
            correct_arr[t]   = correct_t
            surprise_arr[t]  = surprise_t
            true_pos_arr[t]  = true_pos_t
            bel_cell_arr[t]  = bel_cell

            # --- Belief update: qs ∝ A_hat[obs,:] * qs ---
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # --- Dirichlet count update: pA[obs, :] += qs_updated ---
            c.pA[obs, :] += qs_updated

            # --- Value accumulation (Exp 26 mechanism) ---
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            predictability_weight = math.exp(-h_predicted)
            c.value_counts[obs] += predictability_weight

            # --- Random action ---
            action = int(rng.integers(0, n_actions))

            # --- Move ---
            c.true_pos = c.world.move(c.true_pos, action)

            # --- Advance qs through B ---
            c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1

            # --- EWMA update AFTER correctness known ---
            ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

            global_step += 1

        # --- End of chunk: ceiling check (verbatim from exp155/156) ---
        if len(surprise_win) == SURPRISE_WINDOW:
            win_arr = np.array(surprise_win)
            mean_s = float(win_arr.mean())
            slope = float(np.polyfit(np.arange(SURPRISE_WINDOW), win_arr, 1)[0])
            if mean_s > CEILING_MEAN_THRESH and abs(slope) < CEILING_SLOPE_THRESH:
                ceiling_events += 1

    # --- A_hat drift ---
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # -----------------------------------------------------------------------
    # Per-fork aggregates
    # -----------------------------------------------------------------------

    n_correct   = int(correct_arr.sum())
    n_incorrect = int((correct_arr == 0).sum())

    # Per-channel AUROC
    pos_conf_old = conf_old_arr[correct_arr == 1]
    neg_conf_old = conf_old_arr[correct_arr == 0]
    pos_conf_new = conf_new_arr[correct_arr == 1]
    neg_conf_new = conf_new_arr[correct_arr == 0]

    if n_correct >= PRECONDITION_MIN_CLASS and n_incorrect >= PRECONDITION_MIN_CLASS:
        auroc_old = mannwhitney_auroc(pos_conf_old, neg_conf_old)
        auroc_new = mannwhitney_auroc(pos_conf_new, neg_conf_new)
        precondition_failed_class = False
    else:
        auroc_old = float("nan")
        auroc_new = float("nan")
        precondition_failed_class = True

    # Clean vs noisy accuracy per fork (PC3 / diagnostic)
    true_cell_noisy = cell_is_noisy[true_pos_arr]
    clean_mask = ~true_cell_noisy
    noisy_mask = true_cell_noisy

    n_clean = int(clean_mask.sum())
    n_noisy = int(noisy_mask.sum())
    clean_acc = float(correct_arr[clean_mask].mean()) if n_clean > 0 else float("nan")
    noisy_acc = float(correct_arr[noisy_mask].mean()) if n_noisy > 0 else float("nan")

    # Terminal EWMA vs true reliability — Pearson r over 25 cells
    terminal_ewma = ewma.copy()
    # Pearson r
    x = terminal_ewma
    y = cell_true_reliability
    xm = x - x.mean()
    ym = y - y.mean()
    denom_r = math.sqrt(float((xm ** 2).sum()) * float((ym ** 2).sum()))
    if denom_r > 1e-300:
        ewma_calibration_r = float((xm * ym).sum() / denom_r)
    else:
        ewma_calibration_r = 0.0

    return {
        "fork_seed": fork_seed,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "auroc_new": auroc_new,
        "auroc_old": auroc_old,
        "ahat_drift": ahat_drift,
        "detector_events": ceiling_events,
        "clean_acc": clean_acc,
        "noisy_acc": noisy_acc,
        "ewma_calibration_r": ewma_calibration_r,
        "terminal_ewma": terminal_ewma.tolist(),
        "precondition_failed_class": precondition_failed_class,
        # Full arrays for pooled AUROC computation
        "_conf_old": conf_old_arr,
        "_conf_new": conf_new_arr,
        "_correct": correct_arr,
        "_true_pos": true_pos_arr,
    }


# ---------------------------------------------------------------------------
# Per-fork table printer
# ---------------------------------------------------------------------------

def print_fork_table(rows: list[dict]) -> None:
    hdr = (
        f"{'seed':>5}  "
        f"{'n_cor':>6}  "
        f"{'n_inc':>6}  "
        f"{'auroc_new':>10}  "
        f"{'auroc_old':>10}  "
        f"{'new>old':>7}  "
        f"{'clean_acc':>10}  "
        f"{'noisy_acc':>10}  "
        f"{'ewma_r':>7}  "
        f"{'det_ev':>7}  "
        f"{'drift':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        an = (f"{r['auroc_new']:.4f}" if not math.isnan(r['auroc_new']) else "   nan")
        ao = (f"{r['auroc_old']:.4f}" if not math.isnan(r['auroc_old']) else "   nan")
        beats = ("yes" if (not math.isnan(r['auroc_new']) and
                           not math.isnan(r['auroc_old']) and
                           r['auroc_new'] > r['auroc_old']) else "no ")
        ca = (f"{r['clean_acc']:.4f}" if not math.isnan(r['clean_acc']) else "   nan")
        na = (f"{r['noisy_acc']:.4f}" if not math.isnan(r['noisy_acc']) else "   nan")
        print(
            f"{r['fork_seed']:>5}  "
            f"{r['n_correct']:>6}  "
            f"{r['n_incorrect']:>6}  "
            f"{an:>10}  "
            f"{ao:>10}  "
            f"{beats:>7}  "
            f"{ca:>10}  "
            f"{na:>10}  "
            f"{r['ewma_calibration_r']:>7.4f}  "
            f"{r['detector_events']:>7d}  "
            f"{r['ahat_drift']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Check preconditions
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3.  Returns (all_pass, list_of_failures)."""
    failures = []

    for r in rows:
        seed = r["fork_seed"]

        # PC1: class balance
        if r["n_correct"] < PRECONDITION_MIN_CLASS:
            failures.append(
                f"PC1 FAIL: seed={seed} n_correct={r['n_correct']} < {PRECONDITION_MIN_CLASS}"
            )
        if r["n_incorrect"] < PRECONDITION_MIN_CLASS:
            failures.append(
                f"PC1 FAIL: seed={seed} n_incorrect={r['n_incorrect']} < {PRECONDITION_MIN_CLASS}"
            )

        # PC2: A_hat drift
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: seed={seed} ahat_drift={r['ahat_drift']:.4f} >= {PRECONDITION_AHAT_DRIFT}"
            )

    # PC3: pooled clean_acc - noisy_acc >= 0.3
    clean_accs = [r["clean_acc"] for r in rows if not math.isnan(r["clean_acc"])]
    noisy_accs = [r["noisy_acc"] for r in rows if not math.isnan(r["noisy_acc"])]
    if clean_accs and noisy_accs:
        pooled_clean = float(np.mean(clean_accs))
        pooled_noisy = float(np.mean(noisy_accs))
        acc_diff = pooled_clean - pooled_noisy
        if acc_diff < PC3_ACC_DIFF_MIN:
            failures.append(
                f"PC3 FAIL: pooled clean_acc({pooled_clean:.4f}) - "
                f"noisy_acc({pooled_noisy:.4f}) = {acc_diff:.4f} < {PC3_ACC_DIFF_MIN}"
            )
    else:
        failures.append("PC3 FAIL: could not compute pooled clean/noisy accuracy (nan)")

    all_pass = len(failures) == 0
    return all_pass, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1/F1 and P2/F2; return evaluation dict."""
    n = len(rows)

    # Pooled AUROCs
    pool_new = pooled_auroc([r["_conf_new"] for r in rows],
                            [r["_correct"] for r in rows])
    pool_old = pooled_auroc([r["_conf_old"] for r in rows],
                            [r["_correct"] for r in rows])

    # --- P1: channel works ---
    # Per-fork: C-new AUROC > 0.55 in >= 7/8 forks
    forks_new_above_thresh = sum(
        1 for r in rows
        if not math.isnan(r["auroc_new"]) and r["auroc_new"] > P1_PER_FORK_THRESH
    )
    p1_forks_pass = forks_new_above_thresh >= P1_FORKS_AUROC_MIN
    p1_pooled_pass = (not math.isnan(pool_new)) and pool_new >= P1_POOLED_MIN
    p1_pass = p1_forks_pass and p1_pooled_pass

    # F1: <= 4/8 forks with C-new AUROC > 0.5, OR pooled <= 0.52
    f1_forks_count = sum(
        1 for r in rows
        if not math.isnan(r["auroc_new"]) and r["auroc_new"] > 0.5
    )
    f1 = (f1_forks_count <= F1_FORKS_MAX or
          ((not math.isnan(pool_new)) and pool_new <= F1_POOLED_MAX))

    # --- P2: built channel beats natural one ---
    # Per-fork: C-new AUROC > C-old AUROC in >= 7/8 forks
    forks_new_beats_old = sum(
        1 for r in rows
        if (not math.isnan(r["auroc_new"]) and
            not math.isnan(r["auroc_old"]) and
            r["auroc_new"] > r["auroc_old"])
    )
    p2_forks_pass = forks_new_beats_old >= P2_FORKS_MIN

    pooled_diff = (
        pool_new - pool_old
        if not (math.isnan(pool_new) or math.isnan(pool_old))
        else float("nan")
    )
    p2_pooled_pass = (not math.isnan(pooled_diff)) and pooled_diff >= P2_POOLED_DIFF_MIN
    p2_pass = p2_forks_pass and p2_pooled_pass

    # F2: pooled difference <= 0.02
    f2 = (not math.isnan(pooled_diff)) and pooled_diff <= F2_POOLED_DIFF_MAX

    # --- Verdict ---
    positive = p1_pass and p2_pass
    negative = f1 or f2
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    return {
        "pool_new": pool_new,
        "pool_old": pool_old,
        "pooled_diff": pooled_diff,
        "forks_new_above_thresh": forks_new_above_thresh,
        "p1_forks_pass": p1_forks_pass,
        "p1_pooled_pass": p1_pooled_pass,
        "p1_pass": p1_pass,
        "f1_forks_count": f1_forks_count,
        "f1": f1,
        "forks_new_beats_old": forks_new_beats_old,
        "p2_forks_pass": p2_forks_pass,
        "p2_pooled_pass": p2_pooled_pass,
        "p2_pass": p2_pass,
        "f2": f2,
        "verdict_str": verdict_str,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows
# ---------------------------------------------------------------------------

def write_json_rows(rows: list[dict], path: Path, ev: dict | None,
                    smoke: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            row = {
                "exp": 157,
                "fork_seed": r["fork_seed"],
                "n_total": r["n_total"],
                "n_correct": r["n_correct"],
                "n_incorrect": r["n_incorrect"],
                "auroc_new": r["auroc_new"] if not math.isnan(r["auroc_new"]) else None,
                "auroc_old": r["auroc_old"] if not math.isnan(r["auroc_old"]) else None,
                "ahat_drift": r["ahat_drift"],
                "detector_events": r["detector_events"],
                "clean_acc": r["clean_acc"] if not math.isnan(r["clean_acc"]) else None,
                "noisy_acc": r["noisy_acc"] if not math.isnan(r["noisy_acc"]) else None,
                "ewma_calibration_r": r["ewma_calibration_r"],
                "terminal_ewma": r["terminal_ewma"],
                "precondition_failed_class": r["precondition_failed_class"],
            }
            fh.write(json.dumps(row) + "\n")
        if ev is not None:
            summary = {
                "exp": 157,
                "row_type": "summary",
                "pooled_auroc_new": ev["pool_new"] if not math.isnan(ev["pool_new"]) else None,
                "pooled_auroc_old": ev["pool_old"] if not math.isnan(ev["pool_old"]) else None,
                "pooled_diff": ev["pooled_diff"] if not math.isnan(ev["pooled_diff"]) else None,
                "forks_new_above_thresh": ev["forks_new_above_thresh"],
                "forks_new_beats_old": ev["forks_new_beats_old"],
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 157 — internal confidence channel (per-place EWMA)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke run: seeds=[18] only, 8 chunks (800 steps), no verdict written."
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [18] if smoke else SEEDS
    n_chunks = 8 if smoke else N_CHUNKS

    print("=" * 80)
    print("Exp 157 — internal confidence channel (per-place EWMA vs natural max-prob)")
    print("=" * 80)
    print()

    assert SURPRISE_WINDOW == 200, (
        f"PRECONDITION FAILED: SURPRISE_WINDOW={SURPRISE_WINDOW} != 200"
    )
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  [asserted == 200 OK]")
    print(f"CEILING_MEAN_THRESH={CEILING_MEAN_THRESH}")
    print(f"CEILING_SLOPE_THRESH={CEILING_SLOPE_THRESH}")
    print(f"ALPHA={ALPHA}  EWMA_INIT={EWMA_INIT}")
    print(f"P_TRUE_NOISY={P_TRUE_NOISY}  NOISE_SEED_OFFSET={NOISE_SEED_OFFSET}")
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}, n_cells={mirro.world.n_cells}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    rows_world = mirro.world.rows
    cols_world = mirro.world.cols
    print(f"Base cmap: {base_cmap}")

    # Describe checkerboard layout
    noisy_cells = [s for s in range(25) if ((s // cols_world) + (s % cols_world)) % 2 == 0]
    clean_cells = [s for s in range(25) if ((s // cols_world) + (s % cols_world)) % 2 == 1]
    print(f"Noisy cells ({len(noisy_cells)}, row+col even): {noisy_cells}")
    print(f"Clean cells ({len(clean_cells)}, row+col odd):  {clean_cells}")
    print(f"Seeds: {seeds}  n_chunks={n_chunks}  chunk_size={CHUNK_SIZE}  "
          f"n_steps={n_chunks * CHUNK_SIZE}")
    print()

    # -----------------------------------------------------------------------
    # Run all forks
    # -----------------------------------------------------------------------
    fork_rows = []
    for seed in seeds:
        print(f"  Running seed={seed} (n_chunks={n_chunks}) ...", flush=True)
        r = run_fork(mirro, fork_seed=seed, n_chunks=n_chunks, chunk_size=CHUNK_SIZE)
        fork_rows.append(r)

    print()

    # -----------------------------------------------------------------------
    # Per-fork table
    # -----------------------------------------------------------------------
    print("Per-fork table:")
    print_fork_table(fork_rows)
    print()

    # -----------------------------------------------------------------------
    # Ungated diagnostics (always printed)
    # -----------------------------------------------------------------------
    print("Ungated diagnostics:")
    for r in fork_rows:
        print(f"  seed={r['fork_seed']}  "
              f"auroc_old={r['auroc_old']:.4f}  "
              f"det_events={r['detector_events']}  "
              f"ewma_r={r['ewma_calibration_r']:.4f}  "
              f"clean_acc={r['clean_acc']:.4f}  "
              f"noisy_acc={r['noisy_acc']:.4f}")
    print()

    # -----------------------------------------------------------------------
    # Smoke exit
    # -----------------------------------------------------------------------
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # -----------------------------------------------------------------------
    # Precondition check
    # -----------------------------------------------------------------------
    pc_pass, pc_failures = check_preconditions(fork_rows)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp157_rows.json"
        write_json_rows(fork_rows, out_rows_path, ev=None)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # -----------------------------------------------------------------------
    # Evaluate predeclared outcome map
    # -----------------------------------------------------------------------
    ev = evaluate(fork_rows)
    n = ev["n"]

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    # P1 detail
    new_aurocs = [f"{r['auroc_new']:.4f}" if not math.isnan(r['auroc_new']) else "nan"
                  for r in fork_rows]
    print(f"P1 (channel works):")
    print(f"  C-new AUROC per fork: [{', '.join(new_aurocs)}]")
    print(f"  Forks with C-new > {P1_PER_FORK_THRESH}: "
          f"{ev['forks_new_above_thresh']}/{n} (need >={P1_FORKS_AUROC_MIN} for pass)")
    pool_new_s = f"{ev['pool_new']:.4f}" if not math.isnan(ev['pool_new']) else "nan"
    print(f"  Pooled C-new AUROC: {pool_new_s} "
          f"(need >={P1_POOLED_MIN} for pass)")
    print(f"  F1 check: forks C-new>0.5: {ev['f1_forks_count']}/{n} "
          f"(fires if <={F1_FORKS_MAX}); "
          f"pooled <= {F1_POOLED_MAX}: "
          f"{'yes' if (not math.isnan(ev['pool_new']) and ev['pool_new'] <= F1_POOLED_MAX) else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED")
    print(f"  => P1: {p1_status}")
    print()

    # P2 detail
    old_aurocs = [f"{r['auroc_old']:.4f}" if not math.isnan(r['auroc_old']) else "nan"
                  for r in fork_rows]
    print(f"P2 (built channel beats natural one):")
    print(f"  C-old AUROC per fork: [{', '.join(old_aurocs)}]")
    print(f"  Forks with C-new > C-old: "
          f"{ev['forks_new_beats_old']}/{n} (need >={P2_FORKS_MIN} for pass)")
    pooled_diff_s = (f"{ev['pooled_diff']:.4f}" if not math.isnan(ev['pooled_diff'])
                     else "nan")
    print(f"  Pooled diff (C-new - C-old): {pooled_diff_s} "
          f"(need >={P2_POOLED_DIFF_MIN} for pass; F2 fires if <={F2_POOLED_DIFF_MAX})")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED")
    print(f"  => P2: {p2_status}")
    print()

    # -----------------------------------------------------------------------
    # Conjunct summary + VERDICT
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (C-new AUROC > {P1_PER_FORK_THRESH} in >={P1_FORKS_AUROC_MIN}/{n} forks "
          f"AND pooled >={P1_POOLED_MIN}): {p1_status}")
    print(f"P2 (C-new > C-old in >={P2_FORKS_MIN}/{n} forks "
          f"AND pooled diff >={P2_POOLED_DIFF_MIN}): {p2_status}")
    print()
    print(f"VERDICT: {ev['verdict_str']}")
    print()
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Write JSON rows
    # -----------------------------------------------------------------------
    out_rows_path = Path(__file__).parent / "outputs" / "exp157_rows.json"
    write_json_rows(fork_rows, out_rows_path, ev=ev)
    print(f"JSON rows written to {out_rows_path}")

    # -----------------------------------------------------------------------
    # Write verdict JSON
    # -----------------------------------------------------------------------
    arms = {
        "P1_channel_works": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"C-new AUROC > {P1_PER_FORK_THRESH} in "
                f"{ev['forks_new_above_thresh']}/{n} forks (need {P1_FORKS_AUROC_MIN}); "
                f"pooled C-new AUROC={ev['pool_new']:.4f} (need >={P1_POOLED_MIN}); "
                f"F1 forks count={ev['f1_forks_count']}/{n}"
            ),
        },
        "P2_beats_natural": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"C-new > C-old in {ev['forks_new_beats_old']}/{n} forks "
                f"(need {P2_FORKS_MIN}); "
                f"pooled diff={pooled_diff_s} "
                f"(need >={P2_POOLED_DIFF_MIN}; F2 fires if <={F2_POOLED_DIFF_MAX})"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp157_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp157",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N2 prereq build piece 1: internal per-place EWMA confidence channel "
            "vs natural max-prob channel (C-old). World R-PLACE-NOISE: checkerboard "
            "reliability (13 noisy p=0.55, 12 clean). Channel FORM and world noise "
            "placement are PROVIDED; channel VALUES are self-formed online from the "
            "creature's own lived prediction outcomes; nothing is read from the generator."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
