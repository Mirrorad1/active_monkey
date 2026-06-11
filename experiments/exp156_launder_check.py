"""Exp 156 — mechanism check of the detector's structural blindness (Exp 155):
the mislocalization-laundering hypothesis.

Hypothesis (named in Exp 155, UNVERIFIED there): in R-STRUCT context-B
(deranged) phases, the creature absorbs structural mismatch as localization
error — the posterior qs teleports to a cell whose (stale) learned map is
consistent with the deranged observation — keeping per-step surprise below
the detector's mean threshold (CEILING_MEAN_THRESH = 0.7 nats). That, not
window dilution, explains the 0/8 detector silence.

Design: R-STRUCT only (the within-run context-A phases are the control).
FRESH fork seeds 10-17 (the hypothesis was formed post-hoc on seeds 0-9;
LESSONS L7). Forks of mirro only; the spine is never saved. 4000 steps =
40 blocks of 100; blocks alternate context A (base cmap), context B
(derangement [1,2,0]). Steady state of a block = its last 80 steps (first
20 = predeclared transient, excluded from gated rates). qs_post = the
posterior AFTER the Bayes update with that step's observation; map_cell =
argmax(qs_post); mislocalized = (map_cell != true_pos).

Predeclared properties and falsifiers:

  P1 (teleport): per fork, B-phase steady-state mislocalization rate >= 0.8
     AND A-phase steady-state mislocalization rate <= 0.2; both in >= 7/8
     forks. FALSIFIER F1: B-phase steady-state mislocalization <= 0.5 in
     >= 4/8 forks (no teleport — laundering refuted).

  P2 (self-consistency): on mislocalized B-phase steady-state steps, the
     creature's own map at its believed cell predicts what it saw:
     argmax_c A_hat[c, map_cell] == obs at rate >= 0.8 per fork, in >= 7/8
     forks. FALSIFIER F2: pooled (all forks' mislocalized B-steady steps
     concatenated) consistency rate < 0.6.

  P3 (surprise account, load-bearing): per fork, B-phase steady-state mean
     per-step surprise < 0.7 nats (CEILING_MEAN_THRESH) in >= 7/8 forks —
     the detector is silent because the per-step signal ITSELF is
     sub-threshold (strong laundering), not because the 200-step window
     straddles A and B. FALSIFIER F3: B-phase steady-state mean surprise
     >= 0.7 in >= 4/8 forks.

  NAMED ALTERNATIVE (DILUTION, reported either way): if P1 and P2 pass but
     P3 fails, and per fork mean_B >= 0.7 while ALL reconstructed 200-step
     window means < 0.7, the silence is window-straddle averaging — a
     different mechanism; log as laundering-refuted-in-strong-form.

  INSTRUMENT CHECK (no verdict if violated): the reconstructed detector
     (window 200, mean > CEILING_MEAN_THRESH and |slope| <
     CEILING_SLOPE_THRESH, evaluated at block ends) must produce ZERO events
     in every fork (replicating Exp 155); any event => print
     "INSTRUMENT CHECK FAILED" and write no verdict.
     Also PC2 from Exp 155: ahat_drift < 0.05 per fork.

  VERDICT: POSITIVE iff P1, P2, P3 all pass. NEGATIVE iff any falsifier
     fires. Otherwise MIXED. "Not a falsifier" never counts toward POSITIVE.

  Ungated diagnostics (logged, not gated): mean posterior entropy of qs_post
     per phase; transient length per switch (steps from block start until
     mislocalization state first flips, mean per fork per direction); per
     fork mean_A, mean_B, max reconstructed window mean; pooled dilution
     decomposition.
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

SEEDS = list(range(10, 18))  # fresh seeds — hypothesis formed post-hoc on 0-9

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS        # 100
CONTEXT_HALF_PERIOD = 100              # R-STRUCT: alternate every 100 steps

TRANSIENT_STEPS = 20                   # first 20 steps of each block excluded from gated rates
# steady-state = steps [20, 99] within each 100-step block = 80 steps/block

PRECONDITION_AHAT_DRIFT = 0.05         # PC2 from Exp 155

# Predeclared thresholds
P1_B_MISLOC_MIN = 0.8                  # B-phase steady-state misloc rate >= 0.8
P1_A_MISLOC_MAX = 0.2                  # A-phase steady-state misloc rate <= 0.2
P1_FORKS_MIN = 7                       # both conditions in >= 7/8 forks
F1_B_MISLOC_MAX = 0.5                  # B-phase <= 0.5 in >= 4/8 forks => falsifier
F1_FORKS_TRIGGER = 4

P2_PER_FORK_MIN = 0.8                  # consistency rate >= 0.8 per fork
P2_FORKS_MIN = 7                       # in >= 7/8 forks
F2_POOLED_MIN = 0.6                    # pooled rate < 0.6 => falsifier

P3_PER_FORK_MAX = CEILING_MEAN_THRESH  # B-phase steady mean < 0.7 per fork
P3_FORKS_MIN = 7                       # in >= 7/8 forks
F3_FORKS_TRIGGER = 4                   # mean_B >= 0.7 in >= 4/8 forks => falsifier


# ---------------------------------------------------------------------------
# StructCmap — verbatim copy from exp155
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every CONTEXT_HALF_PERIOD steps.

    Context A = base cmap (standard).
    Context B = derangement applied to cell->color lookup.

    The step counter is maintained externally and passed to __call__.
    For __getitem__ use, the caller must set self.current_step first.
    """

    def __init__(self, base_cmap, n_colors, derangement, half_period=100):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.derangement = list(derangement)  # sigma[old_color] = new_color
        self.half_period = int(half_period)
        self.current_step = 0  # caller sets before __getitem__

    def __getitem__(self, s):
        # context index: 0 = A (base), 1 = B (derangement)
        ctx = (self.current_step // self.half_period) % 2
        true_color = self.base[s]
        if ctx == 0:
            return true_color
        else:
            return self.derangement[true_color]

    def __len__(self):
        return len(self.base)


def make_derangement(n_colors: int) -> list[int]:
    """Return a fixed derangement (no fixed points) of range(n_colors).

    For n_colors=3 returns [1, 2, 0].
    For general n: cyclic shift by 1 is always a derangement.
    """
    return [(i + 1) % n_colors for i in range(n_colors)]


# ---------------------------------------------------------------------------
# Step-loop entropy helper
# ---------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector (nats)."""
    p = p + 1e-300
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
# Run one fork in R-STRUCT — full per-step instrumented loop.
# Mirrors exp155's run_fork_regime step semantics EXACTLY; adds per-step
# recording for the laundering mechanism check.
# ---------------------------------------------------------------------------

def run_fork(
    mirro: Creature,
    fork_seed: int,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one R-STRUCT fork (fresh from mirro spine).

    Returns per-fork aggregates and per-step arrays needed for diagnostics.

    Step semantics (verbatim from exp155 run_fork_regime):
      - A_hat pre-step
      - predicted obs/conf before update
      - obs from cmap with cmap_obj.current_step = t
      - surprise = -ln(A_hat[obs,:] @ qs) pre-update
      - Bayes update
      - pA update
      - value accumulation
      - random action from per-chunk rng with chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
      - move
      - advance qs through B
      - age_steps += 1
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW={SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp156_s{fork_seed}_R-STRUCT"
    c = mirro.fork(fork_name)

    from active_loop.creature import World  # noqa: F401 (for type reference)

    n_cells = c.world.n_cells
    n_actions = 4

    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    derangement = make_derangement(n_colors)

    cmap_obj = StructCmap(base_cmap, n_colors, derangement,
                          half_period=CONTEXT_HALF_PERIOD)

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    n_total = n_chunks * chunk_size

    # --- Per-step storage ---
    surprise_arr      = np.empty(n_total, dtype=np.float64)
    obs_arr           = np.empty(n_total, dtype=np.int32)
    true_pos_arr      = np.empty(n_total, dtype=np.int32)
    map_cell_arr      = np.empty(n_total, dtype=np.int32)
    misloc_arr        = np.empty(n_total, dtype=np.bool_)
    self_consist_arr  = np.empty(n_total, dtype=np.bool_)
    phase_arr         = np.empty(n_total, dtype=np.int32)   # 0=A, 1=B
    block_arr         = np.empty(n_total, dtype=np.int32)
    in_steady_arr     = np.empty(n_total, dtype=np.bool_)
    entropy_arr       = np.empty(n_total, dtype=np.float64)

    # Ceiling detector replication (verbatim from exp155)
    surprise_win = _deque(maxlen=SURPRISE_WINDOW)
    ceiling_events = 0
    window_means: list[float] = []   # one per block-end check

    # A_hat drift check
    A_hat_start = c._A_hat().copy()

    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for step_in_chunk in range(chunk_size):
            t = global_step

            # Update step counter for R-STRUCT before reading cmap
            cmap_obj.current_step = t

            # Block index and phase
            block_idx = t // CONTEXT_HALF_PERIOD
            phase_t = block_idx % 2              # 0=A, 1=B
            step_in_block = t % CONTEXT_HALF_PERIOD
            in_steady_t = step_in_block >= TRANSIENT_STEPS

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Predicted observation and confidence (BEFORE update) ---
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            # o_hat = argmax
            _ = int(np.argmax(pred_probs))  # kept for completeness; not used in 156 metrics

            # --- Observe (from regime-specific cmap) ---
            obs = int(cmap_obj[c.true_pos])

            # --- Surprise: -ln(p(o_t)) = -ln(A_hat[obs,:] @ qs) pre-update ---
            p_o = float(A_hat[obs, :] @ c.qs)
            surprise_t = -math.log(p_o + eps)
            surprise_win.append(surprise_t)

            # --- True position BEFORE the move ---
            true_pos_t = c.true_pos

            # --- Self-consistency flag (using SAME pre-update A_hat) ---
            # argmax_c A_hat[c, map_cell_before_update] is computed after Bayes update
            # so we need qs_updated first; but A_hat is fixed (pre-step); compute below.

            # --- Belief update: qs ∝ A_hat[obs,:] * qs ---
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # qs_post = qs_updated (posterior AFTER Bayes update this step)
            map_cell_t = int(np.argmax(qs_updated))

            # Self-consistency: argmax_c A_hat[:, map_cell] == obs
            # A_hat is the same pre-update A_hat used above
            predicted_color_at_belief = int(np.argmax(A_hat[:, map_cell_t]))
            self_consist_t = (predicted_color_at_belief == obs)

            # Posterior entropy
            ent_t = entropy(qs_updated)

            # --- Dirichlet count update: pA[obs, :] += qs_updated ---
            c.pA[obs, :] += qs_updated

            # --- Value accumulation (Exp 26 mechanism) ---
            predicted_obs_dist = A_hat[:, map_cell_t]
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

            # --- Store per-step records ---
            surprise_arr[t]     = surprise_t
            obs_arr[t]          = obs
            true_pos_arr[t]     = true_pos_t
            map_cell_arr[t]     = map_cell_t
            misloc_arr[t]       = (map_cell_t != true_pos_t)
            self_consist_arr[t] = self_consist_t
            phase_arr[t]        = phase_t
            block_arr[t]        = block_idx
            in_steady_arr[t]    = in_steady_t
            entropy_arr[t]      = ent_t

            global_step += 1

        # --- End of chunk: ceiling check (verbatim from exp155) ---
        if len(surprise_win) == SURPRISE_WINDOW:
            win_arr = np.array(surprise_win)
            mean_s = float(win_arr.mean())
            slope = float(np.polyfit(np.arange(SURPRISE_WINDOW), win_arr, 1)[0])
            window_means.append(mean_s)
            if mean_s > CEILING_MEAN_THRESH and abs(slope) < CEILING_SLOPE_THRESH:
                ceiling_events += 1

    # --- A_hat drift ---
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # -----------------------------------------------------------------------
    # Compute per-fork aggregates
    # -----------------------------------------------------------------------

    # Masks
    A_steady  = (phase_arr == 0) & in_steady_arr
    B_steady  = (phase_arr == 1) & in_steady_arr

    def misloc_rate(mask: np.ndarray) -> float:
        n = mask.sum()
        if n == 0:
            return float("nan")
        return float(misloc_arr[mask].sum() / n)

    def mean_surprise(mask: np.ndarray) -> float:
        n = mask.sum()
        if n == 0:
            return float("nan")
        return float(surprise_arr[mask].mean())

    def mean_entropy_phase(mask: np.ndarray) -> float:
        n = mask.sum()
        if n == 0:
            return float("nan")
        return float(entropy_arr[mask].mean())

    A_steady_misloc_rate = misloc_rate(A_steady)
    B_steady_misloc_rate = misloc_rate(B_steady)
    A_steady_mean_surprise = mean_surprise(A_steady)
    B_steady_mean_surprise = mean_surprise(B_steady)
    A_mean_entropy = mean_entropy_phase(A_steady)
    B_mean_entropy = mean_entropy_phase(B_steady)

    # B-phase steady-state consistency (conditional on mislocalized steps)
    B_steady_misloc_mask = B_steady & misloc_arr
    n_b_misloc = int(B_steady_misloc_mask.sum())
    if n_b_misloc > 0:
        B_steady_consistency_rate = float(
            self_consist_arr[B_steady_misloc_mask].sum() / n_b_misloc
        )
    else:
        B_steady_consistency_rate = None

    max_window_mean = float(max(window_means)) if window_means else float("nan")

    # -----------------------------------------------------------------------
    # Transient length diagnostics (ungated)
    # Steps from block start until mislocalization state first flips
    # per direction: A->B switch (block becomes B), B->A switch (block becomes A)
    # Mean across all such transitions in this fork.
    # -----------------------------------------------------------------------
    transient_a_to_b: list[int] = []  # steps until first misloc=True after B-block starts
    transient_b_to_a: list[int] = []  # steps until first misloc=False after A-block starts

    n_blocks = n_total // CONTEXT_HALF_PERIOD
    for blk in range(n_blocks):
        blk_start = blk * CONTEXT_HALF_PERIOD
        blk_end   = blk_start + CONTEXT_HALF_PERIOD
        blk_phase = blk % 2
        blk_misloc = misloc_arr[blk_start:blk_end]
        if blk_phase == 1:
            # B block: look for first step where misloc=True
            flips = np.where(blk_misloc)[0]
            if len(flips) > 0:
                transient_a_to_b.append(int(flips[0]))
        else:
            # A block (blk > 0): look for first step where misloc=False
            if blk > 0:
                flips = np.where(~blk_misloc)[0]
                if len(flips) > 0:
                    transient_b_to_a.append(int(flips[0]))

    mean_transient_a_to_b = float(np.mean(transient_a_to_b)) if transient_a_to_b else float("nan")
    mean_transient_b_to_a = float(np.mean(transient_b_to_a)) if transient_b_to_a else float("nan")

    return {
        "fork_seed": fork_seed,
        # Per-fork aggregates
        "A_steady_misloc_rate":    A_steady_misloc_rate,
        "B_steady_misloc_rate":    B_steady_misloc_rate,
        "A_steady_mean_surprise":  A_steady_mean_surprise,
        "B_steady_mean_surprise":  B_steady_mean_surprise,
        "B_steady_consistency_rate": B_steady_consistency_rate,
        "A_mean_entropy":          A_mean_entropy,
        "B_mean_entropy":          B_mean_entropy,
        "max_window_mean":         max_window_mean,
        "detector_events":         ceiling_events,
        "ahat_drift":              ahat_drift,
        "mean_transient_a_to_b":   mean_transient_a_to_b,
        "mean_transient_b_to_a":   mean_transient_b_to_a,
        "window_means":            window_means,
        # Raw arrays for pooled computation
        "_B_steady_misloc_mask":   B_steady_misloc_mask,
        "_self_consist_arr":       self_consist_arr,
    }


# ---------------------------------------------------------------------------
# Per-fork table printer
# ---------------------------------------------------------------------------

def print_fork_table(rows: list[dict]) -> None:
    hdr = (
        f"{'seed':>5}  "
        f"{'B_misloc':>9}  "
        f"{'A_misloc':>9}  "
        f"{'B_surprise':>11}  "
        f"{'A_surprise':>11}  "
        f"{'consist':>8}  "
        f"{'det_ev':>7}  "
        f"{'drift':>8}  "
        f"{'max_win':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        consist_s = (f"{r['B_steady_consistency_rate']:.4f}"
                     if r['B_steady_consistency_rate'] is not None else "   None")
        print(
            f"{r['fork_seed']:>5}  "
            f"{r['B_steady_misloc_rate']:>9.4f}  "
            f"{r['A_steady_misloc_rate']:>9.4f}  "
            f"{r['B_steady_mean_surprise']:>11.4f}  "
            f"{r['A_steady_mean_surprise']:>11.4f}  "
            f"{consist_s:>8}  "
            f"{r['detector_events']:>7d}  "
            f"{r['ahat_drift']:>8.4f}  "
            f"{r['max_window_mean']:>8.4f}"
        )


# ---------------------------------------------------------------------------
# Write JSONL rows
# ---------------------------------------------------------------------------

def write_json_rows(rows: list[dict], path: Path, smoke: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            # Phase A row
            row_a = {
                "exp": 156,
                "fork_seed": r["fork_seed"],
                "phase": "A",
                "steady_misloc_rate": r["A_steady_misloc_rate"],
                "steady_mean_surprise": r["A_steady_mean_surprise"],
                "mean_entropy": r["A_mean_entropy"],
                "detector_events": r["detector_events"],
                "ahat_drift": r["ahat_drift"],
                "max_window_mean": r["max_window_mean"],
                "mean_transient_b_to_a": r["mean_transient_b_to_a"],
            }
            fh.write(json.dumps(row_a) + "\n")
            # Phase B row
            row_b = {
                "exp": 156,
                "fork_seed": r["fork_seed"],
                "phase": "B",
                "steady_misloc_rate": r["B_steady_misloc_rate"],
                "steady_mean_surprise": r["B_steady_mean_surprise"],
                "mean_entropy": r["B_mean_entropy"],
                "B_steady_consistency_rate": r["B_steady_consistency_rate"],
                "detector_events": r["detector_events"],
                "ahat_drift": r["ahat_drift"],
                "max_window_mean": r["max_window_mean"],
                "mean_transient_a_to_b": r["mean_transient_a_to_b"],
            }
            fh.write(json.dumps(row_b) + "\n")


# ---------------------------------------------------------------------------
# Main evaluation logic
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1/P2/P3 and falsifiers; return evaluation dict."""
    n = len(rows)

    # --- Instrument check: zero detector events in every fork ---
    instrument_ok = all(r["detector_events"] == 0 for r in rows)

    # --- PC2: ahat_drift < 0.05 in every fork ---
    pc2_ok = all(r["ahat_drift"] < PRECONDITION_AHAT_DRIFT for r in rows)

    # --- P1: teleport ---
    b_misloc_high = [r["B_steady_misloc_rate"] >= P1_B_MISLOC_MIN for r in rows]
    a_misloc_low  = [r["A_steady_misloc_rate"] <= P1_A_MISLOC_MAX for r in rows]
    p1_both       = [b and a for b, a in zip(b_misloc_high, a_misloc_low)]
    p1_forks_pass = sum(p1_both)
    p1_pass       = p1_forks_pass >= P1_FORKS_MIN

    # F1: B-phase steady misloc <= 0.5 in >= 4/8 forks
    f1_count = sum(1 for r in rows if r["B_steady_misloc_rate"] <= F1_B_MISLOC_MAX)
    f1       = f1_count >= F1_FORKS_TRIGGER

    # --- P2: self-consistency (per-fork >= 0.8, in >= 7/8 forks) ---
    p2_per_fork = []
    for r in rows:
        cr = r["B_steady_consistency_rate"]
        if cr is None:
            p2_per_fork.append(False)
        else:
            p2_per_fork.append(cr >= P2_PER_FORK_MIN)
    p2_forks_pass = sum(p2_per_fork)
    p2_pass       = p2_forks_pass >= P2_FORKS_MIN

    # F2: pooled consistency rate < 0.6
    all_misloc_b_steps  = np.concatenate([r["_B_steady_misloc_mask"] for r in rows])
    all_self_consist    = np.concatenate([r["_self_consist_arr"] for r in rows])
    pooled_misloc_mask  = all_misloc_b_steps
    n_pooled_misloc     = int(pooled_misloc_mask.sum())
    if n_pooled_misloc > 0:
        pooled_consistency = float(all_self_consist[pooled_misloc_mask].sum() / n_pooled_misloc)
    else:
        pooled_consistency = float("nan")
    f2 = (not math.isnan(pooled_consistency)) and pooled_consistency < F2_POOLED_MIN

    # --- P3: B-phase steady surprise < 0.7 ---
    b_surprise_low = [r["B_steady_mean_surprise"] < P3_PER_FORK_MAX for r in rows]
    p3_forks_pass  = sum(b_surprise_low)
    p3_pass        = p3_forks_pass >= P3_FORKS_MIN

    # F3: mean_B >= 0.7 in >= 4/8 forks
    f3_count = sum(1 for r in rows if r["B_steady_mean_surprise"] >= P3_PER_FORK_MAX)
    f3       = f3_count >= F3_FORKS_TRIGGER

    # --- DILUTION alternative ---
    # P1 and P2 pass but P3 fails; per fork mean_B >= 0.7; ALL window means < 0.7
    dilution_candidate = (p1_pass and p2_pass and not p3_pass)
    if dilution_candidate:
        all_b_high    = all(r["B_steady_mean_surprise"] >= P3_PER_FORK_MAX for r in rows)
        all_win_low   = all(
            all(wm < CEILING_MEAN_THRESH for wm in r["window_means"])
            for r in rows if r["window_means"]
        )
        dilution_flag = all_b_high and all_win_low
    else:
        dilution_flag = False

    # --- Verdict ---
    positive = p1_pass and p2_pass and p3_pass
    negative = f1 or f2 or f3
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    return {
        "instrument_ok": instrument_ok,
        "pc2_ok": pc2_ok,
        "p1_forks_pass": p1_forks_pass,
        "p1_b_misloc_high_per_fork": b_misloc_high,
        "p1_a_misloc_low_per_fork": a_misloc_low,
        "p1_pass": p1_pass,
        "f1_count": f1_count,
        "f1": f1,
        "p2_forks_pass": p2_forks_pass,
        "p2_per_fork": p2_per_fork,
        "pooled_consistency": pooled_consistency,
        "n_pooled_misloc": n_pooled_misloc,
        "p2_pass": p2_pass,
        "f2": f2,
        "p3_forks_pass": p3_forks_pass,
        "b_surprise_low_per_fork": b_surprise_low,
        "p3_pass": p3_pass,
        "f3_count": f3_count,
        "f3": f3,
        "dilution_flag": dilution_flag,
        "verdict_str": verdict_str,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 156 — mislocalization laundering check")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke run: seeds=[10] only, 8 blocks (800 steps), no verdict written."
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [10] if smoke else SEEDS
    n_chunks = 8 if smoke else N_CHUNKS

    print("=" * 80)
    print("Exp 156 — Mislocalization-laundering mechanism check")
    print("=" * 80)
    print()

    assert SURPRISE_WINDOW == 200, (
        f"PRECONDITION FAILED: SURPRISE_WINDOW={SURPRISE_WINDOW} != 200"
    )
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  [asserted == 200 OK]")
    print(f"CEILING_MEAN_THRESH={CEILING_MEAN_THRESH}")
    print(f"CEILING_SLOPE_THRESH={CEILING_SLOPE_THRESH}")
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}")
    base_cmap_check = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    derangement_check = make_derangement(n_colors)
    print(f"Base cmap: {base_cmap_check}")
    print(f"Derangement (context B): {derangement_check}")
    print(f"  (sigma[i]!=i for all i: {all(derangement_check[i] != i for i in range(n_colors))})")
    print(f"Seeds: {seeds}")
    print(f"n_chunks={n_chunks}  chunk_size={CHUNK_SIZE}  "
          f"n_steps={n_chunks * CHUNK_SIZE}  TRANSIENT_STEPS={TRANSIENT_STEPS}")
    print()

    # -----------------------------------------------------------------------
    # Run all forks
    # -----------------------------------------------------------------------
    rows = []
    for seed in seeds:
        print(f"  Running seed={seed} R-STRUCT (n_chunks={n_chunks}) ...", flush=True)
        r = run_fork(mirro, fork_seed=seed, n_chunks=n_chunks, chunk_size=CHUNK_SIZE)
        rows.append(r)

    print()

    # -----------------------------------------------------------------------
    # Per-fork table
    # -----------------------------------------------------------------------
    print("Per-fork table:")
    print_fork_table(rows)
    print()

    # -----------------------------------------------------------------------
    # Ungated diagnostics
    # -----------------------------------------------------------------------
    print("Ungated diagnostics:")
    for r in rows:
        print(f"  seed={r['fork_seed']}  "
              f"mean_entropy_A={r['A_mean_entropy']:.4f}  "
              f"mean_entropy_B={r['B_mean_entropy']:.4f}  "
              f"transient_A->B={r['mean_transient_a_to_b']:.1f}  "
              f"transient_B->A={r['mean_transient_b_to_a']:.1f}")
    print()

    # -----------------------------------------------------------------------
    # Smoke exit
    # -----------------------------------------------------------------------
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # -----------------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------------
    ev = evaluate(rows)

    n = len(seeds)

    # --- Instrument check ---
    if not ev["instrument_ok"]:
        bad = [r["fork_seed"] for r in rows if r["detector_events"] > 0]
        print(f"INSTRUMENT CHECK FAILED: detector events in forks {bad}")
        out_rows_path = Path(__file__).parent / "outputs" / "exp156_rows.json"
        write_json_rows(rows, out_rows_path)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    # --- PC2 ---
    if not ev["pc2_ok"]:
        bad = [(r["fork_seed"], r["ahat_drift"]) for r in rows
               if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT]
        print(f"PRECONDITION FAILED (PC2): ahat_drift >= 0.05 in forks: {bad}")
        out_rows_path = Path(__file__).parent / "outputs" / "exp156_rows.json"
        write_json_rows(rows, out_rows_path)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    # -----------------------------------------------------------------------
    # Conjunct-by-conjunct evaluation
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    # P1
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "FAIL")
    b_rates = [f"{r['B_steady_misloc_rate']:.3f}" for r in rows]
    a_rates = [f"{r['A_steady_misloc_rate']:.3f}" for r in rows]
    print(f"P1 (teleport):")
    print(f"  B-phase misloc rates:  [{', '.join(b_rates)}]")
    print(f"  A-phase misloc rates:  [{', '.join(a_rates)}]")
    print(f"  Forks with B>=0.8 AND A<=0.2: {ev['p1_forks_pass']}/{n} (need >={P1_FORKS_MIN})")
    print(f"  F1 count (B<=0.5): {ev['f1_count']}/{n} (triggers at >={F1_FORKS_TRIGGER}): "
          f"{'FIRES' if ev['f1'] else 'no'}")
    print(f"  => P1: {p1_status}")
    print()

    # P2
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "FAIL")
    consist_rates = []
    for r in rows:
        cr = r["B_steady_consistency_rate"]
        consist_rates.append(f"{cr:.3f}" if cr is not None else "None")
    pooled_s = (f"{ev['pooled_consistency']:.4f}" if not math.isnan(ev["pooled_consistency"])
                else "nan")
    print(f"P2 (self-consistency):")
    print(f"  Per-fork consistency rates (on misloc B-steady steps): "
          f"[{', '.join(consist_rates)}]")
    print(f"  Forks with rate>=0.8: {ev['p2_forks_pass']}/{n} (need >={P2_FORKS_MIN})")
    print(f"  Pooled consistency (n_misloc={ev['n_pooled_misloc']}): {pooled_s} "
          f"(F2 fires if <{F2_POOLED_MIN}): {'FIRES' if ev['f2'] else 'no'}")
    print(f"  => P2: {p2_status}")
    print()

    # P3
    p3_status = "PASS" if ev["p3_pass"] else ("FALSIFIER F3" if ev["f3"] else "FAIL")
    b_surp = [f"{r['B_steady_mean_surprise']:.4f}" for r in rows]
    print(f"P3 (surprise sub-threshold):")
    print(f"  B-phase steady mean surprise: [{', '.join(b_surp)}]")
    print(f"  Forks with mean_B<{P3_PER_FORK_MAX}: {ev['p3_forks_pass']}/{n} "
          f"(need >={P3_FORKS_MIN})")
    print(f"  F3 count (mean_B>={P3_PER_FORK_MAX}): {ev['f3_count']}/{n} "
          f"(triggers at >={F3_FORKS_TRIGGER}): {'FIRES' if ev['f3'] else 'no'}")
    print(f"  => P3: {p3_status}")
    print()

    # Dilution decomposition
    if ev["dilution_flag"]:
        print("DILUTION DECOMPOSITION: P1+P2 pass, P3 fails; per-fork mean_B >= 0.7 AND "
              "all window means < 0.7 => window-straddle averaging (laundering-refuted-in-strong-form)")
    else:
        print("DILUTION DECOMPOSITION: dilution alternative not active")
    print()

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (teleport, B>=0.8 AND A<=0.2 in >={P1_FORKS_MIN}/{n} forks): {p1_status}")
    print(f"P2 (self-consistency, pooled>=0.6, per-fork>=0.8 in >={P2_FORKS_MIN}/{n}): {p2_status}")
    print(f"P3 (B surprise<0.7 in >={P3_FORKS_MIN}/{n} forks): {p3_status}")
    print()
    print(f"VERDICT: {ev['verdict_str']}")
    print()
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Write JSON rows
    # -----------------------------------------------------------------------
    out_rows_path = Path(__file__).parent / "outputs" / "exp156_rows.json"
    write_json_rows(rows, out_rows_path)

    # Write one summary JSONL row
    with out_rows_path.open("a") as fh:
        summary = {
            "exp": 156,
            "row_type": "summary",
            "seeds": seeds,
            "p1_forks_pass": ev["p1_forks_pass"],
            "p1_pass": ev["p1_pass"],
            "f1_count": ev["f1_count"],
            "f1": ev["f1"],
            "p2_forks_pass": ev["p2_forks_pass"],
            "pooled_consistency": ev["pooled_consistency"] if not math.isnan(ev["pooled_consistency"]) else None,
            "n_pooled_misloc": ev["n_pooled_misloc"],
            "p2_pass": ev["p2_pass"],
            "f2": ev["f2"],
            "p3_forks_pass": ev["p3_forks_pass"],
            "p3_pass": ev["p3_pass"],
            "f3_count": ev["f3_count"],
            "f3": ev["f3"],
            "dilution_flag": ev["dilution_flag"],
            "verdict": ev["verdict_str"],
        }
        fh.write(json.dumps(summary) + "\n")

    print(f"JSON rows written to {out_rows_path}")

    # -----------------------------------------------------------------------
    # Write verdict JSON
    # -----------------------------------------------------------------------
    arms = {
        "P1_teleport": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"B-phase misloc>=0.8 AND A-phase misloc<=0.2 in "
                f"{ev['p1_forks_pass']}/{n} forks (need {P1_FORKS_MIN}); "
                f"F1 count (B<=0.5): {ev['f1_count']}/{n}"
            ),
        },
        "P2_self_consistency": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"Per-fork consistency>=0.8 in {ev['p2_forks_pass']}/{n} forks; "
                f"pooled consistency={ev['pooled_consistency']:.4f} "
                f"on {ev['n_pooled_misloc']} misloc B-steady steps"
                if not math.isnan(ev["pooled_consistency"])
                else f"pooled consistency=nan (n_misloc={ev['n_pooled_misloc']})"
            ),
        },
        "P3_surprise_subthreshold": {
            "pass": bool(ev["p3_pass"]),
            "reason": (
                f"B-phase steady mean surprise<0.7 in {ev['p3_forks_pass']}/{n} forks; "
                f"F3 count (mean_B>=0.7): {ev['f3_count']}/{n}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp156_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp156",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "Mechanism check of Exp 155's laundering hypothesis; "
            "R-STRUCT forks only, fresh seeds 10-17."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
