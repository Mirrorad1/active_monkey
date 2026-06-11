"""Exp 155 — N2 prerequisite check on the mirro body (forks only).

Purpose: confirm two preconditions for the N2 (second-order self-model) rung
on the mirro body using fork-only instrumentation (no spine mutation):

  (P1) meta-d' > 0 — the creature's own confidence over the predictive
       distribution tracks its binary prediction accuracy; AUROC of confidence
       over correct vs incorrect trials > 0.5.

  (P2) the surprise-ceiling detector PLUS a residual-structure statistic
       separates irreducible noise (i.i.d.) from structural mismatch (hidden
       context), validating the exocortical alarm as a typed sensor.

Honest design note (carried in comments throughout): the raw ceiling detector
is class-blind (plateau detector only); the classifier = detector +
residual-structure statistic (lag-1 autocorrelation of correct_t binary
sequence).  Lag-1 rho is computed from the creature's own per-step
observations + predictions, which are derived from its internal quantities
(pA-hat and qs).  This is exocortical instrumentation reading internal signals
without modifying them.

Forks: load mirro once; for each fork seed, create fork(f"exp155_s{seed}")
FRESH from the loaded spine per regime (3 independent forks per seed, one per
regime — regimes must NOT contaminate each other).  mirro itself is never
saved, never lived.

Regimes (each: 4000 steps = 40 chunks x live(100) with deterministic
per-chunk seeds derived from the fork seed):
  R-CTRL:   standard world (mirro's cmap, no modification).
  R-NOISE:  NoisyCmap wrapper with p_true=0.7, own seeded RNG — i.i.d.
            irreducible observation noise; analytic irreducible surprise
            ≈ 0.82 nats (> 0.7-nat ceiling threshold).
  R-STRUCT: hidden-context world — cmap alternates every 100 steps between
            context A (standard cmap) and context B (derangement: a fixed
            permutation with no fixed points applied to cell->color lookup).
            Context switch driven by step counter, invisible to the creature.
            The fixed state space cannot represent the hidden context;
            structural mismatch by construction.
            NOTE: detector window = 200, context half-period = 100, so the
            window ALWAYS straddles both contexts — no flicker by design.

Falsifiers / named outcome map (predeclared):

  Preconditions (instrument-grade; any failure on CONFIRM seeds =>
  print "PRECONDITION FAILED" and halt, no verdict):
    PC1: >= 50 correct AND >= 50 incorrect trials per fork in R-NOISE and R-STRUCT.
    PC2: ahat_drift < 0.05 everywhere (stale-map premise).
    PC3: detector window length == 200 (assert from creature constants).

  P1 (meta-d' > 0): in EACH of R-NOISE and R-STRUCT:
    type2_auroc > 0.5 in >= 6/8 confirm forks AND
    pooled (all trials concatenated per regime) auroc >= 0.55.
    FALSIFIER F1: <= 4/8 forks with auroc > 0.5 in either regime,
                  OR pooled auroc <= 0.5 in either. Between bands => MIXED.

  P2a (alarm validity): detector_events >= 1 in >= 6/8 forks in R-NOISE;
    same in R-STRUCT; detector_events == 0 in >= 7/8 forks in R-CTRL.
    FALSIFIER F2a: < 4/8 firing in an alarm regime OR >= 3/8 firing in control.
    Between => MIXED.

  P2b (noise-vs-structural separation): lag1_rho(R-STRUCT) > lag1_rho(R-NOISE)
    in >= 7/8 same-seed pairs AND
    (pooled mean STRUCT rho) - (pooled mean NOISE rho) >= 0.15.
    FALSIFIER F2b: <= 5/8 pairs ordered OR separation < 0.05. Between => MIXED.

  VERDICT: POSITIVE iff P1, P2a, P2b all pass.
           NEGATIVE iff any falsifier fires.
           Otherwise MIXED.
  Print which conjuncts passed/failed explicitly, one line each,
  and a final "VERDICT: ..." line.
  "Not a falsifier" never counts toward POSITIVE.
"""
from __future__ import annotations

import json
import math
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

PILOT_SEEDS = [0, 1]
CONFIRM_SEEDS = list(range(2, 10))  # seeds 2..9

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS   # 100
CONTEXT_HALF_PERIOD = 100          # R-STRUCT: alternate every 100 steps

P_TRUE = 0.7                       # R-NOISE: true-color probability
NOISE_SEED_OFFSET = 50_000         # seeded offset for NoisyCmap RNG

# Analytic irreducible surprise for R-NOISE (p_true=0.7, n_colors=3):
# S* = 0.7*ln(1/0.7) + 0.3*(2/2)*ln(1/0.15) ≈ 0.82 nats
# (referenced in docstring for orientation; not used in any computation)

PRECONDITION_MIN_CLASS = 50        # min correct AND incorrect trials
PRECONDITION_AHAT_DRIFT = 0.05     # max abs change of A_hat columns
P1_PER_REGIME_MIN_FORKS = 6       # >=6/8 confirm forks need auroc > 0.5
P1_POOLED_AUROC_THRESH = 0.55     # pooled auroc >= 0.55
P1_FALSIFIER_AUROC_FORKS = 4      # <= 4/8 forks => F1
P1_FALSIFIER_POOLED = 0.5         # pooled <= 0.5 => F1
P2A_ALARM_MIN = 6                  # >= 6/8 firing in alarm regimes
P2A_CTRL_MAX = 7                   # >= 7/8 silent in ctrl (i.e., <=1/8 fire)
P2A_FALSIFIER_ALARM = 4            # < 4/8 firing => F2a
P2A_FALSIFIER_CTRL = 3             # >= 3/8 firing in ctrl => F2a
P2B_PAIR_MIN = 7                   # >= 7/8 pairs ordered
P2B_SEPARATION_PASS = 0.15        # pooled mean diff >= 0.15
P2B_FALSIFIER_PAIRS = 5           # <= 5/8 ordered => F2b
P2B_FALSIFIER_SEP = 0.05          # separation < 0.05 => F2b

# ---------------------------------------------------------------------------
# Build a fixed derangement of the color labels for R-STRUCT context B.
# A derangement is a permutation with no fixed points.
# We want a permutation sigma of {0, ..., n_colors-1} such that sigma[i] != i.
# For n_colors=3: the only derangements of [0,1,2] are [1,2,0] and [2,0,1].
# We choose [1, 2, 0] (cycle: 0->1->2->0) — same permutation across ALL forks.
# ---------------------------------------------------------------------------

def make_derangement(n_colors: int) -> list[int]:
    """Return a fixed derangement (no fixed points) of range(n_colors).

    For n_colors=3 returns [1, 2, 0].
    For general n: cyclic shift by 1 is always a derangement.
    """
    return [(i + 1) % n_colors for i in range(n_colors)]


# ---------------------------------------------------------------------------
# NoisyCmap — exact copy of exp132 pattern
# ---------------------------------------------------------------------------

class NoisyCmap:
    """Color map wrapper with irreducible observation noise.

    Each lookup returns the true color with probability p_true, otherwise a
    uniform draw among the other colors.  Uses its OWN seeded rng so the
    creature's action stream is untouched (trajectory determinism preserved).
    """

    def __init__(self, base_cmap, n_colors, p_true, seed):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.p_true = float(p_true)
        self.rng = np.random.default_rng(seed)

    def __getitem__(self, s):
        true = self.base[s]
        if self.rng.random() < self.p_true:
            return true
        others = [c for c in range(self.n_colors) if c != true]
        return int(self.rng.choice(others))

    def __len__(self):
        return len(self.base)


# ---------------------------------------------------------------------------
# StructCmap — alternates context A (base) and context B (derangement) every
# CONTEXT_HALF_PERIOD steps.  Step counter is driven externally (passed in).
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


# ---------------------------------------------------------------------------
# Mann-Whitney AUROC (no scipy dependency)
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
    # U = sum_{i in pos, j in neg} I(pos_i > neg_j) + 0.5*I(pos_i == neg_j)
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
# lag-1 Pearson autocorrelation of a binary sequence
# ---------------------------------------------------------------------------

def lag1_rho(seq: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a 1-D array."""
    n = len(seq)
    if n < 2:
        return float("nan")
    x = seq[:-1].astype(float)
    y = seq[1:].astype(float)
    mu_x = x.mean()
    mu_y = y.mean()
    num = ((x - mu_x) * (y - mu_y)).sum()
    denom = math.sqrt(((x - mu_x) ** 2).sum() * ((y - mu_y) ** 2).sum())
    if denom < 1e-300:
        return 0.0
    return float(num / denom)


# ---------------------------------------------------------------------------
# Run one fork in one regime — the core instrumented step loop.
# Uses live(1) per step so the internal ceiling detector accumulates correctly.
# The ceiling check fires at the end of each live() call (step count = 1),
# BUT the window needs 200 entries before it can evaluate — so for single-step
# calls we do NOT get a ceiling event from live(1) itself.  Instead we
# replicate the ceiling detection EXTERNALLY using a rolling deque, matching
# creature.py's semantics exactly.  This preserves internal behavior (pA updates,
# qs updates, rng_counter increments) while allowing per-step hooks.
#
# Per-chunk live(100) pattern is used instead: run live(chunk_seed=...) and
# insert observation hooks around each chunk by temporarily wrapping the cmap.
# But live() does not expose per-step hooks, so we implement the step loop
# directly using the creature's public API pieces to match live() semantics.
#
# Decision: replicate live()'s step semantics directly in this script (reading
# creature.py's step body and mirroring it), calling the creature's _A_hat(),
# pA update, qs update, and movement via world.move(). This is the approach
# that gives per-step access without modifying creature internals.
# The ceiling detector is then implemented in this script using the same
# window/slope logic as creature.py, evaluated at the end of each 100-step chunk
# (matching the chunk boundary at which creature.py would evaluate it).
# ---------------------------------------------------------------------------

def run_fork_regime(
    mirro: Creature,
    fork_seed: int,
    regime: str,
    base_cmap: list,
    n_colors: int,
    derangement: list,
    chunk_size: int = CHUNK_SIZE,
    n_chunks: int = N_CHUNKS,
) -> dict:
    """Run one fork (loaded fresh from mirro) in one regime.

    Returns a dict with per-step arrays and summary statistics.

    The step loop replicates live()'s semantics (reading creature.py lines 303-349):
      - Compute A_hat
      - Observe cmap[true_pos]
      - Compute surprise_t = -ln(A_hat[obs,:] @ qs)  [pre-update]
      - Compute predicted obs and confidence BEFORE the step's update
        (using current qs and A_hat, before pA is updated)
      - Bayesian belief update
      - pA Dirichlet count update
      - Value accumulation (Exp 26 mechanism)
      - Random action (from deterministic per-chunk RNG)
      - Move true_pos
      - Advance qs through B
    """
    # -----------------------------------------------------------------------
    # Sanity check: window length must be 200 (read from creature constants)
    # -----------------------------------------------------------------------
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )

    # -----------------------------------------------------------------------
    # Create fresh fork for this (seed, regime) pair
    # -----------------------------------------------------------------------
    fork_name = f"exp155_s{fork_seed}_{regime}"
    c = mirro.fork(fork_name)

    # -----------------------------------------------------------------------
    # Set up the world cmap for this regime
    # -----------------------------------------------------------------------
    from active_loop.creature import World
    from collections import deque as _deque

    n_cells = c.world.n_cells
    n_actions = 4

    if regime == "R-CTRL":
        cmap_obj = base_cmap  # plain list
    elif regime == "R-NOISE":
        noise_seed = NOISE_SEED_OFFSET + fork_seed
        cmap_obj = NoisyCmap(base_cmap, n_colors, P_TRUE, seed=noise_seed)
    elif regime == "R-STRUCT":
        cmap_obj = StructCmap(base_cmap, n_colors, derangement,
                              half_period=CONTEXT_HALF_PERIOD)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Swap the world's cmap to the regime cmap
    # We do NOT create a new World object — instead we run the step loop
    # manually using cmap_obj for observations while c.world is kept for
    # move() and transition_matrix().
    B = c.world.transition_matrix()  # (n_cells, n_cells, 4)

    # -----------------------------------------------------------------------
    # Per-step storage
    # -----------------------------------------------------------------------
    n_total = n_chunks * chunk_size
    pred_obs_arr = np.empty(n_total, dtype=np.int32)
    conf_arr = np.empty(n_total, dtype=np.float64)
    real_obs_arr = np.empty(n_total, dtype=np.int32)
    correct_arr = np.empty(n_total, dtype=np.int32)
    surprise_arr = np.empty(n_total, dtype=np.float64)

    # For ceiling detection: rolling window (matches creature.py semantics)
    surprise_win = _deque(maxlen=SURPRISE_WINDOW)
    ceiling_events = 0

    # Record A_hat at start (for drift check)
    A_hat_start = c._A_hat().copy()

    # -----------------------------------------------------------------------
    # Step loop
    # -----------------------------------------------------------------------
    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        # Deterministic per-chunk seed derived from fork_seed and chunk index
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for step_in_chunk in range(chunk_size):
            t = global_step

            # Update step counter for R-STRUCT before reading cmap
            if regime == "R-STRUCT":
                cmap_obj.current_step = t

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()  # shape (n_colors, n_cells)

            # --- Predicted observation and confidence (BEFORE update) ---
            # o_hat = argmax_c sum_s qs[s] * A_hat[c, s]
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            o_hat = int(np.argmax(pred_probs))
            conf_t = float(pred_probs.max())

            # --- Observe (from regime-specific cmap) ---
            obs = int(cmap_obj[c.true_pos])

            # --- Surprise: -ln(p(o_t)) = -ln(A_hat[obs,:] @ qs) ---
            p_o = float(A_hat[obs, :] @ c.qs)
            surprise_t = -math.log(p_o + eps)
            surprise_win.append(surprise_t)

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0

            # Store
            pred_obs_arr[t] = o_hat
            conf_arr[t] = conf_t
            real_obs_arr[t] = obs
            correct_arr[t] = correct_t
            surprise_arr[t] = surprise_t

            # --- Belief update: qs ∝ A_hat[obs,:] * qs ---
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # --- Dirichlet count update: pA[obs, :] += qs ---
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
            global_step += 1

        # End of chunk: ceiling check (mirrors creature.py live() end-of-call check)
        if len(surprise_win) == SURPRISE_WINDOW:
            win_arr = np.array(surprise_win)
            mean_s = float(win_arr.mean())
            slope = float(np.polyfit(np.arange(SURPRISE_WINDOW), win_arr, 1)[0])
            if (mean_s > CEILING_MEAN_THRESH and abs(slope) < CEILING_SLOPE_THRESH):
                ceiling_events += 1

    # A_hat at end
    A_hat_end = c._A_hat()

    # ahat_drift: max abs change of A_hat columns start vs end
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # Per-fork statistics
    n_correct = int(correct_arr.sum())
    n_incorrect = int((correct_arr == 0).sum())
    pos_conf = conf_arr[correct_arr == 1]
    neg_conf = conf_arr[correct_arr == 0]

    if n_correct >= PRECONDITION_MIN_CLASS and n_incorrect >= PRECONDITION_MIN_CLASS:
        auroc = mannwhitney_auroc(pos_conf, neg_conf)
        precondition_failed = False
    else:
        auroc = float("nan")
        precondition_failed = True

    rho = lag1_rho(correct_arr)

    return {
        "fork_seed": fork_seed,
        "regime": regime,
        "n_total": n_total,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "type2_auroc": auroc,
        "precondition_failed": precondition_failed,
        "lag1_rho": rho,
        "detector_events": ceiling_events,
        "ahat_drift": ahat_drift,
        # Full arrays — kept for pooled AUROC computation
        "_conf": conf_arr,
        "_correct": correct_arr,
    }


# ---------------------------------------------------------------------------
# Run a set of seeds and return rows
# ---------------------------------------------------------------------------

def run_seeds(
    seeds: list[int],
    mirro: Creature,
    base_cmap: list,
    n_colors: int,
    derangement: list,
    verbose: bool = True,
) -> list[dict]:
    """Run all three regimes for each seed; return list of result dicts."""
    rows = []
    for seed in seeds:
        for regime in ("R-CTRL", "R-NOISE", "R-STRUCT"):
            if verbose:
                print(f"  Running seed={seed} regime={regime} ...", flush=True)
            r = run_fork_regime(
                mirro=mirro,
                fork_seed=seed,
                regime=regime,
                base_cmap=base_cmap,
                n_colors=n_colors,
                derangement=derangement,
            )
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Print per-fork table
# ---------------------------------------------------------------------------

def print_fork_table(rows: list[dict]) -> None:
    hdr = (f"{'seed':>5} {'regime':>8} {'n_cor':>6} {'n_inc':>6} "
           f"{'auroc':>7} {'lag1rho':>8} {'det_ev':>7} {'drift':>8} {'pc_fail':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        auroc_s = f"{r['type2_auroc']:.4f}" if not math.isnan(r['type2_auroc']) else "   nan"
        drift_s = f"{r['ahat_drift']:.4f}"
        print(f"{r['fork_seed']:>5} {r['regime']:>8} {r['n_correct']:>6} {r['n_incorrect']:>6} "
              f"{auroc_s:>7} {r['lag1_rho']:>8.4f} {r['detector_events']:>7d} "
              f"{drift_s:>8} {str(r['precondition_failed']):>8}")


# ---------------------------------------------------------------------------
# Precondition check on a set of rows
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict], label: str) -> tuple[bool, list[str]]:
    """Check all preconditions; return (all_pass, list_of_failures)."""
    failures = []

    # PC3: window length == 200 (asserted in run_fork_regime too, but restate)
    if SURPRISE_WINDOW != 200:
        failures.append(f"PC3 FAIL: SURPRISE_WINDOW={SURPRISE_WINDOW} != 200")

    for r in rows:
        seed = r["fork_seed"]
        regime = r["regime"]

        if regime in ("R-NOISE", "R-STRUCT"):
            if r["n_correct"] < PRECONDITION_MIN_CLASS:
                failures.append(
                    f"PC1 FAIL: seed={seed} {regime} n_correct={r['n_correct']} < {PRECONDITION_MIN_CLASS}"
                )
            if r["n_incorrect"] < PRECONDITION_MIN_CLASS:
                failures.append(
                    f"PC1 FAIL: seed={seed} {regime} n_incorrect={r['n_incorrect']} < {PRECONDITION_MIN_CLASS}"
                )

        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: seed={seed} {regime} ahat_drift={r['ahat_drift']:.4f} >= {PRECONDITION_AHAT_DRIFT}"
            )

    all_pass = len(failures) == 0
    return all_pass, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map on CONFIRM rows
# ---------------------------------------------------------------------------

def evaluate_outcomes(confirm_rows: list[dict]) -> dict:
    """Evaluate P1, P2a, P2b on the confirm set (seeds 2..9)."""
    # Separate by regime
    noise_rows = [r for r in confirm_rows if r["regime"] == "R-NOISE"]
    struct_rows = [r for r in confirm_rows if r["regime"] == "R-STRUCT"]
    ctrl_rows  = [r for r in confirm_rows if r["regime"] == "R-CTRL"]

    n_confirm = len(CONFIRM_SEEDS)  # 8

    # --- P1: per-regime AUROC ---
    def p1_regime(rows: list[dict], regime_name: str) -> dict:
        aurocs = [r["type2_auroc"] for r in rows]
        forks_above = sum(1 for a in aurocs if (not math.isnan(a)) and a > 0.5)
        conf_lists = [r["_conf"] for r in rows]
        corr_lists = [r["_correct"] for r in rows]
        p_auroc = pooled_auroc(conf_lists, corr_lists)
        pass_forks = forks_above >= P1_PER_REGIME_MIN_FORKS
        pass_pooled = (not math.isnan(p_auroc)) and p_auroc >= P1_POOLED_AUROC_THRESH
        falsifier = (forks_above <= P1_FALSIFIER_AUROC_FORKS or
                     (not math.isnan(p_auroc) and p_auroc <= P1_FALSIFIER_POOLED))
        return {
            "regime": regime_name,
            "aurocs": aurocs,
            "forks_above_0.5": forks_above,
            "pooled_auroc": p_auroc,
            "pass_forks": pass_forks,
            "pass_pooled": pass_pooled,
            "falsifier": falsifier,
            "passes": pass_forks and pass_pooled,
        }

    p1_noise = p1_regime(noise_rows, "R-NOISE")
    p1_struct = p1_regime(struct_rows, "R-STRUCT")
    p1_passes = p1_noise["passes"] and p1_struct["passes"]
    p1_falsifier = p1_noise["falsifier"] or p1_struct["falsifier"]

    # --- P2a: alarm validity ---
    noise_fire = sum(1 for r in noise_rows if r["detector_events"] >= 1)
    struct_fire = sum(1 for r in struct_rows if r["detector_events"] >= 1)
    ctrl_fire   = sum(1 for r in ctrl_rows  if r["detector_events"] >= 1)

    p2a_noise_pass   = noise_fire >= P2A_ALARM_MIN
    p2a_struct_pass  = struct_fire >= P2A_ALARM_MIN
    ctrl_silent       = (n_confirm - ctrl_fire) >= (P2A_CTRL_MAX)  # >=7/8 silent
    p2a_passes = p2a_noise_pass and p2a_struct_pass and ctrl_silent

    p2a_falsifier = (
        noise_fire < P2A_FALSIFIER_ALARM or
        struct_fire < P2A_FALSIFIER_ALARM or
        ctrl_fire >= P2A_FALSIFIER_CTRL
    )

    # --- P2b: noise-vs-structural separation ---
    # Per same-seed pairs
    noise_by_seed  = {r["fork_seed"]: r for r in noise_rows}
    struct_by_seed = {r["fork_seed"]: r for r in struct_rows}
    pairs_ordered = 0
    pair_diffs = []
    for seed in CONFIRM_SEEDS:
        rho_noise  = noise_by_seed[seed]["lag1_rho"]
        rho_struct = struct_by_seed[seed]["lag1_rho"]
        if rho_struct > rho_noise:
            pairs_ordered += 1
        pair_diffs.append(rho_struct - rho_noise)

    mean_noise_rho  = float(np.mean([r["lag1_rho"] for r in noise_rows]))
    mean_struct_rho = float(np.mean([r["lag1_rho"] for r in struct_rows]))
    pooled_sep = mean_struct_rho - mean_noise_rho

    p2b_pairs_pass = pairs_ordered >= P2B_PAIR_MIN
    p2b_sep_pass   = pooled_sep >= P2B_SEPARATION_PASS
    p2b_passes = p2b_pairs_pass and p2b_sep_pass

    p2b_falsifier = (pairs_ordered <= P2B_FALSIFIER_PAIRS or
                     pooled_sep < P2B_FALSIFIER_SEP)

    return {
        "p1_noise": p1_noise,
        "p1_struct": p1_struct,
        "p1_passes": p1_passes,
        "p1_falsifier": p1_falsifier,
        "noise_fire": noise_fire,
        "struct_fire": struct_fire,
        "ctrl_fire": ctrl_fire,
        "p2a_noise_pass": p2a_noise_pass,
        "p2a_struct_pass": p2a_struct_pass,
        "ctrl_silent_count": n_confirm - ctrl_fire,
        "ctrl_silent_pass": ctrl_silent,
        "p2a_passes": p2a_passes,
        "p2a_falsifier": p2a_falsifier,
        "pairs_ordered": pairs_ordered,
        "pair_diffs": pair_diffs,
        "mean_noise_rho": mean_noise_rho,
        "mean_struct_rho": mean_struct_rho,
        "pooled_sep": pooled_sep,
        "p2b_pairs_pass": p2b_pairs_pass,
        "p2b_sep_pass": p2b_sep_pass,
        "p2b_passes": p2b_passes,
        "p2b_falsifier": p2b_falsifier,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("Exp 155 — N2 prerequisite check (meta-d' + structured alarm)")
    print("=" * 80)
    print()

    # --- Assert window constant (PC3) ---
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
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    derangement = make_derangement(n_colors)
    print(f"Base cmap: {base_cmap}")
    print(f"Derangement (context B color relabeling): {derangement}")
    print(f"  (sigma[i]!=i for all i: {all(derangement[i] != i for i in range(n_colors))})")
    print()

    # -----------------------------------------------------------------------
    # PILOT: seeds {0, 1}
    # -----------------------------------------------------------------------
    print("=" * 80)
    print(f"PILOT BLOCK (seeds {PILOT_SEEDS})")
    print("=" * 80)
    print()

    pilot_rows = run_seeds(PILOT_SEEDS, mirro, base_cmap, n_colors, derangement)

    print()
    print("Pilot per-fork table:")
    print_fork_table(pilot_rows)
    print()

    # Pilot: check preconditions
    pilot_pc_pass, pilot_pc_failures = check_preconditions(pilot_rows, "PILOT")
    if pilot_pc_failures:
        print("PILOT PRECONDITION FAILURES:")
        for f in pilot_pc_failures:
            print(f"  {f}")
    else:
        print("Pilot preconditions: all PASS")

    # Pilot: also check that at least one alarm regime fires at least one event
    # (validates detector mechanics produce events)
    pilot_noise = [r for r in pilot_rows if r["regime"] == "R-NOISE"]
    pilot_struct = [r for r in pilot_rows if r["regime"] == "R-STRUCT"]
    pilot_noise_fire = sum(1 for r in pilot_noise if r["detector_events"] >= 1)
    pilot_struct_fire = sum(1 for r in pilot_struct if r["detector_events"] >= 1)
    print(f"Pilot detector check: R-NOISE fires in {pilot_noise_fire}/{len(pilot_noise)} forks, "
          f"R-STRUCT fires in {pilot_struct_fire}/{len(pilot_struct)} forks")
    pilot_detector_ok = (pilot_noise_fire >= 1 or pilot_struct_fire >= 1)
    print(f"  >= 1 fire somewhere in alarm regime: {'OK' if pilot_detector_ok else 'FAIL'}")

    if not pilot_pc_pass:
        print()
        print("PRECONDITION FAILED — stopping per pilot protocol.")
        print("Report the failures above to the human before proceeding.")
        return

    if not pilot_detector_ok:
        print()
        print("PRECONDITION FAILED — detector produced zero events in ALL pilot alarm forks.")
        print("Detector may be mis-calibrated. Report to human before proceeding.")
        return

    print()
    print("Pilot preconditions PASS — proceeding to CONFIRM run.")
    print()

    # -----------------------------------------------------------------------
    # CONFIRM: seeds {2..9}
    # -----------------------------------------------------------------------
    print("=" * 80)
    print(f"CONFIRM BLOCK (seeds {CONFIRM_SEEDS})")
    print("=" * 80)
    print()

    confirm_rows = run_seeds(CONFIRM_SEEDS, mirro, base_cmap, n_colors, derangement)

    print()
    print("Confirm per-fork table:")
    print_fork_table(confirm_rows)
    print()

    # Confirm: check preconditions
    confirm_pc_pass, confirm_pc_failures = check_preconditions(confirm_rows, "CONFIRM")
    if confirm_pc_failures:
        print("PRECONDITION FAILED (confirm seeds):")
        for f in confirm_pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        # Still write JSON rows so numbers are recoverable
    else:
        print("Confirm preconditions: all PASS")

    print()

    if not confirm_pc_pass:
        # Write JSON rows without verdict
        _write_json_rows(pilot_rows, confirm_rows, out_verdict=None,
                         out_path=Path(__file__).parent / "outputs" / "exp155_rows.json")
        return

    # -----------------------------------------------------------------------
    # Evaluate predeclared outcome map
    # -----------------------------------------------------------------------
    ev = evaluate_outcomes(confirm_rows)

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP — CONFIRM seeds (2..9)")
    print("=" * 80)
    print()

    # --- P1 detail ---
    for p1_r in (ev["p1_noise"], ev["p1_struct"]):
        rname = p1_r["regime"]
        auroc_strs = " ".join(f"{a:.4f}" if not math.isnan(a) else "nan"
                              for a in p1_r["aurocs"])
        print(f"P1 {rname}: aurocs=[{auroc_strs}]")
        print(f"  forks_above_0.5={p1_r['forks_above_0.5']}/8  (need >=6 for pass, <=4 for F1)")
        print(f"  pooled_auroc={p1_r['pooled_auroc']:.4f}  (need >=0.55 for pass, <=0.5 for F1)")
        status = "PASS" if p1_r["passes"] else ("FALSIFIER F1" if p1_r["falsifier"] else "MIXED")
        print(f"  => {status}")
    print()

    p1_status = "PASS" if ev["p1_passes"] else ("FALSIFIER F1" if ev["p1_falsifier"] else "MIXED")
    print(f"P1 (meta-d'>0) overall: {p1_status}")
    print()

    # --- P2a detail ---
    n_confirm = len(CONFIRM_SEEDS)
    print(f"P2a alarm validity:")
    print(f"  R-NOISE  detector_events>=1 in {ev['noise_fire']}/8 "
          f"(need >=6 for pass, <4 for F2a): "
          f"{'PASS' if ev['p2a_noise_pass'] else 'FAIL'}")
    print(f"  R-STRUCT detector_events>=1 in {ev['struct_fire']}/8 "
          f"(need >=6 for pass, <4 for F2a): "
          f"{'PASS' if ev['p2a_struct_pass'] else 'FAIL'}")
    print(f"  R-CTRL   detector_events==0 in {ev['ctrl_silent_count']}/8 "
          f"(need >=7 silent): "
          f"{'PASS' if ev['ctrl_silent_pass'] else 'FAIL'}")
    p2a_status = "PASS" if ev["p2a_passes"] else ("FALSIFIER F2a" if ev["p2a_falsifier"] else "MIXED")
    print(f"P2a overall: {p2a_status}")
    print()

    # --- P2b detail ---
    print(f"P2b noise-vs-structural separation:")
    seed_detail = " ".join(
        f"s{s}:{d:+.4f}" for s, d in zip(CONFIRM_SEEDS, ev["pair_diffs"])
    )
    print(f"  per-seed rho(STRUCT)-rho(NOISE): {seed_detail}")
    print(f"  pairs ordered (STRUCT>NOISE): {ev['pairs_ordered']}/8 "
          f"(need >=7 for pass, <=5 for F2b)")
    print(f"  mean_noise_rho={ev['mean_noise_rho']:.4f}  "
          f"mean_struct_rho={ev['mean_struct_rho']:.4f}  "
          f"pooled_sep={ev['pooled_sep']:.4f} "
          f"(need >=0.15 for pass, <0.05 for F2b)")
    p2b_status = "PASS" if ev["p2b_passes"] else ("FALSIFIER F2b" if ev["p2b_falsifier"] else "MIXED")
    print(f"P2b overall: {p2b_status}")
    print()

    # -----------------------------------------------------------------------
    # Conjunct summary and VERDICT
    # -----------------------------------------------------------------------
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (meta-d'>0, both regimes): {p1_status}")
    print(f"P2a (alarm validity):          {p2a_status}")
    print(f"P2b (noise/struct separation): {p2b_status}")
    print()

    any_falsifier = ev["p1_falsifier"] or ev["p2a_falsifier"] or ev["p2b_falsifier"]
    all_pass = ev["p1_passes"] and ev["p2a_passes"] and ev["p2b_passes"]

    if all_pass:
        verdict_str = "POSITIVE"
    elif any_falsifier:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    print(f"VERDICT: {verdict_str}")
    print()
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Write JSON rows
    # -----------------------------------------------------------------------
    out_rows_path = Path(__file__).parent / "outputs" / "exp155_rows.json"
    _write_json_rows(pilot_rows, confirm_rows, out_verdict=ev,
                     out_path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # -----------------------------------------------------------------------
    # Write verdict JSON (T6)
    # -----------------------------------------------------------------------
    arms = {
        "P1_meta_d_prime": {
            "pass": bool(ev["p1_passes"]),
            "reason": (
                f"R-NOISE forks>{0.5}: {ev['p1_noise']['forks_above_0.5']}/8, "
                f"pooled={ev['p1_noise']['pooled_auroc']:.4f}; "
                f"R-STRUCT forks>{0.5}: {ev['p1_struct']['forks_above_0.5']}/8, "
                f"pooled={ev['p1_struct']['pooled_auroc']:.4f}"
            ),
        },
        "P2a_alarm_validity": {
            "pass": bool(ev["p2a_passes"]),
            "reason": (
                f"R-NOISE fire={ev['noise_fire']}/8, "
                f"R-STRUCT fire={ev['struct_fire']}/8, "
                f"R-CTRL silent={ev['ctrl_silent_count']}/8"
            ),
        },
        "P2b_noise_struct_sep": {
            "pass": bool(ev["p2b_passes"]),
            "reason": (
                f"pairs_ordered={ev['pairs_ordered']}/8, "
                f"pooled_sep={ev['pooled_sep']:.4f}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp155_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp155",
        arms=arms,
        verdict=verdict_str,
        halted=False,
        notes=(
            "N2 prereq check on mirro forks. "
            "Exocortical instrumentation reading internal qs and A_hat. "
            "Ceiling detector is class-blind; classifier = detector + lag1_rho."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _write_json_rows(
    pilot_rows: list[dict],
    confirm_rows: list[dict],
    out_verdict: dict | None,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in pilot_rows + confirm_rows:
            row = {
                "exp": 155,
                "phase": "pilot" if r["fork_seed"] in PILOT_SEEDS else "confirm",
                "fork_seed": r["fork_seed"],
                "regime": r["regime"],
                "n_total": r["n_total"],
                "n_correct": r["n_correct"],
                "n_incorrect": r["n_incorrect"],
                "type2_auroc": r["type2_auroc"] if not math.isnan(r["type2_auroc"]) else None,
                "precondition_failed": r["precondition_failed"],
                "lag1_rho": r["lag1_rho"],
                "detector_events": r["detector_events"],
                "ahat_drift": r["ahat_drift"],
            }
            fh.write(json.dumps(row) + "\n")
        if out_verdict is not None:
            summary = {
                "exp": 155,
                "phase": "summary",
                "p1_noise_forks_above": out_verdict["p1_noise"]["forks_above_0.5"],
                "p1_noise_pooled_auroc": out_verdict["p1_noise"]["pooled_auroc"],
                "p1_struct_forks_above": out_verdict["p1_struct"]["forks_above_0.5"],
                "p1_struct_pooled_auroc": out_verdict["p1_struct"]["pooled_auroc"],
                "p1_passes": out_verdict["p1_passes"],
                "p1_falsifier": out_verdict["p1_falsifier"],
                "noise_fire": out_verdict["noise_fire"],
                "struct_fire": out_verdict["struct_fire"],
                "ctrl_fire": out_verdict["ctrl_fire"],
                "p2a_passes": out_verdict["p2a_passes"],
                "p2a_falsifier": out_verdict["p2a_falsifier"],
                "pairs_ordered": out_verdict["pairs_ordered"],
                "mean_noise_rho": out_verdict["mean_noise_rho"],
                "mean_struct_rho": out_verdict["mean_struct_rho"],
                "pooled_sep": out_verdict["pooled_sep"],
                "p2b_passes": out_verdict["p2b_passes"],
                "p2b_falsifier": out_verdict["p2b_falsifier"],
            }
            fh.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
