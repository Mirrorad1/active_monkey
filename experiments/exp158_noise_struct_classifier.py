"""Exp 158 — the noise-vs-structural classifier: windowed error rate + lag-1
residual autocorrelation, NO flatness gate (N2 prereq build, piece 2).

Hypothesis: a slope-gate-free classifier over the creature's own correctness
stream separates OK / irreducible-NOISE / STRUCTURAL mismatch, including
across RANDOMIZED structural geometries — fixing both Exp 155 failures (the
plateau detector is class-blind and slept through structure) and Exp 156's
blindness law (no flatness condition anywhere).

Classifier (PROVIDED form; values read only from the creature's own
prediction/observation stream): rolling window W = 200 over the binary
correctness sequence, evaluated every 100 steps once the window is full.
  error_rate = 1 - mean(window)
  if error_rate < 0.05:            label = OK
  elif lag1_rho(window) > 0.3:     label = STRUCTURAL
  else:                            label = NOISE
Threshold justification (committed PRIOR data, not tuned here): Exp 155
measured lag-1 rho 0.9805 in every structural run vs -0.0053 mean under
noise (0.3 sits far from both), and error rates 0.0 (control) vs >= 0.23
(alarm regimes); fresh seeds 26-33 are used here. lag1_rho of a zero-variance
window is defined as 0.0.

Regimes (forks of mirro, spine never saved; 4000 steps = 40 chunks x 100,
exp155 step-loop semantics):
  R-CTRL:        standard world.
  R-NOISE:       NoisyCmap p_true = 0.7 (exp155 form, seed 70_000+fork_seed).
  R-STRUCT-RAND: hidden-context StructCmap with PER-FORK RANDOMIZED geometry
                 (the Exp 155 verifier's determinism flag): derangement drawn
                 from {[1,2,0], [2,0,1]} and half_period drawn from
                 {60, 100, 140}, both via np.random.default_rng(80_000 +
                 fork_seed); geometry logged per fork. All periods < W, so
                 every full window mixes contexts.
  R-PLACE-NOISE: UNGATED PROBE — Exp 157's checkerboard world (p_true 0.55
                 on (row+col)%2==0 cells, seed 90_000+fork_seed). Named
                 prediction (ungated, either outcome informative): the
                 classifier will lean STRUCTURAL here, because the walk's
                 dwell correlates errors by place — place-conditional
                 unreliability is representable structure, not i.i.d. noise.
                 Logged, not gated.

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 >= 50 incorrect trials per fork in R-NOISE and
R-STRUCT-RAND; PC2 ahat_drift < 0.05 per fork per regime; PC3 >= 30
evaluated windows per (fork, regime).

Predeclared properties and falsifiers (fractions over evaluated windows):
  P1 (control): R-CTRL OK-fraction >= 0.95 in >= 7/8 forks.
     FALSIFIER F1: OK-fraction < 0.8 in >= 4/8 forks.
  P2 (noise typed correctly): R-NOISE NOISE-fraction >= 0.9 in >= 7/8 forks
     AND pooled R-NOISE STRUCTURAL-fraction <= 0.05.
     FALSIFIER F2: NOISE-fraction <= 0.6 in >= 4/8 forks OR pooled R-NOISE
     STRUCTURAL-fraction > 0.2 (false-structure alarm, the dangerous error).
  P3 (structure typed correctly, geometry-robust): R-STRUCT-RAND
     STRUCTURAL-fraction >= 0.9 in >= 7/8 forks (forks span the randomized
     geometries). FALSIFIER F3: STRUCTURAL-fraction <= 0.6 in >= 4/8 forks.
  VERDICT: POSITIVE iff P1, P2, P3 all pass. NEGATIVE iff any falsifier
  fires. Otherwise MIXED. "Not a falsifier" never counts toward POSITIVE.

Ungated diagnostics: full label distribution per (fork, regime) including
R-PLACE-NOISE; per-fork randomized geometry (derangement, half_period);
per-regime pooled mean error_rate and mean lag1_rho over evaluated windows;
detector-events equivalent NOT computed (the classifier replaces it).
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
)
from active_loop.verdict import write_verdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEEDS = list(range(26, 34))   # fresh seeds — exp155 used 0-9, exp156 10-17, exp157 18-25

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# Classifier window / evaluation parameters
CLASSIFIER_WINDOW = 200       # W = 200 (same as SURPRISE_WINDOW)
EVAL_EVERY = 100              # evaluate once window is full, every 100 steps
ERROR_RATE_OK_THRESH = 0.05   # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3  # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Regime seed offsets
NOISE_SEED_OFFSET = 70_000        # R-NOISE NoisyCmap rng: seed 70_000+fork_seed
STRUCT_RAND_SEED_OFFSET = 80_000  # R-STRUCT-RAND geometry rng: seed 80_000+fork_seed
PLACE_NOISE_SEED_OFFSET = 90_000  # R-PLACE-NOISE PlaceNoiseCmap rng: seed 90_000+fork_seed

# R-NOISE cmap parameters
P_TRUE_NOISE = 0.7

# R-PLACE-NOISE cmap parameters
P_TRUE_PLACE = 0.55

# R-STRUCT-RAND geometry options
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]
HALF_PERIOD_OPTIONS = [60, 100, 140]

# Precondition thresholds
PRECONDITION_MIN_INCORRECT = 50   # PC1: >= 50 incorrect per fork in alarm regimes
PRECONDITION_AHAT_DRIFT = 0.05    # PC2: max abs A_hat change < 0.05
PRECONDITION_MIN_WINDOWS = 30     # PC3: >= 30 evaluated windows per (fork, regime)

# Verdict thresholds
P1_CTRL_OK_FORKS_MIN = 7          # >= 7/8 forks with OK-fraction >= 0.95
P1_CTRL_OK_FRAC_THRESH = 0.95
F1_CTRL_OK_FORKS_MAX = 4          # F1: OK-fraction < 0.8 in >= 4/8 forks
F1_CTRL_OK_FRAC_LOW = 0.8

P2_NOISE_NOISE_FORKS_MIN = 7      # >= 7/8 forks with NOISE-fraction >= 0.9
P2_NOISE_NOISE_FRAC_THRESH = 0.9
P2_POOLED_STRUCT_FRAC_MAX = 0.05  # pooled R-NOISE STRUCTURAL-fraction <= 0.05
F2_NOISE_FORKS_MAX = 4            # F2a: NOISE-fraction <= 0.6 in >= 4/8 forks
F2_NOISE_FRAC_LOW = 0.6
F2_POOLED_STRUCT_FRAC_HIGH = 0.2  # F2b: pooled R-NOISE STRUCTURAL-fraction > 0.2

P3_STRUCT_STRUCT_FORKS_MIN = 7    # >= 7/8 forks with STRUCTURAL-fraction >= 0.9
P3_STRUCT_STRUCT_FRAC_THRESH = 0.9
F3_STRUCT_FORKS_MAX = 4           # F3: STRUCTURAL-fraction <= 0.6 in >= 4/8 forks
F3_STRUCT_FRAC_LOW = 0.6

ALL_REGIMES = ("R-CTRL", "R-NOISE", "R-STRUCT-RAND", "R-PLACE-NOISE")
SMOKE_REGIMES = ("R-CTRL", "R-STRUCT-RAND")


# ---------------------------------------------------------------------------
# NoisyCmap — verbatim from exp155
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
# StructCmap — verbatim from exp155
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every half_period steps.

    Context A = base cmap (standard).
    Context B = derangement applied to cell->color lookup.

    The step counter is maintained externally; caller must set self.current_step
    before each __getitem__ call.
    """

    def __init__(self, base_cmap, n_colors, derangement, half_period=100):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.derangement = list(derangement)  # sigma[old_color] = new_color
        self.half_period = int(half_period)
        self.current_step = 0  # caller sets before __getitem__

    def __getitem__(self, s):
        ctx = (self.current_step // self.half_period) % 2
        true_color = self.base[s]
        if ctx == 0:
            return true_color
        else:
            return self.derangement[true_color]

    def __len__(self):
        return len(self.base)


# ---------------------------------------------------------------------------
# make_derangement — verbatim from exp155
# ---------------------------------------------------------------------------

def make_derangement(n_colors: int) -> list[int]:
    """Return a fixed derangement (no fixed points) of range(n_colors).

    For n_colors=3 returns [1, 2, 0].
    For general n: cyclic shift by 1 is always a derangement.
    """
    return [(i + 1) % n_colors for i in range(n_colors)]


# ---------------------------------------------------------------------------
# PlaceNoiseCmap — verbatim from exp157
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
            if self.rng.random() < self.p_true:
                return true
            others = [c for c in range(self.n_colors) if c != true]
            return int(self.rng.choice(others))
        else:
            return true

    def __len__(self) -> int:
        return len(self.base)


# ---------------------------------------------------------------------------
# lag1_rho — verbatim from exp155
# ---------------------------------------------------------------------------

def lag1_rho(seq: np.ndarray) -> float:
    """Lag-1 Pearson autocorrelation of a 1-D array.

    Returns 0.0 when denom < 1e-300 (zero-variance window).
    """
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
# Classifier: evaluate a correctness window and return label + diagnostics
# ---------------------------------------------------------------------------

def classify_window(window: np.ndarray) -> tuple[str, float, float]:
    """Classify a correctness window.

    Returns (label, error_rate, rho) where:
      label: "OK" | "STRUCTURAL" | "NOISE"
      error_rate: 1 - mean(window)
      rho: lag1_rho(window) — computed for every evaluated window regardless
           of label (diagnostic value even for OK-labeled windows).

    lag1_rho of a zero-variance window returns 0.0 (the lag1_rho function
    above already returns 0.0 when denom < 1e-300).
    """
    error_rate = 1.0 - float(window.mean())
    rho = lag1_rho(window)
    if error_rate < ERROR_RATE_OK_THRESH:
        label = "OK"
    elif rho > LAG1_RHO_STRUCT_THRESH:
        label = "STRUCTURAL"
    else:
        label = "NOISE"
    return label, error_rate, rho


# ---------------------------------------------------------------------------
# Run one fork in one regime — step loop semantics verbatim from exp155
# ---------------------------------------------------------------------------

def run_fork_regime(
    mirro: Creature,
    fork_seed: int,
    regime: str,
    base_cmap: list,
    n_colors: int,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one fresh fork of mirro in one regime.

    Returns aggregates and per-window records for evaluation.

    Step loop replicates exp155 run_fork_regime semantics EXACTLY:
      - A_hat pre-step
      - pred_probs = A_hat @ qs
      - o_hat = argmax(pred_probs)
      - obs from cmap_obj
      - correct_t = (o_hat == obs)
      - surprise = -ln(A_hat[obs,:] @ qs) pre-update
      - Bayes update
      - pA Dirichlet count update
      - value accumulation (Exp 26 mechanism)
      - random action from per-chunk rng
        (chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF)
      - move
      - advance qs through B
      - age += 1
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW, (
        f"CLASSIFIER_WINDOW={CLASSIFIER_WINDOW} must equal SURPRISE_WINDOW={SURPRISE_WINDOW}"
    )

    fork_name = f"exp158_s{fork_seed}_{regime}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Build regime-specific cmap ---
    geometry_info: dict = {}  # filled for R-STRUCT-RAND

    if regime == "R-CTRL":
        cmap_obj = base_cmap  # plain list
    elif regime == "R-NOISE":
        noise_seed = NOISE_SEED_OFFSET + fork_seed
        cmap_obj = NoisyCmap(base_cmap, n_colors, P_TRUE_NOISE, seed=noise_seed)
    elif regime == "R-STRUCT-RAND":
        # Per-fork randomized geometry: derangement and half_period via seeded rng
        geom_rng = np.random.default_rng(STRUCT_RAND_SEED_OFFSET + fork_seed)
        derangement = DERANGEMENT_OPTIONS[int(geom_rng.integers(0, len(DERANGEMENT_OPTIONS)))]
        half_period = int(HALF_PERIOD_OPTIONS[int(geom_rng.integers(0, len(HALF_PERIOD_OPTIONS)))])
        cmap_obj = StructCmap(base_cmap, n_colors, derangement, half_period=half_period)
        geometry_info = {"derangement": derangement, "half_period": half_period}
    elif regime == "R-PLACE-NOISE":
        place_seed = PLACE_NOISE_SEED_OFFSET + fork_seed
        cmap_obj = PlaceNoiseCmap(
            base_cmap=base_cmap,
            n_colors=n_colors,
            rows=c.world.rows,
            cols=c.world.cols,
            p_true=P_TRUE_PLACE,
            seed=place_seed,
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)

    n_total = n_chunks * chunk_size

    # A_hat at start for drift check
    A_hat_start = c._A_hat().copy()

    # --- Classifier state: rolling correctness window ---
    correct_window = _deque(maxlen=CLASSIFIER_WINDOW)

    # --- Per-window records ---
    window_labels: list[str] = []
    window_error_rates: list[float] = []
    window_rhos: list[float] = []

    # --- Accumulators for preconditions ---
    n_incorrect = 0

    # --- Step loop ---
    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        # Deterministic per-chunk seed derived from fork_seed and chunk index
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(chunk_size):
            t = global_step

            # Update step counter for R-STRUCT-RAND before reading cmap
            if regime == "R-STRUCT-RAND":
                cmap_obj.current_step = t

            # --- Compute A_hat (pre-step) ---
            A_hat = c._A_hat()   # shape (n_colors, n_cells)

            # --- Predicted observation and confidence (BEFORE update) ---
            pred_probs = A_hat @ c.qs   # shape (n_colors,)
            o_hat = int(np.argmax(pred_probs))

            # --- Observe (from regime-specific cmap) ---
            obs = int(cmap_obj[c.true_pos])

            # --- Correctness ---
            correct_t = 1 if (o_hat == obs) else 0
            if correct_t == 0:
                n_incorrect += 1

            # --- Append to rolling window ---
            correct_window.append(correct_t)

            # --- Evaluate classifier at every EVAL_EVERY steps once window full ---
            if (t + 1) % EVAL_EVERY == 0 and len(correct_window) == CLASSIFIER_WINDOW:
                win_arr = np.array(correct_window, dtype=np.float64)
                label, err_rate, rho = classify_window(win_arr)
                window_labels.append(label)
                window_error_rates.append(err_rate)
                window_rhos.append(rho)

            # --- Surprise (for trajectory consistency; not used in classifier) ---
            p_o = float(A_hat[obs, :] @ c.qs)
            _ = -math.log(p_o + eps)  # computed but not stored

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
            global_step += 1

    # A_hat at end for drift check
    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    # --- Per-(fork, regime) aggregates ---
    n_windows = len(window_labels)
    n_ok = window_labels.count("OK")
    n_noise = window_labels.count("NOISE")
    n_structural = window_labels.count("STRUCTURAL")

    frac_ok = n_ok / n_windows if n_windows > 0 else float("nan")
    frac_noise = n_noise / n_windows if n_windows > 0 else float("nan")
    frac_structural = n_structural / n_windows if n_windows > 0 else float("nan")

    mean_error_rate = float(np.mean(window_error_rates)) if window_error_rates else float("nan")
    mean_rho = float(np.mean(window_rhos)) if window_rhos else float("nan")

    result: dict = {
        "fork_seed": fork_seed,
        "regime": regime,
        "n_total": n_total,
        "n_incorrect": n_incorrect,
        "ahat_drift": ahat_drift,
        "n_windows": n_windows,
        "n_ok": n_ok,
        "n_noise": n_noise,
        "n_structural": n_structural,
        "frac_ok": frac_ok,
        "frac_noise": frac_noise,
        "frac_structural": frac_structural,
        "mean_error_rate": mean_error_rate,
        "mean_rho": mean_rho,
    }
    if geometry_info:
        result["geometry"] = geometry_info
    return result


# ---------------------------------------------------------------------------
# Run all seeds for a given set of regimes
# ---------------------------------------------------------------------------

def run_all(
    seeds: list[int],
    mirro: Creature,
    base_cmap: list,
    n_colors: int,
    regimes: tuple[str, ...],
    n_chunks: int = N_CHUNKS,
    verbose: bool = True,
) -> list[dict]:
    """Run all (seed, regime) pairs; return list of result dicts."""
    rows: list[dict] = []
    for seed in seeds:
        for regime in regimes:
            if verbose:
                print(f"  Running seed={seed} regime={regime} ...", flush=True)
            r = run_fork_regime(
                mirro=mirro,
                fork_seed=seed,
                regime=regime,
                base_cmap=base_cmap,
                n_colors=n_colors,
                n_chunks=n_chunks,
                chunk_size=CHUNK_SIZE,
            )
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# Print per-fork table
# ---------------------------------------------------------------------------

def print_fork_table(rows: list[dict]) -> None:
    hdr = (
        f"{'seed':>5}  "
        f"{'regime':>14}  "
        f"{'n_inc':>6}  "
        f"{'n_win':>6}  "
        f"{'frac_ok':>8}  "
        f"{'frac_no':>8}  "
        f"{'frac_st':>8}  "
        f"{'mean_err':>9}  "
        f"{'mean_rho':>9}  "
        f"{'drift':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        fo = f"{r['frac_ok']:.4f}" if not (isinstance(r['frac_ok'], float) and math.isnan(r['frac_ok'])) else "   nan"
        fn = f"{r['frac_noise']:.4f}" if not (isinstance(r['frac_noise'], float) and math.isnan(r['frac_noise'])) else "   nan"
        fs = f"{r['frac_structural']:.4f}" if not (isinstance(r['frac_structural'], float) and math.isnan(r['frac_structural'])) else "   nan"
        me = f"{r['mean_error_rate']:.4f}" if not (isinstance(r['mean_error_rate'], float) and math.isnan(r['mean_error_rate'])) else "    nan"
        mr = f"{r['mean_rho']:.4f}" if not (isinstance(r['mean_rho'], float) and math.isnan(r['mean_rho'])) else "    nan"
        geom = ""
        if "geometry" in r:
            g = r["geometry"]
            geom = f"  [derang={g['derangement']} hp={g['half_period']}]"
        print(
            f"{r['fork_seed']:>5}  "
            f"{r['regime']:>14}  "
            f"{r['n_incorrect']:>6}  "
            f"{r['n_windows']:>6}  "
            f"{fo:>8}  "
            f"{fn:>8}  "
            f"{fs:>8}  "
            f"{me:>9}  "
            f"{mr:>9}  "
            f"{r['ahat_drift']:>8.4f}"
            f"{geom}"
        )


# ---------------------------------------------------------------------------
# Precondition check
# ---------------------------------------------------------------------------

def check_preconditions(rows: list[dict]) -> tuple[bool, list[str]]:
    """Check PC1, PC2, PC3.  Returns (all_pass, list_of_failures)."""
    failures: list[str] = []

    for r in rows:
        seed = r["fork_seed"]
        regime = r["regime"]

        # PC1: >= 50 incorrect in alarm regimes
        if regime in ("R-NOISE", "R-STRUCT-RAND"):
            if r["n_incorrect"] < PRECONDITION_MIN_INCORRECT:
                failures.append(
                    f"PC1 FAIL: seed={seed} {regime} n_incorrect={r['n_incorrect']} "
                    f"< {PRECONDITION_MIN_INCORRECT}"
                )

        # PC2: ahat_drift < 0.05
        if r["ahat_drift"] >= PRECONDITION_AHAT_DRIFT:
            failures.append(
                f"PC2 FAIL: seed={seed} {regime} ahat_drift={r['ahat_drift']:.4f} "
                f">= {PRECONDITION_AHAT_DRIFT}"
            )

        # PC3: >= 30 evaluated windows per (fork, regime)
        if regime in ("R-CTRL", "R-NOISE", "R-STRUCT-RAND"):
            if r["n_windows"] < PRECONDITION_MIN_WINDOWS:
                failures.append(
                    f"PC3 FAIL: seed={seed} {regime} n_windows={r['n_windows']} "
                    f"< {PRECONDITION_MIN_WINDOWS}"
                )

    all_pass = len(failures) == 0
    return all_pass, failures


# ---------------------------------------------------------------------------
# Evaluate predeclared outcome map (P1/F1, P2/F2, P3/F3)
# ---------------------------------------------------------------------------

def evaluate(rows: list[dict]) -> dict:
    """Evaluate P1, P2, P3 and their falsifiers; return evaluation dict."""
    ctrl_rows = [r for r in rows if r["regime"] == "R-CTRL"]
    noise_rows = [r for r in rows if r["regime"] == "R-NOISE"]
    struct_rows = [r for r in rows if r["regime"] == "R-STRUCT-RAND"]

    n_forks = len(SEEDS)  # 8

    # --- P1 / F1: R-CTRL OK-fraction ---
    # P1: OK-fraction >= 0.95 in >= 7/8 forks
    p1_forks_ok = sum(
        1 for r in ctrl_rows
        if not (math.isnan(r["frac_ok"])) and r["frac_ok"] >= P1_CTRL_OK_FRAC_THRESH
    )
    p1_pass = p1_forks_ok >= P1_CTRL_OK_FORKS_MIN

    # F1: OK-fraction < 0.8 in >= 4/8 forks
    f1_forks_low = sum(
        1 for r in ctrl_rows
        if not math.isnan(r["frac_ok"]) and r["frac_ok"] < F1_CTRL_OK_FRAC_LOW
    )
    f1 = f1_forks_low >= F1_CTRL_OK_FORKS_MAX

    # --- P2 / F2: R-NOISE NOISE-fraction ---
    # P2: NOISE-fraction >= 0.9 in >= 7/8 forks AND pooled STRUCTURAL-fraction <= 0.05
    p2_forks_noise = sum(
        1 for r in noise_rows
        if not math.isnan(r["frac_noise"]) and r["frac_noise"] >= P2_NOISE_NOISE_FRAC_THRESH
    )
    p2_forks_pass = p2_forks_noise >= P2_NOISE_NOISE_FORKS_MIN

    # Pooled R-NOISE STRUCTURAL-fraction = total structural windows / total windows
    total_noise_windows = sum(r["n_windows"] for r in noise_rows)
    total_noise_structural = sum(r["n_structural"] for r in noise_rows)
    pooled_noise_struct_frac = (
        total_noise_structural / total_noise_windows
        if total_noise_windows > 0 else float("nan")
    )
    p2_pooled_pass = (
        not math.isnan(pooled_noise_struct_frac)
        and pooled_noise_struct_frac <= P2_POOLED_STRUCT_FRAC_MAX
    )
    p2_pass = p2_forks_pass and p2_pooled_pass

    # F2a: NOISE-fraction <= 0.6 in >= 4/8 forks
    f2_forks_low = sum(
        1 for r in noise_rows
        if not math.isnan(r["frac_noise"]) and r["frac_noise"] <= F2_NOISE_FRAC_LOW
    )
    f2a = f2_forks_low >= F2_NOISE_FORKS_MAX

    # F2b: pooled R-NOISE STRUCTURAL-fraction > 0.2
    f2b = (
        not math.isnan(pooled_noise_struct_frac)
        and pooled_noise_struct_frac > F2_POOLED_STRUCT_FRAC_HIGH
    )
    f2 = f2a or f2b

    # --- P3 / F3: R-STRUCT-RAND STRUCTURAL-fraction ---
    # P3: STRUCTURAL-fraction >= 0.9 in >= 7/8 forks
    p3_forks_struct = sum(
        1 for r in struct_rows
        if not math.isnan(r["frac_structural"]) and r["frac_structural"] >= P3_STRUCT_STRUCT_FRAC_THRESH
    )
    p3_pass = p3_forks_struct >= P3_STRUCT_STRUCT_FORKS_MIN

    # F3: STRUCTURAL-fraction <= 0.6 in >= 4/8 forks
    f3_forks_low = sum(
        1 for r in struct_rows
        if not math.isnan(r["frac_structural"]) and r["frac_structural"] <= F3_STRUCT_FRAC_LOW
    )
    f3 = f3_forks_low >= F3_STRUCT_FORKS_MAX

    # --- Verdict ---
    positive = p1_pass and p2_pass and p3_pass
    negative = f1 or f2 or f3
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    # --- Per-regime pooled diagnostics ---
    def _pooled_mean(rows_list: list[dict], field: str) -> float:
        vals = [r[field] for r in rows_list if not math.isnan(r[field])]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        # R-CTRL
        "p1_forks_ok": p1_forks_ok,
        "p1_pass": p1_pass,
        "f1_forks_low": f1_forks_low,
        "f1": f1,
        "ctrl_mean_error_rate": _pooled_mean(ctrl_rows, "mean_error_rate"),
        "ctrl_mean_rho": _pooled_mean(ctrl_rows, "mean_rho"),
        # R-NOISE
        "p2_forks_noise": p2_forks_noise,
        "p2_forks_pass": p2_forks_pass,
        "pooled_noise_struct_frac": pooled_noise_struct_frac,
        "p2_pooled_pass": p2_pooled_pass,
        "p2_pass": p2_pass,
        "f2a": f2a,
        "f2b": f2b,
        "f2": f2,
        "f2_forks_low": f2_forks_low,
        "noise_mean_error_rate": _pooled_mean(noise_rows, "mean_error_rate"),
        "noise_mean_rho": _pooled_mean(noise_rows, "mean_rho"),
        # R-STRUCT-RAND
        "p3_forks_struct": p3_forks_struct,
        "p3_pass": p3_pass,
        "f3_forks_low": f3_forks_low,
        "f3": f3,
        "struct_mean_error_rate": _pooled_mean(struct_rows, "mean_error_rate"),
        "struct_mean_rho": _pooled_mean(struct_rows, "mean_rho"),
        # Verdict
        "verdict_str": verdict_str,
        "n_forks": n_forks,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + summary
# ---------------------------------------------------------------------------

def write_json_rows(rows: list[dict], path: Path, ev: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            row: dict = {
                "exp": 158,
                "fork_seed": r["fork_seed"],
                "regime": r["regime"],
                "n_total": r["n_total"],
                "n_incorrect": r["n_incorrect"],
                "ahat_drift": r["ahat_drift"],
                "n_windows": r["n_windows"],
                "n_ok": r["n_ok"],
                "n_noise": r["n_noise"],
                "n_structural": r["n_structural"],
                "frac_ok": r["frac_ok"] if not math.isnan(r["frac_ok"]) else None,
                "frac_noise": r["frac_noise"] if not math.isnan(r["frac_noise"]) else None,
                "frac_structural": r["frac_structural"] if not math.isnan(r["frac_structural"]) else None,
                "mean_error_rate": r["mean_error_rate"] if not math.isnan(r["mean_error_rate"]) else None,
                "mean_rho": r["mean_rho"] if not math.isnan(r["mean_rho"]) else None,
            }
            if "geometry" in r:
                row["geometry"] = r["geometry"]
            fh.write(json.dumps(row) + "\n")

        if ev is not None:
            summary: dict = {
                "exp": 158,
                "row_type": "summary",
                "p1_forks_ok": ev["p1_forks_ok"],
                "p1_pass": ev["p1_pass"],
                "f1_forks_low": ev["f1_forks_low"],
                "f1": ev["f1"],
                "p2_forks_noise": ev["p2_forks_noise"],
                "p2_forks_pass": ev["p2_forks_pass"],
                "pooled_noise_struct_frac": (
                    ev["pooled_noise_struct_frac"]
                    if not math.isnan(ev["pooled_noise_struct_frac"]) else None
                ),
                "p2_pooled_pass": ev["p2_pooled_pass"],
                "p2_pass": ev["p2_pass"],
                "f2a": ev["f2a"],
                "f2b": ev["f2b"],
                "f2": ev["f2"],
                "p3_forks_struct": ev["p3_forks_struct"],
                "p3_pass": ev["p3_pass"],
                "f3_forks_low": ev["f3_forks_low"],
                "f3": ev["f3"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 158 — noise-vs-structural classifier (windowed error rate + lag-1 rho)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seeds=[26] only, full 40 chunks, "
            "regimes R-CTRL and R-STRUCT-RAND only, no verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [26] if smoke else SEEDS
    regimes = SMOKE_REGIMES if smoke else ALL_REGIMES

    print("=" * 80)
    print("Exp 158 — noise-vs-structural classifier (windowed error rate + lag-1 rho)")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW={CLASSIFIER_WINDOW}  "
          f"[must be equal; asserted]")
    assert CLASSIFIER_WINDOW == SURPRISE_WINDOW
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"P_TRUE_NOISE={P_TRUE_NOISE}  P_TRUE_PLACE={P_TRUE_PLACE}")
    print(f"NOISE_SEED_OFFSET={NOISE_SEED_OFFSET}  "
          f"STRUCT_RAND_SEED_OFFSET={STRUCT_RAND_SEED_OFFSET}  "
          f"PLACE_NOISE_SEED_OFFSET={PLACE_NOISE_SEED_OFFSET}")
    print(f"Seeds: {seeds}  Regimes: {regimes}  N_CHUNKS={N_CHUNKS}  "
          f"CHUNK_SIZE={CHUNK_SIZE}  N_STEPS={N_CHUNKS * CHUNK_SIZE}")
    print()

    # --- Load mirro spine (read-only) ---
    mirro = Creature.load("creature/state/mirro")
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}, "
          f"n_colors={mirro.world.n_colors}, n_cells={mirro.world.n_cells}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors
    print(f"Base cmap: {base_cmap}")
    print()

    # --- Run ---
    rows = run_all(
        seeds=seeds,
        mirro=mirro,
        base_cmap=base_cmap,
        n_colors=n_colors,
        regimes=regimes,
        n_chunks=N_CHUNKS,
        verbose=True,
    )
    print()

    # --- Per-fork table ---
    print("Per-fork table:")
    print_fork_table(rows)
    print()

    # --- Ungated diagnostics (R-PLACE-NOISE label distribution) ---
    place_rows = [r for r in rows if r["regime"] == "R-PLACE-NOISE"]
    if place_rows:
        print("Ungated diagnostics (R-PLACE-NOISE label distribution):")
        for r in place_rows:
            print(f"  seed={r['fork_seed']}  n_win={r['n_windows']}  "
                  f"frac_ok={r['frac_ok']:.4f}  frac_noise={r['frac_noise']:.4f}  "
                  f"frac_struct={r['frac_structural']:.4f}  "
                  f"mean_err={r['mean_error_rate']:.4f}  mean_rho={r['mean_rho']:.4f}")
        print()

    # --- Smoke exit ---
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # --- Precondition check ---
    pc_pass, pc_failures = check_preconditions(rows)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp158_rows.json"
        write_json_rows(rows, out_rows_path, ev=None)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # --- Evaluate predeclared outcome map ---
    ev = evaluate(rows)
    n = ev["n_forks"]

    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    # P1 detail
    ctrl_rows_ev = [r for r in rows if r["regime"] == "R-CTRL"]
    ctrl_ok_fracs = [f"{r['frac_ok']:.4f}" if not math.isnan(r['frac_ok']) else "nan"
                     for r in ctrl_rows_ev]
    print(f"P1 (R-CTRL OK-fraction >= {P1_CTRL_OK_FRAC_THRESH}):")
    print(f"  frac_ok per fork: [{', '.join(ctrl_ok_fracs)}]")
    print(f"  Forks with frac_ok >= {P1_CTRL_OK_FRAC_THRESH}: "
          f"{ev['p1_forks_ok']}/{n} (need >={P1_CTRL_OK_FORKS_MIN} for pass)")
    print(f"  F1 check: forks with frac_ok < {F1_CTRL_OK_FRAC_LOW}: "
          f"{ev['f1_forks_low']}/{n} (fires if >={F1_CTRL_OK_FORKS_MAX})")
    print(f"  Pooled mean_error_rate={ev['ctrl_mean_error_rate']:.4f}  "
          f"mean_rho={ev['ctrl_mean_rho']:.4f}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED")
    print(f"  => P1: {p1_status}")
    print()

    # P2 detail
    noise_rows_ev = [r for r in rows if r["regime"] == "R-NOISE"]
    noise_noise_fracs = [f"{r['frac_noise']:.4f}" if not math.isnan(r['frac_noise']) else "nan"
                         for r in noise_rows_ev]
    noise_struct_fracs = [f"{r['frac_structural']:.4f}" if not math.isnan(r['frac_structural']) else "nan"
                          for r in noise_rows_ev]
    pnsf = (f"{ev['pooled_noise_struct_frac']:.4f}"
            if not math.isnan(ev["pooled_noise_struct_frac"]) else "nan")
    print(f"P2 (R-NOISE NOISE-fraction >= {P2_NOISE_NOISE_FRAC_THRESH}):")
    print(f"  frac_noise per fork: [{', '.join(noise_noise_fracs)}]")
    print(f"  frac_structural per fork: [{', '.join(noise_struct_fracs)}]")
    print(f"  Forks with frac_noise >= {P2_NOISE_NOISE_FRAC_THRESH}: "
          f"{ev['p2_forks_noise']}/{n} (need >={P2_NOISE_NOISE_FORKS_MIN} for pass)")
    print(f"  Pooled R-NOISE STRUCTURAL-fraction: {pnsf} "
          f"(need <={P2_POOLED_STRUCT_FRAC_MAX} for pass)")
    print(f"  F2a: forks with frac_noise <= {F2_NOISE_FRAC_LOW}: "
          f"{ev['f2_forks_low']}/{n} (fires if >={F2_NOISE_FORKS_MAX})")
    print(f"  F2b: pooled struct frac > {F2_POOLED_STRUCT_FRAC_HIGH}: "
          f"{'YES' if ev['f2b'] else 'no'}")
    print(f"  Pooled mean_error_rate={ev['noise_mean_error_rate']:.4f}  "
          f"mean_rho={ev['noise_mean_rho']:.4f}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED")
    print(f"  => P2: {p2_status}")
    print()

    # P3 detail
    struct_rows_ev = [r for r in rows if r["regime"] == "R-STRUCT-RAND"]
    struct_struct_fracs = [f"{r['frac_structural']:.4f}" if not math.isnan(r['frac_structural']) else "nan"
                           for r in struct_rows_ev]
    geom_strs = [
        f"s{r['fork_seed']}:[{r['geometry']['derangement']},hp={r['geometry']['half_period']}]"
        if "geometry" in r else f"s{r['fork_seed']}:?"
        for r in struct_rows_ev
    ]
    print(f"P3 (R-STRUCT-RAND STRUCTURAL-fraction >= {P3_STRUCT_STRUCT_FRAC_THRESH}):")
    print(f"  Randomized geometries: {', '.join(geom_strs)}")
    print(f"  frac_structural per fork: [{', '.join(struct_struct_fracs)}]")
    print(f"  Forks with frac_structural >= {P3_STRUCT_STRUCT_FRAC_THRESH}: "
          f"{ev['p3_forks_struct']}/{n} (need >={P3_STRUCT_STRUCT_FORKS_MIN} for pass)")
    print(f"  F3: forks with frac_structural <= {F3_STRUCT_FRAC_LOW}: "
          f"{ev['f3_forks_low']}/{n} (fires if >={F3_STRUCT_FORKS_MAX})")
    print(f"  Pooled mean_error_rate={ev['struct_mean_error_rate']:.4f}  "
          f"mean_rho={ev['struct_mean_rho']:.4f}")
    p3_status = "PASS" if ev["p3_pass"] else ("FALSIFIER F3" if ev["f3"] else "MIXED")
    print(f"  => P3: {p3_status}")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (R-CTRL OK-fraction >= {P1_CTRL_OK_FRAC_THRESH} in "
          f">={P1_CTRL_OK_FORKS_MIN}/{n} forks): {p1_status}")
    print(f"P2 (R-NOISE NOISE-fraction >= {P2_NOISE_NOISE_FRAC_THRESH} in "
          f">={P2_NOISE_NOISE_FORKS_MIN}/{n} forks AND pooled STRUCTURAL <= "
          f"{P2_POOLED_STRUCT_FRAC_MAX}): {p2_status}")
    print(f"P3 (R-STRUCT-RAND STRUCTURAL-fraction >= {P3_STRUCT_STRUCT_FRAC_THRESH} in "
          f">={P3_STRUCT_STRUCT_FORKS_MIN}/{n} forks, geometry-robust): {p3_status}")
    print()
    print(f"VERDICT: {ev['verdict_str']}")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp158_rows.json"
    write_json_rows(rows, out_rows_path, ev=ev)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    arms = {
        "P1_control_ok": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"R-CTRL frac_ok >= {P1_CTRL_OK_FRAC_THRESH} in "
                f"{ev['p1_forks_ok']}/{n} forks (need {P1_CTRL_OK_FORKS_MIN}); "
                f"F1 forks_low={ev['f1_forks_low']}/{n}"
            ),
        },
        "P2_noise_typed": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"R-NOISE frac_noise >= {P2_NOISE_NOISE_FRAC_THRESH} in "
                f"{ev['p2_forks_noise']}/{n} forks (need {P2_NOISE_NOISE_FORKS_MIN}); "
                f"pooled struct frac={pnsf} (need <={P2_POOLED_STRUCT_FRAC_MAX}); "
                f"F2a forks_low={ev['f2_forks_low']}/{n}; F2b={ev['f2b']}"
            ),
        },
        "P3_structure_typed_geometry_robust": {
            "pass": bool(ev["p3_pass"]),
            "reason": (
                f"R-STRUCT-RAND frac_structural >= {P3_STRUCT_STRUCT_FRAC_THRESH} in "
                f"{ev['p3_forks_struct']}/{n} forks (need {P3_STRUCT_STRUCT_FORKS_MIN}); "
                f"F3 forks_low={ev['f3_forks_low']}/{n}"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp158_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp158",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N2 prereq build piece 2: windowed error-rate + lag-1 rho classifier, "
            "no flatness gate. Randomized structural geometries across forks. "
            "R-PLACE-NOISE is an ungated probe (place-conditional structure). "
            "Fixes Exp 155 class-blindness and Exp 156 flatness-gate blindness."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
