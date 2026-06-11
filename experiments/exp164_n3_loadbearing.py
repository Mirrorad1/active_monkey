"""Exp 164 — N3 rung 3, the LOAD-BEARING test: N3's trust-driven rewrite of
the classifier window must (i) recover >= 60% of the diagnostic performance
the N2-only agent loses in N2-failure regimes, and (ii) NOT be reducible to
any single offline window retune — because the two failure regimes pull the
dial in OPPOSITE directions, no constant can win both; only regime-adaptive
dial authoring can.

Failure regimes (both provided; instances chosen so the dial requirements
are DISJOINT within the candidate set {200, 400, 800, 1600}):
  FR1 (slow alternation, UNSEEN instance of the Exp 162 class per the
      leakage clause): StructCmap half_period H = 1000 (Exp 162 used 400),
      derangement drawn per fork from {[1,2,0],[2,0,1]} (rng 130_000+seed).
      Detection requires W > 1000 => only W = 1600 works; W <= 800 sees
      mostly pure windows (the Exp 162 blindness).
  FR2 (micro-burst, a NEW instance of the same fixed-dial failure class):
      standard world except two 50-step deranged bursts (steps 1500-1549
      and 3000-3049, same per-fork derangement). Detection requires the
      burst NOT be diluted below the 0.05 OK bar: window error = 50/W,
      so W <= 800 works (error >= 0.0625) and W = 1600 fails (0.03125 <
      0.05 => burst invisible, labeled OK).
  CTRL (no-deception control): standard world.

Agents (post-hoc readouts over the SAME recorded per-step correctness
stream — the classifier is read-only at this rung; diagnosis, not behavior,
is N2's product here; one creature run per (fork, session), four dial
readouts; declared):
  (a) N2-only: fixed W = 200 (the built classifier as-is).
  (c_W) offline retunes: fixed W for each W in {200, 400, 800, 1600} —
      the FULL retune curve, stronger than any single oracle choice.
  (b) N2+N3: starts at W = 200; the Exp 163 forecast-scoring monitor runs
      online under the CURRENT dial (OK forecasts next-100 error < 0.15;
      NOISE forecasts |e_next - e_w| < 0.3; STRUCTURAL unscored); rolling
      trust = 1 - violations among the last 10 scored labels (with fewer
      than 3 scored labels in the rolling buffer since the last dial
      change, trust is NO-EVIDENCE and the dial is kept). Policy
      (PROVIDED): when rolling trust < 0.7, advance the dial cyclically
      200 -> 400 -> 800 -> 1600 -> 200, clear the rolling buffer, continue.
      Labels are recomputed under the current dial at each evaluation
      point (every 100 steps, when at least W steps of history exist).

Diagnostic scores per (agent, regime), over evaluation points every 100
steps (a window evaluates if >= W history exists under that agent's
current dial):
  FR1 score = fraction of evaluated windows labeled STRUCTURAL (the world
      is one hidden-context structure throughout).
  FR2 score = 0.5 * recall + 0.5 * specificity, where ground truth per
      window: STRUCTURAL iff the window contains >= 10 burst steps, OK iff
      it contains 0 burst steps (windows with 1-9 burst steps excluded);
      recall = STRUCTURAL-labeled fraction of true-STRUCTURAL windows,
      specificity = OK-labeled fraction of true-OK windows.
  CTRL score = fraction labeled OK.
  recovery(b, r) = (score_b(r) - score_a(r)) / (ctrl_a - score_a(r)),
      computed per fork (ctrl_a = agent (a)'s CTRL score).

Preconditions (instrument-grade; failure => "PRECONDITION FAILED", rows
written, no verdict): PC1 ahat_drift < 0.05 per (fork, session); PC2 >= 30
evaluation points per (fork, session) at W=200; PC3 input validity:
per-step error rate within FR1 deranged phases >= 0.9 pooled, and within
FR2 bursts >= 0.9 pooled; PC4 ctrl_a - score_a(r) >= 0.2 in both failure
regimes pooled (the gap to recover must exist — rung 1's premise).

Predeclared properties and falsifiers:
  P1 (N3 beats N2-only, the card's >= 60% recovery): recovery(b, FR1) >=
     0.6 AND recovery(b, FR2) >= 0.6, each in >= 7/8 forks.
     FALSIFIER F1: pooled recovery <= 0.3 in either failure regime.
  P2 (not reducible to config): combined(agent) = min(score(FR1),
     score(FR2)); P2 passes iff pooled combined_b - max_W pooled
     combined_c_W >= 0.15 AND combined_b > max_W combined_c_W in >= 7/8
     forks. FALSIFIER F2: some single W has score_c_W(r) >= score_b(r) -
     0.05 in BOTH failure regimes (pooled) — N3 is config, not a layer.
  VERDICT: POSITIVE (RUNG 3 PASSES — N3 owns a regime-adaptive control
  surface over theta_N2 that no constant matches) iff P1 and P2 both pass.
  NEGATIVE iff F1 or F2 fires. Otherwise MIXED. "Not a falsifier" never
  counts toward POSITIVE.

Ungated diagnostics: (b)'s dial trajectory per (fork, session) — expected:
escalate to 1600 and stay in FR1; stay at/near 200 in FR2 and CTRL (no
thrash — rung 4's territory, logged here); per-dial score table (the full
retune curve) per regime; trust traces at dial changes; scored-label
counts.
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

SEEDS = list(range(74, 82))   # fresh seeds — exp155-163 used 0-73

N_STEPS = 4000
N_CHUNKS = 40
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# Classifier parameters (verbatim from exp158/159/162/163)
CLASSIFIER_WINDOW_DEFAULT = 200     # starting W for agent (b) and agent (a)
EVAL_EVERY = 100                    # evaluate every 100 steps once window full
ERROR_RATE_OK_THRESH = 0.05         # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3       # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Dial candidate set (ordered for cyclic advance)
DIAL_CANDIDATES = [200, 400, 800, 1600]

# EWMA channel parameters (verbatim from exp157/159/163)
ALPHA = 0.05        # EWMA learning rate
EWMA_INIT = 0.5     # initial EWMA value per cell

# N3 forecast-scoring thresholds (verbatim from exp163)
OK_VIOLATION_THRESH = 0.15      # L=OK violated iff e_next >= 0.15
NOISE_VIOLATION_DELTA = 0.30    # L=NOISE violated iff |e_next - e_w| >= 0.30
ROLLING_TRUST_WINDOW = 10       # last 10 scored labels for rolling trust
TRUST_FIRE_THRESH = 0.7         # advance dial when trust < 0.7
MIN_SCORED_FOR_TRUST = 3        # < 3 scored since last dial change => NO-EVIDENCE

# FR1 parameters: slow alternation, H=1000, UNSEEN instance
HALF_PERIOD_FR1 = 1000           # H = 1000 (Exp 162 used 400)
FR1_DERANG_SEED_OFFSET = 130_000  # rng 130_000 + fork_seed (leakage-free)

# FR2 parameters: micro-burst
FR2_BURST_STEPS = [(1500, 1549), (3000, 3049)]   # (start_inclusive, end_inclusive)
FR2_BURST_LEN = 50
FR2_DERANG_SEED_OFFSET = 140_000  # rng 140_000 + fork_seed

# Derangement options (verbatim from exp162/163)
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]

# FR2 score ground truth thresholds
FR2_STRUCT_BURST_MIN = 10   # window contains >= 10 burst steps => true STRUCTURAL
FR2_OK_BURST_MAX = 0        # window contains 0 burst steps => true OK
# windows with 1-9 burst steps excluded

# Precondition thresholds
PC1_AHAT_DRIFT_MAX = 0.05          # PC1: ahat_drift < 0.05 per (fork, session)
PC2_MIN_EVAL_POINTS = 30           # PC2: >= 30 eval points per (fork, session) at W=200
PC3_FR1_DERANGED_ERR_MIN = 0.9     # PC3a: per-step error within FR1 deranged phases >= 0.9
PC3_FR2_BURST_ERR_MIN = 0.9        # PC3b: per-step error within FR2 bursts >= 0.9
PC4_GAP_MIN = 0.2                  # PC4: ctrl_a - score_a(r) >= 0.2 pooled per failure regime

# P1 thresholds
P1_RECOVERY_MIN = 0.6              # recovery(b, r) >= 0.6 per fork
P1_FORKS_MIN = 7                   # >= 7/8 forks
F1_POOLED_RECOVERY_MAX = 0.3       # F1: pooled recovery <= 0.3 in either regime

# P2 thresholds
P2_COMBINED_MARGIN = 0.15          # pooled combined_b - max_W pooled combined_c_W >= 0.15
P2_FORKS_MIN = 7                   # combined_b > max_W combined_c_W in >= 7/8 forks
F2_MATCH_MARGIN = 0.05             # F2: some W has score_c_W(r) >= score_b(r) - 0.05 in BOTH regimes

ALL_SESSIONS = ("FR1", "FR2", "CTRL")
FAILURE_SESSIONS = ("FR1", "FR2")


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp162/163
# ---------------------------------------------------------------------------

class StructCmap:
    """Hidden-context cmap: alternates every half_period steps."""

    def __init__(self, base_cmap, n_colors, derangement, half_period=100):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.derangement = list(derangement)
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
# BurstCmap — standard world with two 50-step deranged bursts (FR2)
# ---------------------------------------------------------------------------

class BurstCmap:
    """Standard color map with deranged bursts at specified step ranges."""

    def __init__(self, base_cmap, n_colors, derangement, burst_steps):
        self.base = list(base_cmap)
        self.n_colors = int(n_colors)
        self.derangement = list(derangement)
        # Build a flat set of burst step indices for O(1) membership check
        self._burst_set: set[int] = set()
        for start, end in burst_steps:
            for t in range(start, end + 1):
                self._burst_set.add(t)
        self.current_step = 0  # caller sets before __getitem__

    def in_burst(self, t: int) -> bool:
        return t in self._burst_set

    def __getitem__(self, s):
        true_color = self.base[s]
        if self.current_step in self._burst_set:
            return self.derangement[true_color]
        return true_color

    def __len__(self):
        return len(self.base)


# ---------------------------------------------------------------------------
# lag1_rho — verbatim from exp162/163
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
# classify_window — verbatim from exp162/163
# ---------------------------------------------------------------------------

def classify_window(window: np.ndarray) -> tuple[str, float, float]:
    """Classify a correctness window. Returns (label, error_rate, rho)."""
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
# Draw per-fork derangements
# ---------------------------------------------------------------------------

def draw_fork_derangements(fork_seed: int) -> dict:
    """Draw per-fork derangements for FR1 and FR2.

    FR1 uses rng 130_000 + fork_seed (UNSEEN instance, leakage-free from exp162/163).
    FR2 uses rng 140_000 + fork_seed.
    """
    fr1_rng = np.random.default_rng(FR1_DERANG_SEED_OFFSET + fork_seed)
    fr1_derangement = list(
        DERANGEMENT_OPTIONS[int(fr1_rng.integers(0, len(DERANGEMENT_OPTIONS)))]
    )
    fr2_rng = np.random.default_rng(FR2_DERANG_SEED_OFFSET + fork_seed)
    fr2_derangement = list(
        DERANGEMENT_OPTIONS[int(fr2_rng.integers(0, len(DERANGEMENT_OPTIONS)))]
    )
    return {
        "fr1_derangement": fr1_derangement,
        "fr2_derangement": fr2_derangement,
    }


# ---------------------------------------------------------------------------
# Run one fork in one session — records per-step correctness + context info
# ---------------------------------------------------------------------------

def run_fork_session(
    mirro: Creature,
    fork_seed: int,
    session: str,
    base_cmap: list,
    n_colors: int,
    fork_derangements: dict,
    n_chunks: int = N_CHUNKS,
    chunk_size: int = CHUNK_SIZE,
) -> dict:
    """Run one fresh fork of mirro in one session (FR1, FR2, or CTRL).

    Step loop replicates exp163 semantics exactly.
    Returns per-step correctness array, context/burst membership arrays,
    and ahat_drift. The correctness stream is later post-hoc processed
    by all agent dial policies.
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp164_s{fork_seed}_{session}"
    c = mirro.fork(fork_name)

    n_cells = c.world.n_cells
    n_actions = 4

    # --- Build session-specific cmap ---
    if session == "CTRL":
        cmap_obj = base_cmap   # plain list; no step counter needed
        burst_set: set[int] = set()
        half_period_fr1 = None
    elif session == "FR1":
        derangement = fork_derangements["fr1_derangement"]
        cmap_obj = StructCmap(base_cmap, n_colors, derangement, half_period=HALF_PERIOD_FR1)
        burst_set = set()
        half_period_fr1 = HALF_PERIOD_FR1
    elif session == "FR2":
        derangement = fork_derangements["fr2_derangement"]
        cmap_obj = BurstCmap(base_cmap, n_colors, derangement, FR2_BURST_STEPS)
        # Collect burst step set for membership recording
        burst_set = cmap_obj._burst_set
        half_period_fr1 = None
    else:
        raise ValueError(f"Unknown session: {session}")

    B = c.world.transition_matrix()   # (n_cells, n_cells, 4)
    n_total = n_chunks * chunk_size    # 4000

    A_hat_start = c._A_hat().copy()
    ewma = np.full(n_cells, EWMA_INIT, dtype=np.float64)

    correct_arr = np.empty(n_total, dtype=np.int32)
    # For FR1: context phase (0=A, 1=B) per step
    context_fr1 = np.empty(n_total, dtype=np.int32) if session == "FR1" else None
    # For FR2: burst membership per step (0/1)
    context_fr2 = np.empty(n_total, dtype=np.int32) if session == "FR2" else None

    global_step = 0
    eps = 1e-300

    for chunk_idx in range(n_chunks):
        chunk_seed = (fork_seed * 10_000 + chunk_idx) & 0xFFFFFFFF
        rng = np.random.default_rng(chunk_seed)

        for _step_in_chunk in range(chunk_size):
            t = global_step

            # Set step counter for cmap objects that need it
            if session == "FR1":
                cmap_obj.current_step = t   # type: ignore[union-attr]
            elif session == "FR2":
                cmap_obj.current_step = t   # type: ignore[union-attr]

            # Ground-truth context logging
            if context_fr1 is not None:
                context_fr1[t] = int((t // HALF_PERIOD_FR1) % 2)
            if context_fr2 is not None:
                context_fr2[t] = int(t in burst_set)

            # A_hat pre-step
            A_hat = c._A_hat()

            # Predicted observation
            pred_probs = A_hat @ c.qs
            o_hat = int(np.argmax(pred_probs))

            # bel_cell and EWMA channel confidence (pre-update)
            bel_cell = int(np.argmax(c.qs))

            # Observe
            true_pos_t = c.true_pos
            if session == "CTRL":
                obs = int(base_cmap[true_pos_t])
            else:
                obs = int(cmap_obj[true_pos_t])

            # Correctness
            correct_t = 1 if (o_hat == obs) else 0
            correct_arr[t] = correct_t

            # Surprise (for trajectory consistency)
            p_o = float(A_hat[obs, :] @ c.qs)
            _ = -math.log(p_o + eps)

            # Belief update
            likelihood = A_hat[obs, :]
            qs_updated = likelihood * c.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(n_cells) / n_cells

            # Dirichlet count update
            c.pA[obs, :] += qs_updated

            # Value accumulation (Exp 26 mechanism)
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )
            predictability_weight = math.exp(-h_predicted)
            c.value_counts[obs] += predictability_weight

            # Random action
            action = int(rng.integers(0, n_actions))

            # Move
            c.true_pos = c.world.move(c.true_pos, action)

            # Advance qs through B
            c.qs = B[:, :, action] @ qs_updated

            c.age_steps += 1

            # EWMA update AFTER correctness known
            ewma[bel_cell] = (1.0 - ALPHA) * ewma[bel_cell] + ALPHA * correct_t

            global_step += 1

    A_hat_end = c._A_hat()
    ahat_drift = float(np.abs(A_hat_end - A_hat_start).max())

    return {
        "fork_seed": fork_seed,
        "session": session,
        "n_total": n_total,
        "ahat_drift": ahat_drift,
        "correct_arr": correct_arr,
        "context_fr1": context_fr1,
        "context_fr2": context_fr2,
        "fr1_derangement": fork_derangements["fr1_derangement"],
        "fr2_derangement": fork_derangements["fr2_derangement"],
    }


# ---------------------------------------------------------------------------
# Post-hoc: evaluate fixed-dial agent (N2-only or any offline retune c_W)
# ---------------------------------------------------------------------------

def eval_fixed_dial(correct_arr: np.ndarray, W: int) -> dict:
    """Evaluate a fixed-dial agent over the correctness stream.

    Eval points: t = 99, 199, ... 3999 (every 100 steps).
    Window evaluates iff t+1 >= W (i.e. at least W steps of history).
    Returns list of (t, label, error_rate, rho) eval records.
    """
    n_total = len(correct_arr)
    eval_records: list[dict] = []

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):
        if t + 1 < W:
            continue   # not enough history
        win_arr = correct_arr[max(0, t - W + 1): t + 1].astype(np.float64)
        label, err_rate, rho = classify_window(win_arr)
        eval_records.append({"t": t, "label": label, "e_w": err_rate, "rho": rho})

    return {"W": W, "eval_records": eval_records}


# ---------------------------------------------------------------------------
# Post-hoc: simulate N2+N3 agent (dial rewrite driven by trust monitor)
# ---------------------------------------------------------------------------

def eval_n3_agent(correct_arr: np.ndarray) -> dict:
    """Simulate N2+N3 agent (b) over the correctness stream.

    Starts at W = 200. The Exp 163 forecast-scoring monitor runs online under
    the CURRENT dial. Policy: when rolling trust < 0.7 (and >= 3 scored labels
    in rolling buffer since last dial change), advance dial cyclically
    200->400->800->1600->200, clear rolling buffer, drop pending forecasts.

    Rolling trust = 1 - violations among last 10 scored labels in current buffer.

    Returns eval_records (list of {t, label, e_w, rho, W_used}) and
    dial_trajectory (list of {t, dial_change_to, trust_at_change}).
    """
    n_total = len(correct_arr)
    HORIZON = 100   # forecast horizon (next 100 steps)
    MAX_SCORABLE_T = n_total - 1 - HORIZON   # = 3899

    current_W = DIAL_CANDIDATES[0]   # start at 200
    dial_trajectory: list[dict] = [{"t": -1, "dial_change_to": current_W, "trust_at_change": None}]

    # Rolling scored outcomes since last dial change (deque of 0/1, max 10)
    rolling_outcomes: _deque[int] = _deque(maxlen=ROLLING_TRUST_WINDOW)
    # Pending forecasts: (t_eval, label, e_w) not yet scorable (t_eval+HORIZON not yet passed)
    pending: list[dict] = []
    # Count of scored labels in rolling buffer since last dial change
    scored_since_change = 0

    eval_records: list[dict] = []

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):

        # --- Step 1: mature any pending forecasts where t+100 has elapsed ---
        # A forecast issued at t_eval is scorable once t >= t_eval + HORIZON
        newly_matured: list[dict] = []
        still_pending: list[dict] = []
        for pf in pending:
            if t >= pf["t_eval"] + HORIZON:
                newly_matured.append(pf)
            else:
                still_pending.append(pf)
        pending = still_pending

        for pf in newly_matured:
            t_eval = pf["t_eval"]
            label = pf["label"]
            e_w = pf["e_w"]
            # Compute e_next = mean error of steps t_eval+1 .. t_eval+HORIZON
            next_slice = correct_arr[t_eval + 1: t_eval + 1 + HORIZON]
            e_next = 1.0 - float(next_slice.mean())
            if label == "OK":
                violated = int(e_next >= OK_VIOLATION_THRESH)
            else:  # NOISE
                violated = int(abs(e_next - e_w) >= NOISE_VIOLATION_DELTA)
            rolling_outcomes.append(1 - violated)
            scored_since_change += 1

        # --- Step 2: check trust rule (only when >= MIN_SCORED_FOR_TRUST scored) ---
        if scored_since_change >= MIN_SCORED_FOR_TRUST and len(rolling_outcomes) > 0:
            trust = float(sum(rolling_outcomes) / len(rolling_outcomes))
            if trust < TRUST_FIRE_THRESH:
                # Advance dial cyclically
                idx = DIAL_CANDIDATES.index(current_W)
                current_W = DIAL_CANDIDATES[(idx + 1) % len(DIAL_CANDIDATES)]
                # Clear rolling buffer and drop pending forecasts (declared)
                rolling_outcomes.clear()
                scored_since_change = 0
                pending = []
                dial_trajectory.append({"t": t, "dial_change_to": current_W, "trust_at_change": trust})

        # --- Step 3: evaluate current window under current_W if enough history ---
        if t + 1 >= current_W:
            win_arr = correct_arr[max(0, t - current_W + 1): t + 1].astype(np.float64)
            label, err_rate, rho = classify_window(win_arr)
            eval_records.append({"t": t, "label": label, "e_w": err_rate, "rho": rho, "W_used": current_W})

            # Register forecast if OK or NOISE and scorable horizon exists
            if label != "STRUCTURAL" and t <= MAX_SCORABLE_T:
                pending.append({"t_eval": t, "label": label, "e_w": err_rate})

    return {
        "eval_records": eval_records,
        "dial_trajectory": dial_trajectory,
    }


# ---------------------------------------------------------------------------
# Score functions: FR1, FR2, CTRL
# ---------------------------------------------------------------------------

def score_fr1(eval_records: list[dict]) -> float:
    """FR1 score: fraction of evaluated windows labeled STRUCTURAL."""
    if not eval_records:
        return float("nan")
    n_struct = sum(1 for r in eval_records if r["label"] == "STRUCTURAL")
    return n_struct / len(eval_records)


def score_fr2(eval_records: list[dict], correct_arr: np.ndarray, W: int) -> float:
    """FR2 score = 0.5 * recall + 0.5 * specificity.

    Ground truth per window:
      STRUCTURAL iff window contains >= 10 burst steps (FR2_STRUCT_BURST_MIN)
      OK iff window contains 0 burst steps
      windows with 1-9 burst steps excluded.

    For agent (b) (variable W), use the W_used recorded in each eval record.
    For fixed-dial agents, W is constant.
    """
    # Build burst step set
    burst_set: set[int] = set()
    for start, end in FR2_BURST_STEPS:
        for step in range(start, end + 1):
            burst_set.add(step)

    true_struct: list[str] = []   # predicted labels for true-STRUCTURAL windows
    true_ok: list[str] = []       # predicted labels for true-OK windows

    for rec in eval_records:
        t = rec["t"]
        pred = rec["label"]
        w_used = rec.get("W_used", W)   # W_used from (b); W for fixed-dial
        window_start = max(0, t - w_used + 1)
        burst_count = sum(1 for s in range(window_start, t + 1) if s in burst_set)

        if burst_count >= FR2_STRUCT_BURST_MIN:
            true_struct.append(pred)
        elif burst_count == FR2_OK_BURST_MAX:
            true_ok.append(pred)
        # else: 1-9 burst steps => excluded

    if not true_struct and not true_ok:
        return float("nan")

    recall = (
        sum(1 for p in true_struct if p == "STRUCTURAL") / len(true_struct)
        if true_struct else float("nan")
    )
    specificity = (
        sum(1 for p in true_ok if p == "OK") / len(true_ok)
        if true_ok else float("nan")
    )

    if math.isnan(recall) or math.isnan(specificity):
        # If one is missing, return what we have (degenerate; diagnostics only)
        if not math.isnan(recall):
            return 0.5 * recall
        if not math.isnan(specificity):
            return 0.5 * specificity
        return float("nan")

    return 0.5 * recall + 0.5 * specificity


def score_ctrl(eval_records: list[dict]) -> float:
    """CTRL score: fraction labeled OK."""
    if not eval_records:
        return float("nan")
    n_ok = sum(1 for r in eval_records if r["label"] == "OK")
    return n_ok / len(eval_records)


# ---------------------------------------------------------------------------
# Compute all agent scores for one (fork, session) correctness stream
# ---------------------------------------------------------------------------

def compute_fork_session_scores(
    run_result: dict,
) -> dict:
    """Compute all agent scores for one (fork, session) run result.

    Agents:
      (a) N2-only: fixed W=200
      (c_200, c_400, c_800, c_1600): offline retunes
      (b) N2+N3: dial-rewriting agent

    Returns dict with scores per agent and eval_records / dial trajectory.
    """
    correct_arr = run_result["correct_arr"]
    session = run_result["session"]
    context_fr2 = run_result["context_fr2"]

    # --- Fixed-dial readouts ---
    fixed_results: dict[int, dict] = {}
    for W in DIAL_CANDIDATES:
        fixed_results[W] = eval_fixed_dial(correct_arr, W)

    # --- N2+N3 agent (b) ---
    n3_result = eval_n3_agent(correct_arr)
    n3_eval_records = n3_result["eval_records"]
    dial_trajectory = n3_result["dial_trajectory"]

    # --- Score all agents ---
    scores: dict[str, float] = {}

    for W in DIAL_CANDIDATES:
        recs = fixed_results[W]["eval_records"]
        agent_key = f"c_{W}"
        if session == "FR1":
            scores[agent_key] = score_fr1(recs)
        elif session == "FR2":
            scores[agent_key] = score_fr2(recs, correct_arr, W)
        else:  # CTRL
            scores[agent_key] = score_ctrl(recs)

    # N2-only = c_200 (agent a)
    scores["a"] = scores["c_200"]

    # Agent (b)
    if session == "FR1":
        scores["b"] = score_fr1(n3_eval_records)
    elif session == "FR2":
        scores["b"] = score_fr2(n3_eval_records, correct_arr, W=0)  # W=0 => uses W_used from recs
    else:  # CTRL
        scores["b"] = score_ctrl(n3_eval_records)

    return {
        "scores": scores,
        "n3_eval_records": n3_eval_records,
        "dial_trajectory": dial_trajectory,
        "fixed_eval_records": {W: fixed_results[W]["eval_records"] for W in DIAL_CANDIDATES},
        "n_eval_a": len(fixed_results[200]["eval_records"]),
    }


# ---------------------------------------------------------------------------
# Precondition checks
# ---------------------------------------------------------------------------

def check_preconditions(run_results: list[dict], score_results: dict) -> tuple[bool, list[str]]:
    """Check PC1-PC4.

    score_results: dict[(fork_seed, session)] -> compute_fork_session_scores result
    Returns (all_pass, list_of_failures).
    """
    failures: list[str] = []

    # PC1: ahat_drift < 0.05 per (fork, session)
    # PC2: >= 30 eval points per (fork, session) at W=200
    for rr in run_results:
        seed = rr["fork_seed"]
        session = rr["session"]
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: seed={seed} {session} ahat_drift={rr['ahat_drift']:.4f}"
            )
        sr = score_results[(seed, session)]
        n_eval = sr["n_eval_a"]
        if n_eval < PC2_MIN_EVAL_POINTS:
            failures.append(
                f"PC2 FAIL: seed={seed} {session} n_eval_a={n_eval} < {PC2_MIN_EVAL_POINTS}"
            )

    # PC3: per-step error within FR1 deranged phases >= 0.9 pooled
    fr1_deranged_errors: list[float] = []
    for rr in run_results:
        if rr["session"] != "FR1" or rr["context_fr1"] is None:
            continue
        ctx = rr["context_fr1"]
        correct = rr["correct_arr"]
        deranged_mask = ctx == 1   # context B = deranged
        if deranged_mask.sum() > 0:
            err = 1.0 - float(correct[deranged_mask].mean())
            fr1_deranged_errors.append(err)

    if fr1_deranged_errors:
        pooled_fr1_err = float(np.mean(fr1_deranged_errors))
        if pooled_fr1_err < PC3_FR1_DERANGED_ERR_MIN:
            failures.append(
                f"PC3a FAIL: FR1 pooled deranged-phase error={pooled_fr1_err:.4f} "
                f"< {PC3_FR1_DERANGED_ERR_MIN}"
            )

    # PC3: per-step error within FR2 burst steps >= 0.9 pooled
    fr2_burst_errors: list[float] = []
    for rr in run_results:
        if rr["session"] != "FR2" or rr["context_fr2"] is None:
            continue
        ctx = rr["context_fr2"]
        correct = rr["correct_arr"]
        burst_mask = ctx == 1
        if burst_mask.sum() > 0:
            err = 1.0 - float(correct[burst_mask].mean())
            fr2_burst_errors.append(err)

    if fr2_burst_errors:
        pooled_fr2_err = float(np.mean(fr2_burst_errors))
        if pooled_fr2_err < PC3_FR2_BURST_ERR_MIN:
            failures.append(
                f"PC3b FAIL: FR2 pooled burst error={pooled_fr2_err:.4f} "
                f"< {PC3_FR2_BURST_ERR_MIN}"
            )

    # PC4: ctrl_a - score_a(r) >= 0.2 pooled in both failure regimes
    for session in FAILURE_SESSIONS:
        gaps: list[float] = []
        for rr in run_results:
            if rr["session"] != session:
                continue
            seed = rr["fork_seed"]
            sr = score_results[(seed, session)]
            ctrl_sr = score_results.get((seed, "CTRL"))
            if ctrl_sr is None:
                continue
            score_a = sr["scores"]["a"]
            ctrl_a = ctrl_sr["scores"]["a"]
            if not math.isnan(score_a) and not math.isnan(ctrl_a):
                gaps.append(ctrl_a - score_a)
        if gaps:
            pooled_gap = float(np.mean(gaps))
            if pooled_gap < PC4_GAP_MIN:
                failures.append(
                    f"PC4 FAIL: {session} pooled ctrl_a - score_a = {pooled_gap:.4f} "
                    f"< {PC4_GAP_MIN}"
                )
        else:
            failures.append(f"PC4 FAIL: {session} no valid gap measurements")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Evaluate P1/F1, P2/F2 and compute verdict
# ---------------------------------------------------------------------------

def evaluate(
    run_results: list[dict],
    score_results: dict,
    seeds: list[int],
) -> dict:
    """Evaluate P1/F1 and P2/F2; return evaluation dict.

    pooled = mean over forks (scores are per-fork fractions; declared).
    """
    # --- Build per-fork recovery and combined scores ---
    recovery_b_fr1: list[float] = []
    recovery_b_fr2: list[float] = []
    combined_b: list[float] = []
    combined_c: dict[int, list[float]] = {W: [] for W in DIAL_CANDIDATES}

    # For pooled scores
    pooled_scores_b: dict[str, list[float]] = {"FR1": [], "FR2": [], "CTRL": []}
    pooled_scores_a: dict[str, list[float]] = {"FR1": [], "FR2": [], "CTRL": []}
    pooled_scores_c: dict[int, dict[str, list[float]]] = {
        W: {"FR1": [], "FR2": [], "CTRL": []} for W in DIAL_CANDIDATES
    }

    for seed in seeds:
        ctrl_key = (seed, "CTRL")
        fr1_key = (seed, "FR1")
        fr2_key = (seed, "FR2")

        if ctrl_key not in score_results or fr1_key not in score_results or fr2_key not in score_results:
            continue

        ctrl_sr = score_results[ctrl_key]
        fr1_sr = score_results[fr1_key]
        fr2_sr = score_results[fr2_key]

        score_a_ctrl = ctrl_sr["scores"]["a"]
        score_a_fr1 = fr1_sr["scores"]["a"]
        score_a_fr2 = fr2_sr["scores"]["a"]
        score_b_fr1 = fr1_sr["scores"]["b"]
        score_b_fr2 = fr2_sr["scores"]["b"]

        # Accumulate pooled
        for sess, s_arr in [("FR1", fr1_sr), ("FR2", fr2_sr), ("CTRL", ctrl_sr)]:
            pooled_scores_b[sess].append(s_arr["scores"]["b"])
            pooled_scores_a[sess].append(s_arr["scores"]["a"])
            for W in DIAL_CANDIDATES:
                pooled_scores_c[W][sess].append(s_arr["scores"][f"c_{W}"])

        # Recovery per fork
        # recovery(b, r) = (score_b(r) - score_a(r)) / (ctrl_a - score_a(r))
        if not math.isnan(score_a_ctrl) and not math.isnan(score_a_fr1) and not math.isnan(score_b_fr1):
            denom_fr1 = score_a_ctrl - score_a_fr1
            if abs(denom_fr1) > 1e-12:
                rec_fr1 = (score_b_fr1 - score_a_fr1) / denom_fr1
                recovery_b_fr1.append(rec_fr1)
            else:
                recovery_b_fr1.append(float("nan"))
        else:
            recovery_b_fr1.append(float("nan"))

        if not math.isnan(score_a_ctrl) and not math.isnan(score_a_fr2) and not math.isnan(score_b_fr2):
            denom_fr2 = score_a_ctrl - score_a_fr2
            if abs(denom_fr2) > 1e-12:
                rec_fr2 = (score_b_fr2 - score_a_fr2) / denom_fr2
                recovery_b_fr2.append(rec_fr2)
            else:
                recovery_b_fr2.append(float("nan"))
        else:
            recovery_b_fr2.append(float("nan"))

        # Combined per fork: combined = min(score(FR1), score(FR2))
        if not math.isnan(score_b_fr1) and not math.isnan(score_b_fr2):
            combined_b.append(min(score_b_fr1, score_b_fr2))
        else:
            combined_b.append(float("nan"))

        for W in DIAL_CANDIDATES:
            s_c_fr1 = fr1_sr["scores"][f"c_{W}"]
            s_c_fr2 = fr2_sr["scores"][f"c_{W}"]
            if not math.isnan(s_c_fr1) and not math.isnan(s_c_fr2):
                combined_c[W].append(min(s_c_fr1, s_c_fr2))
            else:
                combined_c[W].append(float("nan"))

    def _pooled(vals: list[float]) -> float:
        v = [x for x in vals if not math.isnan(x)]
        return float(np.mean(v)) if v else float("nan")

    pooled_recovery_fr1 = _pooled(recovery_b_fr1)
    pooled_recovery_fr2 = _pooled(recovery_b_fr2)

    pooled_combined_b = _pooled(combined_b)
    pooled_combined_c = {W: _pooled(combined_c[W]) for W in DIAL_CANDIDATES}
    max_pooled_combined_c = max(
        (v for v in pooled_combined_c.values() if not math.isnan(v)),
        default=float("nan"),
    )
    best_W_combined = max(
        (W for W in DIAL_CANDIDATES if not math.isnan(pooled_combined_c[W])),
        key=lambda W: pooled_combined_c[W],
        default=None,
    )

    # --- P1 / F1 ---
    # P1: recovery(b, FR1) >= 0.6 AND recovery(b, FR2) >= 0.6, each in >= 7/8 forks
    n_forks = len(seeds)
    p1_forks_fr1 = sum(
        1 for r in recovery_b_fr1 if not math.isnan(r) and r >= P1_RECOVERY_MIN
    )
    p1_forks_fr2 = sum(
        1 for r in recovery_b_fr2 if not math.isnan(r) and r >= P1_RECOVERY_MIN
    )
    p1_pass = p1_forks_fr1 >= P1_FORKS_MIN and p1_forks_fr2 >= P1_FORKS_MIN

    # F1: pooled recovery <= 0.3 in either failure regime
    f1 = (
        (not math.isnan(pooled_recovery_fr1) and pooled_recovery_fr1 <= F1_POOLED_RECOVERY_MAX) or
        (not math.isnan(pooled_recovery_fr2) and pooled_recovery_fr2 <= F1_POOLED_RECOVERY_MAX)
    )

    # --- P2 / F2 ---
    # P2: pooled combined_b - max_W pooled combined_c_W >= 0.15
    #     AND combined_b > max_W combined_c_W in >= 7/8 forks
    if not math.isnan(pooled_combined_b) and not math.isnan(max_pooled_combined_c):
        pooled_combined_margin = pooled_combined_b - max_pooled_combined_c
    else:
        pooled_combined_margin = float("nan")

    p2_forks = sum(
        1 for i, cb in enumerate(combined_b)
        if not math.isnan(cb) and all(
            math.isnan(combined_c[W][i]) or cb > combined_c[W][i]
            for W in DIAL_CANDIDATES
        )
    )
    p2_pass = (
        not math.isnan(pooled_combined_margin)
        and pooled_combined_margin >= P2_COMBINED_MARGIN
        and p2_forks >= P2_FORKS_MIN
    )

    # F2: some single W has score_c_W(r) >= score_b(r) - 0.05 in BOTH failure regimes (pooled)
    # i.e., for any W: pooled score_c_W(FR1) >= pooled score_b(FR1) - 0.05
    #                  AND pooled score_c_W(FR2) >= pooled score_b(FR2) - 0.05
    f2 = False
    f2_firing_W = None
    pooled_b_fr1 = _pooled(pooled_scores_b["FR1"])
    pooled_b_fr2 = _pooled(pooled_scores_b["FR2"])
    for W in DIAL_CANDIDATES:
        pooled_cW_fr1 = _pooled(pooled_scores_c[W]["FR1"])
        pooled_cW_fr2 = _pooled(pooled_scores_c[W]["FR2"])
        if (not math.isnan(pooled_cW_fr1) and not math.isnan(pooled_b_fr1) and
                not math.isnan(pooled_cW_fr2) and not math.isnan(pooled_b_fr2)):
            if (pooled_cW_fr1 >= pooled_b_fr1 - F2_MATCH_MARGIN and
                    pooled_cW_fr2 >= pooled_b_fr2 - F2_MATCH_MARGIN):
                f2 = True
                f2_firing_W = W
                break

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
        "n_forks": n_forks,
        # Recovery per fork
        "recovery_b_fr1": recovery_b_fr1,
        "recovery_b_fr2": recovery_b_fr2,
        "pooled_recovery_fr1": pooled_recovery_fr1,
        "pooled_recovery_fr2": pooled_recovery_fr2,
        # P1 / F1
        "p1_forks_fr1": p1_forks_fr1,
        "p1_forks_fr2": p1_forks_fr2,
        "p1_pass": p1_pass,
        "f1": f1,
        # Combined scores
        "combined_b": combined_b,
        "combined_c": combined_c,
        "pooled_combined_b": pooled_combined_b,
        "pooled_combined_c": pooled_combined_c,
        "max_pooled_combined_c": max_pooled_combined_c,
        "best_W_combined": best_W_combined,
        "pooled_combined_margin": pooled_combined_margin,
        # P2 / F2
        "p2_forks": p2_forks,
        "p2_pass": p2_pass,
        "f2": f2,
        "f2_firing_W": f2_firing_W,
        # Pooled scores per agent and regime
        "pooled_scores_b": {s: _pooled(pooled_scores_b[s]) for s in ALL_SESSIONS},
        "pooled_scores_a": {s: _pooled(pooled_scores_a[s]) for s in ALL_SESSIONS},
        "pooled_scores_c": {W: {s: _pooled(pooled_scores_c[W][s]) for s in ALL_SESSIONS}
                            for W in DIAL_CANDIDATES},
        # Verdict
        "verdict_str": verdict_str,
    }


# ---------------------------------------------------------------------------
# Write JSONL rows + summary
# ---------------------------------------------------------------------------

def _nan_to_none(v):
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_json_rows(
    run_results: list[dict],
    score_results: dict,
    ev: dict | None,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    seeds_seen: list[int] = []
    seeds_set: set[int] = set()

    with path.open("w") as fh:
        for rr in run_results:
            seed = rr["fork_seed"]
            session = rr["session"]
            sr = score_results.get((seed, session), {})
            scores = sr.get("scores", {})
            dial_traj = sr.get("dial_trajectory", [])

            row: dict = {
                "exp": 164,
                "fork_seed": seed,
                "session": session,
                "n_total": rr["n_total"],
                "ahat_drift": _nan_to_none(rr["ahat_drift"]),
                "fr1_derangement": rr["fr1_derangement"],
                "fr2_derangement": rr["fr2_derangement"],
                "score_a": _nan_to_none(scores.get("a")),
                "score_b": _nan_to_none(scores.get("b")),
                **{f"score_c_{W}": _nan_to_none(scores.get(f"c_{W}"))
                   for W in DIAL_CANDIDATES},
                "n_eval_a": sr.get("n_eval_a"),
                "dial_changes": len(dial_traj) - 1,  # exclude initial entry
                "dial_trajectory_summary": [
                    {"t": d["t"], "to": d["dial_change_to"], "trust": _nan_to_none(d["trust_at_change"])}
                    for d in dial_traj
                ],
            }
            fh.write(json.dumps(row) + "\n")

            if seed not in seeds_set:
                seeds_set.add(seed)
                seeds_seen.append(seed)

        if ev is not None:
            summary: dict = {
                "exp": 164,
                "row_type": "summary",
                "pooled_recovery_fr1": _nan_to_none(ev["pooled_recovery_fr1"]),
                "pooled_recovery_fr2": _nan_to_none(ev["pooled_recovery_fr2"]),
                "p1_forks_fr1": ev["p1_forks_fr1"],
                "p1_forks_fr2": ev["p1_forks_fr2"],
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "pooled_combined_b": _nan_to_none(ev["pooled_combined_b"]),
                "max_pooled_combined_c": _nan_to_none(ev["max_pooled_combined_c"]),
                "best_W_combined": ev["best_W_combined"],
                "pooled_combined_margin": _nan_to_none(ev["pooled_combined_margin"]),
                "p2_forks": ev["p2_forks"],
                "p2_pass": ev["p2_pass"],
                "f2": ev["f2"],
                "f2_firing_W": ev["f2_firing_W"],
                "verdict": ev["verdict_str"],
            }
            fh.write(json.dumps(summary) + "\n")


# ---------------------------------------------------------------------------
# Print dial trajectories (ungated diagnostic)
# ---------------------------------------------------------------------------

def print_dial_trajectories(score_results: dict, seeds: list[int]) -> None:
    """Print agent (b)'s dial trajectory per (fork, session)."""
    print("Agent (b) dial trajectories:")
    for seed in seeds:
        for session in ALL_SESSIONS:
            sr = score_results.get((seed, session))
            if sr is None:
                continue
            traj = sr["dial_trajectory"]
            traj_strs = [
                f"t={d['t']} W={d['dial_change_to']}"
                + (f" [trust={d['trust_at_change']:.3f}]" if d['trust_at_change'] is not None else "")
                for d in traj
            ]
            print(f"  seed={seed} {session:>5}: {' -> '.join(traj_strs)}")
    print()


# ---------------------------------------------------------------------------
# Print per-dial score table
# ---------------------------------------------------------------------------

def print_score_table(score_results: dict, seeds: list[int], ev: dict) -> None:
    """Print per-dial score table: agents (a), (c_200..c_1600), (b) per regime."""
    def _f(v, w=7):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan".rjust(w)
        return f"{v:.4f}".rjust(w)

    print("Per-dial score table (pooled over forks):")
    header = f"{'agent':>10}  {'FR1_score':>9}  {'FR2_score':>9}  {'CTRL_score':>10}  {'combined':>9}"
    print(header)
    print("-" * len(header))

    ps_a = ev["pooled_scores_a"]
    ps_b = ev["pooled_scores_b"]
    ps_c = ev["pooled_scores_c"]

    # Agent (a) = c_200
    comb_a = min(ps_a.get("FR1", float("nan")), ps_a.get("FR2", float("nan")))
    print(
        f"{'(a) W=200':>10}  {_f(ps_a.get('FR1'))}  {_f(ps_a.get('FR2'))}  "
        f"{_f(ps_a.get('CTRL'), 10)}  {_f(comb_a)}"
    )

    for W in DIAL_CANDIDATES:
        agent_key = f"c_{W}"
        sc = ps_c[W]
        comb = ev["pooled_combined_c"][W]
        print(
            f"{f'(c) W={W}':>10}  {_f(sc.get('FR1'))}  {_f(sc.get('FR2'))}  "
            f"{_f(sc.get('CTRL'), 10)}  {_f(comb)}"
        )

    comb_b = ev["pooled_combined_b"]
    print(
        f"{'(b) N3':>10}  {_f(ps_b.get('FR1'))}  {_f(ps_b.get('FR2'))}  "
        f"{_f(ps_b.get('CTRL'), 10)}  {_f(comb_b)}"
    )
    print()

    print(f"  combined_b - max_W(combined_c) = {_f(ev['pooled_combined_margin'])}  "
          f"(need >= {P2_COMBINED_MARGIN} for P2)")
    print(f"  best single W by combined = W={ev['best_W_combined']}")
    print()

    # Per-fork recovery table
    print("Per-fork recovery table:")
    print(f"  {'seed':>5}  {'rec_FR1':>9}  {'rec_FR2':>9}  {'p1_FR1_ok':>9}  {'p1_FR2_ok':>9}")
    print("  " + "-" * 50)
    for i, seed in enumerate(seeds):
        r_fr1 = ev["recovery_b_fr1"][i] if i < len(ev["recovery_b_fr1"]) else float("nan")
        r_fr2 = ev["recovery_b_fr2"][i] if i < len(ev["recovery_b_fr2"]) else float("nan")
        p1_fr1_ok = "YES" if not math.isnan(r_fr1) and r_fr1 >= P1_RECOVERY_MIN else "no"
        p1_fr2_ok = "YES" if not math.isnan(r_fr2) and r_fr2 >= P1_RECOVERY_MIN else "no"
        print(f"  {seed:>5}  {_f(r_fr1)}  {_f(r_fr2)}  {p1_fr1_ok:>9}  {p1_fr2_ok:>9}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 164 — N3 rung 3, the LOAD-BEARING test"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[74] only, all three sessions, full 4000 steps, "
            "prints scores + (b)'s dial trajectories per regime. No verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [74] if smoke else SEEDS

    print("=" * 80)
    print("Exp 164 — N3 rung 3, the LOAD-BEARING test")
    print("          N3's trust-driven window rewrite must beat N2-only and")
    print("          not reduce to any single offline retune.")
    print("=" * 80)
    print()
    print(f"SURPRISE_WINDOW={SURPRISE_WINDOW}  CLASSIFIER_WINDOW_DEFAULT={CLASSIFIER_WINDOW_DEFAULT}  "
          f"[SURPRISE_WINDOW must be 200; asserted]")
    assert SURPRISE_WINDOW == 200, f"SURPRISE_WINDOW={SURPRISE_WINDOW}, expected 200"
    print(f"EVAL_EVERY={EVAL_EVERY}  ERROR_RATE_OK_THRESH={ERROR_RATE_OK_THRESH}  "
          f"LAG1_RHO_STRUCT_THRESH={LAG1_RHO_STRUCT_THRESH}")
    print(f"DIAL_CANDIDATES={DIAL_CANDIDATES}  starting W={CLASSIFIER_WINDOW_DEFAULT}")
    print(f"N3 trust: TRUST_FIRE_THRESH={TRUST_FIRE_THRESH}  ROLLING_TRUST_WINDOW={ROLLING_TRUST_WINDOW}  "
          f"MIN_SCORED_FOR_TRUST={MIN_SCORED_FOR_TRUST}")
    print(f"N3 forecast: OK_VIOLATION_THRESH={OK_VIOLATION_THRESH}  "
          f"NOISE_VIOLATION_DELTA={NOISE_VIOLATION_DELTA}")
    print(f"FR1: HALF_PERIOD_FR1={HALF_PERIOD_FR1}  FR1_DERANG_SEED_OFFSET={FR1_DERANG_SEED_OFFSET}")
    print(f"FR2: bursts at {FR2_BURST_STEPS}  FR2_DERANG_SEED_OFFSET={FR2_DERANG_SEED_OFFSET}")
    print(f"DERANGEMENT_OPTIONS={DERANGEMENT_OPTIONS}")
    print(f"Seeds: {seeds}  Sessions: {ALL_SESSIONS}  N_CHUNKS={N_CHUNKS}  "
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

    # --- Run all (seed, session) pairs ---
    run_results: list[dict] = []
    score_results: dict = {}

    for seed in seeds:
        fork_derangements = draw_fork_derangements(seed)
        print(
            f"  seed={seed}  "
            f"fr1_derang={fork_derangements['fr1_derangement']}  "
            f"fr2_derang={fork_derangements['fr2_derangement']}",
            flush=True,
        )
        for session in ALL_SESSIONS:
            print(f"    Running seed={seed} session={session} ...", flush=True)
            rr = run_fork_session(
                mirro=mirro,
                fork_seed=seed,
                session=session,
                base_cmap=base_cmap,
                n_colors=n_colors,
                fork_derangements=fork_derangements,
                n_chunks=N_CHUNKS,
                chunk_size=CHUNK_SIZE,
            )
            run_results.append(rr)
            # Post-hoc scoring
            sr = compute_fork_session_scores(rr)
            score_results[(seed, session)] = sr
    print()

    # --- Print dial trajectories (ungated) ---
    print_dial_trajectories(score_results, seeds)

    # --- Print per-fork per-session scores ---
    print("Per-fork per-session scores:")
    hdr = (
        f"{'seed':>5}  {'session':>6}  {'score_a':>8}  "
        + "  ".join(f"c_{W}".rjust(7) for W in DIAL_CANDIDATES)
        + f"  {'score_b':>8}  {'n_eval':>6}  {'n_dchg':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for seed in seeds:
        for session in ALL_SESSIONS:
            sr = score_results.get((seed, session))
            if sr is None:
                continue
            sc = sr["scores"]
            def _f2(v, w=8):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "nan".rjust(w)
                return f"{v:.4f}".rjust(w)
            c_cols = "  ".join(_f2(sc.get(f"c_{W}"), 7) for W in DIAL_CANDIDATES)
            n_dchg = len(sr["dial_trajectory"]) - 1
            print(
                f"{seed:>5}  {session:>6}  {_f2(sc.get('a'))}  "
                f"{c_cols}  {_f2(sc.get('b'))}  "
                f"{sr['n_eval_a']:>6}  {n_dchg:>6}"
            )
    print()

    # --- Smoke exit ---
    if smoke:
        print("SMOKE ONLY — no verdict")
        return

    # --- Precondition check ---
    pc_pass, pc_failures = check_preconditions(run_results, score_results)
    if pc_failures:
        print("PRECONDITION FAILED:")
        for f in pc_failures:
            print(f"  {f}")
        print()
        print("PRECONDITION FAILED — no verdict.")
        out_rows_path = Path(__file__).parent / "outputs" / "exp164_rows.json"
        write_json_rows(run_results, score_results, ev=None, path=out_rows_path)
        print(f"JSON rows written to {out_rows_path} (no verdict)")
        return

    print("Preconditions: all PASS")
    print()

    # --- Evaluate predeclared outcome map ---
    ev = evaluate(run_results, score_results, seeds)
    n = ev["n_forks"]

    # --- Print score table ---
    print_score_table(score_results, seeds, ev)

    # --- P1 / F1 ---
    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    def _fv(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    print(f"P1 (N3 beats N2-only — recovery(b,r) >= {P1_RECOVERY_MIN} in >= {P1_FORKS_MIN}/{n} forks):")
    print(f"  Pooled recovery FR1: {_fv(ev['pooled_recovery_fr1'])}  "
          f"FR2: {_fv(ev['pooled_recovery_fr2'])}")
    print(f"  Forks recovery(FR1) >= {P1_RECOVERY_MIN}: {ev['p1_forks_fr1']}/{n}  "
          f"(need >= {P1_FORKS_MIN})")
    print(f"  Forks recovery(FR2) >= {P1_RECOVERY_MIN}: {ev['p1_forks_fr2']}/{n}  "
          f"(need >= {P1_FORKS_MIN})")
    print(f"  FALSIFIER F1: pooled recovery <= {F1_POOLED_RECOVERY_MAX} in either regime: "
          f"{'YES — F1 FIRES' if ev['f1'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    print(f"P2 (not reducible to config — combined margin >= {P2_COMBINED_MARGIN} pooled, "
          f">= {P2_FORKS_MIN}/{n} forks):")
    print(f"  Pooled combined_b = {_fv(ev['pooled_combined_b'])}  "
          f"max_W(combined_c) = {_fv(ev['max_pooled_combined_c'])} "
          f"(best W={ev['best_W_combined']})")
    print(f"  Pooled combined margin = {_fv(ev['pooled_combined_margin'])}  "
          f"(need >= {P2_COMBINED_MARGIN})")
    print(f"  Forks combined_b > max_W(combined_c): {ev['p2_forks']}/{n}  "
          f"(need >= {P2_FORKS_MIN})")
    f2_str = (
        f"YES — F2 FIRES (W={ev['f2_firing_W']} matches (b) within {F2_MATCH_MARGIN} in both regimes)"
        if ev["f2"] else "no"
    )
    print(f"  FALSIFIER F2: single W matches (b) in BOTH regimes pooled: {f2_str}")
    p2_status = "PASS" if ev["p2_pass"] else ("FALSIFIER F2" if ev["f2"] else "MIXED (not P2, not F2)")
    print(f"  => P2: {p2_status}")
    print()

    # --- Conjunct summary + VERDICT ---
    print("=" * 80)
    print("CONJUNCT SUMMARY + VERDICT")
    print("=" * 80)
    print()
    print(f"P1 (recovery >= {P1_RECOVERY_MIN} in both regimes, >= {P1_FORKS_MIN}/{n} forks): {p1_status}")
    print(f"P2 (not config-reducible, combined margin >= {P2_COMBINED_MARGIN}): {p2_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("RUNG 3 PASSES — N3 owns a regime-adaptive control surface over theta_N2 that no constant matches")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("RUNG 3: FAILED")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp164_rows.json"
    write_json_rows(run_results, score_results, ev=ev, path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    arms = {
        "P1_recovery": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"recovery(b,FR1) >= {P1_RECOVERY_MIN} in {ev['p1_forks_fr1']}/{n} forks; "
                f"recovery(b,FR2) >= {P1_RECOVERY_MIN} in {ev['p1_forks_fr2']}/{n} forks "
                f"(need >= {P1_FORKS_MIN}); "
                f"pooled FR1={_fv(ev['pooled_recovery_fr1'])} FR2={_fv(ev['pooled_recovery_fr2'])}; "
                f"F1 fired={ev['f1']}"
            ),
        },
        "P2_not_config": {
            "pass": bool(ev["p2_pass"]),
            "reason": (
                f"pooled combined_b={_fv(ev['pooled_combined_b'])} "
                f"max_W(combined_c)={_fv(ev['max_pooled_combined_c'])} (W={ev['best_W_combined']}); "
                f"margin={_fv(ev['pooled_combined_margin'])} (need >= {P2_COMBINED_MARGIN}); "
                f"forks_b_wins={ev['p2_forks']}/{n} (need >= {P2_FORKS_MIN}); "
                f"F2 fired={ev['f2']} (firing_W={ev['f2_firing_W']})"
            ),
        },
    }
    verdict_path = Path(__file__).parent / "outputs" / "exp164_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp164",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 3 load-bearing test: N3's trust-driven dial rewrite (200->400->800->1600->200) "
            "must (i) recover >= 60% of N2-only's lost diagnostic performance in both FR1 "
            "(slow alternation H=1000, UNSEEN instance) and FR2 (micro-burst), and "
            "(ii) not reduce to any single offline window setting, because FR1 requires W=1600 "
            "while FR2 requires W<=800 — disjoint optimal dials. "
            "POSITIVE: RUNG 3 PASSES — N3 owns a regime-adaptive control surface. "
            "NEGATIVE: RUNG 3 FAILED."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
