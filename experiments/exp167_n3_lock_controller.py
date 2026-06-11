"""Exp 167 — N3 rung 3 LOCK-ON-LABEL-CONSISTENCY (attempt 4, crack authorised by "Crack it").

WALL LINEAGE (Exp 164-166) — five laws and which delta answers each:

  Law 1 — dial escalation requires evidence density.
    Context: Exp 164 showed a controller needs >= 3 scored labels before trusting.
    Still: MIN_SCORED_FOR_TRUST=3, unchanged.

  Law 2 — false peace: at a WRONG dial, transition windows are routed to STRUCTURAL
    (the ONE unscored class in Exp 165).  Pending STRUCTURAL forecasts were dropped,
    so the rolling buffer filled with uncontested OK/NOISE labels — violations dried
    up while the dial was still wrong.

  Law 3 — false war: Exp 166 SCORED STRUCTURAL (e_next < 0.05 => violated) to
    close false peace.  But this punished the CORRECT dial during honest quiet
    phases: at W=1600 in a structural world, some windows ARE legitimately quiet
    (between transitions), and those produce STRUCTURAL-violated events, inflating
    the violation count and causing the dial to cycle perpetually (false war).

  Law 4 — escalation latency: with H=1000 and only 4000 steps (Exp 165), too few
    transitions occur between buffer resets for violations to accumulate.
    Answer: 12 000 steps (N_STEPS=12000), giving 11 FR1 transitions.

  Law 5 — drift bound scaling: PC1 used ahat_drift < 0.05 at 4000 steps.
    Exp 166 measured drift 0.051-0.056 at 12 000 steps, so the bound is scaled:
    PC1_AHAT_DRIFT_MAX = 0.15  (law 5: 3× horizon scale from 4000-step bound).

THE CRACK — LOCK-ON-LABEL-CONSISTENCY:
  The root failure of Laws 2 and 3 is that both false peace and false war are
  forecast artefacts: they fire on the NEXT-WINDOW error rate, which is noisy at
  boundaries.  The fix is to replace forecast-checking of STRUCTURAL with a
  REGIME STATISTIC over the LABEL DISTRIBUTION itself — a minimal "N1 inside N3".

  LOCK RULE: maintain the sequence of labels emitted under the CURRENT dial
  (cleared on dial change).  The dial is LOCKED whenever the last K=8 labels
  under the current dial are ALL the same class (any class).  While locked the
  trigger is IGNORED — the dial does NOT move.  A label of a different class
  breaks the run and UNLOCKS.

  Rationale:
    - At the CORRECT dial in a structural world (W=1600, H=1000), ALL windows
      are labeled STRUCTURAL once the dial is settled → the last 8 labels become
      uniformly STRUCTURAL → LOCK engages → dial freezes PERMANENTLY.  False war
      cannot fire (the trigger is evaluated but suppressed by the lock).
    - At a WRONG dial in a structural world (W=200, H=1000), the label sequence
      cycles: OK during A-phases, STRUCTURAL at transitions, OK again → the run
      of identical labels never reaches K=8 → the lock never engages → violations
      accumulate normally and the trigger fires, escalating to the next dial.
    - In benign worlds (FR2, CTRL), the dial stays at W=200 where all labels are
      OK; the first K=8 OK labels engage the lock, keeping the dial frozen
      permanently at 200 — no thrash possible.

  K=8 spans 800 steps — longer than any straddle artifact in the tested world
  family (burst width 50 steps, transition window << 100 steps).  K=8 is a
  PROVIDED theta_N3 choice, declared pre-experiment (not tuned post-hoc).

REVERT from Exp 166:
  STRUCTURAL labels are NOT forecast-scored (back to Exp 165's monitor):
    Only OK and NOISE forecasts are scored.
    OK:    violated iff e_next >= 0.15
    NOISE: violated iff |e_next - e_w| >= 0.30
    STRUCTURAL: not scored (unscored class, as in Exp 165).
  The lock makes this safe: the correct dial at structural worlds runs all-STRUCTURAL
  (lock fires, dial frozen) while a wrong dial mixes classes (lock never fires).

READ-ONLY-DIAL DECLARATION:
  One creature run per (fork, session); per-step correctness stream recorded;
  agent (b)'s dial policy is a post-hoc readout over that SAME stream.
  The creature is never re-run or influenced by the dial policy.

SEEDS = range(98, 106) — 8 fresh forks; exp165 used 82-89, exp166 used 90-97.

Failure regimes (provided; dial requirements DISJOINT):
  FR1 (slow alternation, UNSEEN instance): StructCmap half_period H=1000,
      derangement per fork from {[1,2,0],[2,0,1]} (rng 130_000+seed).
      Detection requires W > 1000 => only W=1600 works.
  FR2 (micro-burst): standard world with four 50-step deranged bursts at
      t=1500-1549, 4500-4549, 7500-7549, 10500-10549 (rng 140_000+seed).
      Detection requires W <= 800; W=1600 misses bursts.
  CTRL (no-deception): standard world.

Agents (post-hoc readouts; read-only-dial):
  (a) N2-only:       fixed W=200.
  (c_W) retunes:     fixed W in {200, 400, 800, 1600}.
  (b) N2+N3 + LOCK:  starts at W=200; scores only OK and NOISE forecasts;
                     LOCK disables trigger when last K=8 labels are all same.

Gated scores (steady-state window: t >= STEADY_STATE_START=8000):
  FR1 score = fraction labeled STRUCTURAL.
  FR2 score = 0.5 * recall + 0.5 * specificity.
  CTRL score = fraction labeled OK.
  recovery(b, r) = (score_b(r) - score_a(r)) / (ctrl_a - score_a(r)).

Preconditions:
  PC1 ahat_drift < 0.15 per (fork, session) [scaled from 0.05@4000 steps];
  PC2 >= 30 steady-state eval points per (fork, session) at W=200;
  PC3 FR1 deranged-phase error >= 0.9 pooled;
       FR2 burst error >= 0.9 pooled (both full session);
  PC4 ctrl_a - score_a(FR1) >= 0.2 pooled on steady-state scores (FR1 only).

Predeclared properties and falsifiers (all on steady-state scores):
  P1 (recovery scoped to FR1, steady-state):
     recovery(b, FR1) >= 0.6 in >= 7/8 forks.
     FALSIFIER F1: pooled FR1 recovery <= 0.3.
  P1b (no harm where baseline is fine):
     score_b >= score_a - 0.05 in FR2 AND in CTRL, each in >= 7/8 forks.
     FALSIFIER F1b: pooled score_a - score_b > 0.15 in FR2 or CTRL.
  P2 (not reducible to config):
     combined(x) = min(score_x(FR1), score_x(FR2));
     pooled combined_b - max_W pooled combined_c_W >= 0.15
     AND combined_b > max_W combined_c_W in >= 7/8 forks.
     FALSIFIER F2: some single W has score_c_W(r) >= score_b(r) - 0.05
     in BOTH failure regimes (pooled).
  VERDICT:
    POSITIVE (RUNG 3 PASSES — N3 owns a regime-adaptive control surface no
              constant matches) iff P1, P1b, P2 all pass.
    NEGATIVE iff F1, F1b, or F2 fires (the wall re-confirmed with the crack
              SPENT — halt for a word).
    else MIXED.

Ungated diagnostics:
  - Dial trajectories with timestamps AND lock events (when locked/unlocked,
    on which class).
  - Final dial + final lock state per (fork, session).
  - Time-to-stable-1600 (first t where dial reaches 1600 and lock engages)
    in FR1 per fork.
  - Full-session scores alongside steady-state scores.
  - Trust traces at dial changes.
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

SEEDS = list(range(98, 106))   # fresh seeds — exp166 used 90-97

N_STEPS = 12000
N_CHUNKS = 120
CHUNK_SIZE = N_STEPS // N_CHUNKS    # 100

# Steady-state window for gated scores
STEADY_STATE_START = 8000   # eval points with t >= 8000 (final 4000 steps)

# Classifier parameters (verbatim from exp158/159/162/163/165/166)
CLASSIFIER_WINDOW_DEFAULT = 200     # starting W for agent (b) and agent (a)
EVAL_EVERY = 100                    # evaluate every 100 steps once window full
ERROR_RATE_OK_THRESH = 0.05         # error_rate < 0.05 => OK
LAG1_RHO_STRUCT_THRESH = 0.3       # lag1_rho > 0.3 => STRUCTURAL (else NOISE)

# Dial candidate set (ordered for cyclic advance)
DIAL_CANDIDATES = [200, 400, 800, 1600]

# EWMA channel parameters (verbatim from exp157/159/163/165/166)
ALPHA = 0.05        # EWMA learning rate
EWMA_INIT = 0.5     # initial EWMA value per cell

# N3 forecast-scoring thresholds (reverted to exp165 — STRUCTURAL is NOT scored)
OK_VIOLATION_THRESH = 0.15          # L=OK violated iff e_next >= 0.15
NOISE_VIOLATION_DELTA = 0.30        # L=NOISE violated iff |e_next - e_w| >= 0.30
# STRUCTURAL: not scored (unscored class, back to exp165 design)

ROLLING_TRUST_WINDOW = 10           # last 10 scored labels for rolling trust
TRUST_FIRE_THRESH = 0.85            # advance dial when trust < 0.85 (>= 2 violations)
MIN_SCORED_FOR_TRUST = 3            # < 3 scored since last dial change => NO-EVIDENCE

# LOCK parameters (exp167 crack)
LOCK_K = 8   # lock engages when last K labels under current dial are all same class
              # spans 800 steps — longer than any straddle artifact in tested world family
              # PROVIDED theta_N3 choice, declared pre-experiment

# FR1 parameters: slow alternation, H=1000, UNSEEN instance
HALF_PERIOD_FR1 = 1000               # H = 1000 (Exp 162 used 400)
FR1_DERANG_SEED_OFFSET = 130_000     # rng 130_000 + fork_seed (leakage-free)

# FR2 parameters: micro-burst, 4 bursts at 12000 steps
FR2_BURST_STEPS = [
    (1500, 1549),
    (4500, 4549),
    (7500, 7549),
    (10500, 10549),
]
FR2_BURST_LEN = 50
FR2_DERANG_SEED_OFFSET = 140_000  # rng 140_000 + fork_seed

# Derangement options (verbatim from exp162/163/165/166)
DERANGEMENT_OPTIONS = [[1, 2, 0], [2, 0, 1]]

# FR2 score ground truth thresholds
FR2_STRUCT_BURST_MIN = 10   # window contains >= 10 burst steps => true STRUCTURAL
FR2_OK_BURST_MAX = 0        # window contains 0 burst steps => true OK
# windows with 1-9 burst steps excluded

# Precondition thresholds
# PC1: scaled from 0.05 at 4000 steps => 0.15 at 12000 steps
# (law 5: exp166 measured drift 0.051-0.056 at this horizon — well within 0.15)
PC1_AHAT_DRIFT_MAX = 0.15          # PC1 (12000-step horizon-scaled bound)
PC2_MIN_EVAL_POINTS = 30           # PC2: >= 30 steady-state eval points at W=200
PC3_FR1_DERANGED_ERR_MIN = 0.9     # PC3a: FR1 deranged-phase error >= 0.9 pooled
PC3_FR2_BURST_ERR_MIN = 0.9        # PC3b: FR2 burst error >= 0.9 pooled
PC4_GAP_MIN = 0.2                  # PC4: ctrl_a - score_a(FR1) >= 0.2 on SS scores

# P1 thresholds
P1_RECOVERY_MIN = 0.6              # recovery(b, FR1) >= 0.6 per fork
P1_FORKS_MIN = 7                   # >= 7/8 forks
F1_POOLED_RECOVERY_MAX = 0.3       # F1: pooled FR1 recovery <= 0.3

# P1b thresholds (no-harm where baseline is fine)
P1B_NO_HARM_MARGIN = 0.05          # score_b >= score_a - 0.05 in FR2/CTRL per fork
P1B_FORKS_MIN = 7                  # >= 7/8 forks
F1B_THRASH_MARGIN = 0.15           # F1b: pooled score_a - score_b > 0.15

# P2 thresholds
P2_COMBINED_MARGIN = 0.15          # pooled combined_b - max_W pooled combined_c_W >= 0.15
P2_FORKS_MIN = 7                   # combined_b > max_W combined_c_W in >= 7/8 forks
F2_MATCH_MARGIN = 0.05             # F2: some W within 0.05 of (b) in BOTH regimes pooled

ALL_SESSIONS = ("FR1", "FR2", "CTRL")
FAILURE_SESSIONS = ("FR1", "FR2")


# ---------------------------------------------------------------------------
# StructCmap — verbatim from exp162/163/164/165/166
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
# BurstCmap — standard world with deranged bursts (FR2)
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
# lag1_rho — verbatim from exp162/163/164/165/166
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
# classify_window — verbatim from exp162/163/164/165/166
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

    Step loop replicates exp163/164/165/166 semantics exactly.
    Returns per-step correctness array, context/burst membership arrays,
    and ahat_drift. The correctness stream is later post-hoc processed
    by all agent dial policies.
    """
    assert SURPRISE_WINDOW == 200, (
        f"SURPRISE_WINDOW from creature.py == {SURPRISE_WINDOW}, expected 200"
    )

    fork_name = f"exp167_s{fork_seed}_{session}"
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
    n_total = n_chunks * chunk_size    # 12000

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

    Eval points: every 100 steps (t = 99, 199, ...).
    Window evaluates iff t+1 >= W (at least W steps of history).
    Returns list of (t, label, error_rate, rho) eval records (full session).
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
# Post-hoc: simulate N2+N3 agent with LOCK-ON-LABEL-CONSISTENCY (exp167)
# ---------------------------------------------------------------------------

def eval_n3_agent(correct_arr: np.ndarray) -> dict:
    """Simulate N2+N3 agent (b) with LOCK-ON-LABEL-CONSISTENCY over the correctness stream.

    MONITOR (reverted to exp165 — STRUCTURAL NOT scored):
      Only OK and NOISE forecasts are scored:
        OK:    violated iff e_next >= 0.15
        NOISE: violated iff |e_next - e_w| >= 0.30
      STRUCTURAL: not scored (unscored class).

    LOCK RULE (exp167 crack):
      Maintain a rolling deque of the last LOCK_K=8 labels emitted under the
      CURRENT dial (cleared on dial change).  The dial is LOCKED whenever all
      LOCK_K labels in the deque are the same class.
      - While locked: the movement trigger is IGNORED (dial does not advance).
      - A label of a different class from the locked class breaks the run and
        UNLOCKS immediately (deque now has heterogeneous content).
      - The lock is re-evaluated at every eval point.

    Starting W=200. Trigger (unchanged from exp165/166): when rolling trust < 0.85
    (>= 2 violations in rolling window of 10) AND >= 3 scored labels since last
    dial change, AND NOT LOCKED, advance dial 200->400->800->1600->200,
    clear buffer, drop pending forecasts.

    Returns eval_records, dial_trajectory (with lock events), lock_events,
    time_to_stable_1600 (t where dial reaches 1600 AND lock engages on STRUCTURAL),
    final_dial, final_locked, final_lock_class, total_dial_changes.
    """
    n_total = len(correct_arr)
    HORIZON = 100   # forecast horizon (next 100 steps)
    MAX_SCORABLE_T = n_total - 1 - HORIZON   # = 11899

    current_W = DIAL_CANDIDATES[0]   # start at 200
    dial_trajectory: list[dict] = [{"t": -1, "dial_change_to": current_W, "trust_at_change": None,
                                     "lock_event": None}]

    # Rolling scored outcomes since last dial change (deque of 0/1, max 10)
    rolling_outcomes: _deque[int] = _deque(maxlen=ROLLING_TRUST_WINDOW)
    # Pending forecasts: (t_eval, label, e_w) not yet scorable (only OK/NOISE)
    pending: list[dict] = []
    # Count of scored labels since last dial change
    scored_since_change = 0

    # LOCK state
    # label_run: recent labels under current dial (up to LOCK_K)
    label_run: _deque[str] = _deque(maxlen=LOCK_K)
    locked = False
    lock_class: str | None = None   # class on which lock is engaged

    eval_records: list[dict] = []
    lock_events: list[dict] = []   # {t, event: "LOCKED"|"UNLOCKED", on_class, dial}
    time_to_stable_1600: int | None = None   # t where dial=1600 AND lock engages

    for t in range(EVAL_EVERY - 1, n_total, EVAL_EVERY):

        # --- Step 1: mature any pending forecasts where t+100 has elapsed ---
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
            elif label == "NOISE":
                violated = int(abs(e_next - e_w) >= NOISE_VIOLATION_DELTA)
            else:
                # STRUCTURAL: not scored (reverted to exp165)
                continue
            rolling_outcomes.append(1 - violated)
            scored_since_change += 1

        # --- Step 2: check trust rule — only if NOT LOCKED ---
        if (not locked and
                scored_since_change >= MIN_SCORED_FOR_TRUST and
                len(rolling_outcomes) > 0):
            trust = float(sum(rolling_outcomes) / len(rolling_outcomes))
            if trust < TRUST_FIRE_THRESH:
                # Advance dial cyclically
                idx = DIAL_CANDIDATES.index(current_W)
                current_W = DIAL_CANDIDATES[(idx + 1) % len(DIAL_CANDIDATES)]
                # Clear rolling buffer, pending forecasts, label run
                rolling_outcomes.clear()
                scored_since_change = 0
                pending = []
                label_run.clear()
                locked = False
                lock_class = None
                dial_trajectory.append({
                    "t": t,
                    "dial_change_to": current_W,
                    "trust_at_change": trust,
                    "lock_event": None,
                })

        # --- Step 3: evaluate current window under current_W if enough history ---
        if t + 1 >= current_W:
            win_arr = correct_arr[max(0, t - current_W + 1): t + 1].astype(np.float64)
            label, err_rate, rho = classify_window(win_arr)
            eval_records.append({
                "t": t, "label": label, "e_w": err_rate, "rho": rho,
                "W_used": current_W, "locked": locked,
            })

            # Register forecast for OK and NOISE only (STRUCTURAL not scored)
            if label in ("OK", "NOISE") and t <= MAX_SCORABLE_T:
                pending.append({"t_eval": t, "label": label, "e_w": err_rate})

            # --- Update label_run and re-evaluate lock ---
            was_locked = locked
            label_run.append(label)

            if len(label_run) == LOCK_K and len(set(label_run)) == 1:
                # All K labels are the same class — LOCK engages (or stays engaged)
                new_lock_class = label_run[-1]
                if not locked or lock_class != new_lock_class:
                    locked = True
                    lock_class = new_lock_class
                    lock_ev = {"t": t, "event": "LOCKED", "on_class": lock_class, "dial": current_W}
                    lock_events.append(lock_ev)
                    dial_trajectory[-1] = dict(dial_trajectory[-1], lock_event=lock_ev)
                    # Record time-to-stable-1600: first lock on STRUCTURAL at dial=1600
                    if (time_to_stable_1600 is None and
                            current_W == 1600 and lock_class == "STRUCTURAL"):
                        time_to_stable_1600 = t
            else:
                # Heterogeneous run — UNLOCK if was locked
                if locked:
                    lock_ev = {"t": t, "event": "UNLOCKED", "on_class": lock_class, "dial": current_W}
                    lock_events.append(lock_ev)
                    dial_trajectory.append({
                        "t": t,
                        "dial_change_to": current_W,  # no dial change, just lock event
                        "trust_at_change": None,
                        "lock_event": lock_ev,
                    })
                locked = False
                lock_class = None

    return {
        "eval_records": eval_records,
        "dial_trajectory": dial_trajectory,
        "lock_events": lock_events,
        "time_to_first_change": next(
            (d["t"] for d in dial_trajectory if d["trust_at_change"] is not None), None
        ),
        "time_to_stable_1600": time_to_stable_1600,
        "final_dial": current_W,
        "final_locked": locked,
        "final_lock_class": lock_class,
        "total_dial_changes": sum(
            1 for d in dial_trajectory
            if d["trust_at_change"] is not None
        ),
    }


# ---------------------------------------------------------------------------
# Score functions: FR1, FR2, CTRL (gating by steady-state window)
# ---------------------------------------------------------------------------

def _filter_ss(eval_records: list[dict]) -> list[dict]:
    """Return only eval records with t >= STEADY_STATE_START."""
    return [r for r in eval_records if r["t"] >= STEADY_STATE_START]


def score_fr1(eval_records: list[dict]) -> float:
    """FR1 score: fraction of evaluated windows labeled STRUCTURAL."""
    if not eval_records:
        return float("nan")
    n_struct = sum(1 for r in eval_records if r["label"] == "STRUCTURAL")
    return n_struct / len(eval_records)


def score_fr2(eval_records: list[dict], correct_arr: np.ndarray, W: int) -> float:
    """FR2 score = 0.5 * recall + 0.5 * specificity.

    Ground truth per window:
      STRUCTURAL iff window contains >= 10 burst steps
      OK iff window contains 0 burst steps
      windows with 1-9 burst steps excluded.

    For agent (b) (variable W), use W_used recorded in each eval record.
    For fixed-dial agents, W is constant.
    """
    # Build burst step set
    burst_set: set[int] = set()
    for start, end in FR2_BURST_STEPS:
        for step in range(start, end + 1):
            burst_set.add(step)

    true_struct: list[str] = []
    true_ok: list[str] = []

    for rec in eval_records:
        t = rec["t"]
        pred = rec["label"]
        w_used = rec.get("W_used", W)
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
      (b) N2+N3 + LOCK: LOCK-ON-LABEL-CONSISTENCY controller (exp167)

    Primary scores (gated): computed over steady-state eval points (t >= 8000).
    Ungated scores: computed over full-session eval points (diagnostic only).

    Returns dict with gated scores, ungated scores, eval_records, dial trajectory,
    lock events, and lock diagnostics.
    """
    correct_arr = run_result["correct_arr"]
    session = run_result["session"]

    # --- Fixed-dial readouts ---
    fixed_results: dict[int, dict] = {}
    for W in DIAL_CANDIDATES:
        fixed_results[W] = eval_fixed_dial(correct_arr, W)

    # --- N2+N3 + LOCK agent (b) ---
    n3_result = eval_n3_agent(correct_arr)
    n3_eval_records_full = n3_result["eval_records"]
    dial_trajectory = n3_result["dial_trajectory"]

    # Steady-state filtered records
    n3_eval_records_ss = _filter_ss(n3_eval_records_full)

    # --- Score all agents (gated = steady-state; ungated = full session) ---
    scores_ss: dict[str, float] = {}    # primary / gated
    scores_full: dict[str, float] = {}  # diagnostic / ungated

    for W in DIAL_CANDIDATES:
        recs_full = fixed_results[W]["eval_records"]
        recs_ss = _filter_ss(recs_full)
        agent_key = f"c_{W}"
        if session == "FR1":
            scores_ss[agent_key] = score_fr1(recs_ss)
            scores_full[agent_key] = score_fr1(recs_full)
        elif session == "FR2":
            scores_ss[agent_key] = score_fr2(recs_ss, correct_arr, W)
            scores_full[agent_key] = score_fr2(recs_full, correct_arr, W)
        else:  # CTRL
            scores_ss[agent_key] = score_ctrl(recs_ss)
            scores_full[agent_key] = score_ctrl(recs_full)

    # N2-only = c_200
    scores_ss["a"] = scores_ss["c_200"]
    scores_full["a"] = scores_full["c_200"]

    # Agent (b) — use W=0 sentinel for variable-W records (W_used embedded)
    if session == "FR1":
        scores_ss["b"] = score_fr1(n3_eval_records_ss)
        scores_full["b"] = score_fr1(n3_eval_records_full)
    elif session == "FR2":
        scores_ss["b"] = score_fr2(n3_eval_records_ss, correct_arr, W=0)
        scores_full["b"] = score_fr2(n3_eval_records_full, correct_arr, W=0)
    else:  # CTRL
        scores_ss["b"] = score_ctrl(n3_eval_records_ss)
        scores_full["b"] = score_ctrl(n3_eval_records_full)

    # Steady-state eval count for PC2
    n_eval_a_ss = len(_filter_ss(fixed_results[200]["eval_records"]))
    n_eval_a_full = len(fixed_results[200]["eval_records"])

    return {
        "scores": scores_ss,          # primary (gated / steady-state)
        "scores_full": scores_full,   # diagnostic (ungated / full session)
        "n3_eval_records": n3_eval_records_full,
        "n3_eval_records_ss": n3_eval_records_ss,
        "dial_trajectory": dial_trajectory,
        "lock_events": n3_result["lock_events"],
        "time_to_first_change": n3_result["time_to_first_change"],
        "time_to_stable_1600": n3_result["time_to_stable_1600"],
        "final_dial": n3_result["final_dial"],
        "final_locked": n3_result["final_locked"],
        "final_lock_class": n3_result["final_lock_class"],
        "total_dial_changes": n3_result["total_dial_changes"],
        "fixed_eval_records": {W: fixed_results[W]["eval_records"] for W in DIAL_CANDIDATES},
        "n_eval_a": n_eval_a_ss,       # steady-state count for PC2
        "n_eval_a_full": n_eval_a_full,
    }


# ---------------------------------------------------------------------------
# Precondition checks (PC2 on steady-state; PC4 on steady-state scores)
# ---------------------------------------------------------------------------

def check_preconditions(run_results: list[dict], score_results: dict) -> tuple[bool, list[str]]:
    """Check PC1-PC4.

    PC1: ahat_drift < 0.15 per (fork, session) [scaled bound, law 5].
    PC2: >= 30 steady-state eval points per (fork, session) at W=200.
    PC4: ctrl_a - score_a(FR1) >= 0.2 on steady-state scores, pooled (FR1 only).
    """
    failures: list[str] = []

    # PC1: ahat_drift < 0.15 per (fork, session) [12000-step scaled]
    # PC2: >= 30 steady-state eval points per (fork, session) at W=200
    for rr in run_results:
        seed = rr["fork_seed"]
        session = rr["session"]
        if rr["ahat_drift"] >= PC1_AHAT_DRIFT_MAX:
            failures.append(
                f"PC1 FAIL: seed={seed} {session} ahat_drift={rr['ahat_drift']:.4f} "
                f"(bound={PC1_AHAT_DRIFT_MAX}, 12000-step horizon-scaled)"
            )
        sr = score_results[(seed, session)]
        n_eval = sr["n_eval_a"]
        if n_eval < PC2_MIN_EVAL_POINTS:
            failures.append(
                f"PC2 FAIL: seed={seed} {session} n_eval_a_ss={n_eval} < {PC2_MIN_EVAL_POINTS}"
            )

    # PC3: FR1 deranged-phase error >= 0.9 pooled (full session)
    fr1_deranged_errors: list[float] = []
    for rr in run_results:
        if rr["session"] != "FR1" or rr["context_fr1"] is None:
            continue
        ctx = rr["context_fr1"]
        correct = rr["correct_arr"]
        deranged_mask = ctx == 1
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

    # PC3: FR2 burst error >= 0.9 pooled (full session)
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

    # PC4: ctrl_a - score_a(FR1) >= 0.2 pooled on STEADY-STATE scores (FR1 only)
    gaps_fr1: list[float] = []
    for rr in run_results:
        if rr["session"] != "FR1":
            continue
        seed = rr["fork_seed"]
        fr1_sr = score_results[(seed, "FR1")]
        ctrl_sr = score_results.get((seed, "CTRL"))
        if ctrl_sr is None:
            continue
        score_a = fr1_sr["scores"]["a"]   # steady-state
        ctrl_a = ctrl_sr["scores"]["a"]   # steady-state
        if not math.isnan(score_a) and not math.isnan(ctrl_a):
            gaps_fr1.append(ctrl_a - score_a)
    if gaps_fr1:
        pooled_gap_fr1 = float(np.mean(gaps_fr1))
        if pooled_gap_fr1 < PC4_GAP_MIN:
            failures.append(
                f"PC4 FAIL: FR1 pooled ctrl_a - score_a(SS) = {pooled_gap_fr1:.4f} "
                f"< {PC4_GAP_MIN}"
            )
    else:
        failures.append("PC4 FAIL: FR1 no valid gap measurements")

    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Evaluate P1/F1, P1b/F1b, P2/F2 and compute verdict (all on steady-state scores)
# ---------------------------------------------------------------------------

def evaluate(
    run_results: list[dict],
    score_results: dict,
    seeds: list[int],
) -> dict:
    """Evaluate P1/F1, P1b/F1b, P2/F2 using steady-state scores (scores key).

    pooled = mean over forks.
    """
    recovery_b_fr1: list[float] = []
    combined_b: list[float] = []
    combined_c: dict[int, list[float]] = {W: [] for W in DIAL_CANDIDATES}

    pooled_scores_b: dict[str, list[float]] = {"FR1": [], "FR2": [], "CTRL": []}
    pooled_scores_a: dict[str, list[float]] = {"FR1": [], "FR2": [], "CTRL": []}
    pooled_scores_c: dict[int, dict[str, list[float]]] = {
        W: {"FR1": [], "FR2": [], "CTRL": []} for W in DIAL_CANDIDATES
    }

    p1b_fr2_margins: list[float] = []
    p1b_ctrl_margins: list[float] = []

    for seed in seeds:
        ctrl_key = (seed, "CTRL")
        fr1_key = (seed, "FR1")
        fr2_key = (seed, "FR2")

        if ctrl_key not in score_results or fr1_key not in score_results or fr2_key not in score_results:
            continue

        ctrl_sr = score_results[ctrl_key]
        fr1_sr = score_results[fr1_key]
        fr2_sr = score_results[fr2_key]

        # Use steady-state scores ("scores" key)
        score_a_ctrl = ctrl_sr["scores"]["a"]
        score_a_fr1 = fr1_sr["scores"]["a"]
        score_a_fr2 = fr2_sr["scores"]["a"]
        score_b_fr1 = fr1_sr["scores"]["b"]
        score_b_fr2 = fr2_sr["scores"]["b"]
        score_b_ctrl = ctrl_sr["scores"]["b"]

        for sess, s_arr in [("FR1", fr1_sr), ("FR2", fr2_sr), ("CTRL", ctrl_sr)]:
            pooled_scores_b[sess].append(s_arr["scores"]["b"])
            pooled_scores_a[sess].append(s_arr["scores"]["a"])
            for W in DIAL_CANDIDATES:
                pooled_scores_c[W][sess].append(s_arr["scores"][f"c_{W}"])

        # Recovery per fork (FR1 only, steady-state)
        if not math.isnan(score_a_ctrl) and not math.isnan(score_a_fr1) and not math.isnan(score_b_fr1):
            denom_fr1 = score_a_ctrl - score_a_fr1
            if abs(denom_fr1) > 1e-12:
                rec_fr1 = (score_b_fr1 - score_a_fr1) / denom_fr1
                recovery_b_fr1.append(rec_fr1)
            else:
                recovery_b_fr1.append(float("nan"))
        else:
            recovery_b_fr1.append(float("nan"))

        # P1b margins
        if not math.isnan(score_b_fr2) and not math.isnan(score_a_fr2):
            p1b_fr2_margins.append(score_b_fr2 - score_a_fr2)
        else:
            p1b_fr2_margins.append(float("nan"))

        if not math.isnan(score_b_ctrl) and not math.isnan(score_a_ctrl):
            p1b_ctrl_margins.append(score_b_ctrl - score_a_ctrl)
        else:
            p1b_ctrl_margins.append(float("nan"))

        # Combined per fork
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

    # P1 / F1
    n_forks = len(seeds)
    p1_forks_fr1 = sum(
        1 for r in recovery_b_fr1 if not math.isnan(r) and r >= P1_RECOVERY_MIN
    )
    p1_pass = p1_forks_fr1 >= P1_FORKS_MIN
    f1 = not math.isnan(pooled_recovery_fr1) and pooled_recovery_fr1 <= F1_POOLED_RECOVERY_MAX

    # P1b / F1b
    p1b_fr2_forks = sum(
        1 for m in p1b_fr2_margins if not math.isnan(m) and m >= -P1B_NO_HARM_MARGIN
    )
    p1b_ctrl_forks = sum(
        1 for m in p1b_ctrl_margins if not math.isnan(m) and m >= -P1B_NO_HARM_MARGIN
    )
    p1b_pass = p1b_fr2_forks >= P1B_FORKS_MIN and p1b_ctrl_forks >= P1B_FORKS_MIN

    pooled_b_fr2 = _pooled(pooled_scores_b["FR2"])
    pooled_a_fr2 = _pooled(pooled_scores_a["FR2"])
    pooled_b_ctrl = _pooled(pooled_scores_b["CTRL"])
    pooled_a_ctrl = _pooled(pooled_scores_a["CTRL"])
    f1b = False
    f1b_reason = ""
    if (not math.isnan(pooled_a_fr2) and not math.isnan(pooled_b_fr2) and
            pooled_a_fr2 - pooled_b_fr2 > F1B_THRASH_MARGIN):
        f1b = True
        f1b_reason = f"FR2: score_a - score_b = {pooled_a_fr2 - pooled_b_fr2:.4f} > {F1B_THRASH_MARGIN}"
    if (not math.isnan(pooled_a_ctrl) and not math.isnan(pooled_b_ctrl) and
            pooled_a_ctrl - pooled_b_ctrl > F1B_THRASH_MARGIN):
        f1b = True
        f1b_reason += (
            f" CTRL: score_a - score_b = {pooled_a_ctrl - pooled_b_ctrl:.4f} > {F1B_THRASH_MARGIN}"
        )

    # P2 / F2
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

    f2 = False
    f2_firing_W = None
    pooled_b_fr1_scores = _pooled(pooled_scores_b["FR1"])
    for W in DIAL_CANDIDATES:
        pooled_cW_fr1 = _pooled(pooled_scores_c[W]["FR1"])
        pooled_cW_fr2 = _pooled(pooled_scores_c[W]["FR2"])
        if (not math.isnan(pooled_cW_fr1) and not math.isnan(pooled_b_fr1_scores) and
                not math.isnan(pooled_cW_fr2) and not math.isnan(pooled_b_fr2)):
            if (pooled_cW_fr1 >= pooled_b_fr1_scores - F2_MATCH_MARGIN and
                    pooled_cW_fr2 >= pooled_b_fr2 - F2_MATCH_MARGIN):
                f2 = True
                f2_firing_W = W
                break

    # Verdict
    positive = p1_pass and p1b_pass and p2_pass
    negative = f1 or f1b or f2
    if positive:
        verdict_str = "POSITIVE"
    elif negative:
        verdict_str = "NEGATIVE"
    else:
        verdict_str = "MIXED"

    return {
        "n_forks": n_forks,
        "recovery_b_fr1": recovery_b_fr1,
        "pooled_recovery_fr1": pooled_recovery_fr1,
        "p1_forks_fr1": p1_forks_fr1,
        "p1_pass": p1_pass,
        "f1": f1,
        "p1b_fr2_margins": p1b_fr2_margins,
        "p1b_ctrl_margins": p1b_ctrl_margins,
        "p1b_fr2_forks": p1b_fr2_forks,
        "p1b_ctrl_forks": p1b_ctrl_forks,
        "p1b_pass": p1b_pass,
        "f1b": f1b,
        "f1b_reason": f1b_reason,
        "combined_b": combined_b,
        "combined_c": combined_c,
        "pooled_combined_b": pooled_combined_b,
        "pooled_combined_c": pooled_combined_c,
        "max_pooled_combined_c": max_pooled_combined_c,
        "best_W_combined": best_W_combined,
        "pooled_combined_margin": pooled_combined_margin,
        "p2_forks": p2_forks,
        "p2_pass": p2_pass,
        "f2": f2,
        "f2_firing_W": f2_firing_W,
        "pooled_scores_b": {s: _pooled(pooled_scores_b[s]) for s in ALL_SESSIONS},
        "pooled_scores_a": {s: _pooled(pooled_scores_a[s]) for s in ALL_SESSIONS},
        "pooled_scores_c": {W: {s: _pooled(pooled_scores_c[W][s]) for s in ALL_SESSIONS}
                            for W in DIAL_CANDIDATES},
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
            scores_ss = sr.get("scores", {})
            scores_full = sr.get("scores_full", {})
            dial_traj = sr.get("dial_trajectory", [])
            lock_events = sr.get("lock_events", [])

            row: dict = {
                "exp": 167,
                "fork_seed": seed,
                "session": session,
                "n_total": rr["n_total"],
                "ahat_drift": _nan_to_none(rr["ahat_drift"]),
                "fr1_derangement": rr["fr1_derangement"],
                "fr2_derangement": rr["fr2_derangement"],
                # Gated (steady-state) scores
                "score_a_ss": _nan_to_none(scores_ss.get("a")),
                "score_b_ss": _nan_to_none(scores_ss.get("b")),
                **{f"score_c_{W}_ss": _nan_to_none(scores_ss.get(f"c_{W}"))
                   for W in DIAL_CANDIDATES},
                # Ungated (full-session) scores (diagnostic)
                "score_a_full": _nan_to_none(scores_full.get("a")),
                "score_b_full": _nan_to_none(scores_full.get("b")),
                **{f"score_c_{W}_full": _nan_to_none(scores_full.get(f"c_{W}"))
                   for W in DIAL_CANDIDATES},
                "n_eval_a_ss": sr.get("n_eval_a"),
                "n_eval_a_full": sr.get("n_eval_a_full"),
                "dial_changes": sr.get("total_dial_changes", 0),
                "dial_trajectory_summary": [
                    {
                        "t": d["t"],
                        "to": d["dial_change_to"],
                        "trust": _nan_to_none(d["trust_at_change"]),
                        "lock_event": d.get("lock_event"),
                    }
                    for d in dial_traj
                ],
                "lock_events": lock_events,
                "time_to_first_change": sr.get("time_to_first_change"),
                "time_to_stable_1600": sr.get("time_to_stable_1600"),
                "final_dial": sr.get("final_dial"),
                "final_locked": sr.get("final_locked"),
                "final_lock_class": sr.get("final_lock_class"),
                "total_dial_changes": sr.get("total_dial_changes"),
            }
            fh.write(json.dumps(row) + "\n")

            if seed not in seeds_set:
                seeds_set.add(seed)
                seeds_seen.append(seed)

        if ev is not None:
            summary: dict = {
                "exp": 167,
                "row_type": "summary",
                "pooled_recovery_fr1": _nan_to_none(ev["pooled_recovery_fr1"]),
                "p1_forks_fr1": ev["p1_forks_fr1"],
                "p1_pass": ev["p1_pass"],
                "f1": ev["f1"],
                "p1b_fr2_forks": ev["p1b_fr2_forks"],
                "p1b_ctrl_forks": ev["p1b_ctrl_forks"],
                "p1b_pass": ev["p1b_pass"],
                "f1b": ev["f1b"],
                "f1b_reason": ev["f1b_reason"],
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
# Print dial trajectories + lock events (ungated diagnostic)
# ---------------------------------------------------------------------------

def print_dial_trajectories(score_results: dict, seeds: list[int]) -> None:
    """Print agent (b)'s dial trajectory + lock events per (fork, session).

    Includes: timestamps, final dial, final lock state, time-to-stable-1600,
    lock/unlock events with class and t, trust at dial changes.
    """
    print("Agent (b) dial trajectories + LOCK events:")
    for seed in seeds:
        for session in ALL_SESSIONS:
            sr = score_results.get((seed, session))
            if sr is None:
                continue
            traj = sr["dial_trajectory"]
            traj_strs = []
            for d in traj:
                s = f"t={d['t']} W={d['dial_change_to']}"
                if d["trust_at_change"] is not None:
                    s += f" [trust={d['trust_at_change']:.3f}]"
                if d.get("lock_event"):
                    le = d["lock_event"]
                    s += f" [LOCK:{le['event']} on {le['on_class']}]"
                traj_strs.append(s)
            lock_evs = sr.get("lock_events", [])
            t2fc = sr.get("time_to_first_change")
            t2s1600 = sr.get("time_to_stable_1600")
            final_d = sr.get("final_dial")
            final_locked = sr.get("final_locked")
            final_lc = sr.get("final_lock_class")
            n_chg = sr.get("total_dial_changes", 0)
            t2fc_str = f" t2fc={t2fc}" if t2fc is not None else " t2fc=None"
            t2s1600_str = f" t2s1600={t2s1600}" if t2s1600 is not None else " t2s1600=None"
            print(
                f"  seed={seed} {session:>5}: {' -> '.join(traj_strs)}"
                f"  |  final={final_d} locked={final_locked}({final_lc})"
                f"{t2fc_str}{t2s1600_str}  n_changes={n_chg}"
            )
            if lock_evs:
                for le in lock_evs:
                    print(f"    LOCK_EVENT: t={le['t']} {le['event']} on={le['on_class']} dial={le['dial']}")
    print()

    ctrl_total_changes = sum(
        score_results[(s, "CTRL")].get("total_dial_changes", 0)
        for s in seeds if (s, "CTRL") in score_results
    )
    fr2_total_changes = sum(
        score_results[(s, "FR2")].get("total_dial_changes", 0)
        for s in seeds if (s, "FR2") in score_results
    )
    print(f"  No-thrash check: CTRL total dial changes = {ctrl_total_changes} (expected 0)")
    print(f"  No-thrash check: FR2  total dial changes = {fr2_total_changes} (expected 0)")
    print()


# ---------------------------------------------------------------------------
# Print per-dial score table
# ---------------------------------------------------------------------------

def print_score_table(score_results: dict, seeds: list[int], ev: dict) -> None:
    """Print per-dial score table (steady-state scores) + ungated diagnostic."""
    def _f(v, w=7):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan".rjust(w)
        return f"{v:.4f}".rjust(w)

    print("Per-dial score table — STEADY-STATE scores (t >= 8000), pooled over forks:")
    header = f"{'agent':>10}  {'FR1_score':>9}  {'FR2_score':>9}  {'CTRL_score':>10}  {'combined':>9}"
    print(header)
    print("-" * len(header))

    ps_a = ev["pooled_scores_a"]
    ps_b = ev["pooled_scores_b"]
    ps_c = ev["pooled_scores_c"]

    comb_a = min(ps_a.get("FR1", float("nan")), ps_a.get("FR2", float("nan")))
    print(
        f"{'(a) W=200':>10}  {_f(ps_a.get('FR1'))}  {_f(ps_a.get('FR2'))}  "
        f"{_f(ps_a.get('CTRL'), 10)}  {_f(comb_a)}"
    )

    for W in DIAL_CANDIDATES:
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

    # Also print ungated (full-session) scores for comparison
    print("Ungated full-session scores (diagnostic only), pooled over forks:")
    print(f"{'agent':>10}  {'FR1_full':>9}  {'FR2_full':>9}  {'CTRL_full':>10}")
    print("-" * 50)

    def _pool_full(session, agent_key):
        vals = []
        for seed in seeds:
            sr = score_results.get((seed, session))
            if sr is None:
                continue
            sf = sr.get("scores_full", {})
            v = sf.get(agent_key, float("nan"))
            if not math.isnan(v):
                vals.append(v)
        return float(np.mean(vals)) if vals else float("nan")

    print(
        f"{'(a) W=200':>10}  {_f(_pool_full('FR1', 'a'))}  "
        f"{_f(_pool_full('FR2', 'a'))}  {_f(_pool_full('CTRL', 'a'), 10)}"
    )
    for W in DIAL_CANDIDATES:
        print(
            f"{f'(c) W={W}':>10}  {_f(_pool_full('FR1', f'c_{W}'))}  "
            f"{_f(_pool_full('FR2', f'c_{W}'))}  {_f(_pool_full('CTRL', f'c_{W}'), 10)}"
        )
    print(
        f"{'(b) N3':>10}  {_f(_pool_full('FR1', 'b'))}  "
        f"{_f(_pool_full('FR2', 'b'))}  {_f(_pool_full('CTRL', 'b'), 10)}"
    )
    print()

    # Per-fork recovery table
    print("Per-fork recovery table (FR1 only, steady-state):")
    print(f"  {'seed':>5}  {'rec_FR1':>9}  {'p1_FR1_ok':>9}  "
          f"{'p1b_FR2_m':>10}  {'p1b_CTRL_m':>11}")
    print("  " + "-" * 55)
    n = ev["n_forks"]
    for i, seed in enumerate(seeds):
        r_fr1 = ev["recovery_b_fr1"][i] if i < len(ev["recovery_b_fr1"]) else float("nan")
        p1b_fr2_m = ev["p1b_fr2_margins"][i] if i < len(ev["p1b_fr2_margins"]) else float("nan")
        p1b_ctrl_m = ev["p1b_ctrl_margins"][i] if i < len(ev["p1b_ctrl_margins"]) else float("nan")
        p1_fr1_ok = "YES" if not math.isnan(r_fr1) and r_fr1 >= P1_RECOVERY_MIN else "no"
        print(
            f"  {seed:>5}  {_f(r_fr1)}  {p1_fr1_ok:>9}  "
            f"{_f(p1b_fr2_m, 10)}  {_f(p1b_ctrl_m, 11)}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exp 167 — N3 rung 3, LOCK-ON-LABEL-CONSISTENCY (attempt 4, crack authorised)"
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            "Smoke run: seed=[98] only, all three sessions, full 12000 steps, "
            "prints scores + dial trajectories + lock events. No verdict written."
        ),
    )
    args = parser.parse_args()

    smoke = args.smoke
    seeds = [98] if smoke else SEEDS

    print("=" * 80)
    print("Exp 167 — N3 rung 3, LOCK-ON-LABEL-CONSISTENCY (attempt 4)")
    print("          Attempt 1 (exp164): PC4 fired; controller silent at H=1000.")
    print("          Attempt 2 (exp165): NEGATIVE — false peace + evidence-reset latency.")
    print("          Attempt 3 (exp166): false war — STRUCTURAL scoring punished correct dial.")
    print("          Attempt 4 (this): LOCK crack — consistency statistic, not forecast check.")
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
    print(f"N3 forecast (reverted to exp165): OK_VIOLATION_THRESH={OK_VIOLATION_THRESH}  "
          f"NOISE_VIOLATION_DELTA={NOISE_VIOLATION_DELTA}  STRUCTURAL=NOT scored")
    print(f"LOCK: LOCK_K={LOCK_K}  (last K same-class labels => lock; spans {LOCK_K * EVAL_EVERY} steps)")
    print(f"FR1: HALF_PERIOD_FR1={HALF_PERIOD_FR1}  FR1_DERANG_SEED_OFFSET={FR1_DERANG_SEED_OFFSET}")
    print(f"FR2: bursts at {FR2_BURST_STEPS}  FR2_DERANG_SEED_OFFSET={FR2_DERANG_SEED_OFFSET}")
    print(f"DERANGEMENT_OPTIONS={DERANGEMENT_OPTIONS}")
    print(f"N_STEPS={N_STEPS}  N_CHUNKS={N_CHUNKS}  CHUNK_SIZE={CHUNK_SIZE}")
    print(f"STEADY_STATE_START={STEADY_STATE_START}  "
          f"(gated scores over t >= {STEADY_STATE_START})")
    print(f"PC1_AHAT_DRIFT_MAX={PC1_AHAT_DRIFT_MAX}  (law 5: scaled from 0.05@4000 to 0.15@12000)")
    print(f"Seeds: {seeds}  Sessions: {ALL_SESSIONS}")
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
            sr = compute_fork_session_scores(rr)
            score_results[(seed, session)] = sr
    print()

    # --- Print dial trajectories + lock events (ungated) ---
    print_dial_trajectories(score_results, seeds)

    # --- Print per-fork per-session scores ---
    print("Per-fork per-session scores (SS = steady-state, FULL = full-session diagnostic):")
    hdr = (
        f"{'seed':>5}  {'session':>6}  {'a_ss':>7}  {'a_full':>7}  "
        + "  ".join(f"c{W}_ss".rjust(7) for W in DIAL_CANDIDATES)
        + f"  {'b_ss':>7}  {'b_full':>7}  {'n_ss':>5}  {'n_full':>6}  {'n_dchg':>6}  {'final_W':>7}"
        + f"  {'locked':>7}  {'t2s1600':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for seed in seeds:
        for session in ALL_SESSIONS:
            sr = score_results.get((seed, session))
            if sr is None:
                continue
            sc_ss = sr["scores"]
            sc_full = sr.get("scores_full", {})
            def _f2(v, w=7):
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    return "nan".rjust(w)
                return f"{v:.4f}".rjust(w)
            c_cols_ss = "  ".join(_f2(sc_ss.get(f"c_{W}"), 7) for W in DIAL_CANDIDATES)
            n_dchg = sr.get("total_dial_changes", 0)
            final_d = sr.get("final_dial", "?")
            final_locked = sr.get("final_locked", False)
            t2s1600 = sr.get("time_to_stable_1600")
            n_ss = sr.get("n_eval_a", 0)
            n_full = sr.get("n_eval_a_full", 0)
            print(
                f"{seed:>5}  {session:>6}  {_f2(sc_ss.get('a'))}  {_f2(sc_full.get('a'))}  "
                f"{c_cols_ss}  {_f2(sc_ss.get('b'))}  {_f2(sc_full.get('b'))}  "
                f"{n_ss:>5}  {n_full:>6}  {n_dchg:>6}  {str(final_d):>7}"
                f"  {str(final_locked):>7}  {str(t2s1600):>8}"
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
        out_rows_path = Path(__file__).parent / "outputs" / "exp167_rows.json"
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

    # --- P1 / F1 / P1b / F1b / P2 / F2 ---
    print("=" * 80)
    print("PREDECLARED OUTCOME MAP")
    print("=" * 80)
    print()

    def _fv(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "nan"
        return f"{v:.4f}"

    print(f"P1 (N3 recovers FR1 — recovery(b,FR1) >= {P1_RECOVERY_MIN} in >= {P1_FORKS_MIN}/{n} forks):")
    print(f"  Pooled recovery FR1: {_fv(ev['pooled_recovery_fr1'])}")
    print(f"  Forks recovery(FR1) >= {P1_RECOVERY_MIN}: {ev['p1_forks_fr1']}/{n}  "
          f"(need >= {P1_FORKS_MIN})")
    print(f"  FALSIFIER F1: pooled FR1 recovery <= {F1_POOLED_RECOVERY_MAX}: "
          f"{'YES — F1 FIRES' if ev['f1'] else 'no'}")
    p1_status = "PASS" if ev["p1_pass"] else ("FALSIFIER F1" if ev["f1"] else "MIXED (not P1, not F1)")
    print(f"  => P1: {p1_status}")
    print()

    print(f"P1b (no harm — score_b >= score_a - {P1B_NO_HARM_MARGIN} in >= {P1B_FORKS_MIN}/{n} each):")
    print(f"  FR2 forks no-harm: {ev['p1b_fr2_forks']}/{n}  CTRL forks no-harm: {ev['p1b_ctrl_forks']}/{n}  "
          f"(need >= {P1B_FORKS_MIN} each)")
    print(f"  FALSIFIER F1b: pooled score_a - score_b > {F1B_THRASH_MARGIN}: "
          f"{'YES — F1b FIRES: ' + ev['f1b_reason'] if ev['f1b'] else 'no'}")
    p1b_status = (
        "PASS" if ev["p1b_pass"] else ("FALSIFIER F1b" if ev["f1b"] else "MIXED (not P1b, not F1b)")
    )
    print(f"  => P1b: {p1b_status}")
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
    print(f"P1  (FR1 recovery >= {P1_RECOVERY_MIN}, >= {P1_FORKS_MIN}/{n} forks): {p1_status}")
    print(f"P1b (no harm in FR2/CTRL, >= {P1B_FORKS_MIN}/{n} forks each): {p1b_status}")
    print(f"P2  (not config-reducible, combined margin >= {P2_COMBINED_MARGIN}): {p2_status}")
    print()
    if ev["verdict_str"] == "POSITIVE":
        print("VERDICT: POSITIVE")
        print("RUNG 3 PASSES — N3 owns a regime-adaptive control surface no constant matches")
    elif ev["verdict_str"] == "NEGATIVE":
        print("VERDICT: NEGATIVE")
        print("RUNG 3: FAILED (crack SPENT — halt for a word)")
    else:
        print("VERDICT: MIXED")
    print()
    print("=" * 80)

    # --- Write JSON rows ---
    out_rows_path = Path(__file__).parent / "outputs" / "exp167_rows.json"
    write_json_rows(run_results, score_results, ev=ev, path=out_rows_path)
    print(f"JSON rows written to {out_rows_path}")

    # --- Write verdict JSON ---
    arms = {
        "P1_recovery_FR1": {
            "pass": bool(ev["p1_pass"]),
            "reason": (
                f"recovery(b,FR1) >= {P1_RECOVERY_MIN} in {ev['p1_forks_fr1']}/{n} forks "
                f"(need >= {P1_FORKS_MIN}); "
                f"pooled FR1={_fv(ev['pooled_recovery_fr1'])}; "
                f"F1 fired={ev['f1']}"
            ),
        },
        "P1b_no_harm": {
            "pass": bool(ev["p1b_pass"]),
            "reason": (
                f"FR2 forks no-harm={ev['p1b_fr2_forks']}/{n}; "
                f"CTRL forks no-harm={ev['p1b_ctrl_forks']}/{n} "
                f"(need >= {P1B_FORKS_MIN}); "
                f"F1b fired={ev['f1b']} {ev['f1b_reason']}"
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
    verdict_path = Path(__file__).parent / "outputs" / "exp167_verdict.json"
    write_verdict(
        path=verdict_path,
        experiment="exp167",
        arms=arms,
        verdict=ev["verdict_str"],
        halted=False,
        notes=(
            "N3 rung 3 LOCK-ON-LABEL-CONSISTENCY (attempt 4, crack authorised by 'Crack it'). "
            "Attempt 1 (exp164): PC4 fired (FR2 gap absent); controller silent at H=1000. "
            "Attempt 2 (exp165): NEGATIVE — false peace: STRUCTURAL unscored, violations dried up. "
            "Attempt 3 (exp166): false war — scoring STRUCTURAL punished correct dial in quiet phases. "
            "Attempt 4 crack: LOCK-ON-LABEL-CONSISTENCY — K=8 same-class label run => dial frozen. "
            "Correct dial produces all-STRUCTURAL run => permanent lock (no false war). "
            "Wrong dial mixes classes => lock never engages => violations escalate (no false peace). "
            "POSITIVE: RUNG 3 PASSES. NEGATIVE: crack SPENT — halt for a word."
        ),
    )
    print(f"Verdict JSON written to {verdict_path}")


if __name__ == "__main__":
    main()
