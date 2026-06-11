"""
Exp 155 — N3a-min: shadow failure-mode diagnostic on the four canonical regimes.

Direction: n3-bounded-map-open-world (docs/specs/n3-open-world.md §9, the first
minimal experiment). This is the GATE of the N3 ladder: before any repair
controller is allowed to act, the diagnostic must show — in SHADOW mode (it
predicts, it never touches the creature) — that the creature's own learning
signals carry separable information about WHY surprise is high.

Four regimes, held on a SHARED 4x4 / 4-color geometry so the regime is the only
varying factor (no grid-size / color-count confound):
  A static-bounded  (localized 2x2-block colors)        -> stable_known
  C structural      (scattered aliased layout, Exp 143)  -> structural_inadequacy
  D noisy           (localized base + p_true observation) -> irreducible_noise
  E nonstationary   (localized base, late color remap)    -> nonstationarity

The diagnostic (active_loop.n3_diagnostics, declared rule-based/provided) reads
ONLY internal signals (surprise level/slope, ceiling events, the penalized
held-out replay split test) — never the world label. The load-bearing
discriminator (structural vs. noise) is the SAME penalized replay test the
validated grow decision uses (Exp 152/153/154).

Honesty: the classifier rules+thresholds are PROVIDED by the designer and tied to
already-validated growth constants (ALARM_THRESH=0.7, K_PENALTY=0.05), fixed
BEFORE this run. The empirical question is whether the signals are separable, not
whether the rules are clever. The base M3 loop (single component per color, NO
growth) is copied verbatim-in-arithmetic from exp154's Arm B step body.

PREDECLARED (binding):
  Pass iff, pooled over fresh seeds x layouts:
    (P1) macro-F1 over the four regimes >= 0.70, AND
    (P2) confusion(irreducible_noise -> structural_inadequacy) < 0.20, AND
    (P3) confusion(nonstationarity   -> structural_inadequacy) < 0.20.
  FALSIFIER (any => NEGATIVE, halt the controller work, log which):
    macro-F1 <= 0.40 (~chance for 4 classes), OR either critical confusion >= 0.20.
  Three-way: all of P1-P3 => POSITIVE; clean falsifier => NEGATIVE; otherwise MIXED.

Seeds 20..27 (never run on any prior protocol). Layout seeds 7/11/13 (aliased).
Run: uv run --python .venv python experiments/exp155_n3a_min_shadow.py
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW
from active_loop.creature_continuous import ContinuousPlace
from active_loop.growth import (
    check_ceiling,
    mixture_emission_moments,
    mixture_predictive_logprobs,
)
from active_loop.meta_metrics import (
    confusion_matrix,
    confusion_rate,
    format_confusion,
    macro_f1,
    per_class_prf,
)
from active_loop.n3_diagnostics import (
    REGIMES_4,
    RunSignals,
    classify_regime,
    structural_gain,
)
from active_loop.verdict import write_verdict
from active_loop import worlds

# ---------------------------------------------------------------------------
# Geometry (shared 4x4 / 4-color) and substrate constants (from exp154)
# ---------------------------------------------------------------------------
ROWS, COLS = 4, 4
N_CELLS = ROWS * COLS          # 16
N_COLORS = 4
CELLS_PER_COLOR = 4
D = 2

CELL_CENTERS = np.array([[float(c), float(r)] for r in range(ROWS) for c in range(COLS)])
ARENA = (0.0, float(COLS - 1), 0.0, float(ROWS - 1))
ARENA_CENTER = np.array([(COLS - 1) / 2.0, (ROWS - 1) / 2.0])
ACTION_DELTA = {
    0: np.array([0.0, -1.0]), 1: np.array([0.0, +1.0]),
    2: np.array([-1.0, 0.0]), 3: np.array([+1.0, 0.0]),
}
Q = np.diag(np.array([0.05 ** 2, 0.05 ** 2]))
KAPPA0, NU0 = 1.0, 4.0
S0 = (0.35 ** 2 * (NU0 - D - 1)) * np.eye(D)

# Run lengths (smaller than exp154's 8000 — shadow loop has no growth EM per step)
T_PHASE1 = 1500
T_PHASE2 = 3000
T_TOTAL = T_PHASE1 + T_PHASE2
REMAP_AT_PHASE2 = 2000          # nonstationary: remap fires here; final window is post-remap
FINAL_WINDOW = 600              # final-window for per-color loudness & late_mean
EARLY_LO, EARLY_HI = 200, 700   # phase2 early window (post burn-in, pre-remap) for early_mean
SURPRISE_WINDOW = 200
REPLAY_WINDOW = 400
P_TRUE = 0.7                    # noisy world fidelity; analytic floor(0.7,4) ~ 0.94 nats

SEEDS = list(range(20, 28))      # fresh: never run on any prior protocol
LAYOUT_SEEDS = [7, 11, 13]


def move(cell: int, action: int) -> int:
    r, c = divmod(cell, COLS)
    if action == 0:
        r = max(0, r - 1)
    elif action == 1:
        r = min(ROWS - 1, r + 1)
    elif action == 2:
        c = max(0, c - 1)
    else:
        c = min(COLS - 1, c + 1)
    return r * COLS + c


def localized_cmap() -> np.ndarray:
    """4 colors as contiguous 2x2 quadrant blocks — maximally single-component-adequate."""
    cmap = np.empty(N_CELLS, dtype=int)
    for cell in range(N_CELLS):
        r, c = divmod(cell, COLS)
        cmap[cell] = (r // 2) * 2 + (c // 2)
    return cmap


def remapped_cmap(base: np.ndarray, seed: int) -> np.ndarray:
    """Abrupt nonstationary change: permute the color labels (a genuine world change)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_COLORS)
    return np.array([perm[c] for c in base], dtype=int)


def run_shadow(
    cmap: np.ndarray,
    seed: int,
    *,
    noisy: bool = False,
    cmap_after: np.ndarray | None = None,
) -> tuple[RunSignals, dict]:
    """Run the base M3 loop (single component per color, NO growth) and diagnose.

    Returns (RunSignals, debug_dict). The diagnostic never sees the regime label.
    """
    act_rng = np.random.default_rng(1000 + seed)
    obs_rng = np.random.default_rng(2000 + seed)   # noise draws
    em_rng = np.random.default_rng(3000 + seed)    # structural_gain EM
    actions = act_rng.integers(0, 4, size=T_TOTAL)

    cp = ContinuousPlace(ARENA_CENTER.copy(), np.diag(np.array([4.0, 4.0])), ARENA)
    components: list[list[tuple[float, NIW]]] = []
    counts: list[list[int]] = []
    for _ in range(N_COLORS):
        components.append([(1.0, NIW(m=ARENA_CENTER.copy(), kappa=KAPPA0, nu=NU0, S=S0.copy()))])
        counts.append([1])

    surprise_buf: deque = deque(maxlen=SURPRISE_WINDOW)
    replay_buf: deque = deque(maxlen=REPLAY_WINDOW)
    phase2_pairs: list[tuple[int, float]] = []     # (obs_k, surprise) for phase2
    ceiling_second_half = False
    true_cell = 0

    for t in range(T_TOTAL):
        # --- true color (with nonstationary remap / observation noise) ---
        active_cmap = cmap
        if cmap_after is not None and (t - T_PHASE1) >= REMAP_AT_PHASE2:
            active_cmap = cmap_after
        true_color = int(active_cmap[true_cell])
        if noisy and obs_rng.random() >= P_TRUE:
            others = [c for c in range(N_COLORS) if c != true_color]
            obs_k = int(others[obs_rng.integers(0, len(others))])
        else:
            obs_k = true_color

        Sigma_p_diag = np.maximum(np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]]), 1e-9)
        mu_p = cp.mu
        replay_buf.append((obs_k, mu_p.copy(), Sigma_p_diag.copy()))

        log_probs = mixture_predictive_logprobs(mu_p, Sigma_p_diag, components, convention="normalized")
        surprise_t = float(-log_probs[obs_k])
        surprise_buf.append(surprise_t)

        if t >= T_PHASE1:
            phase2_pairs.append((obs_k, surprise_t))
        if t >= T_TOTAL // 2 and check_ceiling(surprise_buf):
            ceiling_second_half = True

        # --- place + single-component update (no growth) ---
        mu_mix, Sigma_mix_diag, hard_idx = mixture_emission_moments(mu_p, Sigma_p_diag, obs_k, components)
        cp.update(mu_mix, np.diag(Sigma_mix_diag))
        post_mu = cp.mu
        post_Sigma_diag = np.maximum(np.array([cp.Sigma[0, 0], cp.Sigma[1, 1]]), 1e-9)
        old_niw = components[obs_k][hard_idx][1]
        components[obs_k][hard_idx] = (components[obs_k][hard_idx][0],
                                       old_niw.update_moments(post_mu, np.diag(post_Sigma_diag)))
        counts[obs_k][hard_idx] += 1

        action = int(actions[t])
        true_cell = move(true_cell, action)
        cp.predict_clamped_moments(ACTION_DELTA[action], Q)

    # --- feature extraction (no world label) ---
    final = phase2_pairs[-FINAL_WINDOW:]
    early = phase2_pairs[EARLY_LO:EARLY_HI]
    per_color_final = np.full(N_COLORS, -np.inf)
    for k in range(N_COLORS):
        vals = [s for (c, s) in final if c == k]
        if vals:
            per_color_final[k] = float(np.mean(vals))
    loud_color = int(np.argmax(per_color_final))
    loud_mean = float(per_color_final[loud_color])
    early_mean = float(np.mean([s for (_, s) in early])) if early else float("inf")
    late_mean = float(np.mean([s for (_, s) in final])) if final else float("inf")

    loud_pairs = [(mu.copy(), sig.copy()) for (c, mu, sig) in replay_buf if c == loud_color]
    structural, sg = structural_gain(components, loud_color, loud_pairs, em_rng)

    sig = RunSignals(loud_mean=loud_mean, ceiling_fired=ceiling_second_half,
                     early_mean=early_mean, late_mean=late_mean, structural=structural)
    debug = {"loud_color": loud_color, "loud_mean": round(loud_mean, 3),
             "early_mean": round(early_mean, 3), "late_mean": round(late_mean, 3),
             "ceiling": ceiling_second_half, "structural": structural,
             "best_k": sg.get("best_k"), "ho_bestK": sg.get("heldout_surprise_bestK")}
    return sig, debug


def make_runs() -> list[tuple[str, str, dict]]:
    """Enumerate (regime_label, run_name, kwargs) for the full sweep."""
    loc = localized_cmap()
    runs: list[tuple[str, str, dict]] = []
    for seed in SEEDS:
        runs.append(("stable_known", f"A_static.s{seed}", {"cmap": loc, "seed": seed}))
        runs.append(("irreducible_noise", f"D_noisy.s{seed}", {"cmap": loc, "seed": seed, "noisy": True}))
        runs.append(("nonstationarity", f"E_nonstat.s{seed}",
                     {"cmap": loc, "seed": seed, "cmap_after": remapped_cmap(loc, 7000 + seed)}))
        for ls in LAYOUT_SEEDS:
            ali = np.array(worlds.aliased(layout_seed=ls)["cmap"], dtype=int)
            runs.append(("structural_inadequacy", f"C_alias.s{seed}.L{ls}", {"cmap": ali, "seed": seed}))
    return runs


def main(smoke: bool = False) -> None:
    runs = make_runs()
    if smoke:
        # one run per regime, first seed/layout, for fast validity inspection
        seen: set[str] = set()
        runs = [r for r in runs if (r[0] not in seen and not seen.add(r[0]))]

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(s)
        print(s)

    emit("=" * 72)
    emit("Exp 155 — N3a-min shadow failure-mode diagnostic (4 regimes, 4x4/4-color)")
    emit("=" * 72)
    emit(f"seeds={SEEDS}  layouts={LAYOUT_SEEDS}  steps={T_TOTAL} (p1={T_PHASE1},p2={T_PHASE2})")
    emit(f"smoke={smoke}  n_runs={len(runs)}")
    emit("")

    y_true: list[str] = []
    y_pred: list[str] = []
    emit(f"{'run':<22} {'truth':<22} {'pred':<22} {'loud':>5} {'early':>6} {'late':>6} "
         f"{'ceil':>5} {'str':>5} {'K':>2} {'ho_bK':>6}")
    emit("-" * 112)
    for truth, name, kw in runs:
        sig, dbg = run_shadow(**kw)
        pred = classify_regime(sig)
        y_true.append(truth)
        y_pred.append(pred)
        mark = "" if pred == truth else "  <-- MISS"
        emit(f"{name:<22} {truth:<22} {pred:<22} {dbg['loud_mean']:>5} "
             f"{dbg['early_mean']:>6} {dbg['late_mean']:>6} {str(dbg['ceiling']):>5} "
             f"{str(dbg['structural']):>5} {str(dbg['best_k']):>2} {str(dbg['ho_bestK']):>6}{mark}")

    emit("")
    cm = confusion_matrix(y_true, y_pred, REGIMES_4)
    emit("Confusion matrix (rows=truth, cols=prediction):")
    emit(format_confusion(cm, REGIMES_4))
    emit("")
    prf = per_class_prf(cm)
    for lab, c in zip(REGIMES_4, prf):
        emit(f"  {lab:<24} P={c['precision']:.3f} R={c['recall']:.3f} F1={c['f1']:.3f}")
    mf1 = macro_f1(cm)
    c_noise = confusion_rate(cm, REGIMES_4, "irreducible_noise", "structural_inadequacy")
    c_nonstat = confusion_rate(cm, REGIMES_4, "nonstationarity", "structural_inadequacy")
    emit("")
    emit(f"HEADLINE macro-F1 = {mf1:.3f}   (pass >= 0.70, falsifier <= 0.40)")
    emit(f"HEADLINE confusion noise->structural    = {c_noise:.3f}  (pass < 0.20)")
    emit(f"HEADLINE confusion nonstat->structural  = {c_nonstat:.3f}  (pass < 0.20)")

    p1, p2, p3 = mf1 >= 0.70, c_noise < 0.20, c_nonstat < 0.20
    falsified = (mf1 <= 0.40) or (c_noise >= 0.20) or (c_nonstat >= 0.20)
    if p1 and p2 and p3:
        verdict = "POSITIVE"
    elif falsified:
        verdict = "NEGATIVE"
    else:
        verdict = "MIXED"
    emit("")
    emit(f"VERDICT: {verdict}  (P1 macroF1={p1}, P2 noise<0.2={p2}, P3 nonstat<0.2={p3})")

    if not smoke:
        out_dir = Path("experiments/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "exp155.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_verdict(
            out_dir / "exp155_verdict.json",
            experiment="exp155",
            arms={
                "P1_macro_f1": {"pass": bool(p1), "reason": f"macro-F1={mf1:.3f} (>=0.70)"},
                "P2_noise_not_structural": {"pass": bool(p2), "reason": f"rate={c_noise:.3f} (<0.20)"},
                "P3_nonstat_not_structural": {"pass": bool(p3), "reason": f"rate={c_nonstat:.3f} (<0.20)"},
            },
            verdict=verdict,
            halted=(verdict == "NEGATIVE"),
            notes="N3a-min shadow diagnostic; rule-based classifier (provided); fresh seeds 20-27.",
        )
        emit(f"\nwrote experiments/outputs/exp155.txt and exp155_verdict.json")


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
