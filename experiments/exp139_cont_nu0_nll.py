"""
Exp 139 — fresh-seed test of Exp 135's post-hoc nu0-NLL observation (consolidation).

Exp 135 logged (post-hoc, untested): nu0 shifts PREDICTIVE (NLL-based) erosion even
though it is a proven null knob for mean drift — higher nu0 eroded sooner (NLL n_half
174 vs 140 at kappa0=1; 190 vs 146 at kappa0=10). Suspected mechanism (declared before
this run): low nu0 lets the learned covariance WIDEN under noise-phase scatter,
hedging the predictive distribution and postponing the NLL midpoint crossing; high
nu0 pins the covariance narrow, so noise damage shows in NLL sooner. Per
VALIDATION.md, a post-hoc pattern must be tested on FRESH seeds.

Protocol: identical to Exp 135 (hexagon r=1, footprints 0.35^2 I, anchored position
N(pos_t, 0.05^2 I), moment-matched NIW updates, T_struct=600, T_noise=2400,
checkpoints every 100, NLL on 300 held-out structured pairs, NLL-based n_half =
noise dose at the midpoint between end-structured NLL and end-noise plateau) — but
only the continuous agent (no twin needed), cells kappa0 in {1, 10} x nu0 in {4, 20},
and FRESH seeds 8..15 (never run on this protocol).

Predictions (TRUE iff all):
- P1 effect out-of-sample: cell-mean NLL n_half(nu0=4) / n_half(nu0=20) >= 1.15 at
  BOTH kappa0 levels.
- P2 mechanism: cell-mean tr(E[Sigma]) averaged over the 6 words at end-noise is
  >= 1.25x larger for nu0=4 than nu0=20 at both kappa0 levels (widening-as-hedging).
- P3 nu0-null replication: max |delta curve(nu0=4) - delta curve(nu0=20)| < 0.05 at
  every checkpoint, per kappa0 (Exp 135's P4, now out-of-sample).

Falsifier (any triggers NEGATIVE): P1 ratio < 1.0 at either kappa0 (the Exp 135
observation was seed noise — strike it in the log), OR P3 fails (the established
nu0-null does not replicate — flag loudly, it contradicts a logged law). P1 ratio in
[1.0, 1.15) or P2 failing while P1 holds -> MIXED (effect present but weak / effect
real but mechanism wrong — log which). Three-way rule per PROTOCOL step 3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from active_loop.continuous import NIW, predictive_word_logprobs

# ---------------------------------------------------------------------------
# Geometry and constants  (identical to Exp 135)
# ---------------------------------------------------------------------------

D = 2
N_WORDS = 6
SIGMA_K_SCALE = 0.35
RADIUS = 1.0
T_S = 600
T_N = 2400
T_TOTAL = T_S + T_N
ANCHOR_SIGMA_SCALE = 0.05
N_HOLDOUT_PER_CONCEPT = 50
N_HOLDOUT = N_HOLDOUT_PER_CONCEPT * N_WORDS

_angles = np.array([k * np.pi / 3.0 for k in range(N_WORDS)])
CONCEPT_MUS = np.stack([np.array([np.cos(a), np.sin(a)]) for a in _angles])  # (6, 2)

# This experiment's sweep: kappa0 in {1,10} x nu0 in {4,20}
KAPPA0_LIST = [1, 10]
NU0_LIST = [4, 20]
# Row-major enumeration: (kappa0=1,nu0=4)=0, (kappa0=1,nu0=20)=1,
#                        (kappa0=10,nu0=4)=2, (kappa0=10,nu0=20)=3
CELLS = [(k, n) for k in KAPPA0_LIST for n in NU0_LIST]

SEEDS = list(range(8, 16))  # fresh seeds 8..15

M0 = np.zeros(D)
CHECKPOINTS = list(range(100, T_TOTAL + 1, 100))


# ---------------------------------------------------------------------------
# Helpers (identical to Exp 135)
# ---------------------------------------------------------------------------

def _emission_probs_at(pos: np.ndarray) -> np.ndarray:
    log_p = np.array([
        -0.5 * float(np.dot(pos - CONCEPT_MUS[k], pos - CONCEPT_MUS[k])) / SIGMA_K_SCALE**2
        for k in range(N_WORDS)
    ])
    log_p -= float(np.logaddexp.reduce(log_p))
    p = np.exp(log_p)
    return p / p.sum()


def _make_S0(nu0: float) -> np.ndarray:
    return (SIGMA_K_SCALE**2) * (nu0 - D - 1) * np.eye(D)


def _make_fresh_niws(kappa0: float, nu0: float) -> list[NIW]:
    S0 = _make_S0(nu0)
    return [NIW(m=M0.copy(), kappa=float(kappa0), nu=float(nu0), S=S0.copy())
            for _ in range(N_WORDS)]


def _generate_streams(seed: int, cell_index: int) -> dict:
    """Same RNG scheme as Exp 135; fresh seeds (8..15) avoid collisions."""
    rng = np.random.default_rng(seed * 100 + cell_index)

    concepts_s = rng.integers(0, N_WORDS, size=T_S)
    words_s = np.empty(T_S, dtype=int)
    for t in range(T_S):
        c = concepts_s[t]
        pos_t = CONCEPT_MUS[c]
        p = _emission_probs_at(pos_t)
        words_s[t] = rng.choice(N_WORDS, p=p)

    concepts_n = rng.integers(0, N_WORDS, size=T_N)
    words_n = rng.integers(0, N_WORDS, size=T_N)

    holdout_concepts = np.repeat(np.arange(N_WORDS), N_HOLDOUT_PER_CONCEPT)
    holdout_words = np.empty(N_HOLDOUT, dtype=int)
    for i in range(N_HOLDOUT):
        c = holdout_concepts[i]
        pos_c = CONCEPT_MUS[c]
        p = _emission_probs_at(pos_c)
        holdout_words[i] = rng.choice(N_WORDS, p=p)

    return {
        "concepts_s": concepts_s,
        "words_s": words_s,
        "concepts_n": concepts_n,
        "words_n": words_n,
        "holdout_concepts": holdout_concepts,
        "holdout_words": holdout_words,
    }


def _mean_tr_exp_sigma(niws: list[NIW]) -> float:
    """Mean over words of tr(E[Sigma_k])."""
    return float(np.mean([np.trace(niws[k].expected_Sigma()) for k in range(N_WORDS)]))


def _compute_delta(niws: list[NIW], m_at_Ts: list[np.ndarray], c_visited: np.ndarray) -> float:
    denom_vals = [np.linalg.norm(m_at_Ts[k] - c_visited) for k in range(N_WORDS)]
    denom = float(np.mean(denom_vals))
    if denom < 1e-12:
        return 0.0
    numer = float(np.mean([np.linalg.norm(niws[k]._m - m_at_Ts[k]) for k in range(N_WORDS)]))
    return numer / denom


def _cont_holdout_nll(niws: list[NIW], holdout_concepts: np.ndarray,
                      holdout_words: np.ndarray) -> float:
    Sigma_anchor = ANCHOR_SIGMA_SCALE**2 * np.eye(D)
    total = 0.0
    for i in range(N_HOLDOUT):
        c = int(holdout_concepts[i])
        w = int(holdout_words[i])
        mu_post = CONCEPT_MUS[c]
        word_mus = [niws[k]._m for k in range(N_WORDS)]
        word_Sigmas = [niws[k].expected_Sigma() for k in range(N_WORDS)]
        log_probs = predictive_word_logprobs(mu_post, Sigma_anchor, word_mus, word_Sigmas)
        total += -log_probs[w]
    return total / N_HOLDOUT


# ---------------------------------------------------------------------------
# Run one (seed, cell) pair — continuous agent only
# ---------------------------------------------------------------------------

def run_cell_seed(kappa0: float, nu0: float, seed: int, cell_index: int) -> list[dict]:
    streams = _generate_streams(seed, cell_index)
    concepts_s = streams["concepts_s"]
    words_s = streams["words_s"]
    concepts_n = streams["concepts_n"]
    words_n = streams["words_n"]
    holdout_concepts = streams["holdout_concepts"]
    holdout_words = streams["holdout_words"]

    Sigma_anchor = ANCHOR_SIGMA_SCALE**2 * np.eye(D)
    niws = _make_fresh_niws(kappa0, nu0)

    rows = []
    params = {"kappa0": float(kappa0), "nu0": float(nu0)}

    # --- Structured phase ---
    for t in range(T_S):
        c = int(concepts_s[t])
        w = int(words_s[t])
        pos_t = CONCEPT_MUS[c]
        niws[w] = niws[w].update_moments(pos_t, Sigma_anchor)

        step = t + 1
        if step % 100 == 0:
            cont_nll = _cont_holdout_nll(niws, holdout_concepts, holdout_words)
            tr_sigma = _mean_tr_exp_sigma(niws)
            rows.append({
                "exp": 139, "seed": seed, "step": step,
                "metric": "holdout_nll", "value": cont_nll,
                "params": params,
            })
            rows.append({
                "exp": 139, "seed": seed, "step": step,
                "metric": "mean_tr_exp_sigma", "value": tr_sigma,
                "params": params,
            })

    # Snapshot at end of structured phase
    m_at_Ts = [niws[k]._m.copy() for k in range(N_WORDS)]
    c_visited = np.array([CONCEPT_MUS[int(concepts_s[t])] for t in range(T_S)]).mean(axis=0)
    cont_nll_end_struct = _cont_holdout_nll(niws, holdout_concepts, holdout_words)

    # --- Noise phase ---
    noise_checkpoints_data: list[tuple[float, float, float, float]] = []
    # (n_noise_per_word, delta, cont_nll, mean_tr_sigma)

    for t in range(T_N):
        c = int(concepts_n[t])
        w = int(words_n[t])
        pos_t = CONCEPT_MUS[c]
        niws[w] = niws[w].update_moments(pos_t, Sigma_anchor)

        step = T_S + t + 1
        if step % 100 == 0:
            n_noise_per_word = (t + 1) / N_WORDS
            delta_val = _compute_delta(niws, m_at_Ts, c_visited)
            cont_nll = _cont_holdout_nll(niws, holdout_concepts, holdout_words)
            tr_sigma = _mean_tr_exp_sigma(niws)

            noise_checkpoints_data.append((n_noise_per_word, delta_val, cont_nll, tr_sigma))

            rows.append({
                "exp": 139, "seed": seed, "step": step,
                "metric": "holdout_nll", "value": cont_nll,
                "params": params,
            })
            rows.append({
                "exp": 139, "seed": seed, "step": step,
                "metric": "delta", "value": delta_val,
                "params": params,
            })
            rows.append({
                "exp": 139, "seed": seed, "step": step,
                "metric": "mean_tr_exp_sigma", "value": tr_sigma,
                "params": params,
            })

    cont_nll_end_noise = noise_checkpoints_data[-1][2] if noise_checkpoints_data else float("nan")
    end_noise_tr_sigma = noise_checkpoints_data[-1][3] if noise_checkpoints_data else float("nan")

    # NLL-based n_half
    cont_nll_mid = (cont_nll_end_struct + cont_nll_end_noise) / 2.0
    n_half_nll = float("nan")
    for n_dose, _delta, cont_nll, _tr in noise_checkpoints_data:
        if np.isnan(n_half_nll) and cont_nll >= cont_nll_mid:
            n_half_nll = n_dose
            break

    rows.append({
        "exp": 139, "seed": seed, "step": T_TOTAL,
        "metric": "n_half_nll", "value": float(n_half_nll),
        "params": params,
    })
    rows.append({
        "exp": 139, "seed": seed, "step": T_TOTAL,
        "metric": "end_noise_tr_sigma", "value": float(end_noise_tr_sigma),
        "params": params,
    })
    rows.append({
        "exp": 139, "seed": seed, "step": T_S,
        "metric": "end_struct_nll", "value": float(cont_nll_end_struct),
        "params": params,
    })
    rows.append({
        "exp": 139, "seed": seed, "step": T_TOTAL,
        "metric": "end_noise_nll", "value": float(cont_nll_end_noise),
        "params": params,
    })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from collections import defaultdict

    all_rows: list[dict] = []

    for cell_index, (kappa0, nu0) in enumerate(CELLS):
        for seed in SEEDS:
            seed_rows = run_cell_seed(kappa0, nu0, seed, cell_index)
            all_rows.extend(seed_rows)

    # --- Write JSON ---
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp139_rows.json"
    with open(out_path, "w") as f:
        json.dump(all_rows, f)

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------

    # Per-cell accumulators
    cell_n_half_nll: dict = defaultdict(list)
    cell_end_noise_tr_sigma: dict = defaultdict(list)
    cell_end_struct_nll: dict = defaultdict(list)
    cell_end_noise_nll: dict = defaultdict(list)
    # delta at noise checkpoints: (cell_key, n_dose) -> list of deltas
    cell_delta_at_n: dict = defaultdict(list)

    for row in all_rows:
        p = row["params"]
        ck = (p["kappa0"], p["nu0"])
        m = row["metric"]
        v = row["value"]
        step = row["step"]

        if m == "n_half_nll":
            cell_n_half_nll[ck].append(v)
        elif m == "end_noise_tr_sigma":
            cell_end_noise_tr_sigma[ck].append(v)
        elif m == "end_struct_nll":
            cell_end_struct_nll[ck].append(v)
        elif m == "end_noise_nll":
            cell_end_noise_nll[ck].append(v)
        elif m == "delta" and step > T_S:
            n_dose = (step - T_S) / N_WORDS
            cell_delta_at_n[(ck, n_dose)].append(v)

    # ---------------------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------------------
    print("=" * 100)
    print("Exp 139 — fresh-seed test of Exp 135's post-hoc nu0-NLL observation")
    print("Seeds 8..15 (never run on this protocol)")
    print("=" * 100)
    hdr = (f"{'kappa0':>7}  {'nu0':>4}  {'NLL_n_half':>10}  {'EndStructNLL':>12}  "
           f"{'EndNoiseNLL':>11}  {'mean_tr_E[Sigma]':>16}")
    print(hdr)
    print("-" * 80)

    cell_stats: dict = {}
    for kappa0 in KAPPA0_LIST:
        for nu0 in NU0_LIST:
            ck = (float(kappa0), float(nu0))
            n_halves = [v for v in cell_n_half_nll.get(ck, []) if not np.isnan(v)]
            mean_nhalf = float(np.mean(n_halves)) if n_halves else float("nan")
            es_nlls = cell_end_struct_nll.get(ck, [])
            en_nlls = cell_end_noise_nll.get(ck, [])
            tr_sigs = cell_end_noise_tr_sigma.get(ck, [])
            mean_es_nll = float(np.nanmean(es_nlls)) if es_nlls else float("nan")
            mean_en_nll = float(np.nanmean(en_nlls)) if en_nlls else float("nan")
            mean_tr_sig = float(np.nanmean(tr_sigs)) if tr_sigs else float("nan")
            cell_stats[ck] = {
                "mean_nhalf": mean_nhalf,
                "mean_es_nll": mean_es_nll,
                "mean_en_nll": mean_en_nll,
                "mean_tr_sig": mean_tr_sig,
            }
            print(f"{kappa0:>7}  {nu0:>4}  {mean_nhalf:>10.1f}  {mean_es_nll:>12.4f}  "
                  f"{mean_en_nll:>11.4f}  {mean_tr_sig:>16.6f}")

    print()

    # ---------------------------------------------------------------------------
    # P1: NLL n_half ratio nu0=4 / nu0=20 >= 1.15 at both kappa0 levels
    # ---------------------------------------------------------------------------
    print("--- P1: NLL n_half ratio (nu0=4 / nu0=20) >= 1.15 at both kappa0 levels ---")
    p1_ratios = {}
    p1_pass = True
    p1_falsifier = False  # ratio < 1.0
    p1_weak = False       # ratio in [1.0, 1.15)
    for kappa0 in KAPPA0_LIST:
        ck4 = (float(kappa0), 4.0)
        ck20 = (float(kappa0), 20.0)
        nhalf4 = cell_stats[ck4]["mean_nhalf"]
        nhalf20 = cell_stats[ck20]["mean_nhalf"]
        if not np.isnan(nhalf4) and not np.isnan(nhalf20) and nhalf20 > 1e-9:
            ratio = nhalf4 / nhalf20
        else:
            ratio = float("nan")
        p1_ratios[kappa0] = ratio
        ok = not np.isnan(ratio) and ratio >= 1.15
        if np.isnan(ratio) or ratio < 1.0:
            p1_falsifier = True
            p1_pass = False
        elif ratio < 1.15:
            p1_weak = True
            p1_pass = False
        print(f"  kappa0={kappa0:>2}: n_half(nu0=4)={nhalf4:.1f}, n_half(nu0=20)={nhalf20:.1f}, "
              f"ratio={ratio:.4f} -> {'PASS (>=1.15)' if ok else ('FALSIFIER (<1.0)' if ratio < 1.0 else 'WEAK [1.0,1.15)')}")

    print()

    # ---------------------------------------------------------------------------
    # P2: end-noise tr(E[Sigma]) ratio (nu0=4 / nu0=20) >= 1.25 at both kappa0 levels
    # ---------------------------------------------------------------------------
    print("--- P2: end-noise mean tr(E[Sigma]) ratio (nu0=4 / nu0=20) >= 1.25 at both kappa0 ---")
    p2_ratios = {}
    p2_pass = True
    for kappa0 in KAPPA0_LIST:
        ck4 = (float(kappa0), 4.0)
        ck20 = (float(kappa0), 20.0)
        tr4 = cell_stats[ck4]["mean_tr_sig"]
        tr20 = cell_stats[ck20]["mean_tr_sig"]
        if not np.isnan(tr4) and not np.isnan(tr20) and tr20 > 1e-12:
            ratio = tr4 / tr20
        else:
            ratio = float("nan")
        p2_ratios[kappa0] = ratio
        ok = not np.isnan(ratio) and ratio >= 1.25
        if not ok:
            p2_pass = False
        print(f"  kappa0={kappa0:>2}: tr_nu4={tr4:.6f}, tr_nu20={tr20:.6f}, "
              f"ratio={ratio:.4f} -> {'PASS (>=1.25)' if ok else 'FAIL (<1.25)'}")

    print()

    # ---------------------------------------------------------------------------
    # P3: nu0-null replication — max |delta(nu0=4) - delta(nu0=20)| < 0.05 per kappa0
    # ---------------------------------------------------------------------------
    print("--- P3: nu0-null replication (max |delta_nu4 - delta_nu20| < 0.05 per kappa0) ---")
    p3_pass = True
    p3_max_diffs = {}
    for kappa0 in KAPPA0_LIST:
        ck4 = (float(kappa0), 4.0)
        ck20 = (float(kappa0), 20.0)

        noise_doses = sorted(set(
            key[1] for key in cell_delta_at_n.keys()
            if key[0] in (ck4, ck20)
        ))
        max_diff = 0.0
        fail_count = 0
        for n_dose in noise_doses:
            d4 = cell_delta_at_n.get((ck4, n_dose), [])
            d20 = cell_delta_at_n.get((ck20, n_dose), [])
            if not d4 or not d20:
                continue
            diff = abs(float(np.mean(d4)) - float(np.mean(d20)))
            if diff >= 0.05:
                fail_count += 1
            max_diff = max(max_diff, diff)

        p3_max_diffs[kappa0] = max_diff
        ok = max_diff < 0.05
        if not ok:
            p3_pass = False
        print(f"  kappa0={kappa0:>2}: max |delta_nu4 - delta_nu20| = {max_diff:.4f}, "
              f"fail_checkpoints = {fail_count} -> "
              f"{'PASS' if ok else 'FAIL (nu0-null violated — contradicts logged law)'}")

    print()

    # ---------------------------------------------------------------------------
    # VERDICT
    # ---------------------------------------------------------------------------
    falsifiers = []

    # P1 ratio < 1.0 at either kappa0
    if p1_falsifier:
        bad_kappas = [k for k, r in p1_ratios.items() if np.isnan(r) or r < 1.0]
        falsifiers.append(
            f"P1: NLL n_half ratio < 1.0 at kappa0={bad_kappas} — "
            f"Exp 135 nu0-NLL observation was seed noise; strike from log"
        )

    # P3 failure
    if not p3_pass:
        bad_kappas = [k for k, d in p3_max_diffs.items() if d >= 0.05]
        falsifiers.append(
            f"P3: nu0-null VIOLATED at kappa0={bad_kappas} "
            f"(max diff >= 0.05) — CONTRADICTS LOGGED LAW; flag loudly"
        )

    print("--- Tallies ---")
    print(f"  P1 (NLL n_half ratio >= 1.15 at both kappa0): ", end="")
    if p1_falsifier:
        print("FALSIFIER (ratio < 1.0)")
    elif p1_weak:
        print("WEAK (ratio in [1.0, 1.15))")
    else:
        print("PASS")

    print(f"  P2 (tr(E[Sigma]) ratio >= 1.25 at both kappa0): {'PASS' if p2_pass else 'FAIL'}")
    for kappa0 in KAPPA0_LIST:
        print(f"    kappa0={kappa0}: ratio = {p2_ratios.get(kappa0, float('nan')):.4f}")

    print(f"  P3 (nu0-null replication, max diff < 0.05): {'PASS' if p3_pass else 'FAIL — LOGGED LAW VIOLATED'}")
    for kappa0 in KAPPA0_LIST:
        print(f"    kappa0={kappa0}: max_diff = {p3_max_diffs.get(kappa0, float('nan')):.4f}")

    print()

    if falsifiers:
        print("VERDICT: NEGATIVE")
        for f in falsifiers:
            print(f"  FALSIFIER: {f}")
        if not p3_pass:
            print("  *** CRITICAL: nu0-null law does not replicate — "
                  "Exp 135 P4 finding may be seed-specific or model has unexpected nu0 sensitivity ***")
    elif p1_pass and p2_pass and p3_pass:
        print("VERDICT: POSITIVE — all P1/P2/P3 conjuncts satisfied")
        print("  P1: NLL n_half ratio >= 1.15 at both kappa0 levels (effect replicates out-of-sample)")
        for kappa0 in KAPPA0_LIST:
            print(f"    kappa0={kappa0}: ratio = {p1_ratios.get(kappa0, float('nan')):.4f}")
        print("  P2: end-noise tr(E[Sigma]) ratio >= 1.25 at both kappa0 levels (widening-as-hedging confirmed)")
        for kappa0 in KAPPA0_LIST:
            print(f"    kappa0={kappa0}: ratio = {p2_ratios.get(kappa0, float('nan')):.4f}")
        print("  P3: nu0-null replicates on fresh seeds (max diff < 0.05 per kappa0)")
        for kappa0 in KAPPA0_LIST:
            print(f"    kappa0={kappa0}: max_diff = {p3_max_diffs.get(kappa0, float('nan')):.4f}")
    else:
        print("VERDICT: MIXED")
        mixed_notes = []
        if p1_weak:
            bad_kappas = [k for k, r in p1_ratios.items() if not np.isnan(r) and 1.0 <= r < 1.15]
            mixed_notes.append(
                f"P1 weak: ratio in [1.0, 1.15) at kappa0={bad_kappas} — effect present but below threshold"
            )
        if not p2_pass and not p1_falsifier:
            bad_kappas = [k for k, r in p2_ratios.items() if np.isnan(r) or r < 1.25]
            mixed_notes.append(
                f"P2 fails at kappa0={bad_kappas} while P1 holds — effect real but mechanism (widening) wrong or weak"
            )
        if not p3_pass:
            # This would already be a falsifier, but catch in mixed path too
            mixed_notes.append("P3 borderline — check flag above")
        for note in mixed_notes:
            print(f"  NOTE: {note}")

    print()
    print(f"Rows written to: {out_path}")


if __name__ == "__main__":
    main()
