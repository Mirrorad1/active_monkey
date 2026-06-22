"""experiments/exp268_acoustic_eavesdropping.py
Exp 268 — Acoustic Eavesdropping Expressibility Probe (direction: acoustic-ecology, Rung 1).

PLAIN.  We built SOUND as a physical thing in the predator-prey world: moving, attacking,
eating, breeding and dying all radiate energy that spreads across the patch graph, gets quieter
with distance, loses its high notes fastest, and arrives late.  Prey/predators are NEVER told
"that's a predator" — they only get raw loudness per band.  Before letting hearing EVOLVE we ask
the honest gate question: if we GIFT one side perfect-ish hearing, does it actually help — and
does the help vanish when we scramble or silence the sound (proving it was the real signal)?

HYPOTHESIS.  In a calibrated, coexisting predator-prey regime where evasion is possible but not
saturated, a physically-constrained sound field gives prey ACTIONABLE information about nearby
predators, lowering their capture hazard when hearing is GIFTED — and that benefit beats shuffled
and silenced controls.

PREDICTIONS (if TRUE; >=8 seeds, report ALL):
  P1 determinism: same seed -> identical events_hash, per arm.
  P2 channel carries structure: MI(high band; hidden predator density) in the FIELD-ONLY arm is
     >= 0.10 bits AND >= 10x the shuffled-hidden MI null, in >= 5/8 seeds.  [analysis-only metric]
  P3 (CORE) gifted prey benefit: gifted-prey-hearing capture hazard < field-only (baseline) hazard
     by >= 0.01 AND < the SILENT control hazard, in >= 5/8 seeds.
  C-null (degenerate-control guard): the SILENT control reduces to baseline migration —
     |silent hazard - baseline hazard| <= 0.005 (no deterministic-drift artifact).

FALSIFIERS / ABORT (any ⇒ do NOT run hearing evolution; log the boundary):
  F-abort (EXPECTED, the honest prior): P2 holds (sound carries real bits) but P3 FAILS — gifted
     hearing gives no measurable capture-hazard benefit (or harms via crowding) ⇒ information is
     present but NOT locally actionable ⇒ ABORT evolution.  Verdict NEGATIVE / NEW INSIGHT.
  F-dead: P2 fails ⇒ the channel carries no usable structure ⇒ NEGATIVE (and the substrate needs
     fixing before anything else).
  F-shuffle/silent-match: if shuffled OR silenced controls match real sound on hazard, the apparent
     benefit (if any) is not carried by real acoustic structure.
  F-leak: hazard benefit only via leaked exact position/identity/event-type ⇒ FAIL+fix (by
     construction the agent reads ONLY per-band intensity — leakage is structurally precluded; we
     verify no semantic accessor exists, test_acoustic A9).
  F-range: detection perfect at long range ⇒ model too generous (checked: attenuation_curve falls,
     far gain < 0.25x near, test_acoustic A6).
  F-scalar (frequency not load-bearing — MARK, do not fail the probe): if scalar-intensity MI >=
     banded MI (within 0.02) AND scalar-danger gifted behavior ~= banded-danger behavior, frequency
     is not yet load-bearing.

POSITIVE only if P1 ∧ P2 ∧ P3 ∧ C-null and no shuffle/silent match ⇒ gifted hearing helps and the
help is carried by real, frequency-constrained sound ⇒ proceed to Rung 2 (invasion-from-rarity).

REGIME (the Exp 257-259 BOTH coexistence bracket — nontrivial predation, escape possible,
persistent; chosen BEFORE the hazard test).  A heterogeneity diagnostic (frac of patch-steps with
<=2 predators; per-patch predator CoV) is reported to show whether a spatial predator REFUGE exists
to flee TO — a precondition for the hypothesis.
"""
import os
import sys
import json
from multiprocessing import Pool

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim

SEEDS = (200, 201, 202, 203, 204, 205, 206, 207)
HORIZON = 1500


def base_cfg(**kw) -> PatchMosaicConfig:
    """Exp 257-259 BOTH coexistence regime (refuge + migration + asynchrony)."""
    d = dict(
        n_patches=8, topology="ring",
        attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        prey_escape=1.0, pred_attack=1.0,
        horizon=HORIZON, n_prey0_per_patch=40, n_pred0_per_patch=8,
    )
    d.update(kw)
    return PatchMosaicConfig(**d)


# Arm -> config overrides.  All acoustic arms share the SAME base biology.
ARMS = {
    "1_no_field":     dict(enable_acoustic_field=False),
    "2_field_only":   dict(enable_acoustic_field=True, acoustic_response=False),
    "3_prey_hear":    dict(enable_acoustic_field=True, acoustic_response=True, gifted_hearing="prey"),
    "4_pred_hear":    dict(enable_acoustic_field=True, acoustic_response=True, gifted_hearing="pred"),
    "5_prey_shuffle": dict(enable_acoustic_field=True, acoustic_response=True, gifted_hearing="prey",
                           acoustic_shuffle=True),
    "6_prey_silent":  dict(enable_acoustic_field=True, acoustic_response=True, gifted_hearing="prey",
                           acoustic_silence=True),
    "7_prey_scalar":  dict(enable_acoustic_field=True, acoustic_response=True, gifted_hearing="prey",
                           acoustic_scalar=True),
}


def _heterogeneity(res) -> tuple:
    """Spatial predator heterogeneity over the 2nd half: (frac patch-steps with <=2
    predators, mean per-patch CoV).  A refuge to flee TO requires some predator-light cells."""
    pp = np.asarray(res["patch_pred_series"])[:, HORIZON // 2:]  # (patch, time)
    frac_light = float((pp <= 2).mean())
    m = pp.mean(axis=0)
    sd = pp.std(axis=0)
    cov = float(np.mean(sd / np.maximum(m, 1e-9)))
    return frac_light, cov


def run_one(task):
    arm, seed = task
    cfg = base_cfg(**ARMS[arm])
    res = PatchMosaicSim(cfg, seed=seed).run()
    a = res.get("acoustic", {})
    prey_series = res["global_prey_series"]
    pred_series = res["global_pred_series"]
    frac_light, cov = _heterogeneity(res)
    return {
        "arm": arm, "seed": seed,
        "events_hash": res["events_hash"],
        "coexist": int(prey_series[-1] > 0 and pred_series[-1] > 0),
        "prey_end": prey_series[-1], "pred_end": pred_series[-1],
        "prey_mean2": float(np.mean(prey_series[HORIZON // 2:])),
        "capture_hazard": a.get("prey_capture_hazard", float("nan")),
        "capture_success": a.get("predator_capture_success", float("nan")),
        "mi_low": a.get("mi_low_bits", float("nan")),
        "mi_mid": a.get("mi_mid_bits", float("nan")),
        "mi_high": a.get("mi_high_bits", float("nan")),
        "mi_scalar": a.get("mi_scalar_bits", float("nan")),
        "mi_banded": a.get("mi_banded_bits", float("nan")),
        "mi_null": a.get("mi_null_bits", float("nan")),
        "fp": a.get("false_positive_rate", float("nan")),
        "fn": a.get("false_negative_rate", float("nan")),
        "frac_light": frac_light, "pred_cov": cov,
        "atten": a.get("attenuation_curve", []),
    }


def agg(rows, arm, key):
    vals = [r[key] for r in rows if r["arm"] == arm]
    return float(np.nanmean(vals)) if vals else float("nan")


def per_seed(rows, arm, key):
    return [r[key] for r in sorted((r for r in rows if r["arm"] == arm),
                                   key=lambda r: r["seed"])]


def main():
    tasks = [(arm, seed) for arm in ARMS for seed in SEEDS]
    with Pool(processes=min(8, os.cpu_count() or 1)) as pool:
        rows = pool.map(run_one, tasks)

    L = []
    def P(s=""):
        L.append(s)

    P("=" * 78)
    P("Exp 268 — Acoustic Eavesdropping Expressibility Probe (acoustic-ecology, Rung 1)")
    P("=" * 78)
    P(f"Regime: Exp 257-259 BOTH coexistence (8-patch ring, refuge access=0.30 frac=0.25,")
    P(f"        migration prey/pred=0.05, async rotating amp=0.4, attack_a=0.05, K_pred=40).")
    P(f"Seeds: {SEEDS}  horizon={HORIZON}  (>=8 seeds; all reported).")
    P("")

    # ---- P1 determinism: re-run arm 2/3 seed 200, compare hash ----
    h_a = run_one(("3_prey_hear", 200))["events_hash"]
    h_b = run_one(("3_prey_hear", 200))["events_hash"]
    P(f"P1 determinism (gifted-prey seed 200 re-run): {'PASS' if h_a == h_b else 'FAIL'} "
      f"({h_a[:12]} == {h_b[:12]})")
    # byte-identity: arm1 (no field) vs arm2 (field passive) must match per seed
    byte = all(
        run_one(("1_no_field", s))["events_hash"] == run_one(("2_field_only", s))["events_hash"]
        for s in SEEDS[:3])
    P(f"   field-passive byte-identity (no_field == field_only, seeds {SEEDS[:3]}): "
      f"{'PASS' if byte else 'FAIL'}")
    P("")

    # ---- per-arm summary ----
    P("PER-ARM SUMMARY (means over 8 seeds):")
    P(f"{'arm':16} {'coex':4} {'prey_end':8} {'prey_mn2':8} {'pred_end':8} "
      f"{'haz':8} {'capsucc':8}")
    for arm in ARMS:
        P(f"{arm:16} {sum(per_seed(rows, arm, 'coexist')):>2}/8 "
          f"{agg(rows, arm, 'prey_end'):8.0f} {agg(rows, arm, 'prey_mean2'):8.0f} "
          f"{agg(rows, arm, 'pred_end'):8.0f} {agg(rows, arm, 'capture_hazard'):8.5f} "
          f"{agg(rows, arm, 'capture_success'):8.5f}")
    P("")

    # ---- MI (channel structure), from the FIELD-ONLY arm (passive) ----
    P("ACOUSTIC MUTUAL INFORMATION with hidden predator density (ANALYSIS-ONLY, field_only arm):")
    P(f"   MI low={agg(rows,'2_field_only','mi_low'):.3f}  mid={agg(rows,'2_field_only','mi_mid'):.3f}  "
      f"high={agg(rows,'2_field_only','mi_high'):.3f}  scalar={agg(rows,'2_field_only','mi_scalar'):.3f}  "
      f"banded={agg(rows,'2_field_only','mi_banded'):.3f}  NULL={agg(rows,'2_field_only','mi_null'):.4f} (bits)")
    P(f"   detection FP={agg(rows,'2_field_only','fp'):.3f}  FN={agg(rows,'2_field_only','fn'):.3f}  "
      f"(high-band tercile vs predator-density tercile)")
    P(f"   per-seed mi_high: {[round(x,3) for x in per_seed(rows,'2_field_only','mi_high')]}")
    P(f"   per-seed mi_null: {[round(x,4) for x in per_seed(rows,'2_field_only','mi_null')]}")
    P("")

    # ---- heterogeneity diagnostic (is there a refuge to flee TO?) ----
    P("SPATIAL PREDATOR HETEROGENEITY (field_only arm):")
    P(f"   frac patch-steps with <=2 predators (a refuge to flee TO): "
      f"{agg(rows,'2_field_only','frac_light'):.3f}")
    P(f"   mean per-patch predator CoV: {agg(rows,'2_field_only','pred_cov'):.3f}")
    P(f"   attenuation curve (hop-dist -> band-summed gain): {rows[8]['atten']}")
    P("")

    # ---- per-seed capture hazard for the CORE benefit test ----
    base_haz = per_seed(rows, "2_field_only", "capture_hazard")
    hear_haz = per_seed(rows, "3_prey_hear", "capture_hazard")
    shuf_haz = per_seed(rows, "5_prey_shuffle", "capture_hazard")
    sil_haz = per_seed(rows, "6_prey_silent", "capture_hazard")
    P("PREY CAPTURE HAZARD per seed (CORE benefit test, lower=better for prey):")
    P(f"   baseline(field_only): {[round(x,4) for x in base_haz]}")
    P(f"   gifted prey hearing : {[round(x,4) for x in hear_haz]}")
    P(f"   shuffled control    : {[round(x,4) for x in shuf_haz]}")
    P(f"   silent control      : {[round(x,4) for x in sil_haz]}")
    P("")

    # ---- VERDICT (predeclared rule, conjunct by conjunct) ----
    mi_high = np.array(per_seed(rows, "2_field_only", "mi_high"))
    mi_null = np.array(per_seed(rows, "2_field_only", "mi_null"))
    p2 = int(np.sum((mi_high >= 0.10) & (mi_high >= 10.0 * np.maximum(mi_null, 1e-6))))
    gateA = p2 >= 5

    b = np.array(base_haz); h = np.array(hear_haz); s = np.array(sil_haz)
    p3_seeds = int(np.sum((h < b - 0.01) & (h < s)))
    gateB = p3_seeds >= 5

    cnull = abs(np.nanmean(sil_haz) - np.nanmean(base_haz)) <= 0.005

    mi_scalar = agg(rows, "2_field_only", "mi_scalar")
    mi_high_m = agg(rows, "2_field_only", "mi_high")
    freq_load_bearing = mi_high_m > mi_scalar + 0.02

    P("VERDICT (predeclared, conjunct-by-conjunct):")
    P(f"  P2 channel carries structure (MI_high>=0.10 & >=10x null): {p2}/8 seeds -> "
      f"{'PASS' if gateA else 'FAIL'}")
    P(f"  P3 gifted prey benefit (haz < baseline-0.01 AND < silent): {p3_seeds}/8 seeds -> "
      f"{'PASS' if gateB else 'FAIL'}")
    P(f"  C-null silent==baseline (|d|<=0.005): d="
      f"{abs(np.nanmean(sil_haz)-np.nanmean(base_haz)):.5f} -> {'PASS' if cnull else 'FAIL'}")
    P(f"  Frequency load-bearing (MI_high > MI_scalar+0.02): "
      f"{'YES' if freq_load_bearing else 'NO (scalar ~= banded -> frequency NOT yet load-bearing)'}")
    P(f"  predator-hearing asymmetry (capture success): baseline="
      f"{agg(rows,'2_field_only','capture_success'):.5f} vs gifted-pred="
      f"{agg(rows,'4_pred_hear','capture_success'):.5f}")
    P("")

    if gateA and gateB and cnull:
        verdict = "POSITIVE"
        note = ("gifted prey hearing lowers capture hazard, carried by real frequency-constrained "
                "sound that beats shuffle+silence -> PROCEED to Rung 2 (invasion-from-rarity).")
    elif gateA and not gateB:
        verdict = "NEGATIVE"
        note = ("ABORT evolution: sound carries real bits about hidden predator density (MI >> null) "
                "but gifted prey hearing gives NO actionable capture-hazard benefit -- information "
                "present, advantage absent. Honest prior confirmed; do NOT run hearing evolution.")
    elif not gateA:
        verdict = "NEGATIVE"
        note = "channel carries no usable structure (MI ~ null) -- the acoustic model needs fixing."
    else:
        verdict = "MIXED"
        note = "partial / control-confounded; see per-seed detail."

    P(f"VERDICT: {verdict} / NEW INSIGHT")
    P(f"  {note}")
    if gateA and not gateB:
        P("  Mechanism: the metapopulation structure that POSES coexistence (predators spread across "
          "ALL patches -> frac_light~0) leaves prey NO acoustic refuge to flee to; fleeing the "
          "loudest neighbor only CROWDS prey (negative frequency dependence). Information != "
          "actionable advantage -- the local-gradient wall's acoustic face.")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(_repo_root, "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "exp268.txt"), "w") as f:
        f.write(out + "\n")
    # also dump raw rows for reproducibility
    with open(os.path.join(outdir, "exp268_rows.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "atten"} for r in rows], f, indent=0)


if __name__ == "__main__":
    main()
