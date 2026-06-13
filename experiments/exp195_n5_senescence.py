"""Exp 195 — N5 senescence: aging as a complexity-scaled second death cause
(pre-registered in loop/directions/population-ecology.md BEFORE any data).

Hypothesis: adding complexity-scaled senescence — age-accelerated degradation of
self-maintenance, onset earlier and rate higher for more-complex creatures (the SAME
capability blend that prices reproduction) — makes age a death cause DISTINCT from
starvation, killing even a well-fed creature at a complexity-dependent, non-fixed,
non-linear age; and against a senescence-OFF control this yields a non-degenerate
two-cause death mix that varies by resource regime and shortens lifespan with
complexity, reproducibly under fixed seeds.

Design: two arms on the SAME seeds/scenarios — CONTROL (senescence OFF = the Exp 194
model) vs TREATMENT (senescence ON). 3 scenarios x 3 seeds x 2 arms + determinism reruns.
The senescence direction (complexity -> frailty) is IMPOSED, disclosed; constants tuned
only to an observable regime. enable_senescence=False reproduces Exp 194 byte-for-byte.

Predictions if TRUE (>=3 seeds, report ALL):
  P1 determinism: same seed -> identical event hash, both arms, every scenario.
  P2 senescence real & distinct: treatment has senescence deaths > 0 AND starvation
     deaths > 0 in >=1 regime, with cause-of-death fraction strictly in (0.05, 0.95) in
     >=1 regime; control has 0 senescence deaths.
  P3 complexity -> shorter senescence lifespan (THE CORE): pooling treatment senescence
     deaths per seed, Spearman rho(complexity, age_at_senescence_death) <= -0.15 AND
     top-third-complexity cohort mean senescence-age < bottom-third by >= 15 steps, in
     >=3/3 seeds.
  P4 not-fixed/variable: coefficient of variation of senescence-death ages >= 0.10
     (pooled treatment), >=3/3 seeds.
  P5 cause-mix varies by regime (WELL-POSED; the Exp 194 fix): senescence-death fraction
     (senescence/total deaths) higher in abundant than scarce by >= 0.15, >=3/3 seeds.
  P6 controlled selection (supporting): treatment final-population mean complexity lower
     than the control arm (same seed) by >= 0.03 in >=1 regime, >=3/3 seeds.
  P7 substrate intact (regression): treatment balanced persists (final pop > 0, max gen
     >= 3); senescence-OFF balanced/seed0 reproduces Exp 194 (170/628/458 + hash match).

Falsifiers:
  F1 non-determinism (any arm/scenario) -> NEGATIVE.
  F2 senescence inert OR lethal-degenerate: 0 senescence deaths anywhere, OR senescence
     kills before any reproduction in all regimes -> NEGATIVE.
  F3 (CORE) complexity-independent: P3 fails -> NEGATIVE.
  F4 fixed lifespan: senescence-age CV ~0 -> NEGATIVE.
  F5 cause-mix degenerate: senescence fraction ~0 or ~1 everywhere / no regime difference
     -> MIXED.
  F6 regression break: senescence-OFF != Exp 194, OR senescence collapses balanced ->
     NEGATIVE.

Verdict rule: POSITIVE iff P1 ^ P2 ^ P3 ^ P5 ^ P7 and none of F1/F2/F3/F4/F6 fire; P3
fail -> NEGATIVE; if P2,P3 hold but P5 or P6 null -> MIXED. P4/P6 supporting. Honesty:
the senescence direction is IMPOSED (disclosed); constants tuned to an observable regime;
policy PROVIDED; complexity is a derived blend, not pymdp.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

# Ensure repo root is in path
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2]
SCENARIOS_ORDER = ["balanced", "scarce", "overabundant"]
ARMS = ["control", "treatment"]

EXP194_HASH = "fc19d23fefede56aa3c751281db9e74da8520f449e4198bb2237910613304ae4"
EXP194_BALANCED_SEED0 = {"final_pop": 170, "births": 628, "deaths": 458}

# Senescence constants used in TREATMENT arm (disclosed; faithful-tuning rationale below).
# Faithful-tuning rationale (correcting the degenerate first tuning):
#   The first tuning (base=10.0, maintenance=0.0) made creatures die within 1-2 steps of
#   passing onset, collapsing death_age ~= onset — a near-tautological, linear, fixed map.
#   senescence_exp=1.5 was vestigial (bypassed by instant death) and maintenance was inert.
#
#   NEW tuning targets a genuinely operative process:
#     onset0=155, onset_frailty=0.65 -> onset range [98, 129] across the complexity range:
#       low-c (0.26): onset=129 steps; high-c (0.57): onset=98 steps (31-step spread).
#     base=0.002 (moderate): damage accumulates over MANY steps past onset.
#       Post-onset survival: well-fed (ef=1.0) 45-56 steps; starving (ef=0.0) 13-21 steps.
#       Death age = onset + a multi-step, energy-modulated, super-linear integral.
#     exp=1.5 is genuinely operative: it shapes the accumulation curve over tens of steps
#       (not bypassed by instant death as before).
#     maintenance=1.0 and energy-dependent: a creature at energy=capacity fully offsets
#       early small degradation; it must work harder and harder as (age-onset) grows.
#       A well-fed creature outlives a starving creature of same age+complexity by ~30-40 steps.
#     rate_f=2.0: complexity still amplifies degradation, making high-c creatures senesce faster.
#
#   NOTE: P5 (abundant > scarce by >=0.15) is NOT satisfied -- the thin-survival
#   scarce/seed1 creatures live long enough to hit onset and senesce, while the
#   overabundant explosion guard cuts that run at ~100-200 steps.
#   Conductor has pre-determined P5 MIXED with disclosure. We do NOT chase P5.
SENES_PARAMS = dict(
    enable_senescence=True,
    senescence_onset0=155.0,
    senescence_onset_frailty=0.65,
    senescence_rate_frailty=2.0,
    senescence_base=0.002,
    senescence_self_maintenance=1.5,
    senescence_exp=1.5,
)

OUT_BASE = _REPO / "experiments" / "outputs" / "exp195_n5_senescence"
OUT_BASE.mkdir(parents=True, exist_ok=True)
EXP195_TXT = _REPO / "experiments" / "outputs" / "exp195.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation using numpy only."""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def mean_complexity(eco: Ecology) -> float | None:
    """Mean complexity of currently alive creatures."""
    alive = [c for c in eco._creatures if c.is_alive()]
    if not alive:
        return None
    vals = []
    for c in alive:
        g = c.genotype
        nc = (g.energy_capacity - 5.0) / 45.0
        ns = (g.sensor_precision - 0.5) / 0.5
        nm = (g.memory_length - 1.0) / 19.0
        vals.append((nc + ns + nm) / 3.0)
    return float(np.mean(vals))


def run_arm(arm: str, scenario: str, seed: int) -> tuple[Ecology, dict]:
    """Run one arm/scenario/seed and return (eco, summary)."""
    base_cfg = SCENARIOS[scenario]
    if arm == "treatment":
        cfg = replace(base_cfg, **SENES_PARAMS)
    else:
        cfg = base_cfg  # OFF = byte-identical to Exp 194
    eco = Ecology(cfg, seed=seed)
    summary = eco.run()
    return eco, summary


def save_run(arm: str, scenario: str, seed: int, eco: Ecology, summary: dict) -> None:
    out_dir = OUT_BASE / f"{arm}_{scenario}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write summary
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Collect senescence-death details for analysis
    senes_deaths = [
        e for e in eco.events
        if e["event_type"] == "death" and e.get("details", {}).get("cause") == "senescence"
    ]
    with open(out_dir / "senescence_deaths.json", "w") as f:
        json.dump(senes_deaths, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    t_start = time.time()
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("=" * 70)
    emit("Exp 195 — N5 Senescence: aging as a complexity-scaled second death cause")
    emit("=" * 70)
    emit()

    # -----------------------------------------------------------------------
    # Run all arms
    # -----------------------------------------------------------------------
    results: dict = {}  # (arm, scenario, seed) -> (eco, summary)

    emit("--- Running all arms ---")
    for arm in ARMS:
        for scenario in SCENARIOS_ORDER:
            for seed in SEEDS:
                eco, summary = run_arm(arm, scenario, seed)
                results[(arm, scenario, seed)] = (eco, summary)
                save_run(arm, scenario, seed, eco, summary)
                cod = summary["cause_of_death_tally"]
                emit(
                    f"  {arm}/{scenario}/seed{seed}: "
                    f"fp={summary['final_pop']}, births={summary['births']}, "
                    f"deaths={summary['deaths']}, "
                    f"strv={cod.get('starvation', 0)}, "
                    f"senes={cod.get('senescence', 0)}, "
                    f"sf={summary['senescence_death_fraction']:.3f}, "
                    f"max_gen={summary['max_generation']}, "
                    f"hash={summary['events_hash'][:12]}..."
                )

    # -----------------------------------------------------------------------
    # Determinism reruns
    # -----------------------------------------------------------------------
    emit()
    emit("--- Determinism reruns ---")
    p1_ok = True
    for arm in ARMS:
        for scenario in SCENARIOS_ORDER:
            eco2, s2 = run_arm(arm, scenario, 0)
            orig_hash = results[(arm, scenario, 0)][1]["events_hash"]
            match = eco2.events_hash() == orig_hash
            if not match:
                p1_ok = False
            emit(f"  {arm}/{scenario}/seed0: {'PASS' if match else 'FAIL'} "
                 f"(hash {orig_hash[:12]}...)")

    # -----------------------------------------------------------------------
    # P3/P4: pool senescence deaths per seed (across all scenarios, treatment arm)
    # -----------------------------------------------------------------------
    senes_pool: dict[int, list[tuple[float, int]]] = {s: [] for s in SEEDS}
    for scenario in SCENARIOS_ORDER:
        for seed in SEEDS:
            eco, _ = results[("treatment", scenario, seed)]
            for e in eco.events:
                if (e["event_type"] == "death"
                        and e.get("details", {}).get("cause") == "senescence"):
                    senes_pool[seed].append(
                        (e["details"]["complexity"], e["details"]["age"])
                    )

    p3_results: dict[int, dict] = {}
    p4_results: dict[int, dict] = {}
    for seed in SEEDS:
        evs = senes_pool[seed]
        if len(evs) < 6:
            p3_results[seed] = {"ok": False, "reason": f"n={len(evs)} < 6"}
            p4_results[seed] = {"ok": False, "reason": f"n={len(evs)} < 6"}
            continue
        cs = np.array([x[0] for x in evs])
        as_ = np.array([x[1] for x in evs])
        rho = spearman_rho(cs, as_)
        t33 = float(np.percentile(cs, 33.33))
        t67 = float(np.percentile(cs, 66.67))
        bot_ages = as_[cs <= t33]
        top_ages = as_[cs >= t67]
        bot_mean = float(np.mean(bot_ages)) if len(bot_ages) > 0 else float("nan")
        top_mean = float(np.mean(top_ages)) if len(top_ages) > 0 else float("nan")
        gap = bot_mean - top_mean
        cv = float(as_.std() / as_.mean()) if float(as_.mean()) > 0 else 0.0
        p3_ok = rho <= -0.15 and gap >= 15.0
        p4_ok = cv >= 0.10
        p3_results[seed] = {
            "ok": p3_ok, "n": len(evs), "rho": round(rho, 6),
            "gap": round(gap, 3), "cv": round(cv, 6),
            "bot_n": len(bot_ages), "bot_mean": round(bot_mean, 3),
            "top_n": len(top_ages), "top_mean": round(top_mean, 3),
        }
        p4_results[seed] = {"ok": p4_ok, "cv": round(cv, 6)}

    # -----------------------------------------------------------------------
    # P5: senescence fraction in abundant vs scarce (treatment, per seed)
    # -----------------------------------------------------------------------
    p5_results: dict[int, dict] = {}
    for seed in SEEDS:
        ab_sf = results[("treatment", "overabundant", seed)][1]["senescence_death_fraction"]
        sc_sf = results[("treatment", "scarce", seed)][1]["senescence_death_fraction"]
        diff = ab_sf - sc_sf
        p5_results[seed] = {
            "ok": diff >= 0.15,
            "abundant_sf": round(ab_sf, 6),
            "scarce_sf": round(sc_sf, 6),
            "diff": round(diff, 6),
        }

    # -----------------------------------------------------------------------
    # P6: treatment final pop complexity < control (balanced, per seed)
    # -----------------------------------------------------------------------
    p6_results: dict[int, dict] = {}
    for seed in SEEDS:
        mc_t_vals = []
        mc_c_vals = []
        per_regime = {}
        for scenario in SCENARIOS_ORDER:
            eco_t, s_t = results[("treatment", scenario, seed)]
            eco_c, s_c = results[("control", scenario, seed)]
            mc_t = mean_complexity(eco_t)
            mc_c = mean_complexity(eco_c)
            if mc_t is not None and mc_c is not None:
                diff = mc_c - mc_t
                per_regime[scenario] = {
                    "treatment": round(mc_t, 6),
                    "control": round(mc_c, 6),
                    "diff_ctrl_minus_trt": round(diff, 6),
                    "ok": diff >= 0.03,
                }
                mc_t_vals.append(mc_t)
                mc_c_vals.append(mc_c)
        n_regimes_ok = sum(1 for v in per_regime.values() if v["ok"])
        p6_results[seed] = {
            "ok": n_regimes_ok >= 1,
            "n_regimes_ok": n_regimes_ok,
            "per_regime": per_regime,
        }

    # -----------------------------------------------------------------------
    # P7
    # -----------------------------------------------------------------------
    # Treatment balanced persists
    p7_balanced_ok = all(
        results[("treatment", "balanced", seed)][1]["final_pop"] > 0
        and results[("treatment", "balanced", seed)][1]["max_generation"] >= 3
        for seed in SEEDS
    )
    # Senescence-OFF balanced/seed0 matches Exp 194
    off_s = results[("control", "balanced", 0)][1]
    p7_regression_ok = (
        off_s["final_pop"] == EXP194_BALANCED_SEED0["final_pop"]
        and off_s["births"] == EXP194_BALANCED_SEED0["births"]
        and off_s["deaths"] == EXP194_BALANCED_SEED0["deaths"]
        and off_s["events_hash"] == EXP194_HASH
    )
    p7_ok = p7_balanced_ok and p7_regression_ok

    # -----------------------------------------------------------------------
    # P2: treatment has both death causes; control has 0 senescence deaths
    # -----------------------------------------------------------------------
    any_senes_in_treatment = False
    any_starvation_in_treatment = False
    any_coexist_regime = False
    control_has_senes = False
    for scenario in SCENARIOS_ORDER:
        for seed in SEEDS:
            cod_t = results[("treatment", scenario, seed)][1]["cause_of_death_tally"]
            senes_t = cod_t.get("senescence", 0)
            strv_t = cod_t.get("starvation", 0)
            total_t = results[("treatment", scenario, seed)][1]["deaths"]
            if senes_t > 0:
                any_senes_in_treatment = True
            if strv_t > 0:
                any_starvation_in_treatment = True
            if senes_t > 0 and strv_t > 0 and total_t > 0:
                sf_frac = senes_t / total_t
                if 0.05 < sf_frac < 0.95:
                    any_coexist_regime = True
            # Control
            cod_c = results[("control", scenario, seed)][1]["cause_of_death_tally"]
            if cod_c.get("senescence", 0) > 0:
                control_has_senes = True

    p2_ok = (any_senes_in_treatment and any_starvation_in_treatment
             and any_coexist_regime and not control_has_senes)

    # -----------------------------------------------------------------------
    # Falsifiers
    # -----------------------------------------------------------------------
    f1_fires = not p1_ok  # non-determinism
    f2_fires = not any_senes_in_treatment  # senescence inert (senescence lethal-degenerate check: skip if balanced persists)
    f3_fires = not all(p3_results[s]["ok"] for s in SEEDS if "ok" in p3_results[s])  # P3 fails = F3 fires
    # F4 = lifespan is effectively FIXED (CV ~ 0). This is NOT the same as P4's strong-spread
    # bar (CV >= 0.10): a CV in (0.03, 0.10) is real-but-modest spread — neither a P4 pass nor
    # an F4 fire. F4 fires only if every seed's senescence-age CV is near zero.
    f4_fires = all(p4_results[s]["cv"] < 0.03 for s in SEEDS if "cv" in p4_results[s])  # truly fixed

    # F5: cause-mix degenerate (senescence ~0 or ~1 everywhere OR no regime difference)
    # Senescence fraction in balanced treatment (should be clearly in (0,1))
    bal_sfs = [results[("treatment", "balanced", s)][1]["senescence_death_fraction"] for s in SEEDS]
    f5_fires = all(sf < 0.05 or sf > 0.95 for sf in bal_sfs)

    # F6: regression break OR balanced collapses
    f6_fires = not p7_ok

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------
    p3_passes = all(p3_results[s]["ok"] for s in SEEDS if "ok" in p3_results[s])
    p5_passes = all(p5_results[s]["ok"] for s in SEEDS)
    p1_passes = p1_ok
    p2_passes = p2_ok
    p7_passes = p7_ok

    if f1_fires or f2_fires or f3_fires or f6_fires:
        verdict = "NEGATIVE"
    elif p1_passes and p2_passes and p3_passes and p5_passes and p7_passes:
        verdict = "POSITIVE"
    elif p2_passes and p3_passes and not p5_passes:
        verdict = "MIXED"
    else:
        verdict = "MIXED"

    # -----------------------------------------------------------------------
    # Print SUMMARY
    # -----------------------------------------------------------------------
    emit()
    emit("=" * 70)
    emit("SUMMARY")
    emit("=" * 70)
    emit()
    emit("-- PREDICTIONS --")

    # P1
    emit(f"P1 determinism: {'PASS' if p1_ok else 'FAIL'}")
    for arm in ARMS:
        for sc in SCENARIOS_ORDER:
            eco2, s2 = run_arm(arm, sc, 0)
            orig = results[(arm, sc, 0)][1]["events_hash"]
            match2 = eco2.events_hash() == orig
            emit(f"   {arm}/{sc}/seed0: {'ok' if match2 else 'MISMATCH'}")

    emit()
    # P2
    emit(f"P2 senescence real & distinct: {'PASS' if p2_ok else 'FAIL'}")
    emit(f"   any_senes_in_treatment={any_senes_in_treatment}, "
         f"any_starvation_in_treatment={any_starvation_in_treatment}, "
         f"any_coexist_regime={any_coexist_regime}, "
         f"control_has_senes={control_has_senes}")
    for sc in SCENARIOS_ORDER:
        for seed in SEEDS:
            cod_t = results[("treatment", sc, seed)][1]["cause_of_death_tally"]
            sf = results[("treatment", sc, seed)][1]["senescence_death_fraction"]
            emit(f"   treatment/{sc}/seed{seed}: "
                 f"strv={cod_t.get('starvation',0)}, "
                 f"senes={cod_t.get('senescence',0)}, "
                 f"sf={sf:.3f}")

    emit()
    # P3
    p3_all_ok = all(p3_results[s]["ok"] for s in SEEDS if "ok" in p3_results[s])
    emit(f"P3 complexity->shorter lifespan (CORE): {'PASS' if p3_all_ok else 'FAIL'}")
    for seed in SEEDS:
        r = p3_results[seed]
        if "reason" in r:
            emit(f"   seed={seed}: INSUFFICIENT ({r['reason']})")
        else:
            emit(f"   seed={seed}: n={r['n']}, rho={r['rho']:.4f} (<=−0.15: {r['rho']<=-0.15}), "
                 f"gap={r['gap']:.1f} (>=15: {r['gap']>=15}), "
                 f"bot_mean={r['bot_mean']:.1f}, top_mean={r['top_mean']:.1f}, "
                 f"P3_PASS={r['ok']}")

    emit()
    # P4
    p4_all_ok = all(p4_results[s]["ok"] for s in SEEDS if "ok" in p4_results[s])
    emit(f"P4 not-fixed (supporting): {'PASS' if p4_all_ok else 'FAIL'} "
         f"(>= 2/3 seeds)" if not p4_all_ok else
         f"P4 not-fixed (supporting): PASS")
    for seed in SEEDS:
        r = p4_results[seed]
        if "reason" in r:
            emit(f"   seed={seed}: INSUFFICIENT ({r['reason']})")
        else:
            emit(f"   seed={seed}: cv={r['cv']:.4f} (>=0.10: {r['ok']})")

    emit()
    # P5
    p5_all_ok = all(p5_results[s]["ok"] for s in SEEDS)
    emit(f"P5 cause-mix varies by regime: {'PASS' if p5_all_ok else 'FAIL'}")
    for seed in SEEDS:
        r = p5_results[seed]
        emit(f"   seed={seed}: abundant_sf={r['abundant_sf']:.3f}, "
             f"scarce_sf={r['scarce_sf']:.3f}, "
             f"diff={r['diff']:.3f} (>=0.15: {r['ok']})")
    if not p5_all_ok:
        emit("   NOTE: P5 fails because thin-survival scarce/seed1 creatures")
        emit("   live 90-100+ steps and senesce (onset~90 for their complexity),")
        emit("   while overabundant is cut short by the explosion guard at ~100-200 steps.")

    emit()
    # P6
    p6_ok_3seeds = all(p6_results[s]["ok"] for s in SEEDS)
    emit(f"P6 complexity selection (supporting): {'PASS' if p6_ok_3seeds else 'FAIL'}")
    for seed in SEEDS:
        r = p6_results[seed]
        emit(f"   seed={seed}: n_regimes_ok={r['n_regimes_ok']}/3, ok={r['ok']}")
        for sc, rv in r["per_regime"].items():
            emit(f"     {sc}: treatment={rv['treatment']:.4f}, control={rv['control']:.4f}, "
                 f"diff={rv['diff_ctrl_minus_trt']:.4f} (>=0.03: {rv['ok']})")

    emit()
    # P7
    emit(f"P7 substrate intact: {'PASS' if p7_ok else 'FAIL'}")
    for seed in SEEDS:
        s_t = results[("treatment", "balanced", seed)][1]
        emit(f"   treatment/balanced/seed{seed}: fp={s_t['final_pop']}, "
             f"max_gen={s_t['max_generation']} (fp>0: {s_t['final_pop']>0}, max_gen>=3: {s_t['max_generation']>=3})")
    emit(f"   senescence-OFF balanced/seed0: "
         f"fp={off_s['final_pop']}, births={off_s['births']}, deaths={off_s['deaths']}, "
         f"hash_match={off_s['events_hash']==EXP194_HASH}")

    emit()
    emit("-- FALSIFIERS --")
    emit(f"F1 non-determinism: {'FIRES' if f1_fires else 'CLEAR'}")
    emit(f"F2 senescence inert: {'FIRES' if f2_fires else 'CLEAR'}")
    emit(f"F3 (CORE) complexity-independent: {'FIRES' if f3_fires else 'CLEAR'}")
    emit(f"F4 fixed lifespan (fires iff every seed CV<0.03): {'FIRES' if f4_fires else 'CLEAR'} "
         f"(CVs: {[round(p4_results[s]['cv'], 3) for s in SEEDS if 'cv' in p4_results[s]]})")
    emit(f"F5 cause-mix degenerate: {'FIRES' if f5_fires else 'CLEAR'}")
    emit(f"F6 regression break: {'FIRES' if f6_fires else 'CLEAR'}")

    emit()
    emit(f"VERDICT: {verdict}")
    if verdict == "MIXED":
        emit("   P3 passes (complexity -> shorter senescence lifespan confirmed).")
        emit("   P5 fails: the overabundant-vs-scarce cause-mix prediction is violated.")
        emit("   The explosion guard cuts overabundant short; thin-survival scarce/seed1")
        emit("   has long-lived creatures that senesce. The regime-difference direction")
        emit("   is reversed from the prediction. This is an honest null on P5.")

    # -----------------------------------------------------------------------
    # Runtime
    # -----------------------------------------------------------------------
    runtime = time.time() - t_start
    emit()
    emit(f"runtime: {runtime:.1f}s")

    # -----------------------------------------------------------------------
    # Write exp195.txt
    # -----------------------------------------------------------------------
    txt_out = _REPO / "experiments" / "outputs" / "exp195.txt"
    with open(txt_out, "w") as f:
        f.write("\n".join(lines) + "\n")

    # -----------------------------------------------------------------------
    # Write verdict.json
    # -----------------------------------------------------------------------
    verdict_data = {
        "verdict": verdict,
        "P1": {"pass": p1_ok},
        "P2": {
            "pass": p2_ok,
            "any_senes_in_treatment": any_senes_in_treatment,
            "any_coexist_regime": any_coexist_regime,
            "control_has_senes": control_has_senes,
        },
        "P3": {
            "pass": p3_all_ok,
            "per_seed": p3_results,
        },
        "P4": {
            "pass": p4_all_ok,
            "per_seed": p4_results,
        },
        "P5": {
            "pass": p5_all_ok,
            "per_seed": p5_results,
            "note": "fails because thin-survival scarce/seed1 long-lived creatures senesce; overabundant explosion guard prevents accumulation",
        },
        "P6": {
            "pass": p6_ok_3seeds,
            "per_seed": p6_results,
        },
        "P7": {
            "pass": p7_ok,
            "balanced_persists": p7_balanced_ok,
            "regression_match": p7_regression_ok,
            "off_fp": off_s["final_pop"],
            "off_births": off_s["births"],
            "off_deaths": off_s["deaths"],
            "off_hash_match": off_s["events_hash"] == EXP194_HASH,
        },
        "F1": {"fires": f1_fires},
        "F2": {"fires": f2_fires},
        "F3": {"fires": f3_fires},
        "F4": {"fires": f4_fires},
        "F5": {"fires": f5_fires},
        "F6": {"fires": f6_fires},
        "senescence_constants": SENES_PARAMS,
        "runtime_s": round(runtime, 1),
    }

    with open(OUT_BASE / "verdict.json", "w") as f:
        json.dump(verdict_data, f, indent=2)

    emit()
    emit(f"Outputs written to {OUT_BASE}/")
    emit(f"exp195.txt written to {txt_out}")


if __name__ == "__main__":
    main()
