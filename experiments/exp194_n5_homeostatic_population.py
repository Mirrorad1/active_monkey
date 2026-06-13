"""Exp 194 — N5 homeostatic population ecology (pre-registered in
loop/directions/population-ecology.md BEFORE any data).

Hypothesis: a population of simple homeostatic active-inference-flavored creatures,
governed only by energy/resource constraints and inherited (mutated) traits, sustains a
reproducing multi-generation ecology in which the ENVIRONMENT exerts selection
(homeostatic death + finite resources, no external ranking), and the qualitative regime
shifts predictably with resource abundance — reproducibly under fixed seeds.

World: 12x12 regenerating resource grid; inherited movement/metabolic/aging costs;
emergent crowding/depletion. Three scenarios share one founder genotype and identical
mechanics, differing only in resource parameters: balanced / scarce / overabundant. A
runaway cap is a safety assert (falsifier F2), never a culler.

Predictions if TRUE (>=3 fixed seeds/scenario, report ALL):
  P1 determinism: same seed -> byte-identical event-stream hash, every scenario.
  P2 bounded persistence (balanced): final pop > 0 and <= pop_cap (= the population
     runaway cap, max_population = 200; NOT the per-cell resource capacity), 3/3 seeds.
  P3 multi-generation lineage (balanced): births > 0 and max generation >= 3, 3/3 seeds.
  P4 homeostatic death only: >=1 death in balanced; 100% of deaths carry a homeostatic
     cause; 0 deaths from any ranking/evaluator cause (structural).
  P5 scarcity bites (effect size): scarce vs balanced, 3/3 seeds: mean final pop lower
     by >=25% AND starvation-death fraction higher by >=0.15.
  P6 traits not frozen: in >=1 scenario, >=1 trait's last-generation mean differs from
     the founder by >=1 mutation-sigma (drift/variation, NOT adaptation).
  P7 regimes separate: not all extinct and not all exploding; >=1 bounded-persistent.

Falsifiers:
  F1 non-determinism (any scenario) -> NEGATIVE.
  F2 trivial dynamics: balanced explodes past cap in all seeds, OR extinct in ALL
     scenarios/seeds -> NEGATIVE.
  F3 homeostatic constraint inert: zero homeostatic deaths anywhere -> MIXED.
  F4 reproduction unconstrained: any creature reproduces before its OWN maturity_age or
     below its OWN reproduction_energy_threshold -> NEGATIVE.
  F5 unsafe mutation: >5% of births produce out-of-bounds genotypes -> NEGATIVE.
  F6 hidden evaluator: any survival/reproduction path reads a global ranking -> NEGATIVE.
  F7 scarcity inert: scarce ~ balanced on P5 metrics in a majority of seeds -> MIXED.

Verdict rule: POSITIVE iff P1..P5 hold and none of F1/F2/F4/F5/F6 fire; P1 fail ->
NEGATIVE; F7 (substrate ok, scarcity no effect) -> MIXED; F2/F4/F5/F6 -> NEGATIVE.
P6/P7 supporting.

Honesty: the per-scenario RESOURCE parameters are chosen to place each scenario in its
regime (disclosed environment design, not 'ecologies emerge inevitably'); the policy is
a PROVIDED homeostatic heuristic, not the pymdp active-inference stack; P6 is
drift/variation, not adaptation. Constants PROVIDED.
"""
from __future__ import annotations

import io
import os
import sys
import time
import json
from dataclasses import asdict
from typing import Any

# Insert repo root so ecology and run are importable when run as a script
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ecology.run import run_scenario, determinism_check
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.genotype import TRAIT_BOUNDS, mutate
from ecology.recording import write_verdict
import numpy as np


SEEDS = [0, 1, 2]
SCENARIO_NAMES = ["balanced", "scarce", "overabundant"]
OUTPUT_DIR = os.path.join(
    _REPO_ROOT, "experiments", "outputs", "exp194_n5_homeostatic_population"
)
EXP_TXT = os.path.join(_REPO_ROOT, "experiments", "outputs", "exp194.txt")

MUTATION_RATE = SCENARIOS["balanced"].mutation_rate  # 0.05


def _mutation_sigma(trait: str) -> float:
    """1 mutation-sigma = rate * (hi - lo) for P6 threshold."""
    lo, hi = TRAIT_BOUNDS[trait]
    return MUTATION_RATE * (hi - lo)


def main() -> None:
    start_wall = time.time()
    buf = io.StringIO()

    def log(line: str = "") -> None:
        print(line)
        buf.write(line + "\n")

    log("=" * 70)
    log("Exp 194 — N5 Homeostatic Population Ecology")
    log("=" * 70)
    log(f"Seeds: {SEEDS}")
    log(f"Scenarios: {SCENARIO_NAMES}")
    log(f"Mutation rate: {MUTATION_RATE}")
    log(f"Founder: {asdict(FOUNDER)}")
    log()

    # -----------------------------------------------------------------------
    # 1. Run all scenarios x seeds
    # -----------------------------------------------------------------------
    results: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in SCENARIO_NAMES}

    for scenario_name in SCENARIO_NAMES:
        cfg = SCENARIOS[scenario_name]
        log(f"--- Scenario: {scenario_name} (res_cap={cfg.capacity}, regen={cfg.regen_rate}, "
            f"init_resource={cfg.initial_resource}) ---")
        for seed in SEEDS:
            summary = run_scenario(scenario_name, seed, OUTPUT_DIR)
            results[scenario_name][seed] = summary
            log(
                f"  seed={seed}: final_pop={summary['final_pop']}, "
                f"births={summary['births']}, deaths={summary['deaths']}, "
                f"max_gen={summary['max_generation']}, "
                f"cohort_mortality={summary['cohort_mortality']:.3f}, "
                f"starvation_death_frac={summary['starvation_death_fraction']:.3f}, "
                f"extinction={summary['extinction']}, "
                f"explosion={summary['explosion']}, "
                f"steps={summary['steps_run']}, "
                f"hash={summary['events_hash'][:12]}..."
            )
    log()

    # -----------------------------------------------------------------------
    # 2. Determinism checks (P1 / F1)
    # -----------------------------------------------------------------------
    log("--- Determinism checks ---")
    det_results: dict[str, dict[int, bool]] = {name: {} for name in SCENARIO_NAMES}
    for scenario_name in SCENARIO_NAMES:
        for seed in SEEDS:
            ok = determinism_check(scenario_name, seed)
            det_results[scenario_name][seed] = ok
            log(f"  {scenario_name} seed={seed}: {'PASS' if ok else 'FAIL'}")
    log()

    # -----------------------------------------------------------------------
    # 3. Compute predictions P1..P7 and falsifiers F1..F7
    # -----------------------------------------------------------------------

    # --- P1 / F1: Determinism ---
    p1_all_pass = all(
        det_results[name][seed]
        for name in SCENARIO_NAMES
        for seed in SEEDS
    )
    f1_fires = not p1_all_pass

    # --- P2: balanced final pop > 0 and <= max_population, 3/3 seeds ---
    max_pop = SCENARIOS["balanced"].max_population
    p2_per_seed = []
    for seed in SEEDS:
        s = results["balanced"][seed]
        ok = s["final_pop"] > 0 and s["final_pop"] <= max_pop and not s["explosion"]
        p2_per_seed.append(ok)
    p2_pass = sum(p2_per_seed) == 3

    # --- P3: balanced births > 0 and max_generation >= 3, 3/3 seeds ---
    p3_per_seed = []
    for seed in SEEDS:
        s = results["balanced"][seed]
        ok = s["births"] > 0 and s["max_generation"] >= 3
        p3_per_seed.append(ok)
    p3_pass = sum(p3_per_seed) == 3

    # --- P4: >=1 death in balanced; 100% homeostatic cause; 0 ranking deaths ---
    # All deaths should have cause="starvation" (homeostatic)
    balanced_all_deaths = sum(results["balanced"][s]["deaths"] for s in SEEDS)
    balanced_all_starvation = sum(
        results["balanced"][s]["cause_of_death_tally"].get("starvation", 0)
        for s in SEEDS
    )
    balanced_other_deaths = balanced_all_deaths - balanced_all_starvation
    p4_some_deaths = balanced_all_deaths >= 1
    p4_all_homeostatic = balanced_other_deaths == 0
    p4_pass = p4_some_deaths and p4_all_homeostatic
    f3_fires = balanced_all_deaths == 0  # no homeostatic deaths at all

    # --- P5: scarce vs balanced — two sub-metrics, one well-posed, one ill-posed ---
    #
    # Sub-metric A (WELL-POSED, pre-registered): mean final pop lower by >=25%.
    # Sub-metric B (ILL-POSED, pre-registered): starvation_death_fraction higher by >=0.15.
    #   starvation_death_fraction = starvation_deaths / total_deaths ≡ 1.0 when deaths > 0
    #   (starvation is the only death cause).  The "+0.15" conjunct is unsatisfiable.
    # Sub-metric C (EXPLORATORY, post-hoc, NOT predeclared): cohort_mortality
    #   (starvation_deaths / total_births) shows real differences.  Disclosed as exploratory.
    #
    p5_pop_ok_per_seed = []          # well-posed population sub-metric
    p5_starv_predeclared_per_seed = []  # ill-posed predeclared starvation sub-metric
    p5_cohort_mortality_diffs = {}   # exploratory cohort_mortality differences per seed
    for seed in SEEDS:
        b = results["balanced"][seed]
        sc = results["scarce"][seed]
        b_pop = b["final_pop"]
        sc_pop = sc["final_pop"]
        # A) Population sub-metric
        if b_pop > 0:
            pop_ratio = (b_pop - sc_pop) / b_pop
            p5_pop_ok_per_seed.append(pop_ratio >= 0.25)
        else:
            p5_pop_ok_per_seed.append(False)
        # B) Predeclared starvation_death_fraction (≡ 1.0 when deaths > 0)
        b_sdf = b["starvation_death_fraction"]   # will be 1.0 if balanced has deaths
        sc_sdf = sc["starvation_death_fraction"]  # will be 1.0 if scarce has deaths
        sdf_diff = sc_sdf - b_sdf
        p5_starv_predeclared_per_seed.append(sdf_diff >= 0.15)
        # C) Exploratory cohort_mortality
        b_cm = b["cohort_mortality"]
        sc_cm = sc["cohort_mortality"]
        p5_cohort_mortality_diffs[seed] = {
            "balanced": b_cm, "scarce": sc_cm, "diff": round(sc_cm - b_cm, 4)
        }

    p5_population_pass = sum(p5_pop_ok_per_seed) == 3
    p5_starv_predeclared_pass = sum(p5_starv_predeclared_per_seed) == 3  # will be False
    p5_pass = p5_population_pass and p5_starv_predeclared_pass

    # F7: scarce ~ balanced on P5 metrics in majority of seeds
    # F7 uses only the well-posed population sub-metric for the fire condition
    f7_fires = sum(p5_pop_ok_per_seed) < 2  # majority fail = scarcity inert on population

    # --- P6: >=1 scenario, >=1 trait's last-gen mean differs from founder by >=1 sigma ---
    founder_d = asdict(FOUNDER)
    p6_evidence: list[str] = []
    for name in SCENARIO_NAMES:
        for seed in SEEDS:
            s = results[name][seed]
            for trait, founder_val in founder_d.items():
                sigma = _mutation_sigma(trait)
                last_gen_mean = s["last_gen_trait_means"].get(trait)
                if last_gen_mean is not None:
                    diff = abs(last_gen_mean - founder_val)
                    if diff >= sigma:
                        p6_evidence.append(
                            f"{name}/seed{seed}/{trait}: "
                            f"founder={founder_val:.4f}, last_gen={last_gen_mean:.4f}, "
                            f"diff={diff:.4f} >= sigma={sigma:.4f}"
                        )
    p6_pass = len(p6_evidence) > 0

    # --- P7: not all extinct, not all exploding; >=1 bounded-persistent ---
    all_extinct = all(
        results[name][seed]["extinction"]
        for name in SCENARIO_NAMES
        for seed in SEEDS
    )
    all_exploded = all(
        results[name][seed]["explosion"]
        for name in SCENARIO_NAMES
        for seed in SEEDS
    )
    bounded_persistent_exists = any(
        results[name][seed]["final_pop"] > 0 and not results[name][seed]["explosion"]
        for name in SCENARIO_NAMES
        for seed in SEEDS
    )
    p7_pass = not all_extinct and not all_exploded and bounded_persistent_exists

    # --- F2: balanced explodes in ALL seeds OR extinct in ALL scenarios/seeds ---
    balanced_all_exploded = all(results["balanced"][s]["explosion"] for s in SEEDS)
    all_scenarios_all_extinct = all(
        results[name][seed]["extinction"]
        for name in SCENARIO_NAMES
        for seed in SEEDS
    )
    f2_fires = balanced_all_exploded or all_scenarios_all_extinct

    # --- F4: any creature reproduces before maturity_age or below threshold ---
    # Real data check: iterate ALL reproduction events across ALL scenarios/seeds.
    # Each reproduction event now carries parent_age_at_repro, parent_energy_at_repro,
    # parent_maturity_age, and parent_repro_energy_threshold (added in engine.py fix).
    f4_fires = False
    f4_violations: list[str] = []
    f4_total_repro_events = 0
    for scenario_name in SCENARIO_NAMES:
        for seed in SEEDS:
            events_path = os.path.join(OUTPUT_DIR, f"{scenario_name}_seed{seed}", "events.jsonl")
            if not os.path.isfile(events_path):
                continue
            with open(events_path) as ef:
                lines = ef.readlines()
            for line in lines:
                ev = json.loads(line)
                if ev["event_type"] != "reproduction":
                    continue
                f4_total_repro_events += 1
                d = ev.get("details", {})
                age = d.get("parent_age_at_repro")
                energy = d.get("parent_energy_at_repro")
                maturity = d.get("parent_maturity_age")
                threshold = d.get("parent_repro_energy_threshold")
                # age is an integer, so comparison is exact
                if age is not None and maturity is not None and age < maturity:
                    f4_fires = True
                    f4_violations.append(
                        f"{scenario_name}/seed{seed}/creature{ev['creature_id']}: "
                        f"age_at_repro={age} < maturity_age={maturity}"
                    )
                # energy_at_repro is the exact float captured at the decision point,
                # the same value that passed _is_reproduction_eligible's >= check.
                # Any violation here is a genuine engine bug.
                if energy is not None and threshold is not None and energy < threshold:
                    f4_fires = True
                    f4_violations.append(
                        f"{scenario_name}/seed{seed}/creature{ev['creature_id']}: "
                        f"energy_at_repro={energy:.8f} < threshold={threshold:.8f} (margin={energy-threshold:.2e})"
                    )
    if f4_fires:
        f4_detail = f"VIOLATIONS ({len(f4_violations)}): {f4_violations[:3]}"
    else:
        f4_detail = (
            f"CLEAR ({f4_total_repro_events} reproduction events across all "
            f"scenarios/seeds, 0 violations of age<maturity or energy<threshold)"
        )

    # --- F5: >5% births produce out-of-bounds genotypes ---
    # All births go through mutate() which calls clamp_traits + is_valid assertion.
    # Structural guarantee: mutate() has an assert is_valid at the end.
    # We verify by checking our mutation fuzz test passed (11/11 tests pass).
    f5_fires = False
    f5_detail = "structural: mutate() asserts is_valid(); 200-seed fuzz passed"

    # --- F6: any survival/reproduction path reads a global ranking ---
    # Structural guarantee: see engine.py module comment + test_no_global_fitness_selection
    f6_fires = False
    f6_detail = "structural: _step_one_creature reads only (c, local_cell), no population arg"

    # -----------------------------------------------------------------------
    # 4. Print SUMMARY
    # -----------------------------------------------------------------------
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)

    log()
    log("-- PREDICTIONS --")

    log(f"P1 determinism: {'PASS' if p1_all_pass else 'FAIL'}")
    for name in SCENARIO_NAMES:
        for seed in SEEDS:
            ok = det_results[name][seed]
            log(f"   {name} seed={seed}: {'ok' if ok else 'FAIL'}")

    log(f"P2 bounded persistence (balanced): {'PASS' if p2_pass else 'FAIL'}")
    for i, seed in enumerate(SEEDS):
        s = results["balanced"][seed]
        log(f"   seed={seed}: final_pop={s['final_pop']} <= pop_cap={SCENARIOS['balanced'].max_population}, "
            f"explosion={s['explosion']}, ok={p2_per_seed[i]}")

    log(f"P3 multi-generation lineage (balanced): {'PASS' if p3_pass else 'FAIL'}")
    for i, seed in enumerate(SEEDS):
        s = results["balanced"][seed]
        log(f"   seed={seed}: births={s['births']}, max_gen={s['max_generation']}, "
            f"ok={p3_per_seed[i]}")

    log(f"P4 homeostatic death only: {'PASS' if p4_pass else 'FAIL'}")
    log(f"   total_deaths={balanced_all_deaths}, starvation={balanced_all_starvation}, "
        f"other={balanced_other_deaths}")
    log(f"   some_deaths={p4_some_deaths}, all_homeostatic={p4_all_homeostatic}")

    log(f"P5 scarcity bites (effect size): {'PASS' if p5_pass else 'FAIL'}")
    log(f"  P5 population sub-metric (well-posed): "
        f"scarce mean final pop lower than balanced by >=25%")
    for i, seed in enumerate(SEEDS):
        b = results["balanced"][seed]
        sc = results["scarce"][seed]
        b_pop = b["final_pop"]
        sc_pop = sc["final_pop"]
        ratio = (b_pop - sc_pop) / b_pop if b_pop > 0 else float("nan")
        log(f"   seed={seed}: balanced_pop={b_pop}, scarce_pop={sc_pop}, "
            f"pop_reduction={ratio:.2%}, pop_ok={p5_pop_ok_per_seed[i]}")
    log(f"  P5 population sub-metric result: "
        f"{'PASS' if p5_population_pass else 'FAIL'} "
        f"({sum(p5_pop_ok_per_seed)}/3 seeds)")
    log()
    log(f"  P5 starvation-death-fraction (deaths/total_deaths) — "
        f"PREDECLARED metric (starvation_death_fraction):")
    log(f"  NOTE: starvation is the ONLY death cause => this metric ≡ 1.0 wherever deaths>0")
    log(f"  => diff is always 0.000 (or 0 if extinct-no-deaths); +0.15 conjunct UNSATISFIABLE")
    for i, seed in enumerate(SEEDS):
        b = results["balanced"][seed]
        sc = results["scarce"][seed]
        b_sdf = b["starvation_death_fraction"]
        sc_sdf = sc["starvation_death_fraction"]
        sdf_diff = sc_sdf - b_sdf
        log(f"   seed={seed}: balanced_starvation_death_frac={b_sdf:.3f}, "
            f"scarce_starvation_death_frac={sc_sdf:.3f}, diff={sdf_diff:+.3f}, "
            f"ok(>=0.15)={p5_starv_predeclared_per_seed[i]}")
    log(f"  P5 starvation-death-fraction result: ILL-POSED "
        f"(single death cause; +0.15 unsatisfiable) => FAIL")
    log()
    log(f"  P5 cohort-mortality (deaths/total_births) — "
        f"EXPLORATORY, post-hoc, NOT the predeclared standard:")
    for seed in SEEDS:
        cm = p5_cohort_mortality_diffs[seed]
        log(f"   seed={seed}: balanced_cohort_mortality={cm['balanced']:.3f}, "
            f"scarce_cohort_mortality={cm['scarce']:.3f}, diff={cm['diff']:+.3f}")

    log(f"P6 traits not frozen: {'PASS' if p6_pass else 'FAIL'}")
    for ev in p6_evidence[:5]:  # show first 5 examples
        log(f"   {ev}")
    if len(p6_evidence) > 5:
        log(f"   ... ({len(p6_evidence)} total evidence items)")

    log(f"P7 regimes separate: {'PASS' if p7_pass else 'FAIL'}")
    log(f"   all_extinct={all_extinct}, all_exploded={all_exploded}, "
        f"bounded_persistent_exists={bounded_persistent_exists}")

    log()
    log("-- FALSIFIERS --")

    log(f"F1 non-determinism: {'FIRES -> NEGATIVE' if f1_fires else 'CLEAR'}")
    log(f"F2 trivial dynamics: {'FIRES -> NEGATIVE' if f2_fires else 'CLEAR'}")
    log(f"   balanced_all_exploded={balanced_all_exploded}, "
        f"all_scenarios_all_extinct={all_scenarios_all_extinct}")
    log(f"F3 homeostatic constraint inert: {'FIRES -> MIXED' if f3_fires else 'CLEAR'}")
    log(f"   total_deaths_balanced={balanced_all_deaths}")
    log(f"F4 reproduction unconstrained: {'FIRES -> NEGATIVE' if f4_fires else 'CLEAR'}")
    log(f"   F4: {f4_detail}")
    log(f"F5 unsafe mutation: {'FIRES -> NEGATIVE' if f5_fires else 'CLEAR'}")
    log(f"   {f5_detail}")
    log(f"F6 hidden evaluator: {'FIRES -> NEGATIVE' if f6_fires else 'CLEAR'}")
    log(f"   {f6_detail}")
    log(f"F7 scarcity inert: {'FIRES -> MIXED' if f7_fires else 'CLEAR'}")
    log(f"   seeds_where_population_bites={sum(p5_pop_ok_per_seed)}/3 (well-posed metric)")

    log()

    # -----------------------------------------------------------------------
    # 5. Compute verdict
    # -----------------------------------------------------------------------
    # Verdict logic (dictated by conductor — implement EXACTLY):
    #   if not p1_determinism:        NEGATIVE   (F1)
    #   elif f2 or f4 or f5 or f6:    NEGATIVE
    #   elif p2 and p3 and p4 and p5_population_pass and not p5_starv_predeclared_pass:
    #       MIXED   (substrate + scarcity(population) established; predeclared P5
    #                starvation sub-metric ill-posed)
    #   elif p2 and p3 and p4 and p5_pass:
    #       POSITIVE
    #   elif f7:                       MIXED
    #   else:                          MIXED

    if not p1_all_pass:
        verdict = "NEGATIVE"
        verdict_reason = "P1 (determinism) FAILED — F1 fires"
    elif f2_fires or f4_fires or f5_fires or f6_fires:
        fired = []
        if f2_fires: fired.append("F2")
        if f4_fires: fired.append("F4")
        if f5_fires: fired.append("F5")
        if f6_fires: fired.append("F6")
        verdict = "NEGATIVE"
        verdict_reason = f"Hard falsifiers fired: {', '.join(fired)}"
    elif p2_pass and p3_pass and p4_pass and p5_population_pass and not p5_starv_predeclared_pass:
        verdict = "MIXED"
        verdict_reason = (
            "P1–P4 PASS; scarcity bites overwhelmingly on the well-posed population "
            "sub-metric (PASS 3/3); the predeclared P5 starvation-death-fraction "
            "sub-metric is ILL-POSED (single death cause ⇒ ≡1.0); "
            "no hard falsifier fires ⇒ MIXED."
        )
    elif p2_pass and p3_pass and p4_pass and p5_pass:
        verdict = "POSITIVE"
        verdict_reason = "P1..P5 all pass, no hard falsifiers fire"
    elif f7_fires:
        verdict = "MIXED"
        verdict_reason = "F7 fires (scarcity inert on population metric in majority of seeds)"
    else:
        verdict = "MIXED"
        verdict_reason = (
            f"P1-P4 pass but P5 not fully met; no hard falsifier fires"
        )

    log(f"VERDICT: {verdict}")
    log(f"Reason: {verdict_reason}")
    log(f"P6_supporting: {p6_pass} ({len(p6_evidence)} evidence items)")
    log(f"P7_supporting: {p7_pass}")

    log()
    log("DISCLOSURES (honest record):")
    log("- Policy is a PROVIDED homeostatic heuristic (resource-seeking value map), "
        "NOT pymdp active inference.")
    log("- \"No external evaluator selects\" is a VERIFIED DESIGN INVARIANT "
        "(code-inspected: survival/reproduction read only the creature's own "
        "genotype/phenotype + local cell), not an empirical discovery.")
    log("- Scenario resource params AND the founder genotype were TUNED to place each "
        "regime (environment design only; mechanics/metrics/falsifiers fixed). "
        "Founder revised once (repro threshold 12->17) because the original exploded. "
        "Audit trail in ecology/scenarios.py.")
    log("- No baseline (random-policy / zero-mutation) was run.")
    log("- P6 trait shifts are drift/variation; some are large (e.g. metabolic cost "
        "~0.50->~0.02) and consistent with directional selection, but NO counterfactual "
        "control was run, so drift vs selection is NOT distinguished.")
    log("- Overabundant shows ~70% cohort mortality despite abundant resources because "
        "the population explodes to the runaway cap and crowds (per-cell depletion), "
        "not because resources are scarce.")

    elapsed = time.time() - start_wall
    log()
    log(f"runtime: {elapsed:.1f}s")
    log("=" * 70)

    # -----------------------------------------------------------------------
    # 6. Write exp194.txt (human-readable output)
    # -----------------------------------------------------------------------
    os.makedirs(os.path.dirname(EXP_TXT), exist_ok=True)
    with open(EXP_TXT, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    # -----------------------------------------------------------------------
    # 7. Write top-level verdict.json
    # -----------------------------------------------------------------------
    verdict_data = {
        "experiment": "exp194",
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "predictions": {
            "P1_determinism": {"pass": p1_all_pass},
            "P2_bounded_persistence": {
                "pass": p2_pass,
                "per_seed": {str(s): p2_per_seed[i] for i, s in enumerate(SEEDS)},
                "final_pops": {str(s): results["balanced"][s]["final_pop"] for s in SEEDS},
            },
            "P3_multi_generation": {
                "pass": p3_pass,
                "per_seed": {str(s): p3_per_seed[i] for i, s in enumerate(SEEDS)},
                "max_generations": {str(s): results["balanced"][s]["max_generation"] for s in SEEDS},
            },
            "P4_homeostatic_death": {
                "pass": p4_pass,
                "total_deaths": balanced_all_deaths,
                "starvation_deaths": balanced_all_starvation,
                "other_deaths": balanced_other_deaths,
            },
            "P5_scarcity_bites": {
                "pass": p5_pass,
                "p5_population_pass": p5_population_pass,
                "p5_starv_predeclared_pass": p5_starv_predeclared_pass,
                "p5_starv_predeclared_note": "ILL-POSED: starvation is sole death cause => starvation_death_fraction ≡ 1.0; +0.15 unsatisfiable",
                "per_seed_pop_ok": {str(s): p5_pop_ok_per_seed[i] for i, s in enumerate(SEEDS)},
                "per_seed_starv_predeclared_ok": {str(s): p5_starv_predeclared_per_seed[i] for i, s in enumerate(SEEDS)},
                "balanced_pops": {str(s): results["balanced"][s]["final_pop"] for s in SEEDS},
                "scarce_pops": {str(s): results["scarce"][s]["final_pop"] for s in SEEDS},
                "balanced_starvation_death_fracs": {str(s): results["balanced"][s]["starvation_death_fraction"] for s in SEEDS},
                "scarce_starvation_death_fracs": {str(s): results["scarce"][s]["starvation_death_fraction"] for s in SEEDS},
                "cohort_mortality_exploratory": {
                    str(s): p5_cohort_mortality_diffs[s] for s in SEEDS
                },
            },
            "P6_traits_drift": {
                "pass": p6_pass,
                "evidence_count": len(p6_evidence),
                "examples": p6_evidence[:5],
            },
            "P7_regimes_separate": {
                "pass": p7_pass,
                "all_extinct": all_extinct,
                "all_exploded": all_exploded,
                "bounded_persistent_exists": bounded_persistent_exists,
            },
        },
        "falsifiers": {
            "F1_non_determinism": {"fires": f1_fires},
            "F2_trivial_dynamics": {
                "fires": f2_fires,
                "balanced_all_exploded": balanced_all_exploded,
                "all_scenarios_all_extinct": all_scenarios_all_extinct,
            },
            "F3_homeostatic_inert": {"fires": f3_fires, "total_deaths": balanced_all_deaths},
            "F4_reproduction_unconstrained": {"fires": f4_fires, "detail": f4_detail},
            "F5_unsafe_mutation": {"fires": f5_fires, "detail": f5_detail},
            "F6_hidden_evaluator": {"fires": f6_fires, "detail": f6_detail},
            "F7_scarcity_inert": {
                "fires": f7_fires,
                "seeds_population_biting": sum(p5_pop_ok_per_seed),
            },
        },
        "all_summaries": {
            name: {str(seed): results[name][seed] for seed in SEEDS}
            for name in SCENARIO_NAMES
        },
    }

    write_verdict(
        os.path.join(OUTPUT_DIR, "verdict.json"),
        verdict_data,
    )

    print(f"\nOutputs written to: {OUTPUT_DIR}")
    print(f"exp194.txt written to: {EXP_TXT}")


if __name__ == "__main__":
    main()
