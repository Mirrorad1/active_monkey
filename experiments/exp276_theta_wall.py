"""Exp 276a — is INNATE theta (learnable-use) WALLED or DIRECTLY EVOLVABLE?

DIRECTION: loop/directions/learning-guided-evolution.md.  Exp 275 established (POSABLE) that
theta = band_responsiveness (the Exp 201 band-tracker EMA rate) is a FAITHFUL "how well the
creature uses the sensor" axis with a real bad->good range that gates fitness.  Exp 276a makes
theta HERITABLE (Genotype.band_responsiveness) + L30-COSTED (theta_upkeep_floor / theta_cost_slope,
gated behind enable_learnable_use) and asks the binding question for the whole program:

HYPOTHESIS.  With theta heritable and costed, is INNATE theta WALLED — gifted-good yet does NOT
  invade from rarity (the recurring `h` pattern: a useful-when-installed sense that selection will
  not build up locally) — or is it DIRECTLY EVOLVABLE (a rare high-theta mutant invades a low-theta
  resident)?  This is the same-axis, INNATE (no within-life learning) control BEFORE building any
  within-life theta learner: if innate theta already invades, the wall does not even apply to this
  axis; if it is walled, a learning-guided (Baldwin) escape is the motivated next rung.

PREDICTION (WALLED, the expected `h` pattern):  cost-ON, the gifted optimum theta* clearly beats
  gifted-worst (mechanism live, cost fair), BUT a rare high-theta mutant does NOT invade a low-theta
  resident (invasion NOT INVADES; drift-robust selection slope mean_s ~ 0; it does not beat the
  pure-cost / perfect-percept control).

FALSIFIER (DIRECTLY EVOLVABLE => this arc's premise VOID for this axis):  the rare high-theta mutant
  INVADES from rarity with a positive drift-robust slope that beats the pure-cost control.  Then
  innate theta is directly evolvable and there is nothing for a within-life learner to assimilate
  here — the controller must VOID the "theta is walled" framing.

NOTE (L1/L2): the printed [CLAIM] is THIS script's claim.  The CONTROLLER applies the verdict
  POLARITY (invades => VOID the wall framing; walled => proceed to the learnable rung), conjunct by
  conjunct, on the committed raw numbers — NOT this script.

INSTRUMENT.
  Step A (L30 COST CALIBRATION): fixed-cohort early-window benefit probe B(theta) at pinned theta
    {0.05,0.1,0.25,0.5,0.8}, COST OFF (theta_cost_slope=0).  The benefit metric is IN-BAND OCCUPANCY
    — the fraction of the alive cohort sitting on the drifting food band, averaged over the early
    window (the Exp 275 demographic-confound-free measure of the sensor's CORE function; a better
    tracker keeps the creature on the moving band).  It is bounded [0,1] and, unlike a raw/differenced
    per-capita intake (which the first draft used and which came out survival-confounded and
    net-negative on a shrinking cohort), it is not confounded by the survival trajectory.  Report
    B(theta), theta* = argmax, benefit ceiling B(theta*) - B(low).
  Step B (pick theta_cost_slope): choose a slope so cost-ON theta* is STILL the functional optimum
    (gifted-best beats gifted-worst) while the cost at theta* is a FAIR fraction of the benefit
    ceiling.  Report the ratio; do NOT foreordain the verdict.
  Step C (INVASION-FROM-RARITY, binding): ecology.evolvability run_invasion_from_rarity on the
    `band_responsiveness` axis — a rare HIGH-theta mutant (theta* / a step up) invading a LOW-theta
    resident, COST ON, >=8 seeds, WITH the L29 drift-robust selection slope (run_local_pairwise_gradient
    mean_s) AND a pure-cost / perfect-percept control (enable_band_staleness off => tracker inert =>
    theta buys no information, only cost).  Print ALL raw per-seed numbers.
"""
from __future__ import annotations

import dataclasses as D
import math

import numpy as np

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.sense_axis import clamp_founder
from ecology.evolvability import gates as G
from ecology.evolvability.metrics import default_thresholds
from ecology.evolvability.trait_axis import BAND_RESPONSIVENESS_AXIS
from ecology import runtime_budget as RB

# --- regime (mirrors the Exp 275 mild band-staleness forage regime) ---
HORIZON = 300
WIN = (100, 300)
CAL_SEEDS = tuple(range(101, 109))          # 8 seeds for the benefit curve
INV_SEEDS = list(range(200, 224))           # 24 fresh seeds (16 gave a small-sample-fragile slope; controller re-ran at 24)
THETA_PINS = (0.05, 0.10, 0.25, 0.50, 0.80)
THETA_LOW = 0.10                            # low-theta resident (near bottom of the useful range)
GOOD_H = 0.60                               # good sensor precision (organ gifted, cost off)
# Regime tuned (population probe) so the fixed cohort survives well above min_pop (~100-120 alive
# at cost-off) -> the invasion arm is MEASURABLE (not NO_VERDICT from mass extinction) AND the
# in-band-occupancy benefit curve is clean/monotone with theta* clearly above the low resident.
CAPACITY = 50.0
REGEN = 1.0
FOOD_CONC = 6.0
INIT_POP = 100
PERIOD = 150.0                              # FAST drift (Exp 275: sensor separably load-bearing)
BAND_WIDTH = 0.06


def _base_cfg(cost_slope: float, *, learnable: bool) -> "object":
    """Common band-staleness forage base.  theta is heritable+read-from-genotype when learnable=True.

    enable_thermosense=False => the ORGAN (h) is free (cost-off); we isolate the theta channel.
    The L30 theta cost is theta_upkeep_floor + theta_cost_slope * band_responsiveness.
    """
    base = SCENARIOS["balanced"]
    cfg = D.replace(
        base,
        horizon=HORIZON, initial_population=INIT_POP, max_population=8000, capacity=CAPACITY,
        enable_thermosense=False, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        enable_food_coupling=True, thermosense_forage_mode=True, enable_band_staleness=True,
        food_optimal_base=0.5, food_optimal_amplitude=0.35, food_optimal_period=PERIOD,
        food_band_width=BAND_WIDTH, food_concentration=FOOD_CONC, regen_rate=REGEN,
        freeze_learning_rate=True,
        enable_learnable_use=learnable, theta_upkeep_floor=0.0, theta_cost_slope=float(cost_slope),
    )
    return cfg


def _pinned_founder(cfg, theta: float):
    """Founder with the good sensor + pinned theta (band_responsiveness); learning_rate 0.30."""
    f = D.replace(FOUNDER, learning_rate=0.30, band_responsiveness=float(theta))
    return clamp_founder(f, GOOD_H, 0.20)


def _in_band_occupancy(theta: float, seed: int, cost_slope: float) -> tuple[float, int]:
    """Fixed-cohort early-window IN-BAND OCCUPANCY at pinned theta (cost as given).

    Benefit metric = fraction of the alive cohort sitting on the drifting food band, averaged
    over the early window.  This is the Exp 275 demographic-confound-free measure of the sensor's
    CORE function (a better tracker keeps the creature on the moving band), bounded in [0,1] and
    NOT confounded by survival trajectory the way a cumulative/differenced intake would be.
    """
    cfg = _base_cfg(cost_slope, learnable=True)
    cfg = D.replace(cfg, founder=_pinned_founder(cfg, theta), founder_mix=None, mutation_rate=0.0)
    eco = Ecology(cfg, seed=seed)
    w = eco.world
    fr: list[float] = []
    while eco.t < HORIZON and not eco.exploded:
        eco.step()
        if WIN[0] <= eco.t <= WIN[1]:
            al = eco.alive_snapshot()
            if al:
                c0 = w.current_food_optimal
                on = sum(1 for c in al if abs(float(w.temperature[c.phenotype.pos]) - c0) <= BAND_WIDTH)
                fr.append(on / len(al))
        if not eco.has_alive():
            break
    return (float(np.mean(fr)) if fr else float("nan")), eco.alive_count()


def _benefit_curve(cost_slope: float) -> dict[float, float]:
    curve: dict[float, float] = {}
    print(f"--- benefit curve B(theta), cost_slope={cost_slope} (in-band occupancy, Exp 275 metric) ---")
    for th in THETA_PINS:
        per = [_in_band_occupancy(th, s, cost_slope) for s in CAL_SEEDS]
        vals = [r for r, _ in per if not math.isnan(r)]
        ext = sum(1 for _, nlive in per if nlive == 0)
        mu = float(np.mean(vals)) if vals else float("nan")
        curve[th] = mu
        print(f"  theta={th:4}: B={mu:.5f}  (n_valid={len(vals)}/{len(CAL_SEEDS)}, extinct={ext})  "
              f"per_seed={[round(r, 4) for r, _ in per]}")
    return curve


def _grad(base, axis, seeds):
    win, lose = default_thresholds(len(seeds))
    g = G.run_local_pairwise_gradient(base, axis, seeds, win_threshold=win, lose_threshold=lose,
                                      min_valid=max(3, 3 * len(seeds) // 4),
                                      window=(50, HORIZON), min_pop=30)
    a = g.aggregate
    wins = sum(1 for r in g.raw_rows if r["inv_frac_final"] > 0.5)
    return dict(verdict=g.verdict, wins=wins, n=len(seeds), s=a["mean_s"],
                inv=a["mean_inv_frac_final"], extinct=g.validity_flags.get("extinct_fraction"),
                per_seed=[(r["seed"], round(r["inv_frac_final"], 4), round(r["s"], 5)) for r in g.raw_rows])


def main() -> None:
    print("=" * 80)
    print("EXP 276a — INNATE theta (learnable-use / band_responsiveness): WALLED or EVOLVABLE?")
    print("=" * 80)

    # --- L25 runtime pre-flight (BLOCKING) ---
    pf_base = D.replace(_base_cfg(0.2, learnable=True),
                        founder=_pinned_founder(_base_cfg(0.2, learnable=True), 0.5), founder_mix=None)
    workers = RB.recommended_workers_for(pf_base, len(INV_SEEDS), horizon=HORIZON)
    rep = RB.preflight([("exp276a", pf_base, INV_SEEDS[0])], horizon=HORIZON,
                       n_jobs=len(INV_SEEDS), max_workers=workers, require_safe=True)
    workers = max(1, int(rep.get("recommended_workers", workers)))
    print(f"RUNTIME PRE-FLIGHT: safe={rep.get('safe')} workers->{workers} "
          f"proj~{rep.get('proj_total_min')} min  flags={rep.get('flags')}")

    # --- Step A: benefit curve, COST OFF ---
    curve = _benefit_curve(0.0)
    valid = {t: b for t, b in curve.items() if not math.isnan(b)}
    theta_star = max(valid, key=valid.get) if valid else float("nan")
    b_low = curve[THETA_PINS[0]]
    b_star = valid.get(theta_star, float("nan"))
    ceiling = (b_star - b_low) if (not math.isnan(b_star) and not math.isnan(b_low)) else float("nan")
    print(f"\n  theta* (argmax B) = {theta_star}   B(theta*)={b_star:.5f}  B(low={THETA_PINS[0]})={b_low:.5f}")
    print(f"  BENEFIT CEILING B(theta*)-B(low) = {ceiling:.5f} in-band occupancy")

    # --- Step B: pick theta_cost_slope so cost@theta* is a FAIR fraction of the ceiling ---
    # Cost per step at theta = slope * theta.  Pick slope so cost@theta* ~ 30% of the benefit
    # ceiling — a fair bite (not so large it inverts the gifted optimum, not so small it is free).
    target_frac = 0.30
    if not math.isnan(ceiling) and ceiling > 0 and not math.isnan(theta_star) and theta_star > 0:
        cost_slope = target_frac * ceiling / theta_star
    else:
        cost_slope = 0.05  # fallback: a small nonzero bite
    cost_at_star = cost_slope * theta_star
    ratio = (cost_at_star / ceiling) if (not math.isnan(ceiling) and ceiling > 0) else float("nan")
    print(f"\n--- Step B: chosen theta_cost_slope={cost_slope:.5f} => cost@theta*={cost_at_star:.5f} "
          f"= {ratio:.2%} of ceiling ---")

    # cost-ON functional-optimum check: gifted-best (theta*) still beats gifted-worst under cost.
    curve_cost = _benefit_curve(cost_slope)
    valid_c = {t: b for t, b in curve_cost.items() if not math.isnan(b)}
    best_c = max(valid_c, key=valid_c.get) if valid_c else float("nan")
    worst_c = min(valid_c, key=valid_c.get) if valid_c else float("nan")
    gifted_best_beats_worst = (not math.isnan(best_c) and not math.isnan(worst_c)
                               and valid_c.get(best_c, 0) > valid_c.get(worst_c, 0))
    print(f"  cost-ON curve argmax={best_c} argmin={worst_c}  gifted-best>worst={gifted_best_beats_worst}")

    # --- Step C: INVASION FROM RARITY (binding), cost ON ---
    # NON-DEGENERACY GUARD (binding): the mutant theta MUST be a genuine STEP UP from the low
    # resident.  If theta* collapses onto (or below) the resident, step the mutant to the next
    # pin strictly above the resident — otherwise the invasion tests resident-vs-resident (h_mut
    # == h_res => axis.get counts both as resident => f=0.5 forever => a spurious "WALLED").
    theta_mut = float(theta_star)
    if theta_mut <= THETA_LOW + 1e-9:
        above = [t for t in THETA_PINS if t > THETA_LOW + 1e-9]
        theta_mut = float(min(above)) if above else float(THETA_LOW + 0.15)
        print(f"\n[GUARD] theta*={theta_star} <= resident {THETA_LOW}; stepping mutant UP to "
              f"{theta_mut} so the invasion is non-degenerate (h_mut != h_res).")
    assert theta_mut > THETA_LOW + 1e-9, "invasion is degenerate: mutant theta == resident theta"

    axis = D.replace(BAND_RESPONSIVENESS_AXIS,
                     resident_value=float(THETA_LOW), mutant_value=theta_mut)
    base_on = _base_cfg(cost_slope, learnable=True)
    # founder carries the good sensor + resident theta; the invasion runner overwrites theta per arm.
    base_on = D.replace(base_on, founder=_pinned_founder(base_on, THETA_LOW), founder_mix=None)

    win, lose = default_thresholds(len(INV_SEEDS))
    print(f"\n--- Step C: invasion-from-rarity  resident theta={THETA_LOW} -> mutant theta={theta_mut} "
          f"(theta*={theta_star}), COST ON (slope={cost_slope:.5f}), {len(INV_SEEDS)} seeds ---")
    inv = G.run_invasion_from_rarity(base_on, axis, INV_SEEDS,
                                     win_threshold=win, lose_threshold=lose,
                                     min_valid=max(4, 3 * len(INV_SEEDS) // 4),
                                     window=(50, HORIZON), min_pop=30, max_workers=workers)
    print(f"  [INV] verdict={inv.verdict}  increase={inv.aggregate['increase_count']}/{inv.aggregate['n_valid']}"
          f"  extinct_frac={inv.validity_flags.get('extinct_fraction')}")
    for r in inv.raw_rows:
        print(f"    seed={r['seed']:4} f_init={r['f_initial']:.4f} f_final={r['f_final']:.4f} "
              f"increased={r['increased']} final_pop={r['final_pop']} extinct={r['extinct']}")

    # L29 drift-robust selection slope (theta-informative) + pure-cost/perfect-percept control.
    print(f"\n  [SLOPE] drift-robust selection slope (local pairwise gradient), cost ON:")
    info = _grad(base_on, axis, INV_SEEDS)
    print(f"    INFO theta {THETA_LOW}->{theta_star}: verdict={info['verdict']} wins={info['wins']}/{info['n']} "
          f"mean_s={info['s']:+.5f} inv={info['inv']:.3f} extinct_frac={info['extinct']}")
    # CONTROL: perfect-percept / pure-cost — enable_band_staleness=False => tracker inert => theta
    # buys NO information, only its L30 cost.  A positive INFO slope must BEAT this to be real.
    base_ctrl = D.replace(base_on, enable_band_staleness=False)
    ctrl = _grad(base_ctrl, axis, INV_SEEDS)
    print(f"    CTRL pure-cost (tracker off): verdict={ctrl['verdict']} wins={ctrl['wins']}/{ctrl['n']} "
          f"mean_s={ctrl['s']:+.5f} inv={ctrl['inv']:.3f} extinct_frac={ctrl['extinct']}")
    print(f"    INFO per-seed (seed, inv_frac_final, s): {info['per_seed']}")
    print(f"    CTRL per-seed (seed, inv_frac_final, s): {ctrl['per_seed']}")
    print(f"    INFO vs CTRL: mean_s {info['s']:+.5f} vs {ctrl['s']:+.5f} (delta={info['s']-ctrl['s']:+.5f}); "
          f"inv {info['inv']:.3f} vs {ctrl['inv']:.3f}")

    # --- Step D: CAN'T-POSE control — the competitive/equilibrium regime where a wall would be a
    # STRONG test. The posable invasion above uses a resource-RICH short regime (regen=1.0, weak
    # competition, saturating benefit). Re-run the invasion in the competitive Exp-275 regime
    # (regen=0.14, horizon=1500): if it returns NO_VERDICT, the population collapses below the
    # validity floor there (the Exp 242-247 stability-vs-competition boundary), so the wall can only
    # be posed in a WEAK regime. ---
    print("\n  [STEP D] competitive-equilibrium regime (regen=0.14, horizon=1500) — posable there?")
    comp = D.replace(base_on, regen_rate=0.14, horizon=1500)
    comp_inv = G.run_invasion_from_rarity(comp, axis, INV_SEEDS, win_threshold=win, lose_threshold=lose,
                                          min_valid=max(4, 3 * len(INV_SEEDS) // 4),
                                          window=(100, 1500), min_pop=30, max_workers=workers)
    print(f"    [COMP-INV] verdict={comp_inv.verdict} "
          f"increase={comp_inv.aggregate.get('increase_count')}/{comp_inv.aggregate.get('n_valid')} "
          f"(NO_VERDICT here = competitive regime can't sustain a measurable population = CAN'T-POSE)")

    # --- CLAIM (polarity applied by the CONTROLLER, not here) ---
    invades = (inv.verdict == "INVADES" and info["s"] > 0.0
               and info["s"] > ctrl["s"] and info["wins"] > ctrl["wins"])
    claim = "DIRECTLY_EVOLVABLE (invades)" if invades else "WALLED (does not invade from rarity)"
    print(f"\n[CLAIM] innate theta is {claim}")
    print("[NOTE] the CONTROLLER applies verdict polarity: invades => VOID the wall framing; "
          "walled => proceed to the learnable (Baldwin) rung.  This script does NOT decide the verdict.")
    print("[CAVEAT] cost-on; one theta parameterization (tracker EMA rate); the wall is posed only in a "
          "resource-RICH weak-competition regime (Step D: the competitive regime CAN'T-POSE); INNATE "
          "(no within-life theta learner yet).")


if __name__ == "__main__":
    main()
