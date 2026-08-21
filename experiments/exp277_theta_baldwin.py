"""Exp 277 — WITHIN-LIFE theta learning: does a Baldwin effect ASSIMILATE the innate prior?

DIRECTION: loop/directions/learning-guided-evolution.md.  Exp 276 established (CLOSED for the
INNATE control): innate theta = band_responsiveness (the Exp 201 band-tracker EMA rate) does NOT
robustly invade from rarity in the posable RICH regime (cap 50 / regen 1.0 / horizon 300; benefit
curve monotone, theta* ~ 0.8, L30 cost slope ~ 0.072) — a walled "useful-when-installed" sense.
Exp 277 adds a WITHIN-LIFE learner: each creature keeps its INNATE theta (heritable, mutated) AND a
per-life REALIZED theta it LEARNS by a 1-D stochastic hill-climb starting from the innate prior
(imperfect/gradual, so a better innate prior confers a lifetime head-start).  NON-LAMARCKIAN:
offspring inherit the mutated INNATE theta, never the parent's learned realized theta.

HYPOTHESIS.  A within-life learner creates a BALDWIN GRADIENT — because learning takes time to
  converge, a creature born with a BETTER innate theta prior reaches good tracking SOONER and so
  earns higher lifetime fitness than one born with a bad prior even though both eventually learn.
  That gradient can then ASSIMILATE the good value into the GENE POOL: over generations the mean
  INNATE theta CLIMBS (genetic assimilation) even though the same innate axis was WALLED without
  learning (Exp 276).  This is the classic learning-guided-evolution escape of a local-gradient wall.

FOUR BRANCHES (the CONTROLLER applies the verdict, conjunct by conjunct, on the raw numbers below):
  PASS-Baldwin : innate theta CLIMBS over generations with learn-ON, learn-ON beats learn-OFF, the
                 climb SURVIVES the CLAMPED control (not a map-memory free-ride), and the GENOTYPE
                 (newborn/gene-pool innate theta) — not just the phenotype — moves.
  STALL        : innate theta stays low with learn-ON ≈ learn-OFF (no assimilation; learner does
                 not create a usable Baldwin gradient here).
  SHIELDING    : the REALIZED (phenotype) theta climbs but the INNATE (genotype) theta does NOT —
                 learning shields the gene from selection (phenotypic plasticity absorbs it all).
  SUBSTITUTION : the apparent innate climb COLLAPSES under the CLAMPED control — the "assimilation"
                 was actually the resource-map learning_rate free-riding, not theta.

FALSIFIER (of the Baldwin escape):  learn-ON does NOT move the newborn/gene-pool INNATE theta above
  the learn-OFF wall (STALL), OR the climb is pure phenotype (SHIELDING), OR it collapses under the
  CLAMPED control (SUBSTITUTION).  Any of these ⇒ NO genetic assimilation on this axis.

CRITICAL PRECONDITION (Step A):  if a within-life learner ERASES the innate-prior fitness
  difference (all priors reach ~equal lifetime fitness because learning fully compensates), that is
  SHIELDING-BY-CONSTRUCTION — there is NO Baldwin gradient for selection to act on and the evolution
  test (Step B) cannot show assimilation.  Report it honestly; do NOT run Step B on faith.

NOTE (L1/L2): the printed [CLAIM] is THIS script's summary.  The CONTROLLER decides the branch.
  Check per-seed DISPERSION (mean-of-opposites) before ANY "climb" claim.
"""
from __future__ import annotations

import dataclasses as D
import math

import numpy as np

from ecology.engine import Ecology
from ecology.scenarios import SCENARIOS, FOUNDER
from ecology.sense_axis import clamp_founder
from ecology import runtime_budget as RB

# --- regime (the Exp 276 posable RICH regime: cap 50 / regen 1.0; fitness-gating theta) ---
GOOD_H = 0.60                          # good sensor precision (organ gifted, cost off)
CAPACITY = 50.0
REGEN = 1.0
FOOD_CONC = 6.0
PERIOD = 150.0                         # FAST drift (sensor separably load-bearing)
BAND_WIDTH = 0.06
THETA_LOW = 0.10                       # WALLED founder innate theta (Exp 276 low resident)

# --- Step A (pre-flight) knobs ---
A_HORIZON = 300
A_SEEDS = tuple(range(301, 309))       # 8 seeds
A_THETA_PINS = (0.10, 0.25, 0.50, 0.80)

# --- Step B (assimilation) knobs — LONG horizon = many generations ---
B_HORIZON = 1500
B_SEEDS = list(range(320, 328))        # 8 seeds (per arm)
B_INIT_POP = 100
GENE_POOL_WINDOW = 300                 # creatures born in the last window = current gene pool
COST_SLOPE = 0.072                     # the Exp 276 L30 slope (fair cost bite at theta*)
LEARN_PERIOD = 20
LEARN_STEP = 0.10


def _base_cfg(*, learnable: bool, learn: bool, cost_slope: float, horizon: int,
              init_pop: int, mutate: bool, clamp_map: bool):
    """Common band-staleness forage base in the Exp 276 posable RICH regime.

    learnable=True  -> theta is heritable + read from genotype (Exp 276 innate channel).
    learn=True      -> the within-life hill-climb runs (Exp 277 realized-theta channel + L30 on it).
    clamp_map=True  -> freeze_learning_rate: pins the resource-map learning_rate so a map-memory
                       free-ride cannot masquerade as theta assimilation (the SUBSTITUTION control).
    """
    base = SCENARIOS["balanced"]
    return D.replace(
        base,
        horizon=horizon, initial_population=init_pop, max_population=8000, capacity=CAPACITY,
        mutation_rate=(0.05 if mutate else 0.0),
        enable_thermosense=False, enable_temperature=True, temperature_stress_scale=0.0,
        thermosense_active_threshold=0.05, thermosense_noise_base=0.5, thermal_avoidance_weight=4.0,
        enable_food_coupling=True, thermosense_forage_mode=True, enable_band_staleness=True,
        food_optimal_base=0.5, food_optimal_amplitude=0.35, food_optimal_period=PERIOD,
        food_band_width=BAND_WIDTH, food_concentration=FOOD_CONC, regen_rate=REGEN,
        freeze_learning_rate=clamp_map,
        enable_learnable_use=learnable, theta_upkeep_floor=0.0, theta_cost_slope=float(cost_slope),
        enable_theta_learning=learn, theta_learn_period=LEARN_PERIOD, theta_learn_step=LEARN_STEP,
        theta_learn_cost=0.0,
    )


def _pinned_founder(cfg, theta: float):
    """Founder with the good sensor + pinned INNATE theta (band_responsiveness); learning_rate 0.30."""
    f = D.replace(FOUNDER, learning_rate=0.30, band_responsiveness=float(theta))
    return clamp_founder(f, GOOD_H, 0.20)


# ---------------------------------------------------------------------------
# Step A — Baldwin-gradient PRECONDITION: lifetime fitness vs the INNATE prior, learn ON.
# ---------------------------------------------------------------------------
def _lifetime_fitness(theta_prior: float, seed: int, cost_slope: float) -> tuple[float, float, int]:
    """Monomorphic cohort, learn ON, cost ON, innate theta PINNED at theta_prior (mutation off).

    Returns (mean offspring_count over all creatures ever born, mean lifetime resource_eaten, n_born).
    A GOOD innate prior should yield HIGHER lifetime fitness than a BAD one DESPITE learning if a
    Baldwin gradient exists (learning takes LEARN_PERIOD*k steps to climb from the prior).
    """
    cfg = _base_cfg(learnable=True, learn=True, cost_slope=cost_slope, horizon=A_HORIZON,
                    init_pop=B_INIT_POP, mutate=False, clamp_map=False)
    cfg = D.replace(cfg, founder=_pinned_founder(cfg, theta_prior), founder_mix=None)
    eco = Ecology(cfg, seed=seed)
    eco.run()
    born = eco._creatures  # EVERY creature ever born (the _creatures-vs-alive gotcha: this is the cohort)
    if not born:
        return float("nan"), float("nan"), 0
    off = float(np.mean([c.phenotype.offspring_count for c in born]))
    eaten = float(np.mean([c.phenotype.resource_eaten for c in born]))
    return off, eaten, len(born)


def _step_a(cost_slope: float) -> dict[float, tuple[float, float]]:
    print(f"\n--- STEP A: Baldwin-gradient PRECONDITION (lifetime fitness vs INNATE prior, learn ON, "
          f"cost_slope={cost_slope}) ---")
    print("  If a GOOD prior yields HIGHER lifetime fitness than a BAD one DESPITE learning, the")
    print("  Baldwin gradient EXISTS.  If learning ERASES the prior gap (all ~equal), that is")
    print("  SHIELDING-BY-CONSTRUCTION (no assimilation possible) — reported, not run on faith.")
    out: dict[float, tuple[float, float]] = {}
    for th in A_THETA_PINS:
        per = [_lifetime_fitness(th, s, cost_slope) for s in A_SEEDS]
        offs = [o for o, _, n in per if not math.isnan(o) and n > 0]
        eats = [e for _, e, n in per if not math.isnan(e) and n > 0]
        mu_off = float(np.mean(offs)) if offs else float("nan")
        mu_eat = float(np.mean(eats)) if eats else float("nan")
        out[th] = (mu_off, mu_eat)
        print(f"  innate_prior={th:4}: mean_offspring={mu_off:.4f}  mean_eaten={mu_eat:.3f}  "
              f"per_seed_off={[round(o, 3) for o, _, _ in per]}")
    return out


# ---------------------------------------------------------------------------
# Step B — ASSIMILATION test: does the GENE-POOL innate theta climb over generations?
# ---------------------------------------------------------------------------
def _gene_pool_and_phenotype(eco: Ecology) -> tuple[float, float, int, int]:
    """(mean NEWBORN/gene-pool INNATE theta, mean REALIZED theta of ADULTS, n_newborn, n_adult).

    NEWBORN gene pool = creatures BORN in the last GENE_POOL_WINDOW steps (the current genotype
    distribution feeding selection) — the ASSIMILATION signal.  REALIZED theta of ADULTS = the
    learned phenotype of the currently-alive creatures.  We read eco._creatures directly (the
    _creatures-vs-alive gotcha: _creatures is EVERY creature ever born; we filter by birth_t/alive).
    """
    cutoff = eco.t - GENE_POOL_WINDOW
    newborn = [c for c in eco._creatures if c.phenotype.birth_t >= cutoff]
    innate = [c.genotype.band_responsiveness for c in newborn]
    alive = [c for c in eco._creatures if c.is_alive()]
    realized = [c.policy.realized_theta for c in alive
                if c.policy is not None and c.policy.realized_theta is not None]
    mu_innate = float(np.mean(innate)) if innate else float("nan")
    mu_real = float(np.mean(realized)) if realized else float("nan")
    return mu_innate, mu_real, len(newborn), len(alive)


def _run_arm(*, learn: bool, clamp_map: bool, seed: int) -> tuple[float, float, int, int]:
    """One evolving population, founder at the WALLED low innate theta, mutation ON.

    learn-OFF (clamp_map=False)  → byte-reproduces the Exp 276 wall (innate theta stays low).
    learn-ON  (clamp_map=False)  → the assimilation arm.
    learn-ON  (clamp_map=True)   → the CLAMPED / SUBSTITUTION control (map learning_rate frozen).
    """
    cfg = _base_cfg(learnable=True, learn=learn, cost_slope=COST_SLOPE, horizon=B_HORIZON,
                    init_pop=B_INIT_POP, mutate=True, clamp_map=clamp_map)
    cfg = D.replace(cfg, founder=_pinned_founder(cfg, THETA_LOW), founder_mix=None)
    eco = Ecology(cfg, seed=seed)
    eco.run()
    return _gene_pool_and_phenotype(eco)


def _summ(rows: list[tuple[float, float, int, int]]) -> dict[str, float]:
    innate = np.array([r[0] for r in rows if not math.isnan(r[0])])
    real = np.array([r[1] for r in rows if not math.isnan(r[1])])
    return dict(
        innate_mean=float(np.mean(innate)) if innate.size else float("nan"),
        innate_sd=float(np.std(innate)) if innate.size else float("nan"),
        real_mean=float(np.mean(real)) if real.size else float("nan"),
        real_sd=float(np.std(real)) if real.size else float("nan"),
        n=int(innate.size),
    )


def main() -> None:
    print("=" * 84)
    print("EXP 277 — WITHIN-LIFE theta learning: Baldwin assimilation of the WALLED innate prior?")
    print("=" * 84)

    # --- L25 runtime pre-flight (BLOCKING): the evolution loop over many generations is a
    # compute-class jump; keep pop/horizon/seeds right-sized. ---
    pf_base = _base_cfg(learnable=True, learn=True, cost_slope=COST_SLOPE, horizon=B_HORIZON,
                        init_pop=B_INIT_POP, mutate=True, clamp_map=False)
    pf_base = D.replace(pf_base, founder=_pinned_founder(pf_base, THETA_LOW), founder_mix=None)
    workers = RB.recommended_workers_for(pf_base, len(B_SEEDS), horizon=B_HORIZON)
    rep = RB.preflight([("exp277", pf_base, B_SEEDS[0])], horizon=B_HORIZON,
                       n_jobs=len(B_SEEDS), max_workers=workers, require_safe=True)
    workers = max(1, int(rep.get("recommended_workers", workers)))
    print(f"RUNTIME PRE-FLIGHT: safe={rep.get('safe')} workers->{workers} "
          f"proj~{rep.get('proj_total_min')} min  flags={rep.get('flags')}")

    # --- Step A: Baldwin-gradient precondition ---
    fit = _step_a(COST_SLOPE)
    valid = {t: o for t, (o, _) in fit.items() if not math.isnan(o)}
    if valid:
        best_t = max(valid, key=valid.get)
        worst_t = min(valid, key=valid.get)
        gap = valid[best_t] - valid[worst_t]
        # relative gap vs the best (how much of the fitness a bad prior forfeits despite learning)
        rel = (gap / valid[best_t]) if valid[best_t] > 0 else float("nan")
        print(f"\n  [STEP A] best_prior={best_t} (off={valid[best_t]:.4f})  "
              f"worst_prior={worst_t} (off={valid[worst_t]:.4f})  "
              f"Baldwin fitness gap={gap:.4f} ({rel:.1%} of best)")
        print(f"  [STEP A] gap>0 (a good prior beats a bad one DESPITE learning) = {gap > 0} "
              f"— if ~0, learning SHIELDS the prior by construction (no gradient for Step B).")

    # --- Step B: the assimilation test (3 arms x seeds) ---
    print(f"\n--- STEP B: ASSIMILATION over generations (founder innate theta={THETA_LOW}, mutation ON, "
          f"horizon={B_HORIZON}, {len(B_SEEDS)} seeds/arm) ---")
    print("  ARM learn-OFF   = byte-reproduces the Exp 276 wall (innate theta should stay low).")
    print("  ARM learn-ON    = assimilation arm (does newborn/gene-pool INNATE theta climb?).")
    print("  ARM CLAMPED     = learn-ON + freeze_learning_rate (map free-ride removed).")

    off_rows, on_rows, clamp_rows = [], [], []
    for s in B_SEEDS:
        r_off = _run_arm(learn=False, clamp_map=False, seed=s)
        r_on = _run_arm(learn=True, clamp_map=False, seed=s)
        r_cl = _run_arm(learn=True, clamp_map=True, seed=s)
        off_rows.append(r_off); on_rows.append(r_on); clamp_rows.append(r_cl)
        print(f"  seed={s}: OFF innate={r_off[0]:.4f} real={r_off[1]:.4f} (n_new={r_off[2]},alive={r_off[3]}) | "
              f"ON innate={r_on[0]:.4f} real={r_on[1]:.4f} (n_new={r_on[2]},alive={r_on[3]}) | "
              f"CLAMP innate={r_cl[0]:.4f} real={r_cl[1]:.4f} (n_new={r_cl[2]},alive={r_cl[3]})")

    s_off = _summ(off_rows); s_on = _summ(on_rows); s_cl = _summ(clamp_rows)
    print(f"\n  [ARM OFF ]  newborn INNATE theta: mean={s_off['innate_mean']:.4f} sd={s_off['innate_sd']:.4f}  "
          f"| adult REALIZED: mean={s_off['real_mean']:.4f} sd={s_off['real_sd']:.4f}  (n={s_off['n']})")
    print(f"  [ARM ON  ]  newborn INNATE theta: mean={s_on['innate_mean']:.4f} sd={s_on['innate_sd']:.4f}  "
          f"| adult REALIZED: mean={s_on['real_mean']:.4f} sd={s_on['real_sd']:.4f}  (n={s_on['n']})")
    print(f"  [ARM CLAMP] newborn INNATE theta: mean={s_cl['innate_mean']:.4f} sd={s_cl['innate_sd']:.4f}  "
          f"| adult REALIZED: mean={s_cl['real_mean']:.4f} sd={s_cl['real_sd']:.4f}  (n={s_cl['n']})")

    # per-seed dispersion / mean-of-opposites check on the assimilation signal (ON - OFF innate).
    deltas = [on_rows[i][0] - off_rows[i][0] for i in range(len(B_SEEDS))
              if not math.isnan(on_rows[i][0]) and not math.isnan(off_rows[i][0])]
    if deltas:
        d = np.array(deltas)
        print(f"\n  [DISPERSION] per-seed (ON_innate - OFF_innate): mean={float(d.mean()):+.4f} "
              f"sd={float(d.std()):.4f}  per_seed={[round(x, 4) for x in deltas]}")
        print(f"  [DISPERSION] sd>>|mean|? {float(d.std()) > 2 * abs(float(d.mean()))} "
              f"(if True the 'climb' may be a mean-of-opposites — check the per-seed split).")

    # --- CLAIM (branch chosen by the CONTROLLER on the raw numbers, not here) ---
    innate_climbs = (not math.isnan(s_on['innate_mean']) and s_on['innate_mean'] > THETA_LOW + 0.05)
    on_beats_off = (not math.isnan(s_on['innate_mean']) and not math.isnan(s_off['innate_mean'])
                    and s_on['innate_mean'] > s_off['innate_mean'] + 0.02)
    survives_clamp = (not math.isnan(s_cl['innate_mean'])
                      and s_cl['innate_mean'] > s_off['innate_mean'] + 0.02)
    real_climbs = (not math.isnan(s_on['real_mean']) and s_on['real_mean'] > THETA_LOW + 0.05)
    if innate_climbs and on_beats_off and survives_clamp:
        claim = "PASS-Baldwin (innate climbs, ON beats OFF, survives CLAMPED)"
    elif real_climbs and not innate_climbs:
        claim = "SHIELDING (realized climbs but innate does NOT)"
    elif innate_climbs and on_beats_off and not survives_clamp:
        claim = "SUBSTITUTION (innate climb collapses under CLAMPED)"
    else:
        claim = "STALL (innate stays low, ON approx OFF)"
    print(f"\n[CLAIM] {claim}")
    print("[NOTE] the CONTROLLER applies the 4-branch verdict conjunct-by-conjunct on the RAW "
          "per-seed numbers (check dispersion first).  This script does NOT decide the branch.")
    print("[CAVEAT] cost-on; one theta parameterization (tracker EMA rate); one hill-climb "
          "(period=20/step=0.10); the wall is posed only in the resource-RICH weak-competition "
          "regime; NON-LAMARCKIAN by construction (children inherit mutated INNATE theta).")


if __name__ == "__main__":
    main()
