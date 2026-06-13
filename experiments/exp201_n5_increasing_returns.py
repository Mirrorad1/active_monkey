"""Exp 201 — N5: the INCREASING-RETURNS escape — does a convex / speed-gated benefit cross the valley?
(pre-registered in loop/directions/population-ecology.md, BEFORE any verdict-seed data).

Hypothesis (the one untested escape from exp199/200's no-Goldilocks-gradient ceiling): in a regime
with INCREASING returns to precision — a FAST-drifting food band whose center each creature must
PRIVATELY TRACK (an EMA whose responsiveness + reading-noise scale with thermosense_intensity, so a
crude tracker chronically LAGS the moving band into already-depleted cells while a precise tracker
locks on) — a FUNCTIONAL thermosense organ evolves de-novo: gene-pool NEWBORN mean intensity crosses
0.30 in the FAST arm. exp200's fatal flaw was that the band center was handed to the policy for FREE
(world.current_food_optimal); here precision finally buys tracking quality of a MOVING target — which
(unlike a static rich cell) cannot be free-ridden via the resource-memory map or won by creature-id order.

Mechanism (PROVIDED, gated engine; exp194-200 byte-identical, hash-verified): enable_band_staleness
replaces the free read with a per-creature band_estimate EMA tracker; alpha = clamp(intensity*resp,0,1)
and reading-noise = noise_base*(1-intensity).  L19 GUARD: food intake is NEVER reward=f(intensity) —
intensity keys ONLY the tracker; intake falls out of position-vs-depletion via the unchanged consume().

Arms (temp ON, stress 0, cheap efficient organ inefficiency 0.20/floor 0, founder intensity 0.10,
enable_thermosense, forage mode, food_concentration 8, forage weight 4, noise 0.5; horizon 12000;
fresh verdict seeds {33,34,35,36,37}):
  FAST       : the increasing-returns treatment — fast band, the primary de-novo CLIMB test (>0.30 => POSITIVE).
  MEDIUM     : intermediate band speed — predeclared to land BETWEEN SLOW and FAST (speed-gating is graded).
  SLOW       : NULL control — band barely drifts; a crude tracker keeps up (lag harmless) => must stay <0.15.
               (The binding L19 internal falsifier: if SLOW also climbs, the relation was IMPOSED, not earned.)
  CLAMPED_LR : = FAST but learning_rate is FROZEN at the founder value (confound-killer) => isolates whether
               the climb is thermosense, not resource-memory substitution.
  SATURATING : = FAST geometry but enable_band_staleness=False (the exp200 FREE read) => the within-experiment
               saturating baseline; predeclared to stay primitive (~0.08), making FAST-vs-SATURATING the exact
               tracked-vs-free contrast.
  USELESS    : enable_food_coupling=False (organ pure cost) => the primitive anchor (~0.06).
Metric = NEWBORN (gene-pool) mean thermosense_intensity, end = mean over creatures born in [10000,12000].

Pilots (disclosed, per L7): pilot seeds {110,111}. v1 (horizon 4000) found a REAL speed-gated climb
(fast band ~0.18 vs slow null ~0.096) with strip~0 (interference competition does NOT bite — the signal
is pure convex tracking-threshold geometry, not frequency dependence; disclosed scoping). v2 (horizon
12000) found the climb is a TRANSIENT peaking ~t=2000 then DECAYING to a low equilibrium (chosen FAST regime A_base settled 0.13/0.20, the most reliable of a swept set; SLOW null 0.12/0.06);
the regime was fixed to the one giving the climb its best fair shot BEFORE the fresh-seed run.

Predictions (5 fresh seeds; L21 validity = pop>=10 AND a newborn cohort in window):
  P1 determinism: same seed -> identical event hash (FAST + USELESS on seed 33).
  P2 validity: FAST valid >=4/5 seeds (else loosen scarcity in a DISCLOSED pilot before verdict, never after).
  P3 (CORE escape, => POSITIVE): FAST end gene-pool newborn mean intensity > 0.30 in >=4/5 valid seeds
     AND speed-gating holds (FAST_mean > SLOW_mean AND SLOW_mean < 0.15)
     AND CLAMPED_LR also > 0.30 in >=4/5 (the climb is thermosense, not memory substitution).

Falsifiers:
  PRIMARY -> NEGATIVE (the stronger, more general wall): FAST end mean < 0.15 in a majority of valid seeds
     => the primitive ceiling survives INCREASING-RETURNS-to-precision (more general than exp199/200's no-gradient).
  MIXED: FAST end mean in [0.15, 0.30] (climbed above primitive AND above SLOW, speed-gated, but did NOT reach
     functional) -- the predeclared 'convexity flattens above the threshold' outcome, an honest partial result.
  INTERNAL DEGENERACY FALSIFIER (=> any positive DISCARDED as artifact): SLOW_mean >= 0.30 OR USELESS_mean >= 0.15
     (selection appears where it must not -> the relation was plumbing/imposed, not the speed-gated threshold).
  CONFOUND FALSIFIER (=> POSITIVE uninterpretable): FAST > 0.30 but CLAMPED_LR not, OR newborn learning_rate rises
     in FAST vs USELESS (the climb is resource-memory substitution, not thermosense).
  F1 non-determinism -> NEGATIVE (infra).  Regression: enable_band_staleness=False must reproduce committed
     exp194 + exp200 hashes (verified in tests) else NEGATIVE (infra).

Verdict rule: NEGATIVE if not P1; else DEGENERATE-NEGATIVE if the internal degeneracy falsifier fires; else
POSITIVE iff P3; else NEGATIVE iff FAST majority-primitive (<0.15); else MIXED. Repo token POSITIVE/NEGATIVE/MIXED.
Diagnostics (reported, NON-gating): per-arm mean newborn learning_rate (confound indicator); max individual
intensity per arm (high-I individuals arise but do not fix => weak marginal gradient); the r(p) returns-curve
probe (experiments/exp201_returns_probe.py, separate seeds, run + committed BEFORE the verdict). Predicting the
escape FAILS to reach functional (pilots suggest MIXED/NEGATIVE) -- the human's increasing-returns hypothesis is
tested on FRESH seeds.  Founders + costs PROVIDED; the band tracker is a PROVIDED heuristic (not learned).
"""
from __future__ import annotations

import dataclasses as D
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology, EcologyConfig
from ecology.scenarios import SCENARIOS, FOUNDER

# ---------------------------------------------------------------------------
# FIXED experiment constants (pre-registered; do NOT change after verdict data)
# ---------------------------------------------------------------------------
SEEDS = [33, 34, 35, 36, 37]            # fresh, disjoint from exp200 {23-27} + pilots {110,111}
HORIZON = 12000
MAX_POP = 5000
CHECKPOINT_STRIDE = 1000
NEWBORN_WINDOW_START = 10000
NEWBORN_WINDOW_END = HORIZON
MIN_VALID_POP = 10                       # L21: arm/seed VALID iff final_pop >= MIN_VALID_POP + newborn cohort

# Regime geometry (FIXED from the disclosed pilot BEFORE the verdict run) -------
FAST_PERIOD = 60.0
MEDIUM_PERIOD = 120.0
SLOW_PERIOD = 2400.0
BAND_WIDTH = 0.12
REGEN_RATE = 0.20
FOOD_CONC = 8.0
BAND_RESP = 1.0

BASE = dict(
    enable_thermosense=True,
    enable_temperature=True,
    temperature_stress_scale=0.0,
    tolerance_cost_scale=0.0,
    comfort_amplitude=0.0,
    thermosense_upkeep_floor=0.0,
    thermosense_active_threshold=0.05,
    thermosense_noise_base=0.5,
    thermal_avoidance_weight=4.0,
    food_optimal_base=0.5,
    food_optimal_amplitude=0.3,
    food_concentration=FOOD_CONC,
)


# ---------------------------------------------------------------------------
# Founder + config factories (importable for unit tests)
# ---------------------------------------------------------------------------

def founder() -> Any:
    """Primitive foothold 0.10, cheap efficient organ (inefficiency=0.2), tolerance=0.10."""
    return D.replace(
        FOUNDER, thermosense_intensity=0.10, thermosense_inefficiency=0.2,
        temperature_tolerance=0.10,
    )


def _forage(period: float, *, staleness: bool, freeze_lr: bool = False,
            band: float = BAND_WIDTH, regen: float = REGEN_RATE) -> EcologyConfig:
    """A food-coupled forage arm.  staleness=True -> band-staleness tracker (Exp 201);
    staleness=False -> exp200 free read (the SATURATING baseline)."""
    return D.replace(
        SCENARIOS["balanced"], horizon=HORIZON, max_population=MAX_POP, founder=founder(),
        regen_rate=regen,
        enable_food_coupling=True, thermosense_forage_mode=True,
        food_band_width=band, food_optimal_period=period,
        enable_band_staleness=staleness, band_responsiveness=BAND_RESP,
        freeze_learning_rate=freeze_lr, **BASE,
    )


def cfg_useless() -> EcologyConfig:
    """No food coupling -> thermosense useless -> primitive baseline (organ pure cost)."""
    return D.replace(
        SCENARIOS["balanced"], horizon=HORIZON, max_population=MAX_POP, founder=founder(),
        regen_rate=REGEN_RATE,
        enable_food_coupling=False, thermosense_forage_mode=False,
        food_band_width=BAND_WIDTH, food_optimal_period=1500.0,
        enable_band_staleness=False, **BASE,  # BASE food_optimal_amplitude inert (coupling off)
    )


_ARM_CFGS: dict[str, EcologyConfig] = {
    "FAST":       _forage(FAST_PERIOD,   staleness=True),
    "MEDIUM":     _forage(MEDIUM_PERIOD, staleness=True),
    "SLOW":       _forage(SLOW_PERIOD,   staleness=True),
    "CLAMPED_LR": _forage(FAST_PERIOD,   staleness=True, freeze_lr=True),
    "SATURATING": _forage(FAST_PERIOD,   staleness=False),   # exp200 free read, same geometry
    "USELESS":    cfg_useless(),
}
ARMS = ["FAST", "MEDIUM", "SLOW", "CLAMPED_LR", "SATURATING", "USELESS"]


def make_arm_cfg(arm: str) -> EcologyConfig:
    return _ARM_CFGS[arm]


# ---------------------------------------------------------------------------
# Newborn-window gene-pool metric (importable for unit tests)
# ---------------------------------------------------------------------------

def newborn_mean_in_window(creatures: list, lo: int, hi: int, attr: str) -> float:
    """Mean of genotype.<attr> for non-founder creatures with birth_t in [lo, hi]; NaN if none."""
    nb = [c for c in creatures if c.parent_id is not None and lo <= c.phenotype.birth_t <= hi]
    if not nb:
        return float("nan")
    return float(np.mean([getattr(c.genotype, attr) for c in nb]))


def is_valid_run(eco: Ecology, end_nb_intensity: float) -> bool:
    """L21: reached horizon (not exploded) AND final pop >= MIN_VALID_POP AND a newborn cohort."""
    alive = eco._alive()
    return (
        eco.t >= HORIZON and not eco.exploded
        and len(alive) >= MIN_VALID_POP
        and not math.isnan(end_nb_intensity)
    )


# ---------------------------------------------------------------------------
# Core runner: one arm x one seed
# ---------------------------------------------------------------------------

def run_arm(cfg: EcologyConfig, seed: int) -> dict[str, Any]:
    eco = Ecology(cfg, seed=seed)
    horizon = cfg.horizon
    trajectory: list[dict[str, Any]] = []
    checkpoints = set(range(CHECKPOINT_STRIDE, horizon + 1, CHECKPOINT_STRIDE))

    while eco.t < horizon and not eco.exploded:
        eco.step()
        if eco.t in checkpoints:
            alive = eco._alive()
            nb = newborn_mean_in_window(eco._creatures, eco.t - CHECKPOINT_STRIDE, eco.t, "thermosense_intensity")
            mx = max((c.genotype.thermosense_intensity for c in alive), default=float("nan"))
            trajectory.append({
                "t": eco.t, "pop": len(alive),
                "newborn_mean_intensity": float("nan") if math.isnan(nb) else nb,
                "max_intensity": float("nan") if math.isnan(mx) else mx,
            })
        if not eco._alive():
            break

    alive = eco._alive()
    end_nb = newborn_mean_in_window(eco._creatures, NEWBORN_WINDOW_START, horizon, "thermosense_intensity")
    end_lr = newborn_mean_in_window(eco._creatures, NEWBORN_WINDOW_START, horizon, "learning_rate")
    all_int = [c.genotype.thermosense_intensity for c in eco._creatures]
    max_ever = max(all_int) if all_int else float("nan")
    valid = is_valid_run(eco, end_nb)

    return {"trajectory": trajectory, "summary": {
        "seed": seed, "steps_run": eco.t,
        "reached_horizon": eco.t >= HORIZON and not eco.exploded,
        "extinct": len(alive) == 0, "exploded": eco.exploded, "final_pop": len(alive),
        "end_newborn_intensity": end_nb, "end_newborn_learning_rate": end_lr,
        "max_ever_intensity": max_ever, "valid": valid,
        "extinction_step": eco.t if len(alive) == 0 else None,
        "events_hash": eco.events_hash(),
    }}


def check_p1_determinism(hash_fast: str, hash_useless: str) -> bool:
    a = run_arm(make_arm_cfg("FAST"), seed=SEEDS[0])
    b = run_arm(make_arm_cfg("USELESS"), seed=SEEDS[0])
    return (a["summary"]["events_hash"] == hash_fast
            and b["summary"]["events_hash"] == hash_useless)


# ---------------------------------------------------------------------------
# Verdict logic (implement EXACTLY as pre-registered)
# ---------------------------------------------------------------------------

def _arm_mean(end_nb: dict[int, float | None]) -> float:
    vals = [v for v in end_nb.values() if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def _count_above(end_nb: dict[int, float | None], thr: float) -> int:
    return sum(v is not None and v > thr for v in end_nb.values())


def _count_below(end_nb: dict[int, float | None], thr: float) -> int:
    return sum(v is not None and v < thr for v in end_nb.values())


def compute_verdict(end_nb: dict[str, dict[int, float | None]],
                    lr_means: dict[str, float], p1: bool) -> dict[str, Any]:
    fast_mean = _arm_mean(end_nb["FAST"])
    slow_mean = _arm_mean(end_nb["SLOW"])
    useless_mean = _arm_mean(end_nb["USELESS"])

    n_valid_fast = sum(v is not None for v in end_nb["FAST"].values())
    P2 = n_valid_fast >= 4

    speed_gating = (fast_mean > slow_mean) and (slow_mean < 0.15)
    clamped_ok = _count_above(end_nb["CLAMPED_LR"], 0.30) >= 4
    fast_functional = _count_above(end_nb["FAST"], 0.30) >= 4
    P3 = fast_functional and speed_gating and clamped_ok

    # Internal degeneracy falsifier: selection appears where it MUST NOT.
    degenerate = (slow_mean >= 0.30) or (useless_mean >= 0.15)

    # Confound: learning_rate substitution.
    lr_confound = (
        fast_functional and not clamped_ok
    ) or (lr_means.get("FAST", 0.0) - lr_means.get("USELESS", 0.0) > 0.10)

    fast_primitive = _count_below(end_nb["FAST"], 0.15) >= 3  # majority of 5

    if not p1:
        verdict, token = "NEGATIVE (F1 non-determinism, infra)", "NEGATIVE"
    elif degenerate:
        verdict, token = "NEGATIVE / DEGENERATE (speed-gating control failed; relation imposed)", "NEGATIVE"
    elif P3 and not lr_confound:
        verdict, token = "POSITIVE (increasing-returns escape WORKS; functional organ evolves)", "POSITIVE"
    elif P3 and lr_confound:
        verdict, token = "MIXED / CONFOUNDED (functional but learning_rate substitution suspected)", "MIXED"
    elif fast_primitive:
        verdict, token = "NEGATIVE (ceiling general even under increasing returns)", "NEGATIVE"
    else:
        verdict, token = "MIXED (real speed-gated climb above primitive, not functional)", "MIXED"

    return {
        "P1": p1, "P2": P2, "P3": P3,
        "fast_mean": fast_mean, "slow_mean": slow_mean, "useless_mean": useless_mean,
        "speed_gating": speed_gating, "clamped_ok": clamped_ok,
        "fast_functional": fast_functional, "fast_primitive": fast_primitive,
        "degenerate": degenerate, "lr_confound": lr_confound,
        "verdict": verdict, "token": token,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    out_dir = _REPO / "experiments" / "outputs" / "exp201_n5_increasing_returns"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[int, dict[str, Any]]] = {a: {} for a in ARMS}
    print(f"Exp 201: {len(ARMS)} arms x {len(SEEDS)} seeds (horizon={HORIZON}) ...")
    for arm in ARMS:
        cfg = make_arm_cfg(arm)
        for seed in SEEDS:
            print(f"  arm {arm:<11} seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            res = run_arm(cfg, seed=seed)
            s = res["summary"]
            tag = "EXTINCT" if s["extinct"] else f"pop={s['final_pop']}"
            nb = "EXTINCT" if not s["valid"] else f"{s['end_newborn_intensity']:.4f}"
            print(f"done {time.time()-t0:.0f}s  {tag}  end_nb={nb}  valid={s['valid']}")
            results[arm][seed] = res
            with open(out_dir / f"traj_arm{arm}_seed{seed}.json", "w") as f:
                json.dump({"arm": arm, "seed": seed, **res}, f, indent=2)

    print(f"\nP1 determinism (seed {SEEDS[0]}, FAST + USELESS) ...", end=" ", flush=True)
    p1 = check_p1_determinism(
        results["FAST"][SEEDS[0]]["summary"]["events_hash"],
        results["USELESS"][SEEDS[0]]["summary"]["events_hash"],
    )
    print("PASS" if p1 else "FAIL")

    end_nb: dict[str, dict[int, float | None]] = {}
    lr_means: dict[str, float] = {}
    for arm in ARMS:
        end_nb[arm] = {}
        lrs = []
        for seed in SEEDS:
            s = results[arm][seed]["summary"]
            end_nb[arm][seed] = s["end_newborn_intensity"] if s["valid"] else None
            if s["valid"] and not math.isnan(s["end_newborn_learning_rate"]):
                lrs.append(s["end_newborn_learning_rate"])
        lr_means[arm] = float(np.mean(lrs)) if lrs else float("nan")

    v = compute_verdict(end_nb, lr_means, p1)

    # ---- summary text ----
    L: list[str] = []
    L.append("=" * 76)
    L.append("EXP 201 — N5: INCREASING-RETURNS ESCAPE (band-staleness tracker) — SUMMARY")
    L.append("=" * 76)
    L.append("")
    L.append(f"{'Arm':<11}" + "".join(f"  seed{s}" for s in SEEDS) + f"  {'mean':>8}  {'max':>7}  {'nb_lr':>6}  {'n_ext':>5}")
    L.append("-" * 76)
    for arm in ARMS:
        per = "".join(
            f"  {'EXTINCT':>7}" if end_nb[arm][s] is None else f"  {end_nb[arm][s]:>7.4f}"
            for s in SEEDS
        )
        m = _arm_mean(end_nb[arm])
        mx = max(results[arm][s]["summary"]["max_ever_intensity"] for s in SEEDS)
        n_ext = sum(end_nb[arm][s] is None for s in SEEDS)
        mstr = "nan" if math.isnan(m) else f"{m:.4f}"
        lrstr = "nan" if math.isnan(lr_means[arm]) else f"{lr_means[arm]:.3f}"
        L.append(f"{arm:<11}{per}  {mstr:>8}  {mx:>7.3f}  {lrstr:>6}  {n_ext:>5}")
    L.append("")
    L.append("FINAL POPULATIONS per arm/seed:")
    L.append(f"{'Arm':<11}" + "".join(f"  seed{s}" for s in SEEDS))
    L.append("-" * 60)
    for arm in ARMS:
        L.append(f"{arm:<11}" + "".join(f"  {results[arm][s]['summary']['final_pop']:>6}" for s in SEEDS))
    L.append("")
    L.append("PREDICTION / FALSIFIER EVALUATIONS:")
    L.append(f"  P1 determinism (FAST+USELESS seed {SEEDS[0]}):     {'PASS' if v['P1'] else 'FAIL'}")
    L.append(f"  P2 validity (FAST valid >=4/5):                 {'PASS' if v['P2'] else 'FAIL'}")
    L.append(f"  FAST mean={v['fast_mean']:.4f}  SLOW mean={v['slow_mean']:.4f}  USELESS mean={v['useless_mean']:.4f}")
    L.append(f"  speed-gating (FAST>SLOW & SLOW<0.15):           {'YES' if v['speed_gating'] else 'NO'}")
    L.append(f"  FAST functional (>0.30 in >=4/5):               {'YES' if v['fast_functional'] else 'NO'}")
    L.append(f"  CLAMPED_LR functional (>0.30 in >=4/5):         {'YES' if v['clamped_ok'] else 'NO'}")
    L.append(f"  internal degeneracy falsifier fired:            {'YES' if v['degenerate'] else 'NO'}")
    L.append(f"  learning_rate confound flagged:                 {'YES' if v['lr_confound'] else 'NO'}")
    L.append(f"  P3 CORE (functional+speed-gated+clamped-clean):  {'YES' if v['P3'] else 'NO'}")
    L.append("")
    L.append(f"VERDICT: {v['verdict']} (repo token: {v['token']})")
    L.append("")
    L.append(f"runtime: {time.time()-t_start:.0f}s")
    text = "\n".join(L)
    print("\n" + text)

    with open(_REPO / "experiments" / "outputs" / "exp201.txt", "w") as f:
        f.write(text + "\n")

    per_arm_seed: dict[str, Any] = {}
    for arm in ARMS:
        per_arm_seed[arm] = {}
        for seed in SEEDS:
            s = results[arm][seed]["summary"]
            per_arm_seed[arm][str(seed)] = {
                k: s[k] for k in (
                    "valid", "end_newborn_intensity", "end_newborn_learning_rate",
                    "final_pop", "extinct", "max_ever_intensity", "steps_run", "events_hash",
                )
            }
    with open(out_dir / "verdict.json", "w") as f:
        json.dump({
            "experiment": "exp201", "seeds": SEEDS, "per_arm_seed": per_arm_seed,
            "lr_means": lr_means,
            "end_nb": {a: {str(s): end_nb[a][s] for s in SEEDS} for a in ARMS},
            "verdict": v["verdict"], "token": v["token"],
            "predictions": {k: v[k] for k in (
                "P1", "P2", "P3", "speed_gating", "fast_functional", "clamped_ok",
                "degenerate", "lr_confound", "fast_mean", "slow_mean", "useless_mean")},
        }, f, indent=2)
    print(f"\n[saved {out_dir / 'verdict.json'}]")


if __name__ == "__main__":
    main()
