"""experiments/exp248_viability_probe.py — Exp 248 Rung 2: two-trophic viability search.

Rung-1 (geometry probe) was CONFOUNDED: it built genotypes from the raw `founder()`
(energy_capacity=20, bmc=0.5, aging=0.02, NO continuous_regen_rate) on the continuous
depletion substrate, and the prey starved to EXTINCTION by t≈23 even with NO predators.
The continuous-locomotion arc (Exp 240-246) already established a VIABLE monomorphic
continuous config — the Exp-240 calibrated founder (bmc=0.05, mc=0.03, cap=10, thr=4.2,
aging=0.003) + continuous_regen_rate + continuous_capacity. We REUSE that exact config
(imported from exp242_regulated_ess) as the prey base, so viability is not re-derived.
NOTE: the operational base here (regen_rate=1.0, capacity=2.0) PERSISTS to the horizon as a
bounded oscillation with prey tail-mean ~128 (NOT a fixed point — Exp 242 itself graded the
single-population speed ESS CAN'T-POSE). The "N_eq≈374" figure in the source arc corresponds
to the higher capacity=10 regime; viability here only requires PERSISTENCE, which holds.

HYPOTHESIS / PREDICTION: if the substrate's stability-vs-strong-competition wall (Exp 246-247)
also governs a two-trophic system, then NO predation regime will yield persistent coexistence.
PREDECLARED FALSIFIER: if no swept predation-pressure regime keeps BOTH roles alive at the
horizon on all seeds, the prey-escape invasion-from-rarity test is CAN'T-POSE at the viability
gate. [Result: confirmed NEGATIVE — 0 coexistence across 42 best-shot regimes + ~1900
adversarial-verifier runs; predators starve out or boom-bust collapse, no stable band.]

This script:
  (A) CONFIRMS the prey-only viable base persists to a long horizon (sanity that the
      imported config is genuinely viable here).
  (B) SWEEPS predation pressure (N_pred, capture_radius, sensing_radius, assimilation) on
      top of that viable prey base and reports, per regime, whether a persistent TWO-TROPHIC
      coexistence exists at the horizon (both roles alive), with equilibrium headcounts and
      stability (CV). RAW NUMBERS — NO VERDICT.

GO signal (controller judges): ≥1 predation regime with BOTH roles alive at the horizon at
non-trivial, non-exploding headcounts — i.e. a posable predator-prey system. If every regime
either drives prey/predators extinct (too much predation) or has predators die out leaving
the prey base (too little predation), the clean escape-invasion test is NOT posable.
"""
import sys
import os
import importlib.util
import dataclasses as D
import itertools

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.engine import Ecology


def _load(mod_name, rel):
    spec = importlib.util.spec_from_file_location(mod_name, os.path.join(_repo_root, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Reuse the EXACT Exp-242 viable continuous config builders.
_e242 = _load("exp242_regulated_ess", "experiments/exp242_regulated_ess.py")
make_founder = _e242._make_founder      # (speed, ...) -> viable Genotype
make_cfg = _e242._make_cfg              # (speed, cost_slope, regen_rate, ...) -> viable cfg
REGEN_RATE = 1.0                        # mid of exp242 REGEN_SWEEP [0.5,1.0,2.0]
COST_SLOPE = 0.0
HORIZON = 600                           # long enough to see persistence past the transient
SEEDS = [0, 1, 2]


def roles_alive(eco):
    snap = eco.alive_snapshot()
    p = sum(1 for c in snap if c.genotype.role == "prey")
    q = sum(1 for c in snap if c.genotype.role == "predator")
    return p, q


def run_two_trophic(cfg, seed):
    eco = Ecology(cfg, seed=seed)
    prey_series, pred_series = [], []
    while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
        p, q = roles_alive(eco)
        prey_series.append(p)
        pred_series.append(q)
        eco.step()
    p, q = roles_alive(eco)
    prey_series.append(p)
    pred_series.append(q)
    t_end = eco.t
    tail_p = prey_series[max(0, len(prey_series) - 150):]
    tail_q = pred_series[max(0, len(pred_series) - 150):]
    cvp = float(np.std(tail_p) / np.mean(tail_p)) if tail_p and np.mean(tail_p) > 0 else float("nan")
    return {
        "t_end": t_end, "prey_final": p, "pred_final": q,
        "prey_eq": float(np.mean(tail_p)), "pred_eq": float(np.mean(tail_q)),
        "cv_prey": cvp, "exploded": eco.exploded,
        "both_alive_end": (p > 0 and q > 0),
    }


def build_two_trophic_cfg(n_prey, n_pred, capture_radius, sensing_radius, assimilation,
                          regen_rate=REGEN_RATE, pred_bmc=None, pred_mc=None,
                          pred_start_frac=0.75):
    prey_geno = D.replace(make_founder(1.0), role="prey")
    pred_over = {"role": "predator"}
    if pred_bmc is not None:
        pred_over["baseline_metabolic_cost"] = pred_bmc
    if pred_mc is not None:
        pred_over["movement_cost"] = pred_mc
    pred_geno = D.replace(make_founder(1.4), **pred_over)
    cfg = make_cfg(speed=1.0, cost_slope=COST_SLOPE, regen_rate=regen_rate,
                   horizon=HORIZON, founder_mix=((prey_geno, n_prey), (pred_geno, n_pred)))
    return D.replace(
        cfg,
        enable_predation=True,
        freeze_prey_speed=True,
        mutate_predator_speed=False,
        capture_radius=capture_radius,
        sensing_radius=sensing_radius,
        assimilation_efficiency=assimilation,
        pred_start_energy_frac=pred_start_frac,
    )


def confirm_prey_base():
    """(A) Prey-only viable base must persist (sanity check of the imported config)."""
    prey_geno = D.replace(make_founder(1.0), role="prey")
    cfg = make_cfg(speed=1.0, cost_slope=COST_SLOPE, regen_rate=REGEN_RATE,
                   horizon=HORIZON, founder_mix=((prey_geno, 21),))
    out = []
    for s in SEEDS:
        eco = Ecology(cfg, seed=s)
        series = []
        while eco.has_alive() and not eco.exploded and eco.t < cfg.horizon:
            series.append(roles_alive(eco)[0])
            eco.step()
        series.append(roles_alive(eco)[0])
        out.append((eco.t, series[-1], float(np.mean(series[max(0, len(series) - 150):]))))
    return out


def main():
    lines = []
    lines.append("=" * 96)
    lines.append("Exp 248 Rung 2 — TWO-TROPHIC VIABILITY SEARCH on the Exp-242 viable prey base.")
    lines.append("RAW NUMBERS — controller judges go/abort.")
    lines.append(f"prey base: Exp-240 calibrated founder (bmc=0.05,mc=0.03,cap=10,thr=4.2,aging=0.003)")
    lines.append(f"regen_rate={REGEN_RATE} capacity=2.0 cost_slope={COST_SLOPE} horizon={HORIZON} seeds={SEEDS}")
    lines.append("=" * 96)

    # (A) Prey-only base sanity
    base = confirm_prey_base()
    lines.append("(A) PREY-ONLY base (no predators) — must persist:")
    for i, (t_end, n_fin, n_eq) in enumerate(base):
        lines.append(f"    seed={SEEDS[i]}  t_end={t_end}  prey_final={n_fin}  prey_eq(tail)={n_eq:.1f}")
    base_persists = all(t == HORIZON and nf > 0 for (t, nf, _) in base)
    lines.append(f"    => prey base persists to t={HORIZON} on all seeds: {base_persists}")
    lines.append("")

    if not base_persists:
        lines.append("!! Prey base did NOT persist — predation sweep is moot; fix the base first.")
        out = "\n".join(lines)
        print(out)
        _write(out)
        return

    # (B) Two-trophic predation-pressure sweep
    lines.append("(B) TWO-TROPHIC predation-pressure sweep (prey base held viable):")
    lines.append(f"{'n_prey':>6} {'n_pred':>6} {'capR':>5} {'senseR':>7} {'assim':>6} "
                 f"{'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} {'cv_prey':>8} {'BOTH@end':>9} {'explode':>8}")
    lines.append("-" * 96)
    n_preys = [21]
    n_preds = [2, 4, 8]
    capRs = [0.2, 0.4, 0.6]
    senseRs = [1.5, 3.0]
    assims = [0.6]
    go_rows = []
    for n_prey, n_pred, capR, senseR, assim in itertools.product(n_preys, n_preds, capRs, senseRs, assims):
        cfg = build_two_trophic_cfg(n_prey, n_pred, capR, senseR, assim)
        res = [run_two_trophic(cfg, s) for s in SEEDS]
        t_end = np.mean([r["t_end"] for r in res])
        prey_eq = np.mean([r["prey_eq"] for r in res])
        pred_eq = np.mean([r["pred_eq"] for r in res])
        cvp = np.nanmean([r["cv_prey"] for r in res])
        both_all = all(r["both_alive_end"] for r in res)
        explode = any(r["exploded"] for r in res)
        lines.append(
            f"{n_prey:>6} {n_pred:>6} {capR:>5.2f} {senseR:>7.1f} {assim:>6.2f} "
            f"{t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} {cvp:>8.3f} {str(both_all):>9} {str(explode):>8}"
        )
        if both_all and not explode:
            go_rows.append((n_prey, n_pred, capR, senseR, assim, prey_eq, pred_eq, cvp))
    lines.append("")
    lines.append(f"Regimes with BOTH roles alive at t={HORIZON} on ALL seeds (coexistence): {len(go_rows)}")
    for g in go_rows:
        lines.append(f"    COEXIST: n_pred={g[1]} capR={g[2]} senseR={g[3]} assim={g[4]} "
                     f"prey_eq={g[5]:.1f} pred_eq={g[6]:.1f} cv_prey={g[7]:.3f}")
    lines.append("")

    # (C) BEST-SHOT coexistence sweep: a MORE PRODUCTIVE prey base (higher regen → higher
    # N_eq → more biomass to absorb predation) crossed with gentler/finer predation and lower
    # assimilation. If coexistence fails even here, the negative is robust.
    lines.append("(C) BEST-SHOT coexistence sweep (productive prey base + gentle predation):")
    lines.append(f"{'regen':>6} {'n_pred':>6} {'capR':>5} {'senseR':>7} {'assim':>6} "
                 f"{'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} {'cv_prey':>8} {'BOTH@end':>9} {'explode':>8}")
    lines.append("-" * 96)
    c_go = []
    for regen, n_pred, capR, senseR, assim in itertools.product(
            [2.0, 4.0], [3, 5], [0.15, 0.30], [2.0], [0.3, 0.6]):
        cfg = build_two_trophic_cfg(21, n_pred, capR, senseR, assim, regen_rate=regen)
        res = [run_two_trophic(cfg, s) for s in SEEDS]
        t_end = np.mean([r["t_end"] for r in res])
        prey_eq = np.mean([r["prey_eq"] for r in res])
        pred_eq = np.mean([r["pred_eq"] for r in res])
        cvp = np.nanmean([r["cv_prey"] for r in res])
        both_all = all(r["both_alive_end"] for r in res)
        explode = any(r["exploded"] for r in res)
        lines.append(
            f"{regen:>6.1f} {n_pred:>6} {capR:>5.2f} {senseR:>7.1f} {assim:>6.2f} "
            f"{t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} {cvp:>8.3f} {str(both_all):>9} {str(explode):>8}"
        )
        if both_all and not explode:
            c_go.append((regen, n_pred, capR, senseR, assim, prey_eq, pred_eq, cvp))
    lines.append("")
    lines.append(f"(C) coexistence regimes (both roles alive at t={HORIZON}, all seeds): {len(c_go)}")
    for g in c_go:
        lines.append(f"    COEXIST: regen={g[0]} n_pred={g[1]} capR={g[2]} senseR={g[3]} assim={g[4]} "
                     f"prey_eq={g[5]:.1f} pred_eq={g[6]:.1f} cv_prey={g[7]:.3f}")
    lines.append("")

    # (D) PREDATOR best-shot: give the predator a carnivore-efficient metabolism (low bmc/mc),
    # high assimilation, full start energy, on the productive prey base + moderate predation.
    # If even a maximally-efficient predator cannot persist, predator non-viability is NOT a
    # mere predator-energetics mis-calibration — it is the substrate.
    lines.append("(D) PREDATOR best-shot (efficient carnivore metabolism + high assimilation):")
    lines.append(f"{'p_bmc':>6} {'p_mc':>6} {'assim':>6} {'pstart':>7} {'n_pred':>6} {'capR':>5} "
                 f"{'t_end':>7} {'prey_eq':>8} {'pred_eq':>8} {'BOTH@end':>9} {'explode':>8}")
    lines.append("-" * 96)
    d_go = []
    for p_bmc, p_mc, assim, pstart, n_pred, capR in itertools.product(
            [0.02, 0.005], [0.01], [0.9], [0.9], [3, 6], [0.3, 0.5]):
        cfg = build_two_trophic_cfg(21, n_pred, capR, 2.5, assim, regen_rate=2.0,
                                    pred_bmc=p_bmc, pred_mc=p_mc, pred_start_frac=pstart)
        res = [run_two_trophic(cfg, s) for s in SEEDS]
        t_end = np.mean([r["t_end"] for r in res])
        prey_eq = np.mean([r["prey_eq"] for r in res])
        pred_eq = np.mean([r["pred_eq"] for r in res])
        both_all = all(r["both_alive_end"] for r in res)
        explode = any(r["exploded"] for r in res)
        lines.append(
            f"{p_bmc:>6.3f} {p_mc:>6.2f} {assim:>6.2f} {pstart:>7.2f} {n_pred:>6} {capR:>5.2f} "
            f"{t_end:>7.1f} {prey_eq:>8.1f} {pred_eq:>8.1f} {str(both_all):>9} {str(explode):>8}"
        )
        if both_all and not explode:
            d_go.append((p_bmc, p_mc, assim, n_pred, capR, prey_eq, pred_eq))
    lines.append("")
    lines.append(f"(D) coexistence regimes (efficient predator): {len(d_go)}")
    for g in d_go:
        lines.append(f"    COEXIST: p_bmc={g[0]} p_mc={g[1]} assim={g[2]} n_pred={g[3]} capR={g[4]} "
                     f"prey_eq={g[5]:.1f} pred_eq={g[6]:.1f}")
    lines.append("")
    lines.append(f"TOTAL coexistence regimes found (B + C + D): {len(go_rows) + len(c_go) + len(d_go)}")
    out = "\n".join(lines)
    print(out)
    _write(out)


def _write(out):
    out_dir = os.path.join(_repo_root, "experiments", "outputs")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp248_viability.txt"), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/exp248_viability.txt]")


if __name__ == "__main__":
    main()
