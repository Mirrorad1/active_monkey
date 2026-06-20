"""experiments/exp260_onesided_queen_preflight.py — Exp 260: LOCAL-SELECTION-GRADIENT
preflight for the ONE-SIDED Red Queen on the robust Exp-258 patch-mosaic regime.

WHY (the gap Exp 259 left). Exp 259 declared "the Red Queen engages" from (A) a LARGE-step
invasion-from-rarity (escape 1.0->1.4, a 40% jump) and (B) trait-MEAN trajectories (esc 1->2.98
static; esc->0.68 / atk->1.59 co-evolve). The Evolvability-Preflight binding lesson (the closed
sense/memory arc) is that a moving trait-mean is NOT a local selection gradient: a large step can
be positive while the small-eps LOCAL step is flat drift, and a costless trait can ratchet without
an ESS. Before any fuller co-evolution run, this preflight asks the binding question per trait,
ONE-SIDED (focal trait vs a FROZEN monomorphic antagonist): does the trait have a positive LOCAL
selection gradient at the resident, on the substrate where coexistence is actually posed?

METHOD. 50/50 breed-true common garden on the robust Exp-258 BOTH regime (ring-8, refuge
access=0.30 frac=0.25, migration=0.05, rotating async amp=0.4; collapse-prone within-patch
attack_a=0.05 K_pred=40 hmax=0.05). The focal species is split 50/50 into a resident lineage
(trait=r) and a small-eps mutant lineage (trait=r+0.1), interleaved by founding order; BOTH species
breed true (freeze_*_trait), enable_trait_evolution ON so the prey escape COST (the brake) is live.
The antagonist is monomorphic and frozen. Measure the mutant frequency f0=0.5 -> f1 over a window;
selection coefficient s = (logit(f1)-logit(f0))/window (ecology.evolvability.metrics). Across 8
seeds, count wins (s>0) and apply the repo 7/8-STRICT convention default_thresholds(n) (POS iff
wins>=ceil(7n/8); NEG iff wins<=floor(3n/8); else AMBIGUOUS/drift). NEGATIVE is intentionally
easier than POSITIVE so the preflight never over-declares a gradient.

MECHANISM PREDICTION (from the formulas; v_ij=1/(1+escape_k*max(0,escape-attack)), prey birth
*= max(0,1-escape_cost*max(0,escape-1)), predator birth has NO attack cost):
  - PREY escape is COSTED -> expect a positive local gradient at low resident that FLIPS sign at
    high resident (an interior ESS), validating the Exp-259 static-arm saturation as real bounded
    selection rather than a cost-free ratchet.
  - PREDATOR attack is COSTLESS and its benefit SATURATES (once attack>=prey_escape, v=1 for all):
    expect a FLAT gradient at the monomorphic start (prey escape=1.0, attack inert) and a positive
    NON-flipping ratchet only against FAST frozen prey (escape=3.0) -> conditional, no ESS.
  This asymmetry (prey bounded/costed vs predator costless/conditional) would mechanistically
  explain Exp-259 predator dominance.

CONTROLS (binding). Drift nulls where the focal trait is causally INERT must come back NEUTRAL
(not >=7/8): (prey) escape_cost=0 AND predator frozen at attack=2.0 so a 1.1 mutant still has
escape-attack<0 => v=1 (no benefit, no cost); (predator) prey frozen at escape=1.0 so attack>=escape
=> v=1 (no benefit). A cost-null (escape_cost=0, predator attack=1.0) isolates the brake: it must be
MORE positive than the costed gradient. PREDECLARED FALSIFIER: if a drift null itself wins >=7/8,
the gradient readout is a measurement artifact -> NO_VERDICT (downgrade, do not claim selection).

RAW NUMBERS — the controller applies the gate and judges. FRESH seeds 500-507.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
from ecology.evolvability.metrics import selection_coefficient_freq, count_wins, default_thresholds

SEEDS = list(range(500, 508))           # 8 fresh seeds -> default_thresholds(8) = (7, 3)
EPS = 0.1                               # small local step
WINDOW = 400                            # competition window (BOTH regime persists well past this)
MIN_FOCAL = 50                          # focal pop below this at end => seed invalid (cannot measure)


def base_cfg(prey_escape, pred_attack, escape_cost, window=WINDOW):
    """Robust Exp-258 BOTH regime, breed-true (both traits frozen), evolution-cost machinery ON."""
    return PatchMosaicConfig(
        n_patches=8, topology="ring",
        attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=0.0, mutation_sd=0.05,
        escape_cost=escape_cost, escape_baseline=1.0,
        prey_escape=prey_escape, pred_attack=pred_attack,
        freeze_prey_trait=True, freeze_predator_trait=True,
        trait_min=0.0, trait_max=6.0,
        horizon=window, n_prey0_per_patch=40, n_pred0_per_patch=8)


def _focal_traits(sim, focal):
    return [c.trait for p in sim.patches for c in (p.prey if focal == "prey" else p.predators)]


def gradient_probe(focal, resident, eps, antagonist, escape_cost, seed, window=WINDOW):
    """One 50/50 breed-true common-garden run. focal in {'prey','pred'}.

    Returns (f0, f1, s, valid, coexist). f=mutant frequency among the focal species;
    valid requires focal pop >= MIN_FOCAL and antagonist alive and not exploded.
    """
    if focal == "prey":
        cfg = base_cfg(prey_escape=resident, pred_attack=antagonist, escape_cost=escape_cost, window=window)
    else:
        cfg = base_cfg(prey_escape=antagonist, pred_attack=resident, escape_cost=escape_cost, window=window)
    sim = PatchMosaicSim(cfg, seed)

    # Relabel half the focal founders as the mutant lineage (interleaved by founding order so
    # there is no systematic cid-order advantage; the drift null verifies this empirically).
    thr = resident + eps / 2.0
    for patch in sim.patches:
        lst = patch.prey if focal == "prey" else patch.predators
        for k, c in enumerate(lst):
            c.trait = (resident + eps) if (k % 2 == 1) else resident

    def mut_frac():
        traits = _focal_traits(sim, focal)
        if not traits:
            return None
        return sum(t > thr for t in traits) / len(traits)

    f0 = mut_frac()
    exploded = False
    for _ in range(window):
        focal_n = len(_focal_traits(sim, focal))
        ant_n = sum(len(p.predators if focal == "prey" else p.prey) for p in sim.patches)
        if focal_n == 0 or ant_n == 0:
            break
        sim.step()
        if sum(len(p.prey) + len(p.predators) for p in sim.patches) > cfg.pop_cap:
            exploded = True
            break

    focal_n = len(_focal_traits(sim, focal))
    ant_n = sum(len(p.predators if focal == "prey" else p.prey) for p in sim.patches)
    f1 = mut_frac()
    coexist = (focal_n > 0 and ant_n > 0)
    valid = (focal_n >= MIN_FOCAL) and (ant_n > 0) and (not exploded)
    s = selection_coefficient_freq(f0, f1, window) if (f0 is not None and f1 is not None) else float("nan")
    if not valid:
        s = float("nan")
    return f0, f1, s, valid, coexist


def run_cell(focal, resident, antagonist, escape_cost, eps=EPS, seeds=SEEDS):
    """Run all seeds for one (focal, resident, antagonist, cost, eps) cell; summarize."""
    rows = [gradient_probe(focal, resident, eps, antagonist, escape_cost, s) for s in seeds]
    svals = [r[2] for r in rows]
    pairs = [(s, 0.0) for s in svals]          # win = s > 0  (mutant favoured)
    wins, n_valid = count_wins(pairs, eps=0.0)
    f1s = [r[1] for r in rows if r[1] is not None and r[4]]
    coex = sum(1 for r in rows if r[4])
    valid_s = [s for s in svals if not math.isnan(s)]
    win_t, lose_t = default_thresholds(n_valid) if n_valid > 0 else (0, 0)
    if n_valid == 0:
        verdict = "NO_DATA"
    elif wins >= win_t:
        verdict = "POS"
    elif wins <= lose_t:
        verdict = "NEG"
    else:
        verdict = "AMBIG"
    return dict(wins=wins, n_valid=n_valid, mean_s=(float(np.mean(valid_s)) if valid_s else float("nan")),
                mean_f1=(float(np.mean(f1s)) if f1s else float("nan")), coex=coex, n=len(seeds), verdict=verdict)


def fmt(c):
    return (f"{c['wins']:>2}/{c['n_valid']:<2} {c['verdict']:>5}  s={c['mean_s']:+.4f}  "
            f"f1={c['mean_f1']:.3f}  coex={c['coex']}/{c['n']}")


def main():
    smoke = "--smoke" in sys.argv
    seeds = SEEDS[:2] if smoke else SEEDS
    L = []
    L.append("=" * 100)
    L.append("Exp 260 — ONE-SIDED RED QUEEN local-gradient PREFLIGHT on the robust Exp-258 BOTH regime. RAW.")
    L.append(f"50/50 breed-true common garden; eps={EPS}; window={WINDOW}; seeds={seeds}; gate=7/8-strict (default_thresholds).")
    L.append("win = mutant selection coeff s>0; POS iff wins>=ceil(7n/8), NEG iff wins<=floor(3n/8); drift nulls must be NEUTRAL.")
    L.append("=" * 100)

    # ---------------- PREY side: escape vs frozen predator (attack=1.0), cost ON ----------------
    L.append("")
    L.append("(P) PREY-ESCAPE gradient  [predator FROZEN attack=1.0; escape_cost=0.15]  — costed, expect interior ESS")
    L.append(f"{'resident E':>11} | {'wins':>9} {'verdict':>7}  {'mean s':>9}  {'mean f1':>8}  coexist")
    L.append("-" * 78)
    prey_anchors = (1.0, 2.0, 3.0) if smoke else (1.0, 1.5, 2.0, 2.5, 3.0)
    prey_cells = {}
    for E in prey_anchors:
        c = run_cell("prey", E, antagonist=1.0, escape_cost=0.15, seeds=seeds)
        prey_cells[E] = c
        L.append(f"{E:>11.2f} | {fmt(c)}")

    # ---------------- PREY controls ----------------
    L.append("")
    L.append("(P-ctrl) PREY-ESCAPE controls at E=1.0")
    cost_null = run_cell("prey", 1.0, antagonist=1.0, escape_cost=0.0, seeds=seeds)
    drift_null_prey = run_cell("prey", 1.0, antagonist=2.0, escape_cost=0.0, seeds=seeds)  # mutant 1.1 < attack 2.0 => v=1 inert
    L.append(f"{'cost-null':>11} | {fmt(cost_null)}   (escape_cost=0, pred attack=1.0: benefit only -> should be >= costed)")
    L.append(f"{'DRIFT-null':>11} | {fmt(drift_null_prey)}   (escape_cost=0, pred attack=2.0: escape INERT v=1 -> must be NEUTRAL)")

    # ---------------- PREDATOR side: attack vs frozen prey ----------------
    L.append("")
    L.append("(Q) PREDATOR-ATTACK gradient  [prey FROZEN]  — costless; expect FLAT at saturated prey, ratchet vs fast prey")
    L.append(f"{'resident A':>11} | {'E_bg':>4} | {'wins':>9} {'verdict':>7}  {'mean s':>9}  {'mean f1':>8}  coexist")
    L.append("-" * 86)
    # vs FAST frozen prey (escape=3.0): expect positive ratchet up to A~3.0
    pred_anchors = (1.0, 2.0) if smoke else (1.0, 1.5, 2.0, 2.5)
    pred_cells_fast = {}
    for A in pred_anchors:
        c = run_cell("pred", A, antagonist=3.0, escape_cost=0.15, seeds=seeds)
        pred_cells_fast[A] = c
        L.append(f"{A:>11.2f} | {3.0:>4.1f} | {fmt(c)}")
    # vs SATURATED frozen prey (escape=1.0): attack inert (v=1) -> FLAT  [doubles as predator drift null]
    drift_null_pred = run_cell("pred", 1.0, antagonist=1.0, escape_cost=0.15, seeds=seeds)
    L.append(f"{1.0:>11.2f} | {1.0:>4.1f} | {fmt(drift_null_pred)}   <- monomorphic start / DRIFT-null (attack INERT -> must be NEUTRAL)")

    # ---------------- Predeclared synthesis (controller still judges raw numbers) ----------------
    L.append("")
    L.append("PREDECLARED READOUT (controller adjudicates against the raw rows above):")
    L.append(f"  prey gradient sign by E: " + ", ".join(f"E{E:g}={prey_cells[E]['verdict']}(s{prey_cells[E]['mean_s']:+.3f})" for E in prey_anchors))
    flip = [E for E in prey_anchors if prey_cells[E]['verdict'] == 'NEG']
    L.append(f"    -> interior ESS iff POS at low E and flips to NEG at high E. first-NEG anchor(s): {flip if flip else 'none (no flip in range)'}")
    L.append(f"  prey drift-null: {drift_null_prey['verdict']} (wins {drift_null_prey['wins']}/{drift_null_prey['n_valid']}) — FALSIFIED if POS")
    L.append(f"  prey cost-null:  {cost_null['verdict']} (s {cost_null['mean_s']:+.3f}) vs costed E=1.0 s {prey_cells[prey_anchors[0]]['mean_s']:+.3f} — cost is the brake iff cost-null more positive")
    L.append(f"  predator vs fast prey (E=3.0) by A: " + ", ".join(f"A{A:g}={pred_cells_fast[A]['verdict']}(s{pred_cells_fast[A]['mean_s']:+.3f})" for A in pred_anchors))
    L.append(f"  predator monomorphic/drift-null (E=1.0): {drift_null_pred['verdict']} (wins {drift_null_pred['wins']}/{drift_null_pred['n_valid']}) — FALSIFIED if POS")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp260_onesided_queen_preflight_smoke.txt" if smoke else "exp260_onesided_queen_preflight.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
