"""experiments/exp261_costed_predator_attack.py — Exp 261: does COSTING the predator's attack
(symmetric to the prey's escape cost) SYMMETRIZE the patch-mosaic Red Queen, or is the
predator-dominance of Exp-259 terminal?  Single causal bit = `attack_cost` (gated, default 0.0).

WHY. Exp-260 pinned the mechanism behind Exp-259 predator-dominance: prey escape is COSTED
(net local gradient sub-threshold, interior ESS ~2.5-3) while predator attack is COSTLESS
(8/8 conditional ratchet vs fast prey, no ESS).  If that asymmetry is the CAUSE, adding the
missing predator-attack cost should (A) give the predator-attack local gradient an interior
ESS too, and (B) pull the arms-race outcome from predator-dominance toward parity, then (over-
costed) to prey-dominance — a MONOTONE handoff in attack_cost.

HYPOTHESIS / PREDICTION: there exists an attack_cost (predeclared ~0.15, the prey-symmetric
value) at which the Red Queen SYMMETRIZES — BUT only as a TWO-SIDED, GRADIENT-LEVEL, STABLE-
PLATEAU result: |D=atk@3k-esc@3k|<=0.30 AND both traits viable AND a stable multi-checkpoint
plateau AND Part A shows an interior predator ESS at that cost AND the sweep traces
dominance->balance->prey-dominance monotonically.  A single tuned trait-mean crossing between
predator-dominance and prey-dominance is NOT symmetry (the most-dangerous confound).

PREDECLARED FALSIFIER (claim NO symmetrization / NO_VERDICT if ANY):
 (1) baseline attack_cost=0 in Part B does NOT reproduce Exp-259 predator-dominance (D not >=+0.40),
     OR the attack_cost=0 events_hash is NOT byte-identical to the Exp-259 co-evolve config;
 (2) the gap D does NOT decrease monotonically (seed-mean) as attack_cost rises (no brake);
 (3) Part A: attack_cost>=0.15 does NOT create an interior predator ESS (gradient stays POS/non-
     flipping vs fast frozen prey) — the cost didn't bound predator selection like prey's;
 (4) Part A predator DRIFT-NULL (attack causally inert) wins >=7/8 -> readout artifact -> NO_VERDICT.

CONTROLS: attack_cost=0 reproduces Exp-259 (Part B) and Exp-260 predator ratchet (Part A);
byte-identity isolation (attack_cost=0 hash == pinned Exp-259-config golden); predator drift-null
neutral; cost-null contrast (ac=0 more positive than ac>0 at each anchor); prey side HELD FIXED
(escape_cost=0.15) + a prey-ESS sanity row; is_population_valid viability filter + coexistence
parity; birth_p_pred clamp diagnostic (cost must not be shaving a saturated value); full trait
distributions (std) so equal means cannot hide split clades; matched mutation (rate/sd) both species.

trait_max=4.0 for Part B (Exp-259's actual value — its base_cfg did NOT set trait_max so it used
the default 4.0; NOT 6.0).  Part A is breed-true (no mutation) so trait_max never binds.
FRESH disjoint seeds: Part A 600-607, Part B 700-709.  RAW NUMBERS — the controller adjudicates.
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from ecology.patchmosaic import PatchMosaicConfig, PatchMosaicSim
from ecology.evolvability.metrics import selection_coefficient_freq, count_wins, default_thresholds, is_population_valid

SEEDS_A = list(range(600, 608))   # Part A: 8 fresh seeds -> default_thresholds(8) = (7, 3)
SEEDS_B = list(range(700, 710))   # Part B: 10 fresh seeds
EPS = 0.1
WINDOW = 400
MIN_FOCAL = 50
HORIZON_B = 3000
EXP259_GOLDEN = "6f7f3f54096a425a4bb2f23f3eaaa1c4f03ad291a3258e4eaf23da0abc1addb6"  # ac=0, h=300, seed=400, coevolve


# --------------------------------------------------------------------------------------------
# Part A — predator-side local-gradient preflight WITH attack_cost (reuses the Exp-260 instrument)
# --------------------------------------------------------------------------------------------
def cfg_A(prey_escape, pred_attack, attack_cost, escape_cost=0.15):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=0.0, mutation_sd=0.05,
        escape_cost=escape_cost, escape_baseline=1.0, attack_cost=attack_cost, attack_baseline=1.0,
        prey_escape=prey_escape, pred_attack=pred_attack,
        freeze_prey_trait=True, freeze_predator_trait=True,
        trait_min=0.0, trait_max=6.0, horizon=WINDOW, n_prey0_per_patch=40, n_pred0_per_patch=8)


def _focal_traits(sim, focal):
    return [c.trait for p in sim.patches for c in (p.prey if focal == "prey" else p.predators)]


def gradient_probe(focal, resident, antagonist, attack_cost, seed, eps=EPS, escape_cost=0.15, window=WINDOW):
    if focal == "prey":
        cfg = cfg_A(prey_escape=resident, pred_attack=antagonist, attack_cost=attack_cost, escape_cost=escape_cost)
    else:
        cfg = cfg_A(prey_escape=antagonist, pred_attack=resident, attack_cost=attack_cost, escape_cost=escape_cost)
    sim = PatchMosaicSim(cfg, seed)
    thr = resident + eps / 2.0
    for patch in sim.patches:
        lst = patch.prey if focal == "prey" else patch.predators
        for k, c in enumerate(lst):
            c.trait = (resident + eps) if (k % 2 == 1) else resident

    def mut_frac():
        traits = _focal_traits(sim, focal)
        return (sum(t > thr for t in traits) / len(traits)) if traits else None

    f0 = mut_frac()
    exploded = False
    for _ in range(window):
        if len(_focal_traits(sim, focal)) == 0:
            break
        if sum(len(p.predators if focal == "prey" else p.prey) for p in sim.patches) == 0:
            break
        sim.step()
        if sum(len(p.prey) + len(p.predators) for p in sim.patches) > cfg.pop_cap:
            exploded = True
            break
    focal_n = len(_focal_traits(sim, focal))
    ant_n = sum(len(p.predators if focal == "prey" else p.prey) for p in sim.patches)
    f1 = mut_frac()
    valid = (focal_n >= MIN_FOCAL) and (ant_n > 0) and (not exploded)
    s = selection_coefficient_freq(f0, f1, window) if (f0 is not None and f1 is not None and valid) else float("nan")
    return s


def cell_A(focal, resident, antagonist, attack_cost, seeds, **kw):
    svals = [gradient_probe(focal, resident, antagonist, attack_cost, s, **kw) for s in seeds]
    pairs = [(s, 0.0) for s in svals]
    wins, n_valid = count_wins(pairs, eps=0.0)
    valid_s = [s for s in svals if not math.isnan(s)]
    win_t, lose_t = default_thresholds(n_valid) if n_valid > 0 else (0, 0)
    verdict = "NO_DATA" if n_valid == 0 else ("POS" if wins >= win_t else ("NEG" if wins <= lose_t else "AMBIG"))
    return dict(wins=wins, n_valid=n_valid, mean_s=(float(np.mean(valid_s)) if valid_s else float("nan")), verdict=verdict)


# --------------------------------------------------------------------------------------------
# Part B — co-evolving both-traits arms race across the attack_cost sweep (mirrors Exp-259 part_B)
# --------------------------------------------------------------------------------------------
def cfg_B(attack_cost, horizon=HORIZON_B, mutation_rate=0.15):
    return PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05,
        async_mode="rotating", async_amplitude=0.4, async_period=50.0,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=mutation_rate, mutation_sd=0.06,
        escape_cost=0.15, escape_baseline=1.0, attack_cost=attack_cost, attack_baseline=1.0,
        prey_escape=1.0, pred_attack=1.0, freeze_prey_trait=False, freeze_predator_trait=False,
        trait_min=0.0, trait_max=4.0,   # Exp-259's actual value (its base_cfg did not set trait_max)
        horizon=horizon, n_prey0_per_patch=40, n_pred0_per_patch=8)


def _alive(sim):
    return any(p.prey for p in sim.patches) and any(p.predators for p in sim.patches)


def expected_birthp_clamp(sim, cfg):
    """DETERMINISTIC diagnostic (no rng, no substrate change): EXPECTED predator birth prob
    per predator for the current state = pred_birth_per_capture*assimilation*E[captures_j].
    Returns (mean expected birth_p_pred, fraction of predators whose E[bpp] >= 1.0 clamp).
    If the clamp fraction is ~0, the attack cost is NOT merely shaving a saturated value."""
    bpps = []
    for patch in sim.patches:
        N_prey = len(patch.prey)
        preds = patch.predators
        if not preds:
            continue
        sat = 1.0 / (1.0 + cfg.attack_a * cfg.handling_h * N_prey) if N_prey > 0 else 1.0
        access = cfg.refuge_predator_access if (cfg.refuge_mode == "per_patch" and patch.is_refuge) else 1.0
        exp_cap = [0.0] * len(preds)
        for p in patch.prey:
            contribs = []
            tot = 0.0
            for q in preds:
                c = cfg.attack_a * sat * (1.0 / (1.0 + cfg.escape_k * max(0.0, p.trait - q.trait)))
                contribs.append(c)
                tot += c
            if tot <= 0.0:
                continue
            kp = (1.0 - math.exp(-tot)) * access
            for j, c in enumerate(contribs):
                exp_cap[j] += kp * (c / tot)
        for j in range(len(preds)):
            bpps.append(cfg.pred_birth_per_capture * cfg.assimilation * exp_cap[j])
    if not bpps:
        return float("nan"), float("nan")
    return float(np.mean(bpps)), float(np.mean([1.0 if b >= 1.0 else 0.0 for b in bpps]))


def armsrace(attack_cost, seed, horizon=HORIZON_B, checks=(500, 1000, 2000, 3000)):
    cfg = cfg_B(attack_cost, horizon=horizon)
    sim = PatchMosaicSim(cfg, seed)
    esc_at, atk_at, esc_sd, atk_sd = {}, {}, {}, {}
    clamp_at = {}
    while _alive(sim) and sim.t < horizon:
        sim.step()
        if sim.t in checks:
            prey = [c.trait for p in sim.patches for c in p.prey]
            pred = [c.trait for p in sim.patches for c in p.predators]
            esc_at[sim.t] = float(np.mean(prey)) if prey else None
            atk_at[sim.t] = float(np.mean(pred)) if pred else None
            esc_sd[sim.t] = float(np.std(prey)) if prey else None
            atk_sd[sim.t] = float(np.std(pred)) if pred else None
            if sim.t == checks[-1]:
                clamp_at[sim.t] = expected_birthp_clamp(sim, cfg)
    n_prey = sum(len(p.prey) for p in sim.patches)
    n_pred = sum(len(p.predators) for p in sim.patches)
    extinct = not _alive(sim)
    valid = is_population_valid(min(n_prey, n_pred), extinct, n_prey + n_pred > cfg.pop_cap, MIN_FOCAL)
    return dict(t_end=sim.t, esc_at=esc_at, atk_at=atk_at, esc_sd=esc_sd, atk_sd=atk_sd,
                extinct=extinct, n_prey=n_prey, n_pred=n_pred, valid=valid,
                clamp=clamp_at.get(checks[-1], (float("nan"), float("nan"))))


def mean_at(rs, key, t):
    vals = [r[key].get(t) for r in rs if r[key].get(t) is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    smoke = "--smoke" in sys.argv
    seeds_a = SEEDS_A[:2] if smoke else SEEDS_A
    seeds_b = SEEDS_B[:2] if smoke else SEEDS_B
    horizon_b = 1000 if smoke else HORIZON_B
    ac_sweep = [0.0, 0.15] if smoke else [0.0, 0.05, 0.10, 0.15, 0.25, 0.40]
    A_ladder = (1.0, 2.0, 3.0) if smoke else (1.0, 1.5, 2.0, 2.5, 3.0)
    ac_A = [0.0, 0.15] if smoke else [0.0, 0.15, 0.25]

    L = []
    L.append("=" * 104)
    L.append("Exp 261 — COSTED PREDATOR ATTACK: does it symmetrize the patch-mosaic Red Queen? RAW — controller judges.")
    L.append(f"single causal bit = attack_cost (gated, mirror of escape_cost). Part A seeds {seeds_a}; Part B seeds {seeds_b}.")
    L.append("=" * 104)

    # ---- Byte-identity isolation control (binding): attack_cost=0 must reproduce the Exp-259 config hash ----
    g = PatchMosaicSim(PatchMosaicConfig(
        n_patches=8, topology="ring", attack_a=0.05, K_pred_local=40.0, pred_self_limit_hmax=0.05,
        migration_rate_prey=0.05, migration_rate_pred=0.05, async_mode="rotating", async_amplitude=0.4,
        refuge_mode="per_patch", refuge_predator_access=0.30, refuge_fraction=0.25,
        enable_trait_evolution=True, mutation_rate=0.15, mutation_sd=0.06, escape_cost=0.15,
        prey_escape=1.0, pred_attack=1.0, horizon=300, n_prey0_per_patch=40, n_pred0_per_patch=8), 400).run()["events_hash"]
    L.append(f"[CONTROL] byte-identity isolation: attack_cost=0 Exp-259-config hash (h=300,seed=400) = {g[:16]}... "
             f"{'== Exp-259 golden OK (single bit isolated)' if g == EXP259_GOLDEN else '*** DIFFERS — BIT NOT ISOLATED, NO_VERDICT ***'}")
    L.append("")

    # ---- PART A: predator-attack local gradient vs FAST frozen prey (escape=3.0), attack_cost swept ----
    L.append("(A) PREDATOR-ATTACK local gradient vs FAST frozen prey (escape=3.0). 7/8-strict; cost-null = the ac=0 row.")
    header = f"{'attack_cost':>11} | " + " | ".join(f"A={A:g}" for A in A_ladder)
    L.append(header)
    L.append("-" * len(header))
    partA = {}
    for ac in ac_A:
        row = []
        for A in A_ladder:
            c = cell_A("pred", A, antagonist=3.0, attack_cost=ac, seeds=seeds_a)
            partA[(ac, A)] = c
            row.append(f"{c['wins']}/{c['n_valid']}{c['verdict'][:3]}({c['mean_s']:+.3f})")
        L.append(f"{ac:>11.2f} | " + " | ".join(row))
    L.append("")
    # drift-null (attack inert: vs SATURATED frozen prey escape=1.0) + prey-ESS sanity
    dn0 = cell_A("pred", 1.0, antagonist=1.0, attack_cost=0.0, seeds=seeds_a)
    dn25 = cell_A("pred", 1.0, antagonist=1.0, attack_cost=(0.25 if not smoke else 0.15), seeds=seeds_a)
    pe1 = cell_A("prey", 1.0, antagonist=1.0, attack_cost=0.0, seeds=seeds_a)
    pe3 = cell_A("prey", 3.0, antagonist=1.0, attack_cost=0.0, seeds=seeds_a)
    L.append(f"[CTRL] predator DRIFT-NULL (vs saturated prey escape=1.0, attack inert): ac=0 {dn0['wins']}/{dn0['n_valid']} {dn0['verdict']} (s{dn0['mean_s']:+.4f}); ac>0 {dn25['wins']}/{dn25['n_valid']} {dn25['verdict']} (s{dn25['mean_s']:+.4f}) — must NOT be POS")
    L.append(f"[CTRL] prey-ESS sanity (predator frozen, escape_cost=0.15): E=1.0 {pe1['verdict']}(s{pe1['mean_s']:+.4f}), E=3.0 {pe3['verdict']}(s{pe3['mean_s']:+.4f}) — expect POS/AMBIG low, NEG by E=3")
    L.append("")

    # ---- PART B: co-evolving arms race across attack_cost ----
    L.append("(B) CO-EVOLVING arms race (both traits mutable), trait means; D = atk@3k - esc@3k. horizon=" + str(horizon_b))
    # Binding baseline-reproduction: ac=0 on Exp-259's OWN seeds 400-409 (byte-identical to Exp-259 co-evolve)
    if not smoke:
        bse = list(range(400, 410))
        rb = [armsrace(0.0, s, horizon=HORIZON_B) for s in bse]
        rbv = [r for r in rb if r["valid"]] or rb
        be3, ba3 = mean_at(rbv, "esc_at", 3000), mean_at(rbv, "atk_at", 3000)
        bext = sum(r["extinct"] for r in rb)
        L.append(f"[BASELINE ac=0 on Exp-259 seeds 400-409]: esc@3k={be3:.3f} atk@3k={ba3:.3f} D={ba3-be3:+.3f} extinct={bext}/{len(rb)}  "
                 f"(reproduces Exp-259 predator-dominance iff D>=+0.40, esc<~1.0, atk>~1.0)")
    L.append(f"{'attack_cost':>11} | {'esc@3k':>7} {'atk@3k':>7} | {'D':>7} | {'esc_sd':>6} {'atk_sd':>6} | {'pen%':>5} | {'clampf':>6} | {'ext':>4} | label")
    L.append("-" * 104)
    partB = {}
    for ac in ac_sweep:
        rs = [armsrace(ac, s, horizon=horizon_b) for s in seeds_b]
        rsv = [r for r in rs if r["valid"]]
        use = rsv if rsv else rs
        tcheck = 1000 if smoke else 3000
        e3 = mean_at(use, "esc_at", tcheck)
        a3 = mean_at(use, "atk_at", tcheck)
        esd = mean_at(use, "esc_sd", tcheck)
        asd = mean_at(use, "atk_sd", tcheck)
        D = a3 - e3
        ext = sum(r["extinct"] for r in rs)
        pen = ac * max(0.0, a3 - 1.0)  # realized fecundity penalty at the operating attack mean
        clampf = float(np.nanmean([r["clamp"][1] for r in use]))
        partB[ac] = dict(e3=e3, a3=a3, D=D, esd=esd, asd=asd, ext=ext, n_valid=len(rsv), pen=pen, clampf=clampf)
        # provisional label (controller re-adjudicates)
        if ext > 2 or not rsv:
            lab = "COLLAPSE"
        elif D >= 0.40:
            lab = "PRED_DOM"
        elif D <= -0.40:
            lab = "PREY_DOM"
        elif abs(D) <= 0.30:
            lab = "SYMM?"
        else:
            lab = "mid"
        L.append(f"{ac:>11.2f} | {e3:>7.3f} {a3:>7.3f} | {D:>+7.3f} | {esd:>6.3f} {asd:>6.3f} | {pen*100:>4.1f}% | {clampf:>6.3f} | {ext:>2}/{len(rs)} | {lab}")
    L.append("")

    # ---- Predeclared readout (controller adjudicates raw rows) ----
    L.append("PREDECLARED READOUT (controller adjudicates against the raw rows + the two-sided plateau criterion):")
    Ds = [(ac, partB[ac]["D"]) for ac in ac_sweep]
    L.append(f"  Part B gap D by attack_cost: " + ", ".join(f"ac{ac:g}={d:+.3f}" for ac, d in Ds))
    mono = all(Ds[i + 1][1] <= Ds[i][1] + 0.05 for i in range(len(Ds) - 1))
    L.append(f"    -> monotone brake (D non-increasing in attack_cost, tol 0.05): {mono}")
    L.append(f"  baseline ac=0: D={partB[ac_sweep[0]]['D']:+.3f} (PREDATOR_DOMINANCE expected if D>=+0.40); byte-identity isolation: {'OK' if g==EXP259_GOLDEN else 'FAIL'}")
    flipped = [ac for ac in ac_A if ac > 0 and any(partA[(ac, A)]["verdict"] == "NEG" for A in A_ladder)]
    L.append(f"  Part A interior predator ESS (first-NEG up the ladder) appears at attack_cost: {flipped if flipped else 'none (gradient stays POS/AMBIG — falsifier 3)'}")
    L.append(f"  predator drift-null POS? ac=0:{dn0['verdict']=='POS'} ac>0:{dn25['verdict']=='POS'} (POS => NO_VERDICT, falsifier 4)")

    out = "\n".join(L)
    print(out)
    outdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "outputs")
    os.makedirs(outdir, exist_ok=True)
    fname = "exp261_costed_predator_attack_smoke.txt" if smoke else "exp261_costed_predator_attack.txt"
    with open(os.path.join(outdir, fname), "w") as f:
        f.write(out + "\n")
    print(f"\n[written to experiments/outputs/{fname}]")


if __name__ == "__main__":
    main()
