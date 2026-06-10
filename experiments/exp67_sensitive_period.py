"""Exp 67 — the sensitive-period boundary: where does social persuadability end?

Follow-up predeclared in Exp 66. Established so far: the same grounded cue channel that
cannot move an adult (Exp 65 NEGATIVE, ~9,200-count ledger) installs the emitter's favorite
into 800-step newborns (Exp 66, 3/4). This experiment maps the boundary: receiver age
(settle steps) is swept 0/400/800/1600/3200/6400 x 4 seeds, same emitter, same channel,
same 2000-step dyad dose. The 800-step bin with the same birth seeds (300+s) and the same
dyad RNGs is an exact internal replication of Exp 66's four runs.

Predeclared predictions:
  P1 (sensitive period exists): pooled install fraction over young bins (settle <= 800)
     is STRICTLY greater than pooled install fraction over old bins (settle >= 3200),
     AND the 6400 bin has 0 installs.
  P2 (monotonicity, ties allowed): per-bin install fractions are non-increasing with age.
  P3 (count-model adequacy >= 75%): the predeclared proportional-growth criterion
        install  iff  gap_pre * (mass_end_sev / mass_pre) < measured_cue_mass
     (gap_pre = pre-dyad count gap between the receiver's top color and the cue color;
     mass_end_sev = severed-ledger total at dyad end; age-0 runs with mass_pre == 0 are
     predicted INSTALL) matches the observed install outcome in >= 75% of valid runs.
     Provenance: this formula retrodicts Exp 66 at 4/4, including the seed-0 resister
     that a naive static pre-dyad criterion gets wrong. Declared as model adequacy (the
     dose term is the same-run measured cue mass), not a forecast.
Falsifiers:
  F1 = any old bin (>= 3200) has an install fraction >= the youngest valid bin's fraction
     while the youngest bin installs at all -> NO sensitive period; the mass-inertia
     account fails at scale; log NEGATIVE and post a CONSULT.
  F2 = P3 accuracy < 50% -> the count-arithmetic model of social influence is wrong.
  P3 accuracy in [50%, 75%) with P1+P2 passing -> MIXED (boundary real, model inadequate).
Predicted (stated before running): boundary between the 1600 and 3200 bins; P3 ~80%.
Install definition (as Exp 66): fav_on == CUE_COLOR AND fav_sev != CUE_COLOR at dyad end.
Gates: G1 per run: pre-dyad favorite != CUE_COLOR (zero-mass age-0 receivers have
favorite()==0 != 1 and count as valid); >= 3 valid runs per bin else the bin is INVALID;
>= 5 of 6 bins valid else the experiment is INVALID. G2: >= 50 proximity events per run.
Provided priors declared: everything Exp 65/66 declared (channel wiring, speciation world,
dyad world = mirro's grid, random walks, dual-ledger exact severed twin) plus the age sweep
itself. Self-formed: the emitter's cue content; each receiver's pre-dyad values from its
own settled steps. The committed spines never live and are never saved; mirro is forked
once (the emitter), vela is untouched.
"""
from __future__ import annotations

import copy
import sys

import numpy as np
from pathlib import Path

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPECIATION_STEPS = 2000
DYAD_STEPS = 2000
AGES = [0, 400, 800, 1600, 3200, 6400]
SEEDS = [0, 1, 2, 3]
PROX = 1  # Manhattan distance gate
MIN_EVENTS = 50
P3_BAR = 0.75
MIRRO_DIR = Path("creature/state/mirro")

# ---------------------------------------------------------------------------
# Load committed spine (read-only — NEVER call .live() or .save() on mirro)
# ---------------------------------------------------------------------------

print("Exp 67 — the sensitive-period boundary: where does social persuadability end?")
print()

mirro = Creature.load(MIRRO_DIR)

vc_m = mirro.value_counts
tot_m = vc_m.sum()

print(f"mirro: name={mirro.name!r} age={mirro.age_steps} "
      f"hash={mirro._state_hash()[:12]} favorite={mirro.favorite()} "
      f"shares={np.round(vc_m / tot_m, 4)}")
print()

# ---------------------------------------------------------------------------
# Phase 1: emitter speciation (EXACTLY as exp66 — fork name "exp67-emitter")
# Fork mirro; raise 2000 steps in an all-color-1 world.
# ---------------------------------------------------------------------------

print("--- PHASE 1: EMITTER SPECIATION ---")

emitter = mirro.fork("exp67-emitter")

spec_world = World(
    rows=5,
    cols=5,
    cmap=[1] * 25,
    n_colors=mirro.world.n_colors,
)
emitter.world = spec_world
emitter.true_pos = 0

vc_before = emitter.value_counts.copy()
tot_before = vc_before.sum()
print(f"emitter before speciation: favorite={emitter.favorite()} "
      f"value_shares={np.round(vc_before / tot_before, 4)}")

# live() on the fork only — fork is unbound (_state_dir=None), no biography written to disk
emitter.live(SPECIATION_STEPS)

vc_after = emitter.value_counts.copy()
tot_after = vc_after.sum()
print(f"emitter after  speciation: favorite={emitter.favorite()} "
      f"value_shares={np.round(vc_after / tot_after, 4)}")
print()

# Hard check: phase 1 must reproduce favorite=1 exactly
if emitter.favorite() != 1:
    print(f"PHASE-1 REPRODUCTION FAIL — emitter favorite != 1 (got {emitter.favorite()})")
    sys.exit(1)

print(f"PHASE-1 REPRODUCTION OK: emitter favorite == 1")
print()

CUE_COLOR = emitter.favorite()  # expected: 1
print(f"Cue color (emitter favorite): {CUE_COLOR}")
print()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn):
    """Run predicate_fn(); record (name, pass, detail). Exceptions count as FAIL."""
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


# ---------------------------------------------------------------------------
# Phase 2: age sweep — receiver construction per (age, seed)
# ---------------------------------------------------------------------------

print("--- PHASE 2: AGE SWEEP — JUNIOR SETTLING + DYAD RUNS ---")
print()

# run_results keyed by (age, seed)
run_results: dict[tuple[int, int], dict] = {}

# Compact per-run table header
print(
    f"{'age':>6}  {'seed':>4}  {'valid?':>6}  {'events':>6}  "
    f"{'mass_pre':>8}  {'gap_pre':>7}  {'cue_mass':>8}  "
    f"{'proj_gap':>8}  {'diverge':>7}  "
    f"{'fav_sev':>7}  {'fav_on':>6}  {'inst':>4}  {'pred':>4}  {'match':>5}"
)
print("-" * 130)

for age in AGES:
    for seed in SEEDS:
        # --- Receiver construction ---
        # birth seed depends ONLY on seed, NOT on age
        junior = Creature.birth(
            f"exp67-junior-a{age}-s{seed}",
            world=mirro.world,
            seed=300 + seed,
        )
        junior.true_pos = 12  # center start (declared)

        # Settling: junior self-forms initial values for `age` steps
        if age > 0:
            junior.live(age)

        # Pre-dyad record
        jr_vc_pre = junior.value_counts.copy()
        mass_pre = float(jr_vc_pre.sum())
        jr_fav_pre = junior.favorite()
        jr_shares_pre = np.round(jr_vc_pre / mass_pre, 4) if mass_pre > 0 else jr_vc_pre.copy()
        jr_map_acc = junior.map_accuracy()

        # gap_pre: count gap between receiver's top color and the cue color
        # 0.0 when mass_pre == 0 (age-0 case)
        if mass_pre > 0:
            gap_pre = float(jr_vc_pre[int(np.argmax(jr_vc_pre))] - jr_vc_pre[CUE_COLOR])
        else:
            gap_pre = 0.0

        print(f"\n  [age={age} seed={seed}] pre-dyad: favorite={jr_fav_pre} "
              f"shares={jr_shares_pre} mass={mass_pre:.1f} map_acc={jr_map_acc:.3f} "
              f"gap_pre={gap_pre:.2f}")

        # G1 per run: pre-dyad favorite != CUE_COLOR
        # (age-0 receivers have favorite()==0 != 1 and count as valid)
        if jr_fav_pre == CUE_COLOR:
            reason = f"pre-dyad favorite ({jr_fav_pre}) == CUE_COLOR ({CUE_COLOR})"
            run_results[(age, seed)] = {
                "valid": False,
                "g1_reason": reason,
                "age": age,
                "seed": seed,
            }
            print(
                f"  [age={age} seed={seed}] G1 INVALID — {reason}; run excluded"
            )
            print(
                f"  {age:>6}  {seed:>4}  {'INVALID':>6}  {'—':>6}  "
                f"{'—':>8}  {'—':>7}  {'—':>8}  {'—':>8}  {'—':>7}  "
                f"{'—':>7}  {'—':>6}  {'—':>4}  {'—':>4}  {'—':>5}"
            )
            continue

        # Dyad setup
        n_colors = mirro.world.n_colors
        world = mirro.world
        n_cells = world.n_cells
        B = world.transition_matrix()  # (n_cells, n_cells, 4)

        # Deep-copy emitter for this run's dyad
        em = copy.deepcopy(emitter)
        em.world = mirro.world  # run dyad in mirro's world

        # RNGs: depend only on seed — so the 800 bin exactly replicates Exp 66
        rng_em = np.random.default_rng(200000 + 1000 * seed + 0)
        rng_junior = np.random.default_rng(200000 + 1000 * seed + 1)

        # Dual ledger: junior.value_counts is the SEVERED ledger (natural only)
        # cue_extra accumulates ONLY the channel increments
        cue_extra = np.zeros(n_colors)

        events = 0
        gate_weight_sum = 0.0
        cue_color_counts: dict[int, int] = {}

        for step in range(DYAD_STEPS):
            # Read simultaneous start-of-step positions
            pos_em = em.true_pos
            pos_junior = junior.true_pos

            # --- Emitter natural update (replicates live() exactly) ---
            A_hat_em = em._A_hat()
            obs_em = int(world.cmap[pos_em])
            likelihood_em = A_hat_em[obs_em, :]
            qs_upd_em = likelihood_em * em.qs
            denom_em = qs_upd_em.sum()
            if denom_em > 0:
                qs_upd_em = qs_upd_em / denom_em
            else:
                qs_upd_em = np.ones(n_cells) / n_cells
            em.pA[obs_em, :] += qs_upd_em
            map_cell_em = int(np.argmax(qs_upd_em))
            h_em = -np.sum(A_hat_em[:, map_cell_em] * np.log(A_hat_em[:, map_cell_em] + 1e-12))
            w_em = np.exp(-h_em)
            em.value_counts[obs_em] += w_em
            action_em = int(rng_em.integers(0, 4))
            new_pos_em = world.move(pos_em, action_em)
            em.qs = B[:, :, action_em] @ qs_upd_em

            # --- Junior natural update (replicates live() exactly) ---
            A_hat_junior = junior._A_hat()
            obs_junior = int(world.cmap[pos_junior])
            likelihood_junior = A_hat_junior[obs_junior, :]
            qs_upd_junior = likelihood_junior * junior.qs
            denom_junior = qs_upd_junior.sum()
            if denom_junior > 0:
                qs_upd_junior = qs_upd_junior / denom_junior
            else:
                qs_upd_junior = np.ones(n_cells) / n_cells
            junior.pA[obs_junior, :] += qs_upd_junior
            map_cell_junior = int(np.argmax(qs_upd_junior))
            A_hat_junior_col = A_hat_junior[:, map_cell_junior]
            h_junior = -np.sum(A_hat_junior_col * np.log(A_hat_junior_col + 1e-12))
            w_junior = np.exp(-h_junior)
            junior.value_counts[obs_junior] += w_junior  # natural (severed) ledger
            action_junior = int(rng_junior.integers(0, 4))
            new_pos_junior = world.move(pos_junior, action_junior)
            junior.qs = B[:, :, action_junior] @ qs_upd_junior

            # --- Channel (after both natural updates, using start-of-step positions) ---
            r_em, c_em = divmod(pos_em, world.cols)
            r_jr, c_jr = divmod(pos_junior, world.cols)
            manhattan = abs(r_em - r_jr) + abs(c_em - c_jr)
            if manhattan <= PROX:
                cue = int(np.argmax(em.value_counts))  # emitter's current favorite
                cue_extra[cue] += w_junior              # junior's predictability weight
                events += 1
                gate_weight_sum += w_junior
                cue_color_counts[cue] = cue_color_counts.get(cue, 0) + 1

            # --- Apply moves simultaneously ---
            em.true_pos = new_pos_em
            junior.true_pos = new_pos_junior

        # --- End-of-run ledger computations ---
        vc_sev = junior.value_counts.copy()       # severed ledger
        vc_on = junior.value_counts + cue_extra   # on-ledger (severed + channel increments)

        tot_sev = float(vc_sev.sum())
        tot_on = float(vc_on.sum())

        share_sev_cue = float(vc_sev[CUE_COLOR] / tot_sev) if tot_sev > 0 else 0.0
        share_on_cue = float(vc_on[CUE_COLOR] / tot_on) if tot_on > 0 else 0.0
        divergence = share_on_cue - share_sev_cue

        fav_sev = int(np.argmax(vc_sev))
        fav_on = int(np.argmax(vc_on))

        mean_gate_w = gate_weight_sum / events if events > 0 else 0.0
        modal_cue = max(cue_color_counts, key=cue_color_counts.get) if cue_color_counts else -1

        cue_mass = float(cue_extra.sum())
        mass_end_sev = tot_sev

        # P3 per-run prediction
        if mass_pre == 0:
            predicted_install = True
        else:
            projected_gap = gap_pre * (mass_end_sev / mass_pre)
            predicted_install = projected_gap < cue_mass

        # For display: projected_gap (gap_pre * mass_end_sev / mass_pre, or 0 when mass_pre==0)
        if mass_pre > 0:
            proj_gap_display = gap_pre * (mass_end_sev / mass_pre)
        else:
            proj_gap_display = 0.0

        install = (fav_on == CUE_COLOR) and (fav_sev != CUE_COLOR)
        model_match = (predicted_install == install)

        run_results[(age, seed)] = {
            "valid": True,
            "age": age,
            "seed": seed,
            "events": events,
            "mean_gate_w": mean_gate_w,
            "mass_pre": mass_pre,
            "gap_pre": gap_pre,
            "cue_mass": cue_mass,
            "mass_end_sev": mass_end_sev,
            "proj_gap": proj_gap_display,
            "share_sev_cue": share_sev_cue,
            "share_on_cue": share_on_cue,
            "divergence": divergence,
            "fav_sev": fav_sev,
            "fav_on": fav_on,
            "install": install,
            "predicted_install": predicted_install,
            "model_match": model_match,
            "modal_cue": modal_cue,
            "cue_color_counts": cue_color_counts,
            "vc_sev": vc_sev,
            "vc_on": vc_on,
            "jr_fav_pre": jr_fav_pre,
        }

        inst_str = "Y" if install else "N"
        pred_str = "Y" if predicted_install else "N"
        match_str = "Y" if model_match else "N"

        print(
            f"  {age:>6}  {seed:>4}  {'valid':>6}  {events:>6}  "
            f"{mass_pre:>8.1f}  {gap_pre:>7.2f}  {cue_mass:>8.3f}  "
            f"{proj_gap_display:>8.2f}  {divergence:>7.4f}  "
            f"{fav_sev:>7}  {fav_on:>6}  {inst_str:>4}  {pred_str:>4}  {match_str:>5}"
        )

print()

# ---------------------------------------------------------------------------
# Gates section
# ---------------------------------------------------------------------------

print("--- VALIDITY GATES ---")
print()

# G1 per-bin valid counts
print("G1: per-bin valid-run counts (need >= 3 per bin; >= 5 of 6 bins valid):")
bin_valid_counts: dict[int, int] = {}
bin_invalid_bins: list[int] = []

for age in AGES:
    valid_in_bin = sum(
        1 for seed in SEEDS if run_results.get((age, seed), {}).get("valid", False)
    )
    bin_valid_counts[age] = valid_in_bin
    bin_ok = valid_in_bin >= 3
    status = "OK" if bin_ok else "INVALID (< 3 valid runs)"
    print(f"  age={age:>5}: {valid_in_bin}/{len(SEEDS)} valid  [{status}]")
    if not bin_ok:
        bin_invalid_bins.append(age)

n_valid_bins = len(AGES) - len(bin_invalid_bins)
if n_valid_bins < 5:
    print()
    print(f"G1 FAIL — RUN INVALID ({n_valid_bins} valid bins; need >= 5 of 6)")
    sys.exit(1)

print(f"  G1 PASS: {n_valid_bins} of {len(AGES)} bins valid (need >= 5)")
print()

# G2: >= MIN_EVENTS proximity events per valid run
print(f"G2: >= {MIN_EVENTS} proximity events per valid run:")
g2_ok = True
for age in AGES:
    for seed in SEEDS:
        r = run_results.get((age, seed), {})
        if not r.get("valid", False):
            continue
        ev = r["events"]
        ok = ev >= MIN_EVENTS
        if not ok:
            g2_ok = False
        status = "OK" if ok else "FAIL"
        print(f"  age={age:>5} seed={seed}: events={ev}  [{status}]")

if not g2_ok:
    print()
    print("G2 FAIL — RUN INVALID (too few proximity events; redesign before verdict)")
    sys.exit(1)

print(f"  G2 PASS: all valid runs >= {MIN_EVENTS} events")
print()

# ---------------------------------------------------------------------------
# Per-bin summary table
# ---------------------------------------------------------------------------

print("--- PER-BIN SUMMARY ---")
print(f"{'age':>6}  {'n_valid':>7}  {'installs':>8}  {'install_frac':>12}")
print("-" * 44)

bin_fracs: dict[int, float] = {}
bin_installs: dict[int, int] = {}
bin_n_valid: dict[int, int] = {}

for age in AGES:
    n_v = bin_valid_counts[age]
    n_inst = sum(
        1 for seed in SEEDS
        if run_results.get((age, seed), {}).get("valid", False)
        and run_results[(age, seed)]["install"]
    )
    frac = n_inst / n_v if n_v > 0 else float("nan")
    bin_fracs[age] = frac
    bin_installs[age] = n_inst
    bin_n_valid[age] = n_v
    invalid_note = " (INVALID BIN)" if age in bin_invalid_bins else ""
    print(f"  {age:>5}  {n_v:>7}  {n_inst:>8}  {frac:>12.3f}{invalid_note}")

print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

# Valid bins only (for P1/P2 pooling, exclude invalid bins)
valid_ages = [a for a in AGES if a not in bin_invalid_bins]
young_ages = [a for a in [0, 400, 800] if a in valid_ages]
old_ages = [a for a in [3200, 6400] if a in valid_ages]


def _p1():
    # Pool valid runs in young bins
    young_installs = sum(bin_installs[a] for a in young_ages)
    young_n = sum(bin_n_valid[a] for a in young_ages)
    old_installs = sum(bin_installs[a] for a in old_ages)
    old_n = sum(bin_n_valid[a] for a in old_ages)

    young_frac = young_installs / young_n if young_n > 0 else 0.0
    old_frac = old_installs / old_n if old_n > 0 else 0.0

    installs_6400 = bin_installs.get(6400, 0) if 6400 in valid_ages else 0

    passes = (young_frac > old_frac) and (installs_6400 == 0)
    detail = (
        f"young_frac={young_frac:.3f} (bins={young_ages}, installs={young_installs}/{young_n})  "
        f"old_frac={old_frac:.3f} (bins={old_ages}, installs={old_installs}/{old_n})  "
        f"6400_installs={installs_6400}  "
        f"({'young>old AND 6400==0' if passes else 'FAIL'})"
    )
    return passes, detail


def _p2():
    # Monotonicity (ties allowed): fractions non-increasing with age, valid bins only
    fracs_ordered = [(a, bin_fracs[a]) for a in valid_ages]
    violations = []
    for i in range(len(fracs_ordered) - 1):
        a0, f0 = fracs_ordered[i]
        a1, f1 = fracs_ordered[i + 1]
        if f1 > f0:
            violations.append(f"age{a0}({f0:.3f})->age{a1}({f1:.3f}) INCREASES")
    passes = len(violations) == 0
    pairs_str = "  ".join(f"age{a}:{f:.3f}" for a, f in fracs_ordered)
    detail = f"fracs=[{pairs_str}]"
    if violations:
        detail += f"  VIOLATIONS: {'; '.join(violations)}"
    return passes, detail


def _p3():
    all_valid_runs = [
        run_results[(age, seed)]
        for age in AGES for seed in SEEDS
        if run_results.get((age, seed), {}).get("valid", False)
    ]
    n_total = len(all_valid_runs)
    n_match = sum(1 for r in all_valid_runs if r["model_match"])
    accuracy = n_match / n_total if n_total > 0 else 0.0
    mismatches = [
        f"age{r['age']}-s{r['seed']}:inst={'Y' if r['install'] else 'N'}/"
        f"pred={'Y' if r['predicted_install'] else 'N'}"
        for r in all_valid_runs if not r["model_match"]
    ]
    passes = accuracy >= P3_BAR
    detail = (
        f"accuracy={accuracy:.1%} ({n_match}/{n_total})  "
        f"threshold={P3_BAR:.0%}  "
        f"mismatches=[{', '.join(mismatches) if mismatches else 'none'}]"
    )
    return passes, detail


check("P1-sensitive-period-young>old-AND-6400==0", _p1)
check("P2-monotonicity-non-increasing-with-age", _p2)
check("P3-count-model-accuracy>=75pct", _p3)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
print()

failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"{verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print("--- FALSIFIER MAP ---")

p1_passed = checks[0][1]
p2_passed = checks[1][1]
p3_passed = checks[2][1]

# Compute P3 accuracy for falsifier thresholds
all_valid_runs_flat = [
    run_results[(age, seed)]
    for age in AGES for seed in SEEDS
    if run_results.get((age, seed), {}).get("valid", False)
]
p3_acc = (
    sum(1 for r in all_valid_runs_flat if r["model_match"]) / len(all_valid_runs_flat)
    if all_valid_runs_flat else 0.0
)

# F1: any old bin (>= 3200) has fraction >= youngest valid bin fraction
#     AND youngest bin installs at all
youngest_valid_age = valid_ages[0] if valid_ages else None
youngest_frac = bin_fracs.get(youngest_valid_age, 0.0) if youngest_valid_age is not None else 0.0
youngest_installs = bin_installs.get(youngest_valid_age, 0) if youngest_valid_age is not None else 0

f1_fired = False
if youngest_installs > 0:
    for old_age in [a for a in [3200, 6400] if a in valid_ages]:
        if bin_fracs[old_age] >= youngest_frac:
            f1_fired = True
            print(
                f"F1 FIRED: no sensitive period — mass-inertia account fails at scale; "
                f"NEGATIVE; CONSULT required "
                f"(age={old_age} frac={bin_fracs[old_age]:.3f} >= "
                f"youngest bin age={youngest_valid_age} frac={youngest_frac:.3f})"
            )

# F2: P3 accuracy < 50%
f2_fired = p3_acc < 0.50
if f2_fired:
    print(f"F2 FIRED: count-arithmetic model wrong (accuracy {p3_acc:.0%})")

# MIXED: P1+P2 pass, P3 in [50%, 75%)
if p1_passed and p2_passed and not p3_passed and 0.50 <= p3_acc < P3_BAR:
    print(f"P3 shortfall: MIXED — boundary real, model inadequate (accuracy {p3_acc:.0%})")

if p1_passed and p2_passed and p3_passed and not f1_fired:
    print("ALL PASS: sensitive period mapped; count model adequate")

if not f1_fired and not f2_fired:
    print("  No falsifiers fired.")

print()

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

# 800-bin vs Exp 66 replication check
print("--- DIAGNOSTICS: 800-BIN VS EXP 66 REPLICATION ---")
print("(expect exact match to Exp 66 seeds 0/1/2/3; seed 3 G1-invalid there too)")
print(f"{'seed':>4}  {'divergence':>10}  {'fav_sev':>7}  {'fav_on':>6}  {'valid?':>6}")
print("-" * 50)
for seed in SEEDS:
    r = run_results.get((800, seed), {})
    if r.get("valid", False):
        print(
            f"  {seed:>2}   {r['divergence']:>9.4f}  {r['fav_sev']:>6}  "
            f"{r['fav_on']:>5}  {'valid':>6}"
        )
    else:
        reason = r.get("g1_reason", "unknown")
        print(f"  {seed:>2}   {'—':>9}  {'—':>6}  {'—':>5}  INVALID [{reason}]")
print()

# Cue distribution per valid run
any_non_cue = any(
    cue != CUE_COLOR
    for age in AGES for seed in SEEDS
    if run_results.get((age, seed), {}).get("valid", False)
    for cue in run_results[(age, seed)]["cue_color_counts"]
)

print("--- CUE COLOR DISTRIBUTION ---")
if not any_non_cue:
    print("all emissions color 1")
else:
    for age in AGES:
        for seed in SEEDS:
            r = run_results.get((age, seed), {})
            if not r.get("valid", False):
                continue
            total_ev = r["events"]
            cc = r["cue_color_counts"]
            dist_str = "  ".join(
                f"color{c}:{cnt}({100*cnt/total_ev:.1f}%)" for c, cnt in sorted(cc.items())
            )
            print(f"  age={age:>5} seed={seed}: {dist_str}  (total={total_ev})")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if p1_passed and p2_passed and p3_passed:
    print("EXP67: PASS")
elif p1_passed and p2_passed and not p3_passed and 0.50 <= p3_acc < P3_BAR:
    print("EXP67: MIXED [P3]")
else:
    fail_tags = [name.split("-")[0] for name, passed, _ in checks if not passed]
    if f1_fired:
        fail_tags.insert(0, "F1")
    if f2_fired:
        fail_tags.insert(0, "F2") if "F2" not in fail_tags else None
    print(f"EXP67: FAIL {fail_tags}")
