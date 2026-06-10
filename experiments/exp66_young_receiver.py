"""Exp 66 — the young-receiver test: does value-mass inertia explain Exp 65's negative?

Follow-up predeclared in Exp 65 (NOT a re-run of rung 3 — rung 3's adult-to-adult verdict
stands NEGATIVE). Exp 65's diagnosis: social transmission is VALUE-MASS-LIMITED — the cue
mass (~175 counts) was negligible against the adult receiver's ~9,200-count lifetime
ledger. If that diagnosis is right, the SAME channel at the SAME dose into a LOW-mass
receiver must produce a proportionally larger share shift, and may install the emitter's
favorite outright.

Receiver: a newborn SEPARATE ROOT ("junior", per-seed births) — explicitly the
zero/low-history baseline allowed by loop/directions/social-emergence.md; junior is NOT a
mirro descendant and no clade-membership claim is made. Each junior settles 800 steps in
mirro's world to self-form initial values (~600-900 counts) before the dyad.

Emitter: the same speciated mirro-fork as Exp 65 (fork at age 10700 hash 21ccb619f063,
2000 steps in an all-color-1 world) — deterministic from committed state, so phase 1 must
reproduce favorite=1 exactly (checked).

Gates (instrument validity): G1 per seed: emitter favorite != junior's pre-dyad favorite
(else that seed INVALID and excluded; >=4 valid seeds required, else run INVALID).
G2: >=50 proximity events per valid seed.
Predeclared predictions:
  P1 (inertia scaling, primary): junior divergence >= 3x the adult divergence of the SAME
     seed from Exp 65's committed output (s0:0.0160 s1:0.0151 s2:0.0123 s3:0.0147
     s4:0.0133), in >=4/5 valid seeds. Predicted ratio ~3-5x from the ~5x mass ratio.
  P2 (sign): junior divergence > 0 in 5/5 valid seeds.
  P3 (installation, the functional headline; honestly ~50/50 by pre-run arithmetic —
     cue mass ~150 vs natural top-vs-cue share gap ~165 counts): ON-ledger favorite ==
     cue color AND SEVERED favorite != cue color, in >=3/5 valid seeds.
Falsifiers:
  F1 = P1 fails -> the mass-inertia diagnosis of Exp 65 is WRONG (divergence does not
     scale with inverse mass); log NEGATIVE and post a CONSULT (it contradicts an
     explanatory claim already in the log).
  F2 = P2 fails in a valid seed -> wiring/gating bug; HALT for investigation.
  P3 failing alone (P1+P2 pass) -> MIXED: inertia law confirmed, installation not
     achieved at this dose (a dose question, not a law question).
Provided priors declared: everything Exp 65 declared (emission rule Manhattan<=1 emitting
the emitter's current favorite; reception rule value_counts[cue] += receiver's own exp(-H)
predictability weight; speciation world all-color-1; dyad world = mirro's grid; random
walks; dual-ledger exact severed twin) PLUS junior's birth (Creature.birth, separate root,
per-seed birth seeds 300+seed) and its 800-step settling in mirro's world. Self-formed:
the emitter's cue content (its divergent lived history); junior's pre-dyad values (its own
800 settled steps). The committed spines never live and are never saved; mirro is forked
once (emitter), vela is not touched at all.
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
SETTLE_STEPS = 800
DYAD_STEPS = 2000
SEEDS = [0, 1, 2, 3, 4]
PROX = 1  # Manhattan distance gate
P1_RATIO = 3.0
MIN_EVENTS = 50
MIN_VALID_SEEDS = 4
ADULT_DIV = {0: 0.0160, 1: 0.0151, 2: 0.0123, 3: 0.0147, 4: 0.0133}
MIRRO_DIR = Path("creature/state/mirro")

# ---------------------------------------------------------------------------
# Load committed spine (read-only — NEVER call .live() or .save() on mirro)
# ---------------------------------------------------------------------------

print("Exp 66 — young-receiver test: does value-mass inertia explain Exp 65's negative?")
print()

mirro = Creature.load(MIRRO_DIR)

vc_m = mirro.value_counts
tot_m = vc_m.sum()

print(f"mirro: name={mirro.name!r} age={mirro.age_steps} "
      f"hash={mirro._state_hash()[:12]} favorite={mirro.favorite()} "
      f"value_shares={np.round(vc_m / tot_m, 4)}")
print()

# ---------------------------------------------------------------------------
# Phase 1: emitter speciation (IDENTICAL to Exp 65)
# Fork mirro; raise 2000 steps in an all-color-1 world.
# ---------------------------------------------------------------------------

print("--- PHASE 1: EMITTER SPECIATION ---")

emitter = mirro.fork("exp66-emitter")

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

# Hard check: phase 1 must reproduce favorite=1 exactly (deterministic from Exp 65)
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
# Phase 2: junior settling + dyad stepper across seeds
# ---------------------------------------------------------------------------

print("--- PHASE 2: JUNIOR SETTLING + DYAD RUNS ---")
print()

# Storage
seed_results = {}
valid_seeds = []

print(
    f"{'seed':>4}  {'events':>7}  {'mean_gate_w':>11}  {'jr_mass':>7}  "
    f"{'share_sev[cue]':>14}  {'share_on[cue]':>13}  {'divergence':>10}  "
    f"{'ratio':>6}  {'fav_sev':>7}  {'fav_on':>6}  {'em_fav_end':>10}  {'modal_cue':>9}"
)
print("-" * 120)

for seed in SEEDS:
    # --- Junior birth (separate root, not a mirro descendant) ---
    junior = Creature.birth(f"exp66-junior-s{seed}", world=mirro.world, seed=300 + seed)
    junior.true_pos = 12  # center start (declared)

    # --- Settling: junior self-forms initial values ---
    junior.live(SETTLE_STEPS)  # unbound — no biography file written

    jr_vc_post = junior.value_counts.copy()
    jr_tot_post = jr_vc_post.sum()
    jr_fav_pre = junior.favorite()
    jr_shares_pre = np.round(jr_vc_post / jr_tot_post, 4) if jr_tot_post > 0 else jr_vc_post
    jr_map_acc = junior.map_accuracy()

    print(f"\n  [seed={seed}] junior post-settle: favorite={jr_fav_pre} "
          f"shares={jr_shares_pre} mass={jr_tot_post:.1f} map_acc={jr_map_acc:.3f}")

    # --- G1 per seed: emitter favorite != junior pre-dyad favorite ---
    if emitter.favorite() == jr_fav_pre:
        print(f"  [seed={seed}] G1 INVALID — emitter favorite ({emitter.favorite()}) "
              f"== junior pre-dyad favorite ({jr_fav_pre}); seed excluded")
        seed_results[seed] = {"valid": False, "g1_reason": f"emitter_fav==junior_fav=={jr_fav_pre}"}
        continue

    # Seed is G1-valid
    valid_seeds.append(seed)

    n_colors = mirro.world.n_colors
    world = mirro.world
    n_cells = world.n_cells
    B = world.transition_matrix()  # (n_cells, n_cells, 4)

    # Deep-copy emitter for this seed's dyad
    em = copy.deepcopy(emitter)
    em.world = mirro.world  # run dyad in mirro's world
    # em.true_pos inherited from speciation-phase emitter

    # Action RNGs: index 0 = emitter (IDENTICAL to Exp 65 so emitter walks same path per seed)
    #              index 1 = junior
    rng_em = np.random.default_rng(200000 + 1000 * seed + 0)
    rng_junior = np.random.default_rng(200000 + 1000 * seed + 1)

    # Dual ledger: junior.value_counts is the SEVERED ledger (natural live()-mechanism only)
    # cue_extra accumulates ONLY the channel increments
    cue_extra = np.zeros(n_colors)

    events = 0
    gate_weight_sum = 0.0
    cue_color_counts = {}  # track which cue colors were emitted

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
            cue_extra[cue] += w_junior              # junior's predictability weight (this step)
            events += 1
            gate_weight_sum += w_junior
            cue_color_counts[cue] = cue_color_counts.get(cue, 0) + 1

        # --- Apply moves simultaneously ---
        em.true_pos = new_pos_em
        junior.true_pos = new_pos_junior

    # --- End-of-run ledger computations ---
    vc_sev = junior.value_counts.copy()       # severed ledger
    vc_on = junior.value_counts + cue_extra   # on-ledger (severed + channel increments)

    tot_sev = vc_sev.sum()
    tot_on = vc_on.sum()

    share_sev_cue = float(vc_sev[CUE_COLOR] / tot_sev) if tot_sev > 0 else 0.0
    share_on_cue = float(vc_on[CUE_COLOR] / tot_on) if tot_on > 0 else 0.0
    divergence = share_on_cue - share_sev_cue

    fav_sev = int(np.argmax(vc_sev))
    fav_on = int(np.argmax(vc_on))
    em_fav_end = int(np.argmax(em.value_counts))

    mean_gate_w = gate_weight_sum / events if events > 0 else 0.0
    modal_cue = max(cue_color_counts, key=cue_color_counts.get) if cue_color_counts else -1

    ratio = divergence / ADULT_DIV[seed] if ADULT_DIV[seed] > 0 else float("nan")

    seed_results[seed] = {
        "valid": True,
        "events": events,
        "mean_gate_w": mean_gate_w,
        "jr_mass_pre": jr_tot_post,
        "share_sev_cue": share_sev_cue,
        "share_on_cue": share_on_cue,
        "divergence": divergence,
        "ratio": ratio,
        "fav_sev": fav_sev,
        "fav_on": fav_on,
        "em_fav_end": em_fav_end,
        "modal_cue": modal_cue,
        "cue_color_counts": cue_color_counts,
        "vc_sev": vc_sev,
        "vc_on": vc_on,
        "jr_fav_pre": jr_fav_pre,
    }

    print(
        f"  {seed:>2}   {events:>6}   {mean_gate_w:>10.4f}   {jr_tot_post:>6.1f}   "
        f"{share_sev_cue:>13.4f}   {share_on_cue:>12.4f}   {divergence:>9.4f}   "
        f"{ratio:>5.2f}   {fav_sev:>6}   {fav_on:>5}   {em_fav_end:>9}   {modal_cue:>8}"
    )

print()

# ---------------------------------------------------------------------------
# Gate G1 summary: require >= MIN_VALID_SEEDS valid seeds
# ---------------------------------------------------------------------------

print("--- VALIDITY GATE G1 ---")
for seed in SEEDS:
    r = seed_results[seed]
    if r.get("valid", False):
        print(f"  seed={seed}: VALID (emitter_fav={CUE_COLOR} != junior_pre_fav={r['jr_fav_pre']})")
    else:
        reason = r.get("g1_reason", "unknown")
        print(f"  seed={seed}: INVALID [{reason}]")

n_valid = len(valid_seeds)
if n_valid < MIN_VALID_SEEDS:
    print()
    print(f"G1 FAIL — RUN INVALID (only {n_valid} valid seeds; need >= {MIN_VALID_SEEDS})")
    sys.exit(1)

print(f"  G1 PASS: {n_valid} valid seeds >= {MIN_VALID_SEEDS} required")
print()

# ---------------------------------------------------------------------------
# Gate G2: >= MIN_EVENTS proximity cue events per valid seed
# ---------------------------------------------------------------------------

print("--- VALIDITY GATE G2 ---")
g2_ok = True
for seed in valid_seeds:
    ev = seed_results[seed]["events"]
    ok = ev >= MIN_EVENTS
    if not ok:
        g2_ok = False
    status = "OK" if ok else "FAIL"
    print(f"  seed={seed}: events={ev}  [{status}]")

if not g2_ok:
    print()
    print("G2 FAIL — RUN INVALID (too few proximity events; redesign before verdict)")
    sys.exit(1)

print(f"  G2 PASS: all valid seeds >= {MIN_EVENTS} events")
print()

# ---------------------------------------------------------------------------
# Property checks (evaluated over VALID seeds only)
# ---------------------------------------------------------------------------

def _p1():
    pass_seeds = []
    seed_vals = []
    for seed in valid_seeds:
        d = seed_results[seed]["divergence"]
        r = seed_results[seed]["ratio"]
        ok = r >= P1_RATIO
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:div={d:.4f},ratio={r:.2f}({'ok' if ok else 'FAIL'})")
    n_pass = sum(pass_seeds)
    # if only 4 valid, need 4/4; if 5 valid, need 4/5
    threshold = len(valid_seeds) if len(valid_seeds) == MIN_VALID_SEEDS else 4
    return n_pass >= threshold, (
        f"passes={n_pass}/{len(valid_seeds)} (need >={threshold})  [{'; '.join(seed_vals)}]"
    )


def _p2():
    pass_seeds = []
    fail_list = []
    seed_vals = []
    for seed in valid_seeds:
        d = seed_results[seed]["divergence"]
        ok = d > 0
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:{d:.4f}({'ok' if ok else 'FAIL'})")
        if not ok:
            fail_list.append(seed)
    n_pass = sum(pass_seeds)
    detail = f"passes={n_pass}/{len(valid_seeds)}  [{'; '.join(seed_vals)}]"
    if fail_list:
        detail += f"  WARNING: negative divergence in seeds {fail_list} (all passed G2)"
    return n_pass == len(valid_seeds), detail


def _p3():
    pass_seeds = []
    seed_vals = []
    for seed in valid_seeds:
        fav_on = seed_results[seed]["fav_on"]
        fav_sev = seed_results[seed]["fav_sev"]
        ok = (fav_on == CUE_COLOR) and (fav_sev != CUE_COLOR)
        pass_seeds.append(ok)
        seed_vals.append(
            f"s{seed}:fav_on={fav_on},fav_sev={fav_sev}({'ok' if ok else 'FAIL'})"
        )
    n_pass = sum(pass_seeds)
    return n_pass >= 3, f"passes={n_pass}/{len(valid_seeds)} (need >=3)  [{'; '.join(seed_vals)}]"


check("P1-inertia-ratio>=3x-in-4/valid", _p1)
check("P2-sign-divergence>0-in-all-valid", _p2)
check("P3-installation-fav_on==cue-AND-fav_sev!=cue-in-3/valid", _p3)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
print()
print("P1 per-seed ratios (divergence / adult_div):")
for seed in valid_seeds:
    r = seed_results[seed]
    print(f"  seed={seed}: divergence={r['divergence']:.4f}  adult_div={ADULT_DIV[seed]:.4f}  "
          f"ratio={r['ratio']:.2f}x")
print()
print("P3 per-seed favorite pairs (fav_sev, fav_on):")
for seed in valid_seeds:
    r = seed_results[seed]
    print(f"  seed={seed}: fav_sev={r['fav_sev']}  fav_on={r['fav_on']}  "
          f"cue_color={CUE_COLOR}")
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

if not p1_passed:
    print("F1 FIRED: mass-inertia diagnosis WRONG — NEGATIVE; CONSULT required "
          "(divergence does not scale with inverse mass as predicted by Exp 65 diagnosis)")
if not p2_passed:
    print("F2 FIRED: wiring/gating bug — HALT for investigation")

if p1_passed and p2_passed and not p3_passed:
    print("P3 not met: MIXED — inertia law confirmed; installation needs a larger dose")

if p1_passed and p2_passed and p3_passed:
    print("ALL PASS: inertia law confirmed AND favorite installed")

if p1_passed and p2_passed:
    print("  No falsifiers fired.")

print()

# ---------------------------------------------------------------------------
# Diagnostics: per-seed junior post-settle profile, cue distribution, value shares
# ---------------------------------------------------------------------------

print("--- DIAGNOSTICS: JUNIOR POST-SETTLE PROFILES ---")
print()
print(f"{'seed':>4}  {'jr_fav_pre':>10}  {'jr_mass':>7}  {'map_acc':>7}  "
      f"{'jr_shares_pre':>30}")
print("-" * 70)
for seed in SEEDS:
    r = seed_results[seed]
    if not r.get("valid", False):
        print(f"  {seed:>2}   (INVALID — excluded)")
        continue
    # Re-derive settled stats from stored values (we stored jr_fav_pre and jr_mass_pre)
    print(f"  {seed:>2}   {r['jr_fav_pre']:>9}   {r['jr_mass_pre']:>6.1f}  "
          f"  (see pre-dyad printout above)")

print()

# Per-seed cue color distribution
print("--- CUE COLOR DISTRIBUTION PER VALID SEED ---")
for seed in valid_seeds:
    r = seed_results[seed]
    total_ev = r["events"]
    cc = r["cue_color_counts"]
    dist_str = "  ".join(
        f"color{c}:{cnt}({100*cnt/total_ev:.1f}%)" for c, cnt in sorted(cc.items())
    )
    print(f"  seed={seed}: {dist_str}  (total={total_ev})")

print()

# Per-seed severed vs on value shares for all colors
print("--- VALUE SHARES ALL COLORS (end of dyad, valid seeds) ---")
n_colors = mirro.world.n_colors
header = f"{'seed':>4}  " + "  ".join(
    f"sev[{c}]" + " " * 2 + f"on[{c}]" for c in range(n_colors)
)
print("  " + header)
for seed in valid_seeds:
    r = seed_results[seed]
    vc_sev = r["vc_sev"]
    vc_on = r["vc_on"]
    tot_sev = vc_sev.sum()
    tot_on = vc_on.sum()
    vals = "  ".join(
        f"{vc_sev[c]/tot_sev:.4f}  {vc_on[c]/tot_on:.4f}" for c in range(n_colors)
    )
    print(f"  {seed:>2}   {vals}")

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

p1_passed = checks[0][1]
p2_passed = checks[1][1]
p3_passed = checks[2][1]

if p1_passed and p2_passed and p3_passed:
    print("EXP66: PASS")
elif p1_passed and p2_passed and not p3_passed:
    print("EXP66: MIXED [P3]")
else:
    fail_tags = [name.split("-")[0] for name, passed, _ in checks if not passed]
    print(f"EXP66: FAIL {fail_tags}")
