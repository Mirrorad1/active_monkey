"""Exp 88 — graded-uncertainty rung 3: the window theorem — robustness and adaptability
are the same number, and without forgetting, rigidity grows without bound.

The direction's heart. Mechanism this iteration: value-count decay only (value_counts *=
LV per step; pA untouched — isolating the identity ledger from the map ledger; Exp 85-87
covered the map side). The arithmetic makes a theorem-shaped claim: a decayed identity is
a moving window of mass ~1/(1-LV), so BOTH its robustness to transient adverse spells AND
its time to adopt a genuine world change are set by the SAME window length, independent
of age — while a non-decaying identity's evidence gap grows linearly with age, making it
ever more robust AND ever more rigid: the old cannot change their minds in bounded time.

Worlds: identity world W = color 0 on the 13 checkerboard cells ((r+c) even), colors 1
and 2 alternating on the 12 odd cells (6/6) — color 0 is the strong structural favorite
(rate gap ~0.27 mass/step). Change world: all cells color 2.

Cohorts (8 seeds each, births 900+s, start cell 12):
  ADOPTION: arms (LV=1.0, LV=0.999) x ages (young = 6000-step identity, old = 18000):
    after identity formation the world switches PERMANENTLY to all-2; adoption time =
    first 50-step-sampled time with favorite == 2 (cap 6000, censored = cap).
  SPELL (young identity only): arms (LV=1.0, 0.999, 0.997): a 200-step all-2 spell, then
    600 more steps back in W; flipped_during = favorite == 2 at any 50-step sample inside
    the spell or at spell end.
Predeclared (window arithmetic, stated before running):
  P1 (rigidity grows with age, non-decay): median adoption_old / median adoption_young
     for LV=1.0 in [2.2, 3.8] (gap ~ age predicts ~3).
  P2 (window adoption is age-free): the same ratio for LV=0.999 in [0.7, 1.4].
  P3 (robustness ordering, spell cohorts): LV=0.997 flips during the spell in >= 6/8;
     LV=0.999 does NOT flip in >= 6/8; LV=1.0 does NOT flip in >= 6/8.
Falsifiers:
  F1 = P1 fails -> the linear-gap rigidity arithmetic is wrong for non-decay.
  F2 = P2 fails -> the window model of decayed identity is wrong.
  F3 = P3 fails -> the robustness ordering is wrong (window arithmetic fails at short
     horizons).
Any falsifier halts rung 3 for diagnosis. All passing = the window theorem demonstrated:
the design question for any forgetting substrate is matching ONE number (the window) to
the world's transient-vs-change timescales — and the no-forgetting alternative is
unbounded rigidity, quantified here by the age ratio.
Provided priors declared: the worlds, phase lengths, the mechanism (LV, no floor), birth
seeds, sampling grid. Fresh separate-root newborns; spines untouched (nothing loaded).
"""
from __future__ import annotations

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS = list(range(8))          # offsets 0..7; actual birth seed = BIRTH_BASE + offset
BIRTH_BASE = 900
START = 12
YOUNG = 6000
OLD = 18000
ADOPT_CAP = 6000
SPELL = 200
POST_SPELL = 600
SAMPLE = 50

# ---------------------------------------------------------------------------
# Build worlds
# ---------------------------------------------------------------------------

# W: 5x5 grid (25 cells), color 0 on 13 checkerboard cells ((r+c)%2==0),
# colors 1 and 2 alternating on 12 odd cells in ascending index order.
_ROWS, _COLS = 5, 5
_N_CELLS = _ROWS * _COLS
_N_COLORS = 3

_cmap_W = [0] * _N_CELLS
_odd_cells = [i for i in range(_N_CELLS) if ((i // _COLS) + (i % _COLS)) % 2 == 1]
for _k, _idx in enumerate(_odd_cells):
    _cmap_W[_idx] = 1 + (_k % 2)   # alternates 1, 2, 1, 2, ...

# Sanity-check counts
_counts_W = {c: _cmap_W.count(c) for c in range(_N_COLORS)}
assert _counts_W == {0: 13, 1: 6, 2: 6}, (
    f"World W color counts wrong: {_counts_W}"
)

world_W = World(rows=_ROWS, cols=_COLS, cmap=_cmap_W, n_colors=_N_COLORS)

# ALL2: all cells color 2
_cmap_ALL2 = [2] * _N_CELLS
world_ALL2 = World(rows=_ROWS, cols=_COLS, cmap=_cmap_ALL2, n_colors=_N_COLORS)

# Print grids
def _print_grid(label: str, w: World) -> None:
    print(f"  {label}:")
    for r in range(w.rows):
        row_vals = w.cmap[r * w.cols : (r + 1) * w.cols]
        print("   ", " ".join(str(v) for v in row_vals))

print("Exp 88 — graded-uncertainty rung 3: the window theorem.")
print()
print("Worlds:")
_print_grid("W  (identity world)", world_W)
_print_grid("ALL2 (change world)", world_ALL2)
print(f"  W color counts: {_counts_W}")
print()

# Precompute transition matrices (immutable; world geometry identical for both)
_B = world_W.transition_matrix()   # same grid shape, shared B

# ---------------------------------------------------------------------------
# Stepper helpers
# ---------------------------------------------------------------------------

def _make_rng(c: Creature) -> np.random.Generator:
    """Derive RNG identically to live() from (c._seed, c.rng_counter)."""
    combined = (c._seed * 1_000_003 + c.rng_counter) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(combined)


def run_steps(c: Creature, n: int, lv: float) -> None:
    """Run n steps, replicating live() exactly + value_counts *= lv per step (no floor).

    Mechanism: value-count decay only — pA untouched.
    RNG derivation and bookkeeping match live() exactly:
      rng derived once from (c._seed, c.rng_counter) before the loop;
      after the loop: c.age_steps += n; c.rng_counter += 1.
    """
    rng = _make_rng(c)
    B = _B  # precomputed; same grid for all creatures

    do_decay = lv < 1.0

    for _ in range(n):
        # --- value-count decay (identity ledger; pA untouched) ---
        if do_decay:
            c.value_counts *= lv

        # --- A_hat from pA (unchanged) ---
        A = c.pA.copy()
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A / col_sums

        # --- observe ---
        obs = int(c.world.cmap[c.true_pos])

        # --- belief update: qs ∝ likelihood(obs) * prior(qs) ---
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * c.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(c.world.n_cells) / c.world.n_cells

        # --- Dirichlet count learning: pA[obs, :] += qs_upd ---
        c.pA[obs, :] += qs_upd

        # --- value accumulation (Exp 26 mechanism, matching live()) ---
        map_cell = int(np.argmax(qs_upd))
        predicted_obs_dist = A_hat[:, map_cell]
        h_predicted = -np.sum(
            predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
        )
        predictability_weight = np.exp(-h_predicted)
        c.value_counts[obs] += predictability_weight

        # --- choose action, move ---
        action = int(rng.integers(0, 4))
        c.true_pos = c.world.move(c.true_pos, action)

        # --- advance belief through movement model ---
        c.qs = B[:, :, action] @ qs_upd

    # Bookkeeping matches live()
    c.age_steps += n
    c.rng_counter += 1


def run_steps_sampled(
    c: Creature, n: int, lv: float, sample: int
) -> list:
    """Run n steps with sampling every `sample` steps; return list of (t, favorite).

    t is the step count within this call (1-indexed at each sample point).
    favorite = argmax(value_counts).  Sampling at step `sample`, `2*sample`, ...
    and also at the final step if n is not a multiple of sample.
    """
    rng = _make_rng(c)
    B = _B

    do_decay = lv < 1.0
    samples = []

    for step_i in range(1, n + 1):
        # --- value-count decay ---
        if do_decay:
            c.value_counts *= lv

        # --- A_hat ---
        A = c.pA.copy()
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        A_hat = A / col_sums

        # --- observe ---
        obs = int(c.world.cmap[c.true_pos])

        # --- belief update ---
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * c.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(c.world.n_cells) / c.world.n_cells

        # --- Dirichlet count learning ---
        c.pA[obs, :] += qs_upd

        # --- value accumulation ---
        map_cell = int(np.argmax(qs_upd))
        predicted_obs_dist = A_hat[:, map_cell]
        h_predicted = -np.sum(
            predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
        )
        predictability_weight = np.exp(-h_predicted)
        c.value_counts[obs] += predictability_weight

        # --- choose action, move ---
        action = int(rng.integers(0, 4))
        c.true_pos = c.world.move(c.true_pos, action)

        # --- advance belief ---
        c.qs = B[:, :, action] @ qs_upd

        # --- sample ---
        if step_i % sample == 0 or step_i == n:
            samples.append((step_i, int(np.argmax(c.value_counts))))

    c.age_steps += n
    c.rng_counter += 1
    return samples


# ---------------------------------------------------------------------------
# ADOPTION cohorts
# ---------------------------------------------------------------------------
# Arms: (age, lv) = (YOUNG, 1.0), (YOUNG, 0.999), (OLD, 1.0), (OLD, 0.999)
# For each seed: birth in W, form identity (run_steps(age, lv)), G-check favorite==0,
# switch to ALL2, run sampled until favorite==2 (cap ADOPT_CAP).

print("=" * 68)
print("ADOPTION COHORTS")
print("=" * 68)

# Stores: dict keyed by (age_label, lv) -> list of per-seed dicts
adoption_results: dict = {}

ADOPTION_ARMS = [
    ("young", YOUNG, 1.0),
    ("young", YOUNG, 0.999),
    ("old",   OLD,   1.0),
    ("old",   OLD,   0.999),
]

for age_label, age_steps, lv in ADOPTION_ARMS:
    arm_key = (age_label, lv)
    arm_data = []

    print(f"\n-- arm: age={age_label} ({age_steps} steps)  LV={lv} --")

    for s in SEEDS:
        birth_seed = BIRTH_BASE + s
        c = Creature.birth(f"exp88-adopt-{age_label}-lv{lv}-s{s}", world_W, seed=birth_seed)
        c.true_pos = START

        # Identity formation in W
        run_steps(c, age_steps, lv)

        fav_after_form = int(np.argmax(c.value_counts))
        g_ok = (fav_after_form == 0)

        if not g_ok:
            print(f"  seed {birth_seed}: G-check FAIL (favorite={fav_after_form} after formation); excluded from cell")
            arm_data.append({
                "seed": birth_seed, "g_ok": False,
                "adopt_time": None, "censored": None,
            })
            continue

        # Switch world permanently to ALL2
        c.world = world_ALL2

        # Adoption run: sample every SAMPLE steps, cap at ADOPT_CAP
        adopt_time = ADOPT_CAP      # default = censored
        censored = True
        sampled = run_steps_sampled(c, ADOPT_CAP, lv, SAMPLE)
        for t, fav in sampled:
            if fav == 2:
                adopt_time = t
                censored = False
                break

        arm_data.append({
            "seed": birth_seed, "g_ok": True,
            "adopt_time": adopt_time, "censored": censored,
        })
        status = f"adopt@{adopt_time}" + (" [CENSORED]" if censored else "")
        print(f"  seed {birth_seed}: G-check OK  {status}")

    adoption_results[arm_key] = arm_data

# ---------------------------------------------------------------------------
# Adoption table + medians + ratio
# ---------------------------------------------------------------------------

print()
print("--- ADOPTION TABLE ---")
print(f"  {'seed':>5}  {'young/1.0':>10}  {'young/0.999':>12}  {'old/1.0':>10}  {'old/0.999':>12}")
print("  " + "-" * 56)

for s in SEEDS:
    birth_seed = BIRTH_BASE + s
    row = {}
    for age_label, age_steps, lv in ADOPTION_ARMS:
        key = (age_label, lv)
        rec = adoption_results[key][s]
        if rec["g_ok"] and rec["adopt_time"] is not None:
            tag = f"{rec['adopt_time']}" + ("*" if rec["censored"] else "")
        elif not rec["g_ok"]:
            tag = "excl"
        else:
            tag = "n/a"
        row[(age_label, lv)] = tag
    print(
        f"  {birth_seed:>5}  "
        f"{row[('young', 1.0)]:>10}  "
        f"{row[('young', 0.999)]:>12}  "
        f"{row[('old', 1.0)]:>10}  "
        f"{row[('old', 0.999)]:>12}"
    )
print("  (* = censored at cap)")
print()


def arm_median(arm_key: tuple) -> float | None:
    """Compute median adoption time over valid (G-check OK) seeds."""
    data = adoption_results[arm_key]
    times = [r["adopt_time"] for r in data if r["g_ok"] and r["adopt_time"] is not None]
    censor_count = sum(1 for r in data if r["g_ok"] and r.get("censored"))
    valid_count = len(times)
    if valid_count == 0:
        return None, 0, 0
    return float(np.median(times)), valid_count, censor_count


med_young_10, n_y10, c_y10 = arm_median(("young", 1.0))
med_young_999, n_y999, c_y999 = arm_median(("young", 0.999))
med_old_10, n_o10, c_o10 = arm_median(("old", 1.0))
med_old_999, n_o999, c_o999 = arm_median(("old", 0.999))

print("Medians (censored seeds counted at cap):")
print(f"  young / LV=1.000 : {med_young_10}  (n={n_y10}, censored={c_y10})")
print(f"  young / LV=0.999 : {med_young_999}  (n={n_y999}, censored={c_y999})")
print(f"  old   / LV=1.000 : {med_old_10}  (n={n_o10}, censored={c_o10})")
print(f"  old   / LV=0.999 : {med_old_999}  (n={n_o999}, censored={c_o999})")

ratio_p1 = (med_old_10 / med_young_10) if (med_young_10 and med_young_10 > 0) else None
ratio_p2 = (med_old_999 / med_young_999) if (med_young_999 and med_young_999 > 0) else None

print(f"\nAge ratios:")
print(f"  P1 (LV=1.000): old/young = {ratio_p1}")
print(f"  P2 (LV=0.999): old/young = {ratio_p2}")

# ---------------------------------------------------------------------------
# SPELL cohorts
# ---------------------------------------------------------------------------

print()
print("=" * 68)
print("SPELL COHORTS  (young identity only)")
print("=" * 68)

SPELL_LVS = [1.0, 0.999, 0.997]

# Stores: dict keyed by lv -> list of per-seed dicts
spell_results: dict = {}

for lv in SPELL_LVS:
    arm_data = []
    print(f"\n-- arm: LV={lv} --")

    for s in SEEDS:
        birth_seed = BIRTH_BASE + s
        c = Creature.birth(f"exp88-spell-lv{lv}-s{s}", world_W, seed=birth_seed)
        c.true_pos = START

        # Identity formation in W (YOUNG steps)
        run_steps(c, YOUNG, lv)

        fav_after_form = int(np.argmax(c.value_counts))
        g_ok = (fav_after_form == 0)

        if not g_ok:
            print(f"  seed {birth_seed}: G-check FAIL (favorite={fav_after_form}); excluded")
            arm_data.append({
                "seed": birth_seed, "g_ok": False,
                "flipped_during": None, "fav_end": None,
            })
            continue

        # Spell: SPELL steps in ALL2
        c.world = world_ALL2
        spell_samples = run_steps_sampled(c, SPELL, lv, SAMPLE)
        flipped_during = any(fav == 2 for _, fav in spell_samples)

        # Post-spell: POST_SPELL steps back in W
        c.world = world_W
        run_steps(c, POST_SPELL, lv)
        fav_end = int(np.argmax(c.value_counts))

        arm_data.append({
            "seed": birth_seed, "g_ok": True,
            "flipped_during": flipped_during,
            "fav_end": fav_end,
        })

        spell_detail = "FLIPPED" if flipped_during else "held"
        recovery = f"  fav_end={fav_end}"
        print(f"  seed {birth_seed}: G-check OK  spell={spell_detail}{recovery}")

    spell_results[lv] = arm_data

# ---------------------------------------------------------------------------
# Spell table
# ---------------------------------------------------------------------------

print()
print("--- SPELL TABLE ---")
print(f"  {'seed':>5}  {'LV=1.0 flip':>12}  {'LV=0.999 flip':>14}  {'LV=0.997 flip':>14}  "
      f"{'end fav 1.0':>12}  {'end fav .999':>13}  {'end fav .997':>13}")
print("  " + "-" * 90)

for s in SEEDS:
    birth_seed = BIRTH_BASE + s
    row_parts = []
    fav_parts = []
    for lv in SPELL_LVS:
        rec = spell_results[lv][s]
        if rec["g_ok"]:
            row_parts.append("FLIP" if rec["flipped_during"] else "hold")
            fav_parts.append(str(rec["fav_end"]))
        else:
            row_parts.append("excl")
            fav_parts.append("excl")
    print(
        f"  {birth_seed:>5}  "
        f"{row_parts[0]:>12}  {row_parts[1]:>14}  {row_parts[2]:>14}  "
        f"{fav_parts[0]:>12}  {fav_parts[1]:>13}  {fav_parts[2]:>13}"
    )

print()

# Flip counts per arm (over G-ok seeds)
flip_counts = {}
for lv in SPELL_LVS:
    valid = [r for r in spell_results[lv] if r["g_ok"]]
    flip_counts[lv] = sum(1 for r in valid if r["flipped_during"])
    print(f"  LV={lv}: flipped {flip_counts[lv]}/{len(valid)} valid seeds during spell")

recovery_counts = {}
for lv in SPELL_LVS:
    valid = [r for r in spell_results[lv] if r["g_ok"]]
    recovery_counts[lv] = sum(1 for r in valid if r["fav_end"] == 0)
    print(f"  LV={lv}: recovered to fav=0 end of post-spell in {recovery_counts[lv]}/{len(valid)} valid seeds")

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

print()
print("--- PROPERTY CHECKS ---")

PASS_THRESHOLD = 6   # >= 6/8


def check(label: str, passed: bool) -> bool:
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
    return passed


# P1: median adoption old/young ratio for LV=1.0 in [2.2, 3.8]
P1_LO, P1_HI = 2.2, 3.8
if ratio_p1 is not None:
    P1_ok = P1_LO <= ratio_p1 <= P1_HI
else:
    P1_ok = False
check(
    f"P1 (rigidity grows with age, LV=1.0): ratio={ratio_p1} in [{P1_LO}, {P1_HI}]: "
    f"{'PASS' if P1_ok else 'FAIL'}",
    P1_ok,
)

print()

# P2: median adoption old/young ratio for LV=0.999 in [0.7, 1.4]
P2_LO, P2_HI = 0.7, 1.4
if ratio_p2 is not None:
    P2_ok = P2_LO <= ratio_p2 <= P2_HI
else:
    P2_ok = False
check(
    f"P2 (window adoption is age-free, LV=0.999): ratio={ratio_p2} in [{P2_LO}, {P2_HI}]: "
    f"{'PASS' if P2_ok else 'FAIL'}",
    P2_ok,
)

print()

# P3: robustness ordering (three sub-conditions)
# LV=0.997 flips >= 6/8; LV=0.999 does NOT flip >= 6/8; LV=1.0 does NOT flip >= 6/8

valid_n_997 = sum(1 for r in spell_results[0.997] if r["g_ok"])
valid_n_999 = sum(1 for r in spell_results[0.999] if r["g_ok"])
valid_n_10  = sum(1 for r in spell_results[1.0]   if r["g_ok"])

P3a_ok = flip_counts[0.997] >= PASS_THRESHOLD
P3b_ok = (valid_n_999 - flip_counts[0.999]) >= PASS_THRESHOLD
P3c_ok = (valid_n_10  - flip_counts[1.0])   >= PASS_THRESHOLD
P3_ok = P3a_ok and P3b_ok and P3c_ok

check(
    f"P3a (LV=0.997 flips >= {PASS_THRESHOLD}/8): {flip_counts[0.997]}/{valid_n_997}: "
    f"{'PASS' if P3a_ok else 'FAIL'}",
    P3a_ok,
)
check(
    f"P3b (LV=0.999 does NOT flip >= {PASS_THRESHOLD}/8): "
    f"{valid_n_999 - flip_counts[0.999]}/{valid_n_999} held: "
    f"{'PASS' if P3b_ok else 'FAIL'}",
    P3b_ok,
)
check(
    f"P3c (LV=1.0 does NOT flip >= {PASS_THRESHOLD}/8): "
    f"{valid_n_10 - flip_counts[1.0]}/{valid_n_10} held: "
    f"{'PASS' if P3c_ok else 'FAIL'}",
    P3c_ok,
)
check(
    f"P3 (robustness ordering, all three): {'PASS' if P3_ok else 'FAIL'}",
    P3_ok,
)

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

print()
print("--- FALSIFIER MAP ---")

F1_fired = not P1_ok
F2_fired = not P2_ok
F3_fired = not P3_ok

if not F1_fired:
    print(
        f"  F1 did not fire (P1 PASS — rigidity grows with age for LV=1.0; "
        f"old/young ratio={ratio_p1} in [{P1_LO}, {P1_HI}])."
    )
else:
    print(
        f"  F1 FIRED: P1 failed (old/young ratio={ratio_p1}; expected [{P1_LO}, {P1_HI}]) "
        f"-> the linear-gap rigidity arithmetic is wrong for non-decay; rung 3 halted."
    )

if not F2_fired:
    print(
        f"  F2 did not fire (P2 PASS — window adoption is age-free for LV=0.999; "
        f"old/young ratio={ratio_p2} in [{P2_LO}, {P2_HI}])."
    )
else:
    print(
        f"  F2 FIRED: P2 failed (old/young ratio={ratio_p2}; expected [{P2_LO}, {P2_HI}]) "
        f"-> the window model of decayed identity is wrong; rung 3 halted."
    )

if not F3_fired:
    print(
        f"  F3 did not fire (P3 PASS — robustness ordering correct: "
        f"LV=0.997 flips {flip_counts[0.997]}/8, "
        f"LV=0.999 holds {valid_n_999 - flip_counts[0.999]}/8, "
        f"LV=1.0 holds {valid_n_10 - flip_counts[1.0]}/8)."
    )
else:
    details = []
    if not P3a_ok:
        details.append(f"LV=0.997 flips only {flip_counts[0.997]}/8 (need {PASS_THRESHOLD})")
    if not P3b_ok:
        details.append(f"LV=0.999 holds only {valid_n_999 - flip_counts[0.999]}/8 (need {PASS_THRESHOLD})")
    if not P3c_ok:
        details.append(f"LV=1.0 holds only {valid_n_10 - flip_counts[1.0]}/8 (need {PASS_THRESHOLD})")
    print(
        f"  F3 FIRED: P3 failed ({'; '.join(details)}) "
        f"-> robustness ordering wrong (window arithmetic fails at short horizons); rung 3 halted."
    )

print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

fired = [f"F{n}" for n, fired in [(1, F1_fired), (2, F2_fired), (3, F3_fired)] if fired]

if not fired:
    print("EXP88: WINDOW THEOREM DEMONSTRATED")
else:
    msgs = []
    if F1_fired:
        msgs.append("F1 — linear-gap rigidity arithmetic wrong for non-decay")
    if F2_fired:
        msgs.append("F2 — window model of decayed identity wrong")
    if F3_fired:
        msgs.append("F3 — robustness ordering wrong at short horizons")
    print(f"EXP88: {'; '.join(msgs)}; rung 3 halted")
