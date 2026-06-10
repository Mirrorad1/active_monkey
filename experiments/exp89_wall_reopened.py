"""Exp 89 — graded-uncertainty rung 4a: the adult-transmission wall, re-tested with the window.

Exp 65 (NEGATIVE): a grounded cue channel demonstrably transmits (sign 5/5) but ~175 cue
counts cannot move an adult receiver's ~9,200-count lifetime ledger — divergence 0.012-
0.016 against the predeclared 0.02 bar. The window arithmetic (Exp 88) says a LV=0.999
adult's value ledger equilibrates near inflow/(1-LV) ~ 600, so the SAME dose should now
clear the SAME bar with room to spare (expected divergence ~0.1-0.2).

Receiver: the Exp 65-era adult — vela@12750, git-recovered (git show e7220c1~1:...) for
exactness — deepcopied into two arms and PRE-EQUILIBRATED 4000 steps in its own world
before the dyad (the windowed arm reaches its steady-state ledger; the control arm simply
lives 4000 more steps). Emitter substitution (declared): a fresh separate-root creature
speciated 2000 steps in an all-color-1 world (favorite 1 guaranteed-checked); Exp 65 used
a mirro-fork, but only the RECEIVER-side mass arithmetic is under test and the emitter's
sole role is emitting its favorite. Channel, dose, and dual-ledger exactness as Exp 65 —
under decay the ledgers remain exact because decay is linear: the severed ledger and the
cue-extra ledger each decay by LV per step independently.

Predeclared (8 fresh dyad seeds 5-12, rng family 200000+1000*seed+i as Exp 65; the
ORIGINAL 0.02 threshold):
  G1 (per arm x seed): receiver's pre-dyad favorite != 1 (the cue), else the seed is
     excluded for that arm (>= 6 valid required per arm).
  P1 (the wall opens): windowed-arm cue-color share divergence (on - severed) >= 0.02 in
     >= 6/8 valid seeds.
  P2 (the wall stands without the window): control-arm divergence < 0.02 in >= 6/8 valid
     seeds (Exp 65's negative replicates at this state).
  P3 (adult adoption, LOW confidence): windowed arm ON-ledger favorite == 1 AND severed
     favorite != 1, in >= 4/8 valid seeds.
Falsifiers:
  F1 = P1 fails -> the window does NOT open the social door; receiver mass is not the
     whole wall (rung 4a NEGATIVE — a real bound on the mechanism's reach).
  F2 = P2 fails -> Exp 65's wall does not replicate here; comparability broken, diagnose
     before any claim.
Provided priors declared: everything Exp 65 declared (channel, dose, worlds) plus the
mechanism (LV=0.999 on the receiver's value ledgers only, during equilibration and dyad;
pA untouched), the equilibration length, the emitter substitution, fresh seeds. The
committed lines are untouched (vela read from git history; mirro not loaded).
"""
from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np

from active_loop.creature import Creature, World

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LV = 0.999
EQUIL = 4000
DYAD = 2000
SEEDS = list(range(5, 13))   # seeds 5..12 inclusive
PROX = 1                      # Manhattan distance gate
THRESH = 0.02
COMMIT_REF = "e7220c1~1"
EXPECT_AGE = 12750
EXPECT_HASH = "875ac30d715a"
MIN_VALID = 6
MIN_EVENTS = 50
CUE_COLOR = 1                 # emitter-world is all-color-1; favorite must be 1

# ---------------------------------------------------------------------------
# Step 1: recover vela@12750 from git history (exp82 pattern)
# ---------------------------------------------------------------------------

scratch = Path("experiments/outputs/exp89_pre_snapshot")
scratch.mkdir(parents=True, exist_ok=True)

print("Exp 89 — graded-uncertainty rung 4a: the adult-transmission wall, re-tested with the window")
print()
print(f"Recovering vela snapshot from {COMMIT_REF} ...")
print(f"  scratch dir: {scratch}  [untracked; not committed — derivable from git history]")
print()

for fname in ["manifest.json", "arrays.npz"]:
    result = subprocess.run(
        ["git", "show", f"{COMMIT_REF}:creature/state/vela/{fname}"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git show failed for {fname}: {result.stderr.decode()}")
        sys.exit(1)
    (scratch / fname).write_bytes(result.stdout)

recovered = Creature.load(scratch)
print(f"Loaded vela: age={recovered.age_steps}  hash={recovered._state_hash()[:12]}")

if recovered.age_steps != EXPECT_AGE or recovered._state_hash()[:12] != EXPECT_HASH:
    print(
        f"SNAPSHOT MISMATCH: expected age={EXPECT_AGE} hash={EXPECT_HASH}, "
        f"got age={recovered.age_steps} hash={recovered._state_hash()[:12]}"
    )
    sys.exit(1)

print(f"Snapshot verified: age={EXPECT_AGE}  hash={EXPECT_HASH}")
print()

# ---------------------------------------------------------------------------
# Step 2: build emitter (separate-root; speciated in all-color-1 world)
# ---------------------------------------------------------------------------

print("--- EMITTER SPECIATION (separate-root, fresh birth) ---")

spec_world = World(rows=5, cols=5, cmap=[1] * 25, n_colors=recovered.world.n_colors)
em0 = Creature.birth("exp89-emitter", spec_world, seed=42)
em0.true_pos = 0

print(f"  emitter before speciation: favorite={em0.favorite()}  "
      f"value_counts={np.round(em0.value_counts, 4)}")

# Speciate 2000 steps in the all-color-1 world via live()
em0.live(2000)

print(f"  emitter after  speciation: favorite={em0.favorite()}  "
      f"value_counts={np.round(em0.value_counts, 4)}")

if em0.favorite() != CUE_COLOR:
    print(f"EMITTER SPECIATION FAIL: expected favorite=1 after all-color-1 speciation, "
          f"got {em0.favorite()}")
    sys.exit(1)

print(f"  Emitter speciation OK: favorite={em0.favorite()} == CUE_COLOR={CUE_COLOR}")
print()

# ---------------------------------------------------------------------------
# Stepper helpers
# ---------------------------------------------------------------------------

def _make_rng(c: Creature) -> np.random.Generator:
    """Derive RNG identically to live() from (c._seed, c.rng_counter)."""
    combined = (c._seed * 1_000_003 + c.rng_counter) & 0xFFFFFFFFFFFFFFFF
    return np.random.default_rng(combined)


def run_steps_single(c: Creature, n: int, do_decay: bool) -> None:
    """Run n steps replicating live() exactly, optionally decaying value_counts each step.

    Mechanism: value_counts *= LV at step START (before observation/accumulation) when
    do_decay=True; pA untouched. RNG bookkeeping identical to live().

    NOTE on equilibration rngs: the windowed (do_decay=True) and control (do_decay=False)
    arms equilibrate from the same deepcopy of recovered with the SAME derived-rng path.
    Decay multiplies value_counts but consumes no rng draws, so both arms take identical
    random walks and arrive at the same position and pA after EQUIL steps — their positions
    and pA are byte-identical at dyad start; only value_counts differ.
    """
    rng = _make_rng(c)
    world = c.world
    n_cells = world.n_cells
    B = world.transition_matrix()

    for _ in range(n):
        # value-count decay (identity ledger; pA untouched)
        if do_decay:
            c.value_counts *= LV

        # A_hat from pA (unchanged by decay)
        A_hat = c._A_hat()

        # observe
        obs = int(world.cmap[c.true_pos])

        # belief update: qs ∝ likelihood(obs) * prior(qs)
        likelihood = A_hat[obs, :]
        qs_upd = likelihood * c.qs
        denom = qs_upd.sum()
        if denom > 0:
            qs_upd = qs_upd / denom
        else:
            qs_upd = np.ones(n_cells) / n_cells

        # Dirichlet count learning
        c.pA[obs, :] += qs_upd

        # value accumulation (Exp 26 mechanism, matching live())
        map_cell = int(np.argmax(qs_upd))
        predicted_obs_dist = A_hat[:, map_cell]
        h_predicted = -np.sum(predicted_obs_dist * np.log(predicted_obs_dist + 1e-12))
        predictability_weight = np.exp(-h_predicted)
        c.value_counts[obs] += predictability_weight

        # choose action, move
        action = int(rng.integers(0, 4))
        c.true_pos = world.move(c.true_pos, action)

        # advance belief through movement model
        c.qs = B[:, :, action] @ qs_upd

    # Bookkeeping matches live()
    c.age_steps += n
    c.rng_counter += 1


def run_dyad(recv: Creature, em: Creature, seed: int, do_decay: bool) -> dict:
    """Run DYAD paired stepper; return per-seed result dict.

    Both creatures' natural live()-math updates happen first; then channel after both
    updates using start-of-step positions (Manhattan <= PROX).

    Dual ledger: recv.value_counts is the SEVERED ledger (natural only).
    cue_extra accumulates ONLY channel increments.
    Under decay both ledgers decay by LV independently each step — exact because decay
    is linear: (severed + cue_extra) * LV = severed*LV + cue_extra*LV.

    Emitter rng: default_rng(200000+1000*seed+0)
    Receiver rng: default_rng(200000+1000*seed+1)
    Dyad runs in the receiver's world (vela's world); em.world set to recv.world.
    """
    world = recv.world
    n_cells = world.n_cells
    n_colors = world.n_colors
    B = world.transition_matrix()

    rng_em = np.random.default_rng(200000 + 1000 * seed + 0)
    rng_recv = np.random.default_rng(200000 + 1000 * seed + 1)

    # Set emitter to run in receiver's world
    em.world = recv.world
    em.true_pos = 0

    # Dual ledger: cue_extra accumulates channel-only increments
    cue_extra = np.zeros(n_colors)

    events = 0

    for step in range(DYAD):
        # Read simultaneous start-of-step positions
        pos_em = em.true_pos
        pos_recv = recv.true_pos

        # --- value-count decay (both ledgers independently, receiver only) ---
        if do_decay:
            recv.value_counts *= LV
            cue_extra *= LV

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

        # --- Receiver natural update (replicates live() exactly) ---
        A_hat_recv = recv._A_hat()
        obs_recv = int(world.cmap[pos_recv])
        likelihood_recv = A_hat_recv[obs_recv, :]
        qs_upd_recv = likelihood_recv * recv.qs
        denom_recv = qs_upd_recv.sum()
        if denom_recv > 0:
            qs_upd_recv = qs_upd_recv / denom_recv
        else:
            qs_upd_recv = np.ones(n_cells) / n_cells
        recv.pA[obs_recv, :] += qs_upd_recv
        map_cell_recv = int(np.argmax(qs_upd_recv))
        A_hat_recv_col = A_hat_recv[:, map_cell_recv]
        h_recv = -np.sum(A_hat_recv_col * np.log(A_hat_recv_col + 1e-12))
        w_recv = np.exp(-h_recv)
        recv.value_counts[obs_recv] += w_recv  # natural (severed) ledger
        action_recv = int(rng_recv.integers(0, 4))
        new_pos_recv = world.move(pos_recv, action_recv)
        recv.qs = B[:, :, action_recv] @ qs_upd_recv

        # --- Channel (after both natural updates, using start-of-step positions) ---
        r_em, c_em = divmod(pos_em, world.cols)
        r_rv, c_rv = divmod(pos_recv, world.cols)
        manhattan = abs(r_em - r_rv) + abs(c_em - c_rv)
        if manhattan <= PROX:
            cue = int(np.argmax(em.value_counts))  # emitter's current favorite
            cue_extra[cue] += w_recv               # receiver's predictability weight (this step)
            events += 1

        # --- Apply moves simultaneously ---
        em.true_pos = new_pos_em
        recv.true_pos = new_pos_recv

    # --- End-of-run ledger computations ---
    vc_sev = recv.value_counts.copy()       # severed ledger
    vc_on = recv.value_counts + cue_extra   # on-ledger (severed + channel increments)

    tot_sev = vc_sev.sum()
    tot_on = vc_on.sum()

    share_sev_cue = float(vc_sev[CUE_COLOR] / tot_sev) if tot_sev > 0 else 0.0
    share_on_cue = float(vc_on[CUE_COLOR] / tot_on) if tot_on > 0 else 0.0
    divergence = share_on_cue - share_sev_cue

    fav_sev = int(np.argmax(vc_sev))
    fav_on = int(np.argmax(vc_on))

    return {
        "events": events,
        "share_sev_cue": share_sev_cue,
        "share_on_cue": share_on_cue,
        "divergence": divergence,
        "fav_sev": fav_sev,
        "fav_on": fav_on,
        "vc_sev": vc_sev,
        "vc_on": vc_on,
    }


# ---------------------------------------------------------------------------
# Step 3: equilibrate both arms for each seed, then run dyad
# ---------------------------------------------------------------------------
# Arms: "windowed" (do_decay=True) and "control" (do_decay=False).
# Both arms equilibrate from the same deepcopy of recovered with the same rng path
# (decay consumes no rng draws), so positions and pA are byte-identical at dyad start.

ARM_NAMES = ["windowed", "control"]
ARM_DECAY = {"windowed": True, "control": False}

# Storage: arm -> seed -> result
arm_results: dict[str, dict] = {"windowed": {}, "control": {}}
arm_predyad: dict[str, dict] = {"windowed": {}, "control": {}}

print("=" * 78)
print("EQUILIBRATION + DYAD RUNS")
print("=" * 78)

for arm in ARM_NAMES:
    do_decay = ARM_DECAY[arm]
    print(f"\n-- arm: {arm}  (do_decay={do_decay}  EQUIL={EQUIL}  DYAD={DYAD}) --")
    print(
        f"  {'seed':>4}  {'pre_mass':>9}  {'pre_fav':>7}  {'G1':>4}  "
        f"{'events':>7}  {'share_sev':>9}  {'share_on':>8}  {'divergence':>10}  "
        f"{'fav_sev':>7}  {'fav_on':>6}"
    )
    print("  " + "-" * 90)

    for seed in SEEDS:
        # Deep-copy the recovered snapshot for equilibration
        recv = copy.deepcopy(recovered)

        # Equilibrate EQUIL steps (decay on/off per arm; rng derived from rng_counter)
        run_steps_single(recv, EQUIL, do_decay=do_decay)

        pre_mass = float(recv.value_counts.sum())
        pre_fav = recv.favorite()
        g1_ok = (pre_fav != CUE_COLOR)
        arm_predyad[arm][seed] = {
            "pre_mass": pre_mass,
            "pre_fav": pre_fav,
            "g1_ok": g1_ok,
        }

        if not g1_ok:
            print(
                f"  {seed:>4}   {pre_mass:>8.1f}   {pre_fav:>6}   EXCL  "
                f"{'---':>7}  {'---':>9}  {'---':>8}  {'---':>10}  {'---':>7}  {'---':>6}"
            )
            continue

        # Dyad: deep-copy emitter; run paired stepper
        em = copy.deepcopy(em0)
        result = run_dyad(recv, em, seed, do_decay=do_decay)
        arm_results[arm][seed] = result

        # G2-style event gate: warn if below MIN_EVENTS (exclude from P-checks if so)
        ev_ok = result["events"] >= MIN_EVENTS
        ev_tag = f"{result['events']}" if ev_ok else f"{result['events']}(LOW)"

        print(
            f"  {seed:>4}   {pre_mass:>8.1f}   {pre_fav:>6}   {'OK':>4}  "
            f"{ev_tag:>7}  {result['share_sev_cue']:>9.4f}  {result['share_on_cue']:>8.4f}  "
            f"{result['divergence']:>10.4f}  {result['fav_sev']:>7}  {result['fav_on']:>6}"
        )

print()

# ---------------------------------------------------------------------------
# G2-style event gate (per arm): exclude seeds with < MIN_EVENTS
# Report but do not sys.exit (the gate is informational; expunge from P-counts)
# ---------------------------------------------------------------------------

print("--- EVENT GATE (G2-style: events >= MIN_EVENTS per seed) ---")
for arm in ARM_NAMES:
    low_ev = []
    for seed in SEEDS:
        if seed not in arm_results[arm]:
            continue  # G1-excluded
        ev = arm_results[arm][seed]["events"]
        if ev < MIN_EVENTS:
            low_ev.append(seed)
    if low_ev:
        print(f"  {arm}: seeds {low_ev} have < {MIN_EVENTS} events — excluded from P-checks")
    else:
        print(f"  {arm}: all active seeds >= {MIN_EVENTS} events")
print()


def valid_seeds_for_arm(arm: str) -> list[int]:
    """Return seeds that passed G1 AND G2-style event gate for this arm."""
    out = []
    for seed in SEEDS:
        if not arm_predyad[arm][seed]["g1_ok"]:
            continue
        if seed not in arm_results[arm]:
            continue
        if arm_results[arm][seed]["events"] < MIN_EVENTS:
            continue
        out.append(seed)
    return out


valid_windowed = valid_seeds_for_arm("windowed")
valid_control = valid_seeds_for_arm("control")

print(f"Valid seeds — windowed: {valid_windowed}  (n={len(valid_windowed)})")
print(f"Valid seeds — control:  {valid_control}  (n={len(valid_control)})")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

checks: list[tuple[str, bool, str]] = []


def check(name: str, predicate_fn) -> None:
    try:
        passed, detail = predicate_fn()
        checks.append((name, passed, detail))
    except Exception as exc:
        checks.append((name, False, f"exception: {exc}"))


def _p1():
    """Windowed-arm divergence >= THRESH in >= 6/8 valid seeds."""
    if len(valid_windowed) < MIN_VALID:
        return False, f"too few valid seeds ({len(valid_windowed)} < {MIN_VALID})"
    pass_seeds = []
    seed_vals = []
    for seed in valid_windowed:
        d = arm_results["windowed"][seed]["divergence"]
        ok = d >= THRESH
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:{d:.4f}({'ok' if ok else 'FAIL'})")
    n_pass = sum(pass_seeds)
    return n_pass >= MIN_VALID, f"passes={n_pass}/{len(valid_windowed)}  [{'; '.join(seed_vals)}]"


def _p2():
    """Control-arm divergence < THRESH in >= 6/8 valid seeds (wall replicates)."""
    if len(valid_control) < MIN_VALID:
        return False, f"too few valid seeds ({len(valid_control)} < {MIN_VALID})"
    pass_seeds = []
    seed_vals = []
    for seed in valid_control:
        d = arm_results["control"][seed]["divergence"]
        ok = d < THRESH
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:{d:.4f}({'ok' if ok else 'FAIL'})")
    n_pass = sum(pass_seeds)
    return n_pass >= MIN_VALID, f"passes={n_pass}/{len(valid_control)}  [{'; '.join(seed_vals)}]"


def _p3():
    """Windowed-arm ON-ledger favorite==1 AND severed favorite!=1, in >= 4/8 valid seeds."""
    if not valid_windowed:
        return False, "no valid windowed seeds"
    pass_seeds = []
    seed_vals = []
    for seed in valid_windowed:
        r = arm_results["windowed"][seed]
        ok = (r["fav_on"] == CUE_COLOR) and (r["fav_sev"] != CUE_COLOR)
        pass_seeds.append(ok)
        seed_vals.append(
            f"s{seed}:on={r['fav_on']},sev={r['fav_sev']}({'ok' if ok else 'no'})"
        )
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/{len(valid_windowed)}  [{'; '.join(seed_vals)}]"


check("P1-wall-opens-windowed-div>=0.02-in-6/8", _p1)
check("P2-wall-stands-control-div<0.02-in-6/8", _p2)
check("P3-adult-adoption-on-fav==1-AND-sev-fav!=1-in-4/8-LOW-confidence", _p3)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
failed_names: list[str] = []
for name, passed, detail in checks:
    verdict = "PASS" if passed else "FAIL"
    print(f"  {verdict}  {name}: {detail}")
    if not passed:
        failed_names.append(name)

print()

# ---------------------------------------------------------------------------
# Falsifier map
# ---------------------------------------------------------------------------

p1_passed = checks[0][1]
p2_passed = checks[1][1]
p3_passed = checks[2][1]

print("--- FALSIFIER MAP ---")
f1_fired = not p1_passed
f2_fired = not p2_passed

if f1_fired:
    print(
        "  F1 FIRED: P1 failed -> the window does NOT open the social door; "
        "receiver mass is not the whole wall (rung 4a NEGATIVE)."
    )
else:
    print("  F1 did not fire (P1 PASS — windowed arm clears the 0.02 bar).")

if f2_fired:
    print(
        "  F2 FIRED: P2 failed -> Exp 65's wall does not replicate here; "
        "comparability broken — diagnose before any claim."
    )
else:
    print("  F2 did not fire (P2 PASS — control arm wall replicates Exp 65's negative).")

if p3_passed:
    print("  P3 PASS (LOW confidence): adult adoption observed in windowed arm.")
else:
    print("  P3 did not reach >= 4/8 threshold (LOW confidence; not a falsifier).")

print()

# ---------------------------------------------------------------------------
# Diagnostics: per-arm per-seed detail table
# ---------------------------------------------------------------------------

print("--- DIAGNOSTICS ---")
print()

for arm in ARM_NAMES:
    do_decay = ARM_DECAY[arm]
    print(f"  arm={arm}  (do_decay={do_decay}):")
    print(
        f"    {'seed':>4}  {'pre_mass':>9}  {'pre_fav':>7}  {'G1':>4}  "
        f"{'events':>7}  {'share_sev':>9}  {'share_on':>8}  {'divergence':>10}  "
        f"{'fav_sev':>7}  {'fav_on':>6}"
    )
    print("    " + "-" * 88)
    for seed in SEEDS:
        pd = arm_predyad[arm][seed]
        if not pd["g1_ok"]:
            print(
                f"    {seed:>4}   {pd['pre_mass']:>8.1f}   {pd['pre_fav']:>6}   EXCL  "
                f"{'---':>7}  {'---':>9}  {'---':>8}  {'---':>10}  {'---':>7}  {'---':>6}"
            )
            continue
        if seed not in arm_results[arm]:
            print(f"    {seed:>4}   {pd['pre_mass']:>8.1f}   {pd['pre_fav']:>6}   OK    (no dyad result)")
            continue
        r = arm_results[arm][seed]
        ev_tag = f"{r['events']}" + ("" if r["events"] >= MIN_EVENTS else "(LOW)")
        print(
            f"    {seed:>4}   {pd['pre_mass']:>8.1f}   {pd['pre_fav']:>6}   OK    "
            f"{ev_tag:>7}  {r['share_sev_cue']:>9.4f}  {r['share_on_cue']:>8.4f}  "
            f"{r['divergence']:>10.4f}  {r['fav_sev']:>7}  {r['fav_on']:>6}"
        )
    print()

# Per-arm per-seed value shares for all colors
n_colors = recovered.world.n_colors
print("  Value shares all colors (end of dyad):")
for arm in ARM_NAMES:
    print(f"    arm={arm}:")
    hdr = f"    {'seed':>4}  " + "  ".join(
        f"sev[{c}]  on[{c}]" for c in range(n_colors)
    )
    print("  " + hdr)
    for seed in SEEDS:
        if not arm_predyad[arm][seed]["g1_ok"] or seed not in arm_results[arm]:
            continue
        r = arm_results[arm][seed]
        vc_sev = r["vc_sev"]
        vc_on = r["vc_on"]
        tot_sev = vc_sev.sum()
        tot_on = vc_on.sum()
        vals = "  ".join(
            f"{vc_sev[c]/tot_sev:.4f}  {vc_on[c]/tot_on:.4f}" for c in range(n_colors)
        )
        print(f"      {seed:>2}   {vals}")
    print()

# ---------------------------------------------------------------------------
# Final verdict
# ---------------------------------------------------------------------------

if not f1_fired and not f2_fired:
    print(
        "EXP89: THE WALL OPENS (and stands without the window) — "
        "adults are reachable at LV=0.999"
    )
elif f1_fired and not f2_fired:
    print("EXP89: F1 — window does not open the door")
elif not f1_fired and f2_fired:
    print("EXP89: F2 — wall fails to replicate; diagnose")
else:
    print("EXP89: F1 — window does not open the door; F2 — wall fails to replicate; diagnose")
