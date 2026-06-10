"""Exp 65 — clade rung 3: social transmission of value over a grounded cue channel.

Social-emergence direction, rung 3 (loop/directions/social-emergence.md): one clade-mate
(the emitter A) emits a cue carrying its self-formed favorite color; the cue enters the
receiver B's value stream as the M4 extrinsic channel (docs/specs/m4-affective-dyad.md §3),
GROUNDED in B's own intrinsic predictability rather than injected as labeled reward.
Tested against a channel-SEVERED twin of B with exactly identical trajectories.

Hypothesis: a one-way, proximity-gated cue channel carrying A's self-formed favorite,
gated by B's intrinsic predictability weight, produces a measurable value-share shift
in B toward the cued color, relative to the exactly-matched severed twin.

Phase 1 (emitter speciation, the Exp 26 logic): fork mirro, raise the fork 2000 steps in
an all-color-1 world; its favorite must shift to color 1 by lived history (mirro and vela
both currently favor color 2, so an un-speciated emitter would make transmission
unmeasurable).

Setup gates (instrument validity, not verdicts — Exp 61/64 lesson):
  G1 emitter favorite after speciation != receiver favorite (else INVALID, redesign).
  G2 >=50 proximity cue events per dyad run (else INVALID, redesign).
Predeclared predictions:
  P1 (primary, magnitude): cue-color value-share divergence share_on(cue) - share_sev(cue)
     >= 0.02 in >=4/5 seeds.
  P2 (sign): divergence > 0 in 5/5 seeds.
Falsifiers:
  F1 = P1 fails -> rung 3 NEGATIVE: social transmission adds nothing measurable at this
     scale/range; logged as a real negative.
  F2 = P2 fails in any seed that passed G2 -> wiring or gating bug; HALT for investigation
     before any verdict.
Predicted magnitudes (stated before running): ~300-400 proximity events per 2000-step dyad,
intrinsic gate weight 0.5-1.0, divergence ~0.02-0.03. Predicted NO favorite flip in B
(flip needs ~650 net counts on color 1 vs color 2; expected cue mass ~235) — the flip is
a DIAGNOSTIC readout, not a falsifier, either way.

Exactness note (declared): value counts are epiphenomenal to dynamics in this substrate —
the walk policy is random and value_counts never enter qs/pA updates — so the severed twin
is computed in the SAME pass as a dual value ledger; trajectories are identical by
construction, and the on/severed divergence is exactly the channel's contribution.

Provided priors declared: the emission rule (emit current favorite when Manhattan distance
<= 1); the reception rule (B.value_counts[cue] += B's own exp(-H) predictability weight,
the live() intrinsic-valence gate — the toy implementation of M4 §3's grounding); the
speciation world (5x5 all color 1); the shared dyad world (mirro's grid); the random-walk
policy; rung-2's other-agent-here modality is deliberately OMITTED (one mechanism per
iteration; Exp 64 showed it perceptually inert). Self-formed content: WHICH color A
emits (its favorite from its divergent lived history); the gate values (B's own beliefs).
The spines never live: forks only; neither committed line is saved.
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
SEEDS = [0, 1, 2, 3, 4]
PROX = 1  # Manhattan distance gate
MIRRO_DIR = Path("creature/state/mirro")
VELA_DIR = Path("creature/state/vela")
P1_THRESH = 0.02
MIN_EVENTS = 50

# ---------------------------------------------------------------------------
# Load committed spines (read-only — NEVER call .live() or .save() on these)
# ---------------------------------------------------------------------------

print("Exp 65 — clade rung 3: social transmission of value over a grounded cue channel")
print()

mirro = Creature.load(MIRRO_DIR)
vela = Creature.load(VELA_DIR)

vc_m = mirro.value_counts
tot_m = vc_m.sum()
vc_v = vela.value_counts
tot_v = vc_v.sum()

print(f"mirro: name={mirro.name!r} age={mirro.age_steps} "
      f"hash={mirro._state_hash()[:12]} favorite={mirro.favorite()} "
      f"value_shares={np.round(vc_m / tot_m, 4)}")
print(f"vela:  name={vela.name!r}  age={vela.age_steps}  "
      f"hash={vela._state_hash()[:12]} favorite={vela.favorite()} "
      f"value_shares={np.round(vc_v / tot_v, 4)}")
print()

# ---------------------------------------------------------------------------
# Phase 1: emitter speciation
# Fork mirro; raise 2000 steps in an all-color-1 world.
# ---------------------------------------------------------------------------

print("--- PHASE 1: EMITTER SPECIATION ---")

emitter = mirro.fork("exp65-emitter")

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

# Gate G1: emitter favorite must differ from receiver (vela) favorite
if emitter.favorite() == vela.favorite():
    print(f"G1 FAIL — RUN INVALID (emitter favorite == receiver favorite == {vela.favorite()})")
    sys.exit(1)
print(f"G1 PASS: emitter favorite={emitter.favorite()} != vela favorite={vela.favorite()}")
print()

CUE_COLOR = emitter.favorite()  # the speciated color being transmitted (expected: 1)
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
# Phase 2: dyad stepper across seeds
# ---------------------------------------------------------------------------

print("--- PHASE 2: DYAD RUNS ---")
print()

# Make receiver template once (fork appends one biography event to vela — append-only, allowed)
recv_t = vela.fork("exp65-receiver-template")
recv_t.world = mirro.world  # shared grid (mirro's world)
# keep recv_t.true_pos from vela's committed state

# Storage
seed_results = {}

print(
    f"{'seed':>4}  {'events':>7}  {'mean_gate_w':>11}  "
    f"{'share_sev[cue]':>14}  {'share_on[cue]':>13}  {'divergence':>10}  "
    f"{'fav_sev':>7}  {'fav_on':>6}  {'em_fav_end':>10}"
)
print("-" * 100)

for seed in SEEDS:
    # Deep-copy both emitter and receiver for this seed
    em = copy.deepcopy(emitter)
    em.world = mirro.world  # run dyad in mirro's world
    # em.true_pos inherited from speciation-phase emitter (position 0 after speciation walk)

    recv = copy.deepcopy(recv_t)

    n_colors = mirro.world.n_colors
    world = mirro.world
    n_cells = world.n_cells
    B = world.transition_matrix()  # (n_cells, n_cells, 4)

    # Action RNGs: index 0 = emitter, 1 = receiver
    rng_em = np.random.default_rng(200000 + 1000 * seed + 0)
    rng_recv = np.random.default_rng(200000 + 1000 * seed + 1)

    # Dual ledger: recv.value_counts is the SEVERED ledger (natural live()-mechanism only)
    # cue_extra accumulates ONLY the channel increments
    cue_extra = np.zeros(n_colors)

    events = 0
    gate_weight_sum = 0.0
    cue_color_counts = {}  # track which cue colors were emitted

    for step in range(DYAD_STEPS):
        # Read simultaneous start-of-step positions
        pos_em = em.true_pos
        pos_recv = recv.true_pos

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
            gate_weight_sum += w_recv
            cue_color_counts[cue] = cue_color_counts.get(cue, 0) + 1

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
    em_fav_end = int(np.argmax(em.value_counts))  # this seed's dyad-end emitter favorite

    mean_gate_w = gate_weight_sum / events if events > 0 else 0.0

    modal_cue = max(cue_color_counts, key=cue_color_counts.get) if cue_color_counts else -1

    seed_results[seed] = {
        "events": events,
        "mean_gate_w": mean_gate_w,
        "share_sev_cue": share_sev_cue,
        "share_on_cue": share_on_cue,
        "divergence": divergence,
        "fav_sev": fav_sev,
        "fav_on": fav_on,
        "em_fav_end": em_fav_end,
        "modal_cue": modal_cue,
        "cue_color_counts": cue_color_counts,
        "vc_sev": vc_sev,
        "vc_on": vc_on,
    }

    print(
        f"  {seed:>2}   {events:>6}   {mean_gate_w:>10.4f}   "
        f"{share_sev_cue:>13.4f}   {share_on_cue:>12.4f}   {divergence:>9.4f}   "
        f"{fav_sev:>6}   {fav_on:>5}   {em_fav_end:>9}"
    )

print()

# ---------------------------------------------------------------------------
# Gate G2: >= MIN_EVENTS proximity cue events per dyad run, all seeds
# ---------------------------------------------------------------------------

print("--- VALIDITY GATE G2 ---")
g2_ok = True
for seed in SEEDS:
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

print(f"  G2 PASS: all seeds >= {MIN_EVENTS} events")
print()

# ---------------------------------------------------------------------------
# Property checks
# ---------------------------------------------------------------------------

def _p1():
    pass_seeds = []
    seed_vals = []
    for seed in SEEDS:
        d = seed_results[seed]["divergence"]
        ok = d >= P1_THRESH
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:{d:.4f}({'ok' if ok else 'FAIL'})")
    n_pass = sum(pass_seeds)
    return n_pass >= 4, f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"


def _p2():
    pass_seeds = []
    fail_g2_passed = []
    seed_vals = []
    for seed in SEEDS:
        d = seed_results[seed]["divergence"]
        ok = d > 0
        pass_seeds.append(ok)
        seed_vals.append(f"s{seed}:{d:.4f}({'ok' if ok else 'FAIL'})")
        if not ok:
            fail_g2_passed.append(seed)
    n_pass = sum(pass_seeds)
    detail = f"passes={n_pass}/5  [{'; '.join(seed_vals)}]"
    if fail_g2_passed:
        detail += f"  WARNING: negative divergence in seeds {fail_g2_passed} (all passed G2)"
    return n_pass == 5, detail


check("P1-magnitude-divergence>=0.02-in-4/5", _p1)
check("P2-sign-divergence>0-in-5/5", _p2)

# ---------------------------------------------------------------------------
# Print property check results
# ---------------------------------------------------------------------------

print("--- PROPERTY CHECKS ---")
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

if not p1_passed:
    print("F1 FIRED: rung 3 NEGATIVE — social transmission adds nothing measurable at this scale")
if not p2_passed:
    print("F2 FIRED: wiring/gating bug — HALT for investigation")

if p1_passed and p2_passed:
    print("  No falsifiers fired.")

print()

# ---------------------------------------------------------------------------
# Diagnostics: per-seed flip table, emitter favorite, gate weights, modal cue
# ---------------------------------------------------------------------------

print("--- DIAGNOSTICS (not falsifiers) ---")
print()
print(f"{'seed':>4}  {'fav_sev':>7}  {'fav_on':>6}  {'flip_sev':>8}  {'flip_on':>7}  "
      f"{'em_fav_end':>10}  {'modal_cue':>9}  {'mean_gate_w':>11}")
print("-" * 80)

recv_base_fav = vela.favorite()  # receiver's baseline favorite (from committed state)

for seed in SEEDS:
    r = seed_results[seed]
    flip_sev = "YES" if r["fav_sev"] != recv_base_fav else "no"
    flip_on = "YES" if r["fav_on"] != recv_base_fav else "no"
    print(
        f"  {seed:>2}   {r['fav_sev']:>6}   {r['fav_on']:>5}   {flip_sev:>8}   {flip_on:>7}   "
        f"{r['em_fav_end']:>9}   {r['modal_cue']:>8}   {r['mean_gate_w']:>10.4f}"
    )

print()
print(f"receiver baseline favorite (vela committed): {recv_base_fav}")
print()

# Per-seed cue color distribution
print("--- CUE COLOR DISTRIBUTION PER SEED ---")
for seed in SEEDS:
    r = seed_results[seed]
    total_ev = r["events"]
    cc = r["cue_color_counts"]
    dist_str = "  ".join(
        f"color{c}:{cnt}({100*cnt/total_ev:.1f}%)" for c, cnt in sorted(cc.items())
    )
    print(f"  seed={seed}: {dist_str}  (total={total_ev})")

print()

# Per-seed severed vs on value shares for all colors
print("--- VALUE SHARES ALL COLORS (end of dyad) ---")
n_colors = mirro.world.n_colors
header = f"{'seed':>4}  " + "  ".join(
    f"sev[{c}]" + " " * 2 + f"on[{c}]" for c in range(n_colors)
)
print("  " + header)
for seed in SEEDS:
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

if not failed_names:
    print("RUNG 3: PASS")
else:
    print(f"RUNG 3: FAIL {failed_names}")
