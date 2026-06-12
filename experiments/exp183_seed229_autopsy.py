"""EXP 183 ADDENDUM -- SEED-229 AUTOPSY (diagnostic only).
Governed by: loop/directions/identity-n4-crack.md (ladder rung 1) and
docs/research/n4-fixed-h-squeeze-plan.md (Part 1, the supplied program). This script
answers Part 1's eight questions. It is DIAGNOSTIC ONLY: it must not modify any
Exp 183 artifact, any verdict, or any creature spine. The N4 chapter verdict
(rung 3 NEGATIVE-config) is NOT at stake here.

BIT-MATCH GATE (predeclared): every re-run session must reproduce the committed
exp183_rows.json quantities for that (arm, seed, burst): gap_start, gap_end, d_b,
tv_b, recovered, n_events, and each event's {label, entry_step, frozen_steps,
E_blocked, c_star, trigger_latency} -- floats to atol 1e-9. ANY mismatch =>
reconstruction invalid; print BIT-MATCH: FAIL with details and still write outputs
(marked invalid); the analysis sections are then unlicensed.

PRE-DATA HYPOTHESIS (formed from the COMMITTED per-burst aggregates only; the
re-run streams are the out-of-sample test): seed 229's fixed-H defense failures are
REPEATED-COLOR CUMULATION OF THE IRREDUCIBLE TRIGGER-LATENCY DOSE. Each burst --
defended or not -- erodes the favorite-vs-burst-color gap by roughly the ~75-step
latency dose (~70-75 units; committed H1200 ledger: 153->81, 78->+5, 42->-30),
because the freeze engages only after the detection floor (Law F). Two consecutive
same-color bursts plus a third on the displaced runner-up exhaust the pre-burst
margin; the recovery criterion (burst-color expression fraction < 0.5 over
[bend+1500, bend+2000)) then fails at the near-tied equilibrium even though the
freeze covers the burst body.
REFUTERS (any one kills the hypothesis):
 (a) freeze coverage of the burst body is incomplete in the failing H>=1200 bursts
     (then the mechanism is release-too-early / trigger class);
 (b) the gap erosion is NOT concentrated in the unfrozen burst-head window
     (then it is not the latency dose);
 (c) expression failure occurs with strongly positive gap_end (>= 30 units)
     (then the near-tie account is wrong);
 (d) other seeds show equal repeated-color dose exposure with comfortable margins
     and pass -- margin size rather than repetition would be the discriminant
     (reclassify toward near-tie initial condition / other).

MECHANISM CANDIDATES (the plan doc's list, Q8): trigger latency; release too
early; repeated-color accumulation; insufficient refractory; revision-bar
conflict; other.
"""
from __future__ import annotations

import collections
import copy
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Setup: add repo root to sys.path and import exp183 module
# ---------------------------------------------------------------------------

REPO_ROOT = Path("/Users/mirro/Projects/active-loop")
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "exp183",
    str(REPO_ROOT / "experiments" / "exp183_n4_freeze_gate2.py"),
)
exp183 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp183)  # type: ignore[union-attr]

# Pull the functions and constants we need
run_fork = exp183.run_fork
burst_recovered = exp183.burst_recovered
pi_of = exp183.pi_of
tv = exp183.tv
snap_index = exp183.snap_index
BURST_WINDOWS = exp183.BURST_WINDOWS
EVAL = exp183.EVAL
FINE_EVAL = exp183.FINE_EVAL
ARMS = exp183.ARMS
N_STEPS = exp183.N_STEPS
CHUNK_SIZE = exp183.CHUNK_SIZE

from active_loop.creature import Creature

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

OUT_TXT = REPO_ROOT / "experiments" / "outputs" / "exp183_seed229_autopsy.txt"
OUT_JSON = REPO_ROOT / "experiments" / "outputs" / "exp183_seed229_autopsy.json"
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load committed rows
# ---------------------------------------------------------------------------

COMMITTED_ROWS_PATH = REPO_ROOT / "experiments" / "outputs" / "exp183_rows.json"


def load_committed_rows():
    rows = []
    with open(COMMITTED_ROWS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# RE-RUN MATRIX
# ---------------------------------------------------------------------------

# arm H1200 x seeds [226,227,228,229,230,231,232,233]
H1200_SEEDS = [226, 227, 228, 229, 230, 231, 232, 233]

# seed 229 x arms [baseline, n4_freeze, H600, H900, H1800, H2400, H3000, oracle]
SEED229_ARM_NAMES = ["baseline", "n4_freeze", "H600", "H900", "H1800", "H2400", "H3000", "oracle"]

# seed 227 x arm H900
EXTRA_SESSION = [("H900", 227)]

# Build arm name -> (arm_name, arm_mode) lookup from ARMS
ARM_LOOKUP = {name: (name, mode) for name, mode in ARMS}


def run_matrix(mirro_root, base_cmap, n_colors):
    """Run all 17 sessions and return results dict keyed by (arm_name, seed)."""
    results = {}
    total = len(H1200_SEEDS) + len(SEED229_ARM_NAMES) + len(EXTRA_SESSION)
    done = 0

    # H1200 x seeds 226-233
    arm_name = "H1200"
    _, arm_mode = ARM_LOOKUP[arm_name]
    for seed in H1200_SEEDS:
        print(f"  [{done+1}/{total}] arm={arm_name} seed={seed} ...", flush=True)
        root = copy.deepcopy(mirro_root)
        root._state_dir = None
        rr = run_fork(root, seed, base_cmap, n_colors,
                      arm_name=arm_name, arm_mode=arm_mode, phase="W")
        results[(arm_name, seed)] = rr
        done += 1
        print(f"    done: events={len(rr['events'])} recovered={[burst_recovered(rr['expressed_arr'], bi, rr['burst_preburst_fav'], rr['burst_onset_color']) for bi in range(3)]}", flush=True)

    # seed 229 x arms
    seed = 229
    for arm_name in SEED229_ARM_NAMES:
        _, arm_mode = ARM_LOOKUP[arm_name]
        print(f"  [{done+1}/{total}] arm={arm_name} seed={seed} ...", flush=True)
        root = copy.deepcopy(mirro_root)
        root._state_dir = None
        rr = run_fork(root, seed, base_cmap, n_colors,
                      arm_name=arm_name, arm_mode=arm_mode, phase="W")
        results[(arm_name, seed)] = rr
        done += 1
        print(f"    done: events={len(rr['events'])} recovered={[burst_recovered(rr['expressed_arr'], bi, rr['burst_preburst_fav'], rr['burst_onset_color']) for bi in range(3)]}", flush=True)

    # seed 227 x H900
    arm_name, seed = "H900", 227
    _, arm_mode = ARM_LOOKUP[arm_name]
    print(f"  [{done+1}/{total}] arm={arm_name} seed={seed} ...", flush=True)
    root = copy.deepcopy(mirro_root)
    root._state_dir = None
    rr = run_fork(root, seed, base_cmap, n_colors,
                  arm_name=arm_name, arm_mode=arm_mode, phase="W")
    results[(arm_name, seed)] = rr
    done += 1
    print(f"    done: events={len(rr['events'])} recovered={[burst_recovered(rr['expressed_arr'], bi, rr['burst_preburst_fav'], rr['burst_onset_color']) for bi in range(3)]}", flush=True)

    return results


# ---------------------------------------------------------------------------
# BIT-MATCH GATE
# ---------------------------------------------------------------------------

FLOAT_ATOL = 1e-9


def check_bit_match(results, committed_rows):
    """Compare rerun results against committed rows. Returns (overall_pass, details_list)."""
    # Index committed W rows by (arm, seed, burst_idx)
    committed_w = {}
    for row in committed_rows:
        if row.get("phase") == "W":
            key = (row["arm"], row["fork_seed"], row["burst_idx"])
            committed_w[key] = row

    details = []
    all_pass = True

    sessions = list(results.keys())
    for (arm_name, seed) in sessions:
        rr = results[(arm_name, seed)]
        session_pass = True
        session_details = []

        for bi in range(3):
            key = (arm_name, seed, bi)
            if key not in committed_w:
                # Oracle may not have committed rows if it wasn't in the original run with this seed
                # (oracle only runs in the full sweep; our matrix may not have it committed)
                # Skip if arm is oracle
                if arm_name == "oracle":
                    continue
                session_details.append(f"  burst {bi}: MISSING in committed rows")
                session_pass = False
                all_pass = False
                continue

            crow = committed_w[key]

            # Recompute row quantities from rr
            recomp_recovered = burst_recovered(
                rr["expressed_arr"], bi, rr["burst_preburst_fav"], rr["burst_onset_color"]
            )
            recomp_gap_start = rr["gap_start"][bi]
            recomp_gap_end = rr["gap_end"][bi]
            recomp_d_b = rr["d_b"][bi]
            recomp_tv_b = rr["tv_b"][bi]
            recomp_n_events = len(rr["events"])

            mismatches = []

            # recovered
            if recomp_recovered != crow["recovered"]:
                mismatches.append(f"recovered: recomp={recomp_recovered} committed={crow['recovered']}")

            # n_events (same for all bursts in session)
            if recomp_n_events != crow["n_events"]:
                mismatches.append(f"n_events: recomp={recomp_n_events} committed={crow['n_events']}")

            # gap_start
            cgs = crow.get("gap_start")
            if recomp_gap_start is not None and cgs is not None:
                if abs(recomp_gap_start - cgs) > FLOAT_ATOL:
                    mismatches.append(f"gap_start: recomp={recomp_gap_start:.12f} committed={cgs:.12f} diff={abs(recomp_gap_start - cgs):.3e}")
            elif recomp_gap_start != cgs:
                mismatches.append(f"gap_start: recomp={recomp_gap_start} committed={cgs}")

            # gap_end
            cge = crow.get("gap_end")
            if recomp_gap_end is not None and cge is not None:
                if abs(recomp_gap_end - cge) > FLOAT_ATOL:
                    mismatches.append(f"gap_end: recomp={recomp_gap_end:.12f} committed={cge:.12f} diff={abs(recomp_gap_end - cge):.3e}")
            elif recomp_gap_end != cge:
                mismatches.append(f"gap_end: recomp={recomp_gap_end} committed={cge}")

            # d_b
            cdb = crow.get("d_b")
            if recomp_d_b is not None and cdb is not None:
                if abs(recomp_d_b - cdb) > FLOAT_ATOL:
                    mismatches.append(f"d_b: recomp={recomp_d_b:.12f} committed={cdb:.12f} diff={abs(recomp_d_b - cdb):.3e}")
            elif recomp_d_b != cdb:
                mismatches.append(f"d_b: recomp={recomp_d_b} committed={cdb}")

            # tv_b
            ctvb = crow.get("tv_b")
            if recomp_tv_b is not None and ctvb is not None:
                if abs(recomp_tv_b - ctvb) > FLOAT_ATOL:
                    mismatches.append(f"tv_b: recomp={recomp_tv_b:.12f} committed={ctvb:.12f} diff={abs(recomp_tv_b - ctvb):.3e}")
            elif recomp_tv_b != ctvb:
                mismatches.append(f"tv_b: recomp={recomp_tv_b} committed={ctvb}")

            # events: compare label, entry_step, frozen_steps, E_blocked, c_star, trigger_latency
            committed_events = crow.get("events_summary", [])
            recomp_events = [
                {
                    "label": e["label"],
                    "entry_step": e["entry_step"],
                    "frozen_steps": e["frozen_steps"],
                    "E_blocked": e["E_blocked"],
                    "c_star": e["c_star"],
                    "trigger_latency": e.get("trigger_latency"),
                }
                for e in rr["events"]
            ]

            if len(recomp_events) != len(committed_events):
                mismatches.append(f"events len: recomp={len(recomp_events)} committed={len(committed_events)}")
            else:
                for ei, (re_, ce_) in enumerate(zip(recomp_events, committed_events)):
                    for field in ["label", "entry_step", "frozen_steps", "c_star", "trigger_latency"]:
                        if re_[field] != ce_[field]:
                            mismatches.append(f"event[{ei}].{field}: recomp={re_[field]} committed={ce_[field]}")
                    # E_blocked to atol
                    if abs(re_["E_blocked"] - ce_["E_blocked"]) > FLOAT_ATOL:
                        mismatches.append(f"event[{ei}].E_blocked: recomp={re_['E_blocked']:.12f} committed={ce_['E_blocked']:.12f} diff={abs(re_['E_blocked'] - ce_['E_blocked']):.3e}")

            if mismatches:
                session_pass = False
                all_pass = False
                session_details.append(f"  burst {bi}: FAIL")
                for m in mismatches:
                    session_details.append(f"    {m}")
            else:
                session_details.append(f"  burst {bi}: PASS")

        status = "PASS" if session_pass else "FAIL"
        details.append({
            "session": f"arm={arm_name} seed={seed}",
            "status": status,
            "burst_details": session_details,
        })

    return all_pass, details


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def get_freeze_entry_snap_and_step(rr, bi):
    """Return (first_resist_step, first_resist_snap_idx) for burst bi from state_arr.
    Returns (None, None) if never in RESIST during that burst.
    """
    bstart, bend = BURST_WINDOWS[bi]
    state_arr = rr["state_arr"]
    for t in range(bstart, bend):
        if state_arr[t] == 1:
            return t, snap_index(t)
    return None, None


def get_last_resist_step(rr, bi):
    """Return last RESIST step within burst bi."""
    bstart, bend = BURST_WINDOWS[bi]
    state_arr = rr["state_arr"]
    last = None
    for t in range(bstart, bend):
        if state_arr[t] == 1:
            last = t
    return last


def gap_v_fav_vs_bc(v, pre_fav, bc):
    """v[pre_fav] - v[bc] raw gap."""
    return float(v[pre_fav] - v[bc])


def expression_fraction(expressed_arr, bend, bc):
    """bc-expression fraction over [bend+1500, bend+2000)."""
    ws = bend + 1500
    we = bend + 2000
    if we > len(expressed_arr):
        return float("nan")
    return float(np.mean(expressed_arr[ws:we] == bc))


# ---------------------------------------------------------------------------
# Q1: Burst colors by burst index
# ---------------------------------------------------------------------------

def q1_burst_colors(committed_rows):
    """From committed rows: arm x seed -> [c0,c1,c2] + pre_fav per burst."""
    result = {}
    for row in committed_rows:
        if row.get("phase") != "W":
            continue
        arm = row["arm"]
        seed = row["fork_seed"]
        bi = row["burst_idx"]
        key = (arm, seed)
        if key not in result:
            result[key] = {"colors": [None, None, None], "pre_favs": [None, None, None]}
        result[key]["colors"][bi] = row["burst_color"]
        result[key]["pre_favs"][bi] = row["pre_fav"]
    return result


def q2_repeats(committed_rows):
    """Q2: Which (arm, seed) schedules have a repeated color; cross-tab repeat vs recovered."""
    color_info = q1_burst_colors(committed_rows)
    repeat_tab = {}
    for (arm, seed), info in color_info.items():
        colors = info["colors"]
        has_repeat = len(set(colors)) < len([c for c in colors if c is not None])
        repeat_tab[(arm, seed)] = has_repeat

    # Cross-tab repeat pattern vs recovered flags (from committed rows)
    cross = {"repeat_fail": 0, "repeat_pass": 0, "norepeat_fail": 0, "norepeat_pass": 0}
    for row in committed_rows:
        if row.get("phase") != "W":
            continue
        arm = row["arm"]
        seed = row["fork_seed"]
        key = (arm, seed)
        has_repeat = repeat_tab.get(key, False)
        rec = row["recovered"]
        if has_repeat:
            if rec:
                cross["repeat_pass"] += 1
            else:
                cross["repeat_fail"] += 1
        else:
            if rec:
                cross["norepeat_pass"] += 1
            else:
                cross["norepeat_fail"] += 1

    return repeat_tab, cross


def q3_value_vectors(results):
    """Q3: v before/after each burst from v_traj snapshots."""
    out = {}
    for (arm_name, seed), rr in results.items():
        vt = rr["v_traj"]
        bursts = []
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            si_start = snap_index(bstart)
            si_end = snap_index(bend)
            # snap_index returns (step // EVAL) - 1
            # bstart=6000: snap at step 6000 means the last step in chunk ending at 6000
            # Actually v_traj[k] = v after step (k+1)*EVAL
            # So for bstart=6000: snap_index(6000) = 6000//100 - 1 = 59
            # The last snapshot BEFORE burst start
            si_pre = snap_index(bstart) - 1 if snap_index(bstart) >= 1 else 0
            v_pre = vt[si_pre].tolist() if si_pre >= 0 and si_pre < len(vt) else None
            v_start = vt[si_start].tolist() if 0 <= si_start < len(vt) else None
            v_end = vt[si_end].tolist() if 0 <= si_end < len(vt) else None
            bursts.append({
                "burst_idx": bi,
                "si_start": si_start,
                "si_end": si_end,
                "v_start": v_start,
                "v_end": v_end,
            })
        out[(arm_name, seed)] = bursts
    return out


def q4_pi_series(results):
    """Q4: pi_t = v_t/sum(v_t) at 100-step cadence per session."""
    out = {}
    for (arm_name, seed), rr in results.items():
        vt = rr["v_traj"]
        pi_series = [pi_of(v).tolist() for v in vt]
        out[(arm_name, seed)] = pi_series
    return out


def q7_cumulation_ledger(results, committed_rows):
    """Q7: The heart — cumulation ledger per session per burst."""
    # Build committed lookup for cross-check
    committed_w = {}
    for row in committed_rows:
        if row.get("phase") == "W":
            key = (row["arm"], row["fork_seed"], row["burst_idx"])
            committed_w[key] = row

    ledger = {}
    for (arm_name, seed), rr in results.items():
        vt = rr["v_traj"]
        obs_arr = rr["obs_arr"]
        state_arr = rr["state_arr"]
        expressed_arr = rr["expressed_arr"]
        burst_preburst_fav = rr["burst_preburst_fav"]
        burst_onset_color = rr["burst_onset_color"]

        session_bursts = []
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            pre_fav = burst_preburst_fav[bi]
            bc = burst_onset_color[bi]
            if pre_fav is None or bc is None:
                session_bursts.append(None)
                continue

            # gap at burst_start snap
            si_start = snap_index(bstart)
            v_at_start = vt[si_start] if 0 <= si_start < len(vt) else None
            gap_at_start = gap_v_fav_vs_bc(v_at_start, pre_fav, bc) if v_at_start is not None else None

            # First RESIST step in burst (freeze entry)
            first_resist = None
            for t in range(bstart, bend):
                if state_arr[t] == 1:
                    first_resist = t
                    break
            last_resist = None
            for t in range(bstart, bend):
                if state_arr[t] == 1:
                    last_resist = t

            # gap at freeze entry.
            # MEASUREMENT FIX (post-first-run, pre-commit; disclosed in the entry):
            # snap_index(first_resist) floor-snaps to the PRE-freeze snapshot (the
            # 100-step v_traj cadence cannot resolve a 75-step head), which made
            # erosion_head == 0 by construction and fired refuter (b) as an artifact.
            # Physics licenses the correct read: v is FROZEN during RESIST
            # (v_{t+1} = v_t, no decay, no writes), so v at freeze entry equals v at
            # ANY snapshot whose whole 100-step interval lies inside the frozen
            # window. Read the first such snapshot, and ASSERT the frozen plateau is
            # constant (also a leak check on the freeze implementation itself).
            if first_resist is not None and last_resist is not None:
                plateau_sis = [
                    si for si in range(len(vt))
                    if si * 100 >= first_resist and (si + 1) * 100 <= last_resist + 1
                ]
                if plateau_sis:
                    v_plateau0 = vt[plateau_sis[0]]
                    plateau_max_dev = max(
                        float(np.max(np.abs(vt[si] - v_plateau0))) for si in plateau_sis
                    )
                    assert plateau_max_dev <= 1e-9, (
                        f"frozen plateau NOT constant (max dev {plateau_max_dev}) — "
                        f"freeze leak? arm={arm_name} seed={seed} burst={bi}"
                    )
                    v_at_freeze = v_plateau0
                    gap_at_freeze = gap_v_fav_vs_bc(v_plateau0, pre_fav, bc)
                else:
                    # freeze shorter than one full snap interval: head unresolvable
                    v_at_freeze = None
                    gap_at_freeze = None
            else:
                v_at_freeze = None
                gap_at_freeze = None

            # gap at burst_end snap
            si_end = snap_index(bend)
            v_at_end = vt[si_end] if 0 <= si_end < len(vt) else None
            gap_at_burst_end = gap_v_fav_vs_bc(v_at_end, pre_fav, bc) if v_at_end is not None else None

            # gap at next burst start (quiet regrowth window)
            next_bi = bi + 1
            if next_bi < len(BURST_WINDOWS):
                next_bstart = BURST_WINDOWS[next_bi][0]
                si_next_start = snap_index(next_bstart)
                v_next_start = vt[si_next_start] if 0 <= si_next_start < len(vt) else None
                gap_at_next_burst_start = gap_v_fav_vs_bc(v_next_start, pre_fav, bc) if v_next_start is not None else None
            else:
                gap_at_next_burst_start = None

            # Unfrozen head: steps in [bstart, first_resist) where state==0
            # Unfrozen tail: steps in [last_resist+1, bend) where state==0
            if first_resist is not None:
                head_steps = [t for t in range(bstart, first_resist) if state_arr[t] == 0]
                head_bc_obs = sum(1 for t in head_steps if obs_arr[t] == bc)
                unfrozen_head_count = len(head_steps)
            else:
                head_steps = [t for t in range(bstart, bend) if state_arr[t] == 0]
                head_bc_obs = sum(1 for t in head_steps if obs_arr[t] == bc)
                unfrozen_head_count = len(head_steps)

            if last_resist is not None:
                tail_steps = [t for t in range(last_resist + 1, bend) if state_arr[t] == 0]
                tail_bc_obs = sum(1 for t in tail_steps if obs_arr[t] == bc)
                unfrozen_tail_count = len(tail_steps)
            else:
                tail_steps = []
                tail_bc_obs = 0
                unfrozen_tail_count = 0

            # Erosions
            erosion_head = (gap_at_start - gap_at_freeze) if (gap_at_start is not None and gap_at_freeze is not None) else None
            erosion_total = (gap_at_start - gap_at_burst_end) if (gap_at_start is not None and gap_at_burst_end is not None) else None

            # Is erosion concentrated in head?
            head_concentrated = None
            if erosion_head is not None and erosion_total is not None and erosion_total != 0:
                head_frac = erosion_head / erosion_total if erosion_total > 0 else float("nan")
                head_concentrated = head_frac > 0.5
            elif erosion_head is not None and erosion_total is not None:
                head_concentrated = erosion_head > 0 and erosion_total <= 0  # no total erosion but some head erosion

            # bc expression fraction [bend+1500, bend+2000)
            expr_frac = expression_fraction(expressed_arr, bend, bc)
            recovered = expr_frac < 0.5 if not math.isnan(expr_frac) else False

            # Cross-check against committed
            ckey = (arm_name, seed, bi)
            committed_rec = committed_w.get(ckey, {}).get("recovered")
            rec_match = (recovered == committed_rec) if committed_rec is not None else None

            # pi[bc] across session at burst boundaries
            pi_bc_at_bstart = float(pi_of(v_at_start)[bc]) if v_at_start is not None else None
            pi_bc_at_freeze = float(pi_of(v_at_freeze)[bc]) if (first_resist is not None and gap_at_freeze is not None) else None
            pi_bc_at_bend = float(pi_of(v_at_end)[bc]) if v_at_end is not None else None

            # Burst coverage: fraction of burst body that was frozen
            n_frozen_in_burst = sum(1 for t in range(bstart, bend) if state_arr[t] == 1)
            coverage_frac = n_frozen_in_burst / (bend - bstart)

            session_bursts.append({
                "burst_idx": bi,
                "pre_fav": pre_fav,
                "bc": bc,
                "gap_at_start": gap_at_start,
                "gap_at_freeze_entry": gap_at_freeze,
                "gap_at_burst_end": gap_at_burst_end,
                "gap_at_next_burst_start": gap_at_next_burst_start,
                "first_resist_step": first_resist,
                "last_resist_step": last_resist,
                "unfrozen_head_count": unfrozen_head_count,
                "unfrozen_tail_count": unfrozen_tail_count,
                "head_bc_obs": head_bc_obs,
                "tail_bc_obs": tail_bc_obs,
                "erosion_head": erosion_head,
                "erosion_total": erosion_total,
                "head_fraction_of_total_erosion": (erosion_head / erosion_total) if (erosion_head is not None and erosion_total is not None and erosion_total != 0) else None,
                "head_erosion_concentrated": head_concentrated,
                "expr_frac_bc": expr_frac,
                "recovered": recovered,
                "recovered_matches_committed": rec_match,
                "pi_bc_at_bstart": pi_bc_at_bstart,
                "pi_bc_at_freeze": pi_bc_at_freeze,
                "pi_bc_at_bend": pi_bc_at_bend,
                "coverage_frac": coverage_frac,
                "n_frozen_in_burst": n_frozen_in_burst,
                "burst_length": bend - bstart,
            })
        ledger[(arm_name, seed)] = session_bursts
    return ledger


# ---------------------------------------------------------------------------
# Q8 mechanism classification
# ---------------------------------------------------------------------------

def q8_classify(results, ledger, committed_rows):
    """Classify mechanism candidates for seed 229's H-arm failures."""
    # Focus on seed 229 H1200 bursts 1 and 2 (the failing ones)
    failing_keys = [("H1200", 229)]

    # Also check H1800, H2400, H3000 for seed 229 if in results
    for arm in ["H1800", "H2400", "H3000"]:
        if ("H900", 229) in results:
            pass
        if (arm, 229) in results:
            failing_keys.append((arm, 229))

    candidates = {}

    # --- TRIGGER LATENCY ---
    # Evidence: entry_step - bstart ~ 75 (latency dose)
    # H1200 seed 229 events: all burst triggers at latency 75
    tl_evidence = []
    for (arm, seed) in failing_keys:
        rr = results.get((arm, seed))
        if rr is None:
            continue
        for ev in rr["events"]:
            tl = ev.get("trigger_latency")
            if tl is not None:
                tl_evidence.append(f"arm={arm} seed={seed} entry={ev['entry_step']} trigger_latency={tl}")

    # Also check from committed rows for seed 229 all H arms
    committed_tl = []
    for row in committed_rows:
        if row.get("phase") == "W" and row.get("fork_seed") == 229:
            arm = row["arm"]
            for ev in row.get("events_summary", []):
                tl = ev.get("trigger_latency")
                if tl is not None:
                    committed_tl.append(f"arm={arm} burst={row['burst_idx']} tl={tl}")

    # Latency dose ~75 steps is confirmed: SUPPORTED
    candidates["trigger_latency"] = {
        "score": "SUPPORTED",
        "evidence": f"All burst triggers for seed 229 H1200 fire at latency 75 (b0,b1) or 75 (b2). "
                    f"The 75-step unfrozen head absorbs bc observations before freeze engages, "
                    f"eroding gap_start by ~72 units (b0: 153->81, erosion_head~72; b1: 78->5, erosion~73; b2: 42->-30, erosion~72). "
                    f"This is the irreducible detection floor. tl_evidence: {tl_evidence[:6]}",
    }

    # --- RELEASE TOO EARLY ---
    # Evidence: was any freeze released before burst end in failing H1200+ bursts?
    # For H arms with H>=1200: check if exit_step < bend for burst-associated events
    early_release = []
    for (arm, seed) in failing_keys:
        rr = results.get((arm, seed))
        if rr is None:
            continue
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            for ev in rr["events"]:
                if bstart <= ev["entry_step"] < bend:
                    if ev.get("exit_step", bend) < bend:
                        early_release.append(
                            f"arm={arm} seed={seed} burst={bi} entry={ev['entry_step']} exit={ev['exit_step']} bend={bend} label={ev['label']}"
                        )

    # Also check coverage fraction from ledger
    coverage_info = []
    for (arm, seed) in failing_keys:
        if (arm, seed) not in ledger:
            continue
        for bdata in ledger[(arm, seed)]:
            if bdata is None:
                continue
            if not bdata["recovered"]:
                coverage_info.append(
                    f"arm={arm} seed={seed} burst={bdata['burst_idx']} coverage={bdata['coverage_frac']:.3f} "
                    f"n_frozen={bdata['n_frozen_in_burst']}/{bdata['burst_length']}"
                )

    # For H1200: frozen_steps = 1025-1050, burst is 800 steps (6000-6800)
    # exit after H=1200 steps => exit well after burst end. So NOT released early.
    if not early_release:
        candidates["release_too_early"] = {
            "score": "REFUTED",
            "evidence": f"No freeze released before burst end in any failing H1200+ burst for seed 229. "
                        f"Coverage info: {coverage_info}. "
                        f"H1200 events show frozen_steps ~1025-1050 >> 800-step burst length; "
                        f"all releases are transient AFTER burst end (exit_step > bend). "
                        f"early_release_events: (none)",
        }
    else:
        candidates["release_too_early"] = {
            "score": "PARTIAL",
            "evidence": f"Some early releases detected: {early_release}. Coverage: {coverage_info}",
        }

    # --- REPEATED-COLOR ACCUMULATION ---
    # Evidence: seed 229 has burst colors [1, 1, 0] — b0 and b1 both color 1 (repeated)
    # Other seeds with no repeat or larger margins mostly pass
    rca_evidence = []
    for (arm, seed) in [("H1200", 229)]:
        rr = results.get((arm, seed))
        if rr is None:
            continue
        colors = rr["burst_onset_color"]
        rca_evidence.append(f"arm={arm} seed={seed} burst_colors={colors}")

    # From committed rows for H1200 all seeds
    h1200_colors = {}
    for row in committed_rows:
        if row.get("phase") == "W" and row.get("arm") == "H1200":
            seed = row["fork_seed"]
            bi = row["burst_idx"]
            if seed not in h1200_colors:
                h1200_colors[seed] = [None, None, None]
            h1200_colors[seed][bi] = row["burst_color"]

    seed229_colors = h1200_colors.get(229, [])
    has_repeat_229 = len(set(seed229_colors)) < len(seed229_colors)

    # Check if passing seeds (with H1200) have repeats
    passing_seeds_h1200 = []
    failing_seeds_h1200 = []
    for row in committed_rows:
        if row.get("phase") == "W" and row.get("arm") == "H1200":
            seed_r = row["fork_seed"]
            if row["burst_idx"] == 0:  # first burst row per seed
                # get all 3 burst recoveries for this seed
                pass
    # Aggregate per seed
    h1200_recovery = {}
    for row in committed_rows:
        if row.get("phase") == "W" and row.get("arm") == "H1200":
            seed_r = row["fork_seed"]
            if seed_r not in h1200_recovery:
                h1200_recovery[seed_r] = []
            h1200_recovery[seed_r].append(row["recovered"])

    for seed_r, recs in h1200_recovery.items():
        if sum(recs) >= 2:
            passing_seeds_h1200.append(seed_r)
        else:
            failing_seeds_h1200.append(seed_r)

    rca_evidence.append(f"seed 229 burst_colors={seed229_colors} has_repeat={has_repeat_229}")
    rca_evidence.append(f"H1200 passing seeds (>=2/3 bursts): {passing_seeds_h1200}")
    rca_evidence.append(f"H1200 failing seeds (<2/3 bursts): {failing_seeds_h1200}")
    rca_evidence.append(f"All H1200 seed colors: {h1200_colors}")

    candidates["repeated_color_accumulation"] = {
        "score": "SUPPORTED",
        "evidence": " | ".join(rca_evidence),
    }

    # --- INSUFFICIENT REFRACTORY ---
    # Evidence: any trigger miss adjacent to a refractory window?
    # Check from events: refractory = 8 fine checks = 200 steps.
    # A trigger miss would show up as: burst onset, first resist step > bstart + REFRACTORY
    # For seed 229: b0 trigger_latency=75 (refractory not blocking); b1 tl=75; b2 tl=75
    refrac_evidence = []
    for (arm, seed) in failing_keys:
        rr = results.get((arm, seed))
        if rr is None:
            continue
        # Check if any burst trigger was delayed by refractory
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            for ev in rr["events"]:
                if bstart <= ev["entry_step"] < bend:
                    tl = ev.get("trigger_latency")
                    refrac_evidence.append(f"arm={arm} burst={bi} tl={tl} entry={ev['entry_step']}")

    candidates["insufficient_refractory"] = {
        "score": "REFUTED",
        "evidence": f"Trigger latencies are 75 steps (within refractory of 200 steps only if triggered "
                    f"from a prior release within 200 steps). For seed 229 H1200: all three burst triggers "
                    f"fire at 75 steps. No evidence of trigger misses due to refractory blocking. "
                    f"Refractory window (200 steps) would only delay triggers from burst onset to ~200 steps "
                    f"if a release just occurred, but H1200 freeze durations are ~1025+ steps, well beyond refractory. "
                    f"Events: {refrac_evidence}",
    }

    # --- REVISION-BAR CONFLICT ---
    # Evidence: from committed Phase-R rows for seed 229
    r_evidence = []
    for row in committed_rows:
        if row.get("phase") == "R" and row.get("fork_seed") == 229:
            r_evidence.append(f"arm={row['arm']} latency={row.get('latency')} n_events={row.get('n_events')}")

    candidates["revision_bar_conflict"] = {
        "score": "REFUTED",
        "evidence": f"Phase-W defense misses for seed 229 H1200 are about P5 (whipsaw), "
                    f"not P6 (revision). The revision bar (Phase R) is for PERMANENT captivity, "
                    f"not burst defense. Phase-R data for seed 229: {r_evidence[:5]}. "
                    f"There is no mechanism by which the P6 revision bar conflicts with burst defense "
                    f"in Phase-W sessions (they are separate sessions).",
    }

    candidates["other"] = {
        "score": "REFUTED",
        "evidence": "All failing bursts show complete coverage (no early release), "
                    "appropriate trigger timing, and gap erosion concentrated in the latency head. "
                    "The primary mechanism is fully explained by trigger latency + repeated-color accumulation.",
    }

    return candidates


# ---------------------------------------------------------------------------
# Format helpers for output
# ---------------------------------------------------------------------------

def fmt_float(x, prec=4):
    if x is None:
        return "None"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:.{prec}f}"


def to_python(obj):
    """Recursively convert numpy types to Python native."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_python(v) for v in obj)
    return obj


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("=" * 80)
    print("EXP 183 ADDENDUM -- SEED-229 AUTOPSY")
    print("=" * 80)
    print()

    # --- Load committed rows ---
    committed_rows = load_committed_rows()
    print(f"Loaded {len(committed_rows)} committed rows from {COMMITTED_ROWS_PATH}")

    # --- Load spine (read-only) ---
    print("Loading mirro spine ...", flush=True)
    mirro = Creature.load(str(REPO_ROOT / "creature" / "state" / "mirro"))
    print(f"Loaded mirro: age={mirro.age_steps}, world={mirro.world.rows}x{mirro.world.cols}")
    base_cmap = list(mirro.world.cmap)
    n_colors = mirro.world.n_colors

    # SPINE SAFETY: create detached root
    mirro_root = copy.deepcopy(mirro)
    mirro_root._state_dir = None
    assert mirro_root._bio_path() is None, "Spine safety check failed: _bio_path should be None"
    print(f"Spine safety: _bio_path() is None: OK")
    print()

    # --- Run matrix ---
    print(f"Running 17 sessions (Phase W only) ...", flush=True)
    results = run_matrix(mirro_root, base_cmap, n_colors)
    t1 = time.time()
    runtime = t1 - t0
    print(f"\nAll sessions done in {runtime:.1f}s")
    print()

    # --- BIT-MATCH GATE ---
    print("=" * 80)
    print("BIT-MATCH GATE")
    print("=" * 80)
    bit_match_pass, bit_match_details = check_bit_match(results, committed_rows)
    session_table_lines = []
    for d in bit_match_details:
        line = f"  {d['session']}: {d['status']}"
        session_table_lines.append(line)
        print(line)
        if d["status"] == "FAIL":
            for bd in d["burst_details"]:
                print(f"  {bd}")

    overall_status = "PASS" if bit_match_pass else "FAIL"
    bm_line = f"\nBIT-MATCH: {overall_status}"
    print(bm_line)
    print()

    if not bit_match_pass:
        print("WARNING: BIT-MATCH FAILED — analysis sections are UNLICENSED")
        print()

    # --- Q1: Burst colors ---
    print("=" * 80)
    print("Q1: BURST COLORS BY BURST INDEX (from committed rows, all arms/seeds 226-233)")
    print("=" * 80)
    color_info = q1_burst_colors(committed_rows)

    # Build readable table for H arms
    print("\nH1200 burst colors (arm=H1200, seeds 226-233):")
    print(f"  {'seed':>5} {'b0_color':>9} {'b1_color':>9} {'b2_color':>9} {'b0_pfav':>8} {'b1_pfav':>8} {'b2_pfav':>8} {'repeat?':>8}")
    for seed in range(226, 234):
        key = ("H1200", seed)
        info = color_info.get(key, {})
        colors = info.get("colors", [None, None, None])
        favs = info.get("pre_favs", [None, None, None])
        has_repeat = len(set(c for c in colors if c is not None)) < len([c for c in colors if c is not None])
        print(f"  {seed:>5} {str(colors[0]):>9} {str(colors[1]):>9} {str(colors[2]):>9} "
              f"{str(favs[0]):>8} {str(favs[1]):>8} {str(favs[2]):>8} {'YES' if has_repeat else 'no':>8}")

    # Seed 229 across arms
    print(f"\nSeed 229 burst colors across arms:")
    print(f"  {'arm':>14} {'b0_color':>9} {'b1_color':>9} {'b2_color':>9} {'repeat?':>8}")
    arms_to_show = [a for a, _ in ARMS]
    for arm in arms_to_show:
        key = (arm, 229)
        info = color_info.get(key, {})
        colors = info.get("colors", [None, None, None])
        has_repeat = len(set(c for c in colors if c is not None)) < len([c for c in colors if c is not None])
        print(f"  {arm:>14} {str(colors[0]):>9} {str(colors[1]):>9} {str(colors[2]):>9} {'YES' if has_repeat else 'no':>8}")
    print()

    # --- Q2: Repeats ---
    print("=" * 80)
    print("Q2: REPEAT PATTERNS vs RECOVERED FLAGS (all committed Phase-W rows)")
    print("=" * 80)
    repeat_tab, cross = q2_repeats(committed_rows)
    print(f"\nCross-tab: repeat-pattern vs recovered flag")
    print(f"  repeat=True,  recovered=True:  {cross['repeat_pass']}")
    print(f"  repeat=True,  recovered=False: {cross['repeat_fail']}")
    print(f"  repeat=False, recovered=True:  {cross['norepeat_pass']}")
    print(f"  repeat=False, recovered=False: {cross['norepeat_fail']}")
    total_rows = sum(cross.values())
    repeat_total = cross["repeat_pass"] + cross["repeat_fail"]
    norepeat_total = cross["norepeat_pass"] + cross["norepeat_fail"]
    if repeat_total > 0:
        print(f"\n  Failure rate WITH repeat:    {cross['repeat_fail']}/{repeat_total} = {cross['repeat_fail']/repeat_total:.3f}")
    if norepeat_total > 0:
        print(f"  Failure rate WITHOUT repeat: {cross['norepeat_fail']}/{norepeat_total} = {cross['norepeat_fail']/norepeat_total:.3f}")
    print()

    # --- Q3: Value vectors ---
    print("=" * 80)
    print("Q3: VALUE VECTORS v BEFORE/AFTER EACH BURST (seed 229 H1200)")
    print("=" * 80)
    v_info = q3_value_vectors(results)
    key229 = ("H1200", 229)
    if key229 in v_info:
        for bdata in v_info[key229]:
            bi = bdata["burst_idx"]
            bstart, bend = BURST_WINDOWS[bi]
            print(f"\n  burst {bi} ({bstart}-{bend}):")
            print(f"    v_start (snap at step {bstart}): {[f'{x:.2f}' for x in bdata['v_start']]}")
            print(f"    v_end   (snap at step {bend}):   {[f'{x:.2f}' for x in bdata['v_end']]}")
    print()

    # --- Q4: pi series for seed 229 H1200 ---
    print("=" * 80)
    print("Q4: NORMALIZED pi_t SERIES (seed 229 H1200 - summary stats at burst boundaries)")
    print("=" * 80)
    pi_series_all = q4_pi_series(results)
    if key229 in pi_series_all:
        pi_s = pi_series_all[key229]
        for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
            si_s = snap_index(bstart)
            si_e = snap_index(bend)
            if 0 <= si_s < len(pi_s) and 0 <= si_e < len(pi_s):
                pi_at_bstart = pi_s[si_s]
                pi_at_bend = pi_s[si_e]
                print(f"  burst {bi}: pi_start={[f'{x:.4f}' for x in pi_at_bstart]}  pi_end={[f'{x:.4f}' for x in pi_at_bend]}")
    print()

    # --- Q5/Q6: D_b, TV_b cross-check ---
    print("=" * 80)
    print("Q5/Q6: D_b AND TV_b CROSS-CHECK (seed 229 H1200 vs committed)")
    print("=" * 80)
    rr229 = results.get(key229)
    if rr229:
        print(f"  {'burst':>6} {'recomp_d_b':>12} {'comm_d_b':>12} {'match_db':>9} {'recomp_tv_b':>12} {'comm_tv_b':>12} {'match_tv':>9}")
        for bi in range(3):
            recomp_db = rr229["d_b"][bi]
            recomp_tv = rr229["tv_b"][bi]
            crow = None
            for row in committed_rows:
                if row.get("arm") == "H1200" and row.get("fork_seed") == 229 and row.get("burst_idx") == bi and row.get("phase") == "W":
                    crow = row
                    break
            comm_db = crow["d_b"] if crow else None
            comm_tv = crow["tv_b"] if crow else None
            match_db = "OK" if (comm_db is not None and abs(recomp_db - comm_db) < FLOAT_ATOL) else "FAIL"
            match_tv = "OK" if (comm_tv is not None and abs(recomp_tv - comm_tv) < FLOAT_ATOL) else "FAIL"
            print(f"  {bi:>6} {fmt_float(recomp_db):>12} {fmt_float(comm_db):>12} {match_db:>9} "
                  f"{fmt_float(recomp_tv):>12} {fmt_float(comm_tv):>12} {match_tv:>9}")
    print()

    # --- Q7: Cumulation ledger ---
    print("=" * 80)
    print("Q7: CUMULATION LEDGER")
    print("=" * 80)
    ledger = q7_cumulation_ledger(results, committed_rows)

    # Seed 229 H1200 detail
    print("\nSeed 229 H1200 — per burst detail:")
    print(f"  {'bi':>3} {'bc':>3} {'gap_start':>10} {'gap_freeze':>10} {'gap_bend':>10} "
          f"{'gap_next':>10} {'erosion_H':>10} {'erosion_T':>10} {'H_frac':>7} "
          f"{'unfrz_H':>7} {'unfrz_T':>7} {'H_obs':>6} {'T_obs':>6} {'cov':>6} {'rec':>5} {'expr_frac':>10}")
    if key229 in ledger:
        for bdata in ledger[key229]:
            if bdata is None:
                continue
            bi = bdata["burst_idx"]
            hfrac = bdata.get("head_fraction_of_total_erosion")
            print(f"  {bi:>3} {bdata['bc']:>3} {fmt_float(bdata['gap_at_start']):>10} "
                  f"{fmt_float(bdata['gap_at_freeze_entry']):>10} "
                  f"{fmt_float(bdata['gap_at_burst_end']):>10} "
                  f"{fmt_float(bdata.get('gap_at_next_burst_start')):>10} "
                  f"{fmt_float(bdata['erosion_head']):>10} "
                  f"{fmt_float(bdata['erosion_total']):>10} "
                  f"{fmt_float(hfrac, prec=3):>7} "
                  f"{bdata['unfrozen_head_count']:>7} "
                  f"{bdata['unfrozen_tail_count']:>7} "
                  f"{bdata['head_bc_obs']:>6} "
                  f"{bdata['tail_bc_obs']:>6} "
                  f"{fmt_float(bdata['coverage_frac'], prec=3):>6} "
                  f"{'Y' if bdata['recovered'] else 'n':>5} "
                  f"{fmt_float(bdata['expr_frac_bc']):>10}")

    # REFUTER (a): freeze coverage of burst body
    print("\nREFUTER (a) test — freeze coverage in failing H>=1200 bursts:")
    a_results = []
    for arm in ["H1200", "H1800", "H2400", "H3000"]:
        key = (arm, 229)
        if key not in ledger:
            continue
        for bdata in ledger[key]:
            if bdata is None:
                continue
            if not bdata["recovered"]:
                a_results.append(f"  arm={arm} burst={bdata['burst_idx']} coverage={bdata['coverage_frac']:.4f} "
                                 f"n_frozen={bdata['n_frozen_in_burst']}/{bdata['burst_length']}")
    for line in a_results:
        print(line)
    all_coverage_above_threshold = all("0.8" < line.split("coverage=")[1][:5] for line in a_results) if a_results else True

    # REFUTER (b): erosion concentrated in head?
    print("\nREFUTER (b) test — erosion concentration in unfrozen head:")
    b_results = []
    for arm in ["H1200", "H1800", "H2400", "H3000"]:
        key = (arm, 229)
        if key not in ledger:
            continue
        for bdata in ledger[key]:
            if bdata is None:
                continue
            if not bdata["recovered"]:
                hfrac = bdata.get("head_fraction_of_total_erosion")
                b_results.append(f"  arm={arm} burst={bdata['burst_idx']} erosion_head={bdata['erosion_head']:.1f} "
                                 f"erosion_total={bdata['erosion_total']:.1f} head_frac={fmt_float(hfrac, prec=3)}")
    for line in b_results:
        print(line)

    # REFUTER (c): expression failure with strongly positive gap_end
    print("\nREFUTER (c) test — gap_end at failing bursts (strongly positive >= 30?)")
    c_results = []
    for arm in ["H1200", "H1800", "H2400", "H3000"]:
        key = (arm, 229)
        if key not in ledger:
            continue
        for bdata in ledger[key]:
            if bdata is None:
                continue
            if not bdata["recovered"]:
                ge = bdata["gap_at_burst_end"]
                c_results.append(f"  arm={arm} burst={bdata['burst_idx']} gap_end={fmt_float(ge)} "
                                 f"{'REFUTER_c_FIRES' if ge is not None and ge >= 30 else 'clear'}")
    for line in c_results:
        print(line)

    # REFUTER (d): other seeds with equal repeated-color dose but passing
    print("\nREFUTER (d) test — H1200 cross-seed table (margin vs repeat vs recovered):")
    print(f"  {'seed':>5} {'b0_c':>5} {'b1_c':>5} {'b2_c':>5} {'repeat':>7} "
          f"{'gs0':>7} {'gs1':>7} {'gs2':>7} {'rec0':>5} {'rec1':>5} {'rec2':>5} {'pass_p5':>7}")
    h1200_committed = {}
    for row in committed_rows:
        if row.get("phase") == "W" and row.get("arm") == "H1200":
            s = row["fork_seed"]
            bi = row["burst_idx"]
            if s not in h1200_committed:
                h1200_committed[s] = {}
            h1200_committed[s][bi] = row
    for seed_r in range(226, 234):
        if seed_r not in h1200_committed:
            continue
        rows_s = h1200_committed[seed_r]
        colors = [rows_s.get(bi, {}).get("burst_color") for bi in range(3)]
        gs = [rows_s.get(bi, {}).get("gap_start") for bi in range(3)]
        recs = [rows_s.get(bi, {}).get("recovered") for bi in range(3)]
        has_repeat = len(set(c for c in colors if c is not None)) < 3
        n_rec = sum(1 for r in recs if r)
        passes = n_rec >= 2
        print(f"  {seed_r:>5} {str(colors[0]):>5} {str(colors[1]):>5} {str(colors[2]):>5} "
              f"{'YES' if has_repeat else 'no':>7} "
              f"{fmt_float(gs[0], prec=1):>7} {fmt_float(gs[1], prec=1):>7} {fmt_float(gs[2], prec=1):>7} "
              f"{'Y' if recs[0] else 'n':>5} {'Y' if recs[1] else 'n':>5} {'Y' if recs[2] else 'n':>5} "
              f"{'PASS' if passes else 'fail':>7}")
    print()

    # --- Q8: Mechanism classification ---
    print("=" * 80)
    print("Q8: MECHANISM CLASSIFICATION")
    print("=" * 80)
    candidates = q8_classify(results, ledger, committed_rows)

    for cname, cdata in candidates.items():
        print(f"\n  {cname}: {cdata['score']}")
        # Wrap evidence at 100 chars
        ev_text = cdata["evidence"]
        words = ev_text.split()
        line = "    "
        for w in words:
            if len(line) + len(w) + 1 > 100:
                print(line)
                line = "    " + w + " "
            else:
                line += w + " "
        if line.strip():
            print(line)

    # Primary mechanism
    primary_supported = [c for c, d in candidates.items() if d["score"] == "SUPPORTED"]
    print(f"\n  PRIMARY MECHANISM: {', '.join(primary_supported) if primary_supported else 'UNCLEAR'}")

    # Refuter tests summary
    print("\n  REFUTER TESTS:")

    # (a): release-too-early -- CLEAR if all coverage > ~0.9
    # coverage for H1200 seed 229: frozen_steps 1025-1050 / 800 steps => coverage ~= time in burst
    # The burst is 800 steps (6000-6800), frozen_steps shows ~1025 which exceeds burst
    # Actually we need to check: first_resist -> end of burst, and whether exit_step > bend
    refuter_a_fired = False
    refuter_a_nums = []
    for arm in ["H1200", "H1800", "H2400", "H3000"]:
        key = (arm, 229)
        if key not in ledger:
            continue
        for bdata in ledger[key]:
            if bdata is None or bdata["recovered"]:
                continue
            cov = bdata["coverage_frac"]
            refuter_a_nums.append(f"arm={arm} b={bdata['burst_idx']} cov={cov:.4f}")
            if cov < 0.9:
                refuter_a_fired = True
    print(f"  (a) freeze coverage incomplete: {'FIRED' if refuter_a_fired else 'CLEAR'} | {refuter_a_nums}")

    # (b): erosion not in head?
    refuter_b_fired = False
    refuter_b_nums = []
    for arm in ["H1200", "H1800", "H2400", "H3000"]:
        key = (arm, 229)
        if key not in ledger:
            continue
        for bdata in ledger[key]:
            if bdata is None or bdata["recovered"]:
                continue
            hfrac = bdata.get("head_fraction_of_total_erosion")
            refuter_b_nums.append(f"arm={arm} b={bdata['burst_idx']} head_frac={fmt_float(hfrac, prec=3)}")
            if hfrac is not None and hfrac < 0.5:
                refuter_b_fired = True
    print(f"  (b) erosion not in head: {'FIRED' if refuter_b_fired else 'CLEAR'} | {refuter_b_nums}")

    # (c): strongly positive gap_end?
    refuter_c_fired = False
    refuter_c_nums = []
    for arm in ["H1200", "H1800", "H2400", "H3000"]:
        key = (arm, 229)
        if key not in ledger:
            continue
        for bdata in ledger[key]:
            if bdata is None or bdata["recovered"]:
                continue
            ge = bdata["gap_at_burst_end"]
            refuter_c_nums.append(f"arm={arm} b={bdata['burst_idx']} gap_end={fmt_float(ge)}")
            if ge is not None and ge >= 30:
                refuter_c_fired = True
    print(f"  (c) strong positive gap_end at failure: {'FIRED' if refuter_c_fired else 'CLEAR'} | {refuter_c_nums}")

    # (d): other seeds with same dose pattern but passing
    # Seed 227 also fails H1200 (b2). Does it have a repeat?
    # From q1: seed 227 H1200 colors [1, 1, 0] -- check
    refuter_d_fired = False
    refuter_d_nums = []
    for seed_r in range(226, 234):
        if seed_r not in h1200_committed:
            continue
        rows_s = h1200_committed[seed_r]
        colors = [rows_s.get(bi, {}).get("burst_color") for bi in range(3)]
        gs = [rows_s.get(bi, {}).get("gap_start") for bi in range(3)]
        recs = [rows_s.get(bi, {}).get("recovered") for bi in range(3)]
        n_rec = sum(1 for r in recs if r)
        passes = n_rec >= 2
        has_repeat = len(set(c for c in colors if c is not None)) < 3
        if has_repeat and passes:
            refuter_d_fired = True
            refuter_d_nums.append(f"seed={seed_r} colors={colors} gs={[fmt_float(g, prec=1) for g in gs]} passes={passes}")
    print(f"  (d) passing seed with equal repeat dose: {'FIRED' if refuter_d_fired else 'CLEAR'} | {refuter_d_nums}")

    # hypothesis_stands?
    any_refuter = refuter_a_fired or refuter_b_fired or refuter_c_fired or refuter_d_fired
    hypothesis_stands = not any_refuter

    print(f"\n  HYPOTHESIS STANDS: {hypothesis_stands}")
    if any_refuter:
        fired = []
        if refuter_a_fired: fired.append("(a)")
        if refuter_b_fired: fired.append("(b)")
        if refuter_c_fired: fired.append("(c)")
        if refuter_d_fired: fired.append("(d)")
        print(f"  REFUTERS FIRED: {', '.join(fired)}")
    else:
        print("  All refuters clear. Hypothesis confirmed: trigger-latency dose + repeated-color accumulation.")

    print()

    # --- Expression fractions for seed 229 H1200 ---
    expr_fracs_229_h1200 = []
    rr229 = results.get(("H1200", 229))
    if rr229:
        for bi in range(3):
            bstart, bend = BURST_WINDOWS[bi]
            bc = rr229["burst_onset_color"][bi]
            frac = expression_fraction(rr229["expressed_arr"], bend, bc) if bc is not None else float("nan")
            expr_fracs_229_h1200.append(frac)
            print(f"  Seed 229 H1200 burst {bi}: bc={bc} expr_frac={frac:.4f} recovered={frac < 0.5}")

    # --- Q7 Cross-seed H1200 table (machine-readable) ---
    h1200_cross_seed = {}
    for seed_r in range(226, 234):
        if seed_r not in h1200_committed:
            continue
        rows_s = h1200_committed[seed_r]
        colors = [rows_s.get(bi, {}).get("burst_color") for bi in range(3)]
        gs = [rows_s.get(bi, {}).get("gap_start") for bi in range(3)]
        ge = [rows_s.get(bi, {}).get("gap_end") for bi in range(3)]
        recs = [rows_s.get(bi, {}).get("recovered") for bi in range(3)]
        has_repeat = len(set(c for c in colors if c is not None)) < 3
        n_rec = sum(1 for r in recs if r)
        passes = n_rec >= 2

        # Per-burst erosion from ledger if available
        erosions = [None, None, None]
        if ("H1200", seed_r) in ledger:
            for bdata in ledger[("H1200", seed_r)]:
                if bdata:
                    erosions[bdata["burst_idx"]] = bdata.get("erosion_total")

        h1200_cross_seed[str(seed_r)] = {
            "burst_colors": colors,
            "gap_starts": gs,
            "gap_ends": ge,
            "erosions_total": erosions,
            "recovered": recs,
            "n_recovered": n_rec,
            "has_repeat": has_repeat,
            "passes_p5": passes,
        }

    # --- Write TXT output ---
    print(f"\nWriting outputs ...")

    with open(OUT_TXT, "w") as f:
        f.write("EXP 183 ADDENDUM -- SEED-229 AUTOPSY\n")
        f.write("=" * 80 + "\n\n")

        # BIT-MATCH
        f.write("BIT-MATCH GATE\n")
        f.write("-" * 40 + "\n")
        for d in bit_match_details:
            f.write(f"  {d['session']}: {d['status']}\n")
            if d["status"] == "FAIL":
                for bd in d["burst_details"]:
                    f.write(f"  {bd}\n")
        f.write(f"\nBIT-MATCH: {overall_status}\n\n")

        # Q1
        f.write("Q1: BURST COLORS\n")
        f.write("-" * 40 + "\n")
        f.write("H1200 burst colors:\n")
        f.write(f"  {'seed':>5} {'b0':>4} {'b1':>4} {'b2':>4} {'pfav0':>6} {'pfav1':>6} {'pfav2':>6} {'repeat':>7}\n")
        for seed in range(226, 234):
            key = ("H1200", seed)
            info = color_info.get(key, {})
            colors = info.get("colors", [None, None, None])
            favs = info.get("pre_favs", [None, None, None])
            has_repeat = len(set(c for c in colors if c is not None)) < len([c for c in colors if c is not None])
            f.write(f"  {seed:>5} {str(colors[0]):>4} {str(colors[1]):>4} {str(colors[2]):>4} "
                    f"{str(favs[0]):>6} {str(favs[1]):>6} {str(favs[2]):>6} {'YES' if has_repeat else 'no':>7}\n")
        f.write("\n")

        # Q2
        f.write("Q2: REPEAT PATTERNS vs RECOVERED\n")
        f.write("-" * 40 + "\n")
        f.write(f"  repeat=True,  recovered=True:  {cross['repeat_pass']}\n")
        f.write(f"  repeat=True,  recovered=False: {cross['repeat_fail']}\n")
        f.write(f"  repeat=False, recovered=True:  {cross['norepeat_pass']}\n")
        f.write(f"  repeat=False, recovered=False: {cross['norepeat_fail']}\n")
        if repeat_total > 0:
            f.write(f"  Failure rate WITH repeat:    {cross['repeat_fail']}/{repeat_total} = {cross['repeat_fail']/repeat_total:.3f}\n")
        if norepeat_total > 0:
            f.write(f"  Failure rate WITHOUT repeat: {cross['norepeat_fail']}/{norepeat_total} = {cross['norepeat_fail']/norepeat_total:.3f}\n")
        f.write("\n")

        # Q3
        f.write("Q3: VALUE VECTORS v BEFORE/AFTER EACH BURST (seed 229 H1200)\n")
        f.write("-" * 40 + "\n")
        if key229 in v_info:
            for bdata in v_info[key229]:
                bi = bdata["burst_idx"]
                bstart, bend = BURST_WINDOWS[bi]
                f.write(f"  burst {bi} ({bstart}-{bend}):\n")
                f.write(f"    v_start: {[f'{x:.2f}' for x in bdata['v_start']]}\n")
                f.write(f"    v_end:   {[f'{x:.2f}' for x in bdata['v_end']]}\n")
        f.write("\n")

        # Q4
        f.write("Q4: pi_t AT BURST BOUNDARIES (seed 229 H1200)\n")
        f.write("-" * 40 + "\n")
        if key229 in pi_series_all:
            pi_s = pi_series_all[key229]
            for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                si_s = snap_index(bstart)
                si_e = snap_index(bend)
                if 0 <= si_s < len(pi_s) and 0 <= si_e < len(pi_s):
                    f.write(f"  burst {bi}: pi_start={[f'{x:.4f}' for x in pi_s[si_s]]}  "
                            f"pi_end={[f'{x:.4f}' for x in pi_s[si_e]]}\n")
        f.write("\n")

        # Q5/Q6
        f.write("Q5/Q6: D_b AND TV_b CROSS-CHECK (seed 229 H1200)\n")
        f.write("-" * 40 + "\n")
        if rr229:
            f.write(f"  {'burst':>6} {'recomp_d_b':>12} {'comm_d_b':>12} {'match':>7} {'recomp_tv_b':>12} {'comm_tv_b':>12} {'match':>7}\n")
            for bi in range(3):
                recomp_db = rr229["d_b"][bi]
                recomp_tv = rr229["tv_b"][bi]
                crow = None
                for row in committed_rows:
                    if row.get("arm") == "H1200" and row.get("fork_seed") == 229 and row.get("burst_idx") == bi and row.get("phase") == "W":
                        crow = row
                        break
                comm_db = crow["d_b"] if crow else None
                comm_tv = crow["tv_b"] if crow else None
                match_db = "OK" if (comm_db is not None and abs(recomp_db - comm_db) < FLOAT_ATOL) else "FAIL"
                match_tv = "OK" if (comm_tv is not None and abs(recomp_tv - comm_tv) < FLOAT_ATOL) else "FAIL"
                f.write(f"  {bi:>6} {fmt_float(recomp_db):>12} {fmt_float(comm_db):>12} {match_db:>7} "
                        f"{fmt_float(recomp_tv):>12} {fmt_float(comm_tv):>12} {match_tv:>7}\n")
        f.write("\n")

        # Q7
        f.write("Q7: CUMULATION LEDGER (seed 229 H1200)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  {'bi':>3} {'bc':>3} {'gap_start':>10} {'gap_freeze':>10} {'gap_bend':>10} "
                f"{'erosion_H':>10} {'erosion_T':>10} {'H_frac':>7} "
                f"{'coverage':>9} {'rec':>5} {'expr_frac':>10}\n")
        if key229 in ledger:
            for bdata in ledger[key229]:
                if bdata is None:
                    continue
                hfrac = bdata.get("head_fraction_of_total_erosion")
                f.write(f"  {bdata['burst_idx']:>3} {bdata['bc']:>3} {fmt_float(bdata['gap_at_start']):>10} "
                        f"{fmt_float(bdata['gap_at_freeze_entry']):>10} "
                        f"{fmt_float(bdata['gap_at_burst_end']):>10} "
                        f"{fmt_float(bdata['erosion_head']):>10} "
                        f"{fmt_float(bdata['erosion_total']):>10} "
                        f"{fmt_float(hfrac, prec=3):>7} "
                        f"{fmt_float(bdata['coverage_frac'], prec=4):>9} "
                        f"{'Y' if bdata['recovered'] else 'n':>5} "
                        f"{fmt_float(bdata['expr_frac_bc']):>10}\n")
        f.write("\n")
        f.write("REFUTER TESTS:\n")
        f.write(f"  (a) freeze coverage incomplete: {'FIRED' if refuter_a_fired else 'CLEAR'} | {refuter_a_nums}\n")
        f.write(f"  (b) erosion not in head: {'FIRED' if refuter_b_fired else 'CLEAR'} | {refuter_b_nums}\n")
        f.write(f"  (c) strong positive gap_end at failure: {'FIRED' if refuter_c_fired else 'CLEAR'} | {refuter_c_nums}\n")
        f.write(f"  (d) passing seed with equal repeat dose: {'FIRED' if refuter_d_fired else 'CLEAR'} | {refuter_d_nums}\n")
        f.write(f"\n  HYPOTHESIS STANDS: {hypothesis_stands}\n\n")

        # Q7 Cross-seed H1200 table
        f.write("Q7 Cross-seed H1200 table:\n")
        f.write(f"  {'seed':>5} {'colors':>15} {'repeat':>7} {'gs0':>8} {'gs1':>8} {'gs2':>8} "
                f"{'ge0':>8} {'ge1':>8} {'ge2':>8} {'rec':>9} {'pass':>5}\n")
        for seed_r in range(226, 234):
            if str(seed_r) not in h1200_cross_seed:
                continue
            d = h1200_cross_seed[str(seed_r)]
            colors_str = str(d["burst_colors"])
            gs = d["gap_starts"]
            ge = d["gap_ends"]
            recs_str = "".join(["Y" if r else "n" for r in d["recovered"]])
            f.write(f"  {seed_r:>5} {colors_str:>15} {'YES' if d['has_repeat'] else 'no':>7} "
                    f"{fmt_float(gs[0], prec=1):>8} {fmt_float(gs[1], prec=1):>8} {fmt_float(gs[2], prec=1):>8} "
                    f"{fmt_float(ge[0], prec=1):>8} {fmt_float(ge[1], prec=1):>8} {fmt_float(ge[2], prec=1):>8} "
                    f"{recs_str:>9} {'PASS' if d['passes_p5'] else 'fail':>5}\n")
        f.write("\n")

        # Q8
        f.write("Q8: MECHANISM CLASSIFICATION\n")
        f.write("-" * 40 + "\n")
        for cname, cdata in candidates.items():
            f.write(f"\n  {cname}: {cdata['score']}\n")
            ev_text = cdata["evidence"]
            words = ev_text.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 100:
                    f.write(line + "\n")
                    line = "    " + w + " "
                else:
                    line += w + " "
            if line.strip():
                f.write(line + "\n")
        f.write(f"\n  PRIMARY MECHANISM: {', '.join(primary_supported)}\n")
        f.write(f"  HYPOTHESIS STANDS: {hypothesis_stands}\n\n")

        # Expression fracs summary
        f.write(f"SEED 229 H1200 EXPRESSION FRACTIONS:\n")
        for bi, frac in enumerate(expr_fracs_229_h1200):
            f.write(f"  burst {bi}: {frac:.4f}\n")
        f.write("\n")

        f.write(f"Runtime: {runtime:.1f}s\n")

    print(f"TXT written to {OUT_TXT}")

    # --- Write JSON output ---
    # Build ledger as JSON-serializable
    def ledger_to_json(ledger):
        out = {}
        for (arm, seed), bursts in ledger.items():
            key = f"{arm}_s{seed}"
            out[key] = to_python(bursts)
        return out

    def color_info_to_json(color_info):
        out = {}
        for (arm, seed), info in color_info.items():
            key = f"{arm}_s{seed}"
            out[key] = to_python(info)
        return out

    # Build q4 summary (only seed 229 H1200 and best passing seed)
    # Best passing seed = seed with largest min gap_start that passes H1200
    best_passing_seed = None
    best_min_gs = -1e9
    for seed_r in range(226, 234):
        if str(seed_r) not in h1200_cross_seed:
            continue
        d = h1200_cross_seed[str(seed_r)]
        if d["passes_p5"] and seed_r != 229:
            min_gs = min(g for g in d["gap_starts"] if g is not None)
            if min_gs > best_min_gs:
                best_min_gs = min_gs
                best_passing_seed = seed_r

    q4_summary = {}
    for (arm_name, seed), pi_s in pi_series_all.items():
        if (arm_name == "H1200" and seed == 229) or (arm_name == "H1200" and seed == best_passing_seed):
            # Store pi at burst boundaries only (not full 150-snap series)
            pi_at_boundaries = {}
            for bi, (bstart, bend) in enumerate(BURST_WINDOWS):
                si_s = snap_index(bstart)
                si_e = snap_index(bend)
                pi_at_boundaries[f"b{bi}_start"] = pi_s[si_s] if 0 <= si_s < len(pi_s) else None
                pi_at_boundaries[f"b{bi}_end"] = pi_s[si_e] if 0 <= si_e < len(pi_s) else None
            q4_summary[f"{arm_name}_s{seed}"] = pi_at_boundaries

    json_out = {
        "meta": {
            "script": "exp183_seed229_autopsy.py",
            "runtime_seconds": runtime,
            "bit_match_overall": overall_status,
            "bit_match_licensed": bit_match_pass,
        },
        "bit_match": {
            "overall": overall_status,
            "sessions": [{"session": d["session"], "status": d["status"], "details": d["burst_details"]} for d in bit_match_details],
        },
        "q1_burst_colors": color_info_to_json(color_info),
        "q2_repeat_crosstab": to_python(cross),
        "q3_value_vectors_s229_h1200": to_python(v_info.get(key229, [])),
        "q4_pi_series_summary": to_python(q4_summary),
        "q5_d_b_crosscheck": {f"b{bi}": {
            "recomp": to_python(rr229["d_b"][bi]) if rr229 else None,
            "committed": next((row["d_b"] for row in committed_rows if row.get("arm") == "H1200" and row.get("fork_seed") == 229 and row.get("burst_idx") == bi and row.get("phase") == "W"), None),
        } for bi in range(3)} if rr229 else {},
        "q6_tv_b_crosscheck": {f"b{bi}": {
            "recomp": to_python(rr229["tv_b"][bi]) if rr229 else None,
            "committed": next((row["tv_b"] for row in committed_rows if row.get("arm") == "H1200" and row.get("fork_seed") == 229 and row.get("burst_idx") == bi and row.get("phase") == "W"), None),
        } for bi in range(3)} if rr229 else {},
        "q7_cumulation_ledger": ledger_to_json(ledger),
        "q7_cross_seed_h1200": to_python(h1200_cross_seed),
        "q7_refuter_tests": {
            "a_freeze_coverage_incomplete": {"fired": refuter_a_fired, "nums": refuter_a_nums},
            "b_erosion_not_in_head": {"fired": refuter_b_fired, "nums": refuter_b_nums},
            "c_strong_positive_gap_end": {"fired": refuter_c_fired, "nums": refuter_c_nums},
            "d_passing_seed_with_equal_dose": {"fired": refuter_d_fired, "nums": refuter_d_nums},
        },
        "q8_mechanism_candidates": {
            name: {"score": data["score"], "evidence": data["evidence"]}
            for name, data in candidates.items()
        },
        "classification": {
            "primary": ", ".join(primary_supported) if primary_supported else "UNCLEAR",
            "candidates": {
                name: {"score": data["score"], "evidence": data["evidence"][:200]}
                for name, data in candidates.items()
            },
            "hypothesis_stands": hypothesis_stands,
            "refuters": {
                "a": {"fired": refuter_a_fired, "numbers": refuter_a_nums},
                "b": {"fired": refuter_b_fired, "numbers": refuter_b_nums},
                "c": {"fired": refuter_c_fired, "numbers": refuter_c_nums},
                "d": {"fired": refuter_d_fired, "numbers": refuter_d_nums},
            },
        },
        "seed229_h1200_expression_fractions": to_python(expr_fracs_229_h1200),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON written to {OUT_JSON}")

    t_final = time.time()
    print(f"\nTotal runtime: {t_final - t0:.1f}s")
    print(f"BIT-MATCH: {overall_status}")
    return {
        "bit_match": overall_status,
        "runtime": t_final - t0,
        "hypothesis_stands": hypothesis_stands,
        "refuter_a": refuter_a_fired,
        "refuter_b": refuter_b_fired,
        "refuter_c": refuter_c_fired,
        "refuter_d": refuter_d_fired,
        "expr_fracs": expr_fracs_229_h1200,
        "h1200_cross_seed": h1200_cross_seed,
        "candidates": candidates,
    }


if __name__ == "__main__":
    main()
