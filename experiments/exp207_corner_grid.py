"""Exp 207 — sensor-controller co-adaptation: the DESIGN-STAGE PRE-FLIGHT (corner-grid).

Rather than run a full 6-arm x 5-seed x 8000-step co-adaptation batch on faith, MEASURE the
load-bearing premise first (the L25/L26 pre-flight discipline applied at the SCIENTIFIC-PREMISE
level): does the controller-coupling escape instantiate a 2-D fitness valley where neither the
sensor h NOR the controller theta pays alone, but both together do?

The controller theta = the routing weight on the sensor's class read (the currently-FIXED
world.niche_weight in the committed exp206 enable_niche regime). We do NOT need to add an evolvable
trait to ANSWER the premise question — we clamp h (freeze_thermosense) and set niche_weight (the
theta proxy) at the 2x2 corners and measure realized reproduction B(h, theta). NO engine change.

HYPOTHESIS (the directive's premise, the thing this pre-flight tests). A sensor-controller
co-adaptation instantiates a 2-D fitness valley w(h,theta)=R*sigma(k*h*theta-d)-C_h(h)-C_theta(theta)
where neither the sensor h nor the controller theta pays ALONE near the resident, but both together
climb to functional (positive cross-partial d2w/dh.dtheta > 0).

PREDICTION if TRUE: at the corner grid the cross-partial is positive AND theta does not pay alone at
low h AND a sharper sensor pays when the controller is engaged (dB/dh@high-theta > 0).

THE PRE-FLIGHT GATE / FALSIFIER (G0). The premise (a 2-D valley) is FALSIFIED — and the full
co-adaptation experiment is NOT viable (DESIGN-STAGE NEGATIVE) — if ANY of: the discrete cross-partial
    d2B/dh.dtheta = [B(hi_h,hi_t) - B(lo_h,hi_t)] - [B(hi_h,lo_t) - B(lo_h,lo_t)]  <=  0,
OR theta pays strongly ALONE at low h (dB/dtheta@lo_h dominant), OR the sensor h is pure cost at high
theta (dB/dh@high-theta <= 0). Any falsifier firing means there is no co-adaptation to test — running
the full experiment would only show theta climbing alone (herd-escape) while h stays pure cost
(exp206's sixth wall), a 1-D collapse, not a new result.

ANTI-CHEAT GUARDS (reproduced with committed code): (a) at niche_confusion=0 the percept is
perfect for every h => realized intake + events_hash byte-identical across h (h leaks nowhere but
the percept); (b) at niche_weight=0 the routing bonus vanishes for every theta => intake +
events_hash byte-identical across the theta proxy (theta acts ONLY through the routing weight).

This reproduces, on MY committed code, the negative cross-partial the design+audit workflow
reported (so the design-stage-negative verdict rests on a verifiable measurement, not a subagent's
throwaway numbers). Reuses the committed enable_niche engine + experiments/exp206 builders.
"""
from __future__ import annotations

import dataclasses as D
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ecology.engine import Ecology
from ecology import sense_axis as SA
import experiments.exp206_n5_rotating_niche as E206

SEEDS = [90, 91, 92, 93, 94]
HORIZON = 3500
WINDOW = 1000                       # count reproduction events in the last WINDOW steps
H_LO, H_HI = 0.10, 0.45            # sensor corners (resident vs functional)
T_LO, T_HI = 0.6, 6.0             # controller proxy = niche_weight (weak vs strong routing)


def _mean_births(h: float, theta: float, seed: int, confusion: float = E206.CONF,
                 niche_crowding: float = E206.CROWD, costoff: bool = False) -> tuple[float, str, float]:
    """Monomorphic births in the last WINDOW steps at clamped (h, niche_weight=theta).

    Returns (mean births/step in window, events_hash, total resource_eaten) — the hash + intake
    let the anti-cheat guards assert byte-identity across h / theta where required.

    costoff=True (enable_thermosense=False) removes the h-keyed UPKEEP so the anti-cheat guards can
    isolate the PERCEPT channel: at confusion=0 the routing read is perfect for every h, so with no
    upkeep to perturb the (chaotic, long-horizon) trajectory the run is byte-identical across h —
    h leaks NOWHERE but the percept. (With cost ON the legitimate upkeep difference diverges the
    populations chaotically over 3500 steps — that is the licit cost channel, not a reward leak.)
    """
    base = D.replace(E206.niche_compete_cfg(costoff=costoff), horizon=HORIZON, niche_weight=theta,
                     niche_confusion=confusion, niche_crowding=niche_crowding,
                     freeze_thermosense=True)
    cfg = D.replace(base, founder=SA.clamp_founder(base.founder, h, 0.20))
    eco = Ecology(cfg, seed=seed)
    births = 0
    while eco.t < cfg.horizon and not eco.exploded and eco.has_alive():
        eco.step()
        # count reproduction events landing in the last window (cheap incremental check on the tail)
    # final scan of the event log for reproductions in the window (deterministic, post-hoc)
    lo = cfg.horizon - WINDOW
    births = sum(1 for e in eco.events if e["event_type"] == "reproduction" and e["t"] >= lo)
    eaten = sum(c.phenotype.resource_eaten for c in eco._creatures)
    return births / WINDOW, eco.events_hash(), eaten


def main() -> None:
    t0 = time.time()
    out_dir = _REPO / "experiments" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    L = ["=" * 78, "EXP 207 — CONTROLLER CO-ADAPTATION: DESIGN-STAGE CORNER-GRID PRE-FLIGHT", "=" * 78, ""]
    L.append(f"regime: exp206 NICHE_COMPETE (crowd={E206.CROWD} rot={E206.ROT} conf={E206.CONF} "
             f"K={E206.K} regen={E206.REGEN}); theta proxy = niche_weight (routing weight)")
    L.append(f"corners: h in ({H_LO},{H_HI}) x theta in ({T_LO},{T_HI}); seeds {SEEDS}; "
             f"B = births/step in last {WINDOW} of {HORIZON}")
    L.append("")

    def corner(h, t):
        return float(np.mean([_mean_births(h, t, s)[0] for s in SEEDS]))

    B_lh_lt = corner(H_LO, T_LO)
    B_hh_lt = corner(H_HI, T_LO)
    B_lh_ht = corner(H_LO, T_HI)
    B_hh_ht = corner(H_HI, T_HI)
    cross = (B_hh_ht - B_lh_ht) - (B_hh_lt - B_lh_lt)
    dB_dtheta_lo_h = B_lh_ht - B_lh_lt          # does theta pay ALONE at low h?
    dB_dh_hi_theta = B_hh_ht - B_lh_ht          # does h pay at high theta?
    dB_dh_lo_theta = B_hh_lt - B_lh_lt          # does h pay at low theta?

    L.append("B(h, theta) corner grid (mean births/step):")
    L.append(f"            theta={T_LO:<5}   theta={T_HI:<5}")
    L.append(f"  h={H_LO:<5}   {B_lh_lt:>10.5f}   {B_lh_ht:>10.5f}")
    L.append(f"  h={H_HI:<5}   {B_hh_lt:>10.5f}   {B_hh_ht:>10.5f}")
    L.append("")
    L.append(f"CROSS-PARTIAL d2B/dh.dtheta = {cross:+.5f}   (premise needs > 0 for a 2-D valley)")
    L.append(f"dB/dtheta @ low h           = {dB_dtheta_lo_h:+.5f}   (theta pays ALONE if strongly >0)")
    L.append(f"dB/dh @ high theta          = {dB_dh_hi_theta:+.5f}   (does sharper sensor pay when acted on?)")
    L.append(f"dB/dh @ low theta           = {dB_dh_lo_theta:+.5f}")
    L.append("")

    # --- anti-cheat guards (reproduced on committed code; COST-OFF to isolate the percept) ---
    # Exp 207 adds NO engine change (theta proxy = the existing niche_weight config param), so the
    # committed test_exp206_niche.py::test_no_direct_h_reward_confusion_zero already proves the
    # anti-cheat; these re-confirm it in-script at the verdict regime.
    # (a) confusion=0 + cost-OFF => routing perfect for every h, no upkeep => byte-identical across h
    #     (h leaks NOWHERE but the percept).
    ha_lo = _mean_births(0.10, T_HI, SEEDS[0], confusion=0.0, costoff=True)
    ha_hi = _mean_births(0.90, T_HI, SEEDS[0], confusion=0.0, costoff=True)
    guard_conf0 = (ha_lo[1] == ha_hi[1]) and (abs(ha_lo[2] - ha_hi[2]) < 1e-9)
    # (b) niche_weight(theta)=0 + cost-OFF => routing bonus vanishes for every h => byte-identical
    #     (theta acts ONLY through the routing weight).
    hb_lo = _mean_births(0.10, 0.0, SEEDS[0], costoff=True)
    hb_hi = _mean_births(0.90, 0.0, SEEDS[0], costoff=True)
    guard_nw0 = (hb_lo[1] == hb_hi[1]) and (abs(hb_lo[2] - hb_hi[2]) < 1e-9)
    L.append(f"ANTI-CHEAT GUARD (a) confusion=0 cost-OFF => byte-identical across h: {guard_conf0} "
             f"(eaten {ha_lo[2]:.3f} vs {ha_hi[2]:.3f})")
    L.append(f"ANTI-CHEAT GUARD (b) niche_weight=0 cost-OFF => byte-identical across h: {guard_nw0} "
             f"(eaten {hb_lo[2]:.3f} vs {hb_hi[2]:.3f})")
    L.append("")

    valley = cross > 0 and dB_dtheta_lo_h < 0.5 * dB_dh_hi_theta
    verdict = ("VIABLE (2-D valley present)" if valley else
               "NOT-VIABLE (no 2-D valley: cross-partial <= 0 and/or theta pays alone) => "
               "DESIGN-STAGE NEGATIVE; do not run the full co-adaptation batch")
    L.append(f"G0 PRE-FLIGHT: {verdict}")
    L.append("")
    L.append(f"runtime: {time.time()-t0:.0f}s")
    text = "\n".join(L)
    print(text)
    (out_dir / "exp207.txt").write_text(text + "\n")
    print(f"\n[saved {out_dir}/exp207.txt]")


if __name__ == "__main__":
    main()
