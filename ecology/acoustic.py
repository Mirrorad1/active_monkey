"""ecology/acoustic.py — A PHYSICAL acoustic substrate for the patch-mosaic
predator-prey world (Exp 268+, direction: acoustic-ecology).

Sound is built here as a PHYSICAL ENVIRONMENTAL SUBSTRATE, not as messages.  An
ordinary action (movement, attack, feeding, reproduction, death) radiates energy
into a coarse three-band spectrum (low / mid / high).  That energy PROPAGATES
across the patch graph, ATTENUATES with distance (inverse-square-like), is
ABSORBED frequency-dependently (high bands decay fastest), arrives LATE
(propagation delay = distance / sound_speed), and FADES in time.  A listener
perceives only the per-band received intensity against per-band ambient noise,
and detects a band only when its signal-to-noise ratio clears a threshold.

DISTANCE on this substrate.  The patch mosaic is mean-field WITHIN a patch (no
continuous intra-patch position), so the only physically meaningful notion of
distance is the GRAPH-HOP distance between patches on the mosaic topology
(ring / grid2d / smallworld).  A sound emitted in patch j is heard in patch i
attenuated by the hop distance d(i, j) and delayed by round(d(i, j)/sound_speed)
steps.  This is a faithful, honest reduction: the topology IS the space.

WHAT AGENTS NEVER RECEIVE (binding, enforced by construction).  Agents see only
imperfect per-band received intensity (optionally a directional gradient across
their patch's neighbors).  They never receive: the source position, the source
identity, the event type, a predator/prey label, or any semantic tag.  The
spectra below are PHYSICAL SIGNATURES (an attack happens to be broadband and
loud), never labels — the agent is never told "this band means predator".

This module is PURE physics + analysis: no rng of its own, no global evaluator,
no selection.  ecology/patchmosaic.py owns the simulation; it calls these helpers
behind `enable_acoustic_field` (default OFF, byte-identical).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# Bands, fixed coarse spectral resolution: low / mid / high.
BANDS = ("low", "mid", "high")
N_BANDS = 3


# ---------------------------------------------------------------------------
# Event signatures — PHYSICAL, not semantic.
# (source_amplitude, per-band spectrum fractions [low, mid, high]).
# These are the source spectra; propagation/attenuation is applied at receive.
# ---------------------------------------------------------------------------
def default_event_signatures() -> Dict[str, Tuple[float, Tuple[float, float, float]]]:
    """Return the default per-event-type (amplitude, spectrum) signatures.

    - movement:      low/mid, LOW amplitude          (a body shifting)
    - attack:        broadband, HIGH amplitude        (a strike / lunge)
    - feeding:       mid, MEDIUM amplitude            (foraging / consuming)
    - reproduction:  low/mid, configurable amplitude
    - death:         broadband, HIGH amplitude        (a struggle / fall)

    Spectra are PHYSICAL signatures, not labels exposed to agents.
    """
    return {
        "movement":     (0.30, (0.60, 0.40, 0.00)),
        "attack":       (1.00, (0.34, 0.33, 0.33)),
        "feeding":      (0.50, (0.20, 0.60, 0.20)),
        "reproduction": (0.40, (0.50, 0.50, 0.00)),
        "death":        (1.00, (0.34, 0.33, 0.33)),
    }


@dataclass
class AcousticConfig:
    """All acoustic constants — configurable, with physically-motivated defaults.

    These live as fields on PatchMosaicConfig too (flat, for determinism-hashing
    simplicity); this dataclass is the standalone bundle the AcousticField uses.
    """
    sound_speed: float = 4.0                # hops per step; delay = round(dist/speed)
    attenuation_power: float = 2.0          # inverse-square-like (1/(floor+d)^p)
    distance_floor: float = 1.0             # avoids div-by-zero at d=0
    # per-band [low, mid, high]; HIGH must absorb fastest (high > mid > low).
    freq_absorption: Tuple[float, float, float] = (0.05, 0.15, 0.40)
    time_decay: Tuple[float, float, float] = (0.20, 0.35, 0.60)
    ambient_noise: Tuple[float, float, float] = (0.05, 0.04, 0.03)
    detection_threshold: float = 2.0        # SNR floor: received/ambient >= thr to detect
    buffer_window: int = 24                 # steps of emission history retained

    # Heritable / gifted hearing traits (monomorphic "gifted" values in the probe).
    hearing_sensitivity: float = 1.0        # scales received intensity (>1 hears weaker sound)
    hearing_precision: float = 0.0          # 0 = perfect; >0 = multiplicative log-normal-ish noise
    hearing_bandwidth: int = 3              # how many bands resolved (1..3); <3 folds high into mid
    directional_hearing: bool = True        # may compare across neighbor patches

    def absorption(self) -> np.ndarray:
        return np.asarray(self.freq_absorption, dtype=float)

    def tdecay(self) -> np.ndarray:
        return np.asarray(self.time_decay, dtype=float)

    def ambient(self) -> np.ndarray:
        return np.asarray(self.ambient_noise, dtype=float)


# ---------------------------------------------------------------------------
# Hop-distance (the substrate's notion of physical distance)
# ---------------------------------------------------------------------------
def all_pairs_hops(neighbors: List[List[int]]) -> np.ndarray:
    """BFS all-pairs shortest-path (hop) distance over the patch graph.

    neighbors[i] is the sorted adjacency list of patch i.  Returns an (n, n)
    float matrix; unreachable pairs (shouldn't happen — graphs are connected)
    are set to a large finite value.
    """
    n = len(neighbors)
    INF = float(n + 1)
    dist = np.full((n, n), INF, dtype=float)
    for src in range(n):
        dist[src, src] = 0.0
        frontier = deque([src])
        seen = {src}
        while frontier:
            u = frontier.popleft()
            for v in neighbors[u]:
                if v not in seen:
                    seen.add(v)
                    dist[src, v] = dist[src, u] + 1.0
                    frontier.append(v)
    return dist


class AcousticField:
    """Holds the propagation geometry + a rolling emission buffer and computes,
    each step, the per-patch per-band RECEIVED intensity.

    Deterministic and rng-free: given the same emission history it returns the
    same field.  Perception NOISE (hearing_precision) is applied by the caller
    using its own seeded generator, so this object never touches rng.
    """

    def __init__(self, acfg: AcousticConfig, neighbors: List[List[int]]):
        self.acfg = acfg
        self.n = len(neighbors)
        self.hops = all_pairs_hops(neighbors)
        # Propagation delay (steps) per (listener, source) pair.
        speed = max(1e-9, acfg.sound_speed)
        self.delay = np.rint(self.hops / speed).astype(int)
        # Precompute the per-(i,j,band) static gain = 1/(floor+d)^p * exp(-absorp_b * d).
        floor = acfg.distance_floor
        p = acfg.attenuation_power
        absorp = acfg.absorption()  # (3,)
        d = self.hops  # (n, n)
        geom = 1.0 / np.power(floor + d, p)             # (n, n)
        # (n, n, 3): geom * exp(-absorp_b * d)
        self.gain = geom[:, :, None] * np.exp(-absorp[None, None, :] * d[:, :, None])
        # Rolling buffer: each entry is (emit_time, emit_array[n, 3]).
        self._buffer: deque = deque(maxlen=acfg.buffer_window)

    # --- emission -------------------------------------------------------
    def push_emissions(self, t: int, emit: np.ndarray) -> None:
        """Record this step's per-patch per-band emitted energy (n, 3)."""
        self._buffer.append((int(t), np.asarray(emit, dtype=float).copy()))

    # --- propagation ----------------------------------------------------
    def received(self, t: int) -> np.ndarray:
        """Per-patch per-band received intensity at time t (n, 3).

        Sums every buffered emission that has ARRIVED by t:
          arrival = emit_time + delay(listener, source); audible iff arrival <= t.
          contribution = emit[source,band] * gain[listener,source,band]
                         * exp(-time_decay_band * (t - arrival)).
        """
        acfg = self.acfg
        tdecay = acfg.tdecay()  # (3,)
        out = np.zeros((self.n, N_BANDS), dtype=float)
        for emit_time, emit in self._buffer:
            if emit.sum() == 0.0:
                continue
            # arrival[i, j] = emit_time + delay[i, j]; age = t - arrival
            age = t - (emit_time + self.delay)            # (n, n)
            audible = age >= 0                            # bool (n, n)
            if not audible.any():
                continue
            # time-decay per (i, j, band)
            tdf = np.exp(-tdecay[None, None, :] * np.clip(age, 0, None)[:, :, None])
            # contribution[i, b] = sum_j emit[j,b] * gain[i,j,b] * tdf[i,j,b] * audible[i,j]
            contrib = (
                emit[None, :, :]                # (1, n, 3) source energy per band
                * self.gain                     # (n, n, 3) distance+absorption gain
                * tdf                           # (n, n, 3) time decay
                * audible[:, :, None]           # (n, n, 1) arrival mask
            )
            out += contrib.sum(axis=1)
        return out

    def attenuation_curve(self) -> List[Tuple[int, float]]:
        """Diagnostic: received band-summed gain for a unit broadband source at
        each distinct hop distance (listener 0's view), for the range-calibration
        abort check (detection must FALL with distance)."""
        n = self.n
        curve: Dict[int, float] = {}
        for j in range(n):
            d = int(round(self.hops[0, j]))
            g = float(self.gain[0, j, :].sum())
            curve.setdefault(d, g)
        return sorted(curve.items())


# ---------------------------------------------------------------------------
# Perception: turn a true received field into what a (gifted) listener detects.
# ---------------------------------------------------------------------------
def perceive(received_i: np.ndarray, acfg: AcousticConfig,
             noise_rng=None) -> np.ndarray:
    """Return the listener's per-band perceived intensity for one patch.

    - hearing_sensitivity scales the received signal (more sensitive => hears
      weaker sound).
    - hearing_precision injects multiplicative noise (0 => perfect estimate).
    - hearing_bandwidth < 3 folds the unresolved high band into mid (coarser ear).
    noise_rng (if given) is a numpy Generator the CALLER owns — keeps this module
    rng-free and the main simulation stream byte-identical.
    """
    est = np.asarray(received_i, dtype=float) * acfg.hearing_sensitivity
    if acfg.hearing_precision > 0.0 and noise_rng is not None:
        factor = np.exp(noise_rng.normal(0.0, acfg.hearing_precision, size=est.shape))
        est = est * factor
    if acfg.hearing_bandwidth < 3:
        # Fold high into mid (can't resolve the top band).
        est = est.copy()
        est[1] += est[2]
        est[2] = 0.0
    return est


def detected_bands(perceived_i: np.ndarray, acfg: AcousticConfig) -> np.ndarray:
    """Boolean per-band detection vector: SNR >= threshold."""
    amb = acfg.ambient()
    snr = perceived_i / np.maximum(amb, 1e-12)
    return snr >= acfg.detection_threshold


# ---------------------------------------------------------------------------
# Analysis: mutual information & detection error rates (ANALYSIS-ONLY).
# Agents never receive these; they quantify whether the channel carries bits
# about the hidden predator state.
# ---------------------------------------------------------------------------
def mutual_information_bits(obs: List[int], hidden: List[int]) -> float:
    """MI(obs; hidden) in bits, from paired discrete samples.  >=0; 0 = independent."""
    if not obs:
        return 0.0
    obs_a = np.asarray(obs)
    hid_a = np.asarray(hidden)
    n = len(obs_a)
    mi = 0.0
    for o in np.unique(obs_a):
        po = np.mean(obs_a == o)
        for h in np.unique(hid_a):
            ph = np.mean(hid_a == h)
            joint = np.mean((obs_a == o) & (hid_a == h))
            if joint > 0.0 and po > 0.0 and ph > 0.0:
                mi += joint * math.log2(joint / (po * ph))
    return float(max(0.0, mi))


def detection_error_rates(detected: List[bool], near: List[bool]) -> Tuple[float, float]:
    """Return (false_positive_rate, false_negative_rate).

    FP = P(detected | not near); FN = P(not detected | near).  'near' is the
    ground-truth hidden predator-proximity event (analysis-only).
    """
    det = np.asarray(detected, dtype=bool)
    nr = np.asarray(near, dtype=bool)
    fp = float(np.mean(det[~nr])) if (~nr).any() else 0.0
    fn = float(np.mean(~det[nr])) if nr.any() else 0.0
    return fp, fn
