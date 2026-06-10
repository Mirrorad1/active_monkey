"""Persistent-creature substrate for the active-inference RECIPE.

Design intent
-------------
The RECIPE (continuous registered experience + ONE innate anchor + taught labels) is
extended *across* sessions and experiments rather than re-run from scratch each time.
A Creature's Dirichlet counts (``pA``) ARE its learned weights — they persist in git.
A committed state snapshot before and after an experiment makes any empirical claim
resumable. ``fork()`` deep-copies a snapshot and runs a counterfactual twin: the
scientific control that tests "its history made it this way."

Key invariants:
- ``qs`` (place belief) is NEVER reset except at ``birth``. Continuous registered
  experience is the point.
- Movement model (``B``) is INNATE/known — the one anchor. Only the sensory map ``pA``
  is learned.
- All randomness is derived from (manifest seed, rng_counter) so a resumed life is
  deterministic given the committed state, while successive ``live()`` calls differ.
- Biography (``BIOGRAPHY.jsonl``) is append-only — the honest log of a life.

Phase 1 instrumentation (structure-learning)
--------------------------------------------
``_surprise_window`` (deque, NOT hashed, NOT saved): rolling window of per-step surprise
values ``-ln(p(o_t))``.  Used by ``surprise_metrics()`` to detect learning plateaus
(surprise ceiling).

``_replay`` (list of (obs, action) tuples, NOT hashed, NOT saved): in-memory buffer
accumulated during ``live()``.  Flushed to ``<state_dir>/replay.bin`` by ``save_replay()``
which is called automatically by ``save()``.

replay.bin format: flat uint8 pairs [obs, action]*N.  obs < n_colors (< 256), action < 4.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Surprise-ceiling instrumentation constants (Phase 1)
# ---------------------------------------------------------------------------

#: Rolling window size for surprise tracking.
SURPRISE_WINDOW: int = 200

#: Ceiling detection: mean surprise above this value (nats) triggers ceiling flag.
#: 0.7 nats ≈ half of ln(3)=1.099 (uniform surprise over 3 colors).
CEILING_MEAN_THRESH: float = 0.7

#: Ceiling detection: absolute slope below this value (nats/step) triggers ceiling flag.
CEILING_SLOPE_THRESH: float = 5e-4


# ---------------------------------------------------------------------------
# World configuration
# ---------------------------------------------------------------------------

@dataclass
class World:
    """Rectangular grid world with a fixed, aliased color map.

    The color map is the latent structure the creature must discover.  Each cell
    has one color drawn from a palette of size n_colors; multiple cells may share
    the same color (aliasing), which is what makes localisation non-trivial.

    Attributes
    ----------
    rows, cols : grid dimensions.
    cmap : list of int, length rows*cols — color index per cell.
    n_colors : number of distinct colors.
    """
    rows: int
    cols: int
    cmap: list
    n_colors: int

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def move(self, cell: int, action: int) -> int:
        """Wall-clamped deterministic step.  Actions: 0=up, 1=down, 2=left, 3=right."""
        r, c = divmod(cell, self.cols)
        if action == 0:
            r = max(0, r - 1)
        elif action == 1:
            r = min(self.rows - 1, r + 1)
        elif action == 2:
            c = max(0, c - 1)
        else:
            c = min(self.cols - 1, c + 1)
        return r * self.cols + c

    def transition_matrix(self) -> np.ndarray:
        """Build the known (innate) movement model B, shape (n_cells, n_cells, 4).

        B[s', s, a] = 1.0 if move(s, a) == s', else 0.0.
        This is the ONE innate anchor.  Column-normalised (each column sums to 1).
        """
        n = self.n_cells
        B = np.zeros((n, n, 4))
        for s in range(n):
            for a in range(4):
                s_next = self.move(s, a)
                B[s_next, s, a] = 1.0
        return B

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "cmap": list(self.cmap),
            "n_colors": self.n_colors,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        return cls(
            rows=d["rows"],
            cols=d["cols"],
            cmap=d["cmap"],
            n_colors=d["n_colors"],
        )


# ---------------------------------------------------------------------------
# Creature
# ---------------------------------------------------------------------------

class Creature:
    """A creature whose learned state (weights, belief, values, vocab) persists in git.

    Parameters
    ----------
    name : str — unique identifier, used as the state directory name.
    world : World — the world this creature lives in.
    pA : np.ndarray, shape (n_colors, n_cells) — Dirichlet counts (the weights).
    qs : np.ndarray, shape (n_cells,) — current place belief.
    true_pos : int — ground-truth position in the world.
    value_counts : np.ndarray, shape (n_colors,) — grounded valence accumulators.
    vocab : dict[str, np.ndarray] — word → color-count array.
    age_steps : int — total steps lived.
    lineage : list[str] — fork ancestry chain.
    rng_counter : int — monotone counter for deterministic seed derivation.
    _seed : int — the birth seed (manifest seed).
    _state_dir : Path | None — bound when loaded/saved.

    Phase 1 instrumentation (NOT hashed, NOT saved in arrays.npz):
    _surprise_window : deque — rolling window of per-step surprise values (nats).
    _replay : list — in-memory (obs, action) pairs; flushed to replay.bin by save().
    _ceiling_events : list — dicts recording detected surprise-ceiling events.
    """

    def __init__(
        self,
        name: str,
        world: World,
        pA: np.ndarray,
        qs: np.ndarray,
        true_pos: int,
        value_counts: np.ndarray,
        vocab: dict,
        age_steps: int,
        lineage: list,
        rng_counter: int,
        _seed: int,
        _state_dir: Optional[Path] = None,
        *,
        surprise_window: Optional[int] = None,
    ):
        self.name = name
        self.world = world
        self.pA = pA.copy()
        self.qs = qs.copy()
        self.true_pos = int(true_pos)
        self.value_counts = value_counts.copy()
        self.vocab = {w: v.copy() for w, v in vocab.items()}
        self.age_steps = int(age_steps)
        self.lineage = list(lineage)
        self.rng_counter = int(rng_counter)
        self._seed = int(_seed)
        self._state_dir = _state_dir

        # --- Phase 1 instrumentation (NOT hashed, NOT saved in arrays.npz) ---
        _win = SURPRISE_WINDOW if surprise_window is None else int(surprise_window)
        self._surprise_window: deque = deque(maxlen=_win)
        self._replay: list = []          # (obs, action) tuples
        self._ceiling_events: list = []  # dicts recording detected ceiling events

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def birth(
        cls,
        name: str,
        world: World,
        seed: int = 0,
        *,
        surprise_window: Optional[int] = None,
    ) -> "Creature":
        """Create a fresh creature with uniform priors.  Belief is never reset after this."""
        rng = np.random.default_rng(seed)
        n_cells = world.n_cells
        n_colors = world.n_colors
        # Dirichlet concentration: small uniform + tiny seeded jitter (breaks symmetry).
        pA = np.full((n_colors, n_cells), 0.1) + 0.01 * rng.random((n_colors, n_cells))
        qs = np.ones(n_cells) / n_cells  # uniform place belief
        value_counts = np.zeros(n_colors)
        return cls(
            name=name,
            world=world,
            pA=pA,
            qs=qs,
            true_pos=0,
            value_counts=value_counts,
            vocab={},
            age_steps=0,
            lineage=[],
            rng_counter=0,
            _seed=seed,
            surprise_window=surprise_window,
        )

    # ------------------------------------------------------------------
    # Internals: sensory model & belief updates
    # ------------------------------------------------------------------

    def _A_hat(self) -> np.ndarray:
        """Column-normalised sensory likelihood matrix, shape (n_colors, n_cells)."""
        A = self.pA.copy()
        col_sums = A.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1.0, col_sums)
        return A / col_sums

    def _state_hash(self) -> str:
        """SHA-256 over the concatenated raw bytes of all state arrays."""
        h = hashlib.sha256()
        for arr in [self.pA, self.qs, self.value_counts]:
            h.update(arr.tobytes())
        # include vocab arrays in sorted word order for determinism
        for word in sorted(self.vocab):
            h.update(self.vocab[word].tobytes())
        return h.hexdigest()

    def _derive_rng(self) -> np.random.Generator:
        """Derive a reproducible RNG from (birth seed, rng_counter).

        Determinism guarantee: a resumed life (load → live) with the same rng_counter
        produces the same sequence as an uninterrupted run, because the seed is fully
        determined by the committed state.
        """
        combined_seed = (self._seed * 1_000_003 + self.rng_counter) & 0xFFFFFFFFFFFFFFFF
        return np.random.default_rng(combined_seed)

    # ------------------------------------------------------------------
    # Core life
    # ------------------------------------------------------------------

    def live(self, steps: int, seed: Optional[int] = None) -> None:
        """Wander for ``steps`` steps, learning the sensory map continuously.

        Belief (``qs``) carries across calls — this is the RECIPE's key invariant:
        continuous registered experience, never reset.  Each call increments
        ``rng_counter`` once so that successive calls differ while a resumed life
        (save → load → live) is deterministic.

        Learning mechanism (from Exp 20/21 pattern, pure numpy):
        - Observe color at true_pos.
        - Bayesian belief update: qs ∝ A_hat[obs, :] * (B[:, :, a_prev] @ qs_prev).
        - Dirichlet count learning: pA[obs, :] += qs  (soft count accumulation).
        - Value accumulation (Exp 26 mechanism): after the belief update, compute
          predictability weight = exp(-H(A_hat[:, argmax(qs)])) where H is the
          Shannon entropy of the predicted next observation distribution at the MAP
          cell.  Accumulate value_counts[obs] += predictability_weight.
          This means a color accrues value when encountering it coincides with high
          predictability (low surprise) — the functional "comfort" of Exp 26.
        - Pick a random action, wall-clamp true_pos, advance qs through B.

        Parameters
        ----------
        steps : int — number of steps to live.
        seed : int | None — if None, seed is derived deterministically from
            (self._seed, self.rng_counter).  Explicit seed overrides (useful for tests).
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self._derive_rng()

        B = self.world.transition_matrix()  # (n_cells, n_cells, 4)
        n_actions = 4

        for _ in range(steps):
            A_hat = self._A_hat()

            # --- observe ---
            obs = int(self.world.cmap[self.true_pos])

            # --- belief update: qs ∝ likelihood(obs) * prior(qs) ---
            likelihood = A_hat[obs, :]  # shape (n_cells,)

            # --- Phase 1: surprise instrumentation (reads pre-update qs) ---
            # p(o_t) = likelihood @ prior_qs  (using self.qs before reassignment)
            # surprise_t = -ln(p(o_t)); eps guards against zero probability
            eps = 1e-300
            p_o = float(likelihood @ self.qs)
            surprise_t = -math.log(p_o + eps)
            self._surprise_window.append(surprise_t)

            qs_updated = likelihood * self.qs
            denom = qs_updated.sum()
            if denom > 0:
                qs_updated = qs_updated / denom
            else:
                qs_updated = np.ones(self.world.n_cells) / self.world.n_cells

            # --- Dirichlet count learning: pA[obs, :] += qs (soft accumulation) ---
            self.pA[obs, :] += qs_updated

            # --- Value accumulation (Exp 26 mechanism) ---
            # Predictability weight: at MAP cell, how predictable is the next obs?
            map_cell = int(np.argmax(qs_updated))
            predicted_obs_dist = A_hat[:, map_cell]  # P(obs | map_cell)
            h_predicted = -np.sum(
                predicted_obs_dist * np.log(predicted_obs_dist + 1e-12)
            )  # entropy in nats
            predictability_weight = np.exp(-h_predicted)  # 1 = fully predictable, <1 = uncertain
            self.value_counts[obs] += predictability_weight

            # --- choose action, move ---
            action = int(rng.integers(0, n_actions))

            # --- Phase 1: replay buffer (obs, action) accumulated after action chosen ---
            self._replay.append((obs, action))

            self.true_pos = self.world.move(self.true_pos, action)

            # --- advance belief through movement model (innate B) ---
            self.qs = B[:, :, action] @ qs_updated

        self.age_steps += steps
        self.rng_counter += 1

        # --- Phase 1: ceiling detection (once per live() call, not per step) ---
        # Evaluate whether surprise has plateaued at a high floor (learning stalled).
        # Note: under count-decay variants, learning_active would test actual delta_pA;
        # here pA always accumulates so learning is always nominally active.
        ceiling_flag = False
        mean_s: Optional[float] = None
        slope: Optional[float] = None
        if len(self._surprise_window) == self._surprise_window.maxlen:
            win_arr = np.array(self._surprise_window)
            mean_s = float(win_arr.mean())
            slope = float(np.polyfit(np.arange(len(win_arr)), win_arr, 1)[0])
            learning_active = True  # pA always accumulates in this variant
            ceiling_flag = (
                learning_active
                and mean_s > CEILING_MEAN_THRESH
                and abs(slope) < CEILING_SLOPE_THRESH
            )
            if ceiling_flag:
                self._ceiling_events.append(
                    dict(age=self.age_steps, mean=mean_s, slope=slope)
                )

        # Append biography event if state directory is bound
        if mean_s is not None:
            summary = (
                f"lived {steps} steps; map_accuracy={self.map_accuracy():.3f}; "
                f"localize_bits={self.localize_bits():.3f}; "
                f"surprise_mean={mean_s:.3f} slope={slope:+.5f} ceiling={ceiling_flag}"
            )
        else:
            summary = (
                f"lived {steps} steps; map_accuracy={self.map_accuracy():.3f}; "
                f"localize_bits={self.localize_bits():.3f}"
            )
        self._bio_append({
            "event": "live",
            "age_steps": self.age_steps,
            "summary": summary,
            "state_hash": self._state_hash(),
        })

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def sensory_map(self) -> list:
        """Return argmax-tuning per cell: which color each cell most predicts.

        This is the learned sensory map — a direct read of the Dirichlet counts.
        """
        A_hat = self._A_hat()
        return [int(np.argmax(A_hat[:, s])) for s in range(self.world.n_cells)]

    def map_accuracy(self) -> float:
        """Fraction of cells whose learned tuning matches the true colormap."""
        learned = self.sensory_map()
        true = list(self.world.cmap)
        if len(learned) != len(true):
            return 0.0
        return sum(l == t for l, t in zip(learned, true)) / len(true)

    def localize_bits(self) -> float:
        """Shannon entropy of current place belief qs, in bits.  0 = perfectly localised."""
        p = self.qs / (self.qs.sum() + 1e-300)
        return float(-np.sum(p * np.log2(p + 1e-300)))

    def surprise_metrics(self) -> dict:
        """Return surprise-ceiling metrics computed from the current rolling window.

        Returns a dict with keys:
          mean       : float | None — mean surprise (nats) over the window, or None if
                       the window is not yet full.
          slope      : float | None — least-squares nats/step slope, or None if not full.
          ceiling_flag : bool — True iff window is full AND mean > CEILING_MEAN_THRESH
                         AND |slope| < CEILING_SLOPE_THRESH.
          window_len : int — current number of entries in the window.
          events     : list — copy of self._ceiling_events detected so far.
        """
        wlen = len(self._surprise_window)
        if wlen < self._surprise_window.maxlen:
            return dict(mean=None, slope=None, ceiling_flag=False,
                        window_len=wlen, events=list(self._ceiling_events))
        win_arr = np.array(self._surprise_window)
        mean_s = float(win_arr.mean())
        slope = float(np.polyfit(np.arange(wlen), win_arr, 1)[0])
        ceiling_flag = (
            mean_s > CEILING_MEAN_THRESH
            and abs(slope) < CEILING_SLOPE_THRESH
        )
        return dict(mean=mean_s, slope=slope, ceiling_flag=ceiling_flag,
                    window_len=wlen, events=list(self._ceiling_events))

    def save_replay(self, dir_path=None) -> Optional[Path]:
        """Flush the in-memory replay buffer to ``<dir_path>/replay.bin``.

        Appends flat uint8 pairs [obs, action]*N to any existing file, then clears
        ``self._replay``.

        Format: raw bytes, each step = 2 uint8s: [obs, action].
        obs < n_colors (< 256); action < 4.

        Parameters
        ----------
        dir_path : path-like | None — directory to write into.  If None, falls back
            to ``self._state_dir``.  If both are None, returns None without writing.

        Returns
        -------
        Path to the replay file, or None if no directory is bound.
        """
        target = Path(dir_path) if dir_path is not None else self._state_dir
        if target is None:
            return None
        if not self._replay:
            return target / "replay.bin"
        replay_arr = np.array(self._replay, dtype=np.uint8)
        replay_path = target / "replay.bin"
        with replay_path.open("ab") as fh:
            fh.write(replay_arr.tobytes())
        self._replay = []
        return replay_path

    # ------------------------------------------------------------------
    # Values & language
    # ------------------------------------------------------------------

    def favorite(self) -> int:
        """Return the color index with highest accumulated value."""
        return int(np.argmax(self.value_counts))

    def conviction(self) -> float:
        """Normalised max of value distribution — how strong the preference is (0–1)."""
        total = self.value_counts.sum()
        if total == 0:
            return 0.0
        return float(self.value_counts.max() / total)

    def teach_word(self, word: str, color_idx: int, n: int = 8) -> None:
        """Associate a word with a color via n few-shot examples (Exp 34 pattern).

        The creature learns P(word | color) by accumulating soft counts.
        The word-to-color bridge is TAUGHT; the content (what the creature values)
        is SELF-FORMED.  This mirrors the honest framing in Exp 34/35.
        """
        if word not in self.vocab:
            self.vocab[word] = np.ones(self.world.n_colors) * 0.1
        self.vocab[word][color_idx] += n

        self._bio_append({
            "event": "teach_word",
            "age_steps": self.age_steps,
            "summary": f"taught word '{word}' -> color {color_idx} (n={n})",
            "state_hash": self._state_hash(),
        })

    def answer_what_do_you_like(self) -> str:
        """Answer 'what do you like?' in taught words.

        Content is self-formed (value_counts from lived experience).
        Labels are taught (vocab mapping color → word).  Honest framing per RECIPE.
        """
        fav = self.favorite()
        word = self._word_for_color(fav)
        if word is not None:
            return f"I like {word}"
        return f"I like color-{fav} (no word taught yet)"

    def answer_do_you_like(self, word: str) -> str:
        """Answer 'do you like <word>?' using self-formed values and taught labels.

        Returns a statement about whether the queried concept is valued or not,
        with the surprise level for honest calibration.
        """
        color = self._color_for_word(word)
        if color is None:
            return f"I don't know what '{word}' means"
        total = self.value_counts.sum()
        if total == 0:
            return f"I haven't experienced enough to say"
        val_frac = float(self.value_counts[color] / total)
        A_hat = self._A_hat()
        fav_cell = int(np.argmax(self.qs))
        h_bits = float(
            -np.sum(A_hat[:, fav_cell] * np.log2(A_hat[:, fav_cell] + 1e-300))
        )
        if val_frac > 0.5 / self.world.n_colors * 2.0:  # above even-share threshold
            return f"I like {word} (surprise {h_bits:.1f} bits)"
        else:
            return f"{word} unsettles me (surprise {h_bits:.1f} bits)"

    def _word_for_color(self, color: int) -> Optional[str]:
        """Return the best word for a color, or None if no words taught."""
        if not self.vocab:
            return None
        best_word, best_score = None, -1.0
        for word, counts in self.vocab.items():
            total = counts.sum()
            if total > 0:
                score = float(counts[color] / total)
                if score > best_score:
                    best_score, best_word = score, word
        return best_word

    def _color_for_word(self, word: str) -> Optional[int]:
        """Return the most associated color for a word, or None if word unknown."""
        if word not in self.vocab:
            return None
        return int(np.argmax(self.vocab[word]))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path) -> None:
        """Save state to ``dir_path``.

        Creates:
          - ``arrays.npz``: pA, qs, value_counts, vocab arrays.
          - ``manifest.json``: scalars + world config + state_hash.

        The state_hash over the concatenated array bytes lets ``load`` verify
        integrity.  Appends a BIOGRAPHY event.

        Parameters
        ----------
        dir_path : path-like — directory to write into (created if absent).
        """
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        self._state_dir = dir_path

        # Pack vocab arrays into npz-compatible named arrays
        vocab_keys = sorted(self.vocab.keys())
        vocab_arrays = {f"vocab__{w}": self.vocab[w] for w in vocab_keys}

        np.savez(
            dir_path / "arrays.npz",
            pA=self.pA,
            qs=self.qs,
            value_counts=self.value_counts,
            **vocab_arrays,
        )

        hash_ = self._state_hash()
        manifest = {
            "name": self.name,
            "lineage": self.lineage,
            "age_steps": self.age_steps,
            "true_pos": self.true_pos,
            "world": self.world.to_dict(),
            "rng_counter": self.rng_counter,
            "seed": self._seed,
            "state_hash": hash_,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (dir_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

        self._bio_append({
            "event": "save",
            "age_steps": self.age_steps,
            "summary": f"saved to {dir_path}",
            "state_hash": hash_,
        })

        # Phase 1: flush replay buffer (appends to existing replay.bin if present)
        self.save_replay(dir_path)

    @classmethod
    def load(cls, dir_path) -> "Creature":
        """Load creature from ``dir_path``.  Verifies state_hash integrity.

        The loaded creature resumes exactly where it left off — same qs, pA,
        true_pos, rng_counter — so a subsequent ``live()`` call is deterministic.
        """
        dir_path = Path(dir_path)
        manifest = json.loads((dir_path / "manifest.json").read_text())
        arrs = np.load(dir_path / "arrays.npz", allow_pickle=False)

        pA = arrs["pA"]
        qs = arrs["qs"]
        value_counts = arrs["value_counts"]
        vocab = {}
        for key in arrs.files:
            if key.startswith("vocab__"):
                word = key[len("vocab__"):]
                vocab[word] = arrs[key]

        world = World.from_dict(manifest["world"])
        c = cls(
            name=manifest["name"],
            world=world,
            pA=pA,
            qs=qs,
            true_pos=manifest["true_pos"],
            value_counts=value_counts,
            vocab=vocab,
            age_steps=manifest["age_steps"],
            lineage=manifest["lineage"],
            rng_counter=manifest["rng_counter"],
            _seed=manifest["seed"],
            _state_dir=dir_path,
        )

        # Verify integrity
        computed = c._state_hash()
        stored = manifest.get("state_hash", "")
        if stored and computed != stored:
            raise ValueError(
                f"state_hash mismatch for '{manifest['name']}': "
                f"stored={stored[:12]}… computed={computed[:12]}…"
            )
        return c

    # ------------------------------------------------------------------
    # Fork (counterfactual control)
    # ------------------------------------------------------------------

    def fork(self, new_name: str) -> "Creature":
        """Deep-copy this creature, recording lineage for counterfactual tracking.

        Fork is the scientific control mechanism: a forked twin shares the same
        history up to this point, but can be placed in a different world or run with
        a different seed.  If the twin develops different values or knowledge, that
        difference is causally attributable to post-fork experience, not prior
        history.

        Returns
        -------
        Creature — a deep copy with lineage extended to record the parent's name,
        age, and state_hash.
        """
        twin = copy.deepcopy(self)
        twin.name = new_name
        twin._state_dir = None  # unbound: caller must save() to bind
        twin.lineage = self.lineage + [
            f"{self.name}@{self.age_steps}#{self._state_hash()[:12]}"
        ]
        self._bio_append({
            "event": "fork",
            "age_steps": self.age_steps,
            "summary": f"forked into '{new_name}'",
            "state_hash": self._state_hash(),
        })
        return twin

    # ------------------------------------------------------------------
    # Biography
    # ------------------------------------------------------------------

    def _bio_path(self) -> Optional[Path]:
        if self._state_dir is None:
            return None
        return self._state_dir / "BIOGRAPHY.jsonl"

    def _bio_append(self, record: dict) -> None:
        """Append one event to the biography file (append-only)."""
        path = self._bio_path()
        if path is None:
            return
        with path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Creature(name={self.name!r}, age={self.age_steps}, "
            f"world={self.world.rows}x{self.world.cols}, "
            f"map_accuracy={self.map_accuracy():.2f}, "
            f"localize={self.localize_bits():.2f}bits)"
        )
