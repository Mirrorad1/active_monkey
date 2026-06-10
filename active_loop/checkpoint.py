"""Checkpointed continuous-life episodes for the persistent creature (the spine).

Convention (loop/PROTOCOL.md, Exp 58+): every persistent-creature experiment advances
ONE continuous life — the spine, `mirro` by default. An experiment does not re-birth
or branch the spine; it loads the committed snapshot, lives, and saves back, so the
world-model accumulates as a single life that is never reset.

`mirro_episode` makes that guaranteed and uniform:

    from active_loop.checkpoint import mirro_episode

    with mirro_episode("Exp 58") as ep:
        ep.creature.live(500)
        # ... measurements, teach_word, etc. on ep.creature ...
        control = ep.fork_control("exp58-twinA")   # side-control, NOT the spine
        control.live(500)

On entry it loads the spine and records a checkpoint BEFORE (age + state_hash, and a
`checkpoint_before` biography event). On clean exit it saves the spine and records a
checkpoint AFTER. `ep.report()` returns the before/after block to paste into the
EXPERIMENTS.md entry. The helper never git-commits — the experiment's atomic commit
(PROTOCOL step 6) commits the updated snapshot.

Forks remain the scientific control, but only as SIDE-controls: `ep.fork_control(name)`
branches from the spine's pre-experiment state and is saved elsewhere (or not at all).
A fork never becomes the spine, so the spine's continuity is preserved.

Clade model: the spine is the ROOT ANCESTOR / trunk. A branch lived long enough in a
different world is, functionally, a different species; promote it to a committed peer line
with `twin.save(Path("creature/state") / twin.name)`. Every committed checkpoint is also a
restore point — recover from a bad epoch with `git checkout <commit> -- creature/state/mirro/`
rather than wiping in place. The continuity guard (`tests/test_creature_continuity.py`)
forbids only a SILENT trunk reset; a deliberate restart is allowed when logged as an explicit
`rebirth` biography event.

If the body raises, the spine is left untouched (no half-mutated state is saved), so a
failed experiment cannot corrupt the continuous life.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from active_loop.creature import Creature

# The spine — the one continuous creature every Exp 58+ persistent experiment advances.
MIRRO_DIR = Path("creature/state/mirro")


class Episode:
    """Handle to one checkpointed episode of the continuous life.

    Attributes
    ----------
    exp_id : str — the experiment label, e.g. "Exp 58".
    creature : Creature — the loaded spine; live()/teach_word() it directly.
    state_dir : Path — the spine's snapshot directory.
    age_before, hash_before : int, str — checkpoint recorded on entry.
    age_after, hash_after : int | str | None — filled in on clean exit.
    """

    def __init__(self, exp_id: str, creature: Creature, state_dir: Path):
        self.exp_id = exp_id
        self.creature = creature
        self.state_dir = Path(state_dir)
        self.age_before: int = creature.age_steps
        self.hash_before: str = creature._state_hash()
        self.age_after: Optional[int] = None
        self.hash_after: Optional[str] = None

    def fork_control(self, name: str) -> Creature:
        """Branch a counterfactual twin from the spine — a SIDE-control, never the spine.

        The returned creature is unbound (no state directory); save it under a
        non-spine path (e.g. creature/state/<name>/ or a scratch dir) if you need to
        persist it. The spine itself is unaffected.
        """
        return self.creature.fork(name)

    def report(self) -> str:
        """A before/after checkpoint block for the EXPERIMENTS.md entry."""
        after_age = self.age_after if self.age_after is not None else self.creature.age_steps
        after_hash = self.hash_after if self.hash_after is not None else self.creature._state_hash()
        delta = after_age - self.age_before
        return (
            f"[{self.exp_id}] spine={self.creature.name} continuous-life checkpoint\n"
            f"  before: age={self.age_before} state_hash={self.hash_before[:12]}\n"
            f"  after : age={after_age} state_hash={after_hash[:12]}\n"
            f"  delta : +{delta} steps"
        )


@contextmanager
def mirro_episode(exp_id: str, state_dir=MIRRO_DIR) -> Iterator[Episode]:
    """Load the spine, checkpoint before, yield it, then save + checkpoint after.

    Parameters
    ----------
    exp_id : str — experiment label recorded in the biography (e.g. "Exp 58").
    state_dir : path-like — the spine's snapshot directory (default the mirro spine).

    Yields
    ------
    Episode — wrapping the loaded spine. live()/teach_word() on `ep.creature`.

    On clean exit the spine is saved back to ``state_dir`` (this is the AFTER
    checkpoint). On exception the spine is NOT saved, so a failed run leaves the
    continuous life exactly as it was committed.
    """
    state_dir = Path(state_dir)
    if not state_dir.exists():
        raise SystemExit(
            f"no spine at {state_dir} — birth happens in a logged experiment "
            f"(see loop/directions/persistent-creature.md), not here."
        )

    creature = Creature.load(state_dir)
    ep = Episode(exp_id, creature, state_dir)
    creature._bio_append({
        "event": "checkpoint_before",
        "exp": exp_id,
        "age_steps": ep.age_before,
        "state_hash": ep.hash_before,
        "summary": f"{exp_id}: checkpoint before (age {ep.age_before})",
    })

    yield ep  # body raising here propagates WITHOUT the save below — spine untouched.

    creature.save(state_dir)  # AFTER checkpoint (writes arrays + manifest + 'save' event).
    ep.age_after = creature.age_steps
    ep.hash_after = creature._state_hash()
    creature._bio_append({
        "event": "checkpoint_after",
        "exp": exp_id,
        "age_steps": ep.age_after,
        "state_hash": ep.hash_after,
        "summary": f"{exp_id}: checkpoint after (age {ep.age_before} -> {ep.age_after})",
    })
    print(ep.report())
