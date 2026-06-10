"""Continuity guard for the persistent creature spine (mirro).

Fast, stdlib-only (no numpy/jax) — parses the committed snapshot's manifest and
biography to enforce the RECIPE's load-bearing invariant: the spine is ONE
continuous life that is never reset (loop/PROTOCOL.md, Exp 58+).

What it guards:
- `age_steps` in the biography never goes backwards (no reset / no rewind).
- The committed manifest `age_steps` matches the head (max) of the biography — the
  snapshot on disk is the latest point of the life, not a stale or rewound one.
- Every checkpoint_before is followed (eventually) by a non-decreasing age — the
  before/after checkpoint discipline of `mirro_episode` is intact.

If a future experiment re-births or rewinds the spine, this fails before the commit
lands.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SPINE = ROOT / "creature" / "state" / "mirro"

# The spine's birth experiment (Exp 45) logged a re-birth control in the same worktree,
# so two `save age=0` events sit at the very start before any accumulated life. Continuity
# is enforced strictly only once the life passes this birth phase; any reset AFTER that
# (e.g. a future episode wiping an aged spine back toward 0) is still caught.
BIRTH_PHASE_STEPS = 1000


def _manifest():
    return json.loads((SPINE / "manifest.json").read_text(encoding="utf-8"))


def _biography():
    """Return the list of parsed biography events (each a dict)."""
    path = SPINE / "BIOGRAPHY.jsonl"
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _ages():
    return [e["age_steps"] for e in _biography() if "age_steps" in e]


def test_spine_snapshot_present():
    """The continuous spine has a committed snapshot (manifest + biography)."""
    assert (SPINE / "manifest.json").exists(), "spine manifest missing"
    assert (SPINE / "BIOGRAPHY.jsonl").exists(), "spine biography missing"
    assert (SPINE / "arrays.npz").exists(), "spine arrays (weights) missing"


def test_age_never_goes_backwards():
    """Past the birth phase, biography age_steps is monotonic — the life is never reset.

    Anchors on the first event where the accumulated life exceeds BIRTH_PHASE_STEPS, then
    requires age to be non-decreasing forever after. This tolerates the Exp 45 birth-control
    artifact while still catching any genuine reset of an established life.
    """
    ages = _ages()
    assert ages, "no age_steps events in biography"

    anchor = next((i for i, a in enumerate(ages) if a > BIRTH_PHASE_STEPS), None)
    assert anchor is not None, (
        f"spine never accumulated past the birth phase ({BIRTH_PHASE_STEPS} steps)"
    )

    backsteps = [
        (i, ages[i - 1], ages[i])
        for i in range(anchor + 1, len(ages))
        if ages[i] < ages[i - 1]
    ]
    assert not backsteps, (
        "established spine age went backwards (a reset/rewind) at indices: "
        + repr(backsteps)
    )


def test_manifest_matches_biography_head():
    """The committed manifest age == the latest age in the biography (no stale snapshot)."""
    manifest_age = _manifest()["age_steps"]
    ages = _ages()
    assert manifest_age == max(ages), (
        f"manifest age_steps={manifest_age} but biography head={max(ages)} — "
        f"the committed snapshot is not the latest point of the life"
    )


def test_checkpoints_are_well_ordered():
    """Any checkpoint_before/after pairs for one exp keep age non-decreasing.

    Exp 58+ episodes go through mirro_episode, which records checkpoint_before then
    checkpoint_after. Guard that an 'after' for an experiment is never younger than
    its 'before' (i.e. the episode advanced, or at minimum held, the continuous life).
    """
    befores = {}
    for e in _biography():
        if e.get("event") == "checkpoint_before" and "exp" in e:
            befores[e["exp"]] = e["age_steps"]
        elif e.get("event") == "checkpoint_after" and "exp" in e:
            exp = e["exp"]
            if exp in befores:
                assert e["age_steps"] >= befores[exp], (
                    f"{exp}: checkpoint_after age {e['age_steps']} < "
                    f"checkpoint_before age {befores[exp]}"
                )
