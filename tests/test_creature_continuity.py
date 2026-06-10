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

# The sanctioned escape hatch (anti-lock-in): an INTENTIONAL restart is allowed, but only
# when logged as an explicit biography event so it is distinguishable from silent
# corruption. Recovery from a bad epoch is normally `git checkout` of a prior committed
# snapshot (each checkpoint is a restore point); a true in-place restart writes one of these.
RESET_EVENTS = {"rebirth", "reset", "restart"}


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
    """The established spine's age never resets SILENTLY — only via a logged restart.

    Walks the biography in order. An age decrease is a violation UNLESS it is excused by
    one of two sanctioned cases:
      - the birth phase: both the from- and to-age sit at/below BIRTH_PHASE_STEPS (the
        Exp 45 re-birth-control artifact);
      - an explicit restart: a RESET_EVENTS biography event immediately precedes the drop.
    Any other backstep — e.g. an aged spine silently wiped toward 0 — fails. This protects
    the trunk from corruption without locking out an intentional, logged restart.
    """
    violations = []
    last_age = None
    reset_pending = False
    for e in _biography():
        if e.get("event") in RESET_EVENTS:
            reset_pending = True
            continue
        if "age_steps" not in e:
            continue
        a = e["age_steps"]
        if last_age is not None and a < last_age:
            in_birth_phase = last_age <= BIRTH_PHASE_STEPS and a <= BIRTH_PHASE_STEPS
            if not (reset_pending or in_birth_phase):
                violations.append((last_age, a))
        last_age = a
        reset_pending = False  # consumed by the next age observation

    assert not violations, (
        "established spine age went backwards SILENTLY (a reset/rewind without a logged "
        f"{sorted(RESET_EVENTS)} event): " + repr(violations)
    )


def test_manifest_matches_biography_head():
    """The committed manifest age == the LATEST age in the biography (no stale snapshot).

    Uses the last age event rather than the maximum so a sanctioned, logged restart (which
    lowers the current age) is not mistaken for a stale snapshot.
    """
    manifest_age = _manifest()["age_steps"]
    ages = _ages()
    assert manifest_age == ages[-1], (
        f"manifest age_steps={manifest_age} but biography head={ages[-1]} — "
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
