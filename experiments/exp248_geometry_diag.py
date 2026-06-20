"""experiments/exp248_geometry_diag.py — Exp 248 Rung 1 diagnostic companion.

Settles WHY the static expressibility probe (exp248_geometry_probe.py) shows a
sign-unstable escape gradient: is the prey population VIABLE (dense two-trophic
coexistence) or does it COLLAPSE to a sparse near-extinction regime where capture
is a geometry-dominated random-encounter process?

Prints, for a representative cell, the prey/predator headcount trajectory and the
prey death-cause breakdown (predation vs starvation vs crowding vs senescence).
RAW NUMBERS — NO VERDICT.
"""
import sys
import os
import dataclasses as D

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from ecology.engine import Ecology


def _build(s_prey, s_pred):
    # Reuse the probe's exact config builder.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp248_probe", os.path.join(_repo_root, "experiments", "exp248_geometry_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_config(s_prey, s_pred)


def run(s_prey, s_pred, seed, marks=(0, 5, 10, 25, 50, 100, 200, 299)):
    cfg = _build(s_prey, s_pred)
    eco = Ecology(cfg, seed=seed)
    traj = {}
    while True:
        if not eco.has_alive() or eco.exploded or eco.t >= cfg.horizon:
            break
        snap = eco.alive_snapshot()
        if eco.t in marks:
            n_prey = sum(1 for c in snap if c.genotype.role == "prey")
            n_pred = sum(1 for c in snap if c.genotype.role == "predator")
            traj[eco.t] = (n_prey, n_pred)
        eco.step()
    # final
    snap = eco.alive_snapshot()
    n_prey = sum(1 for c in snap if c.genotype.role == "prey")
    n_pred = sum(1 for c in snap if c.genotype.role == "predator")
    traj[eco.t] = (n_prey, n_pred)
    # prey death-cause tally
    causes = {}
    for ev in eco.events:
        if ev["event_type"] == "death":
            cause = ev.get("details", {}).get("cause", "?")
            causes[cause] = causes.get(cause, 0) + 1
    return traj, causes, eco.t


def main():
    print("Exp 248 Rung 1 DIAGNOSTIC — prey/pred trajectory + death causes (raw)")
    print(f"founders: 30 prey + 6 predators (probe config)\n")
    for s_pred in (1.2, 1.4, 1.6):
        for s_prey in (1.0,):  # resident speed
            traj, causes, t_end = run(s_prey, s_pred, seed=0)
            print(f"cell s_prey={s_prey} s_pred={s_pred}  (ended t={t_end})")
            print("   t:  " + "  ".join(f"{t}:{p}/{q}" for t, (p, q) in sorted(traj.items())))
            print(f"   death causes (all roles): {causes}")
            print()


if __name__ == "__main__":
    main()
