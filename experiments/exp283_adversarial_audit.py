"""Additive, post-hoc Exp283 reproduction audit; never a new confirmation.

Retain individual births, match every original ledger row, strengthen the observer
guard beyond the incomplete event hash, and align the descriptive endpoint window.
The original executed sources and raw files are read-only inputs. New evidence
uses exclusive output creation. No scientific gate or original verdict changes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from math import fsum
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from experiments import exp283_boundary_transmission as original

BIRTH_COLUMNS = ["child_id", "parent_id", "birth_t", "parent_h", "child_h"]
OUT = ROOT / "experiments/outputs"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(eco):
    """Every organism's full genotype, including dead organisms, and engine RNG.

    This augments the existing event hash; it is not a full simulator-state proof.
    """
    traits = [[c.creature_id, asdict(c.genotype)] for c in eco._creatures]
    return {
        "events": eco.events_hash(),
        "genotypes": hashlib.sha256(json.dumps(traits, sort_keys=True).encode()).hexdigest(),
        "rng": hashlib.sha256(json.dumps(eco.rng.bit_generator.state, sort_keys=True).encode()).hexdigest(),
        "creatures": len(traits),
    }


def assert_equivalent(watched, bare):
    if watched != bare:
        raise ValueError("Observer changed events, genotypes, creature count, or RNG")


def capture(arm, seed, horizon, observe=True):
    """Use the untouched runner while independently recording append-only births."""
    base = original.Ecology
    instances = []

    class RecordingEcology(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit_births = []
            instances.append(self)

        def step(self):
            parents = {c.creature_id: c.genotype.thermosense_intensity
                       for c in self.alive_snapshot()}
            start = len(self._creatures)
            birth_t = self.t
            super().step()
            for c in self._creatures[start:]:
                if c.parent_id not in parents or c.phenotype.birth_t != birth_t:
                    raise ValueError("Birth lacks a before-census parent or correct timestamp")
                self.audit_births.append([c.creature_id, c.parent_id, birth_t,
                                          parents[c.parent_id], c.genotype.thermosense_intensity])

    original.Ecology = RecordingEcology
    try:
        result = original.run_one(arm, seed, horizon=horizon, observe=observe)
    finally:
        original.Ecology = base
    return result, instances[0]


def window_statistics(births, rows, horizon=3000, width=500):
    lo = max(0, horizon - width)
    children = [b[4] for b in births if lo <= b[2] < horizon]
    # Row t is the after-census for engine births at t-1.
    living = [r for r in rows if lo < r["t"] <= horizon]
    total_births = sum(r["births"] for r in living)
    if len(children) != total_births or not children or not living:
        raise ValueError("Birth window/count mismatch or empty diagnostic")
    return {
        "newborn_count": len(children), "newborn_mean": fsum(children) / len(children),
        "time_mean_living": fsum(r["mean_after"] for r in living) / len(living),
        "birth_weighted_living": fsum(r["births"] * r["mean_after"] for r in living) / total_births,
    }


def observer_checks():
    checks = []
    for arm in original.ARMS:
        watched, eco = capture(arm, original.PILOT_SEED, 120)
        bare = original.Ecology(original.make_config(arm, 120), seed=original.PILOT_SEED)
        while bare.t < 120 and bare.has_alive() and not bare.exploded:
            bare.step()
        wf, bf = fingerprint(eco), fingerprint(bare)
        assert_equivalent(wf, bf)
        if not watched["summary"]["births"]:
            raise ValueError("Observer pilot did not exercise births")
        checks.append({"arm": arm, "seed": original.PILOT_SEED, "steps": 120,
                       "watched": wf, "bare": bf, "equal": True})
        if arm == "NEUTRAL":
            c = eco._creatures[0]
            c.genotype = replace(c.genotype, thermosense_intensity=.123456)
            corrupted = fingerprint(eco)
            if corrupted["events"] != wf["events"]:
                raise ValueError("Negative control did not exercise the event-hash blind spot")
            try:
                assert_equivalent(corrupted, bf)
            except ValueError:
                checks[-1]["focal_corruption_rejected_despite_equal_event_hash"] = True
            else:
                raise ValueError("Focal corruption escaped strengthened guard")
    return checks


def reproduce(job):
    arm, seed = job
    path = OUT / f"exp283_trajectories/{arm.lower()}_seed{seed}.json.gz"
    with gzip.open(path, "rt") as stream:
        old = json.load(stream)
    result, eco = capture(arm, seed, original.HORIZON)
    rows = result["trajectory"]
    if [[r[c] for c in old["columns"]] for r in rows] != old["trajectory"]:
        raise ValueError(f"Original ledger mismatch: {arm}/{seed}")
    for field, value in old["summary"].items():
        if field != "runtime_s" and result["summary"][field] != value:
            raise ValueError(f"Original summary mismatch: {arm}/{seed}/{field}")
    births = eco.audit_births
    if len({b[0] for b in births}) != len(births):
        raise ValueError("Duplicate birth identity")
    counts = np.bincount([b[2] for b in births], minlength=original.HORIZON)
    if counts.tolist() != [r["births"] for r in rows]:
        raise ValueError("Independent per-step birth counts differ")
    stats = window_statistics(births, rows)
    if stats["newborn_mean"] != old["summary"]["final_newborn_mean_last500"]:
        raise ValueError("Independently reconstructed newborn mean differs")
    return {"arm": arm, "seed": seed, "columns": BIRTH_COLUMNS, "births": births,
            "original_trajectory_sha256": sha(path), "all_original_rows_exact": True,
            "all_original_summary_fields_except_runtime_exact": True,
            "rows_checked": len(rows), "window": stats,
            "terminal_living": old["summary"]["final_living_mean"],
            "cumulative_selection": old["summary"]["cumulative_selection"],
            "events_hash": eco.events_hash()}


def reserve_outputs(prefix):
    paths = [prefix.with_suffix(".json"), prefix.with_suffix(".txt"),
             prefix.parent / (prefix.name + "_births")]
    for path in paths:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite {path}; choose a fresh --output-prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive directory creation reserves this prefix against concurrent runs.
    paths[2].mkdir()
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, default=OUT / "exp283_adversarial_audit")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    raw_path = OUT / "exp283.json"
    raw = json.loads(raw_path.read_text())
    for path, expected in raw["source_hashes"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Original executed source has changed: {path}")
    json_path, text_path, birth_dir = reserve_outputs(args.output_prefix)
    start = time.monotonic()
    report = {"scope": "Post-hoc additive same-seed reproduction; not new confirmation",
              "original_raw_sha256": sha(raw_path), "original_source_hashes": raw["source_hashes"],
              "audit_source_sha256": sha(__file__),
              "audit_tests_sha256": sha(ROOT / "tests/test_exp283_adversarial_audit.py"),
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
              "observer_checks": observer_checks(), "runs": [], "arms": {}}
    jobs = [(arm, seed) for arm in original.ARMS for seed in original.SEEDS]
    with text_path.open("x") as log:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for payload in pool.map(reproduce, jobs):
                path = birth_dir / f"{payload['arm'].lower()}_seed{payload['seed']}.json.gz"
                with gzip.open(path, "xt", encoding="utf-8") as stream:
                    json.dump(payload, stream, separators=(",", ":"), allow_nan=False)
                summary = {k: v for k, v in payload.items() if k not in ("columns", "births")}
                summary.update(birth_count=len(payload["births"]), birth_file=path.name, birth_sha256=sha(path))
                report["runs"].append(summary)
                line = f"MATCH {payload['arm']} {payload['seed']} rows={payload['rows_checked']} births={summary['birth_count']}"
                print(line, flush=True)
                log.write(line + "\n")
        for arm in original.ARMS:
            runs = [r for r in report["runs"] if r["arm"] == arm]
            report["arms"][arm] = {metric: {
                "mean": fsum(r["window"][metric] for r in runs) / len(runs),
                "above_002": sum(r["window"][metric] >= .02 for r in runs)}
                for metric in ("newborn_mean", "time_mean_living", "birth_weighted_living")}
        report.update(rows_checked=sum(r["rows_checked"] for r in report["runs"]),
                      births_retained=sum(r["birth_count"] for r in report["runs"]),
                      runtime_s=time.monotonic() - start, status="PASS")
        log.write(json.dumps({k: v for k, v in report.items() if k != "runs"}, indent=2) + "\n")
    with json_path.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "rows": report["rows_checked"],
                      "births": report["births_retained"], "arms": report["arms"]}, indent=2))


if __name__ == "__main__":
    main()
