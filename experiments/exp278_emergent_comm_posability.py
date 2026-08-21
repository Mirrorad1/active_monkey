"""Exp 278 — moving-resource posability harness and numbered runner.

HYPOTHESIS: a current-resource identity signal is a posable instrument when it
is costed, even though a moving resource defeats policies that only reuse history.

Property-level PREDICTION: across eight declared seeds, all non-oracle baselines
remain at or below the history ceiling plus 0.03, while the costed oracle is at
least 0.70 above the strongest baseline, has net reward at least 0.80, and the
static trail positive control reaches at least 0.95.

FALSIFIER: any one of those four conjuncts fails on any declared seed.

Provided-vs-earned ledger: the oracle is provided the current identity mapping and
pays the declared signal cost.  Its ``oracle_net_reward`` is the SHARED/KIN DYAD
payoff (forager success minus scout signal cost), under the honest incentive
assumption that the scout benefits via shared fitness; this does not imply
individual selection for an unrelated scout.  Trail/history/solo/shuffled
policies earn only the observations specified by the harness.  The static control
is provided a repeated resource path as a liveness check, not evidence of
communication.

Non-emergence caveat: this benchmark tests whether a moving-resource condition is
posable and does not show emergent communication, language, or self-formed symbols.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from active_loop.benchmarks.emergent_comm_posability import (
    PosabilityConfig,
    grade_posability,
    run_batch,
)


SEEDS = tuple(range(401, 409))
SMOKE_ROUNDS = 200
POLICY_ROUNDS = len(SEEDS) * 2000 * 5
MAX_PROJECTED_SECONDS = 30.0


def _means(rows):
    fields = (
        "chance",
        "history_ceiling",
        "solo_success",
        "trail_success",
        "history_optimal_success",
        "shuffled_signal_success",
        "oracle_success",
        "oracle_net_reward",
        "static_trail_success",
    )
    return {field: statistics.fmean(getattr(row, field) for row in rows) for field in fields}


def main() -> None:
    config = PosabilityConfig()
    smoke_config = PosabilityConfig(
        n_sites=config.n_sites,
        rounds=SMOKE_ROUNDS,
        signal_cost=config.signal_cost,
    )
    smoke_start = time.perf_counter()
    run_batch((SEEDS[0],), smoke_config, parallel=False)
    smoke_seconds = time.perf_counter() - smoke_start
    projected_seconds = smoke_seconds * (POLICY_ROUNDS / (SMOKE_ROUNDS * 5))

    print("CONFIG " + json.dumps(asdict(config), sort_keys=True))
    print(
        "INCENTIVE SHARED_KIN_DYAD "
        "payoff=forager_success-scout_signal_cost "
        "scout_benefits_via_shared_fitness=true "
        "unrelated_scout_selection=false"
    )
    print(
        "RUNTIME_PREFLIGHT "
        + json.dumps(
            {
                "smoke_rounds": SMOKE_ROUNDS,
                "smoke_seconds": smoke_seconds,
                "projected_policy_rounds": POLICY_ROUNDS,
                "projected_seconds": projected_seconds,
                "max_projected_seconds": MAX_PROJECTED_SECONDS,
                "status": "PASS" if projected_seconds <= MAX_PROJECTED_SECONDS else "ABORT",
            },
            sort_keys=True,
        )
    )
    if projected_seconds > MAX_PROJECTED_SECONDS:
        raise SystemExit("runtime preflight exceeded 30 seconds")

    rows = run_batch(SEEDS, config, parallel=True)
    for row in rows:
        print("ROW " + json.dumps(row.to_dict(), sort_keys=True))

    aggregate = _means(rows)
    gate = grade_posability(rows, config)
    print("AGGREGATE " + json.dumps(aggregate, sort_keys=True))
    for name in (
        "baselines_fail",
        "oracle_advantage",
        "costed_oracle_viable",
        "static_control_live",
    ):
        print(f"GATE {name}={getattr(gate, name)}")
    print("SCRIPT_CLAIM " + gate.verdict)
    print(
        "JSON "
        + json.dumps(
            {
                "config": asdict(config),
                "runtime_preflight": {
                    "smoke_rounds": SMOKE_ROUNDS,
                    "smoke_seconds": smoke_seconds,
                    "projected_policy_rounds": POLICY_ROUNDS,
                    "projected_seconds": projected_seconds,
                },
                "rows": [row.to_dict() for row in rows],
                "aggregate": aggregate,
                "gate": gate.to_dict(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
