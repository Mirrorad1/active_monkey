"""Parallel JSON entrypoint: run ``score_affect_fast`` and print the AffectScoreReport as JSON.

Drop-in for ``python -m eval.affect_score_json`` that emits the identical JSON shape but
scores the seed ensemble in parallel (``active_loop.affect_score_fast``).  Used by the
autopilot's ``score_fn`` so a full score takes ~minutes instead of ~half an hour, while
remaining bit-identical to the frozen scorer (guarded by tests/test_affect_score_fast.py).
"""
from __future__ import annotations

import dataclasses
import json

from active_loop.affect_score_fast import score_affect_fast


def main() -> None:
    report = score_affect_fast()
    print(json.dumps(dataclasses.asdict(report)))


if __name__ == "__main__":
    main()
