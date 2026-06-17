"""Bit-identity guard: the parallel scorer must equal the FROZEN scorer field-for-field.

active_loop.affect_score_fast reuses eval.affect_score._run_session verbatim per seed and
only parallelizes the seed loop, so the AffectScoreReport must be EXACTLY equal (not merely
close) to score_affect on the same config.  This test pins that — if the parallel path ever
diverges (e.g. a refactor changes aggregation), it fails.  Small config so it runs in the
slow suite without a multi-minute cost; bit-identity does not depend on size.
"""
import dataclasses

import pytest

from eval.affect_score import score_affect
from active_loop.affect_score_fast import score_affect_fast

_SEEDS = (20, 21, 22)
_TURNS = 12


@pytest.mark.slow
def test_fast_scorer_is_bit_identical_to_frozen():
    ref = score_affect(seeds=_SEEDS, turns=_TURNS)
    fast = score_affect_fast(seeds=_SEEDS, turns=_TURNS, max_workers=3)
    assert dataclasses.asdict(fast) == dataclasses.asdict(ref)


@pytest.mark.slow
def test_fast_scorer_single_worker_matches():
    """max_workers=1 takes the in-process path; must still equal the frozen scorer."""
    ref = score_affect(seeds=_SEEDS, turns=_TURNS)
    fast = score_affect_fast(seeds=_SEEDS, turns=_TURNS, max_workers=1)
    assert dataclasses.asdict(fast) == dataclasses.asdict(ref)
