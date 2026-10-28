"""Guard confirmatory conjunctions independently of the runner's printed story."""
import copy
import pytest

from experiments.exp279_sleep_replay import assess, reserve_output


def rows(joint=.75, chronological=.25, idle=.25):
    return [dict(seed=s, instruments={'mass': True, 'state': True, 'objective': True, 'evaluation': True},
                 arms={'joint20': {'csel': joint}, 'chronological20': {'csel': chronological},
                       'idle': {'csel': idle}, 'broken20': {'csel': .25},
                       'long300': {'csel': .75}, 'constant': {'csel': 1/3}})
            for s in range(9300, 9316)]


def test_pass_requires_both_baselines_and_effect_and_seed_count():
    assert assess(rows(), confirmatory=True)['verdict'] == 'POSITIVE'
    assert assess(rows(chronological=.75), confirmatory=True)['verdict'] == 'NEGATIVE'
    assert assess(rows(idle=.75), confirmatory=True)['verdict'] == 'NEGATIVE'
    assert assess(rows(joint=.30), confirmatory=True)['verdict'] == 'NEGATIVE'
    sample = rows()
    for row in sample[:5]:
        row['arms']['joint20']['csel'] = 0
    assert assess(sample, confirmatory=True)['verdict'] == 'NEGATIVE'


def test_instrument_failure_or_partial_seed_set_cannot_be_promoted():
    sample = rows()
    sample[0]['instruments']['mass'] = False
    assert assess(sample, confirmatory=True)['verdict'] == 'INVALID'
    assert assess(rows()[:-1], confirmatory=True)['verdict'] == 'INVALID'
    assert assess(rows(), confirmatory=False)['verdict'] == 'SMOKE_ONLY'
    sample = rows()
    sample[-1]['seed'] = sample[0]['seed']
    assert assess(sample, confirmatory=True)['verdict'] == 'INVALID'


def test_missing_or_nonfinite_metric_is_invalid():
    sample = rows()
    sample[0]['arms']['joint20']['csel'] = float('nan')
    assert assess(sample, confirmatory=True)['verdict'] == 'INVALID'
    sample = rows()
    del sample[0]['arms']['idle']
    assert assess(sample, confirmatory=True)['verdict'] == 'INVALID'
    sample = rows()
    del sample[0]['instruments']['mass']
    assert assess(sample, confirmatory=True)['verdict'] == 'INVALID'


def test_existing_output_is_never_overwritten(tmp_path):
    path = tmp_path / 'exp279.txt'
    reserve_output(path)
    path.write_text('original')
    with pytest.raises(FileExistsError):
        reserve_output(path)
    assert path.read_text() == 'original'


def test_report_and_raw_json_cannot_share_a_filename(tmp_path):
    with pytest.raises(ValueError, match='txt'):
        reserve_output(tmp_path / 'run.json')
    assert not (tmp_path / 'run.json').exists()


def test_real_agent_adapter_preserves_response_conditioned_capacity_and_live_state():
    from experiments.exp279_sleep_replay import _model, _snapshot
    from active_loop.affect_agent import DirectHeadAgent
    from active_loop.affect_spec import build_direct_head_model
    from active_loop.sleep_replay import refit, Encounter
    ag = DirectHeadAgent(build_direct_head_model(1, k=4), optimism=2.)
    before = _snapshot(ag).content_hash()
    anchor = _model(ag)
    assert anchor.a0.shape == (6, 4, 5)
    assert anchor.a1.shape == (3, 4, 5)
    fitted, _ = refit(anchor, anchor, (Encounter('observed', 0, 0, 1, 0),))
    assert fitted.a0.shape == (6, 4, 5)
    assert _snapshot(ag).content_hash() == before


@pytest.mark.parametrize('damage', [None, 'empty', 'seed', 'instrument', 'hash', 'nan'])
def test_preflight_requires_real_excluded_seed_and_complete_evidence(damage):
    from experiments.exp279_sleep_replay import check_preflight
    hashes = {'runner': 'abc', 'helper': 'def'}
    smoke = dict(experiment=279, smoke=True, seeds=[9299], source_hashes=hashes.copy(),
                 result={'verdict': 'SMOKE_ONLY'},
                 rows=[dict(seed=9299, instruments=dict(mass=True, state=True,
                                                        objective=True, evaluation=True))],
                 preflight=dict(seconds=20., rss_mb=100., ram_gb=16., memory_safe=True))
    if damage == 'empty':
        smoke['rows'] = []
    elif damage == 'seed':
        smoke['seeds'] = [9300]
    elif damage == 'instrument':
        smoke['rows'][0]['instruments'] = {}
    elif damage == 'hash':
        smoke['source_hashes'] = {}
    elif damage == 'nan':
        smoke['preflight']['seconds'] = float('nan')
    if damage is None:
        check_preflight(smoke, hashes, 2)
    else:
        with pytest.raises(ValueError):
            check_preflight(smoke, hashes, 2)
