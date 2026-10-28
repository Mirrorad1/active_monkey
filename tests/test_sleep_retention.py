"""C1 guards: retention cannot mask failed revision, and acquisition is an outcome."""
import copy
import numpy as np
import pytest
from experiments.exp280_sleep_retention import assess, score_policy, install_model, check_preflight


def rows():
    return [dict(seed=s, instruments=dict(mass=True, state=True, objective=True,
                 evaluation=True, phase=True, checkpoints=True), arms={
        's0':dict(old_all=.75, unchanged=1., new_changed=0., current_all=.5),
        'idle':dict(old_all=.5, unchanged=.5, new_changed=.5, current_all=.5),
        'joint20':dict(old_all=.5, unchanged=1., new_changed=1., current_all=1.),
        'chronological20':dict(old_all=.5, unchanged=.5, new_changed=.5, current_all=.5),
        'incremental20':dict(old_all=.5, unchanged=.5, new_changed=.5, current_all=.5),
        'constant':dict(old_all=1/3, unchanged=1/3, new_changed=1/3, current_all=1/3)})
        for s in range(9400,9416)]


def test_each_scientific_failure_is_negative_and_no_seed_is_filtered():
    assert assess(rows(), confirmatory=True)['verdict']=='POSITIVE'
    for damage in ('acquisition','retention','revision','idle','incremental20','count'):
        sample=rows()
        for r in sample:
            if damage=='acquisition':r['arms']['s0']['old_all']=0.
            if damage=='retention':r['arms']['joint20']['unchanged']=.5
            if damage=='revision':r['arms']['joint20']['new_changed']=.2
            if damage in ('idle','incremental20'):r['arms'][damage]['current_all']=1.
        if damage=='count':
            for r in sample[:5]:r['arms']['joint20']['new_changed']=0.
        result=assess(sample,confirmatory=True)
        assert result['verdict']=='NEGATIVE',damage
        assert result['n']==16


def test_invalid_inputs_and_smoke_cannot_get_scientific_verdict():
    assert assess(rows(), confirmatory=False)['verdict']=='SMOKE_ONLY'
    assert assess(rows()[:-1], confirmatory=True)['verdict']=='INVALID'
    sample=rows();sample[0]['instruments']['phase']=False
    assert assess(sample, confirmatory=True)['verdict']=='INVALID'
    sample=rows();sample[0]['arms']['s0']['old_all']=float('nan')
    assert assess(sample, confirmatory=True)['verdict']=='INVALID'


def test_partition_scores_do_not_reward_obsolete_changed_answers():
    probs=np.eye(5)[np.arange(6)%4]
    scores=score_policy(probs, probs, seed=1)
    assert scores['old_all']==scores['unchanged']==1.
    assert scores['new_changed']==0.
    assert scores['current_all']==.5
    revised=np.array([0,1,2,0,1,2])
    scores=score_policy(np.eye(5)[revised],np.eye(5)[revised],seed=1)
    assert scores['current_all']==scores['new_changed']==scores['unchanged']==1.
    assert scores['old_all']==.5


def test_install_preserves_live_beliefs_rng_and_age():
    from experiments.exp279_sleep_replay import _model, _snapshot
    from active_loop.affect_agent import DirectHeadAgent
    from active_loop.affect_spec import build_direct_head_model
    ag=DirectHeadAgent(build_direct_head_model(1,k=4),seed=1,optimism=2.)
    ag.perceive(0); ag.act(); ag.observe_feedback(0,0)
    before=_snapshot(ag); model=_model(ag)
    model.p0[0] += .25
    model.a0 = model.p0 / model.p0.sum(axis=0)
    install_model(ag,model)
    after=_snapshot(ag)
    assert np.array_equal(np.asarray(ag.agent.pA[0])[0],model.p0.astype(np.float32))
    assert before.rng_state==after.rng_state and before.metadata==after.metadata
    assert all(np.array_equal(v,after.belief_state[k]) for k,v in before.belief_state.items())
    for k in ('B0','B1','C0','C1','D0','D1','pB0','pB1'):
        assert np.array_equal(before.tensors[k],after.tensors[k])


def test_preflight_rejects_damaged_sources_rows_and_resource_estimate():
    hashes={'runner':'abc'}
    smoke=dict(experiment=280,smoke=True,seeds=[9399],source_hashes=hashes,
        rows=[dict(seed=9399,instruments={k:True for k in ('mass','state','objective','evaluation','phase','checkpoints')})],
        result={'verdict':'SMOKE_ONLY'},preflight=dict(seconds=10.,rss_mb=100.,ram_gb=16.))
    check_preflight(smoke,hashes)
    for damage in ('seed','rows','hashes','runtime','memory'):
        s=copy.deepcopy(smoke)
        if damage=='seed':s['seeds']=[9400]
        if damage=='rows':s['rows']=[]
        if damage=='hashes':s['source_hashes']={}
        if damage=='runtime':s['preflight']['seconds']=200.
        if damage=='memory':s['preflight']['rss_mb']=16000.
        with pytest.raises(ValueError):check_preflight(s,hashes)
