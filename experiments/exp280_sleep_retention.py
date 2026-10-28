"""Exp 280 — C1 retention and revision after continued wake (predeclared).

Hypothesis: once-weighted joint replay retains unchanged associations and revises changed
ones after new lived experience, with advantage over online idle and sequential refitting.
Fresh confirmatory seeds9400..9415, excluded smoke9399; no outcome-based tuning/filtering.
Prediction/PASS requires ALL: (A) S0 old-all csel mean>=.5; (R) joint unchanged minus S0
unchanged mean>=-.05 AND no loss in>=12/16 seeds; (V) joint changed-NEW minus S0
changed-NEW mean>=1/3 AND positive in>=12/16; (B) joint current-all minus EACH idle
and incremental20 mean>=1/6 AND positive in>=12/16 separately. Falsifier: any
scientific conjunct fails => NEGATIVE. Instrument failure => INVALID. Failed acquisition
is a normal NEGATIVE, never an exclusion/validity gate. Report every seed/subcriterion.

Provided original DirectHeadAgent K4/U6/R5/V3, optimism2 uniform, lr4, decay.999.
100 initial real interactions c%4, gamma=(1,8,100), alpha1. Common joint20 consolidation
S0 is installed into the SAME donor (A/pA only; preserve belief/PRNG/age). Continue100
real encounters with phase2 code stream seeded seed+400000 (explicitly separate from
phase1), gamma4, alpha1, schedule disabled. Only codes3,4,5 change correct response to
(c%4+1)%4; codes0,1,2 unchanged. Only actual chosen responses yield observed feedback.

Arms share donor's identical200 unique lived encounters: idle donor, joint20, restarted
chronological20, stronger incremental20. All refits start from original INITIAL anchor,
not S0 plus old data again; use donor model as initialization. Each replay20 passes:
4000 fitting +4200 monitoring =8200 record-posteriors, same evidence/capacity/decay,
not identical FLOPs. Incremental20: one joint E/M initialization then19 sequential
responsibility-refresh sweeps; infer against current FULL model then replace cached
weighted old contribution with new. Old records cannot become new independent evidence.
Shared S0 overhead is2000 fit+2100 monitoring. No hidden correct mapping enters helper.

Read-only gamma=alpha4 evaluation on working copies, six raw policy/action distributions.
Separate unchanged csel, old-all, changed-NEW and current-all; exact current-world POS
plus120 balanced heldout actions with common uniforms, no learning. These are familiar
codes/mappings, not compositional transfer. No later online-life benefit claimed.
Constant-response0 diagnostic. No selection/omission of poorly acquired seeds.

Instrument gates:200 unique times0..199; each phase visits all codes and actual feedback
matches harness; once-weighted count mass tolerance1e-6; joint objectives nondecrease
with tolerance1e-8; all source/donor posterior/RNG/age and evaluation state unchanged;
all exported checkpoint hashes roundtrip. Outcomes never determine instrument validity.
Runtime O(passes*N*K), bounded16 seeds,2 threads, no shared RNG; source-matched excluded
pilot required. Conservative serial forecast16*pilot seconds<=1800; two-worker memory
2*pilotRSS<.6RAM. Fresh output paths required; pilot budget failure stops before batch.
This is a stationary-model inference boundary screen after a partial world change,
not evolved sleep, ecological fitness or improved lab research quality.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
# This import sets the established numerical thread limits before NumPy/JAX loads.
from experiments.exp279_sleep_replay import _train, _model, _snapshot, _evaluate, reserve_output
import numpy as np
from active_loop.sleep_replay import Encounter, refit
from active_loop.state import AgentState, AgentProvenance
SEEDS=tuple(range(9400,9416))
SMOKE_SEED=9399
ARMS=('s0','idle','joint20','chronological20','incremental20','constant')
GATES={'mass','state','objective','evaluation','phase','checkpoints'}


def assess(rows, *, confirmatory):
    if not confirmatory:return dict(verdict='SMOKE_ONLY',n=len(rows))
    try:
        if sorted(r['seed'] for r in rows)!=list(SEEDS):
            raise ValueError('requires all unique preregistered seeds')
        for r in rows:
            if not GATES.issubset(r['instruments']) or not all(v is True for v in r['instruments'].values()):
                raise ValueError('instrument check failed')
            for arm in ARMS:
                for key in ('old_all','unchanged','new_changed','current_all'):
                    v=r['arms'][arm][key]
                    if not np.isfinite(v) or not 0<=v<=1:raise ValueError('invalid metric')
        def delta(arm,key,base,basekey=None):
            return np.array([r['arms'][arm][key]-r['arms'][base][basekey or key] for r in rows])
        def comparison(gaps,threshold,*,nonloss=False):
            count=int((gaps>=-1e-12 if nonloss else gaps>1e-12).sum())
            return dict(mean_gap=float(gaps.mean()),count=count,pass_effect=bool(gaps.mean()>=threshold-1e-12),pass_count=count>=12)
        acquired=float(np.mean([r['arms']['s0']['old_all'] for r in rows]))
        checks=dict(acquisition=dict(mean_old_all=acquired,passed=acquired>=.5-1e-12),
                    retention=comparison(delta('joint20','unchanged','s0'),-.05,nonloss=True),
                    revision=comparison(delta('joint20','new_changed','s0'),1/3))
        for base in ('idle','incremental20'):
            checks['vs_'+base]=comparison(delta('joint20','current_all',base),1/6)
        passed=checks['acquisition']['passed'] and all(c['pass_effect'] and c['pass_count'] for k,c in checks.items() if k!='acquisition')
        return dict(verdict='POSITIVE' if passed else 'NEGATIVE',n=len(rows),checks=checks)
    except (KeyError,TypeError,ValueError) as exc:return dict(verdict='INVALID',n=len(rows),reason=str(exc))


def score_policy(policies,probs,*,seed):
    policies,probs=np.asarray(policies),np.asarray(probs)
    if policies.shape!=(6,5) or probs.shape!=(6,5) or not np.isfinite(policies).all() or not np.isfinite(probs).all():
        raise ValueError('six finite response distributions required')
    old=np.arange(6)%4; current=old.copy();current[3:]=(old[3:]+1)%4
    chosen=policies.argmax(axis=1)
    rng=np.random.default_rng(seed+200000);codes=rng.permutation(np.tile(np.arange(6),20));uniforms=rng.random(120)
    cdf=np.cumsum(probs[codes],axis=1);cdf[:,-1]=1.
    actions=(uniforms[:,None]>=cdf).sum(axis=1)
    return dict(old_all=float((chosen==old).mean()),unchanged=float((chosen[:3]==old[:3]).mean()),
        new_changed=float((chosen[3:]==current[3:]).mean()),current_all=float((chosen==current).mean()),
        expected_pos=float(probs[np.arange(6),current].mean()),sampled_pos=float((actions==current[codes]).mean()),
        chosen=chosen.tolist(),policy_probabilities=policies.tolist(),action_probabilities=probs.tolist())


def install_model(ag,model):
    import equinox as eqx
    import jax.numpy as jnp
    ag.agent=eqx.tree_at(lambda a:(a.A,a.pA),ag.agent,
        ([jnp.array(model.a0[None]),jnp.array(model.a1[None])],
         [jnp.array(model.p0[None]),jnp.array(model.p1[None])]))


def snapshot(ag):
    state=_snapshot(ag);state.provenance=AgentProvenance(source_experiments=[280]);return state


def evaluate(model,seed):
    raw=_evaluate(model,seed)
    return dict(**score_policy(raw['policy_probabilities'],raw['action_probabilities'],seed=seed),
                evaluation_state_unchanged=raw['evaluation_state_unchanged'])


def run_seed(seed,checkpoint_dir):
    import equinox as eqx
    import jax.numpy as jnp
    from active_loop.affect_spec import POS,NEU,NEG,ASK
    started=time.perf_counter()
    ag,anchor,initial,first=_train(seed,100)
    initial.provenance=AgentProvenance(source_experiments=[280])
    s0,initial_receipt=refit(anchor,_model(ag),first,passes=20)
    before_install=snapshot(ag)
    install_model(ag,s0)
    acquired=snapshot(ag)
    # Change the declared policy schedule, retaining this individual's lived state.
    ag._gamma_schedule=None
    ag.agent=eqx.tree_at(lambda a:a.gamma,ag.agent,jnp.array([4.]))
    rng=np.random.default_rng(seed+400000);pool=[];second=[]
    for t in range(100,200):
        if not pool:pool=rng.permutation(6).tolist()
        code=int(pool.pop(0));ag.perceive(code);response=ag.act()
        correct=(code%4+1)%4 if code>=3 else code%4
        outcome=POS if response==correct else (NEU if response==ASK else NEG)
        ag.observe_feedback(code,outcome)
        second.append(Encounter(f'{seed}:{t}',t,code,int(response),int(outcome)))
    events=first+tuple(second);wake=_model(ag);source_before=snapshot(ag)
    models={'s0':s0,'idle':wake};receipts={'s0':initial_receipt}
    for name,mode in (('joint20','joint'),('chronological20','chronological'),('incremental20','incremental')):
        models[name],receipts[name]=refit(anchor,wake,events,passes=20,mode=mode)
    arms={name:evaluate(m,seed) for name,m in models.items()}
    const=np.zeros((6,5));const[:,0]=1.;arms['constant']=score_policy(const,const,seed=seed)
    source_after=snapshot(ag)
    mass=True
    for name in ('s0','joint20','chronological20','incremental20'):
        n=100 if name=='s0' else 200
        evidence=sum(4*.999**(n-1-t) for t in range(n));m=models[name]
        mass=mass and all(abs(p.sum()-(a.sum()*.999**n+evidence))<1e-6 for p,a in ((m.p0,anchor.p0),(m.p1,anchor.p1)))
    unchanged_install=(before_install.rng_state==acquired.rng_state and before_install.metadata==acquired.metadata and
        all(np.array_equal(v,acquired.belief_state[k]) for k,v in before_install.belief_state.items()))
    phase=(len(events)==200 and len({e.event_id for e in events})==200 and [e.t for e in events]==list(range(200)) and
        set(e.code for e in first)==set(range(6)) and set(e.code for e in second)==set(range(6)))
    for e in events:
        correct=(e.code%4+1)%4 if e.t>=100 and e.code>=3 else e.code%4
        phase=phase and e.outcome==(POS if e.response==correct else NEU if e.response==ASK else NEG)
    states=dict(initial=initial,anchor=anchor.to_state(),pre_install=before_install,acquired=acquired,
                source_before=source_before,source_after=source_after)
    states.update({name:m.to_state() for name,m in models.items()})
    directory=Path(checkpoint_dir)/f'seed{seed}';directory.mkdir(parents=True,exist_ok=False)
    refs={};roundtrip=True
    for name,state in states.items():
        files=state.save(directory,name)
        restored=AgentState.load(directory,name)
        roundtrip=roundtrip and restored.content_hash()==state.content_hash()
        refs[name]=dict(hash=state.content_hash(),files={k:str(p.relative_to(ROOT)) for k,p in files.items()})
    return dict(seed=seed,arms=arms,events=[asdict(e) for e in events],receipts=receipts,checkpoint_refs=refs,
        seconds=time.perf_counter()-started,instruments=dict(mass=bool(mass),phase=bool(phase),checkpoints=bool(roundtrip),
        state=bool(unchanged_install and source_before.content_hash()==source_after.content_hash()),
        evaluation=all(v.get('evaluation_state_unchanged',True) for v in arms.values()),
        objective=all(bool(np.all(np.diff(receipts[n]['metrics']['objective'])>=-1e-8)) for n in ('s0','joint20'))))


def source_hashes():
    names=('experiments/exp280_sleep_retention.py','experiments/exp279_sleep_replay.py',
           'active_loop/sleep_replay.py','active_loop/sleep.py','active_loop/affect_agent.py','active_loop/affect_spec.py')
    return {n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}


def check_preflight(smoke,hashes):
    try:
        info=smoke['preflight'];rows=smoke['rows']
        valid=(smoke['experiment']==280 and smoke['smoke'] is True and smoke['seeds']==[SMOKE_SEED] and
            smoke['source_hashes']==hashes and smoke['result']['verdict']=='SMOKE_ONLY' and len(rows)==1 and rows[0]['seed']==SMOKE_SEED and
            GATES.issubset(rows[0]['instruments']) and all(v is True for v in rows[0]['instruments'].values()) and
            all(np.isfinite(info[k]) and info[k]>0 for k in ('seconds','rss_mb','ram_gb')) and
            info['seconds']*16<=1800 and info['rss_mb']*2/1000<.6*info['ram_gb'])
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('preflight incomplete, source-stale or exceeds runtime/memory ceiling')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--out',type=Path);parser.add_argument('--preflight',type=Path)
    args=parser.parse_args(argv);hashes=source_hashes();pilot=None
    if not args.smoke:
        if args.preflight is None:parser.error('source-matched excluded pilot required')
        pilot=json.loads(args.preflight.read_text())
        try:check_preflight(pilot,hashes)
        except ValueError as exc:parser.error(str(exc))
    output=(args.out or ROOT/'experiments/outputs'/('exp280_smoke.txt' if args.smoke else 'exp280.txt')).resolve()
    if not output.is_relative_to(ROOT/'experiments/outputs'):parser.error('outputs must be in experiments/outputs')
    reserve_output(output);cp_dir=output.parent/f'{output.stem}_checkpoints';cp_dir.mkdir()
    seeds=(SMOKE_SEED,) if args.smoke else SEEDS;started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:rows=list(pool.map(lambda s:run_seed(s,cp_dir),seeds))
    elapsed=time.perf_counter()-started
    from ecology.runtime_budget import total_ram_gb
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1e6 if sys.platform=='darwin' else 1e3)
    pilot_seconds=elapsed if args.smoke else pilot['preflight']['seconds']
    raw=dict(schema_version=1,experiment=280,smoke=args.smoke,seeds=list(seeds),workers=2,source_hashes=hashes,
        config=dict(phase1=100,phase2=100,passes=20,unchanged_codes=[0,1,2],changed_codes=[3,4,5],
                    phase2_stream='seed+400000',gamma_stage2=4.,alpha_stage2=1.,gamma_eval=4.,alpha_eval=4.),
        preflight=dict(seconds=elapsed,rss_mb=rss,ram_gb=total_ram_gb(),pilot_seconds=pilot_seconds,
                       conservative_serial_forecast=16*pilot_seconds),rows=rows,result=assess(rows,confirmatory=not args.smoke))
    output.with_suffix('.json').write_text(json.dumps(raw,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['Exp280: sleep retention and revision',f'Smoke: {args.smoke}',f'Elapsed seconds: {elapsed:.6f}; peak RSS MB: {rss:.6f}']
    for row in rows:
        lines.append(f"seed {row['seed']} instruments={row['instruments']}")
        for name in ARMS:
            scores={k:row['arms'][name][k] for k in ('old_all','unchanged','new_changed','current_all','expected_pos','sampled_pos')}
            lines.append(f'  {name}: '+json.dumps(scores,sort_keys=True))
    lines+=['Predeclared decision: '+json.dumps(raw['result'],sort_keys=True),'Raw: '+str(output.with_suffix('.json').relative_to(ROOT))]
    report='\n'.join(lines)+'\n';output.write_text(report);print(report)
    return 1 if raw['result']['verdict']=='INVALID' else 0


if __name__=='__main__':sys.exit(main())
