import copy,json
from pathlib import Path
import pytest
from meta_monkey.sleep_archive import build_archive,write_archive,review_archive,load_reviewed_archive


def fixture(root,n=30):
    (root/'experiments/outputs').mkdir(parents=True)
    entries=[];sources=[]
    for i in range(1,n+1):
        script=f'experiments/exp{i}_toy.py';output=f'experiments/outputs/exp{i}.txt'
        (root/script).write_text('toy source\n');(root/output).write_text('observed failure\n')
        entries.append(f'## Exp {i} — toy\n- Plain: A failed toy.\n- Verdict: NEGATIVE / NEW INSIGHT\n- Honest caveat: '+('Long caveat. '*140 if i==1 else 'One toy only.')+'\n')
        sources.append(dict(experiment=i,script=script,output=output))
    (root/'EXPERIMENTS.md').write_text(''.join(entries))
    return dict(count=n,source_commit=None,sources=sources,items=[dict(kind='synthesis',origin='generated',sources=[1,n],text='Failures must remain visible.')])


def test_all30_complete_entries_survive_without_active_budget_truncation(tmp_path):
    packet=fixture(tmp_path);records,synthesis,receipt=build_archive(tmp_path,packet)
    assert records.count('## Exp ')==30
    assert 'Long caveat. '*140 in records
    assert receipt['metrics']['archived_experiments']==30
    assert receipt['budget']['automatic_prompt_attachment'] is False
    assert 'GENERATED RETROSPECTIVE SYNTHESIS' in synthesis
    assert len(receipt['sources'])==90


def test_selection_must_be_exact_latest_window_and_never_silently_omit(tmp_path):
    p=fixture(tmp_path)
    for change in ('omit','duplicate','scope','source'):
        q=copy.deepcopy(p)
        if change=='omit':q['sources']=q['sources'][:-1]
        if change=='duplicate':q['sources'][-1]=q['sources'][0]
        if change=='scope':q['items'][0]['sources']=[31]
        if change=='source':q['sources'][0]['output']='experiments/outputs/exp2.txt'
        with pytest.raises(ValueError):build_archive(tmp_path,q)


def test_generation_must_be_labeled_and_rehearsal_capped(tmp_path):
    p=fixture(tmp_path)
    p['items'][0]['origin']='observed'
    with pytest.raises(ValueError):build_archive(tmp_path,p)
    p=copy.deepcopy(p);p['items']=[]
    for i in range(4):p['items'].append(dict(kind='rehearsal',origin='generated',sources=[1],text='Untested candidate'))
    with pytest.raises(ValueError):build_archive(tmp_path,p)


@pytest.mark.parametrize('tamper',['raw','support','records','synthesis','packet','receipt','review'])
def test_review_binds_every_source_and_report_component(tmp_path,tamper):
    p=fixture(tmp_path)
    extra=tmp_path/'experiments/outputs/exp1_detail.txt';extra.write_text('detail')
    p['sources'][0]['supporting_outputs']=['experiments/outputs/exp1_detail.txt']
    out=tmp_path/'meta/sleep/test-archive';write_archive(tmp_path,p,out)
    r=json.loads((out/'receipt.json').read_text())
    with pytest.raises(ValueError):load_reviewed_archive(tmp_path,out)
    review_archive(tmp_path,out,dict(content_hash=r['content_hash'],reviewer='independent-fixture',accepted=True,notes='all sources checked'))
    assert 'Failures must remain visible.' in load_reviewed_archive(tmp_path,out)
    path={'raw':tmp_path/'experiments/outputs/exp1.txt','support':extra,'records':out/'records.md','synthesis':out/'synthesis.md','packet':out/'packet.json','receipt':out/'receipt.json','review':out/'review.json'}[tamper]
    if tamper in ('review','receipt'):
        value=json.loads(path.read_text());value['accepted' if tamper=='review' else 'decision']=False;path.write_text(json.dumps(value))
    else:path.write_text(path.read_text()+' changed')
    with pytest.raises((ValueError,json.JSONDecodeError)):load_reviewed_archive(tmp_path,out)


def test_archive_and_review_cannot_be_overwritten(tmp_path):
    p=fixture(tmp_path);out=tmp_path/'meta/sleep/test-archive';write_archive(tmp_path,p,out)
    with pytest.raises(FileExistsError):write_archive(tmp_path,p,out)


def test_latest30_means_last30_of_history_not_first30(tmp_path):
    p=fixture(tmp_path,40);p['count']=30;p['sources']=p['sources'][-30:]
    p['items'][0]['sources']=[11,40]
    records,_,r=build_archive(tmp_path,p)
    assert r['metrics']['first']==11 and r['metrics']['last']==40
    assert '## Exp 10 —' not in records and '## Exp 11 —' in records
    p['sources']=p['sources'][:29]
    with pytest.raises(ValueError):build_archive(tmp_path,p)


def test_real_git_source_commit_and_unchanged_evidence_are_required(tmp_path):
    import subprocess
    p=fixture(tmp_path,2)
    def git(*args):return subprocess.check_output(['git',*args],cwd=tmp_path,stderr=subprocess.DEVNULL,text=True).strip()
    git('init');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
    p['source_commit']=git('rev-parse','HEAD')
    build_archive(tmp_path,p)
    p['source_commit']=None
    with pytest.raises(ValueError):build_archive(tmp_path,p)
    p['source_commit']=git('rev-parse','HEAD')
    (tmp_path/'experiments/outputs/exp1.txt').write_text('altered uncommitted evidence')
    with pytest.raises(ValueError,match='committed'):build_archive(tmp_path,p)


def test_committed_post_pin_source_rewrite_is_rejected(tmp_path):
    import subprocess
    p=fixture(tmp_path,2)
    def git(*args):return subprocess.check_output(['git',*args],cwd=tmp_path,stderr=subprocess.DEVNULL,text=True).strip()
    git('init');git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','fixture')
    p['source_commit']=git('rev-parse','HEAD')
    (tmp_path/'experiments/outputs/exp1.txt').write_text('rewritten evidence')
    git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-qm','rewrite')
    with pytest.raises(ValueError):build_archive(tmp_path,p)


@pytest.mark.parametrize('damage',['bytes','reviewer','notes'])
def test_byte_exact_sources_and_complete_review_metadata_are_bound(tmp_path,damage):
    p=fixture(tmp_path);out=tmp_path/'meta/sleep/bound';write_archive(tmp_path,p,out)
    r=json.loads((out/'receipt.json').read_text())
    review_archive(tmp_path,out,dict(content_hash=r['content_hash'],reviewer='independent',accepted=True,notes='checked'))
    if damage=='bytes':
        raw=tmp_path/'experiments/outputs/exp1.txt';raw.write_bytes(raw.read_bytes().replace(b'\n',b'\r\n'))
    else:
        path=out/'review.json';review=json.loads(path.read_text());review[damage]='changed';path.write_text(json.dumps(review))
    with pytest.raises(ValueError):load_reviewed_archive(tmp_path,out)
