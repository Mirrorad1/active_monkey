"""Review-bound archival consolidation; never automatic active-prompt context.

Unlike a bounded sleep brief, an archive preserves every complete selected entry. Generated
synthesis/audit/rehearsal remains interpretation, not new observed scientific evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from active_loop.sleep import make_receipt, canonical_hash
from meta_monkey.sleep import _inside, _cycle_path, _field
from meta_monkey.collect_iteration import find_entries


def _window_at_commit(root,commit):
    if (root/'.git').exists():
        if not isinstance(commit,str) or not re.fullmatch('[0-9a-f]{40}',commit):
            raise ValueError('archive must pin a full source commit')
        result=subprocess.run(['git','show',f'{commit}:EXPERIMENTS.md'],cwd=root,capture_output=True,text=True)
        if result.returncode:raise ValueError('source commit cannot be resolved')
        return find_entries(result.stdout)
    if commit is not None:raise ValueError('synthetic root has no verifiable source commit')
    return find_entries((root/'EXPERIMENTS.md').read_text())



def _source(root,name,commit,*,selector=None):
    path=_inside(root,name);rel=path.relative_to(root).as_posix();data=path.read_bytes()
    committed=None
    if (root/'.git').exists():
        result=subprocess.run(['git','show',f'{commit}:{rel}'],cwd=root,capture_output=True)
        if result.returncode:raise ValueError(f'source not committed at pinned revision: {rel}')
        committed=result.stdout
    if selector is not None:
        text=find_entries(data.decode('utf-8'))[selector]
        selected=text.encode('utf-8')
        if committed is not None and selected!=find_entries(committed.decode('utf-8'))[selector].encode('utf-8'):
            raise ValueError(f'source differs from committed pinned entry: {selector}')
        data=selected
    else:
        if committed is not None and data!=committed:
            raise ValueError(f'source differs from committed pinned bytes: {rel}')
        text=data.decode('utf-8')
    ref=dict(path=rel,sha256=hashlib.sha256(data).hexdigest())
    if selector is not None:ref['experiment']=selector
    return text,ref


def build_archive(root:Path,packet:dict):
    root=root.resolve();count=packet['count'];sources=packet['sources']
    if type(count) is not int or not 1<=count<=30:raise ValueError('archival source count must be1..30')
    entries_at_commit=_window_at_commit(root,packet['source_commit'])
    selected=sorted(entries_at_commit)[-count:]
    if len(selected)!=count or [s['experiment'] for s in sources]!=selected:
        raise ValueError('sources must be the exact ordered latest window at the pinned commit')
    refs=[];records=['# Complete archival evidence\n\nRecorded entries remain authoritative; historical wording is preserved.\n']
    for s in sources:
        n=s['experiment'];entry,ref=_source(root,'EXPERIMENTS.md',packet['source_commit'],selector=n);refs.append(ref)
        if entry!=entries_at_commit[n]:raise ValueError('record differs from the pinned selection commit')
        for key in ('script','output'):
            rel=_inside(root,s[key]).relative_to(root).as_posix()
            matches=(re.fullmatch(rf'experiments/exp{n}_[^/]+\.py',rel) if key=='script' else rel==f'experiments/outputs/exp{n}.txt')
            if not matches:raise ValueError('source must belong to its experiment')
            _,ref=_source(root,rel,packet['source_commit']);refs.append(ref)
        for extra in s.get('supporting_outputs',[]):
            rel=_inside(root,extra).relative_to(root).as_posix()
            if not re.fullmatch(rf'experiments/outputs/exp{n}(?:\.json|[a-z]?_[^\n]+)',rel):
                raise ValueError('supporting output must belong to its experiment')
            _,ref=_source(root,rel,packet['source_commit']);refs.append(ref)
        for extra in s.get('supporting_scripts',[]):
            rel=_inside(root,extra).relative_to(root).as_posix()
            if not re.fullmatch(rf'experiments/exp{n}[a-z]?_[^/]+\.py',rel):
                raise ValueError('supporting script must belong to its experiment')
            _,ref=_source(root,rel,packet['source_commit']);refs.append(ref)
        if not _field(entry,'Verdict') or not _field(entry,'Honest caveat'):
            raise ValueError(f'exp{n} lacks complete recorded verdict/caveat')
        records.append(entry)
        records.append(f'\nArchive source: `{s["script"]}` -> `{s["output"]}`\n')
        for extra in s.get('supporting_outputs',[])+s.get('supporting_scripts',[]):
            records.append(f'Supporting source: `{extra}`\n')
    parts=['# Retrospective sleep synthesis\n',
           f'Window: Exp{selected[0]}–{selected[-1]} ({count} completed records).',
           'CONSOLIDATION / SOURCE-FIDELITY AUDIT. No new simulation, scientific re-grading, or demonstrated lab-quality gain.',
           'Complete verdicts, caveats and historical wording are in records.md. Generated interpretation below is review-bound, not observed data.\n']
    rehearsals=0
    for item in packet['items']:
        kind=item['kind'];ids=item['sources']
        if (item.get('origin')!='generated' or kind not in ('synthesis','audit','rehearsal') or
            not isinstance(ids,list) or not ids or any(type(n) is not int or n not in selected for n in ids) or
            not isinstance(item.get('text'),str) or not item['text'].strip()):
            raise ValueError('generated item needs a valid kind, text and archived source IDs')
        if kind=='rehearsal':rehearsals+=1
        if rehearsals>3:raise ValueError('at most3 untested rehearsals')
        label={'synthesis':'GENERATED RETROSPECTIVE SYNTHESIS','audit':'GENERATED AUDIT NOTE','rehearsal':'GENERATED REHEARSAL — UNTESTED'}[kind]
        parts += [f'## {label} — '+', '.join(f'Exp{n}' for n in ids),item['text'],'']
    records='\n'.join(records);synthesis='\n'.join(parts)+'\n'
    receipt=make_receipt('lab',packet,dict(records=records,synthesis=synthesis),refs,
        dict(max_sources=30,automatic_prompt_attachment=False,active_brief_limits_unchanged=True),
        dict(archived_experiments=count,first=selected[0],last=selected[-1],generated_rehearsals=rehearsals,
             evaluation='retrospective_source_fidelity_only',new_experiment=False),'pending_review')
    return records,synthesis,receipt


def _dump(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def write_archive(root,packet,out):
    records,synthesis,receipt=build_archive(root,packet);out=_cycle_path(root,out)
    out.mkdir(parents=True,exist_ok=False)
    _dump(out/'packet.json',packet);_dump(out/'receipt.json',receipt)
    (out/'records.md').write_text(records);(out/'synthesis.md').write_text(synthesis)


def _verify(root,out):
    out=_cycle_path(root,out);packet=json.loads((out/'packet.json').read_text())
    records,synthesis,receipt=build_archive(root,packet)
    if (records!=(out/'records.md').read_text() or synthesis!=(out/'synthesis.md').read_text() or
        receipt!=json.loads((out/'receipt.json').read_text())):
        raise ValueError('archive, packet or source changed; new review required')
    return out,synthesis,receipt


def review_archive(root,out,review):
    out,_,receipt=_verify(root,out)
    if (review.get('content_hash')!=receipt['content_hash'] or type(review.get('accepted')) is not bool or
        not review.get('reviewer') or not review.get('notes')):raise ValueError('review must bind content hash and identify reviewer/fidelity notes')
    review={**review,'review_hash':canonical_hash({k:v for k,v in review.items() if k!='review_hash'})}
    with (out/'review.json').open('x') as f:json.dump(review,f,indent=2,sort_keys=True);f.write('\n')


def load_reviewed_archive(root,out):
    out,synthesis,receipt=_verify(root,out)
    try:review=json.loads((out/'review.json').read_text())
    except FileNotFoundError as exc:raise ValueError('archive requires independent review') from exc
    if (review.get('accepted') is not True or review.get('content_hash')!=receipt['content_hash'] or
        not review.get('reviewer') or not review.get('notes') or
        review.get('review_hash')!=canonical_hash({k:v for k,v in review.items() if k!='review_hash'})):
        raise ValueError('archive review missing, rejected or stale')
    return synthesis


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0]);subs=parser.add_subparsers(dest='action',required=True)
    p=subs.add_parser('build');p.add_argument('--packet',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p=subs.add_parser('review');p.add_argument('--archive',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p=subs.add_parser('inspect');p.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args(argv);root=Path.cwd()
    try:
        if args.action=='build':write_archive(root,json.loads(args.packet.read_text()),args.out)
        elif args.action=='review':review_archive(root,args.archive,json.loads(args.review.read_text()))
        else:print(load_reviewed_archive(root,args.archive),end='')
        return 0
    except (OSError,KeyError,ValueError,TypeError) as exc:print(f'archive: {exc}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
