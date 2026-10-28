"""Guard the boundaries between recorded evidence, generated advice and reviewed memory."""
import copy
import json
from pathlib import Path

import pytest

from meta_monkey.sleep import build_cycle, write_cycle, review_cycle, load_reviewed_brief


def fixture(root):
    (root / 'experiments/outputs').mkdir(parents=True)
    (root / 'experiments/exp1_toy.py').write_text('observed toy\n')
    (root / 'experiments/outputs/exp1.txt').write_text('metric = 0\n')
    (root / 'EXPERIMENTS.md').write_text(
        '## Exp 1 — toy\n- Plain: Replay failed on the toy.\n'
        '- Verdict: NEGATIVE / NEW INSIGHT\n- Honest caveat: One toy only.\n')
    return dict(sources=[dict(experiment=1, script='experiments/exp1_toy.py',
                             output='experiments/outputs/exp1.txt')],
                items=[dict(kind='boundary', source=1, excerpt='Replay failed on the toy.')])


def approved(root, packet):
    out = root / 'meta/sleep/test'
    write_cycle(root, packet, out)
    receipt = json.loads((out / 'receipt.json').read_text())
    review_cycle(root, out, dict(content_hash=receipt['content_hash'],
                                reviewer='independent-fixture', accepted=True,
                                notes='Source fidelity and negative retention checked.'))
    return out


def test_negative_is_retained_with_verdict_caveat_and_provenance(tmp_path):
    packet = fixture(tmp_path)
    before = (tmp_path / 'EXPERIMENTS.md').read_bytes()
    brief, receipt = build_cycle(tmp_path, packet)
    assert 'NEGATIVE' in brief and 'One toy only.' in brief
    assert 'Replay failed on the toy.' in brief
    assert 'experiments/outputs/exp1.txt' in brief
    assert receipt['scope'] == 'lab' and receipt['decision'] == 'pending_review'
    assert receipt['sources'] and receipt['before_hash'] and receipt['after_hash']
    assert (tmp_path / 'EXPERIMENTS.md').read_bytes() == before


@pytest.mark.parametrize('change', ['invent', 'missing', 'escape', 'generated_fact', 'drop_negative'])
def test_unwarranted_evidence_is_rejected(tmp_path, change):
    packet = fixture(tmp_path)
    if change == 'invent':
        packet['items'][0]['excerpt'] = 'Replay succeeded.'
    elif change == 'missing':
        packet['sources'][0]['output'] = 'absent.txt'
    elif change == 'escape':
        packet['sources'][0]['output'] = '../outside.txt'
    elif change == 'generated_fact':
        packet['items'][0]['origin'] = 'generated'
    else:
        packet['items'][0]['kind'] = 'finding'
    with pytest.raises((ValueError, FileNotFoundError)):
        build_cycle(tmp_path, packet)


def test_symlink_source_escape_is_rejected(tmp_path):
    packet = fixture(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + '-outside.txt')
    outside.write_text('external')
    (tmp_path / 'linked.txt').symlink_to(outside)
    packet['sources'][0]['output'] = 'linked.txt'
    with pytest.raises(ValueError):
        build_cycle(tmp_path, packet)


@pytest.mark.parametrize('key,wrong', [('script', 'experiments/exp2_toy.py'),
                                     ('output', 'experiments/outputs/exp2.txt')])
def test_existing_but_misattributed_experiment_files_are_rejected(tmp_path, key, wrong):
    packet = fixture(tmp_path)
    (tmp_path / wrong).write_text('another experiment')
    packet['sources'][0][key] = wrong
    with pytest.raises(ValueError, match='belong'):
        build_cycle(tmp_path, packet)


def test_budget_and_source_caps_fail_without_dropping_evidence(tmp_path):
    packet = fixture(tmp_path)
    with pytest.raises(ValueError, match='budget'):
        build_cycle(tmp_path, packet, max_bytes=20)
    packet['sources'] *= 9
    with pytest.raises(ValueError):
        build_cycle(tmp_path, packet)


def test_generated_rehearsal_is_explicit_and_deterministic(tmp_path):
    packet = fixture(tmp_path)
    packet['items'].append(dict(kind='rehearsal', origin='generated', source=1,
                                text='Could a simpler baseline explain an apparent gain?'))
    a = build_cycle(tmp_path, packet)
    b = build_cycle(tmp_path, copy.deepcopy(packet))
    assert a == b
    assert 'GENERATED' in a[0]


@pytest.mark.parametrize('tamper', ['output', 'script', 'log', 'brief', 'packet', 'review', 'receipt'])
def test_review_is_bound_to_exact_evidence_and_generated_content(tmp_path, tamper):
    packet = fixture(tmp_path)
    out = approved(tmp_path, packet)
    assert 'Replay failed' in load_reviewed_brief(tmp_path, out)
    path = {'output': tmp_path / 'experiments/outputs/exp1.txt',
            'script': tmp_path / 'experiments/exp1_toy.py',
            'log': tmp_path / 'EXPERIMENTS.md',
            'brief': out / 'brief.md', 'packet': out / 'packet.json',
            'review': out / 'review.json', 'receipt': out / 'receipt.json'}[tamper]
    if tamper in ('review', 'receipt'):
        value = json.loads(path.read_text())
        value['accepted' if tamper == 'review' else 'decision'] = False
        path.write_text(json.dumps(value))
    else:
        path.write_text(path.read_text() + ' changed')
    with pytest.raises((ValueError, json.JSONDecodeError)):
        load_reviewed_brief(tmp_path, out)


def test_unreviewed_and_overwrite_and_external_output_are_rejected(tmp_path):
    packet = fixture(tmp_path)
    out = tmp_path / 'meta/sleep/test'
    write_cycle(tmp_path, packet, out)
    with pytest.raises(ValueError):
        load_reviewed_brief(tmp_path, out)
    with pytest.raises(FileExistsError):
        write_cycle(tmp_path, packet, out)
    with pytest.raises(ValueError):
        write_cycle(tmp_path, packet, tmp_path / 'experiments/outputs')


def test_conflicting_evidence_is_retained_and_not_resolved_by_summary(tmp_path):
    packet = fixture(tmp_path)
    with (tmp_path / 'EXPERIMENTS.md').open('a') as f:
        f.write('## Exp 2 — conflicting toy\n- Verdict: POSITIVE\n'
                '- Honest caveat: Different condition.\n- Plain: Replay helped.\n')
    (tmp_path / 'experiments/exp2_toy.py').write_text('toy2')
    (tmp_path / 'experiments/outputs/exp2.txt').write_text('metric = 1')
    packet['sources'].append(dict(experiment=2, script='experiments/exp2_toy.py',
                                  output='experiments/outputs/exp2.txt'))
    packet['items'].append(dict(kind='finding', source=2, excerpt='Replay helped.'))
    brief, _ = build_cycle(tmp_path, packet)
    assert 'Replay failed' in brief and 'Replay helped' in brief
    assert 'Different condition.' in brief and 'One toy only.' in brief


def test_wrapped_verdicts_and_caveats_cannot_be_silently_truncated(tmp_path):
    packet = fixture(tmp_path)
    log = tmp_path / 'EXPERIMENTS.md'
    log.write_text('## Exp 1 — toy\n- Plain: Replay failed on the toy.\n'
                   '- Verdict: NEGATIVE\n  Limited to the tested regime.\n'
                   '- Honest caveat: One toy only.\n  No evidence of ecological benefit.\n'
                   '- Next: A separate test.\n')
    brief, _ = build_cycle(tmp_path, packet)
    assert 'Limited to the tested regime.' in brief
    assert 'No evidence of ecological benefit.' in brief
    assert '- Next:' not in brief


def test_compose_only_attaches_reviewed_cycle(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('sleep_test_compose',
                        Path(__file__).parents[1] / 'loop/compose.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('PREMISE.md', 'directions/demo.md', 'personas/default.md'):
        path = tmp_path / 'loop' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    monkeypatch.setattr(module, 'LOOP_DIR', tmp_path / 'loop')
    packet = fixture(tmp_path)
    out = tmp_path / 'meta/sleep/test'
    original = module.compose('demo', 'default', None)
    write_cycle(tmp_path, packet, out)
    with pytest.raises(ValueError):
        module.compose('demo', 'default', None, str(out))
    receipt = json.loads((out / 'receipt.json').read_text())
    review_cycle(tmp_path, out, dict(content_hash=receipt['content_hash'],
                                    reviewer='other', accepted=True, notes='checked'))
    attached = module.compose('demo', 'default', None, str(out))
    assert attached.startswith(original)
    assert '=== ADVISORY SLEEP BRIEF ===' in attached
    assert 'Replay failed on the toy.' in attached
