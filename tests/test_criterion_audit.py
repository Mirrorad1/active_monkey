"""Criterion guards: hand-built counterexamples precede the raw-data regression."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit():
    return importlib.import_module('experiments.exp281_criterion_audit')


def sweep():
    return [(0, .6), (.05, .4), (.1, .2), (.15, .1), (.25, -.5), (.4, -1)]


def acoustic():
    return {key: [value] * 8 for key, value in {
        'mi_low': .4, 'mi_mid': .3, 'mi_high': .2, 'mi_null': .001,
        'mi_scalar': .5, 'mi_banded': .5, 'hazard_baseline': .4,
        'hazard_hearing': .38, 'hazard_silent': .4, 'hazard_shuffle': .4,
        'hazard_scalar': .38,
    }.items()}


def test_monotonicity_is_necessary_even_when_middle_means_look_symmetric(audit):
    rows = sweep()
    rows[3] = (.15, .28)
    result = audit.check261(rows)
    assert result['necessary_pass'] is False
    assert result['max_adjacent_rise'] == pytest.approx(.08)
    assert result['failed_transitions'] == [[.1, .15]]
    assert result['full_symmetry'] == 'NOT_SUPPORTED'


def test_monotonicity_accepts_boundary_without_claiming_full_symmetry(audit):
    rows = sweep()
    rows[3] = (.15, .25)
    result = audit.check261(rows)
    assert result['necessary_pass'] is True
    assert result['full_symmetry'] == 'NOT_ESTABLISHED_BY_THIS_NECESSARY_CHECK'


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'nonfinite', 'wrong_cost'])
def test_monotonicity_refuses_invalid_sweep(audit, change):
    rows = sweep()
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows[-1] = rows[0]
    if change == 'nonfinite': rows[0] = (0, float('nan'))
    if change == 'wrong_cost': rows[-1] = (.5, -1)
    with pytest.raises(ValueError):
        audit.check261(rows)


@pytest.mark.parametrize('counts, passed', [
    ({'scarce': (7, 8), 'abundant': (6, 8), 'drift': (6, 8)}, True),
    ({'scarce': (8, 8), 'abundant': (7, 8), 'drift': (4, 8)}, False),
    ({'scarce': (6, 8), 'abundant': (0, 8), 'drift': (4, 8)}, False),
    ({'scarce': (8, 8), 'abundant': (0, 8), 'drift': (7, 8)}, False),
])
def test_resident_zero_conjunction_counts_abundance_and_drift(audit, counts, passed):
    assert audit.check263(counts)['necessary_pass'] is passed


@pytest.mark.parametrize('counts', [
    {}, {'scarce': (8, 8), 'abundant': (0, 7), 'drift': (0, 8)},
    {'scarce': (9, 8), 'abundant': (0, 8), 'drift': (0, 8)},
    {'scarce': (float('nan'), 8), 'abundant': (0, 8), 'drift': (0, 8)},
])
def test_resident_zero_rejects_incomplete_or_invalid_counts(audit, counts):
    with pytest.raises(ValueError): audit.check263(counts)


def test_acoustic_positive_necessary_checks_leave_behavior_unspecified(audit):
    result = audit.check268(acoustic())
    assert result['necessary_pass'] is True
    assert result['p2_count'] == 8
    assert result['p3_count'] == 8
    assert result['cnull_pass'] is True
    assert result['fscalar_mi_pass'] is True
    assert result['scalar_ge_best_single_band'] is True
    assert result['fscalar_behavior'] == 'UNSPECIFIED_TOLERANCE_NOT_GRADED'
    assert result['frequency_dispensability'] == 'UNESTABLISHED'


def test_fscalar_compares_banded_not_high_and_keeps_necessary_failure(audit):
    data = acoustic()
    data['mi_banded'] = [.8] * 8
    data['mi_low'] = [.6] * 8
    result = audit.check268(data)
    assert result['p2_pass'] is True
    assert result['p3_pass'] is True
    assert result['fscalar_mi_pass'] is False
    assert result['scalar_ge_best_single_band'] is False
    assert result['necessary_pass'] is False
    assert result['frequency_dispensability'] == 'UNESTABLISHED'


def test_acoustic_main_failure_and_cnull_follow_original_rules(audit):
    data = acoustic()
    data['hazard_hearing'] = [.5] * 8
    data['hazard_silent'] = [.41] * 8
    result = audit.check268(data)
    assert result['p3_count'] == 0
    assert result['p3_pass'] is False
    assert result['cnull_pass'] is False
    assert result['necessary_pass'] is False


@pytest.mark.parametrize('change', ['missing', 'short', 'nonfinite', 'out_of_range'])
def test_acoustic_refuses_invalid_arrays(audit, change):
    data = acoustic()
    if change == 'missing': del data['mi_banded']
    if change == 'short': data['mi_high'].pop()
    if change == 'nonfinite': data['hazard_hearing'][0] = float('inf')
    if change == 'out_of_range': data['hazard_hearing'][0] = -1
    with pytest.raises(ValueError): audit.check268(data)


def test_numeric_adapters_ignore_verdict_prose_and_keep_original_sections(audit):
    raw261 = (ROOT / 'experiments/outputs/exp261.txt').read_text()
    raw263 = (ROOT / 'experiments/outputs/exp263.txt').read_text()
    assert audit.parse261(raw261) == [(0, .917), (.05, .361), (.1, -.274),
                                        (.15, -.194), (.25, -1.087), (.4, -1.909)]
    assert audit.parse263(raw263) == {'scarce': (8, 8), 'abundant': (7, 8), 'drift': (4, 8)}
    assert audit.parse263(raw263.replace('POS', 'NEG')) == audit.parse263(raw263)
    rows = json.loads((ROOT / 'experiments/outputs/exp268_rows.json').read_text())
    arrays = audit.parse268(rows)
    assert len(arrays['mi_banded']) == 8
    assert arrays['mi_banded'][0] == rows[8]['mi_banded']
    result = audit.check268(arrays)
    assert result['p2_count'] == 8
    assert result['p3_count'] == 0
    assert result['cnull_pass'] is True
    assert result['fscalar_mi_pass'] is False
    assert result['scalar_ge_best_single_band'] is False


def test_adapters_refuse_missing_and_duplicate_numeric_rows(audit):
    raw261 = (ROOT / 'experiments/outputs/exp261.txt').read_text()
    row261 = next(line for line in raw261.splitlines() if '|  +0.917 |' in line)
    with pytest.raises(ValueError): audit.parse261(raw261.replace(row261, ''))
    with pytest.raises(ValueError): audit.parse261(raw261.replace(row261, row261 + '\n' + row261))
    raw263 = (ROOT / 'experiments/outputs/exp263.txt').read_text()
    row263 = next(line for line in raw263.splitlines() if 'SCARCE |    8/8' in line)
    with pytest.raises(ValueError): audit.parse263(raw263.replace(row263, ''))
    with pytest.raises(ValueError): audit.parse263(raw263.replace(row263, row263 + '\n' + row263))
    rows = json.loads((ROOT / 'experiments/outputs/exp268_rows.json').read_text())
    with pytest.raises(ValueError): audit.parse268(rows[:-1])
    with pytest.raises(ValueError): audit.parse268(rows + [rows[8]])
    rows[8]['mi_scalar'] = float('nan')
    with pytest.raises(ValueError): audit.parse268(rows)


def test_build_preserves_source_refs_full_numbers_and_negative_validity(audit):
    report = audit.build_audit(ROOT)
    assert report['verdict'] == 'NEGATIVE'
    assert report['classification'] == 'CONSOLIDATION'
    assert report['instrument_valid'] is True
    assert report['source_commit'] == '824ec190fbac8dacdb87680dd838c2644cf8679d'
    assert len(report['sources']) == 10
    assert {ref['experiment'] for ref in report['sources'] if 'experiment' in ref} == {261, 263, 268}
    assert all(len(ref['sha256']) == 64 for ref in report['sources'])
    assert report['checks']['261']['max_adjacent_rise'] == pytest.approx(.08)
    assert report['checks']['263']['abundance_not_pos'] is False
    assert report['checks']['268']['p3_count'] == 0
    assert report['inputs']['268']['mi_banded'][0] > .8
    json.dumps(report, allow_nan=False)


def test_build_refuses_altered_pinned_source(audit, tmp_path):
    (tmp_path / '.git').symlink_to(ROOT / '.git', target_is_directory=True)
    for path in audit.SOURCE_FILES:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, target)
    target = tmp_path / 'experiments/outputs/exp263.txt'
    target.write_text(target.read_text().replace('7/8      POS', '6/8      POS'))
    report = audit.build_audit(tmp_path)
    assert report['verdict'] == 'INVALID'
    assert report['instrument_valid'] is False
    assert 'pinned' in report['reason']


def test_write_report_fresh_paths_only_and_full_precision(audit, tmp_path):
    report = {'verdict': 'NEGATIVE', 'value': .12345678912345678}
    target = tmp_path / 'experiments/outputs/fresh.txt'
    audit.write_report(report, target, root=tmp_path)
    assert json.loads(target.with_suffix('.json').read_text()) == report
    assert '.12345678912345678' in target.read_text()
    with pytest.raises(FileExistsError): audit.write_report(report, target, root=tmp_path)
    assert json.loads(target.with_suffix('.json').read_text()) == report


@pytest.mark.parametrize('name', ['outside.txt', 'experiments/outputs/bad.json'])
def test_writer_refuses_escape_or_non_txt_output(audit, tmp_path, name):
    with pytest.raises(ValueError):
        audit.write_report({}, tmp_path / name, root=tmp_path)
    assert not (tmp_path / name).exists()


def test_writer_refuses_existing_json_without_creating_txt(audit, tmp_path):
    target = tmp_path / 'experiments/outputs/fresh.txt'
    target.parent.mkdir(parents=True)
    target.with_suffix('.json').write_text('preserve')
    with pytest.raises(FileExistsError): audit.write_report({}, target, root=tmp_path)
    assert not target.exists()
    assert target.with_suffix('.json').read_text() == 'preserve'


def test_acoustic_original_seed_count_thresholds_are_five_of_eight(audit):
    data = acoustic()
    data['mi_high'] = [.1] * 5 + [.099] * 3
    data['mi_null'] = [.01] * 8
    data['hazard_hearing'] = [.38] * 5 + [.5] * 3
    result = audit.check268(data)
    assert (result['p2_count'], result['p3_count']) == (5, 5)
    assert result['necessary_pass'] is True
    data['mi_high'][4] = .099
    data['hazard_hearing'][4] = .5
    result = audit.check268(data)
    assert (result['p2_count'], result['p3_count']) == (4, 4)
    assert result['p2_pass'] is False
    assert result['p3_pass'] is False


def test_acoustic_hazard_gap_is_strict_and_silent_comparison_is_strict(audit):
    data = acoustic()
    data['hazard_hearing'] = [.4 - .01] * 8
    assert audit.check268(data)['p3_count'] == 0
    data['hazard_hearing'] = [.38] * 8
    data['hazard_silent'] = [.38] * 8
    assert audit.check268(data)['p3_count'] == 0


def test_acoustic_cnull_uses_mean_delta_and_fscalar_includes_point_zero_two(audit):
    data = acoustic()
    data['hazard_silent'] = [.38, .42] * 4
    data['mi_scalar'] = [.48] * 8
    result = audit.check268(data)
    assert result['cnull_pass'] is True
    assert result['fscalar_mi_pass'] is True
    data['mi_scalar'] = [.479] * 8
    assert audit.check268(data)['fscalar_mi_pass'] is False


def test_build_reports_missing_pinned_source_as_invalid(audit, tmp_path):
    (tmp_path / '.git').symlink_to(ROOT / '.git', target_is_directory=True)
    report = audit.build_audit(tmp_path)
    assert report['verdict'] == 'INVALID'
    assert report['instrument_valid'] is False


def test_writer_rejects_symlink_escape_and_nonfinite_before_reservation(audit, tmp_path):
    outputs = tmp_path / 'experiments/outputs'
    outputs.mkdir(parents=True)
    outside = tmp_path / 'elsewhere'
    outside.mkdir()
    (outputs / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        audit.write_report({}, outputs / 'escape/report.txt', root=tmp_path)
    with pytest.raises(ValueError):
        audit.write_report({'bad': float('nan')}, outputs / 'nonfinite.txt', root=tmp_path)
    assert not (outside / 'report.txt').exists()
    assert not (outputs / 'nonfinite.txt').exists()
