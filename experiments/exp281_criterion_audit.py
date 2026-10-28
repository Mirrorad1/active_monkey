"""Exp 281 — retrospective original-criterion reconciliation / CONSOLIDATION.

HYPOTHESIS (audit question, NOT a prospective scientific prediction): do all selected
original necessary criteria/subclaim assertions for Exp261/263/268 survive their own
committed numeric evidence? FALSIFIER: any selected necessary criterion fails ->
NEGATIVE; missing, malformed, duplicate, nonfinite, or altered selected input -> INVALID.
These cases and their outcomes were already exposed by the Exp251–280 archive. This
is deterministic retrospective CONSOLIDATION, not new insight or an L1 causal test of
improved research quality. No world is simulated; no persistent creature is touched.

261: six original Part B D values must be nonincreasing in cost, allowing the original
implemented .05 tolerance. Failure defeats full symmetrization; passing this necessary
check alone cannot establish the ungraded plateau/gradient conjunction. The descriptive
cost lever and later MIXED reading remain distinct.
263: at original resident aggression zero, scarce>=7/8 AND abundant<7/8 AND drift<7/8.
Later higher-resident and invasion observations cannot repair this original conjunction.
268: original eight-seed P2>=5, P3>=5, and mean-hazard C-null; F-scalar's MI antecedent is
scalar>=BANDED-.02, not scalar>=high-.02. Also report scalar vs best single-band MI.
Behavior ~= has no specified tolerance: never grade it or infer full F-scalar. A failed
MI antecedent leaves frequency dispensability unestablished, not usefulness proved.
P1/all-arm determinism, leakage, range, and complete historical eligibility are not
retested. Procedural source validity is separate from scientific claim eligibility.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
import re
from statistics import fmean
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from active_loop.sleep import canonical_hash
from experiments.exp279_sleep_replay import reserve_output
from meta_monkey.sleep_archive import _source

SOURCE_COMMIT = '824ec190fbac8dacdb87680dd838c2644cf8679d'
SCRIPTS = {
    261: 'experiments/exp261_costed_predator_attack.py',
    263: 'experiments/exp263_intraspecific_contest_emergence.py',
    268: 'experiments/exp268_acoustic_eavesdropping.py',
}
SOURCE_FILES = tuple(SCRIPTS.values()) + (
    'experiments/outputs/exp261.txt', 'experiments/outputs/exp263.txt',
    'experiments/outputs/exp268.txt', 'experiments/outputs/exp268_rows.json',
    'EXPERIMENTS.md',
)
COSTS = (0, .05, .1, .15, .25, .4)
SEEDS268 = tuple(range(200, 208))
ARMS268 = ('1_no_field', '2_field_only', '3_prey_hear', '4_pred_hear',
           '5_prey_shuffle', '6_prey_silent', '7_prey_scalar')
MI_KEYS = ('mi_low', 'mi_mid', 'mi_high', 'mi_null', 'mi_scalar', 'mi_banded')
HAZARD_ARMS = {
    'hazard_baseline': '2_field_only', 'hazard_hearing': '3_prey_hear',
    'hazard_shuffle': '5_prey_shuffle', 'hazard_silent': '6_prey_silent',
    'hazard_scalar': '7_prey_scalar',
}


def _number(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('selected values must be finite real numbers')
    return value


def check261(rows):
    """Check the original necessary cost monotonicity condition, never a plateau proxy."""
    try:
        if len(rows) != 6 or any(len(row) != 2 for row in rows):
            raise ValueError('requires six cost/D pairs')
        pairs = sorted((_number(cost), _number(gap)) for cost, gap in rows)
        if tuple(cost for cost, _ in pairs) != COSTS:
            raise ValueError('requires each original cost exactly once')
    except (TypeError, KeyError) as exc:
        raise ValueError('malformed cost/D pairs') from exc
    # Decimal on recorded numeric strings avoids floating arithmetic moving an exact
    # .05 boundary to either side; it adds no tolerance to the original .05 rule.
    rises = [float(Decimal(str(b[1])) - Decimal(str(a[1])))
             for a, b in zip(pairs, pairs[1:])]
    failed = [[pairs[i][0], pairs[i + 1][0]] for i, rise in enumerate(rises) if rise > .05]
    return dict(necessary_pass=not failed, adjacent_rises=rises,
                max_adjacent_rise=max(rises), tolerance=.05,
                failed_transitions=failed,
                full_symmetry='NOT_SUPPORTED' if failed else 'NOT_ESTABLISHED_BY_THIS_NECESSARY_CHECK',
                preserved='Descriptive cost lever / later MIXED interpretation; no new plateau threshold.')


def check263(counts):
    """Check scarcity specificity at resident zero, with eight original gradient seeds."""
    try:
        if set(counts) != {'scarce', 'abundant', 'drift'}:
            raise ValueError('requires scarce, abundant, drift counts')
        for wins, total in counts.values():
            if type(wins) is not int or type(total) is not int or total != 8 or not 0 <= wins <= total:
                raise ValueError('requires valid integer wins out of eight')
    except (TypeError, KeyError) as exc:
        raise ValueError('malformed resident-zero counts') from exc
    scarce = counts['scarce'][0] >= 7
    abundant = counts['abundant'][0] < 7
    drift = counts['drift'][0] < 7
    return dict(necessary_pass=scarce and abundant and drift,
                scarce_pos=scarce, abundance_not_pos=abundant, drift_not_pos=drift,
                resident_aggression=0, threshold_wins=7,
                original_scarcity_specific='NOT_SUPPORTED' if not (scarce and abundant and drift)
                else 'NECESSARY_CONJUNCTION_PASSES',
                preserved='Later invasion and four-seed higher-resident diagnostics are distinct evidence.')


def check268(arrays):
    """Grade original numeric acoustic predicates; leave unspecified behavior ungraded."""
    try:
        for key in MI_KEYS + tuple(HAZARD_ARMS):
            values = arrays[key]
            if len(values) != 8:
                raise ValueError(f'{key} requires exactly eight values')
            for value in values:
                _number(value)
                if value < 0 or (key.startswith('hazard_') and value > 1):
                    raise ValueError(f'{key} outside its numeric domain')
    except (TypeError, KeyError) as exc:
        raise ValueError('missing/malformed selected acoustic arrays') from exc
    p2 = [high >= .10 and high >= 10 * max(null, 1e-6)
          for high, null in zip(arrays['mi_high'], arrays['mi_null'])]
    p3 = [hear < baseline - .01 and hear < silent
          for baseline, hear, silent in zip(arrays['hazard_baseline'], arrays['hazard_hearing'],
                                            arrays['hazard_silent'])]
    means = {key: fmean(values) for key, values in arrays.items()}
    cnull_delta = abs(means['hazard_silent'] - means['hazard_baseline'])
    scalar = means['mi_scalar']
    banded = means['mi_banded']
    best = max(means[key] for key in ('mi_low', 'mi_mid', 'mi_high'))
    fscalar_mi = scalar >= banded - .02
    main_pass = sum(p2) >= 5 and sum(p3) >= 5 and cnull_delta <= .005
    return dict(necessary_pass=main_pass and fscalar_mi,
                main_numeric_pass=main_pass, means=means,
                p2_by_seed=p2, p2_count=sum(p2), p2_pass=sum(p2) >= 5,
                p3_by_seed=p3, p3_count=sum(p3), p3_pass=sum(p3) >= 5,
                cnull_delta=cnull_delta, cnull_pass=cnull_delta <= .005,
                fscalar_mi_pass=fscalar_mi, fscalar_mi_threshold=banded - .02,
                scalar_minus_banded=scalar - banded,
                best_single_band_mi=best, scalar_ge_best_single_band=scalar >= best,
                scalar_minus_best_single_band=scalar - best,
                fscalar_behavior='UNSPECIFIED_TOLERANCE_NOT_GRADED',
                frequency_dispensability='UNESTABLISHED',
                preserved='Main hearing-benefit negative persists; failing MI antecedent does not prove frequency usefulness.')


def _section(text, start, stop):
    if text.count(start) != 1:
        raise ValueError(f'missing/duplicate section {start}')
    tail = text.split(start, 1)[1]
    if tail.count(stop) != 1:
        raise ValueError(f'missing/duplicate section boundary {stop}')
    return tail.split(stop, 1)[0]


def parse261(text):
    """Read only the six original Part B numeric table rows, ignoring verdict prose."""
    table = _section(text, '(B) CO-EVOLVING', 'PREDECLARED READOUT')
    rows = []
    for line in table.splitlines():
        cells = [cell.strip() for cell in line.split('|')]
        if len(cells) != 8:
            continue
        try:
            cost = float(cells[0])
        except ValueError:
            continue
        try:
            rows.append((cost, float(cells[2])))
        except ValueError as exc:
            raise ValueError('malformed Part B D value') from exc
    check261(rows)
    return sorted(rows)


def parse263(text):
    """Read resident-zero gradient counts separately from invasion/higher-resident rows."""
    gradient = _section(text, '(A) PREY-AGGR', '(B) CONTROLS')
    controls = _section(text, '(B) CONTROLS', '(C) INVASION-FROM-RARITY')
    counts = {}
    for block, names in ((gradient, {'SCARCE': 'scarce', 'ABUNDANT': 'abundant'}),
                         (controls, {'DRIFT_NULL': 'drift'})):
        for line in block.splitlines():
            cells = line.split('|')
            name = cells[0].strip()
            if name not in names:
                continue
            key = names[name]
            if key in counts or len(cells) < 2:
                raise ValueError('duplicate/malformed resident-zero row')
            match = re.match(r'\s*(\d+)\s*/\s*(\d+)\b', cells[1])
            if not match:
                raise ValueError('malformed resident-zero win counts')
            counts[key] = tuple(map(int, match.groups()))
    check263(counts)
    return counts


def parse268(rows):
    """Extract full-precision relevant arrays from the complete 7-arm/8-seed raw grid.

    The no-field arm legitimately recorded NaN for unavailable acoustic metrics; those
    are not inputs to these predicates. All selected MI and hazard values must be finite.
    """
    try:
        if not isinstance(rows, list):
            raise ValueError('acoustic raw data must be a row list')
        by_cell = {}
        for row in rows:
            arm, seed = row['arm'], row['seed']
            if arm not in ARMS268 or type(seed) is not int or seed not in SEEDS268:
                raise ValueError('unexpected acoustic arm/seed')
            key = (arm, seed)
            if key in by_cell:
                raise ValueError('duplicate acoustic arm/seed')
            by_cell[key] = row
        if len(by_cell) != len(ARMS268) * len(SEEDS268):
            raise ValueError('missing acoustic arm/seed')
        arrays = {key: [by_cell['2_field_only', seed][key] for seed in SEEDS268]
                  for key in MI_KEYS}
        arrays.update({key: [by_cell[arm, seed]['capture_hazard'] for seed in SEEDS268]
                       for key, arm in HAZARD_ARMS.items()})
    except (TypeError, KeyError) as exc:
        raise ValueError('missing/malformed acoustic row') from exc
    check268(arrays)
    return arrays


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key {key}')
        result[key] = value
    return result


def build_audit(root=ROOT):
    """Byte-check pinned scripts, canonical raw files and selected original log entries."""
    root = Path(root).resolve()
    report = dict(experiment=281, classification='CONSOLIDATION',
                  evaluation='retrospective_original_criterion_reconciliation',
                  source_commit=SOURCE_COMMIT, instrument_valid=False,
                  sources=[], inputs={}, checks={})
    try:
        if not (root / '.git').exists():
            raise ValueError('requires a git repository to verify pinned source bytes')
        texts = {}
        for name in SOURCE_FILES:
            if name == 'EXPERIMENTS.md':
                for number in (261, 263, 268):
                    _, ref = _source(root, name, SOURCE_COMMIT, selector=number)
                    report['sources'].append(ref)
            else:
                texts[name], ref = _source(root, name, SOURCE_COMMIT)
                report['sources'].append(ref)
        inputs = {
            '261': parse261(texts['experiments/outputs/exp261.txt']),
            '263': parse263(texts['experiments/outputs/exp263.txt']),
            '268': parse268(json.loads(texts['experiments/outputs/exp268_rows.json'],
                                      object_pairs_hook=_unique_object)),
        }
        checks = {'261': check261(inputs['261']), '263': check263(inputs['263']),
                  '268': check268(inputs['268'])}
        report.update(instrument_valid=True, inputs=inputs, checks=checks,
                      verdict='POSITIVE' if all(c['necessary_pass'] for c in checks.values()) else 'NEGATIVE',
                      historical_scientific_eligibility={
                          '261_full_symmetrization': checks['261']['full_symmetry'],
                          '263_original_scarcity_specific': checks['263']['original_scarcity_specific'],
                          '268_hearing_benefit_numeric': 'PASSES_NECESSARY_CHECKS' if checks['268']['main_numeric_pass'] else 'NOT_SUPPORTED',
                          '268_frequency_dispensability': 'UNESTABLISHED',
                      },
                      limits=[
                          'Cases/outcomes known before this retrospective audit; no prospective prediction.',
                          'No new worlds, weight updates, persistent-state advance, or causal L1 evaluation.',
                          'Necessary checks cannot establish omitted historical conjuncts.',
                          'No fabricated plateau or behavior-equivalence tolerance; no all-arm determinism claim.',
                          '268 full-precision MI/hazards use committed row JSON; canonical printed summary is separately pinned.',
                          '268 no-field unavailable acoustic NaNs are excluded from selected inputs.',
                          'Later 261 MIXED/cost-lever and 263 invasion/higher-resident observations remain distinct.',
                      ])
    except (OSError, KeyError, TypeError, ValueError, OverflowError) as exc:
        report.update(verdict='INVALID', reason=str(exc))
    report['content_hash'] = canonical_hash(report)
    return report


def write_report(report, out, *, root=ROOT):
    """Reserve new .txt/.json output only below experiments/outputs; never overwrite."""
    root = Path(root).resolve()
    out = Path(out)
    if not out.is_absolute():
        out = root / out
    out = out.resolve()
    allowed = (root / 'experiments/outputs').resolve()
    if out.suffix != '.txt' or not out.is_relative_to(allowed):
        raise ValueError('output must be a .txt path under experiments/outputs')
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    reserve_output(out)
    with out.with_suffix('.json').open('x') as stream:
        stream.write(serialized)
    out.write_text('Exp 281 — retrospective original-criterion audit\n'
                   'CONSOLIDATION / cases and outcomes already exposed; no new simulation.\n'
                   'Full-precision selected numbers and source refs follow as JSON.\n\n' + serialized)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=Path('experiments/outputs/exp281.txt'))
    args = parser.parse_args(argv)
    try:
        report = build_audit(ROOT)
        write_report(report, args.out, root=ROOT)
    except (OSError, ValueError) as exc:
        print(f'criterion audit: {exc}', file=sys.stderr)
        return 1
    print(f"{report['verdict']} / CONSOLIDATION; instrument_valid={report['instrument_valid']}")
    return 1 if report['verdict'] == 'INVALID' else 0


if __name__ == '__main__':
    sys.exit(main())
