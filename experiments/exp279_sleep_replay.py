"""Exp 279 — sleep-inspired once-weighted replay of short dyad sessions.

PREDECLARATION (before scientific execution; approved framework 2026-10-27):
Hypothesis: joint offline re-estimation of latent assignments from the same 100 lived
encounters improves code-response discrimination beyond idle and chronological refitting.
Prediction: joint20 minus EACH of chronological20 and idle has mean correct_select gap
>= 1/6 AND strictly positive gap on >=12/16 seeds, separately for BOTH baselines.
Falsifier: any scientific conjunct fails => NEGATIVE. Instrument failure => INVALID.
This is a toy algorithmic inference screen, not MoE knowledge seeding, evolved sleep,
an ecology result, lab self-improvement evidence, or reliable short-session rescue.

Frozen design: K=4,U=6,R=5,V=3 DirectHeadAgent; original direct-head scaffold,
optimism=2 uniform, lr=4, decay=.999; wake gamma linearly 1->8 over 100 turns,
alpha=1. Local seeded block-shuffled codes; POS iff response==code%4, NEU iff ASK,
NEG otherwise. Learner/replay module never receives the evaluator's correct mapping.
Confirmatory seeds 9300..9315; smoke9299 excluded, no tuning on smoke outcomes.

All same-history arms start from the same 100-turn checkpoint and immutable pre-wake
anchor. Arms idle, chronological20, joint20, broken20, plus constant-response0 control
and a LONG300 reference (same init but 300 real encounters and its own gamma schedule;
extra data, NOT matched). Each replay arm performs 20*N=2000 latent-record posterior
fitting evaluations plus 2100 monitoring evaluations (4100 total), same shapes,
prior and evidence weight. Joint20 maximizes weighted
log sum_z[D[z]A0[c,z,r]A1[v,z,r]] plus decayed anchor pseudo-count log penalties.
Rebuild counts FROM ANCHOR each pass: p = anchor*.999**100 +
4*sum_t[.999**(99-t)*responsibility_t]. No cumulative re-adding of replay data.
Chronological20 restarts the same sequential once-weighted pass from the anchor each
time; start A affects its first record, then counts determine A. It is a deliberately
redundant budget control, NOT the strongest possible offline optimizer. Broken20
permutes logged outcomes, using no hidden labels, and is GENERATED corrupted data.

Both infer code/outcome given the actual chosen response. No unchosen outcomes or
extra world queries in sleep. Joint uses wake parameters to initialize the E-step.
This changes inference/optimization order, not capacity; matched work counts are
record-posterior evaluations, NOT a claim of identical FLOPs or wall-clock.

Read-only evaluator: gamma=alpha=4, no schedule, original structural B/C/D. For each
code, infer from D with NEU as in existing correct_select; record q_pi and exact
stochastic-action probabilities softmax(4*log_stable(q_pi)). Csel is fraction of six
codes whose argmax response equals code%4. Also report exact expected POS and sampled
POS on a separate 120-balanced-code stream with common uniform draws. No evaluation
feedback is used to learn. This tests new episodes of a KNOWN mapping, not transfer.
The primary metric is discrimination, not a stochastic POS-rate or count threshold.

Instrument checks: unique ordered event IDs/timestamps, 100 observations; both refits
have declared anchor+once-weighted evidence mass (absolute tolerance1e-6), joint
regularized objective nondecreasing (tolerance1e-8); chronology repeated-pass exact
equivalence guarded by unit tests; source/live posterior/PRNG/age hash unchanged across
all refits and evaluations. Constant-control csel is independently bounded at2/6.
Input validity never depends on a favorable discrimination score. All seed results,
traces, checkpoints, raw policies, compute counts and source hashes are retained.

L25 bound: 16 independent seeds*(100+300) lived steps, 3*(20+21)*100 replay posterior
evaluations per seed; memory O(N*K + U*K*V*R), not all-pairs. Smoke projects wall-clock
and resident memory before batch. Use 2 bounded threads: the dominant JAX backend
releases the GIL; no global NumPy RNG is shared. Fresh output path required.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from active_loop.sleep import canonical_hash
from active_loop.sleep_replay import Encounter, ReplayModel, refit
from active_loop.state import AgentState, AgentProvenance

SEEDS = tuple(range(9300, 9316))
SMOKE_SEED = 9299
ARMS = ('idle', 'chronological20', 'joint20', 'broken20', 'constant', 'long300')


def reserve_output(path):
    path = Path(path)
    if path.suffix != '.txt':
        raise ValueError('report output must end in .txt so raw JSON has a distinct path')
    if path.exists() or path.with_suffix('.json').exists() or (path.parent / f'{path.stem}_checkpoints').exists():
        raise FileExistsError('experiment output already exists; never overwrite raw results')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x'):
        pass


def assess(rows, *, confirmatory):
    if not confirmatory:
        return dict(verdict='SMOKE_ONLY')
    try:
        if sorted(row['seed'] for row in rows) != list(SEEDS):
            return dict(verdict='INVALID', reason='requires all unique preregistered seeds')
        required = {'mass', 'state', 'objective', 'evaluation'}
        if any(not required.issubset(row['instruments']) or
               not all(v is True for v in row['instruments'].values()) for row in rows):
            return dict(verdict='INVALID', reason='instrument check failed')
        for row in rows:
            for arm in ARMS:
                value = row['arms'][arm]['csel']
                if not np.isfinite(value) or not 0 <= value <= 1:
                    return dict(verdict='INVALID', reason='nonfinite/out-of-range metric')
        comparisons = {}
        for baseline in ('idle', 'chronological20'):
            gaps = np.array([row['arms']['joint20']['csel'] - row['arms'][baseline]['csel'] for row in rows])
            comparisons[baseline] = dict(mean_gap=float(gaps.mean()),
                                          positive_seeds=int((gaps > 1e-12).sum()),
                                          pass_effect=bool(gaps.mean() >= 1/6 - 1e-12),
                                          pass_count=bool((gaps > 1e-12).sum() >= 12))
        passed = all(c['pass_effect'] and c['pass_count'] for c in comparisons.values())
        return dict(verdict='POSITIVE' if passed else 'NEGATIVE', comparisons=comparisons)
    except (KeyError, TypeError, ValueError):
        return dict(verdict='INVALID', reason='missing/malformed result')


def _snapshot(ag):
    tensors = {}
    for field in ('A', 'pA', 'B', 'pB', 'C', 'D'):
        for i, value in enumerate(getattr(ag.agent, field)):
            tensors[f'{field}{i}'] = np.asarray(value)
    belief = {}
    for field in ('_prior', '_qs_perceive', '_last_qs'):
        values = getattr(ag, field)
        if values is not None:
            belief.update({f'{field}_{i}': np.asarray(v) for i, v in enumerate(values)})
    return AgentState('M4a-affect-dyad-DirectHeadAgent', 'DirectHeadAgent',
        tensors=tensors, belief_state=belief, rng_state=np.asarray(ag._key).tolist(),
        provenance=AgentProvenance(source_experiments=[279]),
        metadata=dict(lived_turns=ag._t, lv=ag.lv, lr_pA=ag.lr_pA,
                      gamma_schedule=ag._gamma_schedule,
                      last_action=None if ag._last_action is None else np.asarray(ag._last_action).tolist()))


def _model(ag):
    return ReplayModel(np.asarray(ag.agent.A[0])[0], np.asarray(ag.agent.A[1])[0],
                       np.asarray(ag.agent.pA[0])[0], np.asarray(ag.agent.pA[1])[0],
                       np.asarray(ag._D[0]).reshape(-1))


def _train(seed, turns):
    from active_loop.affect_agent import DirectHeadAgent
    from active_loop.affect_spec import build_direct_head_model, POS, NEU, NEG, ASK
    ag = DirectHeadAgent(build_direct_head_model(seed, k=4), seed=seed,
                         gamma=1., alpha=1., lr_pA=4., lv=.999, optimism=2.,
                         gamma_schedule=(1., 8., turns))
    anchor, initial = _model(ag), _snapshot(ag)
    rng, pool, events = np.random.default_rng(seed), [], []
    for t in range(turns):
        if not pool:
            pool = rng.permutation(6).tolist()
        code = int(pool.pop(0))
        ag.perceive(code)
        response = ag.act()
        outcome = POS if response == code % 4 else (NEU if response == ASK else NEG)
        ag.observe_feedback(code, outcome)
        events.append(Encounter(f'{seed}:{t}', t, code, int(response), int(outcome)))
    return ag, anchor, initial, tuple(events)


def _evaluate(model, seed):
    import jax
    import jax.numpy as jnp
    from pymdp.maths import log_stable
    from active_loop.affect_agent import DirectHeadAgent
    from active_loop.affect_spec import build_direct_head_model, NEU
    m = build_direct_head_model(seed, k=4)
    m['A'] = [jnp.array(model.a0[None]), jnp.array(model.a1[None])]
    m['pA'] = [jnp.array(model.p0[None]), jnp.array(model.p1[None])]
    ag = DirectHeadAgent(m, seed=seed, gamma=4., alpha=4.)
    before = _snapshot(ag).content_hash()
    policies, action_probs = [], []
    for code in range(6):
        q = ag.agent.infer_states([jnp.array([code]), jnp.array([NEU])], ag._D)
        q_pi, _ = ag.agent.infer_policies(q)
        policies.append(np.asarray(q_pi).reshape(-1).tolist())
        action_probs.append(np.asarray(jax.nn.softmax(4 * log_stable(q_pi), axis=-1)).reshape(-1))
    probs = np.array(action_probs)
    csel = ag.correct_select({code: code % 4 for code in range(6)})
    rng = np.random.default_rng(seed + 200000)
    codes = rng.permutation(np.tile(np.arange(6), 20))
    uniforms = rng.random(len(codes))
    cdf = np.cumsum(probs[codes], axis=1)
    cdf[:, -1] = 1.
    actions = (uniforms[:, None] >= cdf).sum(axis=1)
    return dict(csel=csel, expected_pos=float(probs[np.arange(6), np.arange(6) % 4].mean()),
                sampled_pos=float((actions == codes % 4).mean()),
                policy_probabilities=policies, action_probabilities=probs.tolist(),
                evaluation_state_unchanged=before == _snapshot(ag).content_hash())


def run_seed(seed, turns=100, passes=20, checkpoint_dir=None):
    start_time = time.perf_counter()
    ag, anchor, initial, events = _train(seed, turns)
    wake, live_before = _model(ag), _snapshot(ag)
    models, receipts = {'idle': wake}, {}
    timings = {}
    for name, mode in (('chronological20', 'chronological'), ('joint20', 'joint')):
        begin = time.perf_counter()
        models[name], receipts[name] = refit(anchor, wake, events, passes=passes, mode=mode)
        timings[name] = time.perf_counter() - begin
    rng = np.random.default_rng(seed + 100000)
    shuffled = rng.permutation([e.outcome for e in events])
    broken = tuple(Encounter('generated:' + e.event_id, e.t, e.code, e.response, int(v))
                   for e, v in zip(events, shuffled))
    models['broken20'], receipts['broken20'] = refit(anchor, wake, broken, passes=passes)
    receipts['broken20']['origin_note'] = 'GENERATED corrupted diagnostic; not actual experience'
    receipts['broken20']['content_hash'] = canonical_hash(
        {k: v for k, v in receipts['broken20'].items() if k != 'content_hash'})
    arms = {name: _evaluate(m, seed) for name, m in models.items()}
    arms['constant'] = dict(csel=2/6, expected_pos=2/6, sampled_pos=2/6)
    long_ag, _, long_initial, long_events = _train(seed, turns * 3)
    models['long300'] = _model(long_ag)
    arms['long300'] = _evaluate(models['long300'], seed)
    live_after = _snapshot(ag)
    horizon = events[-1].t + 1
    evidence = float(sum(4 * .999 ** (horizon-1-e.t) for e in events))
    mass = all(abs(m.p0.sum() - (anchor.p0.sum() * .999**horizon + evidence)) < 1e-6 and
               abs(m.p1.sum() - (anchor.p1.sum() * .999**horizon + evidence)) < 1e-6
               for name, m in models.items() if name not in ('idle', 'long300'))
    objective_ok = bool(np.all(np.diff(receipts['joint20']['metrics']['objective']) >= -1e-8))
    states = dict(initial=initial, live_before=live_before, live_after=live_after,
                  long_initial=long_initial, long_final=_snapshot(long_ag))
    states.update({name: m.to_state() for name, m in models.items()})
    refs = {}
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir) / f'seed{seed}'
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        for name, state in states.items():
            files = state.save(checkpoint_dir, name)
            refs[name] = dict(hash=state.content_hash(), files={key: str(p.relative_to(ROOT)) for key, p in files.items()})
    else:
        refs = {name: dict(hash=state.content_hash()) for name, state in states.items()}
    return dict(seed=seed, arms=arms, receipts=receipts, checkpoint_refs=refs,
        events=[asdict(e) for e in events], long_events=[asdict(e) for e in long_events],
        instruments=dict(mass=bool(mass), objective=objective_ok,
                         state=live_before.content_hash() == live_after.content_hash(),
                         evaluation=all(v.get('evaluation_state_unchanged', True) for v in arms.values())),
        replay_seconds=timings, seconds=time.perf_counter() - start_time)


def _dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def check_preflight(smoke, expected_hashes, workers):
    required = {'mass', 'state', 'objective', 'evaluation'}
    try:
        rows = smoke['rows']
        info = smoke['preflight']
        valid = (smoke['experiment'] == 279 and smoke['smoke'] is True and
                 smoke['seeds'] == [SMOKE_SEED] and len(rows) == 1 and
                 rows[0]['seed'] == SMOKE_SEED and
                 required.issubset(rows[0]['instruments']) and
                 all(v is True for v in rows[0]['instruments'].values()) and
                 smoke['result']['verdict'] == 'SMOKE_ONLY' and
                 smoke['source_hashes'] == expected_hashes and
                 all(np.isfinite(info[key]) and info[key] > 0
                     for key in ('seconds', 'rss_mb', 'ram_gb')) and
                 info['seconds'] * 16 / workers <= 1200 and info['memory_safe'] is True and
                 info['rss_mb'] * 2 / 1000 < .6 * info['ram_gb'])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        valid = False
    if not valid:
        raise ValueError('preflight incomplete, stale or over runtime/memory budget; run a fresh excluded-seed smoke')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--preflight', type=Path, default=ROOT/'experiments/outputs/exp279_smoke.json')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 2:
        parser.error('worker ceiling is 2')
    output = args.out or ROOT / 'experiments/outputs' / ('exp279_smoke.txt' if args.smoke else 'exp279.txt')
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'experiments/outputs'):
        parser.error('raw experiment outputs must live under experiments/outputs')
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (Path(__file__).resolve(), ROOT/'active_loop/sleep_replay.py',
                              ROOT/'active_loop/affect_agent.py', ROOT/'active_loop/affect_spec.py')}
    if not args.smoke:
        smoke_path = args.preflight.resolve()
        if not smoke_path.exists():
            parser.error('run excluded-seed smoke preflight first')
        smoke = json.loads(smoke_path.read_text())
        try:
            check_preflight(smoke, source_hashes, args.workers)
        except ValueError as exc:
            parser.error(str(exc))
    reserve_output(output)
    cp_dir = output.parent / f'{output.stem}_checkpoints'
    cp_dir.mkdir()
    seeds = (SMOKE_SEED,) if args.smoke else SEEDS
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda seed: run_seed(seed, checkpoint_dir=cp_dir), seeds))
    elapsed = time.perf_counter() - started
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if sys.platform == 'darwin' else 1e3)
    from ecology.runtime_budget import total_ram_gb
    summary = assess(rows, confirmatory=not args.smoke)
    raw = dict(schema_version=1, experiment=279, smoke=args.smoke, seeds=list(seeds),
        workers=args.workers, source_hashes=source_hashes,
        config=dict(wake=100, long_wake=300, passes=20, gamma_eval=4., alpha_eval=4.),
        preflight=dict(seconds=elapsed, rss_mb=rss_mb, ram_gb=total_ram_gb(),
                       projected_seconds_2_workers=elapsed * 16 / 2,
                       memory_safe=bool(rss_mb * 2 / 1000 < .6 * total_ram_gb())),
        result=summary, rows=rows)
    _dump(output.with_suffix('.json'), raw)
    lines = ['Exp 279: once-weighted sleep replay', f'Smoke: {args.smoke}',
             f'Elapsed seconds: {elapsed:.3f}; peak RSS MB: {rss_mb:.1f}']
    for row in rows:
        lines.append(f"seed {row['seed']} instruments={row['instruments']}")
        for name in ARMS:
            a = row['arms'][name]
            lines.append(f"  {name}: csel={a['csel']:.9f} expected_pos={a['expected_pos']:.9f} sampled_pos={a['sampled_pos']:.9f}")
    lines.append('Predeclared decision: ' + json.dumps(summary, sort_keys=True))
    lines.append('Raw: ' + str(output.with_suffix('.json').relative_to(ROOT)))
    report = '\n'.join(lines) + '\n'
    output.write_text(report)
    print(report)
    return 0 if summary['verdict'] != 'INVALID' else 1


if __name__ == '__main__':
    sys.exit(main())
