"""Compile evidence excerpts and generated proposals into a reviewed advisory sleep brief.

No model calls, experiment selection, scientific grading or source mutation. Exact excerpts
provide provenance, not a substitute for independent interpretation of raw evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from active_loop.sleep import canonical_hash, make_receipt
# Reuse the collector's file-loader fallback: loop/ is not an installed package when
# compose.py is executed as a script and the repository root is absent from sys.path.
from meta_monkey.collect_iteration import find_entries


def _inside(root: Path, path) -> Path:
    root = root.resolve()
    candidate = Path(path)
    candidate = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f'source/output escapes repository: {path}')
    return candidate


def _read(root, name, *, selector=None):
    path = _inside(root, name)
    text = path.read_text()
    if selector is not None:
        try:
            text = find_entries(text)[selector]
        except KeyError as exc:
            raise ValueError(f'missing experiment {selector}') from exc
    # Real repositories require committed, unchanged source evidence. Synthetic test roots
    # without git use ordinary files; no claim of historical verification is made for them.
    if (root / '.git').exists():
        rel = path.relative_to(root.resolve()).as_posix()
        result = subprocess.run(['git', 'show', f'HEAD:{rel}'], cwd=root,
                                capture_output=True, text=True)
        committed = result.stdout
        if selector is not None:
            committed = find_entries(committed).get(selector, '')
        if result.returncode or committed != text:
            raise ValueError(f'source is not committed and unchanged: {name}')
    ref = dict(path=path.relative_to(root.resolve()).as_posix(),
               sha256=hashlib.sha256(text.encode()).hexdigest())
    if selector is not None:
        ref['experiment'] = selector
    return text, ref


def _field(entry, label):
    # Markdown fields can wrap across physical lines. Stop at the next top-level field
    # or heading, never silently discard qualifications on continuation lines.
    match = re.search(rf'^- {re.escape(label)}[^:\n]*:[ \t]*(.*(?:\n(?!- [^\n]*:|## )[^\n]*)*)',
                      entry, re.M)
    return match[1].strip() if match else None


def build_cycle(root: Path, packet: dict, max_bytes: int = 1500) -> tuple[str, dict]:
    root = root.resolve()
    sources = packet['sources']
    if not 0 < len(sources) <= 8 or not 0 < max_bytes <= 1500:
        raise ValueError('source cap is eight; byte budget must be 1..1500')
    entries, refs = {}, []
    lines = ['Advisory sleep brief; sources retain authority. No new result.']
    for source in sources:
        n = source['experiment']
        if type(n) is not int or n in entries:
            raise ValueError('experiment IDs must be distinct integers')
        entry, ref = _read(root, 'EXPERIMENTS.md', selector=n)
        refs.append(ref)
        for key in ('script', 'output'):
            actual = _inside(root, source[key]).relative_to(root).as_posix()
            belongs = (actual == f'experiments/outputs/exp{n}.txt' if key == 'output' else
                       re.fullmatch(rf'experiments/exp{n}_[^/]+\.py', actual) is not None)
            if not belongs:
                raise ValueError(f'{key} must belong to exp{n} under the current experiment protocol')
            _, ref = _read(root, source[key])
            refs.append(ref)
        verdict = _field(entry, 'Verdict')
        caveat = _field(entry, 'Honest caveat')
        if not verdict or not caveat:
            raise ValueError(f'exp{n} lacks a recorded verdict or honest caveat')
        entries[n] = entry
        # Include the complete recorded verdict/caveat; fail rather than truncate evidence.
        lines += [f'Exp {n}: {verdict}', f'Caveat: {caveat}',
                  f'Raw: {source["output"]}']
        if ('NEGATIVE' in verdict or 'MIXED' in verdict) and not any(
            item.get('source') == n and item.get('kind') == 'boundary'
            for item in packet['items']
        ):
            raise ValueError(f'exp{n} requires a negative/mixed boundary')
    generated = 0
    for item in packet['items']:
        n, kind = item['source'], item['kind']
        if n not in entries:
            raise ValueError('item must cite a retrieved experiment')
        if kind in ('finding', 'boundary'):
            excerpt = item['excerpt']
            if item.get('origin', 'recorded') != 'recorded' or not excerpt.strip() or excerpt not in entries[n]:
                raise ValueError('recorded evidence must be an exact, nonempty source excerpt')
            lines.append(f'{kind.upper()} [exp{n}]: {excerpt}')
        elif kind in ('hypothesis', 'rehearsal'):
            if item.get('origin') != 'generated' or not item.get('text', '').strip():
                raise ValueError('proposals require explicit generated provenance')
            generated += 1
            if generated > 3:
                raise ValueError('at most three generated proposals per cycle')
            lines.append(f'GENERATED {kind.upper()} [exp{n}]: {item["text"]}')
        else:
            raise ValueError(f'unknown item kind {kind}')
    brief = '\n'.join(lines) + '\n'
    size = len(brief.encode())
    if size > max_bytes:
        raise ValueError(f'brief exceeds byte budget ({size}>{max_bytes}); narrow retrieval explicitly')
    receipt = make_receipt('lab', packet, brief, refs,
                           dict(max_sources=8, max_bytes=max_bytes, used_bytes=size),
                           dict(experiments=len(entries), generated=generated,
                                rejected=[], evaluation='source_fidelity_only'), 'pending_review')
    return brief, receipt


def _cycle_path(root, output):
    out = _inside(root, output)
    base = (root / 'meta/sleep').resolve()
    if not out.is_relative_to(base) or out == base:
        raise ValueError('cycle output must be a child of meta/sleep/')
    return out


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_cycle(root: Path, packet: dict, output: Path) -> None:
    brief, receipt = build_cycle(root, packet)
    out = _cycle_path(root, output)
    out.mkdir(parents=True, exist_ok=False)
    _json(out / 'packet.json', packet)
    _json(out / 'receipt.json', receipt)
    (out / 'brief.md').write_text(brief)


def _verify(root, output):
    out = _cycle_path(root, output)
    packet = json.loads((out / 'packet.json').read_text())
    brief, receipt = build_cycle(root, packet)
    if receipt != json.loads((out / 'receipt.json').read_text()) or brief != (out / 'brief.md').read_text():
        raise ValueError('cycle or source content changed; new review required')
    return out, brief, receipt


def review_cycle(root: Path, output: Path, review: dict) -> None:
    out, _, receipt = _verify(root, output)
    if (review.get('content_hash') != receipt['content_hash'] or
        type(review.get('accepted')) is not bool or not review.get('reviewer') or
        not review.get('notes')):
        raise ValueError('review must bind receipt hash, identify reviewer and record fidelity notes')
    with (out / 'review.json').open('x') as f:
        json.dump(review, f, indent=2, sort_keys=True)
        f.write('\n')


def load_reviewed_brief(root: Path, directory: Path) -> str:
    out, brief, receipt = _verify(root, directory)
    path = out / 'review.json'
    if not path.exists():
        raise ValueError('cycle requires independent review before attachment')
    review = json.loads(path.read_text())
    if (review.get('accepted') is not True or not review.get('reviewer') or
        not review.get('notes') or review.get('content_hash') != receipt['content_hash']):
        raise ValueError('cycle review missing, rejected or stale')
    return brief


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('build')
    p.add_argument('--packet', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('review')
    p.add_argument('--cycle', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p = sub.add_parser('inspect')
    p.add_argument('--cycle', type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path.cwd()
    try:
        if args.action == 'build':
            write_cycle(root, json.loads(args.packet.read_text()), args.out)
        elif args.action == 'review':
            review_cycle(root, args.cycle, json.loads(args.review.read_text()))
        else:
            print(load_reviewed_brief(root, args.cycle), end='')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'sleep: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
