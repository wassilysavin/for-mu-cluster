"""Explain the saved reference qualification without loading a model."""

from private_runtime import configure_private_runtime

configure_private_runtime()

import argparse
from collections import Counter
from hashlib import file_digest, sha256
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parent
RUN_ID = 'train-fit-reference-v1'
EXCERPT_CHARS = 1000


def digest(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode()
    return sha256(encoded).hexdigest()


def inside(path):
    path = Path(path).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Report files must stay inside this checkout')
    return path


def read(path):
    return json.loads(inside(path).read_text())


def record(path):
    path = inside(path)
    with path.open('rb') as stream:
        checksum = file_digest(stream, 'sha256').hexdigest()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
            'sha256': checksum}


def verify(entry):
    path = inside(ROOT / entry['path'])
    if record(path) != entry:
        raise ValueError(f"Saved report hash changed: {entry['path']}")
    return path


def metric(rows, field):
    passed = sum(row.get(field) is True and (field != 'success'
                 or row.get('critical_status') == 'none_observed') for row in rows)
    return {'passed': passed, 'planned': len(rows),
            'accuracy': passed / len(rows) if rows else None}


def load_report(run_id=RUN_ID):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}', run_id):
        raise ValueError('Invalid run ID')
    directory = inside(ROOT / 'runs' / run_id)
    manifest = read(directory / 'experiment.json')
    identity = read(directory / 'identity.json')
    result = read(directory / 'result.json')
    binding = digest(manifest)
    if (manifest['identity_sha256'] != digest(identity)
            or result['identity_sha256'] != manifest['identity_sha256']
            or read(directory / 'manifest-binding.json') != {'sha256': binding}):
        raise ValueError('Saved experiment identities do not agree')
    for filename in ('experiment.json', 'manifest-binding.json'):
        if record(directory / filename) not in result['artifacts']:
            raise ValueError(f'Saved manifest differs from the terminal artifact inventory: {filename}')
    receipt_path = directory / 'stages/reference-transport.json'
    qualification = read(receipt_path)
    if (record(receipt_path) not in result['artifacts']
            or qualification != result['reference_transport']
            or qualification['identity_sha256'] != manifest['identity_sha256']):
        raise ValueError('Reference admission differs from the terminal report')
    policy = identity['policy']
    if policy['qualification_cases'] != 32 or policy['transport_minimum'] != 30:
        raise ValueError('Unexpected reference admission policy')
    planned = manifest['train_panel']['qualification_ids']
    if len(planned) != 32 or len(set(planned)) != 32:
        raise ValueError('Reference admission must contain 32 distinct planned cases')
    expected = {str((directory / 'reference/qualification' / f'train-{gpu}.jsonl').relative_to(ROOT)): gpu
                for gpu in range(4)}
    entries = qualification['observations']
    if len(entries) != 4 or {entry['path'] for entry in entries} != set(expected):
        raise ValueError('Expected exactly the four saved qualification shards')
    observations = []
    for entry in entries:
        gpu = expected[entry['path']]
        worker = digest({'manifest': binding, 'model': 'reference', 'phase': 'qualification',
                         'gpu': gpu, 'weights': manifest['reference']['fingerprint']})
        path = verify(entry)
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                shard = int(sha256(row['case_id'].encode()).hexdigest()[:8], 16) % 4
                if shard != gpu or row['generation']['diagnostic_worker_sha256'] != worker:
                    raise ValueError('Saved response belongs to another reference worker')
                observations.append(row)
    ids = [row['case_id'] for row in observations]
    if len(ids) != len(set(ids)) or set(ids) != set(planned):
        raise ValueError('Reference observations are missing, duplicated or unplanned')
    supported = [row for row in observations if row['semantic_eligible'] and row['cohort'] == 'supported']
    counts = {'transport_valid': metric(observations, 'transport_valid'), 'supported': metric(supported, 'success')}
    if any(qualification['all_fixed_train_panel'][key] != value for key, value in counts.items()):
        raise ValueError('Saved admission counts differ from the observations')
    gates = {'native_format': counts['transport_valid']['passed'] >= 30,
             'complete_supported_answer': counts['supported']['passed'] >= 1}
    passed = all(gates.values())
    status = 'COMPLETE_DIAGNOSTIC' if passed else 'TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED'
    if qualification['passed'] is not passed or result['status'] != status:
        raise ValueError('Saved reference admission decision does not reproduce')
    return {'run_id': run_id, 'passed': passed, 'counts': counts, 'gates': gates,
            'observations': observations, 'path': receipt_path}


def select_examples(report, count):
    rows = sorted(report['observations'], key=lambda row: row['case_id'])
    selected = []
    if not report['gates']['native_format']:
        invalid = [row for row in rows if row.get('transport_valid') is not True]
        if invalid:
            selected.append(invalid[0])
    if not report['gates']['complete_supported_answer']:
        supported = [row for row in rows if row['semantic_eligible'] and row['cohort'] == 'supported']
        supported.sort(key=lambda row: (row.get('transport_valid') is not True, row['case_id']))
        if supported and supported[0] not in selected:
            selected.append(supported[0])
    failed = [row for row in rows if row.get('success') is not True]
    categories = {(row.get('error_kind'), row['generation'].get('finish_reason'), row.get('transport_valid'))
                  for row in selected}
    for row in failed:
        key = (row.get('error_kind'), row['generation'].get('finish_reason'), row.get('transport_valid'))
        if key not in categories:
            categories.add(key)
            selected.append(row)
    selected += [row for row in failed if row not in selected]
    return selected[:count]


def show(report, examples=2):
    print('Reference admission:', 'PASS' if report['passed'] else 'NOT QUALIFIED')
    for key, label, minimum in (('transport_valid', 'Valid native format', 30),
                                ('supported', 'Complete supported answers', 1)):
        value = report['counts'][key]
        print(f"{label}: {value['passed']}/{value['planned']} (required: at least {minimum})")
    print('Failed gates:', ', '.join(key for key, passed in report['gates'].items() if not passed) or 'none')
    for field, label in (('finish_reason', 'Generation stops'), ('error_kind', 'Scoring errors')):
        values = Counter(str((row['generation'] if field == 'finish_reason' else row).get(field) or 'none')
                         for row in report['observations'])
        print(label + ':', json.dumps(dict(sorted(values.items())), ensure_ascii=True))
    for index, row in enumerate(select_examples(report, examples), 1):
        generation = row['generation']
        raw = generation.get('raw_text', '')
        print(f"Example {index}:", json.dumps({'case_id': row['case_id'], 'cohort': row['cohort'],
              'error': row.get('error_kind'), 'stop': generation.get('finish_reason'),
              'input_tokens': generation.get('input_tokens'), 'output_tokens': len(generation.get('raw_ids', [])),
              'raw_chars': len(raw)}, ensure_ascii=True))
        print(f'Raw response (first {EXCERPT_CHARS} characters):', json.dumps(raw[:EXCERPT_CHARS], ensure_ascii=True))
        if len(raw) > EXCERPT_CHARS:
            print('Raw response (last 200 characters):', json.dumps(raw[-200:], ensure_ascii=True))
    print('Saved identities, shard hashes and recorded counts agree. No model was run or saved results changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=RUN_ID)
    parser.add_argument('--examples', type=int, choices=range(5), default=2)
    args = parser.parse_args()
    show(load_report(args.run_id), args.examples)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f'Reference inspection error: {error}')
