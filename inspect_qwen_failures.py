"""Explain saved Qwen qualification failures without loading a model."""

import argparse
from collections import Counter
import json
import sys

import qualify_reference as qualification


def json_key(value):
    if value is None:
        return ('null',)
    if isinstance(value, bool):
        return ('boolean', value)
    if isinstance(value, (int, float)):
        return ('number', value)
    if isinstance(value, str):
        return ('string', value)
    if isinstance(value, list):
        return ('array', tuple(json_key(item) for item in value))
    if isinstance(value, dict):
        return ('object', tuple((key, json_key(value[key])) for key in sorted(value)))
    raise ValueError('Expected a JSON value')


def argument_differences(expected, actual, path='arguments'):
    if json_key(expected) == json_key(actual):
        return []
    if json_key(expected)[0] != json_key(actual)[0]:
        return [{'kind': 'argument_type', 'path': path, 'expected': expected, 'actual': actual}]
    if isinstance(expected, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            nested = path + '.' + key
            if key not in actual:
                differences.append({'kind': 'missing_argument', 'path': nested, 'expected': expected[key]})
            elif key not in expected:
                differences.append({'kind': 'extra_argument', 'path': nested, 'actual': actual[key]})
            else:
                differences.extend(argument_differences(expected[key], actual[key], nested))
        return differences
    if isinstance(expected, list) and len(expected) == len(actual):
        return [difference for index, (left, right) in enumerate(zip(expected, actual))
                for difference in argument_differences(left, right, f'{path}[{index}]')]
    return [{'kind': 'argument_value', 'path': path, 'expected': expected, 'actual': actual}]


def align_calls(actual, expected):
    if json_key(actual) == json_key(expected):
        return {'cost': 0, 'differences': [], 'alignment_ambiguous': False}
    if Counter(map(json_key, actual)) == Counter(map(json_key, expected)):
        return {'cost': 1, 'differences': [{'kind': 'call_order', 'expected': expected, 'actual': actual}],
                'alignment_ambiguous': False}
    costs = [[0] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    ways = [[1] * (len(actual) + 1) for _ in range(len(expected) + 1)]
    steps = {}
    for i in range(1, len(expected) + 1):
        costs[i][0] = i * 2
        steps[i, 0] = (i - 1, 0, [{'kind': 'missing_call', 'expected': expected[i - 1]}])
    for j in range(1, len(actual) + 1):
        costs[0][j] = j * 2
        steps[0, j] = (0, j - 1, [{'kind': 'extra_call', 'actual': actual[j - 1]}])
    for i, wanted in enumerate(expected, 1):
        for j, received in enumerate(actual, 1):
            if wanted['name'] == received['name']:
                diff = argument_differences(wanted['arguments'], received['arguments'])
                penalty = int(bool(diff))
            else:
                diff = [{'kind': 'wrong_function', 'expected': wanted, 'actual': received}]
                penalty = 3
            options = [(costs[i - 1][j - 1] + penalty, i - 1, j - 1, diff),
                       (costs[i - 1][j] + 2, i - 1, j, [{'kind': 'missing_call', 'expected': wanted}]),
                       (costs[i][j - 1] + 2, i, j - 1, [{'kind': 'extra_call', 'actual': received}])]
            minimum = min(option[0] for option in options)
            best = [option for option in options if option[0] == minimum]
            costs[i][j] = minimum
            ways[i][j] = min(2, sum(ways[x][y] for _, x, y, _ in best))
            _, x, y, differences = best[0]
            steps[i, j] = (x, y, differences)
    i, j = len(expected), len(actual)
    groups = []
    while i or j:
        i, j, differences = steps[i, j]
        groups.append(differences)
    return {'cost': costs[-1][-1], 'differences': [d for group in reversed(groups) for d in group],
            'alignment_ambiguous': ways[-1][-1] > 1}


def compare_calls(actual, allowed_batches):
    if not allowed_batches:
        raise ValueError('The frozen case has no allowed call batches')
    unique = {json_key(batch): batch for batch in allowed_batches}
    candidates = [{**align_calls(actual, batch), 'expected': batch} for batch in unique.values()]
    minimum = min(candidate['cost'] for candidate in candidates)
    closest = [candidate for candidate in candidates if candidate['cost'] == minimum]
    chosen = closest[0]
    return {**chosen, 'kinds': sorted({d['kind'] for d in chosen['differences']}),
            'closest_alternatives': len(closest),
            'alignment_ambiguous': len(closest) > 1 or chosen['alignment_ambiguous']}


def explain_case(item, row):
    passed = row['success'] is True and row['critical_status'] == 'none_observed'
    semantic = row.get('semantic_result') or {}
    record = {'case_id': item['case_id'], 'question': item['question'], 'group': row.get('group'),
              'passed': passed, 'transport_valid': row['transport_valid'],
              'critical_status': row['critical_status'], 'critical_category': semantic.get('critical_category'),
              'evaluator_failure': semantic.get('failure_category'), 'execution_error': row.get('error_kind'),
              'selected_reference_targets': semantic.get('selected_reference_targets'),
              'expected_reference_targets': semantic.get('expected_reference_targets'),
              'structural': None, 'actual_calls': None}
    if passed:
        return record
    generation = row['generation']
    if not row['transport_valid']:
        record['raw_excerpt'] = generation.get('reference_raw_text', '')[:500]
        return record
    marker = generation['eos_token_text']
    raw = generation['reference_raw_text']
    if not marker or not raw.endswith(marker):
        raise ValueError('Saved Hermes response lost its verified terminal token')
    actual = qualification.transport.parse_calls(raw[:-len(marker)])
    record['actual_calls'] = actual
    record['structural'] = compare_calls(actual, item['allowed_batches'])
    return record


def build_report(cases, rows):
    by_id = {row['case_id']: row for row in rows}
    case_ids = {case['case']['case_id'] for case in cases}
    if (len(by_id) != len(rows) or len(case_ids) != len(cases)
            or len(cases) != len(rows) or set(by_id) != case_ids):
        raise ValueError('Cases and saved observations differ')
    items = [explain_case(envelope['case'], by_id[envelope['case']['case_id']]) for envelope in cases]
    failed = [item for item in items if not item['passed']]
    structures, fields = Counter(), Counter()
    for item in failed:
        comparison = item['structural']
        if comparison is None:
            continue
        structures.update(comparison['kinds'])
        fields.update({difference['path'] for difference in comparison['differences'] if 'path' in difference})
    return {'total': len(items), 'passed': len(items) - len(failed), 'failed': len(failed),
            'valid_format': sum(item['transport_valid'] for item in items),
            'critical': sum(item['critical_status'] == 'detected' for item in items),
            'unresolved': sum(item['critical_status'] == 'review_required' for item in items),
            'evaluator_failures': dict(Counter(item['evaluator_failure'] or item['execution_error'] or 'not_classified' for item in failed)),
            'execution_errors': dict(Counter(item['execution_error'] for item in failed if item['execution_error'])),
            'critical_categories': dict(Counter(item['critical_category'] for item in failed if item['critical_category'])),
            'structural_counts': dict(structures), 'argument_fields': dict(fields),
            'ambiguous_alignments': sum(bool(item['structural'] and item['structural']['alignment_ambiguous']) for item in failed),
            'identical_allowed_but_unpassed': sum(bool(item['structural'] and item['structural']['cost'] == 0) for item in failed),
            'items': items}


def load_report():
    source = qualification.load_source()
    child = qualification.output_directory()
    result = qualification.diagnostic.read(child / 'result.json')
    manifest = qualification.diagnostic.read(child / 'experiment.json')
    cases = qualification.select_cases(source['manifest']['train_panel'])
    qualification.verify_manifest(manifest, source, cases)
    for entry in result['artifacts']:
        qualification.diagnostic.verify(entry)
    if result != qualification.collect(child, manifest, source, cases):
        raise ValueError('Saved qualification report no longer reproduces')
    rows = qualification.diagnostic.merged_rows([child / f'gpu-{gpu}.jsonl' for gpu in range(4)], manifest['case_ids'])
    report = build_report(cases, rows)
    if (report['total'] != 32 or report['passed'] != result['metrics']['supported']['passed']
            or report['valid_format'] != result['metrics']['transport_valid']['passed']):
        raise ValueError('Failure analysis counts differ from the completed qualification')
    report['path'] = child / 'result.json'
    return report


def examples_for(report, count):
    failed = sorted((item for item in report['items'] if not item['passed']), key=lambda item: item['case_id'])
    selected = []
    for predicate in (lambda item: not item['transport_valid'],
                      lambda item: item['critical_status'] == 'detected',
                      lambda item: item['evaluator_failure'] == 'execution_error',
                      lambda item: item['critical_status'] == 'review_required'):
        matches = [item for item in failed if predicate(item) and item not in selected]
        if matches:
            selected.append(matches[0])
    def signals(item):
        comparison = item['structural'] or {'kinds': ['invalid_format'], 'differences': []}
        return {('kind', kind) for kind in comparison['kinds']} | {
            ('field', difference['path']) for difference in comparison['differences'] if 'path' in difference}
    while len(selected) < count:
        remaining = [item for item in failed if item not in selected]
        if not remaining:
            break
        covered = set().union(*(signals(item) for item in selected))
        selected.append(max(remaining, key=lambda item: len(signals(item) - covered)))
    return selected[:count]


def excerpt(value, limit=240):
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False)
    return text if len(text) <= limit else text[:limit] + '...'


def show(report, examples=4, all_cases=False):
    print(f'Qwen saved qualification: {report["passed"]}/{report["total"]} passed; {report["failed"]} not passed; format {report["valid_format"]}/{report["total"]}.')
    print(f'Recorded critical: {report["critical"]}; unresolved review: {report["unresolved"]}.')
    for key, label in (('evaluator_failures', 'Evaluator failures'), ('execution_errors', 'Error codes'),
                       ('critical_categories', 'Critical categories'), ('structural_counts', 'Structural signals')):
        print(label + ':', excerpt(report[key], 900))
    top_fields = sorted(report['argument_fields'].items(), key=lambda item: (-item[1], item[0]))[:8]
    print('Argument fields (cases, top 8):', excerpt(dict(top_fields), 650))
    print(f'Ambiguous closest-program alignments: {report["ambiguous_alignments"]}; exact allowed program but not passed: {report["identical_allowed_but_unpassed"]}.')
    if all_cases:
        for item in report['items']:
            if not item['passed']:
                signals = item['structural']['kinds'] if item['structural'] else ['invalid_format']
                print(excerpt({'case': item['case_id'], 'evaluation': item['evaluator_failure'] or item['execution_error'],
                               'critical': item['critical_status'], 'signals': signals}, 500))
    else:
        for index, item in enumerate(examples_for(report, examples), 1):
            print(f'Example {index}:', excerpt({'case': item['case_id'], 'evaluation': item['evaluator_failure'] or item['execution_error'],
                  'critical': item['critical_status']}, 350))
            print('  Question:', excerpt(item['question'], 220))
            if item['structural'] is None:
                print('  Raw response:', excerpt(item.get('raw_excerpt'), 320))
            else:
                print('  Closest allowed calls:', excerpt(item['structural']['expected'], 350))
                print('  Qwen calls:', excerpt(item['actual_calls'], 350))
                print('  Differences:', excerpt(item['structural']['differences'], 450))
    print('Structural signals can overlap and describe differences from a closest frozen allowed program; they do not adjudicate unresolved reviews.')
    print('Saved identities, scores and hashes verified. No model loaded or saved results changed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples', type=int, choices=range(7), default=4)
    parser.add_argument('--all', action='store_true', help='List one diagnostic line for each nonpassing case')
    args = parser.parse_args()
    show(load_report(), args.examples, args.all)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        sys.exit(f'Qwen failure inspection error: {error}')
