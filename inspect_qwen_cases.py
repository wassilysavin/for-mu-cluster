"""Inspect thirteen saved Qwen cases without model inference or result changes."""

import argparse
from collections import Counter
import json
import sys

import inspect_qwen_failures as failures
import qualify_reference as qualification


EXPECTED_COUNTS = {'contract_not_implemented': 1, 'review_required': 4, 'invalid_arguments': 8}
DISCUSSED_REVIEW_ID = 'control:history-v3-train-error_and_success-18-fresh_after_success-control'


def load_evidence():
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
    rows = qualification.diagnostic.merged_rows(
        [child / f'gpu-{gpu}.jsonl' for gpu in range(4)], manifest['case_ids'])
    summary = failures.build_report(cases, rows)
    if (summary['total'] != 32 or summary['passed'] != result['metrics']['supported']['passed']
            or summary['valid_format'] != result['metrics']['transport_valid']['passed']):
        raise ValueError('Inspection counts differ from the completed qualification')
    return source, cases, rows, result


def select_cases(cases, rows):
    by_id = {row['case_id']: row for row in rows}
    case_ids = {envelope['case']['case_id'] for envelope in cases}
    if (len(by_id) != len(rows) or len(case_ids) != len(cases)
            or len(cases) != len(rows) or set(by_id) != case_ids):
        raise ValueError('Cases and saved observations differ')
    selected = []
    for envelope in cases:
        item = envelope['case']
        row = by_id[item['case_id']]
        categories = []
        if row.get('error_kind') in ('contract_not_implemented', 'invalid_arguments'):
            categories.append(row['error_kind'])
        if row['critical_status'] == 'review_required':
            categories.append('review_required')
        if not categories:
            continue
        if (len(categories) != 1 or row['success'] is not False or row['transport_valid'] is not True
                or (categories[0] == 'review_required' and row.get('error_kind') != 'semantic_unresolved')):
            raise ValueError('Selected case has inconsistent saved failure labels')
        selected.append({'case': item, 'observation': row, 'category': categories[0]})
    counts = dict(Counter(item['category'] for item in selected))
    if counts != EXPECTED_COUNTS:
        raise ValueError(f'Expected the reported 1 contract, 4 review and 8 argument cases; found {counts}')
    rank = {category: index for index, category in enumerate(EXPECTED_COUNTS)}
    selected.sort(key=lambda item: (rank[item['category']], item['case']['case_id']))
    return selected, counts


def reference_ids(value):
    if isinstance(value, dict):
        direct = {item for key, item in value.items()
                  if key in ('target', '_replace', '_resolve', '_supersede', 'revision_target', 'pending_target')
                  and isinstance(item, str)}
        return direct | set().union(*(reference_ids(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(reference_ids(item) for item in value), set())
    return set()


def history_context(item, calls):
    before = item['before']
    exchanges = before.get('exchanges', [])
    targets = reference_ids([calls, item['allowed_batches']])
    turn_ids = {target.split('.')[0] for target in targets if target.startswith('t')}
    selected = [exchange for index, exchange in enumerate(exchanges)
                if index in (0, len(exchanges) - 1) or exchange.get('id') in targets
                or exchange.get('semantic_turn_id') in turn_ids]
    answers = [answer for turn in before.get('semantic', {}).get('turns', [])
               for answer in turn.get('answers', []) if answer.get('id') in targets]
    return {'exchanges': selected, 'omitted_exchanges': len(exchanges) - len(selected),
            'referenced_answers': answers, 'pending': before.get('semantic', {}).get('pending', [])}


def build_report(cases, rows, rpc):
    selected, counts = select_cases(cases, rows)
    items = []
    for entry in selected:
        item, observation = entry['case'], entry['observation']
        inspection = rpc.request({'action': 'inspect', 'case': item, 'observation': observation},
                                 'inspection')['inspection']
        if (inspection['case_id'] != item['case_id'] or inspection['error_kind'] != observation['error_kind']
                or inspection['critical_status'] != observation['critical_status']):
            raise ValueError('Detailed inspection disagrees with saved case identity or labels')
        items.append({'case_id': item['case_id'], 'category': entry['category'], 'question': item['question'],
                      'allowed_batches': item['allowed_batches'], 'before': item['before'],
                      'context': history_context(item, inspection['model_calls']), 'inspection': inspection})
    return {'counts': counts, 'total': len(items), 'items': items, 'unresolved': counts['review_required']}


def load_report():
    source, cases, rows, _ = load_evidence()
    with qualification.diagnostic.NodeRPC('inspect_qwen_cases.mjs', source['registry']) as rpc:
        return build_report(cases, rows, rpc)


def compact(value, limit=None):
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit] + f' ... [clipped {len(text) - limit} chars; use --full]'


def show_context(context):
    for exchange in context['exchanges']:
        print('  History:', compact({'id': exchange.get('id'), 'question': exchange.get('question'),
                                    'outcome': exchange.get('outcome')}))
        print('    Calls:', compact(exchange.get('model_calls'), 240),
              'Reply:', compact(exchange.get('assistant'), 200))
    if context['omitted_exchanges']:
        print(f'  History: {context["omitted_exchanges"]} other exchanges omitted; use --full.')
    if context['referenced_answers']:
        print('  Addressed answers:', compact(context['referenced_answers'], 500))
    if context['pending']:
        print('  Pending:', compact(context['pending'], 350))


def show_review(review):
    for alternative in review['alternatives']:
        delivered, semantic = alternative['delivered'], alternative['semantic']
        print('  Allowed alternative:', compact({'index': alternative['index'], 'calls': alternative['calls'],
                                                'display_match': delivered['equal'], 'state_match': semantic['equal']}))
        if not delivered['equal']:
            print('    Reply difference:', compact({key: delivered[key] for key in
                  ('first_difference', 'actual_chars', 'expected_chars', 'actual_excerpt', 'expected_excerpt')}))
        if not semantic['equal']:
            print('    State differences:', compact(semantic['entries'], 850))
            if semantic['truncated']:
                print(f'    State differences limited to {semantic["limit"]}; --full includes complete actual/expected states.')


def show(report, full=False):
    if full:
        print(compact(report))
        return
    print(f'Saved Qwen cases: {report["total"]}; ' + compact(report['counts']))
    for index, item in enumerate(report['items'], 1):
        inspection = item['inspection']
        print(f'[{index}] {item["category"]}: ' + compact(item['case_id']))
        print('  Question:', compact(item['question']))
        show_context(item['context'])
        print('  Qwen calls:', compact(inspection['model_calls']))
        trace = inspection['trace']
        if trace.get('error'):
            print('  Exact failure:', compact({'stage': trace['stage'], 'call_index': trace.get('call_index'),
                                               'error': trace['error'], 'attribution': trace.get('attribution')}))
            for schema in inspection.get('schemas', []):
                if not schema['valid']:
                    print('  Violated schema:', compact({key: schema[key] for key in
                          ('call_index', 'attributed', 'source', 'name', 'path', 'fragment') if key in schema}, 650))
        if inspection.get('contract'):
            contract = inspection['contract']
            print('  Binding coverage:', compact({key: contract[key] for key in
                  ('domain_call', 'schema_valid', 'exact_binding_present', 'same_function_count', 'same_function_subject_count')}))
            print('  Existing binding examples:', compact(contract['same_function_subject_examples'] or
                                                          contract['same_function_examples'], 600))
        if inspection.get('review'):
            show_review(inspection['review'])
        else:
            print('  Frozen allowed batches:', compact(item['allowed_batches'], 700))
    print(f'{report["unresolved"]} reviews remain unresolved. Allowed programs are frozen labels, not an independent scientific adjudication.')
    print('Missing exact bindings do not alone identify model error versus registry coverage. Call indices are zero-based; null means no unique attribution.')
    print('All 32 saved identities, scores and hashes verified. CPU executor replay only; no model, downloads or saved-result changes.')


def remaining_reviews(report):
    reviews = [(index, item) for index, item in enumerate(report['items'], 1)
               if item['category'] == 'review_required']
    ids = [item['case_id'] for _, item in reviews]
    if (report['unresolved'] != 4 or report['counts'].get('review_required') != 4
            or len(ids) != 4 or len(set(ids)) != 4 or ids.count(DISCUSSED_REVIEW_ID) != 1
            or any(item['inspection']['critical_status'] != 'review_required'
                   or (item['inspection'].get('review') or {}).get('status') != 'review_required'
                   for _, item in reviews)):
        raise ValueError('Expected four saved reviews including the previously displayed case')
    return [(index, item) for index, item in reviews if item['case_id'] != DISCUSSED_REVIEW_ID]


def short_value(value, limit):
    text = compact(value)
    marker = ' ... [clipped; --full]'
    if limit <= len(marker):
        raise ValueError('Display limit is too short for an explicit clipping marker')
    return text if len(text) <= limit else text[:limit - len(marker)] + marker


def reply_excerpt(text, utf16_offset):
    # The Node comparator records UTF-16 offsets, not Python character offsets.
    position = len(text.encode('utf-16-le')[:utf16_offset * 2].decode('utf-16-le', errors='ignore'))
    return text[max(0, position - 4):position + 76]


def exchange_map(item):
    before = item['before']
    turns = {turn['id']: turn for turn in before.get('semantic', {}).get('turns', [])}
    result = []
    for exchange in before.get('exchanges', []):
        turn_id = exchange.get('semantic_turn_id')
        if turn_id is not None and turn_id not in turns:
            raise ValueError('Saved exchange refers to a missing semantic turn')
        turn = turns.get(turn_id, {})
        result.append([exchange['id'], exchange['outcome'], exchange.get('error_code'), turn_id,
                       [answer['id'] for answer in turn.get('answers', [])],
                       [part['kind'] for part in turn.get('response_parts', [])]])
    return result


def show_remaining_reviews(report):
    selected = remaining_reviews(report)
    print('Remaining for discussion: 3/4 saved reviews. Previously displayed case excluded only from this view.')
    for index, item in selected:
        inspection = item['inspection']
        review = inspection['review']
        alternatives = review['alternatives']
        if not alternatives:
            raise ValueError('Saved review has no frozen alternatives')
        first = alternatives[0]
        print(f'[{index}] ' + short_value(item['case_id'], 120))
        print('  Question:', short_value(item['question'], 260))
        print('  All exchanges [id, outcome, error, turn, answer IDs, part kinds]:', compact(exchange_map(item)))
        print('  Qwen:', short_value(inspection['model_calls'], 340))
        print(f'  Frozen alternative {first["index"]}/{len(alternatives)} (zero-based):',
              short_value(first['calls'], 340))
        display = sum(value['delivered']['equal'] for value in alternatives)
        state = sum(value['semantic']['equal'] for value in alternatives)
        both = sum(value['delivered']['equal'] and value['semantic']['equal'] for value in alternatives)
        print(f'  All {len(alternatives)} alternatives: display_match={display}; state_match={state}; both_match={both}.')
        offset = first['delivered']['first_difference']
        if offset is not None:
            actual = reply_excerpt(review['actual']['delivered'], offset)
            expected = reply_excerpt(first['expected']['delivered'], offset)
            print(f'  Reply excerpts at alternative {first["index"]} difference {offset}:',
                  'actual=' + short_value(actual, 140), 'expected=' + short_value(expected, 140))
        semantic = first['semantic']
        paths = [value['path'] for value in semantic['entries']]
        suffix = '+' if semantic['truncated'] else ''
        print(f'  Sampled state paths (first 3/{len(paths)}{suffix}, alternative {first["index"]}):',
              short_value(paths[:3], 120))
    print('All 4 saved reviews remain unresolved; scores unchanged. Excerpts are partial; --full retains all histories and alternatives.')
    print('Frozen labels are not independent scientific adjudication.')
    print('All 32 saved cases authenticated. No model run, download or saved-result change.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    view = parser.add_mutually_exclusive_group()
    view.add_argument('--full', action='store_true', help='Print complete diagnostic JSON and selected histories')
    view.add_argument('--remaining-reviews', action='store_true',
                      help='Briefly show the three reviews after the already displayed first review; scores stay unchanged')
    args = parser.parse_args()
    report = load_report()
    if args.remaining_reviews:
        show_remaining_reviews(report)
    else:
        show(report, full=args.full)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        sys.exit(f'Qwen case inspection error: {error}')
