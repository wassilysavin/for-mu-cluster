"""Read-only controls for targeted inspection of 13 saved Qwen cases."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import inspect_qwen_cases as report
import qualify_reference as qualification
import test_qualify_reference as fixtures
import test_inspect_qwen_failures as saved_fixtures


def targeted_cases():
    cases, rows = [], []
    for index in range(32):
        passed = index < 2
        critical = 'none_observed'
        failure = error = None
        if index == 2:
            error, failure = 'contract_not_implemented', 'execution_error'
        elif 3 <= index <= 6:
            error = failure = 'semantic_unresolved'
            critical = 'review_required'
        elif 7 <= index <= 14:
            error, failure = 'invalid_arguments', 'execution_error'
        elif 15 <= index <= 27:
            error = failure = 'omitted_reference'
            critical = 'detected'
        elif index == 28:
            error = failure = 'wrong_reference'
            critical = 'detected'
        elif index == 29:
            error = failure = 'unnecessary_clarification'
        elif index == 30:
            error, failure = 'unknown_function', 'execution_error'
        elif index == 31:
            error, failure = 'incomplete_model_output', 'execution_error'
        item, row = saved_fixtures.ExplanationFixture.example(
            index, passed=passed, critical=critical, valid=index != 31)
        item['question'] = f'Complete question {index}: ' + 'history conditions ' * 120 + 'UNCHANGED_END'
        row['error_kind'] = error
        row['semantic_result'].update(failure_category=failure)
        cases.append({'case': item}); rows.append(row)
    return cases, rows


class InspectionRPC:
    def __init__(self):
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def request(self, request, expected):
        if request['action'] != 'inspect' or expected != 'inspection':
            raise AssertionError('Unexpected inspection protocol')
        self.requests.append(deepcopy(request))
        item, row = request['case'], request['observation']
        inspection = {'case_id': item['case_id'], 'error_kind': row['error_kind'],
                      'critical_status': row['critical_status'],
                      'model_calls': [{'name': 'repeat_answer', 'arguments': {'target': 't2.c1'}}],
                      'trace': {'stage': 'complete', 'error': None}, 'schemas': [], 'review': None}
        if row['critical_status'] == 'review_required':
            inspection['review'] = {'status': 'review_required',
                'actual': {'text': 'FULL_ACTUAL_REPLY', 'semantic': {'full': 'FULL_ACTUAL_STATE'}},
                'alternatives': [{
                'index': 0, 'calls': item['allowed_batches'][0],
                'expected': {'text': 'FULL_ALLOWED_REPLY', 'semantic': {'full': 'FULL_ALLOWED_STATE'}},
                'delivered': {'equal': False, 'first_difference': 12,
                              'actual_chars': 99, 'expected_chars': 100,
                              'actual_excerpt': 'actual scientific reply',
                              'expected_excerpt': 'allowed scientific reply'},
                'semantic': {'equal': False, 'truncated': False, 'limit': 16,
                             'entries': [{'path': 'turns[0].subject',
                    'actual': 'TR', 'expected': 'BIDS.basepath'}]}}]}
        else:
            inspection['trace'] = {'stage': 'registry', 'call_index': 0,
                                   'error': {'code': row['error_kind']}, 'attribution': 'exact call'}
            inspection['schemas'] = [{'source': 'registry', 'name': 'repeat_answer',
                'path': 'arguments', 'valid': False,
                'fragment': {'required': ['target'], 'description': 'schema detail ' * 150}}]
        return {'inspection': inspection}


class SelectionTests(unittest.TestCase):
    def test_exact13_selected_in_expected_categories_without_truncating_questions(self):
        cases, rows = targeted_cases()
        before = deepcopy((cases, rows))
        selected, counts = report.select_cases(cases, rows)
        self.assertEqual(counts, {'contract_not_implemented': 1, 'review_required': 4, 'invalid_arguments': 8})
        self.assertEqual(len(selected), 13)
        self.assertEqual({entry['case']['case_id'] for entry in selected},
                         {f'case-{index:02d}' for index in range(2, 15)})
        for entry in selected:
            self.assertTrue(entry['case']['question'].endswith('UNCHANGED_END'))
            self.assertGreater(len(entry['case']['question']), 1000)
            self.assertEqual(entry['observation']['case_id'], entry['case']['case_id'])
        self.assertEqual((cases, rows), before)

    def test_incomplete_duplicate_or_unplanned_evidence_rejected(self):
        cases, rows = targeted_cases()
        mutations = [(cases[:-1], rows), (cases, rows[:-1]),
                     (cases + [deepcopy(cases[0])], rows),
                     (cases, rows + [deepcopy(rows[0])]),
                     (cases, [{**rows[0], 'case_id': 'unplanned'}, *rows[1:]])]
        for altered_cases, altered_rows in mutations:
            with self.subTest(cases=len(altered_cases), rows=len(altered_rows)), self.assertRaises(ValueError):
                report.select_cases(altered_cases, altered_rows)

    def test_target_counts_must_match_reported_frozen_categories(self):
        cases, rows = targeted_cases()
        rows[7]['error_kind'] = 'unknown_function'
        with self.assertRaises(ValueError):
            report.select_cases(cases, rows)

    def test_overlapping_categories_are_rejected_instead_of_counted_twice(self):
        cases, rows = targeted_cases()
        rows[2]['critical_status'] = 'review_required'
        with self.assertRaises(ValueError):
            report.select_cases(cases, rows)

    def test_targeted_pass_or_invalid_transport_is_not_inspected_as_expected_failure(self):
        for index, field, value in ((2, 'success', True), (3, 'success', True),
                                     (7, 'transport_valid', False), (3, 'transport_valid', False)):
            cases, rows = targeted_cases()
            rows[index][field] = value
            with self.subTest(index=index, field=field), self.assertRaises(ValueError):
                report.select_cases(cases, rows)

    def test_review_required_retains_the_expected_unresolved_error(self):
        cases, rows = targeted_cases()
        rows[3]['error_kind'] = 'wrong_reference'
        with self.assertRaises(ValueError):
            report.select_cases(cases, rows)


class EvidenceTests(fixtures.Fixture):
    prepare = saved_fixtures.SavedReportTests.prepare
    prohibit_inference = saved_fixtures.SavedReportTests.prohibit_inference

    def test_authenticates_saved32_without_models_weights_or_file_changes(self):
        cases, _, source, result = self.prepare()
        self.prohibit_inference()
        self.write(self.parent / 'untouched.json', {'closed': True})
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        loaded_source, loaded_cases, rows, loaded_result = report.load_evidence()
        self.assertEqual(loaded_source, source)
        self.assertEqual(loaded_cases, cases)
        self.assertEqual(loaded_result, result)
        self.assertEqual(len(rows), 32)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_tampered_shard_or_summary_fails_before_inspection(self):
        _, _, _, result = self.prepare()
        self.prohibit_inference()
        path = self.child / 'gpu-0.jsonl'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaises(ValueError):
            report.load_evidence()
        path.write_bytes(original)
        changed = deepcopy(result)
        changed['metrics']['supported']['passed'] = 3
        self.write(self.child / 'result.json', changed)
        with self.assertRaises(ValueError):
            report.load_evidence()

    def test_changed_manifest_and_missing_terminal_report_fail_closed(self):
        self.prepare()
        self.prohibit_inference()
        manifest = self.child / 'experiment.json'
        before = manifest.read_bytes()
        value = json.loads(before)
        value['changed'] = True
        self.write(manifest, value)
        with self.assertRaises(ValueError):
            report.load_evidence()
        manifest.write_bytes(before)
        (self.child / 'result.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            report.load_evidence()

    def test_parent_authentication_failure_is_propagated(self):
        self.prepare()
        self.prohibit_inference()
        with patch.object(qualification, 'load_source', side_effect=ValueError('parent changed')):
            with self.assertRaisesRegex(ValueError, 'parent changed'):
                report.load_evidence()


class ReportTests(unittest.TestCase):
    def test_inspects_only13_cases_and_preserves_full_evidence_without_reclassifying_reviews(self):
        cases, rows = targeted_cases()
        before = deepcopy((cases, rows))
        rpc = InspectionRPC()
        result = report.build_report(cases, rows, rpc)
        self.assertEqual((result['total'], result['unresolved']), (13, 4))
        self.assertEqual(result['counts'], report.EXPECTED_COUNTS)
        self.assertEqual(len(rpc.requests), 13)
        self.assertEqual({request['case']['case_id'] for request in rpc.requests},
                         {f'case-{index:02d}' for index in range(2, 15)})
        selected_cases = {entry['case']['case_id']: entry['case'] for entry in cases}
        for item in result['items']:
            expected = selected_cases[item['case_id']]
            self.assertEqual(item['question'], expected['question'])
            self.assertEqual(item['allowed_batches'], expected['allowed_batches'])
            self.assertEqual(item['before'], expected['before'])
            if item['category'] == 'review_required':
                self.assertEqual(item['inspection']['critical_status'], 'review_required')
                self.assertEqual(item['inspection']['error_kind'], 'semantic_unresolved')
        self.assertEqual((cases, rows), before)

    def test_replay_identity_or_label_mismatch_aborts(self):
        cases, rows = targeted_cases()
        for field, value in (('case_id', 'different'), ('error_kind', 'different'),
                             ('critical_status', 'none_observed')):
            rpc = InspectionRPC()
            original = rpc.request

            def changed(request, expected):
                response = original(request, expected)
                if response['inspection']['critical_status'] == 'review_required':
                    response['inspection'][field] = value
                return response

            rpc.request = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                report.build_report(cases, rows, rpc)

    def test_context_includes_addressed_middle_exchange_and_keeps_full_history(self):
        cases, rows = targeted_cases()
        item = cases[2]['case']
        exchanges = [{'id': f'e{index}', 'semantic_turn_id': f't{index}',
                      'question': f'question {index}', 'assistant': f'reply {index}'}
                     for index in range(1, 5)]
        item['before'] = {'exchanges': exchanges, 'semantic': {
            'turns': [{'answers': [{'id': 't2.c1', 'subject': 'TR'}]}], 'pending': ['unchanged']}}
        result = report.build_report(cases, rows, InspectionRPC())
        selected = next(entry for entry in result['items'] if entry['case_id'] == item['case_id'])
        context = selected['context']
        self.assertEqual([exchange['id'] for exchange in context['exchanges']], ['e1', 'e2', 'e4'])
        self.assertEqual(context['omitted_exchanges'], 1)
        self.assertEqual(context['referenced_answers'], [{'id': 't2.c1', 'subject': 'TR'}])
        self.assertEqual(context['pending'], ['unchanged'])
        self.assertEqual(selected['before']['exchanges'], exchanges)

    def test_terminal_output_marks_clipping_and_does_not_adjudicate_reviews(self):
        cases, rows = targeted_cases()
        cases[2]['case']['question'] += '\x1b[31m\nUNTRUSTED_END'
        result = report.build_report(cases, rows, InspectionRPC())
        output = io.StringIO()
        with redirect_stdout(output):
            report.show(result)
        text = output.getvalue()
        self.assertNotIn('\x1b', text)
        self.assertIn('\\u001b', text)
        self.assertEqual(text.count('UNCHANGED_END'), 13)
        self.assertIn('UNTRUSTED_END', text)
        self.assertIn('clipped', text)
        self.assertIn('--full', text)
        self.assertIn('4 reviews remain unresolved', text)
        self.assertIn('not an independent scientific adjudication', text)
        self.assertIn('actual scientific reply', text)
        self.assertIn('allowed scientific reply', text)
        for sentinel in ('FULL_ACTUAL_REPLY', 'FULL_ACTUAL_STATE', 'FULL_ALLOWED_REPLY', 'FULL_ALLOWED_STATE'):
            self.assertNotIn(sentinel, text)

    def test_full_output_is_lossless_json(self):
        cases, rows = targeted_cases()
        result = report.build_report(cases, rows, InspectionRPC())
        output = io.StringIO()
        with redirect_stdout(output):
            report.show(result, full=True)
        self.assertEqual(json.loads(output.getvalue()), result)

    def test_failed_authentication_does_not_open_inspection_rpc(self):
        with patch.object(report, 'load_evidence', side_effect=ValueError('tampered saved evidence')):
            with patch.object(qualification.diagnostic, 'NodeRPC') as rpc:
                with self.assertRaisesRegex(ValueError, 'tampered saved evidence'):
                    report.load_report()
                rpc.assert_not_called()


class RemainingReviewTests(unittest.TestCase):
    @staticmethod
    def shifted_history():
        return {'exchanges': [
            {'id': 'e1', 'outcome': 'committed', 'semantic_turn_id': 't1'},
            {'id': 'e2', 'outcome': 'committed', 'semantic_turn_id': 't2'},
            {'id': 'e3', 'outcome': 'committed', 'semantic_turn_id': 't3'},
            {'id': 'e4', 'outcome': 'error', 'error_code': 'invalid_arguments'},
            {'id': 'e5', 'outcome': 'committed', 'semantic_turn_id': 't4'}],
            'semantic': {'turns': [
                {'id': 't4', 'answers': [{'id': 't4.c1'}], 'response_parts': [{'kind': 'answer'}]},
                {'id': 't3', 'answers': [], 'response_parts': [{'kind': 'replayed_exchange'}]},
                {'id': 't2', 'answers': [], 'response_parts': [{'kind': 'reference_clarification'}]},
                {'id': 't1', 'answers': [{'id': 't1.c1'}], 'response_parts': [{'kind': 'answer'}]}]}}

    def test_exchange_map_keeps_failures_and_resolves_turns_by_id_not_position(self):
        item = {'before': self.shifted_history()}
        before = deepcopy(item)
        self.assertEqual(report.exchange_map(item), [
            ['e1', 'committed', None, 't1', ['t1.c1'], ['answer']],
            ['e2', 'committed', None, 't2', [], ['reference_clarification']],
            ['e3', 'committed', None, 't3', [], ['replayed_exchange']],
            ['e4', 'error', 'invalid_arguments', None, [], []],
            ['e5', 'committed', None, 't4', ['t4.c1'], ['answer']]])
        self.assertEqual(item, before)

    def test_brief_view_prints_all_exchanges_even_if_context_excluded_them(self):
        result = self.prepared()
        for _, item in report.remaining_reviews(result):
            item['before'] = self.shifted_history()
            item['context']['exchanges'] = [item['before']['exchanges'][0], item['before']['exchanges'][-1]]
        before = deepcopy(result)
        text = self.rendered(result)
        for _, item in report.remaining_reviews(result):
            self.assertIn(report.compact(report.exchange_map(item)), text)
        self.assertEqual(text.count('All exchanges [id, outcome, error, turn, answer IDs, part kinds]:'), 3)
        self.assertLessEqual(len(text), 6500)
        self.assertEqual(result, before)

    def test_missing_semantic_turn_in_exchange_map_is_rejected(self):
        item = {'before': self.shifted_history()}
        item['before']['exchanges'][-1]['semantic_turn_id'] = 't99'
        with self.assertRaisesRegex(ValueError, 'missing semantic turn'):
            report.exchange_map(item)

    @staticmethod
    def prepared():
        cases, rows = targeted_cases()
        result = report.build_report(cases, rows, InspectionRPC())
        discussed = next(item for item in result['items'] if item['case_id'] == 'case-03')
        discussed['case_id'] = report.DISCUSSED_REVIEW_ID
        discussed['inspection']['case_id'] = report.DISCUSSED_REVIEW_ID
        for item in result['items']:
            review = item['inspection']['review']
            if review is not None:
                review['actual']['delivered'] = review['actual']['text']
                for alternative in review['alternatives']:
                    alternative['expected']['delivered'] = alternative['expected']['text']
        return result

    @staticmethod
    def rendered(result):
        output = io.StringIO()
        with redirect_stdout(output):
            report.show_remaining_reviews(result)
        return output.getvalue()

    def test_selection_keeps_original_indices_and_review_flags_without_mutation(self):
        result = self.prepared()
        before = deepcopy(result)
        selected = report.remaining_reviews(result)
        self.assertEqual([index for index, _ in selected], [3, 4, 5])
        self.assertEqual([item['case_id'] for _, item in selected], ['case-04', 'case-05', 'case-06'])
        for index, item in selected:
            self.assertIs(item, result['items'][index - 1])
            self.assertEqual(item['inspection']['critical_status'], 'review_required')
            self.assertEqual(item['inspection']['error_kind'], 'semantic_unresolved')
        self.rendered(result)
        self.assertEqual(result, before)
        self.assertEqual((result['unresolved'], result['counts']['review_required']), (4, 4))

    def test_discussed_case_is_excluded_by_exact_identity_after_reordering(self):
        result = self.prepared()
        result['items'] = list(reversed(result['items']))
        expected = [(index, item) for index, item in enumerate(result['items'], 1)
                    if item['category'] == 'review_required' and item['case_id'] != report.DISCUSSED_REVIEW_ID]
        self.assertEqual(report.remaining_reviews(result), expected)
        self.assertEqual(len(expected), 3)
        self.assertNotIn(report.DISCUSSED_REVIEW_ID, [item['case_id'] for _, item in expected])

    def test_missing_duplicate_or_inconsistent_review_evidence_is_rejected(self):
        mutations = []
        missing_identity = self.prepared()
        missing_identity['items'][1]['case_id'] = 'another-control-case'
        mutations.append(missing_identity)
        duplicate_identity = self.prepared()
        duplicate_identity['items'][2]['case_id'] = report.DISCUSSED_REVIEW_ID
        duplicate_identity['items'][2]['inspection']['case_id'] = report.DISCUSSED_REVIEW_ID
        mutations.append(duplicate_identity)
        missing_review = self.prepared()
        missing_review['items'][2]['inspection']['review'] = None
        mutations.append(missing_review)
        missing_item = self.prepared()
        del missing_item['items'][2]
        mutations.append(missing_item)
        for field in ('unresolved', 'counts'):
            wrong_count = self.prepared()
            if field == 'unresolved':
                wrong_count[field] = 3
            else:
                wrong_count[field]['review_required'] = 3
            mutations.append(wrong_count)
        for index, result in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                report.remaining_reviews(result)

    def test_short_value_is_ascii_and_its_total_length_includes_the_clipping_marker(self):
        escaped = report.short_value({'message': '\x1b[31m\n\tПривет'}, 120)
        self.assertTrue(escaped.isascii())
        self.assertNotIn('\x1b', escaped)
        self.assertNotIn('\n', escaped)
        self.assertNotIn('\t', escaped)
        self.assertIn('\\u001b', escaped)
        self.assertEqual(json.loads(escaped), {'message': '\x1b[31m\n\tПривет'})
        for limit in (40, 120):
            clipped = report.short_value({'message': 'x' * 10000}, limit)
            with self.subTest(limit=limit):
                self.assertLessEqual(len(clipped), limit)
                self.assertIn('clipped', clipped)
                self.assertTrue(clipped.isascii())

    def test_compact_output_contains_only_three_reviews_and_counts_all_alternatives(self):
        result = self.prepared()
        for _, item in report.remaining_reviews(result):
            template = item['inspection']['review']['alternatives'][0]
            alternatives = []
            for index, (display_match, state_match) in enumerate(((False, False), (True, False), (False, True))):
                alternative = deepcopy(template)
                alternative['index'] = index
                alternative['delivered']['equal'] = display_match
                alternative['semantic']['equal'] = state_match
                alternatives.append(alternative)
            item['inspection']['review']['alternatives'] = alternatives
        text = self.rendered(result)
        for index, item in report.remaining_reviews(result):
            self.assertIn(f'[{index}]', text)
            self.assertIn(item['case_id'], text)
        self.assertNotIn('[2]', text)
        self.assertNotIn('contract_not_implemented', text)
        self.assertNotIn('invalid_arguments', text)
        self.assertIn('4', text)
        self.assertIn('3', text)
        self.assertGreaterEqual(text.count('display_match'), 3)
        self.assertGreaterEqual(text.count('state_match'), 3)
        self.assertGreaterEqual(text.count('both_match'), 3)
        self.assertEqual(text.count('All 3 alternatives: display_match=1; state_match=1; both_match=0.'), 3)
        self.assertIn('Question:', text)
        self.assertIn('repeat_answer', text)
        self.assertIn('TR', text)
        for _, item in report.remaining_reviews(result):
            item['inspection']['review']['alternatives'][1]['delivered']['equal'] = False
            item['inspection']['review']['alternatives'][2]['semantic']['equal'] = False
        self.assertNotEqual(text, self.rendered(result))

    def test_last_of_256_alternatives_contributes_to_totals_without_expanding_output(self):
        result = self.prepared()
        for _, item in report.remaining_reviews(result):
            review = item['inspection']['review']
            template = review['alternatives'][0]
            review['alternatives'] = []
            for index in range(256):
                alternative = deepcopy(template)
                alternative['index'] = index
                alternative['delivered']['equal'] = index == 255
                alternative['semantic']['equal'] = index == 255
                review['alternatives'].append(alternative)
        text = self.rendered(result)
        self.assertEqual(text.count('All 256 alternatives: display_match=1; state_match=1; both_match=1.'), 3)
        self.assertEqual(text.count('Frozen alternative 0/256'), 3)
        self.assertNotIn('Frozen alternative 255/', text)
        self.assertLessEqual(len(text), 6500)

    def test_large_fields_stay_short_and_terminal_control_characters_are_escaped(self):
        result = self.prepared()
        large = '\x1b[31m\n\tПривет' + 'x' * 30000
        for _, item in report.remaining_reviews(result):
            item['question'] = large
            item['inspection']['model_calls'][0]['arguments']['untrusted'] = large
            item['context'] = {
                'exchanges': [{'id': f'e{index}', 'question': large, 'assistant': large,
                               'model_calls': [{'name': 'history_call', 'arguments': {'text': large}}]}
                              for index in range(4)],
                'omitted_exchanges': 20,
                'referenced_answers': [{'id': 't2.c1', 'text': large}],
                'pending': [{'text': large}]}
            review = item['inspection']['review']
            review['actual']['delivered'] = large
            template = review['alternatives'][0]
            review['alternatives'] = []
            for index in range(3):
                alternative = deepcopy(template)
                alternative['index'] = index
                alternative['calls'][0]['arguments']['untrusted'] = large
                alternative['expected']['delivered'] = large + 'changed'
                alternative['semantic']['entries'] = [
                    {'path': f'turns[{entry}].subject', 'actual': large, 'expected': large + 'changed'}
                    for entry in range(16)]
                alternative['semantic']['truncated'] = True
                review['alternatives'].append(alternative)
        before = deepcopy(result)
        text = self.rendered(result)
        self.assertLessEqual(len(text), 6500)
        self.assertTrue(text.isascii())
        self.assertNotIn('\x1b', text)
        self.assertNotIn('\t', text)
        self.assertIn('\\u001b', text)
        self.assertIn('clipped', text)
        self.assertIn('sample', text.lower())
        self.assertGreaterEqual(text.count('display_match'), 3)
        self.assertGreaterEqual(text.count('state_match'), 3)
        self.assertEqual(result, before)

    def test_reply_snippet_starts_four_characters_before_the_actual_mismatch(self):
        result = self.prepared()
        selected = report.remaining_reviews(result)[0][1]
        review = selected['inspection']['review']
        prefix = 'OLD_PREFIX_NOT_NEAR_DIFFERENCE' + 'x' * 140
        context = 'CTX_'
        actual = prefix + context + 'ACTUAL_MISMATCH' + 'z' * 100
        expected = prefix + context + 'EXPECTED_MISMATCH' + 'z' * 100
        difference = len(prefix + context)
        review['actual']['delivered'] = actual
        alternative = review['alternatives'][0]
        alternative['expected']['delivered'] = expected
        alternative['delivered'].update(
            first_difference=difference, actual_chars=len(actual), expected_chars=len(expected),
            actual_excerpt=actual[difference - 80:difference + 80],
            expected_excerpt=expected[difference - 80:difference + 80])
        text = self.rendered(result)
        self.assertIn(context + 'ACTUAL_MISMATCH', text)
        self.assertIn(context + 'EXPECTED_MISMATCH', text)
        self.assertNotIn('x' + context, text)
        self.assertNotIn('OLD_PREFIX_NOT_NEAR_DIFFERENCE', text)

    def test_reply_snippet_converts_saved_utf16_offset_before_clipping_escaped_emoji(self):
        result = self.prepared()
        review = report.remaining_reviews(result)[0][1]['inspection']['review']
        prefix = 'prefix' + '\U0001f600' * 100
        actual = prefix + 'ACTUAL_MISMATCH' + 'x' * 100
        expected = prefix + 'EXPECTED_MISMATCH' + 'x' * 100
        review['actual']['delivered'] = actual
        alternative = review['alternatives'][0]
        alternative['expected']['delivered'] = expected
        alternative['delivered']['first_difference'] = len(prefix.encode('utf-16-le')) // 2
        text = self.rendered(result)
        escaped_context = '\\ud83d\\ude00' * 4
        self.assertIn(escaped_context + 'ACTUAL_MISMATCH', text)
        self.assertIn(escaped_context + 'EXPECTED_MISMATCH', text)
        self.assertNotIn('\U0001f600', text)


class CommandLineTests(unittest.TestCase):
    def test_remaining_reviews_uses_authenticated_loader_and_specialized_display(self):
        result = RemainingReviewTests.prepared()
        with patch.object(report, 'load_report', return_value=result) as load:
            with patch.object(report, 'show_remaining_reviews') as remaining:
                with patch.object(report, 'show') as normal:
                    with patch('sys.argv', ['inspect_qwen_cases.py', '--remaining-reviews']):
                        report.main()
        load.assert_called_once_with()
        remaining.assert_called_once_with(result)
        normal.assert_not_called()

    def test_full_and_default_display_routes_remain_available(self):
        result = RemainingReviewTests.prepared()
        for argv, full in (([], False), (['--full'], True)):
            with self.subTest(argv=argv), patch.object(report, 'load_report', return_value=result) as load:
                with patch.object(report, 'show_remaining_reviews') as remaining:
                    with patch.object(report, 'show') as normal:
                        with patch('sys.argv', ['inspect_qwen_cases.py', *argv]):
                            report.main()
            load.assert_called_once_with()
            normal.assert_called_once_with(result, full=full)
            remaining.assert_not_called()

    def test_remaining_reviews_and_full_are_mutually_exclusive_before_loading(self):
        with patch.object(report, 'load_report') as load:
            with patch('sys.argv', ['inspect_qwen_cases.py', '--remaining-reviews', '--full']):
                with patch('sys.stderr', new=io.StringIO()), self.assertRaises(SystemExit) as error:
                    report.main()
        self.assertEqual(error.exception.code, 2)
        load.assert_not_called()


class CompleteReportTests(fixtures.Fixture):
    prepare = saved_fixtures.SavedReportTests.prepare
    prohibit_inference = saved_fixtures.SavedReportTests.prohibit_inference

    @staticmethod
    def receipt(item, generation):
        row = fixtures.Fixture.receipt(item, generation)
        template = targeted_cases()[1][int(item['case_id'].split('-')[1])]
        for key in ('error_kind', 'critical_status', 'semantic_result'):
            row[key] = deepcopy(template[key])
        return row

    def test_authenticated_report_replays_scores_without_inference_downloads_or_result_writes(self):
        self.prepare()
        self.prohibit_inference()
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        inspection = InspectionRPC()
        scripts = []

        def rpc(script, registry):
            scripts.append(script)
            return inspection if script == 'inspect_qwen_cases.mjs' else self.rpc()

        with patch.object(qualification.diagnostic, 'NodeRPC', side_effect=rpc):
            result = report.load_report()
        self.assertEqual(scripts, ['diagnose-data.mjs', 'inspect_qwen_cases.mjs'])
        self.assertEqual((result['total'], result['unresolved']), (13, 4))
        self.assertEqual(len(inspection.requests), 13)
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()})


if __name__ == '__main__':
    unittest.main()
