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
