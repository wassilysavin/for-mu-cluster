"""Offline controls for the read-only Hermes failure report."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
import unittest
from unittest.mock import patch

import inspect_qwen_failures as report
import qualify_reference as qualification
import test_qualify_reference as fixtures


def call(name='explain_purpose', **arguments):
    return {'name': name, 'arguments': arguments}


class JsonEqualityTests(unittest.TestCase):
    def test_json_type_equality_keeps_booleans_distinct_from_numbers(self):
        self.assertNotEqual(report.json_key(True), report.json_key(1))
        self.assertNotEqual(report.json_key(False), report.json_key(0))
        self.assertEqual(report.json_key(1), report.json_key(1.0))
        self.assertEqual(report.json_key({'a': 1, 'b': [False, None]}),
                         report.json_key({'b': [False, None], 'a': 1.0}))
        self.assertNotEqual(report.json_key({'a': None}), report.json_key({}))
        self.assertNotEqual(report.json_key([1, 2]), report.json_key([2, 1]))

    def test_argument_comparison_preserves_missing_extra_null_zero_and_false(self):
        differences = report.argument_differences(
            {'required': 0, 'known': None, 'enabled': False, 'value': {'count': 1}},
            {'known': None, 'enabled': 0, 'extra': False, 'value': {'count': 2}})
        by_path = {entry['path']: entry for entry in differences}
        self.assertNotIn('arguments.known', by_path)
        self.assertEqual(by_path['arguments.required']['kind'], 'missing_argument')
        self.assertEqual(by_path['arguments.extra']['kind'], 'extra_argument')
        self.assertEqual(by_path['arguments.enabled']['kind'], 'argument_type')
        self.assertEqual(by_path['arguments.value.count']['kind'], 'argument_value')


class CallDifferenceTests(unittest.TestCase):
    def compare(self, actual, *expected):
        return report.compare_calls(actual, list(expected))

    def test_any_complete_allowed_batch_can_match_without_gold_only_bias(self):
        first = [call('repeat_exchange', target='e1')]
        alternative = [call('repeat_answer', target='t1c1'), call('repeat_answer', target='t1c2')]
        result = self.compare(alternative, first, alternative)
        self.assertEqual(result['cost'], 0)
        self.assertEqual(result['differences'], [])
        self.assertEqual(result['kinds'], [])

    def test_missing_call_and_extra_duplicate_are_counted_with_multiplicity(self):
        first, second = call(subject='TR'), call('explain_default', subject='TR')
        missing = self.compare([second], [first, second])
        self.assertIn('missing_call', missing['kinds'])
        self.assertNotIn('wrong_function', missing['kinds'])
        extra = self.compare([first, first], [first])
        self.assertIn('extra_call', extra['kinds'])
        self.assertEqual(sum(entry['kind'] == 'extra_call' for entry in extra['differences']), 1)

    def test_order_only_mismatch_does_not_invent_function_or_argument_changes(self):
        first, second = call(subject='TR'), call('explain_default', subject='TR')
        result = self.compare([second, first], [first, second])
        self.assertEqual(result['kinds'], ['call_order'])

    def test_function_and_argument_mismatches_are_distinct(self):
        wrong_function = self.compare([call('explain_default', subject='TR')], [call(subject='TR')])
        self.assertIn('wrong_function', wrong_function['kinds'])
        wrong_argument = self.compare([call(subject='BIDS.basepath')], [call(subject='TR')])
        self.assertIn('argument_value', wrong_argument['kinds'])
        self.assertNotIn('wrong_function', wrong_argument['kinds'])

    def test_ambiguous_equal_cost_allowed_programs_are_not_silently_collapsed(self):
        result = self.compare([call(subject='actual')], [call(subject='left')], [call(subject='right')])
        self.assertEqual(result['closest_alternatives'], 2)
        self.assertGreater(result['cost'], 0)

    def test_argument_type_is_not_erased_by_python_bool_number_equality(self):
        result = self.compare([call(enabled=1)], [call(enabled=True)])
        self.assertIn('argument_type', result['kinds'])


class ExplanationFixture(unittest.TestCase):
    @staticmethod
    def example(index=0, actual=None, *, passed=False, critical='review_required', valid=True):
        item = fixtures.Fixture.case(index)['case']
        item.update(question=f'Question {index}', gold_calls=[call(subject='TR')],
                    allowed_batches=[[call(subject='TR')]], before={'exchanges': []})
        raw = fixtures.Fixture.raw(item, valid=valid)
        if valid:
            actual = [call(subject='wrong')] if actual is None else actual
            raw['raw_text'] = ''.join('<tool_call>' + json.dumps(value) + '</tool_call>'
                                      for value in actual) + '<|im_end|>'
        generation = qualification.converted_generation(raw, item, 'fixture')
        row = fixtures.Fixture.receipt(item, generation)
        row.update(success=passed, critical_status=critical,
                   error_kind=None if passed else 'semantic_unresolved',
                   semantic_result={'success': passed, 'critical_status': critical,
                                    'failure_category': None if passed else 'semantic_unresolved'})
        return item, row


class ExplanationTests(ExplanationFixture):
    def test_recorded_semantic_pass_does_not_become_a_structural_failure(self):
        item, row = self.example(actual=[call('equivalent_program', subject='TR')],
                                 passed=True, critical='none_observed')
        result = report.explain_case(item, row)
        self.assertTrue(result['passed'])
        self.assertIsNone(result['structural'])

    def test_transport_failure_does_not_attempt_to_invent_missing_semantic_calls(self):
        item, row = self.example(valid=False)
        result = report.explain_case(item, row)
        self.assertFalse(result['passed'])
        self.assertIsNone(result['structural'])
        self.assertFalse(result['transport_valid'])

    def test_recorded_critical_and_unresolved_remain_separate_from_differences(self):
        item, row = self.example(critical='detected')
        result = report.explain_case(item, row)
        self.assertEqual(result['critical_status'], 'detected')
        item, row = self.example(actual=[call(subject='TR')], critical='review_required')
        result = report.explain_case(item, row)
        self.assertEqual(result['critical_status'], 'review_required')
        self.assertFalse(result['passed'])
        self.assertEqual(result['structural']['differences'], [])

    def test_success_flag_alone_does_not_clear_unresolved_or_detected_critical(self):
        for status in ('review_required', 'detected'):
            item, row = self.example(actual=[call(subject='TR')], passed=True, critical=status)
            with self.subTest(status=status):
                result = report.explain_case(item, row)
                self.assertFalse(result['passed'])
                self.assertEqual(result['critical_status'], status)
                self.assertEqual(result['structural']['cost'], 0)

    def test_invalid_saved_native_claim_is_never_interpreted_as_a_good_call(self):
        item, row = self.example()
        row['generation']['reference_raw_text'] = '<tool_call>{broken}</tool_call><|im_end|>'
        with self.assertRaises((ValueError, AssertionError)):
            report.explain_case(item, row)


class SavedReportTests(fixtures.Fixture):
    def prepare(self):
        cases, manifest, source = fixtures.CompletedTests.completed(self, valid=31, semantic=2)
        for index, envelope in enumerate(cases):
            envelope['case'].update(question=f'Question {index}', gold_calls=[call(subject='TR')],
                                    allowed_batches=[[call(subject='TR')]], before={'exchanges': []})
        self.stack.enter_context(patch.object(qualification.diagnostic, 'NodeRPC',
                                             side_effect=lambda *args: self.rpc()))
        result = qualification.collect(self.child, manifest, source, cases)
        self.write(self.child / 'result.json', result)
        self.stack.enter_context(patch.object(qualification, 'load_source', return_value=source))
        self.stack.enter_context(patch.object(qualification, 'select_cases', return_value=cases))
        self.stack.enter_context(patch.object(qualification, 'verify_manifest'))
        return cases, manifest, source, result

    def prohibit_inference(self):
        for module, names in ((qualification, ('launch', 'check_dependencies')),
                              (qualification.transport, ('Runtime',)),
                              (qualification.reference, ('prepare_reference', 'validate_receipt')),
                              (qualification.diagnostic, ('student_provider', 'dev_worker', 'compare'))):
            for name in names:
                self.stack.enter_context(patch.object(module, name,
                    side_effect=AssertionError(f'Report must not invoke {name}')))

    def test_reads_all32_and_reports30_without_model_or_changes_to_saved_files(self):
        self.prepare()
        self.prohibit_inference()
        self.write(self.parent / 'closed-source.json', {'closed': True})
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        result = report.load_report()
        self.assertEqual((result['total'], result['passed'], result['failed']), (32, 2, 30))
        self.assertEqual(len(result['items']), 32)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_tampered_shard_or_terminal_counts_are_rejected(self):
        _, _, _, result = self.prepare()
        self.prohibit_inference()
        path = self.child / 'gpu-0.jsonl'
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        with self.assertRaises(ValueError):
            report.load_report()
        path.write_bytes(original)
        changed = deepcopy(result)
        changed['metrics']['supported']['passed'] = 3
        self.write(self.child / 'result.json', changed)
        with self.assertRaises(ValueError):
            report.load_report()

    def test_changed_child_manifest_is_rejected_by_terminal_artifact_hash(self):
        self.prepare()
        self.prohibit_inference()
        path = self.child / 'experiment.json'
        value = json.loads(path.read_text())
        value['added'] = 'changed after the completed run'
        self.write(path, value)
        with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
            report.load_report()

    def test_missing_terminal_result_is_not_a_request_to_generate(self):
        self.prepare()
        self.prohibit_inference()
        (self.child / 'result.json').unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            report.load_report()

    def test_report_propagates_parent_authentication_failure(self):
        self.prepare()
        self.prohibit_inference()
        with patch.object(qualification, 'load_source', side_effect=ValueError('parent mismatch')):
            with self.assertRaisesRegex(ValueError, 'parent mismatch'):
                report.load_report()


class AggregateAndDisplayTests(ExplanationFixture):
    def report(self):
        cases, rows = [], []
        for index in range(32):
            passed = index < 2
            status = 'none_observed' if passed or index == 2 else 'detected' if index == 3 else 'review_required'
            item, row = self.example(index, passed=passed, critical=status, valid=index != 2)
            item['question'] = 'Untrusted \x1b[31m\nquestion ' + 'x' * 10000
            cases.append({'case': item}); rows.append(row)
        return report.build_report(cases, rows), cases, rows

    def test_all30_unpassed_are_accounted_and_reviews_not_reclassified(self):
        result, _, _ = self.report()
        self.assertEqual((result['total'], result['passed'], result['failed']), (32, 2, 30))
        self.assertEqual(result['valid_format'], 31)
        self.assertEqual(result['critical'], 1)
        self.assertEqual(result['unresolved'], 28)
        self.assertEqual(result['structural_counts']['argument_value'], 29)
        self.assertEqual(result['argument_fields']['arguments.subject'], 29)

    def test_default_output_is_bounded_and_control_characters_are_escaped(self):
        result, _, _ = self.report()
        output = io.StringIO()
        with redirect_stdout(output):
            report.show(result)
        text = output.getvalue()
        self.assertLess(len(text), 12000)
        self.assertLess(len(text.splitlines()), 45)
        self.assertEqual(text.count('Example '), 4)
        self.assertNotIn('\x1b', text)
        self.assertIn('\\u001b', text)
        self.assertIn('do not adjudicate unresolved reviews', text)

    def test_all_mode_lists_every_nonpassing_case_once_and_no_verbose_examples(self):
        result, _, _ = self.report()
        output = io.StringIO()
        with redirect_stdout(output):
            report.show(result, all_cases=True)
        text = output.getvalue()
        lines = [line for line in text.splitlines() if line.startswith('{')]
        self.assertEqual(len(lines), 30)
        ids = [json.loads(line)['case'] for line in lines]
        self.assertEqual(len(set(ids)), 30)
        self.assertNotIn('case-00', ids)
        self.assertNotIn('case-01', ids)
        self.assertNotIn('Example ', text)

    def test_missing_duplicate_or_unplanned_rows_never_change_denominator(self):
        _, cases, rows = self.report()
        for changed in (rows[:-1], rows + [deepcopy(rows[0])],
                        [{**rows[0], 'case_id': 'unexpected'}, *rows[1:]]):
            with self.subTest(count=len(changed)), self.assertRaises(ValueError):
                report.build_report(cases, changed)

    def test_examples_cover_transport_critical_and_unresolved(self):
        result, _, _ = self.report()
        examples = report.examples_for(result, 4)
        self.assertEqual(len(examples), 4)
        self.assertTrue(any(not item['transport_valid'] for item in examples))
        self.assertTrue(any(item['critical_status'] == 'detected' for item in examples))
        self.assertTrue(any(item['critical_status'] == 'review_required' for item in examples))


if __name__ == '__main__':
    unittest.main()
