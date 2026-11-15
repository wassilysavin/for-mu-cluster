"""CPU controls for admission, model identity, recovery and planned denominators."""

import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import diagnose as d


class DiagnosticControls(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.patch = patch.object(d, 'ROOT', self.root)
        self.patch.start()
        self.runs = self.root / 'runs'
        self.runs.mkdir()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def save(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_incomplete_latest_recovery_never_falls_back_to_old_closed_parent(self):
        parent = self.runs / 'source'
        self.save(parent / 'experiment.json', {})
        self.save(parent / 'result.json', {'status': 'old'})
        newest = parent / 'revisions/rope-alignment-v1/recovery/checkpoint-io-v1'
        self.save(newest / 'experiment.json', {})
        with self.assertRaisesRegex(RuntimeError, 'no terminal result'):
            d.latest_source(self.runs, 'source')
        self.save(newest / 'result.json', {'status': 'finished'})
        self.assertEqual(d.latest_source(self.runs, 'source'), newest)

    def test_source_path_rejects_escape_before_listing(self):
        parent = self.runs / 'source'
        parent.mkdir()
        outside = self.root / 'unrelated'
        outside.mkdir()
        (parent / 'revisions').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            d.latest_source(self.runs, 'source')

    def test_read_and_write_recovery_reject_symlinks_outside_checkout(self):
        outside = self.root.parent / (self.root.name + '-outside.jsonl')
        outside.write_text('{"kept":true}\n')
        try:
            link = self.root / 'linked.jsonl'
            link.symlink_to(outside)
            with self.assertRaises(ValueError): d.read(link)
            with self.assertRaises(ValueError): d.recover_jsonl(link)
            self.assertEqual(outside.read_text(), '{"kept":true}\n')
        finally:
            outside.unlink()

    def test_partial_jsonl_recovers_only_uncommitted_tail(self):
        path = self.root / 'observations.jsonl'
        path.write_bytes(b'{"case_id":"one"}\n{"case_id":')
        self.assertEqual(d.recover_jsonl(path), [{'case_id': 'one'}])
        self.assertEqual(path.read_bytes(), b'{"case_id":"one"}\n')

    def test_complete_tail_without_newline_is_preserved(self):
        path = self.root / 'observations.jsonl'
        path.write_text('{"case_id":"one"}')
        self.assertEqual(d.recover_jsonl(path), [{'case_id': 'one'}])
        self.assertTrue(path.read_bytes().endswith(b'\n'))

    def test_corruption_inside_committed_rows_is_not_truncated(self):
        path = self.root / 'observations.jsonl'
        data = b'{"case_id":"one"}\nnot-json\n'
        path.write_bytes(data)
        with self.assertRaises(ValueError): d.recover_jsonl(path)
        self.assertEqual(path.read_bytes(), data)

    def test_merge_rejects_duplicate_missing_and_extra_cases(self):
        first, second = self.root / 'one.jsonl', self.root / 'two.jsonl'
        first.write_text('{"case_id":"a"}\n')
        second.write_text('{"case_id":"a"}\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            d.merged_rows([first, second], ['a', 'b'])
        second.write_text('{"case_id":"c"}\n')
        with self.assertRaisesRegex(ValueError, 'missing or adding'):
            d.merged_rows([first, second], ['a', 'b'])
        second.write_text('{"case_id":"b"}\n')
        self.assertEqual([row['case_id'] for row in d.merged_rows([second, first], ['a', 'b'])], ['a', 'b'])

    def test_worker_identity_changes_with_checkpoint_model_phase_and_gpu(self):
        manifest = {'reference': {'fingerprint': 'ref'}, 'students': {
            'a': {'checkpoint': {'sha256': 'one'}}, 'b': {'checkpoint': {'sha256': 'two'}}}}
        baseline = d.worker_binding(manifest, 'a', 'measure', 0)
        self.assertNotEqual(baseline, d.worker_binding(manifest, 'b', 'measure', 0))
        self.assertNotEqual(baseline, d.worker_binding(manifest, 'a', 'qualification', 0))
        self.assertNotEqual(baseline, d.worker_binding(manifest, 'a', 'measure', 1))
        manifest['students']['a']['checkpoint']['sha256'] = 'changed'
        self.assertNotEqual(baseline, d.worker_binding(manifest, 'a', 'measure', 0))

    def test_transport_admission_includes_longest_without_dev_selection(self):
        cases = [{'case': {'case_id': f'case-{index}', 'group': f'group-{index % 8}',
                           'semantic_eligible': True}, 'input_token_count': index + 10} for index in range(80)]
        selected = d.qualification_ids(cases)
        self.assertEqual(len(selected), 32)
        self.assertEqual(len(set(selected)), 32)
        self.assertEqual(selected[0], 'case-79')
        self.assertEqual(selected, d.qualification_ids(list(reversed(cases))))

    def test_unresolved_and_critical_are_not_semantic_passes(self):
        base = {'semantic_eligible': True, 'cohort': 'supported', 'success': True,
                'native_exact_match': False, 'canonical_call_match': False, 'transport_valid': True}
        score = d.metrics([{**base, 'critical_status': status} for status in
                           ('none_observed', 'review_required', 'detected')])
        self.assertEqual(score['supported']['passed'], 1)
        self.assertEqual(score['supported']['planned'], 3)
        self.assertEqual(score['unresolved'], 1)
        self.assertEqual(score['critical_detected'], 1)

    def test_paired_unresolved_cases_are_not_called_joint_errors(self):
        base = {'panel': 'fixed_prefix', 'cohort': 'supported', 'critical_status': 'none_observed'}
        student = [{**base, 'case_id': 'a', 'success': False}, {**base, 'case_id': 'b', 'success': True},
                   {**base, 'case_id': 'c', 'success': False, 'critical_status': 'review_required'}]
        reference = [{**base, 'case_id': key, 'success': True} for key in ('a', 'b', 'c')]
        counts = d.paired_cells(student, reference)['fixed_prefix']
        self.assertEqual(counts['reference_only_pass'], 1)
        self.assertEqual(counts['both_pass'], 1)
        self.assertEqual(counts['neither_pass'], 0)
        self.assertEqual(counts['unresolved_any'], 1)

    def test_insufficient_disk_admission_does_not_write(self):
        with patch.object(d, 'disk_usage', return_value=type('Space', (), {'free': d.RESERVE})()):
            with self.assertRaises(RuntimeError): d.require_space(self.root, 1)
        self.assertEqual(list(self.root.iterdir()), [self.runs])

    def source_fixture(self):
        source = self.runs / 'source'
        source.mkdir()
        def checkpoint(name, step, lineage):
            path = source / f'{name}.safetensors'
            header = json.dumps({'__metadata__': {'step': str(step), 'lineage': json.dumps(lineage)}}).encode()
            path.write_bytes(struct.pack('<Q', len(header)) + header)
            return d.record(path)
        lineage = {'identity_sha256': 'full-course', 'initial_checkpoint_sha256': 'original'}
        initial = checkpoint('baseline', 7205, lineage)
        dummy = source / 'asset.json'
        dummy.write_text('{}')
        assets = {key: d.record(dummy) for key in ('registry', 'tokenizer', 'history_train_specs')}
        assets['initial_checkpoint'] = initial
        prepared = {'collection_receipt': {'fingerprint': {key: assets[asset]['sha256'] for key, asset in (
            ('checkpoint_sha256', 'initial_checkpoint'), ('registry_sha256', 'registry'),
            ('tokenizer_sha256', 'tokenizer'), ('specs_sha256', 'history_train_specs'))}}}
        origin = {'checkpoint': initial, 'lineage': lineage}
        manifest = {'assets': assets, 'code': {'files': []}, 'origin': origin, 'config': {}}
        binding = d.history.digest(manifest)
        observations, timings = source / 'observations.jsonl', source / 'timings.jsonl'
        observations.write_text(''); timings.write_text('')
        def evaluation(weights):
            return {'checkpoint': weights, 'observations': d.record(observations), 'timings': d.record(timings),
                    'seconds': 0, 'score': {'infrastructure_failures': 0}}
        identities, arms = {}, {}
        for arm in ('control', 'treatment'):
            name = f'{arm}-1729'
            identity = {'experiment_id': name, 'initial_checkpoint': initial,
                        'prepared_sha256': d.history.digest(prepared), 'train': {'arm': name, 'schedule_sha256': name}}
            identities[name] = identity
            weights = checkpoint(name, 7991, {'identity_sha256': d.history.digest(identity),
                                              'initial_checkpoint_sha256': initial['sha256']})
            arms[arm] = evaluation(weights)
        result = {'identity_sha256': binding, 'status': 'not_promoted', 'baseline': evaluation(initial),
                  'pairs': [{'seed': 1729, **arms}]}
        self.save(source / 'experiment.json', manifest)
        self.save(source / 'result.json', result)
        self.save(source / 'identities.json', identities)
        self.save(source / 'stages/official-parity.json', {'identity_sha256': binding, 'result': {'passed': True}})
        self.save(source / 'stages/prepare.json', {'identity_sha256': binding, 'result': prepared})
        train = source / 'train.jsonl'; train.write_text('{}\n')
        schedules = {name: SimpleNamespace(sha256=name) for name in identities}
        return source, result, train, schedules, checkpoint

    def source_call(self, train, schedules):
        with patch.object(d.history, 'validate_prepared', return_value=((train, d.record(train)), schedules)), \
             patch.object(d.history, 'validate_score'):
            return d.source_context(self.runs, 'source')

    def test_complete_negative_pair_is_admitted_without_choosing_a_winner(self):
        source, result, train, schedules, _ = self.source_fixture()
        context = self.source_call(train, schedules)
        self.assertEqual(list(context['students']), ['baseline-7205', 'control-1729', 'treatment-1729'])
        result['status'] = 'budget_stopped'
        self.save(source / 'result.json', result)
        self.assertEqual(len(self.source_call(train, schedules)['students']), 3)

    def test_incomplete_pair_and_second_seed_substitution_are_rejected(self):
        source, result, train, schedules, _ = self.source_fixture()
        pair = result['pairs'][0]
        for replacement in ([], [{**pair, 'seed': 2718}], [{'seed': 1729, 'control': pair['control']}]):
            with self.subTest(replacement=replacement):
                self.save(source / 'result.json', {**result, 'pairs': replacement})
                with self.assertRaisesRegex(RuntimeError, 'both complete seed-1729'):
                    self.source_call(train, schedules)

    def test_wrong_checkpoint_lineage_is_rejected_even_with_updated_file_hash(self):
        source, result, train, schedules, write_checkpoint = self.source_fixture()
        result['pairs'][0]['control']['checkpoint'] = write_checkpoint('control-1729', 7991,
            {'identity_sha256': 'a-different-model', 'initial_checkpoint_sha256': result['baseline']['checkpoint']['sha256']})
        self.save(source / 'result.json', result)
        with self.assertRaisesRegex(ValueError, 'lineage'):
            self.source_call(train, schedules)

    def test_wrong_checkpoint_step_is_rejected_without_falling_back(self):
        source, result, train, schedules, write_checkpoint = self.source_fixture()
        result['pairs'][0]['control']['checkpoint'] = write_checkpoint('control-1729', 7990, {})
        self.save(source / 'result.json', result)
        with self.assertRaisesRegex(ValueError, 'checkpoint step'):
            self.source_call(train, schedules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
