"""CPU controls for admission, model identity, recovery and planned denominators."""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
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

    def test_empty_or_partial_completion_never_falls_back_to_closed_recovery(self):
        source = self.runs / 'source/revisions/rope-alignment-v1/recovery/checkpoint-io-v1'
        self.save(source / 'experiment.json', {})
        self.save(source / 'result.json', {'status': 'budget_stopped'})
        completed = source / d.completion.COMPLETION_NAME
        completed.mkdir()
        with self.assertRaisesRegex(RuntimeError, 'no terminal result'):
            d.latest_source(self.runs, 'source')
        self.save(completed / 'experiment.json', {})
        with self.assertRaisesRegex(RuntimeError, 'no terminal result'):
            d.latest_source(self.runs, 'source')
        self.save(completed / 'result.json', {'status': 'finished'})
        self.assertEqual(d.latest_source(self.runs, 'source'), completed)
        (completed / 'experiment.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'no terminal result'):
            d.latest_source(self.runs, 'source')

    def test_completion_symlink_cannot_substitute_the_old_closed_source(self):
        source = self.runs / 'source/revisions/rope-alignment-v1/recovery/checkpoint-io-v1'
        self.save(source / 'experiment.json', {})
        self.save(source / 'result.json', {'status': 'budget_stopped'})
        (source / d.completion.COMPLETION_NAME).symlink_to(source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'completion directory.*symlink'):
            d.latest_source(self.runs, 'source')

    def test_completion_uses_authenticated_parent_context_and_child_evidence(self):
        source = self.runs / 'source/revisions/rope-alignment-v1/recovery/checkpoint-io-v1'
        completed = source / d.completion.COMPLETION_NAME
        self.save(source / 'experiment.json', {'origin': 'original'})
        self.save(source / 'result.json', {'status': 'budget_stopped'})
        self.save(completed / 'experiment.json', {'origin': 'completion'})
        self.save(completed / 'result.json', {'status': 'not_promoted'})
        context = {'directory': source, 'manifest': {'origin': 'original'},
                   'result': {'status': 'not_promoted'}, 'prepared': {'unchanged': True},
                   'train': source / 'train.jsonl', 'students': {'baseline-7205': {},
                       'control-1729': {}, 'treatment-1729': {}},
                   'evidence': [d.record(source / 'result.json'), d.record(completed / 'result.json')]}
        with patch.object(d.completion, 'load_completed_source', return_value=context) as load:
            self.assertIs(d.source_context(self.runs, 'source'), context)
        load.assert_called_once_with(self.runs, 'source')

    def test_invalid_completion_provenance_never_falls_back_to_old_result(self):
        source = self.runs / 'source/revisions/rope-alignment-v1/recovery/checkpoint-io-v1'
        completed = source / d.completion.COMPLETION_NAME
        self.save(source / 'experiment.json', {})
        self.save(source / 'result.json', {'status': 'budget_stopped'})
        self.save(completed / 'experiment.json', {})
        self.save(completed / 'result.json', {'status': 'not_promoted'})
        with patch.object(d.completion, 'load_completed_source', side_effect=ValueError('Completion identity changed')) as load:
            with self.assertRaisesRegex(ValueError, 'Completion identity changed'):
                d.source_context(self.runs, 'source')
        load.assert_called_once_with(self.runs, 'source')

    def test_closed_legacy_diagnostic_uses_historical_code_without_hardware_or_a_lease(self):
        from source_provenance import WEEKLY_REVISION, archived_record, verify_sources
        import run
        repo = Path(__file__).resolve().parents[1]
        names = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only',
                                         WEEKLY_REVISION, 'evaluation/'], text=True).splitlines()
        names.extend(['model.py', 'ledger.py', 'private_runtime.py', 'cluster', 'history.py', 'history_prepare.py',
                      'run.py', 'requirements.txt', 'experiments.json', 'experiments-history.json', 'diagnose.py',
                      'diagnose-data.mjs', 'diagnose-dev.mjs', 'reference_model.py', 'finish_history.py'])
        code = {'git_commit': WEEKLY_REVISION, 'dirty': False,
                'files': [archived_record(repo, WEEKLY_REVISION, name) for name in sorted(names)]}
        preparer = next(e for e in code['files'] if e['path'] == 'diagnose-data.mjs')
        parent_code = {**code, 'files': [preparer]}
        source = {'manifest': {'code': parent_code, 'assets': {}, 'config': {}}, 'students': {},
                  'result': {'completion_policy': d.completion.POLICY}, 'evidence': [], 'prepared': {}}
        output = self.runs / d.RUN_ID
        self.save(output / 'disk-budget.json', {'schema': 1, 'limit_bytes': d.DISK_LIMIT,
                  'reserve_bytes': d.RESERVE, 'auxiliary_allowance_bytes': d.AUXILIARY_ALLOWANCE,
                  'cache_roots': list(d.reference_model.CACHE_ROOTS), 'baseline': {}})
        contexts = []
        for name in ('diagnostic-train-contexts.jsonl', 'diagnostic-train-contexts.json'):
            path = self.root / 'assets' / name
            self.save(path, {})
            contexts.append(d.record(path))
        identity = {'code': code, 'source_evidence': [preparer], 'dependencies': {
                    'transformers': '4.57.3', 'huggingface-hub': '0.36.2', **d.reference_model.REFERENCE_DEPENDENCIES},
                    'hardware': {'devices': [{'index': n} for n in range(4)]}, 'contexts': contexts,
                    'disk_budget': d.record(output / 'disk-budget.json'), 'policy': d.diagnostic_policy()}
        self.save(output / 'identity.json', identity)
        manifest = {'identity_sha256': d.history.digest(identity), 'students': {}, 'source': {'assets': {}, 'config': {}},
                    'train_panel': {}, 'reference': {}, 'disk_budget': identity['disk_budget'],
                    'no_training_performed': True, 'sealed_test_accessed': False}
        self.save(output / 'experiment.json', manifest)
        self.save(output / 'train-panel.json', {})
        self.save(output / 'manifest-binding.json', {'sha256': d.history.digest(manifest)})
        final = {'identity_sha256': manifest['identity_sha256'], 'artifacts': []}
        self.save(output / 'result.json', final)
        args = SimpleNamespace(run_id=d.RUN_ID, source_run_id='original')
        self.assertFalse((self.root / 'diagnose-data.mjs').exists())
        with (patch.object(d, 'source_context', return_value=source), patch.object(d, 'check_inputs'),
              patch.object(d, 'verify_sources', side_effect=lambda root, value: verify_sources(repo, value)),
              patch.object(d, 'validate_panel') as panel, patch.object(d.reference_model, 'validate_receipt'),
              patch.object(d, 'execution_lease', side_effect=AssertionError('Report tried to acquire lease')),
              patch.object(d, 'version', side_effect=AssertionError('Report tried to import model dependencies')),
              patch.object(run, 'four_gpu_identity', side_effect=AssertionError('Report probed GPUs')),
              patch.object(d, 'launch', side_effect=AssertionError('Report tried to launch worker')),
              patch.object(d, 'publish_json', side_effect=AssertionError('Report tried to publish'))):
            before = {f: f.read_bytes() for f in self.root.rglob('*') if f.is_file()}
            with patch.object(d, 'measure_or_report', return_value=final) as validate_stages:
                self.assertEqual(d.compare(args), final)
                validate_stages.assert_called_once_with(output, identity, manifest, lease=None)
            panel.assert_called_once_with({}, source, preparer)
            with self.assertRaisesRegex(ValueError, 'missing its transport stage'):
                d.compare(args)
            self.assertEqual(before, {f: f.read_bytes() for f in self.root.rglob('*') if f.is_file()})
            for replacement, message in [([*code['files'][:-1]], 'inventory differs'),
                                         ([{**e, 'sha256': 'f' * 64} if e == preparer else e
                                           for e in code['files']], 'Frozen artifact changed')]:
                self.save(output / 'identity.json', {**identity, 'code': {**code, 'files': replacement}})
                with self.assertRaisesRegex(ValueError, message):
                    d.compare(args)
            self.save(output / 'identity.json', {**identity, 'source_evidence': []})
            with self.assertRaisesRegex(ValueError, 'source evidence differs'):
                d.compare(args)

    def test_terminal_verify_requires_the_exact_qualification_and_missing_stage_stays_missing(self):
        output = self.runs / d.RUN_ID
        manifest = {'identity_sha256': 'id', 'students': {'baseline-7205': {}}}
        final = {'identity_sha256': 'id', 'artifacts': []}
        self.save(output / 'result.json', final)
        stage = output / 'stages/reference-transport.json'
        admitted = {'passed': False, 'observations': []}
        self.save(stage, admitted)
        with (patch.object(d, 'launch', side_effect=AssertionError('Report tried to launch')),
              patch.object(d, 'qualify', return_value=admitted)):
            with self.assertRaisesRegex(ValueError, 'missing its baseline-7205 stage'):
                d.measure_or_report(output, {}, manifest, lease=None)
            self.assertFalse((output / 'stages/baseline-7205.json').exists())
            self.save(stage, {**admitted, 'passed': True})
            with self.assertRaisesRegex(ValueError, 'transport decision differs'):
                d.measure_or_report(output, {}, manifest, lease=None)

    def test_completed_result_recomputes_each_model_and_compares_terminal_inventory(self):
        output = self.runs / d.RUN_ID
        origin = output / 'dev-observations.jsonl'
        receipt = output / 'train-observations.jsonl'
        for path in [origin, receipt, *(output / n for n in ('experiment.json', 'manifest-binding.json',
                      'disk-budget.json', 'train-panel.json', 'train-panel.jsonl'))]:
            self.save(path, {})
        dev = {'observations': d.record(origin)}
        students = {name: dev for name in ('baseline-7205', 'control-1729', 'treatment-1729')}
        manifest = {'identity_sha256': 'id', 'students': students}
        admitted = {'passed': False, 'observations': [d.record(receipt)]}
        self.save(output / 'stages/reference-transport.json', admitted)
        train = {'observations': [d.record(receipt)], 'own_seen_train_fit': {'supported': {'passed': 3, 'planned': 4}}}
        for name in students:
            self.save(output / 'stages' / f'{name}.json', {'model': name, 'identity_sha256': 'id', 'train': train,
                      'dev': dev, 'dev_reused_from_authenticated_source': True})
        identity = {'source_evidence': []}
        with (patch.object(d, 'launch', side_effect=AssertionError('Unexpected worker')),
              patch.object(d, 'qualify', return_value=admitted), patch.object(d, 'collect_train', return_value=train),
              patch.object(d, 'print_summary')):
            final = d.measure_or_report(output, identity, manifest, lease=1)
            self.assertEqual(final['status'], 'TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED')
            before = {p: p.read_bytes() for p in output.rglob('*') if p.is_file()}
            with patch.object(d, 'publish_json', side_effect=AssertionError('Terminal result must remain read-only')):
                self.assertEqual(d.measure_or_report(output, identity, manifest, lease=None), final)
                self.assertEqual(before, {p: p.read_bytes() for p in output.rglob('*') if p.is_file()})
                self.save(output / 'result.json', {**final, 'artifacts': final['artifacts'][1:]})
                with self.assertRaisesRegex(ValueError, 'differs from recomputed'):
                    d.measure_or_report(output, identity, manifest, lease=None)
                self.save(output / 'result.json', final)
                path = output / 'stages/treatment-1729.json'
                self.save(path, {**d.read(path), 'train': {**train, 'own_seen_train_fit': {}}})
                with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                    d.measure_or_report(output, identity, manifest, lease=None)

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

    def test_budget_failure_during_tail_repair_never_truncates_valid_record(self):
        path = self.root / 'observations.jsonl'
        original = b'{"case_id":"one"}'
        path.write_bytes(original)
        with patch.object(d, 'guarded_write', side_effect=ValueError('Budget cache containment failed')):
            with self.assertRaisesRegex(ValueError, 'Budget cache containment'):
                d.recover_jsonl(path)
        self.assertEqual(path.read_bytes(), original)

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

    def test_disk_budget_preserves_baseline_and_never_credits_deleted_cache(self):
        runtime = self.root / '.runtime'
        runtime.mkdir()
        old = runtime / 'old-cache'; old.write_bytes(b'o' * 16384)
        output = self.runs / 'diagnostic'; output.mkdir()
        with patch.object(d.reference_model, 'CACHE_ROOTS', ('.runtime/reference',), create=True):
            budget = d.DiskBudget(output, create=True)
            original = budget.path.read_bytes()
            baseline = budget.usage()['controlled_and_cache_growth_bytes']
            fresh = runtime / 'new-cache'; fresh.write_bytes(b'n' * 32768)
            added = budget.usage()['controlled_and_cache_growth_bytes'] - baseline
            self.assertGreaterEqual(added, 32768)
            old.unlink()
            self.assertGreaterEqual(budget.usage()['controlled_and_cache_growth_bytes'] - baseline, 32768)
            resumed = d.DiskBudget(output)
            self.assertEqual(resumed.path.read_bytes(), original)
            self.assertEqual(resumed.usage(), budget.usage())

    def test_disk_budget_charges_cached_reference_and_deduplicates_hardlinks(self):
        reference = self.root / '.runtime/reference'
        reference.mkdir(parents=True)
        weights = reference / 'model'; weights.write_bytes(b'w' * 32768)
        output = self.runs / 'diagnostic'; output.mkdir()
        with patch.object(d.reference_model, 'CACHE_ROOTS', ('.runtime/reference',), create=True):
            budget = d.DiskBudget(output, create=True)
            before = budget.usage()
            (reference / 'same-model').hardlink_to(weights)
            after = budget.usage()
            self.assertGreaterEqual(before['reference_bytes'], 32768)
            self.assertLess(after['reference_bytes'] - before['reference_bytes'], 4096)
            self.assertLess(after['charged_bytes'] - before['charged_bytes'], 4096)

    def test_disk_cap_is_shared_by_concurrent_writers_and_rejects_before_append(self):
        output = self.runs / 'diagnostic'; output.mkdir()
        with patch.object(d.reference_model, 'CACHE_ROOTS', ('.runtime/reference',), create=True), \
                patch.object(d, 'DISK_LIMIT', 180000), patch.object(d, 'RESERVE', 0), \
                patch.object(d, 'AUXILIARY_ALLOWANCE', 0):
            budget = d.DiskBudget(output, create=True)
            def write(index):
                path = output / f'worker-{index}.jsonl'
                try:
                    with path.open('ab') as stream:
                        d.append_observation(stream, {'case_id': str(index), 'raw': 'x' * 60000})
                except RuntimeError:
                    self.assertEqual(path.stat().st_size, 0)
                    return False
                return True
            with patch.object(d, 'ACTIVE_BUDGET', budget), ThreadPoolExecutor(max_workers=4) as pool:
                outcomes = list(pool.map(write, range(4)))
            self.assertTrue(any(outcomes))
            self.assertFalse(all(outcomes))
            self.assertLessEqual(budget.usage()['charged_bytes'], d.DISK_LIMIT)

    def test_disk_inventory_rejects_external_symlink(self):
        runtime = self.root / '.runtime'; runtime.mkdir()
        (runtime / 'escape').symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            d.inventory(runtime)

    def source_fixture(self, source=None):
        source = source or self.runs / 'source'
        source.mkdir(parents=True)
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
        (self.root / 'run.py').write_text('# Frozen source fixture.\n')
        manifest = {'assets': assets, 'code': d.code_record(self.root, ['run.py']), 'origin': origin, 'config': {}}
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

    def completion_fixture(self):
        parent = self.runs / 'source/revisions/rope-alignment-v1/recovery/checkpoint-io-v1'
        parent, result, train, _, checkpoint = self.source_fixture(parent)
        manifest, prepared = d.read(parent / 'experiment.json'), d.read(parent / 'stages/prepare.json')['result']
        identities = d.read(parent / 'identities.json')
        outcome = {'passed': False, 'status': 'not_promoted'}
        result['pairs'][0]['decision'] = outcome
        self.save(parent / 'result.json', {**result, 'status': 'budget_stopped', 'pairs': []})
        self.save(parent / 'stages/baseline.json', {
            'identity_sha256': d.history.digest(manifest), 'result': result['baseline']})
        regressions = {}
        for arm in ('control', 'treatment'):
            name = f'{arm}-1729'
            weights = checkpoint(name + '-midpoint', 7598, {
                'identity_sha256': d.history.digest(identities[name]),
                'initial_checkpoint_sha256': manifest['origin']['checkpoint']['sha256']})
            regressions[arm] = {**result['pairs'][0][arm], 'checkpoint': weights}
        self.save(parent / 'stages/control-1729-regression.json', {
            'identity_sha256': d.history.digest(manifest), 'result': regressions['control']})
        for suffix, evaluated in (('midpoint', regressions['control']), ('trained', result['pairs'][0]['control'])):
            self.save(parent / 'stages' / f'control-1729-{suffix}.json', {
                'identity_sha256': d.history.digest(manifest), 'result': {'checkpoint': evaluated['checkpoint']}})
        inherited = {'identity_sha256': 'original-budget', 'maximum_updates': d.history.MAX_UPDATES,
                     'charged_updates': 800, 'maximum_seconds': d.history.MAX_SECONDS,
                     'elapsed_seconds': 140000.0, 'stages': [], 'inflight': None}
        old_budget = self.runs / 'source/budget.json'
        self.save(old_budget, inherited)
        static = [d.record(path) for path in sorted(parent.rglob('*')) if path.is_file()] + [d.record(old_budget)]
        source = {'directory': parent, 'manifest': manifest, 'prepared': prepared, 'train': train,
                  'identities': identities, 'budget': inherited, 'static': static}
        output = parent / d.completion.COMPLETION_NAME
        (self.root / 'finish_history.py').write_text('# Synthetic validator fixture.\n')
        completion = {'schema': 1, 'policy': d.completion.POLICY,
                      'source_directory': str(parent.relative_to(self.root)),
                      'source_manifest': d.record(parent / 'experiment.json'),
                      'inherited_budget': inherited,
                      'source_snapshot': {'immutable': static, 'append_only_prefixes': [],
                          'admitted_latest_checkpoints': [], 'original_result_present': True},
                      'code': d.code_record(self.root, ['finish_history.py'])}
        binding = d.history.digest(completion)
        self.save(output / 'experiment.json', completion)
        self.save(output / 'manifest-binding.json', {'sha256': binding})
        budget = {**inherited, 'identity_sha256': binding, 'inherited_sha256': d.history.digest(inherited),
                  'maximum_seconds': None, 'charged_updates': 1600, 'elapsed_seconds': 280000.0,
                  'prior_time_limit_seconds': inherited['maximum_seconds']}
        self.save(output / 'budget.json', budget)
        for name, value in [('treatment-1729-regression', regressions['treatment']),
                ('treatment-1729-midpoint', {'checkpoint': regressions['treatment']['checkpoint']}),
                ('treatment-1729-trained', {'checkpoint': result['pairs'][0]['treatment']['checkpoint']}), *[
                (f'{arm}-1729-final', result['pairs'][0][arm]) for arm in ('control', 'treatment')]]:
            self.save(output / 'stages' / f'{name}.json', {'identity_sha256': binding, 'result': value})
        artifacts = [d.record(output / name) for name in ('experiment.json', 'manifest-binding.json', 'budget.json')]
        artifacts += [d.record(path) for path in sorted((output / 'stages').glob('*.json'))]
        result.update(identity_sha256=binding, completion_policy=d.completion.POLICY, budget=budget,
                      original_budget=inherited, artifacts=artifacts, TEST_opened=False,
                      exploratory=True, product_acceptance=False)
        self.save(output / 'result.json', result)
        return source, output, result, outcome

    def completion_call(self, source, outcome):
        with patch.object(d.completion, 'ROOT', self.root), \
                patch.object(d.completion, 'load_source', return_value=source), \
                patch.object(d.history, 'validate_score'), patch.object(d.history, 'decision', return_value=outcome):
            return d.source_context(self.runs, 'source')

    def test_real_completion_loader_retains_parent_and_child_provenance(self):
        source, output, result, outcome = self.completion_fixture()
        original_result = (source['directory'] / 'result.json').read_bytes()
        context = self.completion_call(source, outcome)
        self.assertEqual(context['directory'], source['directory'])
        self.assertEqual(context['manifest'], source['manifest'])
        self.assertEqual(context['train'], source['train'])
        self.assertEqual(context['result'], result)
        self.assertEqual(list(context['students']), ['baseline-7205', 'control-1729', 'treatment-1729'])
        evidence = {entry['path']: entry for entry in context['evidence']}
        for entry in [*source['static'], *result['artifacts'], d.record(output / 'result.json')]:
            self.assertEqual(evidence[entry['path']], entry)
        for entry in context['evidence']:
            d.verify(entry)
        self.assertEqual((source['directory'] / 'result.json').read_bytes(), original_result)

    def test_real_completion_loader_rejects_wrong_result_identity(self):
        source, output, result, outcome = self.completion_fixture()
        self.save(output / 'result.json', {**result, 'identity_sha256': 'another-completion'})
        with self.assertRaisesRegex(ValueError, 'Completion result identity'):
            self.completion_call(source, outcome)

    def test_real_completion_loader_rejects_changed_parent_receipt(self):
        source, _, _, outcome = self.completion_fixture()
        self.save(source['directory'] / 'result.json', {'status': 'changed'})
        with self.assertRaisesRegex(ValueError, 'Frozen continuation input changed'):
            self.completion_call(source, outcome)

    def test_real_completion_loader_rejects_incomplete_artifact_inventory(self):
        source, output, result, outcome = self.completion_fixture()
        self.save(output / 'result.json', {**result, 'artifacts': result['artifacts'][1:]})
        with self.assertRaisesRegex(ValueError, 'Completion artifact inventory changed'):
            self.completion_call(source, outcome)

    def test_real_completion_loader_rejects_missing_training_stage(self):
        source, output, result, outcome = self.completion_fixture()
        missing = output / 'stages/treatment-1729-midpoint.json'
        missing.unlink()
        artifacts = [entry for entry in result['artifacts'] if entry['path'] != str(missing.relative_to(self.root))]
        self.save(output / 'result.json', {**result, 'artifacts': artifacts})
        with self.assertRaisesRegex(ValueError, 'missing.*stage receipt'):
            self.completion_call(source, outcome)

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
